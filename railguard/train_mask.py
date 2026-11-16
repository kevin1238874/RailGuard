"""Lazy COCO train segmentation for the optional M2 challenger.

Importing this module is deliberately cheap: neither Torch nor Ultralytics is
imported until :meth:`YoloSegTrainMaskProvider._load_model` is called.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np


@dataclass
class TrainMaskResult:
    mask: np.ndarray
    metadata: dict[str, Any]


class YoloSegTrainMaskProvider:
    """Union class-6 COCO instance masks behind a lazy Ultralytics adapter."""

    def __init__(
        self,
        weights: str = "yolo11n-seg.pt",
        confidence: float = 0.25,
        imgsz: int = 640,
        device: str | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        self.weights = str(weights)
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        self.device = device
        self._model = model
        self._cold_load_ms: float | None = 0.0 if model is not None else None

    def _load_model(self) -> Any:
        if self._model is None:
            started = perf_counter()
            from ultralytics import YOLO  # lazy by contract

            weights = Path(self.weights)
            if not weights.is_absolute():
                repo_weight = Path(__file__).resolve().parent.parent / weights
                if repo_weight.exists():
                    weights = repo_weight
            self._model = YOLO(str(weights))
            self._cold_load_ms = (perf_counter() - started) * 1000.0
        return self._model

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def predict(self, image: np.ndarray) -> TrainMaskResult:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Expected uint8 BGR color image")
        h, w = image.shape[:2]
        model = self._load_model()
        started = perf_counter()
        kwargs: dict[str, Any] = {
            "source": image,
            "classes": [6],
            "conf": self.confidence,
            "imgsz": self.imgsz,
            "retina_masks": True,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        results = model.predict(**kwargs)
        inference_ms = (perf_counter() - started) * 1000.0

        union = np.zeros((h, w), np.uint8)
        raw_instances = kept_instances = 0
        min_component = max(1, int(np.ceil(0.002 * h * w)))
        for result in results or []:
            masks_obj = getattr(result, "masks", None)
            data = getattr(masks_obj, "data", None) if masks_obj is not None else None
            if data is None:
                continue
            masks = self._as_numpy(data)
            if masks.ndim == 2:
                masks = masks[None, ...]
            for raw_mask in masks:
                raw_instances += 1
                binary = (raw_mask > 0.5).astype(np.uint8)
                if binary.shape != (h, w):
                    binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
                count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
                for label in range(1, count):
                    if int(stats[label, cv2.CC_STAT_AREA]) < min_component:
                        continue
                    union[labels == label] = 1
                    kept_instances += 1

        coverage = float(np.mean(union > 0))
        k = max(5, int(round(0.015 * min(h, w))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        core = cv2.erode(union, kernel) if np.any(union) else union.copy()
        core_nonempty = bool(np.any(core))
        valid = bool(0.005 <= coverage <= 0.85 and core_nonempty)
        if raw_instances == 0:
            fallback_reason = "no_train_detections"
        elif coverage < 0.005:
            fallback_reason = "train_mask_too_small"
        elif coverage > 0.85:
            fallback_reason = "train_mask_too_large"
        elif not core_nonempty:
            fallback_reason = "empty_eroded_train_core"
        else:
            fallback_reason = None
        metadata: dict[str, Any] = {
            "provider": "yolo_seg",
            "weights": self.weights,
            "confidence": self.confidence,
            "imgsz": self.imgsz,
            "device": self.device,
            "raw_instances": raw_instances,
            "kept_components": kept_instances,
            "coverage": round(coverage, 8),
            "erosion_radius_px": k,
            "eroded_core_nonempty": core_nonempty,
            "valid": valid,
            "fallback_reason": fallback_reason,
            "cold_load_ms": None if self._cold_load_ms is None else round(self._cold_load_ms, 3),
            "warm_inference_ms": round(inference_ms, 3),
        }
        return TrainMaskResult(mask=union.astype(bool), metadata=metadata)


_DEFAULT_PROVIDER: YoloSegTrainMaskProvider | None = None


def get_default_train_mask_provider() -> YoloSegTrainMaskProvider:
    """Return a cached singleton YOLO-seg train mask provider."""
    global _DEFAULT_PROVIDER
    if _DEFAULT_PROVIDER is None:
        _DEFAULT_PROVIDER = YoloSegTrainMaskProvider()
    return _DEFAULT_PROVIDER
