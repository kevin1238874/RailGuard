"""Unified person detections and lazy Ultralytics adapters.

The module does not download or load a model at import time. Tests can inject a
small object exposing ``predict`` through ``model=``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional

import numpy as np

Point = tuple[float, float]


def _finite_bbox(bbox) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must contain x1, y1, x2, y2")
    value = tuple(float(item) for item in bbox)
    if not all(np.isfinite(value)) or value[2] < value[0] or value[3] < value[1]:
        raise ValueError("bbox must be finite and ordered x1 <= x2, y1 <= y2")
    return value


@dataclass
class Person:
    """Detector-independent person instance."""

    bbox: tuple[float, float, float, float]
    confidence: float = 1.0
    object_id: Optional[str] = None
    ankles: Optional[list[tuple[float, float, float]]] = None
    runtime_ms: float = 0.0
    detector: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bbox = _finite_bbox(self.bbox)
        self.confidence = float(self.confidence)
        self.runtime_ms = float(self.runtime_ms)
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must lie in [0, 1]")
        if not np.isfinite(self.runtime_ms) or self.runtime_ms < 0:
            raise ValueError("runtime_ms must be finite and nonnegative")

    @property
    def bbox_bottom(self) -> Point:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    def foot_points(self, method: str = "bbox", min_conf: float = 0.35,
                    fallback: str = "bbox") -> tuple[list[Point], str]:
        """Return points and explicitly distinguish pose fallback/abstention."""
        aliases = {"bbox_bottom": "bbox", "pose_ankle": "pose"}
        method = aliases.get(method, method)
        if method not in {"bbox", "pose"}:
            raise ValueError("method must be bbox/bbox_bottom or pose/pose_ankle")
        if fallback not in {"bbox", "abstain"}:
            raise ValueError("fallback must be bbox or abstain")
        if not 0 <= min_conf <= 1:
            raise ValueError("min_conf must lie in [0, 1]")
        if method == "bbox":
            return [self.bbox_bottom], "bbox_bottom_center"

        x1, y1, x2, y2 = self.bbox
        reliable: list[Point] = []
        for ankle in self.ankles or []:
            if len(ankle) != 3:
                continue
            x, y, score = map(float, ankle)
            if (np.isfinite((x, y, score)).all() and score >= min_conf
                    and x > 0 and y > 0 and x1 <= x <= x2 and y1 <= y <= y2):
                reliable.append((x, y))
        if reliable:
            return reliable[:2], "two_ankles" if len(reliable) > 1 else "one_ankle"
        if fallback == "bbox":
            return [self.bbox_bottom], "bbox_fallback"
        return [], "abstain"


def _as_list(value):
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _scalar(value) -> float:
    value = _as_list(value)
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("model returned an empty scalar")
        value = value[0]
    return float(value)


def _validate_imgsz(imgsz: int | tuple[int, int]) -> int | tuple[int, int]:
    if isinstance(imgsz, bool):
        raise ValueError("imgsz must be a positive integer or (height, width)")
    if isinstance(imgsz, int):
        if imgsz <= 0:
            raise ValueError("imgsz must be positive")
        return imgsz
    if (not isinstance(imgsz, (tuple, list)) or len(imgsz) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in imgsz)):
        raise ValueError("imgsz must be a positive integer or (height, width)")
    return tuple(imgsz)


class YOLODetector:
    """B3 detection / A pose adapter for COCO person class 0.

    Supports direct pose, bbox, and detect-crop-pose refinement for distant/small people.
    """

    def __init__(self, method: str = "bbox", weights: str | None = None,
                 confidence: float = 0.25, device: str | int | None = None, *,
                 imgsz: int | tuple[int, int] = 640, iou: float = 0.7,
                 conf: float | None = None, model=None,
                 enable_crop_pose: bool = True):
        if method not in {"bbox", "pose"}:
            raise ValueError("method must be bbox or pose")
        if conf is not None:
            confidence = conf
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("conf must be numeric")
        if not 0 <= float(confidence) <= 1:
            raise ValueError("conf must lie in [0, 1]")
        if isinstance(iou, bool) or not isinstance(iou, (int, float)) or not 0 < float(iou) <= 1:
            raise ValueError("iou must lie in (0, 1]")
        if weights is not None and (not isinstance(weights, str) or not weights.strip()):
            raise ValueError("weights must be a nonempty path or model name")
        if device is not None and not isinstance(device, (str, int)):
            raise ValueError("device must be None, an integer, or a nonempty string")
        if isinstance(device, str) and not device.strip():
            raise ValueError("device must not be empty")

        self.method = method
        self.weights = weights or ("yolo11n-pose.pt" if method == "pose" else "yolo11n.pt")
        self.confidence = float(confidence)
        self.imgsz = _validate_imgsz(imgsz)
        self.iou = float(iou)
        self.device = device
        self.enable_crop_pose = enable_crop_pose
        self._det_model = None
        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Ultralytics missing; install the optional 'vision' dependencies") from exc
            model = YOLO(self.weights)
            if self.method == "pose" and self.enable_crop_pose:
                import os
                det_path = "yolo11n.pt"
                if os.path.exists(det_path):
                    self._det_model = YOLO(det_path)
        self.model = model
        self.last_runtime_ms = 0.0
        self.last_config: dict[str, Any] = {}

    def predict(self, image: np.ndarray) -> list[Person]:
        if not isinstance(image, np.ndarray) or image.ndim not in {2, 3} or image.size == 0:
            raise ValueError("image must be a nonempty grayscale or color numpy array")
        opts: dict[str, Any] = {
            "source": image, "verbose": False, "conf": self.confidence,
            "iou": self.iou, "imgsz": self.imgsz,
        }
        if self.device is not None:
            opts["device"] = self.device
        started = perf_counter()
        outputs = self.model.predict(**opts)
        self.last_runtime_ms = (perf_counter() - started) * 1000
        self.last_config = {
            "weights": self.weights, "conf": self.confidence, "iou": self.iou,
            "imgsz": self.imgsz, "device": self.device,
            "runtime_ms": self.last_runtime_ms,
        }
        if outputs is None or len(outputs) == 0:
            return []
        output = outputs[0]
        boxes = getattr(output, "boxes", None)
        if boxes is None:
            return []
        people: list[Person] = []
        image_h, image_w = image.shape[:2]
        keypoints = getattr(output, "keypoints", None)

        # Nếu đang chạy Pose và có _det_model hỗ trợ: dùng YOLO Det để tìm trọn vẹn người nhỏ ở xa
        if self.method == "pose" and self._det_model is not None:
            det_opts = {
                "source": image, "verbose": False, "conf": max(0.15, self.confidence * 0.75),
                "iou": self.iou, "imgsz": self.imgsz,
            }
            if self.device is not None:
                det_opts["device"] = self.device
            det_res = self._det_model.predict(**det_opts)
            if det_res and len(det_res) > 0 and getattr(det_res[0], "boxes", None) is not None:
                det_boxes = det_res[0].boxes
                for box in det_boxes:
                    if int(_scalar(box.cls)) != 0:
                        continue
                    b_conf = float(_scalar(box.conf))
                    if b_conf < self.confidence:
                        continue
                    bbox_raw = _as_list(box.xyxy)
                    while isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 1 and isinstance(bbox_raw[0], (list, tuple)):
                        bbox_raw = bbox_raw[0]
                    bx1, by1, bx2, by2 = _finite_bbox(bbox_raw)
                    # Kiểm tra xem box này đã được cover bởi kết quả pose trực tiếp chưa
                    matched_keypoints = None
                    if keypoints is not None:
                        xy_all = _as_list(getattr(keypoints, "xy", None))
                        conf_all = _as_list(getattr(keypoints, "conf", None))
                        for p_idx, p_box in enumerate(boxes):
                            if int(_scalar(p_box.cls)) != 0:
                                continue
                            px1, py1, px2, py2 = _finite_bbox(_as_list(p_box.xyxy)[0])
                            # IoU matching giữa det box và pose box
                            ix1, iy1 = max(bx1, px1), max(by1, py1)
                            ix2, iy2 = min(bx2, px2), min(by2, py2)
                            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                            union = (bx2 - bx1) * (by2 - by1) + (px2 - px1) * (py2 - py1) - inter
                            if union > 0 and inter / union > 0.45:
                                if xy_all is not None and p_idx < len(xy_all):
                                    matched_keypoints = (xy_all[p_idx], conf_all[p_idx] if conf_all is not None else [1.0] * 17)
                                break

                    ankles = []
                    if matched_keypoints is not None:
                        pts, scores = matched_keypoints
                        for k_idx in (15, 16):
                            if k_idx < len(pts):
                                score = scores[k_idx] if k_idx < len(scores) else 0.0
                                ankles.append((float(pts[k_idx][0]), float(pts[k_idx][1]), float(score)))
                    else:
                        # Chạy Pose trực tiếp trên crop người nhỏ phóng to (crop-pose)
                        bh = by2 - by1
                        pad = int(bh * 0.15)
                        cy1, cy2 = max(0, int(by1 - pad)), min(image_h, int(by2 + pad))
                        cx1, cx2 = max(0, int(bx1 - pad)), min(image_w, int(bx2 + pad))
                        crop = image[cy1:cy2, cx1:cx2]
                        if crop.size > 0 and crop.shape[0] >= 16 and crop.shape[1] >= 16:
                            sub_res = self.model.predict(source=crop, verbose=False, conf=max(0.15, self.confidence * 0.7), imgsz=256)
                            if sub_res and len(sub_res) > 0 and getattr(sub_res[0], "keypoints", None) is not None:
                                sub_xy = _as_list(getattr(sub_res[0].keypoints, "xy", None))
                                sub_conf = _as_list(getattr(sub_res[0].keypoints, "conf", None))
                                if sub_xy and len(sub_xy) > 0:
                                    s_pts = sub_xy[0]
                                    s_scores = sub_conf[0] if sub_conf else [1.0] * 17
                                    for k_idx in (15, 16):
                                        if k_idx < len(s_pts):
                                            score = s_scores[k_idx] if k_idx < len(s_scores) else 0.0
                                            ankles.append((float(s_pts[k_idx][0] + cx1), float(s_pts[k_idx][1] + cy1), float(score)))

                    metadata = dict(self.last_config)
                    metadata["legs_cropped"] = bool(by2 >= image_h - 1)
                    metadata["bbox_clipped"] = bool(bx1 <= 0 or by1 <= 0 or bx2 >= image_w - 1 or by2 >= image_h - 1)
                    metadata["missing_ankles"] = 2 - sum(int(x > 0 and y > 0 and s >= self.confidence) for x, y, s in ankles)
                    people.append(Person(
                        (bx1, by1, bx2, by2), b_conf, ankles=ankles,
                        runtime_ms=self.last_runtime_ms,
                        detector="yolo_pose_crop_enhanced",
                        metadata=metadata,
                    ))
                return people
        for index, box in enumerate(boxes):
            if int(_scalar(box.cls)) != 0:
                continue
            bbox_raw = _as_list(box.xyxy)
            while (isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 1
                   and isinstance(bbox_raw[0], (list, tuple))):
                bbox_raw = bbox_raw[0]
            bbox = _finite_bbox(bbox_raw)
            ankles = None
            metadata: dict[str, Any] = dict(self.last_config)
            if self.method == "pose":
                ankles = []
                if keypoints is not None:
                    xy_all = _as_list(getattr(keypoints, "xy", None))
                    conf_all = _as_list(getattr(keypoints, "conf", None))
                    if xy_all is not None and index < len(xy_all):
                        points = xy_all[index]
                        scores = (conf_all[index] if conf_all is not None and index < len(conf_all)
                                  else [1.0] * len(points))
                        for keypoint_index in (15, 16):
                            if keypoint_index < len(points):
                                point = points[keypoint_index]
                                score = scores[keypoint_index] if keypoint_index < len(scores) else 0.0
                                ankles.append((float(point[0]), float(point[1]), float(score)))
                x1, y1, x2, y2 = bbox
                metadata["legs_cropped"] = bool(y2 >= image_h - 1)
                metadata["bbox_clipped"] = bool(x1 <= 0 or y1 <= 0 or x2 >= image_w - 1 or y2 >= image_h - 1)
                metadata["missing_ankles"] = 2 - sum(
                    int(x > 0 and y > 0 and score >= self.confidence) for x, y, score in ankles)
            people.append(Person(
                bbox, max(0.0, min(1.0, _scalar(box.conf))), ankles=ankles,
                runtime_ms=self.last_runtime_ms,
                detector="yolo_pose" if self.method == "pose" else "yolo_detection",
                metadata=metadata,
            ))
        return people
