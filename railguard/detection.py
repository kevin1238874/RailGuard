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


def _box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def _unwrap_bbox(raw):
    while isinstance(raw, (list, tuple)) and len(raw) == 1 and isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    return _finite_bbox(raw)


class YOLODetector:
    """B3 detection / A pose adapter for COCO person class 0.

    Optional detect -> crop -> pose (``enable_crop_pose=True``) is an explicit
    research branch for small/distant people. It is never enabled implicitly
    (e.g. by a weights file happening to exist in the working directory), and
    its detector weights are recorded in ``last_config`` so runs stay
    reproducible. Direct pose detections are always kept; the detection model
    only *adds* people that the pose model missed.
    """

    def __init__(self, method: str = "bbox", weights: str | None = None,
                 confidence: float = 0.25, device: str | int | None = None, *,
                 imgsz: int | tuple[int, int] = 640, iou: float = 0.7,
                 conf: float | None = None, model=None,
                 enable_crop_pose: bool = False,
                 crop_pose_det_weights: str | None = None, det_model=None,
                 crop_pose_imgsz: int = 256, crop_match_iou: float = 0.45):
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
        if not 0 < float(crop_match_iou) <= 1:
            raise ValueError("crop_match_iou must lie in (0, 1]")

        self.method = method
        self.weights = weights or ("yolo11n-pose.pt" if method == "pose" else "yolo11n.pt")
        self.confidence = float(confidence)
        self.imgsz = _validate_imgsz(imgsz)
        self.iou = float(iou)
        self.device = device
        self.enable_crop_pose = bool(enable_crop_pose) and method == "pose"
        self.crop_pose_imgsz = _validate_imgsz(crop_pose_imgsz)
        self.crop_match_iou = float(crop_match_iou)
        self.crop_pose_det_weights = ((crop_pose_det_weights or "yolo11n.pt")
                                      if self.enable_crop_pose else None)
        self._det_model = None
        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Ultralytics missing; install the optional 'vision' dependencies") from exc
            model = YOLO(self.weights)
            if self.enable_crop_pose and det_model is None:
                det_model = YOLO(self.crop_pose_det_weights)
        if self.enable_crop_pose:
            if det_model is None:
                raise ValueError("enable_crop_pose with an injected model also requires det_model")
            self._det_model = det_model
        self.model = model
        self.last_runtime_ms = 0.0
        self.last_config: dict[str, Any] = {}

    def _opts(self, image, *, conf=None, imgsz=None) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "source": image, "verbose": False,
            "conf": self.confidence if conf is None else conf,
            "iou": self.iou, "imgsz": self.imgsz if imgsz is None else imgsz,
        }
        if self.device is not None:
            opts["device"] = self.device
        return opts

    def predict(self, image: np.ndarray) -> list[Person]:
        if not isinstance(image, np.ndarray) or image.ndim not in {2, 3} or image.size == 0:
            raise ValueError("image must be a nonempty grayscale or color numpy array")
        started = perf_counter()
        outputs = self.model.predict(**self._opts(image))
        self.last_runtime_ms = (perf_counter() - started) * 1000
        self.last_config = {
            "weights": self.weights, "conf": self.confidence, "iou": self.iou,
            "imgsz": self.imgsz, "device": self.device,
            "runtime_ms": self.last_runtime_ms,
            "crop_pose": self.enable_crop_pose,
            "crop_pose_det_weights": self.crop_pose_det_weights,
        }
        people = self._parse(outputs, image.shape[:2], self.method)
        if self._det_model is not None:
            people = self._crop_pose_merge(image, people)
            self.last_runtime_ms = (perf_counter() - started) * 1000
            self.last_config["runtime_ms"] = self.last_runtime_ms
            for person in people:
                person.runtime_ms = self.last_runtime_ms
                person.metadata["runtime_ms"] = self.last_runtime_ms
        return people

    def _parse(self, outputs, image_hw, method: str,
               offset: tuple[float, float] = (0.0, 0.0)) -> list[Person]:
        """Convert one Ultralytics result into Person records, optionally shifted."""
        if outputs is None or len(outputs) == 0:
            return []
        output = outputs[0]
        boxes = getattr(output, "boxes", None)
        if boxes is None:
            return []
        image_h, image_w = image_hw
        ox, oy = offset
        keypoints = getattr(output, "keypoints", None)
        xy_all = _as_list(getattr(keypoints, "xy", None)) if keypoints is not None else None
        conf_all = _as_list(getattr(keypoints, "conf", None)) if keypoints is not None else None
        people: list[Person] = []
        for index, box in enumerate(boxes):
            if int(_scalar(box.cls)) != 0:
                continue
            x1, y1, x2, y2 = _unwrap_bbox(_as_list(box.xyxy))
            bbox = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
            ankles = None
            metadata: dict[str, Any] = dict(self.last_config)
            if method == "pose":
                ankles = []
                if xy_all is not None and index < len(xy_all):
                    points = xy_all[index]
                    scores = (conf_all[index] if conf_all is not None and index < len(conf_all)
                              else [1.0] * len(points))
                    for keypoint_index in (15, 16):
                        if keypoint_index < len(points):
                            px, py = float(points[keypoint_index][0]), float(points[keypoint_index][1])
                            score = scores[keypoint_index] if keypoint_index < len(scores) else 0.0
                            # Ultralytics reports an unpredicted keypoint as (0, 0); keep that marker.
                            if px > 0 and py > 0:
                                px, py = px + ox, py + oy
                            ankles.append((px, py, float(score)))
                self._pose_metadata(metadata, bbox, ankles, image_h, image_w)
            people.append(Person(
                bbox, max(0.0, min(1.0, _scalar(box.conf))), ankles=ankles,
                runtime_ms=self.last_runtime_ms,
                detector="yolo_pose" if method == "pose" else "yolo_detection",
                metadata=metadata,
            ))
        return people

    def _pose_metadata(self, metadata, bbox, ankles, image_h, image_w) -> None:
        x1, y1, x2, y2 = bbox
        metadata["legs_cropped"] = bool(y2 >= image_h - 1)
        metadata["bbox_clipped"] = bool(x1 <= 0 or y1 <= 0 or x2 >= image_w - 1 or y2 >= image_h - 1)
        metadata["missing_ankles"] = 2 - sum(
            int(x > 0 and y > 0 and score >= self.confidence) for x, y, score in ankles)

    def _crop_pose_merge(self, image: np.ndarray, pose_people: list[Person]) -> list[Person]:
        """Keep every direct pose person; add crop-pose people for unmatched det boxes."""
        image_h, image_w = image.shape[:2]
        det_people = self._parse(self._det_model.predict(**self._opts(image)),
                                 (image_h, image_w), "bbox")

        # Greedy one-to-one IoU matching: one pose person cannot absorb two det boxes.
        pairs = sorted(((_box_iou(d.bbox, p.bbox), i, j)
                        for i, d in enumerate(det_people)
                        for j, p in enumerate(pose_people)), reverse=True)
        used_det: set[int] = set()
        used_pose: set[int] = set()
        for iou, i, j in pairs:
            if iou < self.crop_match_iou:
                break
            if i not in used_det and j not in used_pose:
                used_det.add(i)
                used_pose.add(j)

        merged = list(pose_people)
        for i, det in enumerate(det_people):
            if i in used_det:
                continue
            bx1, by1, bx2, by2 = det.bbox
            pad = int((by2 - by1) * 0.15)
            cx1, cy1 = max(0, int(bx1 - pad)), max(0, int(by1 - pad))
            cx2, cy2 = min(image_w, int(bx2 + pad)), min(image_h, int(by2 + pad))
            crop = image[cy1:cy2, cx1:cx2]
            ankles: list[tuple[float, float, float]] = []
            if crop.size > 0 and crop.shape[0] >= 16 and crop.shape[1] >= 16:
                sub_out = self.model.predict(**self._opts(
                    crop, conf=max(0.15, self.confidence * 0.7), imgsz=self.crop_pose_imgsz))
                candidates = self._parse(sub_out, (image_h, image_w), "pose", offset=(cx1, cy1))
                # A crop may contain bystanders: use the pose person that best overlaps
                # the target det box, never simply the first result.
                best = max(candidates, key=lambda p: _box_iou(p.bbox, det.bbox), default=None)
                if best is not None and _box_iou(best.bbox, det.bbox) >= 0.3:
                    ankles = list(best.ankles or [])
            metadata = dict(self.last_config)
            self._pose_metadata(metadata, det.bbox, ankles, image_h, image_w)
            metadata["crop_pose_added"] = True
            merged.append(Person(
                det.bbox, det.confidence, ankles=ankles,
                runtime_ms=self.last_runtime_ms,
                detector="yolo_pose_crop", metadata=metadata,
            ))
        return merged
