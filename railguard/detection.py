"""Detections and Ultralytics adapters (weights downloaded on first inference)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

Point = tuple[float, float]


@dataclass
class Person:
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0
    object_id: Optional[str] = None
    ankles: Optional[list[tuple[float, float, float]]] = None  # (x,y,score) pairs

    def foot_points(self, method: str = "bbox", min_conf: float = 0.35,
                    fallback: str = "bbox") -> tuple[list[Point], str]:
        if method not in {"bbox", "pose"}:
            raise ValueError("method must be bbox or pose")
        if fallback not in {"bbox", "abstain"}:
            raise ValueError("fallback must be bbox or abstain")
        x1, _, x2, y2 = self.bbox
        bottom_center = ((x1 + x2) / 2.0, y2)
        if method == "bbox":
            return [bottom_center], "bbox_bottom_center"
        reliable = [(x, y) for x, y, conf in (self.ankles or [])
                    if conf >= min_conf and x > 0 and y > 0]
        if reliable:
            return reliable, "two_ankles" if len(reliable) > 1 else "one_ankle"
        return ([bottom_center], "bbox_fallback") if fallback == "bbox" else ([], "missing_keypoints")


class YOLODetector:
    """COCO person class=0. Custom weights must use the same class index."""

    def __init__(self, method: str = "bbox", weights: str | None = None,
                 confidence: float = 0.25, device: str | None = None):
        if method not in {"bbox", "pose"}:
            raise ValueError("method must be bbox or pose")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics missing; run: pip install -e '.[vision]'") from exc
        self.method, self.confidence, self.device = method, confidence, device
        self.model = YOLO(weights or ("yolo11n-pose.pt" if method == "pose" else "yolo11n.pt"))

    def predict(self, image) -> list[Person]:
        opts = {"source": image, "verbose": False, "conf": self.confidence}
        if self.device:
            opts["device"] = self.device
        output = self.model.predict(**opts)[0]
        if output.boxes is None:
            return []
        people = []
        for index, box in enumerate(output.boxes):
            if int(box.cls[0].item()) != 0:
                continue
            bbox = tuple(map(float, box.xyxy[0].cpu().tolist()))
            ankles = None
            if self.method == "pose" and output.keypoints is not None:
                points = output.keypoints.xy[index].cpu().tolist()
                conf = output.keypoints.conf
                scores = conf[index].cpu().tolist() if conf is not None else [1.0] * len(points)
                if len(points) >= 17:
                    ankles = [(float(points[k][0]), float(points[k][1]), float(scores[k]))
                              for k in (15, 16)]  # COCO ankle indices
            people.append(Person(bbox, float(box.conf[0].item()), ankles=ankles))
        return people
