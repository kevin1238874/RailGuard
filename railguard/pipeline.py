"""Geometric risk analysis; no location is guessed without explicit opt-in."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from .boundaries import estimate_zone
from .detection import Person
from .geometry import HazardZone


@dataclass
class Decision:
    bbox: tuple[float, float, float, float]
    confidence: float
    foot_points: list[tuple[float, float]]
    foot_source: str
    intrusion: bool | None
    object_id: str | None = None

    def as_dict(self) -> dict:
        return {"bbox": list(self.bbox), "confidence": self.confidence,
                "foot_points": [list(p) for p in self.foot_points],
                "foot_source": self.foot_source, "intrusion": self.intrusion,
                "object_id": self.object_id}


def analyze(image: np.ndarray, people: list[Person], zone: HazardZone | None = None,
            foot_method: str = "bbox", auto_boundary: bool = False,
            boundary_method: str = "color_hough", paint_color: str = "both",
            tolerance_px: float = 0.0, keypoint_confidence: float = 0.35,
            pose_fallback: str = "bbox") -> tuple[list[Decision], HazardZone | None, str]:
    status = "manually_calibrated" if zone else "missing_calibration"
    if zone is None and auto_boundary:
        estimate = estimate_zone(image, method=boundary_method, color=paint_color)
        zone, status = estimate.zone, estimate.status
    if zone is None:
        return [], None, status
    decisions = []
    for person in people:
        feet, source = person.foot_points(foot_method, keypoint_confidence, pose_fallback)
        intrusion = any(zone.contains(point, tolerance_px=tolerance_px) for point in feet) if feet else None
        decisions.append(Decision(person.bbox, person.confidence, feet, source,
                                  intrusion, person.object_id))
    return decisions, zone, status


def draw(image: np.ndarray, decisions: list[Decision], zone: HazardZone | None,
         status: str = "") -> np.ndarray:
    canvas = image.copy()
    if zone is not None:
        poly = zone.polygon().astype(np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [poly], (0, 160, 240))
        cv2.addWeighted(overlay, 0.24, canvas, 0.76, 0, canvas)
        for seg, color in ((zone.safety_line, (0, 220, 240)),
                           (zone.platform_edge, (255, 145, 30))):
            cv2.line(canvas, tuple(map(int, seg.a)), tuple(map(int, seg.b)), color, 2)
    for dec in decisions:
        color = (35, 55, 235) if dec.intrusion is True else ((50, 180, 65) if dec.intrusion is False else (180, 180, 180))
        x1, y1, x2, y2 = map(int, dec.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        for foot in dec.foot_points:
            cv2.circle(canvas, tuple(map(int, foot)), 5, color, -1)
        tag = "DANGER" if dec.intrusion is True else "SAFE" if dec.intrusion is False else "UNKNOWN"
        cv2.putText(canvas, tag, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(canvas, status[:75], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)
    return canvas
