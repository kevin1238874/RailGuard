"""Two camera-space lines L (safety) and E (platform edge) define risk zone Z.

The area between the projected segments is a quadrilateral; any-point ankle
intrusion is used for the pose approach. Image origin is top-left, y downwards.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import cv2
import numpy as np

Point = tuple[float, float]


@dataclass(frozen=True)
class Segment:
    a: Point
    b: Point

    @classmethod
    def from_list(cls, points) -> "Segment":
        if not isinstance(points, (list, tuple)) or len(points) != 2 or any(
            not isinstance(p, (list, tuple)) or len(p) != 2 for p in points
        ):
            raise ValueError("Each line must have exactly two [x, y] points")
        arr = [tuple(map(float, p)) for p in points]
        if not all(math.isfinite(v) for p in arr for v in p):
            raise ValueError("Line coordinates must be finite")
        if arr[0] == arr[1]:
            raise ValueError("Line endpoints must be distinct")
        return cls(arr[0], arr[1])

    def length(self) -> float:
        return math.dist(self.a, self.b)

    def y_at(self, x: float) -> float | None:
        if abs(self.b[0] - self.a[0]) < 1e-10:
            return None
        return self.a[1] + (x - self.a[0]) * (self.b[1] - self.a[1]) / (self.b[0] - self.a[0])


@dataclass(frozen=True)
class HazardZone:
    safety_line: Segment
    platform_edge: Segment

    @classmethod
    def from_dict(cls, raw: dict) -> "HazardZone":
        return cls(Segment.from_list(raw["safety_line"]),
                   Segment.from_list(raw["platform_edge"]))

    def to_dict(self) -> dict:
        return {"safety_line": [list(self.safety_line.a), list(self.safety_line.b)],
                "platform_edge": [list(self.platform_edge.a), list(self.platform_edge.b)]}

    def polygon(self) -> np.ndarray:
        # Sort the endpoints along the dominant axis to avoid crossed connections.
        ends = self.safety_line.a, self.safety_line.b, self.platform_edge.a, self.platform_edge.b
        use_x = (abs(self.safety_line.b[0] - self.safety_line.a[0]) >=
                 abs(self.safety_line.b[1] - self.safety_line.a[1]))
        index = 0 if use_x else 1
        left = sorted((self.safety_line.a, self.safety_line.b), key=lambda p: p[index])
        right = sorted((self.platform_edge.a, self.platform_edge.b), key=lambda p: p[index])
        poly = np.asarray([left[0], left[1], right[1], right[0]], dtype=np.float32)
        area = abs(float(cv2.contourArea(poly)))
        if area < 1e-5:
            raise ValueError("Degenerate risk zone: the line pair encloses no area")
        # Reject a bow-tie polygon where consecutive edges cross.
        if _intersect(poly[0], poly[1], poly[2], poly[3]) or _intersect(poly[1], poly[2], poly[3], poly[0]):
            raise ValueError("Invalid crossing line endpoints for risk zone")
        return poly

    def contains(self, point: Point, tolerance_px: float = 0.0) -> bool:
        if tolerance_px < 0:
            raise ValueError("tolerance_px must be >= 0")
        return cv2.pointPolygonTest(self.polygon(), tuple(map(float, point)), True) >= -tolerance_px


def _intersect(a, b, c, d) -> bool:
    def orient(p, q, r):
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    v1, v2, v3, v4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return v1 * v2 < 0 and v3 * v4 < 0


def normalized_to_pixels(zone: HazardZone, width: int, height: int) -> HazardZone:
    if width < 2 or height < 2:
        raise ValueError("Image dimensions must be >=2")
    def rescale(line):
        return Segment((line.a[0] * (width - 1), line.a[1] * (height - 1)),
                       (line.b[0] * (width - 1), line.b[1] * (height - 1)))
    return HazardZone(rescale(zone.safety_line), rescale(zone.platform_edge))
