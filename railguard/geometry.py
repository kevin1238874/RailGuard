"""Camera-space L/E geometry and orientation-independent M4 point rules.

``L`` is the safety-line edge, ``E`` is the platform edge, and ``Z`` is the
finite quadrilateral between them. The calibrated unsafe direction is from L
toward E and then into the half-plane beyond E. It is derived from both lines,
never from increasing image y, so mirrored, diagonal, and vertical views use
the same rules.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

Point = tuple[float, float]
VALID_STATUSES = {"SAFE", "DANGER", "UNKNOWN"}
UNSAFE_SIDE_CONVENTION = "beyond_platform_edge"


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

    @property
    def midpoint(self) -> Point:
        return ((self.a[0] + self.b[0]) / 2.0,
                (self.a[1] + self.b[1]) / 2.0)

    def y_at(self, x: float) -> float | None:
        if abs(self.b[0] - self.a[0]) < 1e-10:
            return None
        return self.a[1] + (x - self.a[0]) * (self.b[1] - self.a[1]) / (self.b[0] - self.a[0])

    def raw_signed_distance(self, point: Point) -> float:
        """Signed perpendicular distance; sign initially follows endpoint order."""
        length = self.length()
        if not math.isfinite(length) or length <= 1e-10:
            raise ValueError("Line endpoints must be finite and distinct")
        x, y = map(float, point)
        return ((self.b[0] - self.a[0]) * (y - self.a[1])
                - (self.b[1] - self.a[1]) * (x - self.a[0])) / length

    def projection(self, point: Point) -> float:
        """Return the point's longitudinal parameter (0=a, 1=b)."""
        dx, dy = self.b[0] - self.a[0], self.b[1] - self.a[1]
        denominator = dx * dx + dy * dy
        if not math.isfinite(denominator) or denominator <= 1e-20:
            raise ValueError("Line endpoints must be finite and distinct")
        return ((point[0] - self.a[0]) * dx + (point[1] - self.a[1]) * dy) / denominator


@dataclass(frozen=True)
class PointAssessment:
    point: Point
    status: str
    reason: str
    reasons: tuple[str, ...]
    signed_distance_to_l: float | None
    signed_distance_to_e: float | None
    inside_z: bool | None

    def as_dict(self) -> dict:
        return {
            "point": list(self.point),
            "status": self.status,
            "reason": self.reason,
            "reasons": list(self.reasons),
            "signed_distance_to_L": self.signed_distance_to_l,
            "signed_distance_to_E": self.signed_distance_to_e,
            "inside_Z": self.inside_z,
        }


@dataclass(frozen=True)
class HazardZone:
    safety_line: Segment
    platform_edge: Segment
    # E locates the direction; this convention says the unsafe half-plane
    # continues away from L after E rather than ending at finite polygon Z.
    unsafe_side: str = UNSAFE_SIDE_CONVENTION

    @classmethod
    def from_dict(cls, raw: dict) -> "HazardZone":
        if not isinstance(raw, dict):
            raise ValueError("zone must be a mapping")
        return cls(
            Segment.from_list(raw["safety_line"]),
            Segment.from_list(raw["platform_edge"]),
            raw.get("unsafe_side", UNSAFE_SIDE_CONVENTION),
        )

    def to_dict(self) -> dict:
        return {
            "safety_line": [list(self.safety_line.a), list(self.safety_line.b)],
            "platform_edge": [list(self.platform_edge.a), list(self.platform_edge.b)],
            "unsafe_side": self.unsafe_side,
        }

    def polygon(self) -> np.ndarray:
        if self.unsafe_side != UNSAFE_SIDE_CONVENTION:
            raise ValueError(
                f"unsafe_side must be {UNSAFE_SIDE_CONVENTION!r}; got {self.unsafe_side!r}")
        # Sort endpoints along L's dominant axis to avoid dependence on click order.
        use_x = (abs(self.safety_line.b[0] - self.safety_line.a[0]) >=
                 abs(self.safety_line.b[1] - self.safety_line.a[1]))
        index = 0 if use_x else 1
        left = sorted((self.safety_line.a, self.safety_line.b), key=lambda p: p[index])
        right = sorted((self.platform_edge.a, self.platform_edge.b), key=lambda p: p[index])
        poly = np.asarray([left[0], left[1], right[1], right[0]], dtype=np.float32)
        if not np.isfinite(poly).all():
            raise ValueError("Line coordinates must be finite")
        area = abs(float(cv2.contourArea(poly)))
        if area < 1e-5:
            raise ValueError("Degenerate risk zone: the line pair encloses no area")
        if (_intersect(poly[0], poly[1], poly[2], poly[3])
                or _intersect(poly[1], poly[2], poly[3], poly[0])):
            raise ValueError("Invalid crossing line endpoints for risk zone")
        if abs(self.safety_line.raw_signed_distance(self.platform_edge.midpoint)) < 1e-7:
            raise ValueError("Invalid calibration: E does not define an unsafe side of L")
        if abs(self.platform_edge.raw_signed_distance(self.safety_line.midpoint)) < 1e-7:
            raise ValueError("Invalid calibration: L and E do not define opposite sides of E")
        return poly

    def contains(self, point: Point, tolerance_px: float = 0.0) -> bool:
        if not math.isfinite(tolerance_px) or tolerance_px < 0:
            raise ValueError("tolerance_px must be finite and >= 0")
        return cv2.pointPolygonTest(self.polygon(), tuple(map(float, point)), True) >= -tolerance_px

    def corridor_polygon(self, width: int, height: int, y_top: float | None = None) -> np.ndarray:
        """Compute the continuous hazard corridor polygon between L and E,

        clipped cleanly to the image rectangle [0, width-1] x [y_top, height-1].
        Properly handles perspective vanishing points and frame corners (e.g. bottom-left
        or bottom-right platform cutoffs) without leaving triangular gaps.
        """
        def get_x(pA, pB, y):
            dy = pB[1] - pA[1]
            dx = pB[0] - pA[0]
            if abs(dy) < 1e-7:
                return float(pA[0])
            return float(pA[0] + dx * (y - pA[1]) / dy)

        def get_y(pA, pB, x):
            dx = pB[0] - pA[0]
            dy = pB[1] - pA[1]
            if abs(dx) < 1e-7:
                return float(pA[1])
            return float(pA[1] + dy * (x - pA[0]) / dx)

        p1, p2 = np.asarray(self.safety_line.a, float), np.asarray(self.safety_line.b, float)
        q1, q2 = np.asarray(self.platform_edge.a, float), np.asarray(self.platform_edge.b, float)

        # Vanishing point perspective check
        A = np.column_stack([p2 - p1, -(q2 - q1)])
        b = q1 - p1
        if y_top is None:
            y_top = 0.0
            if abs(float(np.linalg.det(A))) > 1e-5:
                params = np.linalg.solve(A, b)
                inter = p1 + float(params[0]) * (p2 - p1)
                if 0 <= inter[1] < height * 0.75:
                    y_top = float(min(height - 10, max(0, inter[1] + 15)))

        y_bot = float(height - 1)
        y_events = {float(y_top), float(y_bot)}
        for pA, pB in [(p1, p2), (q1, q2)]:
            for x_edge in [0.0, float(width - 1)]:
                y_edge = get_y(pA, pB, x_edge)
                if y_top < y_edge < y_bot:
                    y_events.add(float(y_edge))

        sorted_y = sorted(y_events)
        right_pts = []
        for y in sorted_y:
            xL = get_x(p1, p2, y)
            xE = get_x(q1, q2, y)
            xr = float(np.clip(max(xL, xE), 0, width - 1))
            if not right_pts or abs(right_pts[-1][0] - xr) > 1e-3 or abs(right_pts[-1][1] - y) > 1e-3:
                right_pts.append((xr, y))

        left_pts = []
        for y in reversed(sorted_y):
            xL = get_x(p1, p2, y)
            xE = get_x(q1, q2, y)
            xl = float(np.clip(min(xL, xE), 0, width - 1))
            if not left_pts or abs(left_pts[-1][0] - xl) > 1e-3 or abs(left_pts[-1][1] - y) > 1e-3:
                left_pts.append((xl, y))

        poly = right_pts + left_pts
        return np.array(poly, dtype=np.float32)

    def signed_distances(self, point: Point) -> tuple[float, float]:
        """Distances positive in the calibrated L -> E -> track direction."""
        self.polygon()
        d_l_raw = self.safety_line.raw_signed_distance(point)
        l_reference = self.safety_line.raw_signed_distance(self.platform_edge.midpoint)
        d_l = d_l_raw * (1.0 if l_reference > 0 else -1.0)

        # Unsafe of E is the side opposite L, independent of endpoint order.
        d_e_raw = self.platform_edge.raw_signed_distance(point)
        safe_reference = self.platform_edge.raw_signed_distance(self.safety_line.midpoint)
        d_e = d_e_raw * (-1.0 if safe_reference > 0 else 1.0)
        return float(d_l), float(d_e)

    def observable_horizon_y(self, width: int, height: int) -> float | None:
        """Return visible horizon y-clip when L and E converge to a forward VP inside frame."""
        if width < 2 or height < 2:
            return None
        vp = _line_intersection(self.safety_line, self.platform_edge)
        if vp is None:
            return None
        s_dx = abs(self.safety_line.b[0] - self.safety_line.a[0])
        s_dy = abs(self.safety_line.b[1] - self.safety_line.a[1])
        e_dx = abs(self.platform_edge.b[0] - self.platform_edge.a[0])
        e_dy = abs(self.platform_edge.b[1] - self.platform_edge.a[1])
        if s_dy < s_dx * 0.4 or e_dy < e_dx * 0.4:
            return None
        if abs(self.safety_line.b[1] - self.safety_line.a[1]) < 1e-6 or abs(self.platform_edge.b[1] - self.platform_edge.a[1]) < 1e-6:
            return None
        y_mid = 0.25 * (self.safety_line.a[1] + self.safety_line.b[1]
                        + self.platform_edge.a[1] + self.platform_edge.b[1])
        if not (0.0 <= vp[1] < y_mid and vp[1] < float(height - 1)):
            return None
        slope_l = (self.safety_line.b[0] - self.safety_line.a[0]) / (self.safety_line.b[1] - self.safety_line.a[1])
        slope_e = (self.platform_edge.b[0] - self.platform_edge.a[0]) / (self.platform_edge.b[1] - self.platform_edge.a[1])
        rate = abs(slope_l - slope_e)
        if rate < 1e-6:
            return None
        min_sep = max(4.0, 0.012 * float(min(width, height)))
        y_top = vp[1] + min_sep / rate
        if y_top >= float(height - 1):
            return None
        return float(max(0.0, y_top))

    def clipped_boundary_segments(self, width: int, height: int) -> tuple[Segment, Segment]:
        """Clip L and E to image bounds and finite forward vanishing-point horizon."""
        horizon_y = self.observable_horizon_y(width, height)
        y_min = horizon_y if horizon_y is not None else 0.0
        clipped_l = _clip_line_to_rect(self.safety_line, 0.0, y_min, float(width - 1), float(height - 1))
        clipped_e = _clip_line_to_rect(self.platform_edge, 0.0, y_min, float(width - 1), float(height - 1))
        return clipped_l or self.safety_line, clipped_e or self.platform_edge

    def assess_point(self, point: Point, image_shape: tuple[int, int], *,
                     tolerance_px: float = 0.0,
                     image_edge_margin_px: float = 1.0) -> PointAssessment:
        """Apply the M4 tri-state rule to one observed footpoint."""
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            return _assessment((math.nan, math.nan), "UNKNOWN", "invalid_footpoint")
        try:
            point = (float(point[0]), float(point[1]))
        except (TypeError, ValueError):
            return _assessment((math.nan, math.nan), "UNKNOWN", "invalid_footpoint")
        if not all(math.isfinite(value) for value in point):
            return _assessment((math.nan, math.nan), "UNKNOWN", "invalid_footpoint")
        if (not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2
                or image_shape[0] < 2 or image_shape[1] < 2):
            return _assessment(point, "UNKNOWN", "invalid_image_dimensions")
        if (not math.isfinite(tolerance_px) or tolerance_px < 0
                or not math.isfinite(image_edge_margin_px) or image_edge_margin_px < 0):
            raise ValueError("tolerances must be finite and >= 0")

        d_l, d_e = self.signed_distances(point)
        height, width = int(image_shape[0]), int(image_shape[1])
        x, y = point
        if x < 0 or y < 0 or x > width - 1 or y > height - 1:
            return _assessment(point, "UNKNOWN", "point_out_of_bounds", d_l, d_e, None)

        # Above a finite perspective horizon L and E cross/invert, so classification is indeterminate.
        horizon_y = self.observable_horizon_y(width, height)
        if horizon_y is not None and y < horizon_y:
            return _assessment(point, "UNKNOWN", "outside_observable_zone", d_l, d_e, False)

        near_edge = (x <= image_edge_margin_px or y <= image_edge_margin_px
                     or x >= width - 1 - image_edge_margin_px
                     or y >= height - 1 - image_edge_margin_px)
        inside_z = self.contains(point, tolerance_px=tolerance_px)
        if not inside_z and height >= 2 and width >= 2:
            try:
                corr = self.corridor_polygon(width, height)
                inside_z = bool(cv2.pointPolygonTest(corr, tuple(map(float, point)), True) >= -tolerance_px)
            except Exception:
                pass
        longitudinal_slack_l = tolerance_px / max(self.safety_line.length(), 1e-12)
        longitudinal_slack_e = tolerance_px / max(self.platform_edge.length(), 1e-12)
        on_l_segment = -longitudinal_slack_l <= self.safety_line.projection(point) <= 1 + longitudinal_slack_l
        on_e_segment = -longitudinal_slack_e <= self.platform_edge.projection(point) <= 1 + longitudinal_slack_e

        # Beyond-E intentionally precedes finite-Z: leaving the polygon past E
        # cannot turn an observed violation into SAFE.
        if d_e > tolerance_px:
            reasons = (("beyond_platform_edge", "near_image_edge") if near_edge
                       else ("beyond_platform_edge",))
            return _assessment(point, "DANGER", reasons[0], d_l, d_e, inside_z, reasons)
        if abs(d_e) <= tolerance_px and on_e_segment:
            reasons = (("on_platform_edge", "near_image_edge") if near_edge
                       else ("on_platform_edge",))
            return _assessment(point, "DANGER", reasons[0], d_l, d_e, inside_z, reasons)
        if abs(d_l) <= tolerance_px and on_l_segment:
            reasons = (("on_safety_line", "near_image_edge") if near_edge
                       else ("on_safety_line",))
            return _assessment(point, "DANGER", reasons[0], d_l, d_e, inside_z, reasons)
        if inside_z:
            reasons = (("inside_hazard_zone", "near_image_edge") if near_edge
                       else ("inside_hazard_zone",))
            return _assessment(point, "DANGER", reasons[0], d_l, d_e, inside_z, reasons)
        if near_edge:
            return _assessment(point, "UNKNOWN", "near_image_edge", d_l, d_e, inside_z)
        if d_l < -tolerance_px and on_l_segment:
            return _assessment(point, "SAFE", "safe_side_of_safety_line", d_l, d_e, inside_z)
        if d_l >= -tolerance_px and d_e <= tolerance_px:
            return _assessment(point, "UNKNOWN", "outside_observable_zone", d_l, d_e, inside_z)
        return _assessment(point, "UNKNOWN", "inconsistent_line_relationship", d_l, d_e, inside_z)


def _assessment(point: Point, status: str, reason: str,
                d_l: float | None = None, d_e: float | None = None,
                inside_z: bool | None = None,
                reasons: tuple[str, ...] | None = None) -> PointAssessment:
    return PointAssessment(point, status, reason, reasons or (reason,), d_l, d_e, inside_z)


def _line_intersection(seg_a: Segment, seg_b: Segment) -> Point | None:
    x1, y1 = seg_a.a
    x2, y2 = seg_a.b
    x3, y3 = seg_b.a
    x4, y4 = seg_b.b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) <= 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    if not (math.isfinite(px) and math.isfinite(py)):
        return None
    return float(px), float(py)


def _clip_line_to_rect(seg: Segment, x_min: float, y_min: float,
                       x_max: float, y_max: float) -> Segment | None:
    x1, y1 = seg.a
    x2, y2 = seg.b
    dx, dy = x2 - x1, y2 - y1
    hits: list[Point] = []
    if abs(dy) > 1e-9:
        for yb in (y_min, y_max):
            t = (yb - y1) / dy
            xb = x1 + t * dx
            if x_min - 1e-6 <= xb <= x_max + 1e-6:
                hits.append((float(min(x_max, max(x_min, xb))), float(yb)))
    if abs(dx) > 1e-9:
        for xb in (x_min, x_max):
            t = (xb - x1) / dx
            yb = y1 + t * dy
            if y_min - 1e-6 <= yb <= y_max + 1e-6:
                hits.append((float(xb), float(min(y_max, max(y_min, yb)))))
    unique: list[Point] = []
    for pt in hits:
        if not any(math.hypot(pt[0] - q[0], pt[1] - q[1]) < 1e-4 for q in unique):
            unique.append(pt)
    if len(unique) < 2:
        return None
    # Sort hits along the segment's direction to keep deterministic ordering.
    unique.sort(key=lambda pt: (pt[0] - x1) * dx + (pt[1] - y1) * dy)
    return Segment(unique[0], unique[-1])


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

    return HazardZone(rescale(zone.safety_line), rescale(zone.platform_edge), zone.unsafe_side)
