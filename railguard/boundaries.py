"""Conservative orientation-independent candidate safety-zone estimation.

HSV paint cues and Canny/Hough line segments are suggestions only: their
semantics (paint vs. platform edge) cannot be guaranteed from RGB alone.
Always inspect the suggested polygon before committing a calibration.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

import cv2
import numpy as np

from .geometry import Segment, HazardZone


@dataclass
class LineResult:
    zone: HazardZone | None
    status: str
    diagnostics: dict


def paint_mask(image: np.ndarray, color: str = "both", clahe: bool = False) -> np.ndarray:
    if clahe:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(l)
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (16, 55, 100), (42, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 165), (179, 68, 255))
    if color == "yellow":
        mask = yellow
    elif color == "white":
        mask = white
    elif color == "both":
        mask = cv2.bitwise_or(yellow, white)
    else:
        raise ValueError("color must be yellow, white or both")
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _candidate_segments(image: np.ndarray, mode: str, color: str, clahe: bool) -> list[Segment]:
    h, w = image.shape[:2]
    if mode == "color":
        edges = cv2.Canny(paint_mask(image, color, clahe), 35, 110)
    elif mode == "canny":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 65, 145)
    else:
        raise ValueError("Unknown line extractor")
    raw = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=max(18, min(h, w) // 14),
        minLineLength=max(24, round(min(h, w) * .20)),
        maxLineGap=max(10, round(min(h, w) * .06)),
    )
    if raw is None:
        return []
    segments = []
    for x0, y0, x1, y1 in raw[:, 0, :]:
        seg = Segment((float(x0), float(y0)), (float(x1), float(y1)))
        if seg.length() >= min(h, w) * .19:
            segments.append(seg)
    return segments


def _direction(seg: Segment) -> np.ndarray:
    vector = np.array(seg.b, float) - np.array(seg.a, float)
    vector /= np.linalg.norm(vector)
    if vector[0] < -1e-9 or (abs(vector[0]) < 1e-9 and vector[1] < 0):
        vector *= -1
    return vector


def _extend_to_image(point: np.ndarray, direction: np.ndarray,
                     width: int, height: int) -> Segment | None:
    intersections: list[tuple[float, float]] = []
    for dim, value, other_dim, upper in ((0, 0., 1, height - 1.),
                                          (0, width - 1., 1, height - 1.),
                                          (1, 0., 0, width - 1.),
                                          (1, height - 1., 0, width - 1.)):
        if abs(direction[dim]) <= 1e-9:
            continue
        t = (value - point[dim]) / direction[dim]
        candidate = point + t * direction
        if -1e-5 <= candidate[other_dim] <= upper + 1e-5:
            xy = (float(np.clip(candidate[0], 0, width - 1)),
                  float(np.clip(candidate[1], 0, height - 1)))
            if not any(math.dist(xy, p) <= .01 for p in intersections):
                intersections.append(xy)
    if len(intersections) < 2:
        return None
    first, second = max(
        ((a, b) for i, a in enumerate(intersections) for b in intersections[i + 1:]),
        key=lambda pair: math.dist(*pair),
    )
    return Segment(first, second)


def _fit_segments(segments: list[Segment], width: int, height: int,
                  distance_px: float = 12.0) -> list[tuple[Segment, float]]:
    """Group near-collinear lines regardless of angle; return image-clipped lines."""
    if not segments:
        return []
    ranked: list[tuple[float, Segment, list[Segment]]] = []
    for seed in segments:
        normal = np.array([-_direction(seed)[1], _direction(seed)[0]])
        center = (np.asarray(seed.a) + np.asarray(seed.b)) / 2
        supporting = []
        for item in segments:
            if abs(float(np.dot(_direction(seed), _direction(item)))) < math.cos(math.radians(9)):
                continue
            midpoint = (np.asarray(item.a) + np.asarray(item.b)) / 2
            if abs(float(np.dot(midpoint - center, normal))) < distance_px:
                supporting.append(item)
        ranked.append((sum(s.length() for s in supporting), seed, supporting))
    ranked.sort(key=lambda x: x[0], reverse=True)
    fitted: list[tuple[Segment, float]] = []
    for score, seed, members in ranked:
        pts = np.asarray([p for member in members for p in (member.a, member.b)], np.float32)
        if len(pts) < 2:
            continue
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, .01, .01).reshape(-1)
        line = _extend_to_image(np.asarray([x0, y0], float),
                                np.asarray([vx, vy], float), width, height)
        if line is None:
            continue
        mid = (np.asarray(line.a) + np.asarray(line.b)) / 2
        if any(abs(float(_direction(existing)[0] * (mid[1] - existing.a[1]) -
                         _direction(existing)[1] * (mid[0] - existing.a[0]))) <= distance_px
               and abs(float(np.dot(_direction(existing), _direction(line)))) >= .97
               for existing, _ in fitted):
            continue
        fitted.append((line, score))
        if len(fitted) >= 12:
            break
    return fitted


def estimate_zone(image: np.ndarray, method: str = "color_hough", color: str = "both",
                  clahe: bool = False, min_gap_frac: float = .06,
                  max_gap_frac: float = .78) -> LineResult:
    """Suggest paint-line/edge pairs, including steep and near-vertical lines.

    Returns no zone when evidence is insufficient. Even a nonempty candidate
    requires human semantic verification. No automatic result is ground truth.
    """
    if method not in {"color_hough", "canny_hough", "color_ransac", "hybrid"}:
        raise ValueError(f"Unknown method: {method}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected BGR color image")
    h, w = image.shape[:2]
    if w < 32 or h < 32:
        return LineResult(None, "uncertain_image_too_small", {})
    paint = _candidate_segments(image, "color", color, clahe)
    edges = _candidate_segments(image, "canny", color, clahe)
    threshold = max(5, min(h, w) * (.022 if method in {"color_ransac", "hybrid"} else .015))
    colored_lines = _fit_segments(paint, w, h, threshold)
    edge_lines = _fit_segments(edges, w, h, threshold)
    if method in {"color_hough", "color_ransac", "hybrid"} and not colored_lines:
        return LineResult(None, "uncertain_no_paint_line", {"paint_segments": len(paint), "edge_segments": len(edges)})
    anchors = edge_lines if method == "canny_hough" else colored_lines
    options: list[tuple[float, HazardZone]] = []
    for a, a_support in anchors:
        a_dir = _direction(a)
        a_normal = np.array([-a_dir[1], a_dir[0]])
        a_mid = (np.asarray(a.a) + np.asarray(a.b)) / 2
        for b, b_support in edge_lines:
            if abs(float(np.dot(a_dir, _direction(b)))) < math.cos(math.radians(10)):
                continue
            b_mid = (np.asarray(b.a) + np.asarray(b.b)) / 2
            gap = abs(float(np.dot(b_mid - a_mid, a_normal)))
            if not (min(h, w) * min_gap_frac <= gap <= min(h, w) * max_gap_frac):
                continue
            try:
                zone = HazardZone(a, b)
                zone.polygon()
            except ValueError:
                continue
            # Prefer strong, distinct parallel structures; never conflate score
            # with a calibrated confidence in the lines' real-world semantics.
            support = min(a_support, a.length() * 2) + min(b_support, b.length() * 2)
            score = support - .10 * gap
            options.append((score, zone))
    diagnostics = {"method": method, "paint_segments": len(paint),
                   "edge_segments": len(edges), "paint_lines": len(colored_lines),
                   "edge_lines": len(edge_lines), "candidate_pairs": len(options)}
    if not options:
        return LineResult(None, "uncertain_no_pair", diagnostics)
    zone = max(options, key=lambda item: item[0])[1]
    return LineResult(zone, "estimated_review_required", diagnostics)
