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


def _axis_point(line: Segment, axis: int, value: float) -> np.ndarray | None:
    """Point on an infinite fitted line at a given x or y (perspective-safe)."""
    a, b = np.asarray(line.a, dtype=float), np.asarray(line.b, dtype=float)
    if abs(b[axis] - a[axis]) < 1e-6:
        return None
    return a + (value - a[axis]) * (b - a) / (b[axis] - a[axis])


def _sample_gray(gray: np.ndarray, point: np.ndarray) -> float | None:
    x, y = np.rint(point).astype(int)
    h, w = gray.shape
    if x < 2 or y < 2 or x >= w - 2 or y >= h - 2:
        return None
    return float(np.mean(gray[y-2:y+3, x-2:x+3]))


def _edge_evidence(gray: np.ndarray, line: Segment) -> tuple[float, float]:
    """Contrast across a proposed edge and center-dark line evidence.

    A solid dark synthetic line has a dark core; a true platform edge usually
    separates a brighter platform from a darker train/platform gap. Both can
    have strong Canny edges, so this cue is just one factor, not a classifier.
    """
    h, w = gray.shape
    tangent = _direction(line)
    normal = np.array([-tangent[1], tangent[0]])
    offset = max(8.0, min(w, h) * .035)
    contrasts, cores = [], []
    for t in np.linspace(.12, .88, 29):
        p = np.asarray(line.a) * (1-t) + np.asarray(line.b) * t
        left = _sample_gray(gray, p + normal * offset)
        right = _sample_gray(gray, p - normal * offset)
        center = _sample_gray(gray, p)
        if None in (left, right, center):
            continue
        contrasts.append(abs(left - right))
        cores.append(max(0., .5 * (left + right) - center))
    if len(contrasts) < 9:
        return 0., 0.
    return float(np.median(contrasts)), float(np.median(cores))


def _zone_evidence(gray: np.ndarray, a: Segment, b: Segment) -> tuple[float, float, float] | None:
    """Measure gap and surface brightness *between* two lines.

    We sample matched x (for near-horizontal) or y (for steep) coordinates;
    this avoids erroneous quadrilaterals formed by independently clipped
    diagonal lines with differing endpoints.
    """
    h, w = gray.shape
    direction = _direction(a)
    axis = 0 if abs(direction[0]) >= abs(direction[1]) else 1
    limit = w - 1 if axis == 0 else h - 1
    gaps, inside = [], []
    for t in np.linspace(.15, .85, 15):
        v = limit * t
        pa = _axis_point(a, axis, v)
        pb = _axis_point(b, axis, v)
        if pa is None or pb is None:
            continue
        gap = float(np.linalg.norm(pb - pa))
        samples = [_sample_gray(gray, pa*(1-u)+pb*u) for u in (.25,.50,.75)]
        valid = [val for val in samples if val is not None]
        if len(valid) < 2:
            continue
        inside.extend(valid)
        gaps.append(gap)
    if len(gaps) < 8:
        return None
    return float(np.median(gaps)), float(np.median(inside)), float(np.mean(np.asarray(inside) > 88))


def _outside_dark_drop(gray: np.ndarray, a: Segment, b: Segment) -> float:
    """Positive if the area *outside* the edge is darker than platform interior.

    Crucial distinction: both edges of a paint stripe are strong Hough lines,
    but neither separates a well-lit platform from a dark track gap.
    """
    h, w = gray.shape
    tangent = _direction(a)
    axis = 0 if abs(tangent[0]) >= abs(tangent[1]) else 1
    limit = w - 1 if axis == 0 else h - 1
    differences = []
    offset = max(12., min(h, w) * .042)
    for t in np.linspace(.18, .82, 21):
        pa = _axis_point(a, axis, limit * t)
        pb = _axis_point(b, axis, limit * t)
        if pa is None or pb is None:
            continue
        direction = pb - pa
        gap = float(np.linalg.norm(direction))
        if gap < 1:
            continue
        direction /= gap
        before = _sample_gray(gray, pb - direction * min(offset, gap * .32))
        after = _sample_gray(gray, pb + direction * offset)
        if before is not None and after is not None:
            differences.append(before - after)
    return float(np.median(differences)) if len(differences) >= 8 else 0.


def _paint_line_evidence(gray: np.ndarray, line: Segment) -> tuple[float, float]:
    """Support for a narrow BRIGHT stripe surrounded by walkable platform.

    Train body reflections also produce long white Hough lines; unlike platform
    paint, one side of a train reflection is frequently the dark track gap.
    Requiring light texture on *both* sides is deliberately conservative.
    Returns (fraction_of_good_samples, median_brightness_contrast).
    """
    tangent = _direction(line)
    normal = np.array([-tangent[1], tangent[0]])
    # Near samples search around either side of a thin painted stripe; far
    # samples must be pavement rather than the stripe itself or a dark track.
    far = max(23., min(gray.shape) * .065)
    contrast = []
    good = []
    for t in np.linspace(.12, .88, 29):
        p = np.asarray(line.a) * (1-t) + np.asarray(line.b) * t
        sides = [_sample_gray(gray, p + normal * delta) for delta in (-far, far)]
        near = [_sample_gray(gray, p + normal * delta) for delta in
                (-.66*far, -.33*far, 0., .33*far, .66*far)]
        if any(v is None for v in sides) or all(v is None for v in near):
            continue
        peak = max(v for v in near if v is not None)
        floor = min(sides)
        contrast.append(peak - float(np.mean(sides)))
        good.append(floor >= 60 and peak >= 170 and peak - floor >= 28)
    if len(good) < 8:
        return 0., 0.
    return float(np.mean(good)), float(np.median(contrast))


def _far_side_darkness(gray: np.ndarray, a: Segment, b: Segment) -> tuple[float, float]:
    """Compare pavement just inside E to the rail/track area outside E.

    Direction is defined from paint anchor A to edge candidate B, making this
    consistent even when the image is rotated or lines are near vertical.
    """
    h, w = gray.shape
    tangent = _direction(a)
    axis = 0 if abs(tangent[0]) >= abs(tangent[1]) else 1
    limit = w - 1 if axis == 0 else h - 1
    inside, outside = [], []
    offset = max(17., min(h, w) * .055)
    for t in np.linspace(.12, .88, 23):
        pa = _axis_point(a, axis, limit * t)
        pb = _axis_point(b, axis, limit * t)
        if pa is None or pb is None:
            continue
        v = pb - pa
        d = float(np.linalg.norm(v))
        if d <= offset * 2:
            continue
        v /= d
        a_val = _sample_gray(gray, pb - v * min(offset, d*.22))
        b_val = _sample_gray(gray, pb + v * offset)
        if a_val is not None and b_val is not None:
            inside.append(a_val)
            outside.append(b_val)
    if len(inside) < 8:
        return 0., 255.
    return float(np.median(inside)), float(np.median(outside))


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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    options: list[tuple[float, HazardZone, dict]] = []
    # The earlier support-only ranking repeatedly picked the two *sides of the
    # same painted stripe*. Demand a real gap and evidence of platform surface
    # on the strip's interior, plus a structural transition at the far edge.
    for a, a_support in anchors:
        paint_fraction, paint_contrast = _paint_line_evidence(gray, a)
        # Strongly reject train reflections and other bright edges that are
        # not actually a painted stripe *on a walkable surface*.
        if method != "canny_hough" and paint_fraction < .42:
            continue
        direction = _direction(a)
        for b, b_support in edge_lines:
            if abs(float(np.dot(direction, _direction(b)))) < math.cos(math.radians(32)):
                continue
            evidence = _zone_evidence(gray, a, b)
            if evidence is None:
                continue
            gap, interior_brightness, interior_light_fraction = evidence
            if not (max(min(h, w) * min_gap_frac, min(h, w) * .12) <= gap <= min(h, w) * max_gap_frac):
                continue
            # A dark platform/train gap is *not* the area of pedestrian risk.
            # These thresholds are heuristic. In low light prefer UNKNOWN to a
            # confidently wrong polygon (manually calibrate that scene).
            if interior_brightness < 58:
                continue
            # Low-contrast grey pavement is valid; track gap is not.
            if interior_light_fraction < .30 and interior_brightness < 95:
                # Uniform synthetic/test pavement can be dimmer, but then a
                # distinct DARK edge stripe is needed to justify a candidate.
                _, darkcore_for_grey = _edge_evidence(gray, b)
                if darkcore_for_grey < 12:
                    continue
            edge_contrast, edge_darkcore = _edge_evidence(gray, b)
            dark_drop = _outside_dark_drop(gray, a, b)
            inside_side, outside_side = _far_side_darkness(gray, a, b)
            real_dark_gap = inside_side >= 75 and outside_side <= 105 and inside_side-outside_side >= 32
            synthetic_dark_edge = edge_darkcore >= 14 and edge_contrast >= 16
            # Bounding a whole *platform region* is not the same as bounding
            # the two bright sides of the painted safety stripe.
            if not (real_dark_gap or synthetic_dark_edge):
                continue
            try:
                zone = HazardZone(a, b)
                zone.polygon()
            except ValueError:
                continue
            # Balance painted-line continuity, far-edge consistency, gap size,
            # and a well-lit surface between the two structures. Scores are
            # NOT probabilities or a calibrated confidence estimate.
            score = (paint_fraction * 150 + min(paint_contrast, 135) * .25
                     + (80 if real_dark_gap else 0)
                     + min(a_support / max(a.length(), 1), 2.5) * 15
                     + min(b_support / max(b.length(), 1), 3.5) * 11
                     + min(edge_contrast, 130) * .70
                     + min(edge_darkcore, 80) * .20
                     + max(-100, min(dark_drop, 125)) * 1.6
                     + max(0, min(inside_side - outside_side, 125)) * 1.2
                     + interior_light_fraction * 52
                     - (gap / min(h, w)) * 20)
            options.append((score, zone, {"gap_px": round(gap, 1),
                                                 "interior_brightness": round(interior_brightness, 1),
                                                 "edge_contrast": round(edge_contrast, 1),
                                                 "edge_darkcore": round(edge_darkcore, 1),
                                                 "outside_dark_drop": round(dark_drop, 1),
                                                 "paint_support_fraction": round(paint_fraction, 3),
                                                 "outside_brightness": round(outside_side, 1)}))
    diagnostics = {"method": method, "paint_segments": len(paint),
                   "edge_segments": len(edges), "paint_lines": len(colored_lines),
                   "edge_lines": len(edge_lines), "candidate_pairs": len(options)}
    if not options:
        return LineResult(None, "uncertain_no_pair", diagnostics)
    options.sort(key=lambda item: item[0], reverse=True)
    score, zone, details = options[0]
    diagnostics.update(details)
    diagnostics["top_pair_score"] = round(score, 2)
    # This is still only a proposal. Surface cues cannot label a safety line
    # and platform edge unambiguously in every station/camera.
    return LineResult(zone, "estimated_review_required", diagnostics)
