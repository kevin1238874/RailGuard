"""Conservative orientation-independent candidate safety-zone estimation.

HSV paint cues and Canny/Hough line segments are suggestions only: their
semantics (paint vs. platform edge) cannot be guaranteed from RGB alone.
Always inspect the suggested polygon before committing a calibration.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np

from .geometry import Segment, HazardZone
from .line_models import fit_line, fit_line_ransac, line_clustering


M2_METHODS = (
    "hsv_canny_hough", "lab_canny_hough",
    "hsv_hough_cluster", "lab_hough_cluster",
    "hsv_hough_ransac", "lab_hough_ransac",
    "hybrid_region_evidence",
)
M1_METHODS = ("raw", "clahe", "gamma")
_LEGACY_METHODS = {
    "color_hough": "hsv_canny_hough",
    "canny_hough": "hsv_hough_cluster",
    "color_ransac": "hsv_hough_ransac",
    "hybrid": "hybrid_region_evidence",
}


@dataclass
class LineResult:
    zone: HazardZone | None
    status: str
    diagnostics: dict
    candidates: list[dict[str, Any]] | None = None


def paint_mask(image: np.ndarray, color: str = "both", clahe: bool = False,
               color_space: str = "hsv") -> np.ndarray:
    if clahe:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(l)
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    color_space = color_space.lower()
    if color_space == "hsv":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(converted, (14, 45, 75), (45, 255, 255))
        value = converted[:, :, 2]
        p90 = float(np.percentile(value, 90))
        adaptive_white = int(np.clip(max(160.0, p90 + .12*(float(value.max())-p90)), 0, 255))
        white = cv2.inRange(converted, (0, 0, adaptive_white), (179, 72, 255))
    elif color_space == "lab":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # LAB provides a luminance cue for white and b-channel separation for
        # yellow. These are candidate masks, not semantic classifications.
        lightness = converted[:, :, 0]
        p90 = float(np.percentile(lightness, 90))
        adaptive_white = int(np.clip(max(160.0, p90 + .12*(float(lightness.max())-p90)), 0, 255))
        white = cv2.inRange(converted, (adaptive_white, 108, 108), (255, 148, 148))
        yellow = cv2.inRange(converted, (75, 115, 142), (255, 160, 235))
    else:
        raise ValueError("color_space must be 'hsv' or 'lab'")
    if color == "yellow":
        mask = yellow
    elif color == "white":
        mask = white
    elif color == "both":
        mask = cv2.bitwise_or(yellow, white)
    else:
        raise ValueError("color must be yellow, white or both")
    # Lọc hình thái khử nhiễu đốm nhỏ nhưng bảo toàn cả line ngang và dọc
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))


def _candidate_segments(image: np.ndarray, mode: str, color: str, clahe: bool,
                        color_space: str = "hsv") -> list[Segment]:
    h, w = image.shape[:2]
    if mode == "color":
        edges = cv2.Canny(paint_mask(image, color, clahe, color_space), 35, 110)
    elif mode == "canny":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 65, 145)
    else:
        raise ValueError("Unknown line extractor")
    raw = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=max(16, min(h, w) // 16),
        minLineLength=max(20, round(min(h, w) * .10)),
        maxLineGap=max(10, round(min(h, w) * .06)),
    )
    if raw is None:
        return []
    segments = []
    min_len = max(20.0, min(h, w) * .09)
    # OpenCV 4 returns (N, 1, 4); OpenCV 5 returns (N, 4).
    for x0, y0, x1, y1 in np.asarray(raw).reshape(-1, 4):
        # Khử các đoạn ngang ở sát đỉnh nóc nhà/bầu trời
        if max(y0, y1) < min(h, w) * 0.22 and abs(y1 - y0) < abs(x1 - x0) * 0.35:
            continue
        seg = Segment((float(x0), float(y0)), (float(x1), float(y1)))
        if seg.length() >= min_len:
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
    """Backward-compatible PCA cluster fit (explicitly not called RANSAC)."""
    return [(line, stats["support_length"])
            for line, stats in _model_segments(
                segments, width, height, "cluster", distance_px=distance_px)]


def _model_segments(segments: list[Segment], width: int, height: int,
                    fit_method: str, *, distance_px: float,
                    random_state: int = 0, max_trials: int = 240,
                    residual_threshold: float | None = None,
                    min_inlier_ratio: float = .45) -> list[tuple[Segment, dict[str, Any]]]:
    """Create clipped line candidates using raw Hough, clustering, or RANSAC."""
    if not segments:
        return []
    if fit_method == "hough":
        sources = [(segment, (segment,), segment.length()) for segment in segments]
    elif fit_method in {"cluster", "ransac"}:
        sources = [(None, cluster.segments, cluster.support_length)
                   for cluster in line_clustering(
                       segments, angle_threshold_deg=10,
                       distance_threshold=distance_px)]
    else:
        raise ValueError(f"unknown fit method: {fit_method}")
    fitted: list[tuple[Segment, dict[str, Any]]] = []
    for seed_index, (raw_line, members, support) in enumerate(sources):
        # Lấy đa điểm mẫu (sample 5 điểm đều trên từng đoạn segment thay vì chỉ 2 đầu mút)
        sample_pts = []
        for m in members:
            pA = np.asarray(m.a, float)
            pB = np.asarray(m.b, float)
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                sample_pts.append(pA * (1.0 - alpha) + pB * alpha)
        points = np.asarray(sample_pts, dtype=float)

        # Trọng số ưu tiên phối cảnh: Các điểm có y càng lớn (càng ở gần camera)
        # có trọng số cao hơn, chống nhiễu loạn ở điểm quá xa chân trời
        y_coords = points[:, 1]
        y_span = max(float(y_coords.max() - y_coords.min()), 1.0)
        y_norm = (y_coords - y_coords.min()) / y_span
        weights = 1.0 + 2.5 * (y_norm ** 1.5)

        try:
            if fit_method == "ransac" and len(points) >= 4:
                result = fit_line_ransac(
                    points, max_trials=max_trials,
                    residual_threshold=(max(1.5, distance_px * .55)
                                        if residual_threshold is None else residual_threshold),
                    min_inlier_ratio=min_inlier_ratio,
                    random_state=random_state + seed_index,
                )
                inliers = points[result.inlier_mask]
                inlier_w = weights[result.inlier_mask]
                model = fit_line(inliers, weights=inlier_w)
                stats = {
                    "fit": "ransac", "support_length": round(float(support), 3),
                    "segment_count": len(members),
                    "point_count": len(points), "inlier_count": int(result.inlier_mask.sum()),
                    "inlier_ratio": round(result.inlier_ratio, 4),
                    "median_residual_px": round(float(np.median(result.residuals[result.inlier_mask])), 4),
                    "trials": result.trials,
                }
            elif fit_method == "hough":
                model = fit_line(points)
                stats = {"fit": "hough_fitline", "support_length": round(float(support), 3),
                         "segment_count": 1, "point_count": 2,
                         "inlier_count": 2, "inlier_ratio": 1.0,
                         "median_residual_px": 0.0}
            else:
                model = fit_line(points, weights=weights)
                stats = {"fit": "cluster_pca", "support_length": round(float(support), 3),
                         "segment_count": len(members), "point_count": len(points),
                         "inlier_count": len(points), "inlier_ratio": 1.0,
                         "median_residual_px": round(float(np.median(model.residuals(points))), 4)}
        except ValueError:
            continue
        line = _extend_to_image(model.point, model.direction, width, height)
        if line is None:
            continue
        midpoint = (np.asarray(line.a) + np.asarray(line.b)) * .5
        duplicate = False
        for existing, _ in fitted:
            normal = np.array([-_direction(existing)[1], _direction(existing)[0]])
            existing_mid = (np.asarray(existing.a) + np.asarray(existing.b)) * .5
            if (abs(float(np.dot(_direction(existing), _direction(line)))) >= .985
                    and abs(float(np.dot(midpoint - existing_mid, normal))) <= distance_px * .65):
                duplicate = True
                break
        if not duplicate:
            fitted.append((line, stats))
        if len(fitted) >= 18:
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
    adaptive_reference = float(np.median(gray))
    return (float(np.median(gaps)), float(np.median(inside)),
            float(np.mean(np.asarray(inside) > adaptive_reference)))


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


def _paint_line_evidence(gray: np.ndarray, line: Segment, mask: np.ndarray | None = None) -> tuple[float, float]:
    """Support for a narrow BRIGHT stripe surrounded by walkable platform.

    Train body reflections also produce long white Hough lines; unlike platform
    paint, one side of a train reflection is frequently the dark track gap.
    Requiring light texture on *both* sides is deliberately conservative.
    Returns (fraction_of_good_samples, median_brightness_contrast).
    """
    tangent = _direction(line)
    normal = np.array([-tangent[1], tangent[0]])
    far = max(20., min(gray.shape) * .06)
    p10, p90 = np.percentile(gray, (10, 90))
    image_median = float(np.median(gray))
    dynamic_range = max(float(p90 - p10), 1.0)
    contrast = []
    good = []
    floors = []
    for t in np.linspace(.12, .88, 29):
        p = np.asarray(line.a) * (1-t) + np.asarray(line.b) * t
        sides = [_sample_gray(gray, p + normal * delta) for delta in (-far, far)]
        near = [_sample_gray(gray, p + normal * delta) for delta in
                (-.66*far, -.33*far, 0., .33*far, .66*far)]
        if any(v is None for v in sides) or all(v is None for v in near):
            continue
        peak = max(v for v in near if v is not None)
        floor = min(sides)
        floors.append(floor)
        contrast.append(peak - float(np.mean(sides)))
        good.append(floor >= 50.0 and peak - floor >= max(5.0, dynamic_range * .08))

    # Real platform paint must have pavement on both sides (floor >= 50).
    # Train reflections border the dark track gap (floor ~ 20-30), which is rejected.
    median_floor = float(np.median(floors)) if floors else 0.0
    if median_floor < 48.0:
        return 0., 0.

    mask_frac = 0.0
    if mask is not None:
        h_m, w_m = mask.shape[:2]
        hits = []
        for t in np.linspace(.10, .90, 31):
            p = np.asarray(line.a) * (1-t) + np.asarray(line.b) * t
            xi, yi = int(round(p[0])), int(round(p[1]))
            if 0 <= xi < w_m and 0 <= yi < h_m:
                patch = mask[max(0, yi-2):min(h_m, yi+3), max(0, xi-2):min(w_m, xi+3)]
                hits.append(bool(np.any(patch > 0)))
        if hits:
            mask_frac = float(np.mean(hits))

    if len(good) < 6:
        if mask_frac >= 0.22:
            return mask_frac, max(float(np.median(contrast)) if contrast else 0.0, dynamic_range * 0.12)
        return 0., 0.

    frac = max(float(np.mean(good)), mask_frac)
    med_c = float(np.median(contrast)) if contrast else 0.0
    return frac, med_c


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


def _two_sided_region_evidence(gray: np.ndarray, safety: Segment,
                               edge: Segment) -> dict[str, Any]:
    """Measure both sides of E using the L->E normal, never image left/right.

    Thresholds are normalized by robust image contrast. The evidence therefore
    describes a relative surface transition instead of requiring one universal
    brightness value across cameras and illumination conditions.
    """
    h, w = gray.shape
    axis = 0 if abs(_direction(safety)[0]) >= abs(_direction(safety)[1]) else 1
    limit = w - 1 if axis == 0 else h - 1
    offset = max(6.0, min(h, w) * .045)
    platform_values: list[float] = []
    outside_values: list[float] = []
    platform_texture: list[float] = []
    outside_texture: list[float] = []
    signed_drops: list[float] = []
    for t in np.linspace(.12, .88, 31):
        pa = _axis_point(safety, axis, limit * t)
        pb = _axis_point(edge, axis, limit * t)
        if pa is None or pb is None:
            continue
        toward_edge = pb - pa
        gap = float(np.linalg.norm(toward_edge))
        if gap <= offset * 1.4:
            continue
        toward_edge /= gap
        pin = pb - toward_edge * min(offset, gap * .25)
        pout = pb + toward_edge * offset
        xi, yi = np.rint(pin).astype(int)
        xo, yo = np.rint(pout).astype(int)
        radius = max(2, round(min(h, w) * .009))
        if (xi-radius < 0 or yi-radius < 0 or xi+radius >= w or yi+radius >= h
                or xo-radius < 0 or yo-radius < 0 or xo+radius >= w or yo+radius >= h):
            continue
        inside_patch = gray[yi-radius:yi+radius+1, xi-radius:xi+radius+1].astype(float)
        outside_patch = gray[yo-radius:yo+radius+1, xo-radius:xo+radius+1].astype(float)
        inside_mean, outside_mean = float(inside_patch.mean()), float(outside_patch.mean())
        platform_values.append(inside_mean)
        outside_values.append(outside_mean)
        platform_texture.append(float(inside_patch.std()))
        outside_texture.append(float(outside_patch.std()))
        signed_drops.append(inside_mean - outside_mean)
    p10, p90 = np.percentile(gray, (10, 90))
    robust_range = max(float(p90 - p10), float(np.std(gray)), 1.0)
    if len(platform_values) < 9:
        return {"reliable": False, "sample_count": len(platform_values),
                "reason": "insufficient_two_sided_samples"}
    drop = float(np.median(signed_drops))
    drop_normalized = drop / robust_range
    consistency = float(np.mean(np.asarray(signed_drops) > robust_range * .12))
    texture_delta = (float(np.median(platform_texture))
                     - float(np.median(outside_texture))) / robust_range
    reliable = drop_normalized >= .28 and consistency >= .58
    return {
        "reliable": reliable,
        "sample_count": len(platform_values),
        "platform_brightness": round(float(np.median(platform_values)), 3),
        "outside_brightness": round(float(np.median(outside_values)), 3),
        "brightness_drop": round(drop, 3),
        "brightness_drop_normalized": round(drop_normalized, 4),
        "drop_consistency": round(consistency, 4),
        "platform_texture": round(float(np.median(platform_texture)), 3),
        "outside_texture": round(float(np.median(outside_texture)), 3),
        "texture_delta_normalized": round(texture_delta, 4),
        "reason": None if reliable else "weak_or_inconsistent_region_transition",
    }


def _painted_stroke_convention(mask: np.ndarray, line: Segment, edge: Segment,
                               convention: str) -> tuple[Segment, dict[str, Any]]:
    """Move the detected *edge* of a bright paint stripe to its centerline.

    Hough often detects either side of the same ~10–25 px painted band;
    matching a person's foot to that side produces systematic boundary errors.
    Sample short runs of segmented paint orthogonal to the candidate line and
    use the median run-center displacement. Keep the original if evidence is
    too fragmented or line clipping would fail.
    """
    h, w = mask.shape
    direction = _direction(line)
    normal = np.array([-direction[1], direction[0]])
    reach = max(18, int(min(h, w) * .075))
    offsets = np.arange(-reach, reach+1, dtype=float)
    if convention not in {"safe_side_edge", "stripe_center"}:
        raise ValueError("stripe_convention must be 'safe_side_edge' or 'stripe_center'")
    low_offsets, high_offsets = [], []
    for t in np.linspace(.10, .90, 35):
        p = np.asarray(line.a)*(1-t)+np.asarray(line.b)*t
        xs = np.rint(p[0] + normal[0]*offsets).astype(int)
        ys = np.rint(p[1] + normal[1]*offsets).astype(int)
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        flags = np.zeros(len(xs), dtype=bool)
        flags[inside] = mask[ys[inside], xs[inside]] > 0
        indices = np.flatnonzero(flags)
        if not len(indices):
            continue
        runs = np.split(indices, np.flatnonzero(np.diff(indices)>1)+1)
        runs = [run for run in runs if 3 <= len(run) <= max(15, reach*1.6)]
        if not runs:
            continue
        closest = min(runs, key=lambda run: abs(float(np.mean(offsets[run]))))
        low_offsets.append(float(offsets[closest[0]]))
        high_offsets.append(float(offsets[closest[-1]]))
    metadata: dict[str, Any] = {
        "stripe_convention": convention,
        "stroke_samples": len(low_offsets),
        "paint_mask_nonzero": int(np.count_nonzero(mask)),
        "paint_mask_shape": list(mask.shape),
    }
    if len(low_offsets) < 12:
        metadata.update({"stripe_refinement": "insufficient_mask_support",
                         "stripe_shift_px": 0.0})
        return line, metadata
    low, high = float(np.median(low_offsets)), float(np.median(high_offsets))
    center = (low + high) * .5
    line_mid = (np.asarray(line.a) + np.asarray(line.b)) * .5 + normal * center
    edge_mid = (np.asarray(edge.a) + np.asarray(edge.b)) * .5
    edge_sign = float(np.dot(edge_mid - line_mid, normal))
    delta = low if (convention != "stripe_center" and edge_sign >= 0) else (high if convention != "stripe_center" else center)

    # Dịch chuyển vạch sơn theo pháp tuyến mà bảo toàn các điểm đầu mút đã cắt phối cảnh
    normal_shift = normal * delta
    p_a = (float(np.clip(line.a[0] + normal_shift[0], 0, w - 1)),
           float(np.clip(line.a[1] + normal_shift[1], 0, h - 1)))
    p_b = (float(np.clip(line.b[0] + normal_shift[0], 0, w - 1)),
           float(np.clip(line.b[1] + normal_shift[1], 0, h - 1)))
    shifted = Segment(p_a, p_b)

    metadata.update({
        "stripe_refinement": "applied",
        "stripe_shift_px": round(delta, 3),
        "stripe_width_px": round(high - low + 1.0, 3),
        "stripe_center_shift_px": round(center, 3),
    })
    return (shifted or line), metadata


def _center_painted_stroke(mask: np.ndarray, line: Segment) -> tuple[Segment, float]:
    """Compatibility helper retained for earlier callers and tests."""
    centered, metadata = _painted_stroke_convention(mask, line, line, "stripe_center")
    return centered, float(metadata.get("stripe_shift_px", 0.0))


def _preprocess_m1(image: np.ndarray, method: str) -> np.ndarray:
    from .preprocessing import PREPROCESSORS

    if method == "raw":
        return image.copy()
    if method == "clahe":
        return PREPROCESSORS["clahe"](image)
    if method == "gamma":
        return PREPROCESSORS["gamma"](image, gamma=1.4)
    raise ValueError(f"Unknown M1 preprocessing method: {method}")


def _segment_payload(line: Segment, width: int, height: int) -> dict[str, Any]:
    pixels = [list(map(float, line.a)), list(map(float, line.b))]
    normalized = [[point[0] / (width - 1), point[1] / (height - 1)] for point in pixels]
    return {"pixels": pixels, "normalized": normalized}


def estimate_zone(image: np.ndarray, method: str = "color_hough", color: str = "both",
                  clahe: bool = False, min_gap_frac: float = .06,
                  max_gap_frac: float = .78, *, preprocess: str = "raw",
                  stripe_convention: str = "safe_side_edge",
                  random_state: int = 0, ransac_max_trials: int = 240,
                  ransac_residual_threshold: float | None = None,
                  ransac_min_inlier_ratio: float = .45) -> LineResult:
    """Rank review-only L/E proposals across the Phase-P2 method matrix.

    Pixel geometry is returned in ``zone`` for compatibility. Every ranked
    candidate also records pixel and normalized endpoints explicitly. If no E
    has adequate relative, two-sided evidence, the function abstains.
    """
    requested_method = method
    method = _LEGACY_METHODS.get(method, method)
    if method not in M2_METHODS:
        raise ValueError(f"Unknown method: {requested_method}")
    if preprocess not in M1_METHODS:
        raise ValueError(f"Unknown M1 preprocessing method: {preprocess}")
    if stripe_convention not in {"safe_side_edge", "stripe_center"}:
        raise ValueError("stripe_convention must be 'safe_side_edge' or 'stripe_center'")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR color image")
    h, w = image.shape[:2]
    base_diagnostics: dict[str, Any] = {
        "method": method, "requested_method": requested_method,
        "preprocess": preprocess, "stripe_convention": stripe_convention,
        "validation_status": "IMPLEMENTED_NOT_VALIDATED_ON_REAL_DATA",
        "coordinate_spaces": ["pixels", "normalized"],
    }
    if w < 32 or h < 32:
        return LineResult(None, "uncertain_image_too_small", base_diagnostics, [])
    work = _preprocess_m1(image, "clahe" if clahe and preprocess == "raw" else preprocess)
    color_space = "lab" if method.startswith("lab_") else "hsv"
    distance_px = max(4.0, min(h, w) * .022)
    if method == "hybrid_region_evidence":
        paint_hsv = _candidate_segments(work, "color", color, False, "hsv")
        paint_lab = _candidate_segments(work, "color", color, False, "lab")
        paint = paint_hsv + paint_lab
        fit_method = "ransac"
        color_space = "hybrid_hsv_lab"
        colored_lines = (_model_segments(
            paint, w, h, fit_method, distance_px=distance_px,
            random_state=random_state, max_trials=ransac_max_trials,
            residual_threshold=ransac_residual_threshold,
            min_inlier_ratio=ransac_min_inlier_ratio)
            + _model_segments(paint, w, h, "hough", distance_px=distance_px))
        edges = _candidate_segments(work, "canny", color, False, "hsv")
        edge_lines = (_model_segments(
            edges, w, h, fit_method, distance_px=distance_px,
            random_state=random_state + 10000, max_trials=ransac_max_trials,
            residual_threshold=ransac_residual_threshold,
            min_inlier_ratio=ransac_min_inlier_ratio)
            + _model_segments(edges, w, h, "hough", distance_px=distance_px))
    else:
        paint = _candidate_segments(work, "color", color, False, color_space)
        fit_method = ("ransac" if method.endswith("_ransac")
                      else "cluster" if method.endswith("_cluster") else "hough")
        edges = _candidate_segments(work, "canny", color, False, "hsv")
        colored_lines = _model_segments(
            paint, w, h, fit_method, distance_px=distance_px,
            random_state=random_state, max_trials=ransac_max_trials,
            residual_threshold=ransac_residual_threshold,
            min_inlier_ratio=ransac_min_inlier_ratio)
        edge_lines = _model_segments(
            edges, w, h, fit_method, distance_px=distance_px,
            random_state=random_state + 10000, max_trials=ransac_max_trials,
            residual_threshold=ransac_residual_threshold,
            min_inlier_ratio=ransac_min_inlier_ratio)
    base_diagnostics.update({
        "color_space": color_space, "line_fit": fit_method,
        "paint_segments": len(paint), "edge_segments": len(edges),
        "paint_lines": len(colored_lines), "edge_lines": len(edge_lines),
    })
    if not colored_lines:
        return LineResult(None, "uncertain_no_paint_line", base_diagnostics, [])
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    mask_space = "hsv" if color_space.startswith("hybrid") else color_space
    mask = paint_mask(work, color, False, mask_space)
    robust_range = max(float(np.percentile(gray, 90) - np.percentile(gray, 10)),
                       float(np.std(gray)), 1.0)
    raw_options: list[tuple[float, HazardZone, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for safety, safety_stats in colored_lines:
        paint_fraction, paint_contrast = _paint_line_evidence(gray, safety, mask)
        if paint_fraction < .20:
            continue
        _, stroke_probe = _painted_stroke_convention(
            mask, safety, safety, "stripe_center")
        stroke_width = float(stroke_probe.get("stripe_width_px", 0.0))
        direction = _direction(safety)
        s_dx = abs(safety.b[0] - safety.a[0])
        s_dy = abs(safety.b[1] - safety.a[1])
        if max(safety.a[1], safety.b[1]) < h * 0.35:
            continue
        for edge, edge_stats in edge_lines:
            e_dx = abs(edge.b[0] - edge.a[0])
            e_dy = abs(edge.b[1] - edge.a[1])
            if max(edge.a[1], edge.b[1]) < h * 0.35:
                continue
            # Tính nhất quán về độ dốc phối cảnh: Không ghép một vạch chạy dọc sân ga với một vạch ngang cắt thân tàu
            if (s_dy > s_dx * 1.3 and e_dx > e_dy * 1.3) or (e_dy > e_dx * 1.3 and s_dx > s_dy * 1.3):
                continue
            angle_cosine = abs(float(np.dot(direction, _direction(edge))))
            if angle_cosine < math.cos(math.radians(35)):
                continue
            zone_evidence = _zone_evidence(gray, safety, edge)
            if zone_evidence is None:
                continue
            gap, interior_brightness, interior_light_fraction = zone_evidence
            # Vùng nguy hiểm trên sân ga không thể có độ sáng lòng thấp như hố ray (interior_brightness >= 40)
            if interior_brightness < 40.0:
                continue
            if not (min(h, w) * max(min_gap_frac, .04) <= gap
                    <= min(h, w) * max_gap_frac):
                continue
            # If E is only the opposite boundary of L's segmented paint run,
            # the pair describes stripe thickness, not a platform hazard zone.
            if stroke_width > 0 and gap <= stroke_width * 2.0:
                continue
            edge_contrast, edge_darkcore = _edge_evidence(gray, edge)
            region = _two_sided_region_evidence(gray, safety, edge)
            narrow_dark_structure = (edge_darkcore / robust_range >= .12) or (edge_contrast / robust_range >= .16)
            is_reliable = bool(region.get("reliable", False))
            drop_norm = float(region.get("brightness_drop_normalized", 0.0))
            if not is_reliable and not narrow_dark_structure:
                continue
            # Nếu không phải cấu trúc tối hẹp và độ sụt sáng quá thấp, loại bỏ
            if not is_reliable and not narrow_dark_structure and drop_norm < 0.05:
                continue
            if drop_norm < -0.15:
                continue

            def _at_y(seg, y_val):
                dy = seg.b[1] - seg.a[1]
                dx = seg.b[0] - seg.a[0]
                return (float(seg.a[0] + dx * (y_val - seg.a[1]) / (dy + 1e-6)), float(y_val))

            # Tính điểm tụ phối cảnh (vanishing point) giữa safety line và platform edge
            p1, p2 = np.asarray(safety.a, float), np.asarray(safety.b, float)
            q1, q2 = np.asarray(edge.a, float), np.asarray(edge.b, float)
            A = np.column_stack([p2 - p1, -(q2 - q1)])
            b_vec = q1 - p1
            vp_inter = None
            vp_bonus = 0.0
            if abs(float(np.linalg.det(A))) > 1e-5:
                params = np.linalg.solve(A, b_vec)
                vp_inter = p1 + float(params[0]) * (p2 - p1)
                # Thưởng điểm lớn nếu 2 vạch hội tụ vào dải chân trời hợp lý
                if 0 <= vp_inter[1] < h * 0.70 and -w * 0.5 <= vp_inter[0] <= w * 1.5:
                    vp_bonus = 100.0
                elif vp_inter[1] >= h: # Phân kỳ về phía chân trời
                    vp_bonus = -80.0

            # Kiểm tra độ rộng dải ở đáy so với đỉnh để loại bỏ vạch phân kỳ
            is_steep = (abs(p2[1] - p1[1]) >= abs(p2[0] - p1[0]) * 0.7)
            g_bot = abs(_at_y(safety, h - 1)[0] - _at_y(edge, h - 1)[0])
            g_top = abs(_at_y(safety, 0)[0] - _at_y(edge, 0)[0])
            if is_steep and g_bot < g_top * 0.65:
                continue

            # Căn chỉnh phối cảnh và cắt ngắn trước điểm giao nhau
            safe_seg = safety
            edge_seg = edge
            if vp_inter is not None and 0 <= vp_inter[1] < h * 0.70:
                y_cut = min(h - 1, max(0, int(vp_inter[1] + 25)))
                y_base = float(h - 1)

                def _clip_base(seg, y_b, y_c):
                    pt = _at_y(seg, y_b)
                    if pt[0] < 0:
                        y_at_0 = seg.a[1] + (0 - seg.a[0]) * (seg.b[1] - seg.a[1]) / (seg.b[0] - seg.a[0] + 1e-6)
                        return (0.0, float(np.clip(y_at_0, min(y_b, y_c), max(y_b, y_c))))
                    elif pt[0] > w - 1:
                        y_at_w = seg.a[1] + (w - 1 - seg.a[0]) * (seg.b[1] - seg.a[1]) / (seg.b[0] - seg.a[0] + 1e-6)
                        return (float(w - 1), float(np.clip(y_at_w, min(y_b, y_c), max(y_b, y_c))))
                    return pt

                s_a = _clip_base(safety, y_base, float(y_cut))
                e_a = _clip_base(edge, y_base, float(y_cut))

                if abs(y_base - vp_inter[1]) > 1e-4:
                    ratio = (y_cut - vp_inter[1]) / (y_base - vp_inter[1])
                    s_bx = vp_inter[0] + ratio * (s_a[0] - vp_inter[0])
                    e_bx = vp_inter[0] + ratio * (e_a[0] - vp_inter[0])
                    s_b = (float(np.clip(s_bx, 0, w - 1)), float(y_cut))
                    e_b = (float(np.clip(e_bx, 0, w - 1)), float(y_cut))
                else:
                    s_b = _at_y(safety, float(y_cut))
                    e_b = _at_y(edge, float(y_cut))

                try:
                    z_cand = HazardZone(Segment(s_a, s_b), Segment(e_a, e_b))
                    z_cand.polygon()
                    safe_seg, edge_seg = z_cand.safety_line, z_cand.platform_edge
                    zone = z_cand
                except ValueError:
                    try:
                        zone = HazardZone(safe_seg, edge_seg)
                        zone.polygon()
                    except ValueError:
                        continue
            else:
                try:
                    zone = HazardZone(safe_seg, edge_seg)
                    zone.polygon()
                except ValueError:
                    continue
            region_score = (max(0.0, drop_norm) * 160.0
                            + float(region.get("drop_consistency", 0)) * 90.0)
            rel_bonus = 140.0 if is_reliable else -100.0
            # Ưu tiên các vạch chạy dọc/xiên theo chiều sâu sân ga (dy > dx)
            vert_bonus = 50.0 if s_dy >= s_dx * 1.5 else (-40.0 if s_dx >= s_dy * 0.9 else 0.0)
            supp_bonus = min(float(safety_stats["support_length"]) / max(min(h, w), 1), 3.0) * 20.0
            score = (
                paint_fraction * 70.0
                + supp_bonus
                + vert_bonus
                + min(paint_contrast / robust_range, 2.0) * 20.0
                + min(float(edge_stats["support_length"]) / max(edge.length(), 1), 3.0) * 12.0
                + region_score
                + rel_bonus
                + vp_bonus
                + min(edge_darkcore / robust_range, 2.0) * 15.0
                - (gap / min(h, w)) * 10.0
            )
            evidence = {
                "gap_px": round(gap, 3),
                "gap_normalized": round(gap / min(h, w), 5),
                "interior_brightness_descriptive": round(interior_brightness, 3),
                "interior_light_fraction_descriptive": round(interior_light_fraction, 4),
                "edge_contrast": round(edge_contrast, 3),
                "edge_darkcore": round(edge_darkcore, 3),
                "paint_support_fraction": round(paint_fraction, 4),
                "paint_contrast": round(paint_contrast, 3),
                "two_sided_region": region,
                "edge_evidence_type": ("region_transition" if region.get("reliable", False)
                                       else "narrow_dark_structure_penalized"),
            }
            raw_options.append((score, zone, evidence, safety_stats, edge_stats))
    raw_options.sort(key=lambda item: item[0], reverse=True)
    ranked: list[dict[str, Any]] = []
    refined_options: list[tuple[float, HazardZone, dict[str, Any]]] = []
    for score, raw_zone, evidence, safety_stats, edge_stats in raw_options[:12]:
        refined_safety, stripe_metadata = _painted_stroke_convention(
            mask, raw_zone.safety_line, raw_zone.platform_edge, stripe_convention)
        try:
            zone = HazardZone(refined_safety, raw_zone.platform_edge)
            zone.polygon()
        except ValueError:
            continue
        record = {
            "rank": len(ranked) + 1,
            "safety_line": _segment_payload(zone.safety_line, w, h),
            "platform_edge": _segment_payload(zone.platform_edge, w, h),
            "coordinate_space": "pixels_and_normalized",
            "support": {"safety": safety_stats, "platform_edge": edge_stats},
            "evidence": evidence,
            "diagnostics": stripe_metadata,
            "score": round(float(score), 4),
            "uncertainty_reason": "heuristic_rgb_semantics_require_review",
            "estimated_review_required": True,
        }
        ranked.append(record)
        refined_options.append((score, zone, record))
        if len(ranked) >= 3:
            break
    diagnostics = {**base_diagnostics, "candidate_pairs": len(raw_options),
                   "top_candidates": ranked}
    if not refined_options:
        diagnostics["uncertainty_reason"] = "no_reliable_two_sided_platform_edge_evidence"
        return LineResult(None, "uncertain_no_platform_edge", diagnostics, [])
    score, zone, record = refined_options[0]
    diagnostics.update(record["evidence"])
    diagnostics.update({
        key: record["evidence"]["two_sided_region"].get(key)
        for key in ("platform_brightness", "outside_brightness",
                    "brightness_drop_normalized", "drop_consistency")
        if key in record["evidence"]["two_sided_region"]
    })
    diagnostics.update(record["diagnostics"])
    diagnostics["top_pair_score"] = round(float(score), 4)
    # Surface semantics remain heuristic: this is explicitly not calibration
    # ground truth and no probability-like confidence is exposed.
    return LineResult(zone, "estimated_review_required", diagnostics, ranked)
