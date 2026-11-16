"""Conservative orientation-independent candidate safety-zone estimation.

HSV paint cues and Canny/Hough line segments are suggestions only: their
semantics (paint vs. platform edge) cannot be guaranteed from RGB alone.
Always inspect the suggested polygon before committing a calibration.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Mapping

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


def _paint_masks(image: np.ndarray, *, clahe: bool = False,
                 color_space: str = "hsv") -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint yellow and bright-white platform-paint masks."""
    if clahe:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(l)
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    color_space = color_space.lower()
    if color_space == "hsv":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Bound yellow hue <= 30 so lime-green train bodies (H >= 31) do not trigger
        yellow = cv2.inRange(converted, (14, 45, 75), (30, 255, 255))
        value = converted[:, :, 2]
        sat = converted[:, :, 1]
        if float(np.mean(yellow > 0)) > 0.23:
            # Suppress wide warm-stone/granite background floods while keeping vivid or bright stripes
            v_thr = int(max(165.0, float(np.percentile(value[yellow > 0], 75))))
            keep = ((sat >= 135) & (value >= 135)) | ((value >= v_thr) & (sat >= 50))
            yellow = np.where((yellow > 0) & keep, np.uint8(255), np.uint8(0))
        p90 = float(np.percentile(value, 90))
        adaptive_white = int(np.clip(max(185.0, min(238.0, p90 + .20*(float(value.max())-p90))), 0, 255))
        white = cv2.inRange(converted, (0, 0, adaptive_white), (179, 65, 255))
    elif color_space == "lab":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # LAB provides a luminance cue for white and b-channel separation for
        # yellow. These are candidate masks, not semantic classifications.
        lightness = converted[:, :, 0]
        b_ch = converted[:, :, 2]
        p90 = float(np.percentile(lightness, 90))
        adaptive_white = int(np.clip(max(185.0, min(238.0, p90 + .20*(float(lightness.max())-p90))), 0, 255))
        white = cv2.inRange(converted, (adaptive_white, 110, 110), (255, 146, 146))
        yellow = cv2.inRange(converted, (75, 115, 140), (255, 155, 235))
        if float(np.mean(yellow > 0)) > 0.23:
            # Suppress wide warm-beige coping stone floods while keeping saturated yellow or bright stripes
            l_thr = float(np.percentile(lightness[yellow > 0], 75))
            keep = (b_ch >= 152) | ((lightness >= max(145.0, l_thr)) & (b_ch >= 143))
            yellow = np.where((yellow > 0) & keep, np.uint8(255), np.uint8(0))
    else:
        raise ValueError("color_space must be 'hsv' or 'lab'")
    # Yellow owns warm pixels so one stripe cannot supply both semantic roles.
    white = cv2.bitwise_and(white, cv2.bitwise_not(yellow))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned: list[np.ndarray] = []
    for raw in (yellow, white):
        value = cv2.morphologyEx(raw, cv2.MORPH_OPEN, open_kernel)
        cleaned.append(cv2.morphologyEx(value, cv2.MORPH_CLOSE, close_kernel))
    return cleaned[0], cleaned[1]


def paint_mask(image: np.ndarray, color: str = "both", clahe: bool = False,
               color_space: str = "hsv") -> np.ndarray:
    """Backward-compatible selected/combined paint mask."""
    yellow, white = _paint_masks(image, clahe=clahe, color_space=color_space)
    if color == "yellow":
        return yellow
    if color == "white":
        return white
    if color == "both":
        return cv2.bitwise_or(yellow, white)
    raise ValueError("color must be yellow, white or both")


def _dark_inset_mask(image: np.ndarray) -> np.ndarray:
    """Isolate narrow low-chroma dark inset strips flanked on both sides by brighter floor."""
    h, w = image.shape[:2]
    d = min(h, w)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0].astype(np.float32)
    chroma = np.hypot(lab[:, :, 1].astype(np.float32) - 128.0, lab[:, :, 2].astype(np.float32) - 128.0)
    off = max(12, int(round(0.045 * d)))
    rad = max(4, int(round(0.018 * d)))
    k = 2 * rad + 1
    mean_x = cv2.blur(l_ch, (k, 5))
    center = cv2.blur(l_ch, (5, 5))
    left_m = np.zeros_like(l_ch)
    right_m = np.zeros_like(l_ch)
    left_m[:, off:] = mean_x[:, :-off]
    right_m[:, :-off] = mean_x[:, off:]
    valley = np.minimum(left_m, right_m) - center
    return ((valley >= 8.0) & (chroma <= 34.0)).astype(np.uint8) * 255


def _candidate_segments(image: np.ndarray, mode: str, color: str, clahe: bool,
                        color_space: str = "hsv",
                        custom_mask: np.ndarray | None = None,
                        exclude_mask: np.ndarray | None = None) -> list[Segment]:
    h, w = image.shape[:2]
    if mode == "color":
        edges = cv2.Canny(paint_mask(image, color, clahe, color_space), 35, 110)
    elif mode == "mask" and custom_mask is not None:
        edges = cv2.Canny(custom_mask, 35, 110)
    elif mode == "canny":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 65, 145)
    else:
        raise ValueError("Unknown line extractor")
    if exclude_mask is not None and np.any(exclude_mask):
        # Erode slightly deeper before pre-Canny suppression so YOLO boundary jitter never erases true platform-edge pixels.
        margin_k = max(5, int(round(0.014 * min(h, w))) | 1)
        margin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_k, margin_k))
        interior_exclude = cv2.erode(np.asarray(exclude_mask, dtype=np.uint8), margin_kernel) > 0
        edges[interior_exclude] = 0
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
                    min_inlier_ratio: float = .45,
                    max_lines: int | None = None) -> list[tuple[Segment, dict[str, Any]]]:
    """Create clipped line candidates using sorted Hough, tight clustering, or multi-pass RANSAC."""
    if not segments:
        return []
    if fit_method == "hough":
        ordered = sorted(segments, key=lambda segment: segment.length(), reverse=True)
        steep_lower = [
            s for s in ordered
            if abs(s.b[1] - s.a[1]) >= abs(s.b[0] - s.a[0]) * 0.85
            and max(s.a[1], s.b[1]) >= height * 0.45
        ]
        steep_ids = {id(s) for s in steep_lower}
        non_steep = [s for s in ordered if id(s) not in steep_ids]
        sources = (
            [(segment, (segment,), segment.length(), True) for segment in steep_lower]
            + [(segment, (segment,), segment.length(), False) for segment in non_steep]
        )
    elif fit_method in {"cluster", "ransac"}:
        tight_dist = max(4.0, distance_px * 0.30)
        clusters = line_clustering(
            segments, angle_threshold_deg=4.5, distance_threshold=tight_dist)
        sources = [(None, cluster.segments, cluster.support_length, False) for cluster in clusters]
    else:
        raise ValueError(f"unknown fit method: {fit_method}")

    def _sample_points_weights(members: tuple[Segment, ...]) -> tuple[np.ndarray, np.ndarray]:
        sample_pts = []
        for m in members:
            pA = np.asarray(m.a, float)
            pB = np.asarray(m.b, float)
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                sample_pts.append(pA * (1.0 - alpha) + pB * alpha)
        pts = np.asarray(sample_pts, dtype=float)
        y_coords = pts[:, 1]
        y_span = max(float(y_coords.max() - y_coords.min()), 1.0)
        y_norm = (y_coords - y_coords.min()) / y_span
        wts = 1.0 + 2.5 * (y_norm ** 1.5)
        return pts, wts

    fitted: list[tuple[Segment, dict[str, Any]]] = []
    if max_lines is None:
        max_lines = 44 if fit_method == "hough" else 36
    steep_cap = max(20, max_lines - 4)

    def _append_fitted(line: Segment, stats: dict[str, Any]) -> bool:
        pa_n, pb_n = np.asarray(line.a), np.asarray(line.b)
        d_new = _direction(line)
        dup_tol = max(4.0, distance_px * .38)
        for existing, _ in fitted:
            d_ex = _direction(existing)
            if abs(float(np.dot(d_ex, d_new))) < .9975:
                continue
            normal = np.array([-d_ex[1], d_ex[0]])
            existing_mid = (np.asarray(existing.a) + np.asarray(existing.b)) * .5
            if max(abs(float(np.dot(pa_n - existing_mid, normal))),
                   abs(float(np.dot(pb_n - existing_mid, normal)))) <= dup_tol:
                return False
        fitted.append((line, stats))
        return True

    def _maybe_append_top_member(members_seq: tuple[Segment, ...], base_stats: dict[str, Any]) -> None:
        if len(members_seq) < 3 or len(fitted) >= max_lines:
            return
        top_seg = max(members_seq, key=lambda m: m.length())
        if top_seg.length() < min(width, height) * 0.14:
            return
        tp, tw = _sample_points_weights((top_seg,))
        tm = fit_line(tp, weights=tw)
        tl = _extend_to_image(tm.point, tm.direction, width, height)
        if tl is not None:
            _append_fitted(tl, {**base_stats, "segment_count": 1,
                                "support_length": round(float(top_seg.length()), 3)})

    deferred_fg: list[tuple[Segment, dict[str, Any]]] = []

    def _maybe_append_foreground_refit(members_seq: tuple[Segment, ...],
                                       base_line: Segment | None,
                                       base_stats: dict[str, Any]) -> None:
        """Fit a secondary line anchored strictly on Mid [0.45h,0.70h) and Near [0.70h,0.96h] points."""
        if base_line is None or len(members_seq) < 2 or len(deferred_fg) >= 8:
            return
        fg_list: list[np.ndarray] = []
        y_mid_lo, y_mid_hi = 0.45 * height, 0.70 * height
        y_near_hi = 0.96 * height
        for seg in members_seq:
            n_pts = max(3, min(13, int(round(seg.length() / 14.0)) + 1))
            pa = np.asarray(seg.a, float)
            pb = np.asarray(seg.b, float)
            for alpha in np.linspace(0.0, 1.0, n_pts):
                fg_list.append(pa * (1.0 - alpha) + pb * alpha)
        if not fg_list:
            return
        pts = np.asarray(fg_list, dtype=float)
        ys = pts[:, 1]
        in_fg = (ys >= y_mid_lo) & (ys <= y_near_hi)
        if int(np.count_nonzero(in_fg)) < 8:
            return
        fg_pts = pts[in_fg]
        fg_ys = fg_pts[:, 1]
        if int(np.count_nonzero(fg_ys < y_mid_hi)) < 3 or int(np.count_nonzero(fg_ys >= y_mid_hi)) < 3:
            return
        if float(fg_ys.max() - fg_ys.min()) < 0.18 * height:
            return
        fg_norm = np.clip((fg_ys - y_mid_lo) / max(y_near_hi - y_mid_lo, 1.0), 0.0, 1.0)
        fg_w = 1.0 + 3.0 * (fg_norm ** 1.5)
        try:
            fg_model = fit_line(fg_pts, weights=fg_w)
            res_fg = fg_model.residuals(fg_pts)
            thresh_fg = max(2.5, distance_px * 0.50) if residual_threshold is None else float(residual_threshold)
            inliers = res_fg <= thresh_fg
            if int(np.count_nonzero(inliers)) >= 8:
                in_pts = fg_pts[inliers]
                in_ys = in_pts[:, 1]
                if int(np.count_nonzero(in_ys < y_mid_hi)) >= 2 and int(np.count_nonzero(in_ys >= y_mid_hi)) >= 2:
                    fg_model = fit_line(in_pts, weights=fg_w[inliers])
        except ValueError:
            return
        fg_line = _extend_to_image(fg_model.point, fg_model.direction, width, height)
        if fg_line is None:
            return
        if abs(float(np.dot(_direction(base_line), _direction(fg_line)))) < math.cos(math.radians(6.5)):
            return
        deferred_fg.append((fg_line, {**base_stats, "foreground_refit": True}))

    steep_fitted = 0
    for seed_index, (_, members, support, is_steep_bucket) in enumerate(sources):
        if len(fitted) >= max_lines:
            break
        if fit_method == "hough" and is_steep_bucket and steep_fitted >= steep_cap:
            continue
        points, weights = _sample_points_weights(members)
        try:
            if fit_method == "ransac" and len(points) >= 4:
                thresh = (max(1.5, distance_px * .50)
                          if residual_threshold is None else residual_threshold)
                try:
                    result = fit_line_ransac(
                        points, max_trials=max_trials,
                        residual_threshold=thresh,
                        min_inlier_ratio=min_inlier_ratio,
                        random_state=random_state + seed_index,
                    )
                except ValueError:
                    # Fallback gracefully on multi-modal clusters instead of discarding the line.
                    result = fit_line_ransac(
                        points, max_trials=max_trials,
                        residual_threshold=max(thresh * 1.35, 2.5),
                        min_inlier_ratio=max(0.25, min_inlier_ratio * 0.55),
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
                line = _extend_to_image(model.point, model.direction, width, height)
                if line is not None:
                    _append_fitted(line, stats)
                # Sequential multi-pass RANSAC peel if a dense cluster holds adjacent parallel edges
                if len(members) >= 8 and len(fitted) < max_lines:
                    rem_pts = points
                    rem_w = weights
                    cur_res = result
                    for peel in range(2):
                        outlier_mask = cur_res.residuals > max(thresh * 1.3, 2.5)
                        if int(outlier_mask.sum()) < max(12, int(0.22 * len(points))):
                            break
                        rem_pts = rem_pts[outlier_mask]
                        rem_w = rem_w[outlier_mask]
                        try:
                            cur_res = fit_line_ransac(
                                rem_pts, max_trials=max_trials,
                                residual_threshold=thresh,
                                min_inlier_ratio=max(0.28, min_inlier_ratio * 0.65),
                                random_state=random_state + 500 * (peel + 1) + seed_index,
                            )
                        except ValueError:
                            break
                        m_p = fit_line(rem_pts[cur_res.inlier_mask], weights=rem_w[cur_res.inlier_mask])
                        l_p = _extend_to_image(m_p.point, m_p.direction, width, height)
                        if l_p is not None and len(fitted) < max_lines:
                            s_p = {
                                "fit": "ransac", "support_length": round(float(support * cur_res.inlier_ratio * 0.6), 3),
                                "segment_count": len(members),
                                "point_count": len(rem_pts), "inlier_count": int(cur_res.inlier_mask.sum()),
                                "inlier_ratio": round(cur_res.inlier_ratio, 4),
                                "median_residual_px": round(float(np.median(cur_res.residuals[cur_res.inlier_mask])), 4),
                                "trials": cur_res.trials,
                            }
                            _append_fitted(l_p, s_p)
                _maybe_append_top_member(members, stats)
                _maybe_append_foreground_refit(members, line, stats)
                continue
            elif fit_method == "hough":
                model = fit_line(points)
                stats = {"fit": "hough_fitline", "support_length": round(float(support), 3),
                         "segment_count": 1, "point_count": len(points),
                         "inlier_count": len(points), "inlier_ratio": 1.0,
                         "median_residual_px": 0.0}
            else:
                model = fit_line(points, weights=weights)
                med_res = float(np.median(model.residuals(points)))
                stats = {"fit": "cluster_pca", "support_length": round(float(support), 3),
                         "segment_count": len(members), "point_count": len(points),
                         "inlier_count": len(points), "inlier_ratio": 1.0,
                         "median_residual_px": round(med_res, 4)}
                # Preserve long individual member edges when tight clustering still spans adjacent lines.
                if len(members) >= 8 and med_res > max(3.5, distance_px * 0.30):
                    _maybe_append_top_member(members, stats)
        except ValueError:
            continue
        line = _extend_to_image(model.point, model.direction, width, height)
        if line is not None and _append_fitted(line, stats):
            if is_steep_bucket:
                steep_fitted += 1
        _maybe_append_foreground_refit(members, line, stats)
    for fg_line, fg_stats in deferred_fg:
        if len(fitted) >= max_lines + 4:
            break
        _append_fitted(fg_line, fg_stats)
    return fitted


_T_VALS_15 = np.linspace(.15, .85, 15)
_T_VALS_21 = np.linspace(.18, .82, 21)
_T_VALS_29 = np.linspace(.12, .88, 29)
_T_VALS_31 = np.linspace(.10, .90, 31)
_T_VALS_23 = np.linspace(0.06, 0.94, 23)
_T_VALS_15_TRENCH = np.linspace(0.10, 0.90, 15)
_ZONE_US = np.array((0.25, 0.50, 0.75))
_SAFE_BG_US = np.array((0.04, 0.08, 0.14))
_TRENCH_US = np.array((0.18, 0.34, 0.50, 0.66, 0.80, 0.89, 0.95, 1.00))
_DILATE_5X5 = np.ones((5, 5), np.uint8)
_COS_35_DEG = math.cos(math.radians(35))


def _axis_points_vec(line: Segment, axis: int, values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    ax, ay = line.a
    bx, by = line.b
    da = (bx - ax) if axis == 0 else (by - ay)
    if -1e-6 < da < 1e-6:
        return None
    t = (values - (ax if axis == 0 else ay)) / da
    return ax + t * (bx - ax), ay + t * (by - ay)


def _station_frames(safety: Segment, edge: Segment, axis: int, t_vals: np.ndarray,
                    min_gap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute vectorized paired stations along safety and edge where gap >= min_gap."""
    sax, say = safety.a
    sbx, sby = safety.b
    pLx = sax + t_vals * (sbx - sax)
    pLy = say + t_vals * (sby - say)
    pts_e = _axis_points_vec(edge, axis, pLx if axis == 0 else pLy)
    if pts_e is None:
        return None
    pEx, pEy = pts_e
    dx = pEx - pLx
    dy = pEy - pLy
    gap = np.hypot(dx, dy)
    keep = gap >= min_gap
    if not np.any(keep):
        return None
    gap_k = gap[keep]
    dx_k = dx[keep]
    dy_k = dy[keep]
    return pLx[keep], pLy[keep], pEx[keep], pEy[keep], dx_k, dy_k, gap_k, dx_k / gap_k, dy_k / gap_k


def _fast_median(vals: list[float]) -> float:
    n = len(vals)
    s = sorted(vals)
    mid = n >> 1
    return s[mid] if (n & 1) else 0.5 * (s[mid - 1] + s[mid])


def _box_mean5(channel: np.ndarray) -> np.ndarray:
    """Precompute 5x5 local mean for O(1) point sampling inside interior."""
    return cv2.boxFilter(channel.astype(np.float32), -1, (5, 5), borderType=cv2.BORDER_REPLICATE)


def _axis_point_xy(line: Segment, axis: int, value: float) -> tuple[float, float] | None:
    ax, ay = line.a
    bx, by = line.b
    da = (bx - ax) if axis == 0 else (by - ay)
    if -1e-6 < da < 1e-6:
        return None
    t = (value - (ax if axis == 0 else ay)) / da
    return (ax + t * (bx - ax), ay + t * (by - ay))


def _axis_point(line: Segment, axis: int, value: float) -> np.ndarray | None:
    """Point on an infinite fitted line at a given x or y (perspective-safe)."""
    pt = _axis_point_xy(line, axis, value)
    return None if pt is None else np.array(pt, dtype=float)


def _sample_gray(gray: np.ndarray, point: np.ndarray,
                 gray_box: np.ndarray | None = None) -> float | None:
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    h, w = gray.shape
    if x < 2 or y < 2 or x >= w - 2 or y >= h - 2:
        return None
    if gray_box is not None:
        return float(gray_box[y, x])
    return float(np.mean(gray[y-2:y+3, x-2:x+3]))


def _edge_evidence(gray: np.ndarray, line: Segment,
                   gray_box: np.ndarray | None = None) -> tuple[float, float]:
    """Contrast across a proposed edge and center-dark line evidence.

    A solid dark synthetic line has a dark core; a true platform edge usually
    separates a brighter platform from a darker train/platform gap. Both can
    have strong Canny edges, so this cue is just one factor, not a classifier.
    """
    h, w = gray.shape
    if gray_box is None:
        gray_box = _box_mean5(gray)
    tangent = _direction(line)
    normal = np.array([-tangent[1], tangent[0]])
    offset = max(8.0, min(w, h) * .035)
    contrasts, cores = [], []
    ax, ay = line.a
    bx, by = line.b
    nx_off, ny_off = float(normal[0] * offset), float(normal[1] * offset)
    for t in _T_VALS_29:
        px = ax * (1.0 - t) + bx * t
        py = ay * (1.0 - t) + by * t
        lx, ly = int(round(px + nx_off)), int(round(py + ny_off))
        rx, ry = int(round(px - nx_off)), int(round(py - ny_off))
        cx, cy = int(round(px)), int(round(py))
        if (lx < 2 or ly < 2 or lx >= w - 2 or ly >= h - 2
                or rx < 2 or ry < 2 or rx >= w - 2 or ry >= h - 2
                or cx < 2 or cy < 2 or cx >= w - 2 or cy >= h - 2):
            continue
        left = float(gray_box[ly, lx])
        right = float(gray_box[ry, rx])
        center = float(gray_box[cy, cx])
        contrasts.append(abs(left - right))
        cores.append(max(0., .5 * (left + right) - center))
    if len(contrasts) < 9:
        return 0., 0.
    return float(np.median(contrasts)), float(np.median(cores))


def _zone_evidence(gray: np.ndarray, a: Segment, b: Segment, *,
                   gray_box: np.ndarray | None = None,
                   image_median: float | None = None) -> tuple[float, float, float] | None:
    """Measure gap and surface brightness *between* two lines.

    We sample matched x (for near-horizontal) or y (for steep) coordinates;
    this avoids erroneous quadrilaterals formed by independently clipped
    diagonal lines with differing endpoints.
    """
    h, w = gray.shape
    if gray_box is None:
        gray_box = _box_mean5(gray)
    direction = _direction(a)
    axis = 0 if abs(direction[0]) >= abs(direction[1]) else 1
    limit = float(w - 1 if axis == 0 else h - 1)
    lo = max(min(a.a[axis], a.b[axis]), min(b.a[axis], b.b[axis]), 0.0)
    hi = min(max(a.a[axis], a.b[axis]), max(b.a[axis], b.b[axis]), limit)
    if hi - lo < 8.0:
        return None
    vals = lo + (hi - lo) * _T_VALS_15
    pa = _axis_points_vec(a, axis, vals)
    pb = _axis_points_vec(b, axis, vals)
    if pa is None or pb is None:
        return None
    pax, pay = pa
    pbx, pby = pb
    u = _ZONE_US[None, :]
    xis = np.rint(pax[:, None] * (1.0 - u) + pbx[:, None] * u).astype(np.intp)
    yis = np.rint(pay[:, None] * (1.0 - u) + pby[:, None] * u).astype(np.intp)
    valid_pt = (xis >= 2) & (xis < w - 2) & (yis >= 2) & (yis < h - 2)
    valid_station = np.count_nonzero(valid_pt, axis=1) >= 2
    if np.count_nonzero(valid_station) < 8:
        return None
    gaps = np.hypot(pbx - pax, pby - pay)[valid_station]
    valid_mask = valid_pt & valid_station[:, None]
    inside = gray_box[yis[valid_mask], xis[valid_mask]].astype(np.float64)
    adaptive_reference = float(np.median(gray)) if image_median is None else image_median
    return (float(np.median(gaps)), float(np.median(inside)),
            float(np.count_nonzero(inside > adaptive_reference) / inside.size))


def _outside_dark_drop(gray: np.ndarray, a: Segment, b: Segment, *,
                       gray_box: np.ndarray | None = None) -> float:
    """Positive if the area *outside* the edge is darker than platform interior.

    Crucial distinction: both edges of a paint stripe are strong Hough lines,
    but neither separates a well-lit platform from a dark track gap.
    """
    h, w = gray.shape
    if gray_box is None:
        gray_box = _box_mean5(gray)
    tangent = _direction(a)
    axis = 0 if abs(tangent[0]) >= abs(tangent[1]) else 1
    limit = w - 1 if axis == 0 else h - 1
    differences: list[float] = []
    offset = max(12., min(h, w) * .042)
    for t in _T_VALS_21:
        v = limit * float(t)
        pa = _axis_point_xy(a, axis, v)
        pb = _axis_point_xy(b, axis, v)
        if pa is None or pb is None:
            continue
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        gap = math.hypot(dx, dy)
        if gap < 1.0:
            continue
        ux, uy = dx / gap, dy / gap
        b_step = min(offset, gap * .32)
        bx_i, by_i = int(round(pb[0] - ux * b_step)), int(round(pb[1] - uy * b_step))
        ax_i, ay_i = int(round(pb[0] + ux * offset)), int(round(pb[1] + uy * offset))
        if (2 <= bx_i < w - 2 and 2 <= by_i < h - 2
                and 2 <= ax_i < w - 2 and 2 <= ay_i < h - 2):
            differences.append(float(gray_box[by_i, bx_i]) - float(gray_box[ay_i, ax_i]))
    return float(np.median(differences)) if len(differences) >= 8 else 0.


def _paint_line_evidence(gray: np.ndarray, line: Segment, mask: np.ndarray | None = None, *,
                         gray_box: np.ndarray | None = None,
                         gray_p10: float | None = None,
                         gray_p90: float | None = None,
                         mask_dilated: np.ndarray | None = None) -> tuple[float, float]:
    """Support for a narrow BRIGHT stripe surrounded by walkable platform.

    Train body reflections also produce long white Hough lines; unlike platform
    paint, one side of a train reflection is frequently the dark track gap.
    Requiring light texture on *both* sides is deliberately conservative.
    Returns (fraction_of_good_samples, median_brightness_contrast).
    """
    h, w = gray.shape
    if gray_box is None:
        gray_box = _box_mean5(gray)
    if gray_p10 is None or gray_p90 is None:
        p10, p90 = np.percentile(gray, (10, 90))
    else:
        p10, p90 = gray_p10, gray_p90
    dynamic_range = max(float(p90 - p10), 1.0)
    min_peak_diff = max(5.0, dynamic_range * .08)

    tangent = _direction(line)
    nx, ny = float(-tangent[1]), float(tangent[0])
    far = max(20., min(h, w) * .06)
    ax, ay = line.a
    bx, by = line.b

    contrast: list[float] = []
    good: list[bool] = []
    floors: list[float] = []
    for t in _T_VALS_29:
        px = ax * (1.0 - float(t)) + bx * float(t)
        py = ay * (1.0 - float(t)) + by * float(t)
        sx0, sy0 = int(round(px - nx * far)), int(round(py - ny * far))
        sx1, sy1 = int(round(px + nx * far)), int(round(py + ny * far))
        if not (2 <= sx0 < w - 2 and 2 <= sy0 < h - 2 and 2 <= sx1 < w - 2 and 2 <= sy1 < h - 2):
            continue
        s0 = float(gray_box[sy0, sx0])
        s1 = float(gray_box[sy1, sx1])
        near_vals: list[float] = []
        for mul in (-.66, -.33, 0., .33, .66):
            delta = far * mul
            qx, qy = int(round(px + nx * delta)), int(round(py + ny * delta))
            if 2 <= qx < w - 2 and 2 <= qy < h - 2:
                near_vals.append(float(gray_box[qy, qx]))
        if not near_vals:
            continue
        peak = max(near_vals)
        floor = s0 if s0 <= s1 else s1
        floors.append(floor)
        contrast.append(peak - 0.5 * (s0 + s1))
        good.append(floor >= 50.0 and peak - floor >= min_peak_diff)

    # Real platform paint must have pavement on both sides (floor >= 50).
    # Train reflections border the dark track gap (floor ~ 20-30), which is rejected.
    median_floor = _fast_median(floors) if floors else 0.0
    if median_floor < 48.0:
        return 0., 0.

    mask_frac = 0.0
    if mask is not None:
        m_lut = mask_dilated if mask_dilated is not None else cv2.dilate(mask, _DILATE_5X5)
        h_m, w_m = m_lut.shape[:2]
        hits = 0
        total_h = 0
        for t in _T_VALS_31:
            xi = int(round(ax * (1.0 - float(t)) + bx * float(t)))
            yi = int(round(ay * (1.0 - float(t)) + by * float(t)))
            if 0 <= xi < w_m and 0 <= yi < h_m:
                total_h += 1
                if m_lut[yi, xi] > 0:
                    hits += 1
        if total_h:
            mask_frac = hits / total_h

    if len(good) < 6:
        if mask_frac >= 0.22:
            return mask_frac, max(_fast_median(contrast) if contrast else 0.0, dynamic_range * 0.12)
        return 0., 0.

    frac = max(sum(1 for g in good if g) / len(good), mask_frac)
    med_c = _fast_median(contrast) if contrast else 0.0
    return frac, med_c


def _line_mask_fraction(line: Segment, mask_dilated: np.ndarray) -> float:
    ax, ay = line.a
    bx, by = line.b
    h, w = mask_dilated.shape
    hits = total = 0
    for t in _T_VALS_31:
        x = int(round(ax * (1.0 - float(t)) + bx * float(t)))
        y = int(round(ay * (1.0 - float(t)) + by * float(t)))
        if 0 <= x < w and 0 <= y < h:
            total += 1
            hits += int(mask_dilated[y, x] > 0)
    return hits / total if total else 0.0


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

    # Parallel shift along line normal, re-extended cleanly to the viewport bounds.
    p_ref = (np.asarray(line.a, float) + np.asarray(line.b, float)) * 0.5 + normal * delta
    shifted = _extend_to_image(p_ref, direction, w, h)
    if shifted is None:
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
    return shifted, metadata


def _center_painted_stroke(mask: np.ndarray, line: Segment) -> tuple[Segment, float]:
    """Compatibility helper retained for earlier callers and tests."""
    centered, metadata = _painted_stroke_convention(mask, line, line, "stripe_center")
    return centered, float(metadata.get("stripe_shift_px", 0.0))


def _edge_anchored_stripe_candidate(
    mask: np.ndarray,
    ref_line: Segment,
    edge: Segment,
    convention: str,
) -> tuple[Segment, dict[str, Any]] | None:
    """Locate the closest painted stripe on the platform side of `edge`, weighted toward foreground."""
    h, w = mask.shape
    d = float(min(h, w))
    edge_dir = _direction(edge)
    edge_norm = np.array([-edge_dir[1], edge_dir[0]], dtype=float)
    side_dot = float(np.dot(
        (np.asarray(ref_line.a, float) + np.asarray(ref_line.b, float)) * 0.5
        - (np.asarray(edge.a, float) + np.asarray(edge.b, float)) * 0.5,
        edge_norm,
    ))
    if abs(side_dot) < 1e-6:
        return None
    to_platform = edge_norm if side_dot > 0 else -edge_norm

    min_off = max(4.0, 0.012 * d)
    max_off = max(72.0, 0.58 * d)
    offsets = np.arange(min_off, max_off + 1.0, 1.0, dtype=float)
    groove_gap = max(3, min(7, round(0.010 * d)))
    max_run_w = max(20.0, 0.16 * d)

    station_runs: list[tuple[np.ndarray, float, float]] = []
    for t in np.linspace(0.14, 0.94, 36):
        pe = np.asarray(edge.a, float) * (1.0 - t) + np.asarray(edge.b, float) * t
        if not (0.08 * h <= pe[1] <= 0.96 * h):
            continue
        xs = np.rint(pe[0] + to_platform[0] * offsets).astype(int)
        ys = np.rint(pe[1] + to_platform[1] * offsets).astype(int)
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not np.any(inside):
            continue
        flags = np.zeros(len(xs), dtype=bool)
        flags[inside] = mask[ys[inside], xs[inside]] > 0
        idx = np.flatnonzero(flags)
        if not len(idx):
            continue
        sub_runs = [r for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(r) >= 2]
        if not sub_runs:
            continue
        # Bridge narrow tactile-tile ribs within a single stripe without merging separate floor lines.
        merged: list[np.ndarray] = [sub_runs[0]]
        for r in sub_runs[1:]:
            if int(r[0]) - int(merged[-1][-1]) - 1 <= groove_gap and (int(r[-1]) - int(merged[-1][0]) + 1) <= max_run_w:
                merged[-1] = np.arange(int(merged[-1][0]), int(r[-1]) + 1)
            else:
                merged.append(r)
        valid_runs = [r for r in merged if 3 <= len(r) <= max_run_w]
        if not valid_runs:
            continue
        nearest = valid_runs[0]
        lo_o = float(offsets[nearest[0]])
        hi_o = float(offsets[nearest[-1]])
        target_o = hi_o if convention != "stripe_center" else 0.5 * (lo_o + hi_o)
        pt = pe + to_platform * target_o
        if 0.12 * h <= pt[1] <= 0.96 * h:
            station_runs.append((pt, float(hi_o - lo_o + 1.0), float(pe[1])))

    if len(station_runs) < 8:
        return None
    fg_runs = [sr for sr in station_runs if sr[2] >= 0.45 * h]
    if len(fg_runs) < 4:
        return None
    fg_med_w = float(np.median([sr[1] for sr in fg_runs]))
    if fg_med_w < 3.0 or fg_med_w > max(32.0, 0.16 * d):
        return None

    pts = np.asarray([sr[0] for sr in station_runs], dtype=float)
    ys_s = pts[:, 1]
    if float(ys_s.max() - ys_s.min()) < 0.20 * h:
        return None
    if int(np.count_nonzero(ys_s < 0.65 * h)) < 3 or int(np.count_nonzero(ys_s >= 0.65 * h)) < 3:
        return None
    y_norm = np.clip((ys_s - 0.15 * h) / max(0.80 * h, 1.0), 0.0, 1.0)
    wts = 1.0 + 3.0 * (y_norm ** 1.5)
    tol_px = max(3.5, 0.45 * fg_med_w)
    # Seed line via weighted 2-point consensus anchored on mid/near stations to ignore upper edge glare.
    seed_idx = np.flatnonzero(ys_s >= 0.42 * h)
    if len(seed_idx) > 10:
        seed_idx = seed_idx[np.linspace(0, len(seed_idx) - 1, 10, dtype=int)]
    best_inliers: np.ndarray | None = None
    best_wsum = -1.0
    for ii in range(len(seed_idx)):
        p1 = pts[seed_idx[ii]]
        for jj in range(ii + 1, len(seed_idx)):
            p2 = pts[seed_idx[jj]]
            vec = p2 - p1
            vnorm = float(np.hypot(vec[0], vec[1]))
            if vnorm < 0.12 * h:
                continue
            unit = vec / vnorm
            nvec = np.array([-unit[1], unit[0]])
            res = np.abs((pts - p1) @ nvec)
            inl = res <= tol_px
            wsum = float(np.sum(wts[inl]))
            if wsum > best_wsum:
                best_wsum = wsum
                best_inliers = inl
    try:
        seed_pts = pts[best_inliers] if (best_inliers is not None and int(np.count_nonzero(best_inliers)) >= 8) else pts
        seed_wts = wts[best_inliers] if (best_inliers is not None and int(np.count_nonzero(best_inliers)) >= 8) else wts
        model = fit_line(seed_pts, weights=seed_wts)
        inliers = model.residuals(pts) <= tol_px
        if int(np.count_nonzero(inliers)) < 8:
            return None
        in_pts = pts[inliers]
        if int(np.count_nonzero(in_pts[:, 1] < 0.65 * h)) < 3 or int(np.count_nonzero(in_pts[:, 1] >= 0.65 * h)) < 3:
            return None
        model = fit_line(in_pts, weights=wts[inliers])
    except ValueError:
        return None

    anc_seg = _extend_to_image(model.point, model.direction, w, h)
    if anc_seg is None or abs(float(np.dot(_direction(anc_seg), edge_dir))) < math.cos(math.radians(45.0)):
        return None
    return anc_seg, {
        "stripe_convention": convention,
        "stripe_refinement": "edge_anchored",
        "stripe_width_px": round(fg_med_w, 2),
        "edge_anchored_inliers": int(np.count_nonzero(inliers)),
    }


def _painted_stroke_candidates(mask: np.ndarray, line: Segment, edge: Segment,
                               convention: str) -> list[tuple[Segment, dict[str, Any]]]:
    """Return parallel-shifted stripe line plus an optional Near/Mid foreground angular refit."""
    base_seg, base_meta = _painted_stroke_convention(mask, line, edge, convention)
    out: list[tuple[Segment, dict[str, Any]]] = [(base_seg, base_meta)]
    if base_meta.get("stripe_refinement") != "applied":
        return out
    h, w = mask.shape
    direction = _direction(line)
    normal = np.array([-direction[1], direction[0]])
    reach = max(18, int(min(h, w) * .075))
    offsets = np.arange(-reach, reach + 1, dtype=float)
    edge_sign = float(np.dot(
        (np.asarray(edge.a) + np.asarray(edge.b)) * .5 - (np.asarray(base_seg.a) + np.asarray(base_seg.b)) * .5,
        normal,
    ))
    med_w = max(float(base_meta.get("stripe_width_px", 4.0)) - 1.0, 2.0)
    wide_tactile = med_w >= 18.0
    groove_gap = max(3, min(8, round(0.011 * min(h, w)))) if wide_tactile else 0
    max_fg_span = max(med_w * 2.8, med_w + 16.0) if wide_tactile else max(med_w * 1.65, med_w + 6.0)
    near_fg_span = (
        max(med_w * 2.8, med_w + 18.0)
        if (wide_tactile or med_w >= 4.0)
        else max_fg_span
    )
    min_fg_ratio = 0.45 if wide_tactile else 0.50
    max_reach_mult = 1.9 if wide_tactile else 1.6
    samples: list[np.ndarray] = []
    for t in np.linspace(.08, .94, 42):
        # Sample around the already parallel-shifted base_seg so wide shifts stay within +/-reach.
        p = np.asarray(base_seg.a, float) * (1.0 - t) + np.asarray(base_seg.b, float) * t
        xs = np.rint(p[0] + normal[0] * offsets).astype(int)
        ys = np.rint(p[1] + normal[1] * offsets).astype(int)
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        flags = np.zeros(len(xs), dtype=bool)
        flags[inside] = mask[ys[inside], xs[inside]] > 0
        indices = np.flatnonzero(flags)
        if not len(indices):
            continue
        raw_runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        if wide_tactile:
            # Bridge narrow tactile-tile grooves so foreground perspective taper is sampled across the full block.
            sub_runs = [run for run in raw_runs if len(run) >= 2]
            if not sub_runs:
                continue
            merged_runs: list[np.ndarray] = [sub_runs[0]]
            for run in sub_runs[1:]:
                if (int(run[0]) - int(merged_runs[-1][-1]) - 1 <= groove_gap
                        and (int(run[-1]) - int(merged_runs[-1][0]) + 1) <= reach * max_reach_mult):
                    merged_runs[-1] = np.arange(int(merged_runs[-1][0]), int(run[-1]) + 1)
                else:
                    merged_runs.append(run)
            raw_runs = merged_runs
        runs = [run for run in raw_runs if 3 <= len(run) <= max(15, reach * max_reach_mult)]
        if not runs:
            continue
        closest = min(runs, key=lambda run: abs(float(np.mean(offsets[run]))))
        run_w = float(len(closest) - 1)
        lo_o = float(offsets[closest[0]])
        hi_o = float(offsets[closest[-1]])
        off = (lo_o if (convention != "stripe_center" and edge_sign >= 0)
               else (hi_o if convention != "stripe_center" else 0.5 * (lo_o + hi_o)))
        pt = p + normal * off
        persp_t = float(np.clip((pt[1] - 0.60 * h) / max(0.25 * h, 1.0), 0.0, 1.0))
        local_max_span = max_fg_span + persp_t * (near_fg_span - max_fg_span)
        if not (med_w * min_fg_ratio <= run_w <= min(reach * max_reach_mult, local_max_span)):
            continue
        if 0.45 * h <= pt[1] <= 0.96 * h:
            samples.append(pt)
    if len(samples) < 8:
        return out
    pts = np.asarray(samples, dtype=float)
    ys_s = pts[:, 1]
    y_mid_lo, y_mid_hi, y_near_hi = 0.45 * h, 0.70 * h, 0.96 * h
    if int(np.count_nonzero(ys_s < y_mid_hi)) < 4 or int(np.count_nonzero(ys_s >= y_mid_hi)) < 4:
        return out
    if float(ys_s.max() - ys_s.min()) < 0.18 * h:
        return out
    y_norm = np.clip((ys_s - y_mid_lo) / max(y_near_hi - y_mid_lo, 1.0), 0.0, 1.0)
    wts = 1.0 + 3.0 * (y_norm ** 1.5)
    try:
        model = fit_line(pts, weights=wts)
        res = model.residuals(pts)
        inliers = res <= max(3.0, 0.45 * med_w)
        if int(np.count_nonzero(inliers)) >= 8:
            in_pts = pts[inliers]
            in_ys = in_pts[:, 1]
            if int(np.count_nonzero(in_ys < y_mid_hi)) >= 3 and int(np.count_nonzero(in_ys >= y_mid_hi)) >= 3:
                model = fit_line(in_pts, weights=wts[inliers])
    except ValueError:
        return out
    fg_seg = _extend_to_image(model.point, model.direction, w, h)
    if fg_seg is None:
        return out
    base_dir = _direction(base_seg)
    cos_diff = abs(float(np.dot(base_dir, _direction(fg_seg))))
    if cos_diff > math.cos(math.radians(0.35)) or cos_diff < math.cos(math.radians(8.0)):
        return out
    base_norm = np.array([-base_dir[1], base_dir[0]])
    base_res = np.abs((pts - np.asarray(base_seg.a, float)) @ base_norm)
    fg_res = model.residuals(pts)
    if float(np.median(fg_res)) > float(np.median(base_res)) * 0.90:
        return out
    if max(math.hypot(fg_seg.a[0] - base_seg.a[0], fg_seg.a[1] - base_seg.a[1]),
           math.hypot(fg_seg.b[0] - base_seg.b[0], fg_seg.b[1] - base_seg.b[1])) < 2.0:
        return out
    out.append((fg_seg, {**base_meta, "stripe_refinement": "applied_fg_angular"}))
    return out


def _preprocess_m1(image: np.ndarray, method: str) -> np.ndarray:
    from .preprocessing import PREPROCESSORS

    if method == "raw":
        return image.copy()
    if method == "clahe":
        return PREPROCESSORS["clahe"](image)
    if method == "gamma":
        return PREPROCESSORS["gamma"](image, gamma=1.4)
    raise ValueError(f"Unknown M1 preprocessing method: {method}")


def _cap_line_pool(
    pool: list[tuple[Segment, dict[str, Any], str]],
    max_paint: int = 18,
    max_edge: int = 20,
    min_y: float = 0.0,
    min_steep_y: float = 0.0,
) -> list[tuple[Segment, dict[str, Any], str]]:
    """Keep lower-reaching paint and edge lines, prioritizing steep bottom-reaching edges."""
    eligible = [item for item in pool if max(item[0].a[1], item[0].b[1]) >= min_y]
    paints = [item for item in eligible if item[2] == "paint"]
    train_bounds = [item for item in eligible if item[2] == "train_boundary"]
    edges = [item for item in eligible if item[2] not in {"paint", "train_boundary"}]

    def _pool_rank(it: tuple[Segment, dict[str, Any], str]) -> float:
        seg = it[0]
        supp = float(it[1].get("support_length", 0.0))
        dx = abs(seg.b[0] - seg.a[0])
        dy = abs(seg.b[1] - seg.a[1])
        is_steep_bottom = (dy >= dx * 0.7) and (max(seg.a[1], seg.b[1]) >= min_steep_y)
        return supp * (2.2 if is_steep_bottom else 1.0)

    paints.sort(key=_pool_rank, reverse=True)
    edges.sort(key=_pool_rank, reverse=True)
    return paints[:max_paint] + edges[:max_edge] + train_bounds


def _rescale_segment(seg: Segment, sx: float, sy: float, w: int, h: int) -> Segment:
    return Segment(
        (float(np.clip(seg.a[0] * sx, 0.0, w - 1.0)), float(np.clip(seg.a[1] * sy, 0.0, h - 1.0))),
        (float(np.clip(seg.b[0] * sx, 0.0, w - 1.0)), float(np.clip(seg.b[1] * sy, 0.0, h - 1.0))),
    )


def _segment_payload(line: Segment, width: int, height: int) -> dict[str, Any]:
    pixels = [list(map(float, line.a)), list(map(float, line.b))]
    normalized = [[point[0] / (width - 1), point[1] / (height - 1)] for point in pixels]
    return {"pixels": pixels, "normalized": normalized}


def _merge_line_pool(lines_with_meta: list[tuple[Segment, dict[str, Any], str]],
                     distance_px: float) -> list[tuple[Segment, dict[str, Any], str]]:
    """Deduplicate nearly identical fitted lines while keeping sharp Hough geometry."""
    pool: list[tuple[Segment, dict[str, Any], str]] = []
    dup_tol = max(4.0, distance_px * 0.38)
    for line, stats, source_tag in lines_with_meta:
        d_new = _direction(line)
        pa_new, pb_new = np.asarray(line.a), np.asarray(line.b)
        merged_idx = None
        for idx, (existing, ex_stats, ex_tag) in enumerate(pool):
            if (source_tag == "train_boundary") != (ex_tag == "train_boundary"):
                continue
            d_ex = _direction(existing)
            if abs(float(np.dot(d_ex, d_new))) < 0.9975:
                continue
            n_ex = np.array([-d_ex[1], d_ex[0]])
            mid_ex = (np.asarray(existing.a) + np.asarray(existing.b)) * 0.5
            if max(abs(float(np.dot(pa_new - mid_ex, n_ex))),
                   abs(float(np.dot(pb_new - mid_ex, n_ex)))) <= dup_tol:
                merged_idx = idx
                break
        if merged_idx is None:
            pool.append((line, stats, source_tag))
        else:
            existing, ex_stats, ex_tag = pool[merged_idx]
            new_is_ransac = stats.get("fit") == "ransac" and ex_stats.get("fit") != "ransac"
            better_support = float(stats.get("support_length", 0.0)) > float(ex_stats.get("support_length", 0.0))
            if new_is_ransac or (stats.get("fit") == ex_stats.get("fit") and better_support and source_tag == "paint"):
                pool[merged_idx] = (line, stats, source_tag)
    return pool


def _clip_perspective_pair(safety: Segment, edge: Segment,
                           width: int, height: int) -> tuple[HazardZone | None, float | None]:
    """Clip steep converging L/E segments cleanly below any finite horizon VP."""
    p1, p2 = np.asarray(safety.a, float), np.asarray(safety.b, float)
    q1, q2 = np.asarray(edge.a, float), np.asarray(edge.b, float)
    mat = np.column_stack([p2 - p1, -(q2 - q1)])
    vp_y: float | None = None
    vp_inter: np.ndarray | None = None
    if abs(float(np.linalg.det(mat))) > 1e-5:
        params = np.linalg.solve(mat, q1 - p1)
        inter = p1 + float(params[0]) * (p2 - p1)
        if 0.0 <= inter[1] < height * 0.70 and -width * 0.5 <= inter[0] <= width * 1.5:
            vp_inter = inter
            vp_y = float(inter[1])

    is_steep = max(abs(p2[1] - p1[1]), abs(q2[1] - q1[1])) >= max(abs(p2[0] - p1[0]), abs(q2[0] - q1[0])) * 0.7
    if vp_inter is None or not is_steep:
        try:
            z = HazardZone(safety, edge)
            z.polygon()
            return z, vp_y
        except ValueError:
            return None, vp_y

    y_top = float(min(height - 10, max(0, math.ceil(vp_inter[1] + max(8.0, min(height, width) * 0.025)))))
    y_bot = float(height - 1)
    if y_top >= y_bot - 5.0:
        return None, vp_y

    def _clip_y_bounds(seg: Segment, yb: float, yt: float) -> tuple[tuple[float, float], tuple[float, float]] | None:
        a, b = np.asarray(seg.a, float), np.asarray(seg.b, float)
        dx, dy = b[0] - a[0], b[1] - a[1]
        if abs(dy) < 1e-6:
            return None
        xb = a[0] + dx * (yb - a[1]) / dy
        if xb < 0.0:
            y0 = a[1] + (0.0 - a[0]) * dy / (dx + 1e-9)
            pa = (0.0, float(np.clip(y0, yt, yb)))
        elif xb > width - 1.0:
            yw = a[1] + (width - 1.0 - a[0]) * dy / (dx + 1e-9)
            pa = (float(width - 1.0), float(np.clip(yw, yt, yb)))
        else:
            pa = (float(xb), yb)

        xt = a[0] + dx * (yt - a[1]) / dy
        if xt < 0.0:
            y0t = a[1] + (0.0 - a[0]) * dy / (dx + 1e-9)
            pb = (0.0, float(np.clip(y0t, yt, yb)))
        elif xt > width - 1.0:
            ywt = a[1] + (width - 1.0 - a[0]) * dy / (dx + 1e-9)
            pb = (float(width - 1.0), float(np.clip(ywt, yt, yb)))
        else:
            pb = (float(xt), yt)
        if math.hypot(pb[0] - pa[0], pb[1] - pa[1]) < 4.0:
            return None
        return pa, pb

    cs = _clip_y_bounds(safety, y_bot, y_top)
    ce = _clip_y_bounds(edge, y_bot, y_top)
    if cs is None or ce is None:
        return None, vp_y
    try:
        z = HazardZone(Segment(*cs), Segment(*ce))
        z.polygon()
        return z, vp_y
    except ValueError:
        return None, vp_y


def _sample_channel(channel: np.ndarray, point: np.ndarray | tuple[float, float],
                    box_map: np.ndarray | None = None) -> float | None:
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    h, w = channel.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    if box_map is not None and 2 <= x < w - 2 and 2 <= y < h - 2:
        return float(box_map[y, x])
    x0, x1 = max(0, x - 2), min(w, x + 3)
    y0, y1 = max(0, y - 2), min(h, y + 3)
    return float(np.mean(channel[y0:y1, x0:x1]))


def _evaluate_paint_band(lab_l: np.ndarray, mask: np.ndarray,
                         safety: Segment, edge: Segment,
                         p10: float, contrast: float, *,
                         lab_l_box: np.ndarray | None = None,
                         mask_bool: np.ndarray | None = None) -> dict[str, Any]:
    """Assess whether candidate L sits on a bounded, platform-safe paint/tactile band."""
    h, w = lab_l.shape
    if lab_l_box is None:
        lab_l_box = _box_mean5(lab_l)
    d = float(min(h, w))
    axis = 0 if abs(_direction(safety)[0]) >= abs(_direction(safety)[1]) else 1
    frames = _station_frames(safety, edge, axis, _T_VALS_23, 6.0)
    if frames is None or frames[0].size < 8:
        return {"passed": False, "quality": 0.0, "reason": "insufficient_l_stations"}

    pLx, pLy, pEx, pEy, dx, dy, gap, nx, ny = frames
    valid_stations = int(pLx.size)
    mask_b = mask_bool if mask_bool is not None else (mask if mask.dtype == bool else (mask > 0))

    near_tol = max(4.0, 0.018 * d)
    e_rad = max(4, int(0.014 * d))
    off_max = max(10.0, 0.032 * d)
    inv_contrast = 1.0 / contrast

    # Fast path: check near-line paint support before wide width-run scans.
    near_limit = int(round(near_tol))
    near_offs = np.arange(-near_limit, near_limit + 1, dtype=np.float64)[None, :]
    nxs = np.rint(pLx[:, None] + nx[:, None] * near_offs).astype(np.intp)
    nys = np.rint(pLy[:, None] + ny[:, None] * near_offs).astype(np.intp)
    n_in = (nxs >= 0) & (nxs < w) & (nys >= 0) & (nys < h)
    n_hits = np.zeros_like(n_in)
    n_hits[n_in] = mask_b[nys[n_in], nxs[n_in]]
    has_near_arr = np.any(n_hits, axis=1)
    hits_near = int(np.count_nonzero(has_near_arr))
    lower_mask = pLy >= h * 0.45
    lower_total = int(np.count_nonzero(lower_mask))
    lower_hits = int(np.count_nonzero(has_near_arr & lower_mask))
    l_support = hits_near / valid_stations
    lower_support = (lower_hits / lower_total) if lower_total >= 3 else l_support
    if l_support < 0.42 or lower_support < 0.35:
        return {
            "passed": False,
            "quality": 0.0,
            "l_support": round(l_support, 4),
            "lower_support": round(lower_support, 4),
            "e_paint_support": 0.0,
        }

    reach = np.maximum(14.0, np.minimum(0.14 * d, 0.45 * gap)).astype(np.intp)
    max_reach = int(reach.max())
    offs = np.arange(-max_reach, max_reach + 1, dtype=np.float64)[None, :]
    xs = np.rint(pLx[:, None] + nx[:, None] * offs).astype(np.intp)
    ys = np.rint(pLy[:, None] + ny[:, None] * offs).astype(np.intp)
    in_mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h) & (np.abs(offs) <= reach[:, None])
    hits = np.zeros_like(in_mask)
    hits[in_mask] = mask_b[ys[in_mask], xs[in_mask]]

    e_offs = np.arange(-e_rad, e_rad + 1, dtype=np.float64)[None, :]
    exs = np.rint(pEx[:, None] + nx[:, None] * e_offs).astype(np.intp)
    eys = np.rint(pEy[:, None] + ny[:, None] * e_offs).astype(np.intp)
    e_in = (exs >= 0) & (exs < w) & (eys >= 0) & (eys < h)
    e_hits_arr = np.zeros_like(e_in)
    e_hits_arr[e_in] = mask_b[eys[e_in], exs[e_in]]
    e_paint_hits = int(np.count_nonzero(np.any(e_hits_arr, axis=1)))

    # Bridge narrow internal tactile-tile grooves only on rows showing genuine multi-ribbed structure.
    groove_k = max(3, min(7, int(round(0.010 * d)))) | 1
    closed_hits = cv2.morphologyEx(hits.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, groove_k), np.uint8)) > 0
    wide_gap_rows = gap > (4.0 * groove_k)
    rib_rows = np.zeros(valid_stations, dtype=bool)
    min_rib_span = max(18.0, 0.026 * d)
    for ri in np.flatnonzero(wide_gap_rows):
        row_cls = closed_hits[ri]
        if not np.any(row_cls):
            continue
        row_raw = hits[ri]
        c_pad = np.pad(row_cls, 1)
        c_ed = np.flatnonzero(c_pad[1:] ^ c_pad[:-1])
        for cs, ce in zip(c_ed[0::2], c_ed[1::2]):
            span = ce - cs
            if span < min_rib_span:
                continue
            dist = 0 if (cs <= max_reach <= ce - 1) else min(abs(int(cs) - max_reach), abs(int(ce) - 1 - max_reach))
            if dist > off_max:
                continue
            sub_pad = np.pad(row_raw[cs:ce], 1)
            s_ed = np.flatnonzero(sub_pad[1:] ^ sub_pad[:-1])
            s_lens = [int(b_ - a_) for a_, b_ in zip(s_ed[0::2], s_ed[1::2]) if (b_ - a_) >= 2]
            if len(s_lens) >= 2 and max(s_lens) <= 0.65 * span:
                rib_rows[ri] = True
                break
    use_closed = rib_rows if int(np.count_nonzero(rib_rows)) >= 3 else np.zeros(valid_stations, dtype=bool)
    band_hits = np.where(use_closed[:, None], closed_hits, hits)

    padded = np.zeros((valid_stations, offs.shape[1] + 2), dtype=bool)
    padded[:, 1:-1] = band_hits
    edges = padded[:, 1:] ^ padded[:, :-1]
    row_has_hit = np.any(band_hits, axis=1)

    widths: list[float] = []
    width_ratios: list[float] = []
    reach_hits = 0
    rec_rows: list[int] = []
    rec_widths: list[float] = []
    for i in np.flatnonzero(row_has_hit):
        idx = np.flatnonzero(edges[i])
        starts = idx[0::2] - max_reach
        ends = idx[1::2] - 1 - max_reach
        abs_mids = np.abs(0.5 * (starts + ends))
        k = int(np.argmin(abs_mids))
        if abs_mids[k] <= off_max:
            run_w = float(ends[k] - starts[k] + 1)
            widths.append(run_w)
            width_ratios.append(run_w / max(float(gap[i]), 1.0))
            r_i = int(reach[i])
            if int(starts[k]) == -r_i or int(ends[k]) == r_i:
                reach_hits += 1
            rec_rows.append(int(i))
            rec_widths.append(run_w)

    recovery_scores: list[float] = []
    if rec_rows:
        rr = np.asarray(rec_rows, dtype=np.intp)
        rw = np.asarray(rec_widths, dtype=np.float64)
        steps = np.stack((np.maximum(rw * 1.6, 0.04 * d), np.maximum(rw * 2.5, 0.07 * d)), axis=1)
        rxs = np.rint(pLx[rr, None] - steps * nx[rr, None]).astype(np.intp)
        rys = np.rint(pLy[rr, None] - steps * ny[rr, None]).astype(np.intp)
        r_in = (rxs >= 0) & (rxs < w) & (rys >= 0) & (rys < h)
        r_vals = np.full(rxs.shape, -1.0, dtype=np.float64)
        r_vals[r_in] = lab_l_box[rys[r_in], rxs[r_in]]
        has_rec = np.any(r_in, axis=1)
        thresh_rec = p10 + 0.12 * contrast
        if np.any(has_rec):
            passed_rec = np.max(r_vals[has_rec], axis=1) >= thresh_rec
            recovery_scores = [1.0 if ok else 0.0 for ok in passed_rec]

    bg_steps = _SAFE_BG_US[None, :] * d
    bxs = np.rint(pLx[:, None] - bg_steps * nx[:, None]).astype(np.intp)
    bys = np.rint(pLy[:, None] - bg_steps * ny[:, None]).astype(np.intp)
    b_in = (bxs >= 2) & (bxs < w - 2) & (bys >= 2) & (bys < h - 2)

    mxs = np.rint(pLx + 0.50 * dx).astype(np.intp)
    mys = np.rint(pLy + 0.50 * dy).astype(np.intp)
    m_in = (mxs >= 2) & (mxs < w - 2) & (mys >= 2) & (mys < h - 2)
    interior_bg_arr = (lab_l_box[mys[m_in], mxs[m_in]].astype(np.float64) - p10) * inv_contrast

    e_support = e_paint_hits / valid_stations
    w_med = _fast_median(widths) if widths else 0.0
    reach_frac = (reach_hits / len(widths)) if widths else 1.0
    if len(width_ratios) >= 4:
        med_r = _fast_median(width_ratios)
        if med_r > 1e-6:
            mad_r = _fast_median([abs(wr - med_r) for wr in width_ratios])
            width_cv = mad_r / med_r
        else:
            width_cv = 1.0
    else:
        width_cv = 1.0
    recovery_frac = (sum(recovery_scores) / len(recovery_scores)) if recovery_scores else 0.5

    mid_paint_hits = int(np.count_nonzero(mask_b[mys[m_in], mxs[m_in]])) if np.any(m_in) else 0
    mid_paint_frac = (mid_paint_hits / int(np.count_nonzero(m_in))) if np.any(m_in) else 1.0
    clear_curb_exception = (
        mid_paint_frac <= 0.72
        and reach_frac <= 0.28
        and recovery_frac >= 0.72
        and l_support >= 0.60
    )
    e_paint_ok = (e_support <= min(0.72, max(0.44, 0.80 * l_support))) or clear_curb_exception

    has_bg = int(np.count_nonzero(b_in)) >= 6
    if has_bg:
        safe_bg_raw = lab_l_box[bys[b_in], bxs[b_in]].astype(np.float64)
        safe_bg_med = float(np.median((safe_bg_raw - p10) * inv_contrast))
        safe_bg_abs_med = float(np.median(safe_bg_raw))
    else:
        safe_bg_med = 1.0
        safe_bg_abs_med = 255.0
    interior_bg_med = float(np.median(interior_bg_arr)) if interior_bg_arr.size > 0 else safe_bg_med
    ls_vals, ok_ls = _sample_box_arr(lab_l_box, pLx, pLy)
    l_strip_med = float(np.median((ls_vals[ok_ls] - p10) * inv_contrast)) if np.any(ok_ls) else safe_bg_med
    # Reject inverted pairs whose "safe side" lands in a pitch-black track pit (L < 45),
    # while allowing dark-grey station floors (L >= 45) bordered by light coping slabs.
    safe_bg_ok = not (
        safe_bg_med < 0.28
        and safe_bg_abs_med < 45.0
        and (interior_bg_med - safe_bg_med) >= 0.28
    )
    max_band_w = max(28.0, 0.095 * d)
    excessive_noise = (width_cv > 0.76 and reach_frac > 0.25 and e_support > 0.35)
    wide_coping_slab = (
        l_support >= 0.70
        and lower_support >= 0.80
        and max_band_w < w_med <= 0.18 * d
        and reach_frac <= 0.96
        and width_cv <= 0.18
        and recovery_frac >= 0.80
        and e_paint_ok
        and safe_bg_ok
    )
    passed = (
        l_support >= 0.42
        and lower_support >= 0.35
        and 3.0 <= w_med <= max_band_w
        and reach_frac <= 0.38
        and (3.0 <= w_med <= max_band_w or wide_coping_slab)
        and (reach_frac <= 0.38 or wide_coping_slab)
        and width_cv <= 0.90
        and not excessive_noise
        and e_paint_ok
        and safe_bg_ok
    )
    quality = (
        # band quality weighted toward full-span continuity, foreground support, and paint recovery
        0.40 * min(l_support / 0.95, 1.0)
        + 0.25 * min(lower_support / 0.95, 1.0)
        + 0.20 * max(0.0, 1.0 - max(0.0, w_med - 0.06 * d) / max(0.06 * d, 1.0)) * (1.0 - min(reach_frac, 1.0))
        + 0.15 * max(0.0, 1.0 - max(0.0, e_support - 0.35 * l_support))
        - 0.35 * min(max(0.0, interior_bg_med - safe_bg_med - 0.25), 0.50)
        - 0.30 * min(max(0.0, width_cv - 0.48) / 0.42, 1.0)
        - 0.45 * max(0.0, 0.80 - recovery_frac)
        - (0.04 if wide_coping_slab else 0.0)
    )
    return {
        "passed": passed,
        "quality": round(float(np.clip(quality, 0.0, 1.0)), 4),
        "l_support": round(l_support, 4),
        "lower_support": round(lower_support, 4),
        "e_paint_support": round(e_support, 4),
        "stripe_width_px": round(w_med, 2),
        "reach_hit_frac": round(reach_frac, 4),
        "width_cv": round(width_cv, 4),
        "recovery_frac": round(recovery_frac, 4),
        "wide_coping_slab": wide_coping_slab,
        "safe_bg_med": round(safe_bg_med, 4),
        "interior_bg_med": round(interior_bg_med, 4),
        "l_strip_med": round(l_strip_med, 4),
    }


def _sample_box_arr(box_map: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = box_map.shape
    xi = np.rint(xs).astype(np.intp)
    yi = np.rint(ys).astype(np.intp)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    vals = np.empty(xi.shape, dtype=np.float64)
    vals[ok] = box_map[yi[ok], xi[ok]]
    return vals, ok


def _evaluate_trench_drop(lab_l: np.ndarray, lab_b: np.ndarray, mask: np.ndarray,
                          safety: Segment, edge: Segment,
                          contrast: float, *,
                          lab_l_box: np.ndarray | None = None,
                          lab_b_box: np.ndarray | None = None,
                          mask_dilated: np.ndarray | None = None,
                          p10: float | None = None) -> dict[str, Any]:
    """Assess persistent curb-to-track brightness drop vs transient dark paint borders."""
    h, w = lab_l.shape
    if lab_l_box is None:
        lab_l_box = _box_mean5(lab_l)
    if lab_b_box is None:
        lab_b_box = _box_mean5(lab_b)
    d = float(min(h, w))
    axis = 0 if abs(_direction(safety)[0]) >= abs(_direction(safety)[1]) else 1
    min_gap = max(6.0, 0.03 * d)
    frames = _station_frames(safety, edge, axis, _T_VALS_23, min_gap)
    if frames is None or frames[0].size < 8:
        return {"passed": False, "quality": 0.0, "reason": "insufficient_e_stations"}

    _, _, pEx, pEy, _, _, gap, nx, ny = frames
    o_near_d = max(3.0, 0.015 * d)
    o_mid_d = max(6.0, 0.032 * d)
    o_far_d = max(10.0, 0.058 * d)
    inv_contrast = 1.0 / contrast

    d1 = np.minimum(0.025 * d, 0.30 * gap)
    d2 = np.minimum(0.050 * d, 0.60 * gap)
    i1, ok1 = _sample_box_arr(lab_l_box, pEx - d1 * nx, pEy - d1 * ny)
    i2, ok2 = _sample_box_arr(lab_l_box, pEx - d2 * nx, pEy - d2 * ny)
    oc, ok_c = _sample_box_arr(lab_l_box, pEx, pEy)
    on, ok_n = _sample_box_arr(lab_l_box, pEx + o_near_d * nx, pEy + o_near_d * ny)
    om, ok_m = _sample_box_arr(lab_l_box, pEx + o_mid_d * nx, pEy + o_mid_d * ny)
    of, ok_f = _sample_box_arr(lab_l_box, pEx + o_far_d * nx, pEy + o_far_d * ny)
    o_ext_d = max(24.0, 0.16 * d)
    oe, ok_e = _sample_box_arr(lab_l_box, pEx + o_ext_d * nx, pEy + o_ext_d * ny)

    valid = (ok1 | ok2) & ok_n & ok_m
    n_valid = int(np.count_nonzero(valid))
    if n_valid < 8:
        return {"passed": False, "quality": 0.0, "reason": "insufficient_e_stations"}

    inside = np.where(ok1 & ok2, 0.5 * (i1 + i2), np.where(ok1, i1, i2))[valid]
    pEx, pEy, nx, ny = pEx[valid], pEy[valid], nx[valid], ny[valid]
    oc, ok_c = oc[valid], ok_c[valid]
    on, om, of, ok_f = on[valid], om[valid], of[valid], ok_f[valid]
    oe, ok_e = oe[valid], ok_e[valid]

    ex = np.rint(pEx).astype(np.intp)
    ey = np.rint(pEy).astype(np.intp)
    e_ok = (ex >= 0) & (ex < w) & (ey >= 0) & (ey < h)
    if mask_dilated is not None:
        e_on_paint_count = int(np.count_nonzero(mask_dilated[ey[e_ok], ex[e_ok]] > 0))
    else:
        e_on_paint_count = 0
        for px_i, py_i in zip(ex[e_ok], ey[e_ok]):
            x0, x1 = max(0, int(px_i) - 3), min(w, int(px_i) + 4)
            y0, y1 = max(0, int(py_i) - 3), min(h, int(py_i) + 4)
            if np.any(mask[y0:y1, x0:x1] > 0):
                e_on_paint_count += 1

    dn = (inside - on) * inv_contrast
    dm = (inside - om) * inv_contrast
    d_peak = np.maximum(dn, dm)
    use_far_peak = ok_f & (of <= om + 0.08 * contrast)
    if np.any(use_far_peak):
        df_all = (inside - of) * inv_contrast
        d_peak = np.where(use_far_peak, np.maximum(d_peak, df_all), d_peak)

    core_both = ok_c & ok_f
    core_only_c = ok_c & ~ok_f
    core_vals = np.concatenate((
        np.maximum(0.0, 0.5 * (inside[core_both] + of[core_both]) - oc[core_both]) * inv_contrast,
        np.maximum(0.0, inside[core_only_c] - oc[core_only_c]) * inv_contrast,
    ))

    n_far = int(np.count_nonzero(ok_f))
    if n_far > 0:
        d_far_arr = (inside[ok_f] - of[ok_f]) * inv_contrast
        rebounds_arr = (of[ok_f] - np.minimum(on[ok_f], om[ok_f])) * inv_contrast
        bf, ok_bf = _sample_box_arr(lab_b_box, pEx[ok_f] + o_far_d * nx[ok_f], pEy[ok_f] + o_far_d * ny[ok_f])
        far_chromatic_count = int(np.count_nonzero(ok_bf & (np.abs(bf - 128.0) >= 12.0)))
        far_chromatic_frac = far_chromatic_count / n_far
        med_far = float(np.median(d_far_arr))
        p75_far = float(np.percentile(d_far_arr, 75))
        med_rebound = float(np.median(rebounds_arr))
    else:
        med_far = float(np.median(dm))
        p75_far = float(np.percentile(dm, 75)) if dm.size > 0 else 0.0
        med_rebound = 0.0
        far_chromatic_frac = 0.0

    med_near = float(np.median(dn))
    med_mid = float(np.median(dm))
    med_peak = float(np.median(d_peak))
    med_core = float(np.median(core_vals)) if core_vals.size > 0 else 0.0
    pos_frac = float(np.count_nonzero(d_peak > 0.10)) / n_valid
    e_paint_frac = e_on_paint_count / n_valid

    # Reject a painted stripe + dark border rebound masquerading as E, while
    # allowing an unpainted dark-core curb/rail (e.g. synthetic rail line fixtures).
    is_dark_border_rebound = (
        med_rebound > 0.24
        and med_far < 0.34
        and far_chromatic_frac < 0.35
        and not (med_core >= 0.45 and e_paint_frac < 0.15)
    )
    solid_dark_curb = (med_core >= 0.45 and med_near >= 0.20 and e_paint_frac < 0.15)
    passed = (
        (
            (med_near >= 0.10 or (med_near >= 0.0 and med_mid >= 0.35 and med_far >= 0.45 and pos_frac >= 0.85))
            and med_peak >= 0.16
            and (med_mid >= 0.10 or (med_near >= 0.24 and pos_frac >= 0.50))
            and pos_frac >= 0.50
            and not is_dark_border_rebound
        )
        or solid_dark_curb
    )
    persistence = max(0.0, min((med_mid + max(med_far, 0.0)) / 1.00, 1.0))
    if far_chromatic_frac >= 0.35 and med_near >= 0.22:
        persistence = max(persistence, min(med_near / 0.45, 1.0))
    if solid_dark_curb:
        persistence = max(persistence, min(med_core / 0.80, 0.75))
        persistence = max(persistence, min(med_core / 0.65, 0.90))
    quality = (
        0.15 * min(max(med_near, med_core * 0.80, 0.0) / 0.38, 1.0)
        + 0.30 * min(max(med_peak, med_core * 0.85, 0.0) / 0.60, 1.0)
        + 0.35 * persistence
        + 0.20 * min(max(pos_frac, 0.75 if solid_dark_curb else 0.0) / 0.90, 1.0)
        - (0.35 if is_dark_border_rebound else 0.0)
    )
    med_ext = (
        round(float(np.median((oe[ok_e] - p10) * inv_contrast)), 4)
        if (p10 is not None and int(np.count_nonzero(ok_e)) >= 4)
        else None
    )
    return {
        "passed": passed,
        "quality": round(float(np.clip(quality, 0.0, 1.0)), 4),
        "med_drop_near": round(med_near, 4),
        "med_drop_mid": round(med_mid, 4),
        "med_drop_far": round(med_far, 4),
        "p75_drop_far": round(p75_far, 4),
        "med_ext": med_ext,
        "med_core": round(med_core, 4),
        "med_rebound": round(med_rebound, 4),
        "pos_frac": round(pos_frac, 4),
        "dark_border_rebound": is_dark_border_rebound,
    }


def _has_interior_trench(lab_l: np.ndarray, safety: Segment, edge: Segment,
                         p10: float, contrast: float, *,
                         lab_l_box: np.ndarray | None = None) -> bool:
    """Reject candidate corridors [L, E] that enclose a deep dark track trench."""
    h, w = lab_l.shape
    if lab_l_box is None:
        lab_l_box = _box_mean5(lab_l)
    axis = 0 if abs(_direction(safety)[0]) >= abs(_direction(safety)[1]) else 1
    frames = _station_frames(safety, edge, axis, _T_VALS_15_TRENCH, 8.0)
    if frames is None or frames[0].size < 6:
        return False

    pLx, pLy, _, _, dx, dy, _, _, _ = frames
    u = _TRENCH_US[None, :]
    xis = np.rint(pLx[:, None] + dx[:, None] * u).astype(np.intp)
    yis = np.rint(pLy[:, None] + dy[:, None] * u).astype(np.intp)
    in_bounds = (xis >= 0) & (xis < w) & (yis >= 0) & (yis < h)
    row_ok = np.all(in_bounds, axis=1)
    valid = int(np.count_nonzero(row_ok))
    if valid < 6:
        return False

    vals = (lab_l_box[yis[row_ok], xis[row_ok]].astype(np.float64) - p10) * (1.0 / contrast)
    # Check whether an internal station (u = 0.34..0.89) plunges sharply below both flanks
    inner_rel = np.argmin(vals[:, 1:-2], axis=1)
    inner_idx = 1 + inner_rel
    rows_idx = np.arange(valid)
    inner_min = vals[rows_idx, inner_idx]
    # Left flank maximum over cols 0..inner_idx-1 (mask columns >= inner_idx)
    col_grid = np.arange(vals.shape[1])[None, :]
    left_vals = np.where(col_grid < inner_idx[:, None], vals, -np.inf)
    left_hi = np.max(left_vals, axis=1)
    right_hi = vals[:, -1]
    valley_stations = int(np.count_nonzero(
        (inner_min <= 0.28) & ((left_hi - inner_min) >= 0.26) & ((right_hi - inner_min) >= 0.24)
    ))
    if valley_stations / valid >= 0.33:
        return True

    med_prof = np.median(vals, axis=0)
    m_inner_idx = 1 + int(np.argmin(med_prof[1:-2]))
    m_inner_min = float(med_prof[m_inner_idx])
    m_left_hi = float(np.max(med_prof[:m_inner_idx]))
    m_right_hi = float(med_prof[-1])
    asymmetric_trench = (
        m_inner_min <= 0.16 and (m_left_hi - m_inner_min) >= 0.35 and m_right_hi <= 0.45
    )
    return bool(
        asymmetric_trench
        or (m_inner_min <= 0.28 and (m_left_hi - m_inner_min) >= 0.25 and (m_right_hi - m_inner_min) >= 0.22)
    )


def _classify_vp(safety: Segment, edge: Segment, width: int,
                 height: int) -> dict[str, float | str | None]:
    """Report the unbounded line intersection and its perspective class."""
    p = np.asarray(safety.a, dtype=float)
    q = np.asarray(edge.a, dtype=float)
    ds = np.asarray(safety.b, dtype=float) - p
    de = np.asarray(edge.b, dtype=float) - q
    ns = float(np.linalg.norm(ds))
    ne = float(np.linalg.norm(de))
    if ns <= 1e-9 or ne <= 1e-9:
        return {"vp_x": None, "vp_y": None, "vp_class": "near_parallel",
                "angular_separation_deg": 0.0}
    cosine = float(np.clip(abs(np.dot(ds, de)) / (ns * ne), 0.0, 1.0))
    angular = math.degrees(math.acos(cosine))
    mat = np.column_stack((ds, -de))
    det = float(np.linalg.det(mat))
    if abs(det) <= 1e-9:
        return {"vp_x": None, "vp_y": None, "vp_class": "near_parallel",
                "angular_separation_deg": round(angular, 4)}
    t = float(np.linalg.solve(mat, q - p)[0])
    vp = p + t * ds
    vx, vy = float(vp[0]), float(vp[1])
    if angular < 1.5 or abs(vy) > 12.0 * height:
        vp_class = "near_parallel"
    elif vx < -0.5 * width or vx > 1.5 * width:
        vp_class = "outside_lateral"
    elif vy < -0.5 * height:
        vp_class = "far_above"
    elif vy < 0.70 * height:
        vp_class = "plausible"
    elif vy < height:
        vp_class = "inside_low"
    else:
        vp_class = "below_image"
    return {"vp_x": round(vx, 4), "vp_y": round(vy, 4),
            "vp_class": vp_class, "angular_separation_deg": round(angular, 4)}


_VP_ADJUSTMENTS = {
    "plausible": 0.0,
    "far_above": -14.0,
    "near_parallel": -8.0,
    "inside_low": -14.0,
    "below_image": -18.0,
    "outside_lateral": -10.0,
}


def _vp_horizon_penalty(vp_y: float | None, line: Segment, height: int, width: int | None = None) -> float:
    """Penalize steep corridors whose intersection prematurely truncates the upper platform."""
    if vp_y is None or vp_y < 0.0 or height <= 0:
        return 0.0
    dx = abs(line.b[0] - line.a[0])
    dy = abs(line.b[1] - line.a[1])
    if dy < dx * 0.7:
        return 0.0
    is_landscape = width is not None and width > height
    thr = 0.36 if is_landscape else 0.19
    cap = 0.48 if is_landscape else 0.45
    return max(0.0, min(vp_y / float(height), cap) - thr) * 480.0


def _axial_angle(line: Segment) -> float:
    return math.degrees(math.atan2(line.b[1] - line.a[1],
                                   line.b[0] - line.a[0])) % 180.0


def _axial_distance(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def _yellow_orientation_consensus(line_meta: list[dict[str, Any]], d: float) -> dict[str, Any]:
    eligible = [
        item for item in line_meta
        if float(item.get("yellow_frac", 0.0)) >= 0.35
        and float(item.get("yellow_purity", 0.0)) >= 0.70
        and float(item["stats"].get("support_length", 0.0)) >= 0.18 * d
    ]
    if len(eligible) < 2:
        return {"valid": False, "angle_deg": None, "eligible_lines": len(eligible)}
    weights = np.asarray([float(x["stats"].get("support_length", 0.0)) for x in eligible])
    angles = np.asarray([_axial_angle(x["line"]) for x in eligible])
    best: tuple[float, float, int] | None = None
    for center in angles:
        chosen = np.asarray([_axial_distance(float(a), float(center)) <= 8.0 for a in angles])
        weight = float(np.sum(weights[chosen]))
        count = int(np.count_nonzero(chosen))
        if best is None or (weight, count, -float(center)) > (best[0], best[2], -best[1]):
            # Axial circular mean via doubled angles.
            radians = np.deg2rad(2.0 * angles[chosen])
            mean = 0.5 * math.degrees(math.atan2(float(np.sum(weights[chosen] * np.sin(radians))),
                                                  float(np.sum(weights[chosen] * np.cos(radians))))) % 180.0
            best = (weight, mean, count)
    assert best is not None
    valid = best[2] >= 2 and best[0] >= 0.45 * float(np.sum(weights))
    return {"valid": valid, "angle_deg": round(best[1], 4) if valid else None,
            "eligible_lines": len(eligible), "cluster_lines": best[2],
            "weight_fraction": round(best[0] / max(float(np.sum(weights)), 1e-9), 4)}


def _orientation_adjustment(line: Segment, consensus: Mapping[str, Any]) -> float:
    if not consensus.get("valid") or consensus.get("angle_deg") is None:
        return 0.0
    distance = _axial_distance(_axial_angle(line), float(consensus["angle_deg"]))
    if distance <= 8.0:
        return 14.0
    if distance < 18.0:
        return 14.0 * (18.0 - distance) / 10.0
    if distance < 28.0:
        return -8.0 * (distance - 18.0) / 10.0
    return -8.0


def _line_mask_run_width(mask: np.ndarray, line: Segment, y: float,
                         reach: int) -> tuple[bool, float]:
    point = _axis_point_xy(line, 1, y)
    if point is None:
        return False, 0.0
    direction = _direction(line)
    normal = np.array((-direction[1], direction[0]))
    offsets = np.arange(-reach, reach + 1, dtype=float)
    xs = np.rint(point[0] + offsets * normal[0]).astype(np.intp)
    ys = np.rint(point[1] + offsets * normal[1]).astype(np.intp)
    inside = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
    hits = np.zeros(offsets.size, bool)
    hits[inside] = mask[ys[inside], xs[inside]] > 0
    indices = np.flatnonzero(hits)
    if indices.size == 0:
        return False, 0.0
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    run = min(runs, key=lambda value: abs(float(np.mean(offsets[value]))))
    return abs(float(np.mean(offsets[run]))) <= max(5.0, 0.25 * reach), float(len(run))


def _premature_color_crossing(mask: np.ndarray, line: Segment,
                               vp: Mapping[str, Any]) -> bool:
    """Role-aware connected same-color continuation through an implausible VP."""
    vy_raw = vp.get("vp_y")
    if vy_raw is None:
        return False
    vy = float(vy_raw)
    h, w = mask.shape
    if not (0.16 * h <= vy <= 0.85 * h):
        return False
    reach = max(8, int(round(0.08 * min(h, w))))
    above_y = np.linspace(max(0.0, vy - 0.16 * h), max(0.0, vy - 0.03 * h), 7)
    below_y = np.linspace(min(h - 1.0, vy + 0.03 * h), min(h - 1.0, vy + 0.16 * h), 7)
    above = [_line_mask_run_width(mask, line, float(y), reach) for y in above_y]
    below = [_line_mask_run_width(mask, line, float(y), reach) for y in below_y]
    if sum(hit for hit, _ in above) < 4 or sum(hit for hit, _ in above[-3:]) < 2:
        return False
    below_widths = [width for hit, width in below if hit]
    above_widths = [width for hit, width in above if hit]
    if len(below_widths) < 3 or not above_widths:
        return False
    ratio = float(np.median(above_widths)) / max(float(np.median(below_widths)), 1.0)
    if not 0.5 <= ratio <= 1.8:
        return False
    corridor_y = np.linspace(max(0.0, vy - 0.03 * h), min(h - 1.0, vy + 0.03 * h), 13)
    corridor = [_line_mask_run_width(mask, line, float(y), reach)[0] for y in corridor_y]
    gap_run = longest = 0
    for hit in corridor:
        gap_run = 0 if hit else gap_run + 1
        longest = max(longest, gap_run)
    gap_px = longest * (corridor_y[-1] - corridor_y[0]) / max(len(corridor_y) - 1, 1)
    return bool(gap_px <= 0.03 * h)


def _trough_features(values: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.size < 7:
        return np.array([]), np.array([]), np.array([])
    minima = np.flatnonzero((values[1:-1] < values[:-2]) & (values[1:-1] <= values[2:])) + 1
    candidates: list[tuple[float, float, float, float]] = []
    for idx in minima:
        radius = max(6, min(24, values.size // 6))
        left = values[max(0, idx - radius):idx]
        right = values[idx + 1:min(values.size, idx + radius + 1)]
        if left.size == 0 or right.size == 0:
            continue
        prominence = min(float(np.max(left) - values[idx]), float(np.max(right) - values[idx]))
        raw_drop = min(float(np.max(raw[max(0, idx - radius):idx]) - raw[idx]),
                       float(np.max(raw[idx + 1:min(raw.size, idx + radius + 1)]) - raw[idx]))
        if prominence < 0.18 or raw_drop < 18.0:
            continue
        threshold = values[idx] + 0.5 * prominence
        lo = idx
        hi = idx
        while lo > 0 and values[lo - 1] <= threshold:
            lo -= 1
        while hi + 1 < values.size and values[hi + 1] <= threshold:
            hi += 1
        candidates.append((prominence, float(idx), prominence, float(hi - lo + 1)))
    # One physical rib can contain several pixel-scale minima. Deterministic
    # prominence-first suppression keeps its deepest trough only.
    selected: list[tuple[float, float, float]] = []
    min_distance = max(4.0, 0.045 * values.size)
    for _, position, prominence, width in sorted(candidates, key=lambda x: (-x[0], x[1])):
        if all(abs(position - existing[0]) >= min_distance for existing in selected):
            selected.append((position, prominence, width))
    selected.sort(key=lambda x: x[0])
    positions = [item[0] for item in selected]
    prominences = [item[1] for item in selected]
    widths = [item[2] for item in selected]
    return np.asarray(positions), np.asarray(prominences), np.asarray(widths)


def _periodic_pitch(values: np.ndarray, min_lag: int, max_lag: int) -> tuple[float, float]:
    centered = values.astype(float) - float(np.mean(values))
    energy = float(np.dot(centered, centered))
    if energy <= 1e-9:
        return 0.0, 1.0
    samples: list[tuple[int, float]] = []
    upper = min(max_lag, values.size // 2)
    for lag in range(max(2, min_lag), upper + 1):
        left, right = centered[:-lag], centered[lag:]
        denom = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
        corr = float(np.dot(left, right) / denom) if denom > 1e-9 else -1.0
        samples.append((lag, corr))
    local = [item for index, item in enumerate(samples)
             if 0 < index < len(samples) - 1
             and item[1] >= samples[index - 1][1]
             and item[1] >= samples[index + 1][1]
             and item[1] >= 0.12]
    if local:
        best_lag, best_corr = local[0]
    else:
        best_lag, best_corr = max(samples, key=lambda item: (item[1], -item[0]))
    return float(best_lag), float(np.clip((1.0 - max(best_corr, 0.0)) * 0.25, 0.0, 1.0))


def _evaluate_repeated_ribs(lab_l: np.ndarray, paint_union: np.ndarray,
                            safety: Segment, edge: Segment, p10: float,
                            contrast: float, trench: Mapping[str, Any],
                            band: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Detect deep, matched-pitch train-roof ribs on both sides of a pair."""
    h, w = lab_l.shape
    d = float(min(h, w))
    axis = 0 if abs(_direction(safety)[0]) >= abs(_direction(safety)[1]) else 1
    frames = _station_frames(safety, edge, axis, _T_VALS_23, max(6.0, 0.03 * d))
    base = {"passed": False, "roof_rib_reject": False, "observable_stations": 0,
            "rib_positive_stations": 0, "rib_positive_fraction": 0.0}
    if frames is None:
        return {**base, "reason": "insufficient_rib_stations"}
    pLx, pLy, _, _, _, _, gaps, nx, ny = frames
    radius = 0.22 * d
    step = 1.0
    pitches: list[float] = []
    transitions: list[int] = []
    safe_ranges: list[float] = []
    safe_trough_counts: list[int] = []
    far_trough_counts: list[int] = []
    station_details: list[dict[str, float | int]] = []
    positive = observable = 0
    prominence_floor = max(0.18, 0.60 * max(float(trench.get("med_drop_near", 0.0)),
                                             float(trench.get("med_drop_mid", 0.0)),
                                             float(trench.get("med_core", 0.0))))
    for lx, ly, gap, nxi, nyi in zip(pLx, pLy, gaps, nx, ny):
        safe_r = np.arange(-radius, -0.02 * d + step, step)
        far_r = np.arange(float(gap) + 0.02 * d, float(gap) + radius + step, step)
        def sample(rs: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
            xs = np.rint(lx + rs * nxi).astype(np.intp)
            ys = np.rint(ly + rs * nyi).astype(np.intp)
            ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            if int(np.count_nonzero(ok)) < 0.75 * rs.size:
                return None
            raw = lab_l[ys[ok], xs[ok]].astype(float)
            z = cv2.GaussianBlur(((raw - p10) / contrast).reshape(1, -1), (0, 0), 0.6).ravel()
            return raw, z
        safe = sample(safe_r)
        far = sample(far_r)
        if safe is None or far is None:
            continue
        observable += 1
        safe_ranges.append(float(np.percentile(safe[1], 95) - np.percentile(safe[1], 5)))
        sp, spr, swidth = _trough_features(safe[1], safe[0])
        fp, fpr, fwidth = _trough_features(far[1], far[0])
        safe_trough_counts.append(int(sp.size))
        far_trough_counts.append(int(fp.size))
        if sp.size < 2 or fp.size < 2:
            continue
        min_lag = max(4, int(round(0.012 * d)))
        max_lag = max(min_lag + 1, int(round(0.08 * d)))
        med_s, cv_s = _periodic_pitch(safe[1], min_lag, max_lag)
        med_f, cv_f = _periodic_pitch(far[1], min_lag, max_lag)
        pitch_cv = max(cv_s, cv_f)
        pitch_candidates: list[tuple[float, float]] = []
        for div_s in (1, 2, 3, 4):
            for div_f in (1, 2, 3, 4):
                ps, pf = med_s / div_s, med_f / div_f
                mismatch = abs(ps - pf) / max(max(ps, pf), 1e-9)
                pitch_candidates.append((mismatch, 0.5 * (ps + pf)))
        pitch_match, med_pitch = min(pitch_candidates, key=lambda item: (item[0], -item[1]))
        width_ratios = np.concatenate((swidth / max(med_s, 1e-9),
                                       fwidth / max(med_f, 1e-9)))
        # Half-prominence basins are bounded by adjacent rib midpoints.
        width_ratios = np.minimum(width_ratios, 0.55)
        width_ratio = float(np.median(width_ratios))
        bright_ratio = float(1.0 - width_ratio)
        med_prom = float(np.median(np.concatenate((spr, fpr))))
        transition_count = int(2 * (sp.size + fp.size))
        bright_runs = int(max(sp.size - 1, 0) + max(fp.size - 1, 0))
        station_positive = (
            bright_runs >= 3
            and 0.12 <= width_ratio <= 0.55
            and 0.20 <= bright_ratio <= 0.78
            and pitch_cv <= 0.28
            and pitch_match <= 0.25
            and med_prom >= prominence_floor
        )
        station_details.append({"safe_troughs": int(sp.size), "far_troughs": int(fp.size),
                                "pitch_cv": round(pitch_cv, 4), "pitch_match": round(pitch_match, 4),
                                "trough_width_pitch": round(width_ratio, 4),
                                "bright_width_pitch": round(bright_ratio, 4),
                                "median_prominence": round(med_prom, 4)})
        if station_positive:
            positive += 1
            pitches.append(med_pitch)
            transitions.append(transition_count)
    fraction = positive / observable if observable else 0.0
    if len(pitches) >= 2:
        harmonic_cvs: list[float] = []
        for target in pitches:
            normalized = [min((pitch / div for div in (1, 2, 3, 4)),
                              key=lambda value: abs(value - target)) for pitch in pitches]
            harmonic_cvs.append(float(np.std(normalized) / max(np.mean(normalized), 1e-9)))
        cross_cv = min(harmonic_cvs)
    else:
        cross_cv = float("inf")
    safe_range = float(np.median(safe_ranges)) if safe_ranges else 0.0
    med_transitions = float(np.median(transitions)) if transitions else 0.0
    med_safe_troughs = float(np.median(safe_trough_counts)) if safe_trough_counts else 0.0
    med_far_troughs = float(np.median(far_trough_counts)) if far_trough_counts else 0.0
    safe_bg_med = float(band.get("safe_bg_med", 1.0)) if band is not None else 1.0
    interior_bg_med = float(band.get("interior_bg_med", 0.0)) if band is not None else 0.0
    dark_roof_contrast = (
        band is not None
        and safe_bg_med <= 0.49
        and interior_bg_med >= 0.80
        and (interior_bg_med - safe_bg_med) >= 0.28
    )
    primary_rib_field = (
        observable >= 8
        and (
            (fraction >= 0.55 and cross_cv <= 0.20)
            or (
                dark_roof_contrast
                and fraction >= 0.80
                and cross_cv <= 0.26
                and med_far_troughs >= 4.0
                and med_transitions >= 18.0
            )
        )
        and safe_range >= 0.30
        and med_transitions >= 4.0
        and med_safe_troughs >= 3.5
    )
    dense_bilateral_ribs = (
        dark_roof_contrast
        and observable >= 8
        and fraction >= 0.14
        and cross_cv <= 0.14
        and safe_range >= 0.38
        and med_transitions >= 18.0
        and med_safe_troughs >= 4.0
        and med_far_troughs >= 4.0
    )
    reject = bool(primary_rib_field or dense_bilateral_ribs)
    return {
        "passed": reject, "roof_rib_reject": reject,
        "observable_stations": observable, "rib_positive_stations": positive,
        "rib_positive_fraction": round(fraction, 4),
        "cross_station_pitch_cv": None if not np.isfinite(cross_cv) else round(cross_cv, 4),
        "safe_window_dynamic_range": round(safe_range, 4),
        "median_transition_count": round(med_transitions, 3),
        "prominence_floor": round(prominence_floor, 4),
        "median_safe_troughs": med_safe_troughs,
        "median_far_troughs": med_far_troughs,
        "station_details": station_details[:8],
    }


def _prepare_train_exclusion(mask: np.ndarray | None, height: int, width: int,
                             mode: str) -> dict[str, Any]:
    result: dict[str, Any] = {"valid": False, "core": None, "mask": None,
                              "boundary": None, "fallback_reason": None}
    if mode not in {"yolo_seg", "hybrid_yolo_seg"}:
        return result
    if mask is None or not np.any(mask):
        result["fallback_reason"] = "missing_or_empty_exclusion_mask"
        return result
    binary = (mask > 0).astype(np.uint8)
    coverage = float(np.mean(binary))
    k = max(5, int(round(0.015 * min(height, width))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    core = cv2.erode(binary, kernel)
    if coverage < 0.005:
        reason = "train_mask_too_small"
    elif coverage > 0.85:
        reason = "train_mask_too_large"
    elif not np.any(core):
        reason = "empty_eroded_train_core"
    else:
        reason = None
    result.update({"valid": reason is None, "core": core.astype(bool), "mask": binary.astype(bool),
                   "boundary": cv2.morphologyEx(binary, cv2.MORPH_GRADIENT,
                                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))).astype(bool),
                   "fallback_reason": reason, "coverage": round(coverage, 8), "erosion_radius_px": k})
    return result


def _train_overlap_metrics(context: Mapping[str, Any], safety: Segment,
                           edge: Segment) -> dict[str, Any]:
    core = context.get("core")
    if not context.get("valid") or core is None:
        return {"valid_stations": 0, "l_train_core_overlap": 0.0,
                "safe_side_train_overlap": 0.0, "e_train_core_overlap": 0.0,
                "e_train_boundary_distance_px": None, "outside_train_overlap": 0.0,
                "e_train_raw_overlap": 0.0, "train_boundary_confirmed": False,
                "train_overhang_confirmed": False, "rejected": False}
    core = np.asarray(core, dtype=bool)
    h, w = core.shape
    d = float(min(h, w))
    axis = 0 if abs(_direction(safety)[0]) >= abs(_direction(safety)[1]) else 1
    frames = _station_frames(safety, edge, axis, _T_VALS_23, max(4.0, 0.02 * d))
    if frames is None:
        return {"valid_stations": 0, "l_train_core_overlap": 0.0,
                "safe_side_train_overlap": 0.0, "e_train_core_overlap": 0.0,
                "e_train_boundary_distance_px": None, "outside_train_overlap": 0.0,
                "e_train_raw_overlap": 0.0, "train_boundary_confirmed": False,
                "train_overhang_confirmed": False, "rejected": False}
    pLx, pLy, pEx, pEy, _, _, gaps, nx, ny = frames
    valid = int(pLx.size)
    radius = max(4, int(round(0.018 * d)))
    offsets = np.arange(-radius, radius + 1, dtype=float)[None, :]
    lxs = np.rint(pLx[:, None] + nx[:, None] * offsets).astype(np.intp)
    lys = np.rint(pLy[:, None] + ny[:, None] * offsets).astype(np.intp)
    lin = (lxs >= 0) & (lxs < w) & (lys >= 0) & (lys < h)
    tube_hits = np.zeros_like(lin)
    tube_hits[lin] = core[lys[lin], lxs[lin]]
    tube_frac = float(np.count_nonzero(tube_hits) / max(np.count_nonzero(lin), 1))
    lxi, lyi = np.rint(pLx).astype(np.intp), np.rint(pLy).astype(np.intp)
    in_l = (lxi >= 0) & (lxi < w) & (lyi >= 0) & (lyi < h)
    station_frac = float(np.mean(core[lyi[in_l], lxi[in_l]])) if np.any(in_l) else 0.0
    l_overlap = max(tube_frac, station_frac)
    strip_hits = strip_total = 0
    for lx, ly, gap, nxi, nyi in zip(pLx, pLy, gaps, nx, ny):
        length = min(0.18 * d, max(0.08 * d, 0.75 * float(gap)))
        rs = np.linspace(0.02 * d, length, 7)
        xs = np.rint(lx - rs * nxi).astype(np.intp)
        ys = np.rint(ly - rs * nyi).astype(np.intp)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        strip_total += int(np.count_nonzero(ok))
        strip_hits += int(np.count_nonzero(core[ys[ok], xs[ok]]))
    safe_overlap = strip_hits / max(strip_total, 1)
    exi, eyi = np.rint(pEx).astype(np.intp), np.rint(pEy).astype(np.intp)
    in_e = (exi >= 0) & (exi < w) & (eyi >= 0) & (eyi < h)
    e_overlap = float(np.mean(core[eyi[in_e], exi[in_e]])) if np.any(in_e) else 0.0
    boundary = context.get("boundary")
    if boundary is not None and np.any(in_e):
        boundary_arr = np.asarray(boundary, dtype=np.uint8)
        distance = cv2.distanceTransform((~boundary_arr.astype(bool)).astype(np.uint8), cv2.DIST_L2, 3)
        b_dists = distance[eyi[in_e], exi[in_e]]
        e_boundary_distance: float | None = round(float(np.median(b_dists)), 3)
        e_boundary_p75_distance: float | None = round(float(np.percentile(b_dists, 75)), 3)
    else:
        e_boundary_distance = None
        e_boundary_p75_distance = None
    raw_mask = np.asarray(context.get("mask"), dtype=bool) if context.get("mask") is not None else core
    e_raw_overlap = float(np.mean(raw_mask[eyi[in_e], exi[in_e]])) if np.any(in_e) else 0.0
    if np.any(in_e):
        ey_in = pEy[in_e]
        fg_e = ey_in >= float(np.median(ey_in))
        e_raw_fg_overlap = float(np.mean(raw_mask[eyi[in_e][fg_e], exi[in_e][fg_e]])) if np.any(fg_e) else e_raw_overlap
    else:
        e_raw_fg_overlap = 0.0
    out_hits = out_total = 0
    out_rs = np.linspace(0.015 * d, 0.10 * d, 5)
    for ex, ey, nxi, nyi in zip(pEx, pEy, nx, ny):
        xs = np.rint(ex + out_rs * nxi).astype(np.intp)
        ys = np.rint(ey + out_rs * nyi).astype(np.intp)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        out_total += int(np.count_nonzero(ok))
        out_hits += int(np.count_nonzero(raw_mask[ys[ok], xs[ok]]))
    outside_train_overlap = out_hits / max(out_total, 1)
    train_boundary_confirmed = bool(
        valid >= 8
        and outside_train_overlap >= 0.45
        and e_overlap <= 0.35
        and e_raw_overlap <= 0.45
        and e_raw_fg_overlap <= 0.30
        and (e_boundary_distance is None or e_boundary_distance <= max(20.0, 0.035 * d))
        and (e_boundary_p75_distance is None or e_boundary_p75_distance <= max(32.0, 0.055 * d))
    )
    train_overhang_confirmed = bool(
        valid >= 8
        and outside_train_overlap >= 0.70
        and l_overlap <= 0.25
        and safe_overlap <= 0.15
        and (e_boundary_distance is None or e_boundary_distance <= max(36.0, 0.065 * d))
    )
    rejected = bool(valid >= 8 and (l_overlap >= 0.35 or safe_overlap >= 0.25))
    return {"valid_stations": valid, "l_train_core_overlap": round(l_overlap, 4),
            "safe_side_train_overlap": round(safe_overlap, 4),
            "e_train_core_overlap": round(e_overlap, 4),
            "e_train_raw_overlap": round(e_raw_overlap, 4),
            "e_train_fg_raw_overlap": round(e_raw_fg_overlap, 4),
            "outside_train_overlap": round(outside_train_overlap, 4),
            "train_boundary_confirmed": train_boundary_confirmed,
            "train_overhang_confirmed": train_overhang_confirmed,
            "e_train_boundary_distance_px": e_boundary_distance,
            "e_train_boundary_p75_distance_px": e_boundary_p75_distance,
            "rejected": rejected}


def _train_score_adjustment(overlap: Mapping[str, Any], *, wide_coping_slab: bool = False) -> float:
    """Reward train-boundary adherence and penalize E running inside the train core."""
    if not overlap.get("valid_stations"):
        return 0.0
    if wide_coping_slab and overlap.get("train_overhang_confirmed"):
        return 24.0
    bonus = 32.0 if overlap.get("train_boundary_confirmed") else 0.0
    e_core = float(overlap.get("e_train_core_overlap", 0.0))
    penalty = 95.0 * min(max(e_core - 0.30, 0.0) / 0.35, 1.0) if e_core >= 0.35 else 0.0
    if not overlap.get("train_boundary_confirmed") and float(overlap.get("e_train_raw_overlap", 0.0)) >= 0.65:
        penalty = max(penalty, 45.0)
    return bonus - penalty


def _extract_train_boundary_candidates(
    train_context: Mapping[str, Any],
    bgr_image: np.ndarray,
    h: int,
    w: int,
) -> tuple[str | None, list[tuple[Segment, dict[str, Any], str]]]:
    """Infer unambiguous train side from YOLO mask and emit E-only platform-facing boundary proposals."""
    if not train_context.get("valid") or train_context.get("mask") is None:
        return None, []
    mask = np.asarray(train_context["mask"], dtype=bool)
    y0, y1 = int(round(0.35 * h)), max(int(round(0.35 * h)) + 1, int(round(0.96 * h)))
    border_w = max(3, int(round(0.025 * w)))
    left_touch = float(np.mean(mask[y0:y1, :border_w].any(axis=1)))
    right_touch = float(np.mean(mask[y0:y1, w - border_w:].any(axis=1)))
    if left_touch >= 0.40 and right_touch <= 0.15:
        train_side = "left"
    elif right_touch >= 0.40 and left_touch <= 0.15:
        train_side = "right"
    else:
        return None, []

    num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    anchor_w = max(6, int(round(0.08 * w)))
    anchor_slice = labels[:, :anchor_w] if train_side == "left" else labels[:, w - anchor_w:]
    anchored_ids = set(np.unique(anchor_slice).tolist()) - {0}
    min_cc_area = max(40, int(0.015 * h * w))
    keep_ids = [cid for cid in anchored_ids if int(stats[cid, cv2.CC_STAT_AREA]) >= min_cc_area]
    side_mask = np.isin(labels, keep_ids) if keep_ids else mask

    raw_rows: list[tuple[float, float]] = []
    for y in range(int(round(0.38 * h)), int(round(0.96 * h))):
        xs = np.flatnonzero(side_mask[y])
        if not xs.size:
            continue
        x_val = float(xs.max()) if train_side == "left" else float(xs.min())
        if 4.0 <= x_val <= w - 5.0:
            raw_rows.append((x_val, float(y)))
    if len(raw_rows) < 24:
        return train_side, []

    arr = np.asarray(raw_rows, dtype=float)
    xs_r, ys_r = arr[:, 0], arr[:, 1]
    clean_mask = np.ones(len(arr), dtype=bool)
    jump_tol = max(12.0, 0.06 * w)
    for idx in range(len(arr)):
        lo, hi = max(0, idx - 3), min(len(arr), idx + 4)
        if abs(xs_r[idx] - float(np.median(xs_r[lo:hi]))) > jump_tol:
            clean_mask[idx] = False
    clean_pts = arr[clean_mask]
    if len(clean_pts) < 24 or float(clean_pts[:, 1].max() - clean_pts[:, 1].min()) < 0.28 * h:
        return train_side, []

    fg_sel = clean_pts[:, 1] >= 0.42 * h
    fit_pts = clean_pts[fg_sel] if int(np.count_nonzero(fg_sel)) >= 20 else clean_pts
    d = float(min(h, w))
    try:
        sil_res = fit_line_ransac(
            fit_pts, max_trials=160,
            residual_threshold=max(2.5, 0.012 * d),
            min_inlier_ratio=0.45, random_state=17,
        )
        in_pts = fit_pts[sil_res.inlier_mask]
    except ValueError:
        in_pts = fit_pts
    if len(in_pts) < 12:
        return train_side, []
    y_norm = np.clip((in_pts[:, 1] - 0.42 * h) / max(0.54 * h, 1.0), 0.0, 1.0)
    wts = 1.0 + 3.0 * (y_norm ** 1.5)
    try:
        sil_model = fit_line(in_pts, weights=wts)
    except ValueError:
        return train_side, []
    sil_line = _extend_to_image(sil_model.point, sil_model.direction, w, h)
    if sil_line is None:
        return train_side, []

    span_len = math.hypot(sil_line.b[0] - sil_line.a[0], sil_line.b[1] - sil_line.a[1])
    proposals: list[tuple[Segment, dict[str, Any], str]] = [(
        sil_line,
        {"fit": "train_mask_boundary", "support_length": round(span_len, 3),
         "segment_count": len(in_pts), "can_be_L": False, "can_be_E": True,
         "train_side": train_side},
        "train_boundary",
    )]

    # Narrow platformward snap: locate dark gap / crisp curb edge just outside train silhouette
    gray_f = cv2.GaussianBlur(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 0)
    out_sign = 1.0 if train_side == "left" else -1.0
    neg_lim = max(2, int(round(0.008 * d)))
    pos_lim = max(10, int(round(0.032 * d)))
    snap_pts: list[tuple[float, float]] = []
    plat_inner = max(3, int(round(0.006 * d)))
    plat_outer = max(10, int(round(0.020 * d)))
    for y_i in range(int(round(0.42 * h)), int(round(0.94 * h)), 2):
        pt_s = _axis_point(sil_line, 1, float(y_i))
        if pt_s is None:
            continue
        x_s = float(pt_s[0])
        best_sc = 0.0
        best_x = None
        for off in range(-neg_lim, pos_lim + 1):
            xc = int(round(x_s + out_sign * off))
            if xc < 3 or xc >= w - 3:
                continue
            x_p0 = int(np.clip(round(xc + out_sign * plat_inner), 0, w - 1))
            x_p1 = int(np.clip(round(xc + out_sign * plat_outer), 0, w - 1))
            lo_px, hi_px = min(x_p0, x_p1), max(x_p0, x_p1)
            if hi_px <= lo_px:
                continue
            plat_mean = float(np.mean(gray_f[y_i, lo_px:hi_px + 1]))
            pit_val = float(gray_f[y_i, xc])
            grad_h = abs(float(gray_f[y_i, xc + 2]) - float(gray_f[y_i, xc - 2]))
            sc = (plat_mean - pit_val) + 0.55 * grad_h
            if sc > best_sc:
                best_sc = sc
                best_x = float(xc)
        if best_x is not None and best_sc >= 14.0:
            snap_pts.append((best_x, float(y_i)))
    if len(snap_pts) >= 18:
        sp_arr = np.asarray(snap_pts, dtype=float)
        if float(sp_arr[:, 1].max() - sp_arr[:, 1].min()) >= 0.22 * h:
            try:
                sp_res = fit_line_ransac(
                    sp_arr, max_trials=140,
                    residual_threshold=max(2.0, 0.010 * d),
                    min_inlier_ratio=0.45, random_state=31,
                )
                sp_in = sp_arr[sp_res.inlier_mask]
                if len(sp_in) >= 12:
                    sp_norm = np.clip((sp_in[:, 1] - 0.42 * h) / max(0.52 * h, 1.0), 0.0, 1.0)
                    sp_model = fit_line(sp_in, weights=1.0 + 3.0 * (sp_norm ** 1.5))
                    snap_line = _extend_to_image(sp_model.point, sp_model.direction, w, h)
                    if snap_line is not None and abs(float(np.dot(_direction(sil_line), _direction(snap_line)))) >= math.cos(math.radians(6.0)):
                        if max(math.hypot(snap_line.a[0] - sil_line.a[0], snap_line.a[1] - sil_line.a[1]),
                               math.hypot(snap_line.b[0] - sil_line.b[0], snap_line.b[1] - sil_line.b[1])) >= 2.0:
                            snap_len = math.hypot(snap_line.b[0] - snap_line.a[0], snap_line.b[1] - snap_line.a[1])
                            proposals.append((
                                snap_line,
                                {"fit": "train_mask_boundary_snap", "support_length": round(snap_len, 3),
                                 "segment_count": len(sp_in), "can_be_L": False, "can_be_E": True,
                                 "train_side": train_side},
                                "train_boundary",
                            ))
            except ValueError:
                pass
    return train_side, proposals


def _pair_in_list(pairs: list[tuple[Segment, Segment]], safety: Segment,
                  edge: Segment, tol_px: float) -> bool:
    """Return True when (safety, edge) is within tol_px midpoints of an existing pair."""
    return any(
        math.hypot(safety.midpoint[0] - s.midpoint[0], safety.midpoint[1] - s.midpoint[1]) < tol_px
        and math.hypot(edge.midpoint[0] - e.midpoint[0], edge.midpoint[1] - e.midpoint[1]) < tol_px
        for s, e in pairs
    )


def _distinct_candidate_geometry(first: HazardZone, second: HazardZone,
                                 width: int, height: int) -> tuple[bool, float]:
    errors: list[float] = []
    d = float(min(width, height))
    for frac in (0.2, 0.5, 0.8):
        for one, two in ((first.safety_line, second.safety_line),
                         (first.platform_edge, second.platform_edge)):
            p = np.asarray(one.a) * (1.0 - frac) + np.asarray(one.b) * frac
            axis = 0 if abs(_direction(one)[0]) >= abs(_direction(one)[1]) else 1
            q = _axis_point(two, axis, float(p[axis]))
            if q is not None:
                errors.append(float(np.linalg.norm(p - q)) / d)
    mean_sep = float(np.mean(errors)) if errors else 0.0
    side_a = np.sign(first.platform_edge.midpoint[0] - first.safety_line.midpoint[0])
    side_b = np.sign(second.platform_edge.midpoint[0] - second.safety_line.midpoint[0])
    return bool(mean_sep >= 0.06 or side_a != side_b), mean_sep


def estimate_zone(image: np.ndarray, method: str = "color_hough", color: str = "both",
                  clahe: bool = False, min_gap_frac: float = .06,
                  max_gap_frac: float = .78, *, preprocess: str = "raw",
                  stripe_convention: str = "safe_side_edge",
                  random_state: int = 0, ransac_max_trials: int = 240,
                  ransac_residual_threshold: float | None = None,
                  ransac_min_inlier_ratio: float = .45,
                  train_suppression: str = "structural",
                  exclusion_mask: np.ndarray | None = None,
                  exclusion_metadata: Mapping[str, Any] | None = None,
                  _dark_inset_pass: bool = False,
                  _precomputed: Mapping[str, Any] | None = None) -> LineResult:
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
    suppression_modes = {"off", "structural", "yolo_seg", "hybrid_yolo_seg"}
    if train_suppression not in suppression_modes:
        raise ValueError(f"train_suppression must be one of {sorted(suppression_modes)}")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR color image")
    orig_h, orig_w = image.shape[:2]
    if exclusion_mask is not None:
        if exclusion_mask.ndim != 2 or exclusion_mask.shape != (orig_h, orig_w):
            raise ValueError("exclusion_mask must have the same height and width as image")
        if exclusion_mask.dtype not in (np.bool_, np.uint8):
            raise ValueError("exclusion_mask must be boolean or uint8")
    base_diagnostics: dict[str, Any] = {
        "method": method, "requested_method": requested_method,
        "preprocess": preprocess, "stripe_convention": stripe_convention,
        "train_suppression": train_suppression,
        "exclusion_metadata": dict(exclusion_metadata or {}),
        "validation_status": "IMPLEMENTED_NOT_VALIDATED_ON_REAL_DATA",
        "coordinate_spaces": ["pixels", "normalized"],
    }
    if orig_w < 32 or orig_h < 32:
        return LineResult(None, "uncertain_image_too_small", base_diagnostics, [])
    work_full = _preprocess_m1(image, "clahe" if clahe and preprocess == "raw" else preprocess)
    if min(orig_h, orig_w) > 720:
        down_scale = 600.0 / float(min(orig_h, orig_w))
        w = max(32, int(round(orig_w * down_scale)))
        h = max(32, int(round(orig_h * down_scale)))
        work = cv2.resize(work_full, (w, h), interpolation=cv2.INTER_AREA)
    else:
        down_scale = 1.0
        w, h = orig_w, orig_h
        work = work_full
    sx_orig = (orig_w - 1.0) / max(w - 1.0, 1.0) if (w != orig_w) else 1.0
    sy_orig = (orig_h - 1.0) / max(h - 1.0, 1.0) if (h != orig_h) else 1.0
    eff_trials = min(int(ransac_max_trials), 90) if down_scale < 1.0 else int(ransac_max_trials)
    base_diagnostics["work_resolution"] = [w, h]
    base_diagnostics["work_scale"] = round(down_scale, 4)
    work_exclusion = None
    if exclusion_mask is not None:
        binary_exclusion = (exclusion_mask > 0).astype(np.uint8)
        work_exclusion = (cv2.resize(binary_exclusion, (w, h), interpolation=cv2.INTER_NEAREST)
                          if (w, h) != (orig_w, orig_h) else binary_exclusion)
    train_context = _prepare_train_exclusion(work_exclusion, h, w, train_suppression)
    if _precomputed is not None:
        auto_train_fallback = _precomputed["auto_train_fallback"]
        exclusion_mask = _precomputed["exclusion_mask"]
        base_diagnostics["exclusion_metadata"] = dict(_precomputed["exclusion_metadata"])
        train_context = _precomputed["train_context"]
    else:
        auto_train_fallback = None
        if train_suppression in {"yolo_seg", "hybrid_yolo_seg"} and exclusion_mask is None:
            from railguard.train_mask import get_default_train_mask_provider

            train_res = get_default_train_mask_provider().predict(image)
            merged_meta = dict(train_res.metadata)
            merged_meta.update(base_diagnostics.get("exclusion_metadata") or {})
            base_diagnostics["exclusion_metadata"] = merged_meta
            auto_train_fallback = train_res.metadata.get("fallback_reason")
            if np.any(train_res.mask):
                exclusion_mask = train_res.mask
        work_exclusion = None
        if exclusion_mask is not None:
            binary_exclusion = (exclusion_mask > 0).astype(np.uint8)
            work_exclusion = (cv2.resize(binary_exclusion, (w, h), interpolation=cv2.INTER_NEAREST)
                              if (w, h) != (orig_w, orig_h) else binary_exclusion)
        train_context = _prepare_train_exclusion(work_exclusion, h, w, train_suppression)
    base_diagnostics["train_prior_valid"] = bool(train_context.get("valid"))
    base_diagnostics["train_prior_fallback_reason"] = train_context.get("fallback_reason")
    base_diagnostics["train_prior_fallback_reason"] = (
        auto_train_fallback if (not train_context.get("valid") and auto_train_fallback is not None)
        else train_context.get("fallback_reason")
    )
    if "coverage" in train_context:
        base_diagnostics["train_prior_coverage"] = train_context["coverage"]

    color_space = "lab" if method.startswith("lab_") else "hsv"
    distance_px = max(4.0, min(h, w) * .022)
    hough_cap = 44 if down_scale == 1.0 else 26
    model_cap = 36 if down_scale == 1.0 else 18
    dark_mask = _dark_inset_mask(work) if _dark_inset_pass else None

    def _run_dark_inset_fallback(cached_edges: tuple[Any, Any] | None = None) -> LineResult | None:
        if _dark_inset_pass:
            return None
        pre = {
            "auto_train_fallback": auto_train_fallback,
            "exclusion_mask": exclusion_mask,
            "exclusion_metadata": base_diagnostics.get("exclusion_metadata") or {},
            "train_context": train_context,
        }
        if cached_edges is not None:
            pre["edges"], pre["edge_lines"] = cached_edges
        fb = estimate_zone(
            image, method=requested_method, color=color, clahe=clahe,
            min_gap_frac=min_gap_frac, max_gap_frac=max_gap_frac, preprocess=preprocess,
            stripe_convention=stripe_convention, random_state=random_state,
            ransac_max_trials=ransac_max_trials,
            ransac_residual_threshold=ransac_residual_threshold,
            ransac_min_inlier_ratio=ransac_min_inlier_ratio,
            train_suppression=train_suppression, _dark_inset_pass=True, _precomputed=pre,
        )
        if fb.zone is not None:
            fb.diagnostics["fallback_pass"] = "dark_inset"
            return fb
        return None
    train_exclude = train_context["core"] if train_context.get("valid") else None
    if method == "hybrid_region_evidence":
        paint_hsv = (
            _candidate_segments(work, "mask", color, False, custom_mask=dark_mask, exclude_mask=train_exclude)
            if dark_mask is not None else _candidate_segments(work, "color", color, False, "hsv", exclude_mask=train_exclude)
        )
        paint_lab = [] if dark_mask is not None else _candidate_segments(work, "color", color, False, "lab", exclude_mask=train_exclude)
        paint = paint_hsv + paint_lab
        fit_method = "ransac"
        color_space = "hybrid_hsv_lab"
        colored_lines = (_model_segments(
            paint, w, h, fit_method, distance_px=distance_px,
            random_state=random_state, max_trials=eff_trials,
            residual_threshold=ransac_residual_threshold,
            min_inlier_ratio=ransac_min_inlier_ratio, max_lines=model_cap)
            + _model_segments(paint, w, h, "hough", distance_px=distance_px, max_lines=hough_cap))
        edges = (
            _precomputed["edges"]
            if (_precomputed is not None and "edges" in _precomputed)
            else _candidate_segments(work, "canny", color, False, "hsv", exclude_mask=train_exclude)
        )
        edge_lines = (_model_segments(
            edges, w, h, fit_method, distance_px=distance_px,
            random_state=random_state + 10000, max_trials=eff_trials,
            residual_threshold=ransac_residual_threshold,
            min_inlier_ratio=ransac_min_inlier_ratio, max_lines=model_cap)
            + _model_segments(edges, w, h, "hough", distance_px=distance_px, max_lines=hough_cap))
    else:
        paint = (
            _candidate_segments(work, "mask", color, False, custom_mask=dark_mask, exclude_mask=train_exclude)
            if dark_mask is not None else _candidate_segments(work, "color", color, False, color_space, exclude_mask=train_exclude)
        )
        fit_method = ("ransac" if method.endswith("_ransac")
                      else "cluster" if method.endswith("_cluster") else "hough")
        cap_m = hough_cap if fit_method == "hough" else model_cap
        edges = (
            _precomputed["edges"]
            if (_precomputed is not None and "edges" in _precomputed)
            else _candidate_segments(work, "canny", color, False, "hsv", exclude_mask=train_exclude)
        )
        colored_lines = _model_segments(
            paint, w, h, fit_method, distance_px=distance_px,
            random_state=random_state, max_trials=eff_trials,
            residual_threshold=ransac_residual_threshold,
            min_inlier_ratio=ransac_min_inlier_ratio, max_lines=cap_m)
        edge_lines = _model_segments(
            edges, w, h, fit_method, distance_px=distance_px,
            random_state=random_state + 10000, max_trials=eff_trials,
            residual_threshold=ransac_residual_threshold,
            min_inlier_ratio=ransac_min_inlier_ratio, max_lines=cap_m)
    base_diagnostics.update({
        "color_space": color_space, "line_fit": fit_method,
        "paint_segments": len(paint), "edge_segments": len(edges),
        "paint_lines": len(colored_lines), "edge_lines": len(edge_lines),
    })
    if not colored_lines:
        fb_res = _run_dark_inset_fallback((edges, edge_lines))
        if fb_res is not None:
            return fb_res
        return LineResult(None, "uncertain_no_paint_line", base_diagnostics, [])

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    lab_img = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    lab_l = lab_img[:, :, 0].astype(float)
    lab_b = lab_img[:, :, 2].astype(float)
    gray_box = _box_mean5(gray)
    lab_l_box = _box_mean5(lab_l)
    lab_b_box = _box_mean5(lab_b)
    gray_p10, gray_median, gray_p90 = [float(v) for v in np.percentile(gray, (10, 50, 90))]
    p10, p90 = np.percentile(lab_l, (10, 90))
    contrast = max(float(p90 - p10), float(np.std(lab_l)), 1.0)
    robust_range = max(float(gray_p90 - gray_p10), float(np.std(gray)), 1.0)

    yellow_hsv, white_hsv = _paint_masks(work, color_space="hsv")
    yellow_lab, white_lab = _paint_masks(work, color_space="lab")
    yellow_only = cv2.bitwise_or(yellow_hsv, yellow_lab)
    bright_white = cv2.bitwise_and(cv2.bitwise_or(white_hsv, white_lab),
                                   cv2.bitwise_not(yellow_only))
    mask_hsv = paint_mask(work, color, False, "hsv")
    mask_lab = paint_mask(work, color, False, "lab")
    mask = mask_lab if color_space == "lab" else (cv2.bitwise_or(mask_hsv, mask_lab) if color_space.startswith("hybrid") else mask_hsv)
    mask_union = cv2.bitwise_or(mask_hsv, mask_lab)
    if dark_mask is not None:
        yellow_only = np.zeros_like(dark_mask)
        bright_white = dark_mask
        mask = dark_mask
        mask_union = dark_mask
        color_space = "dark_inset"
    mask_union_bool = mask_union > 0
    mask_union_dil = cv2.dilate(mask_union, _DILATE_5X5)
    yellow_dil = cv2.dilate(yellow_only, _DILATE_5X5)
    white_dil = cv2.dilate(bright_white, _DILATE_5X5)
    mask_dil = mask_union_dil if color_space.startswith("hybrid") else cv2.dilate(mask, _DILATE_5X5)

    # Deduplicate candidate lines across paint + canny + YOLO train boundary into one unified pool
    train_side, train_bound_lines = _extract_train_boundary_candidates(train_context, work, h, w)
    if train_side is not None:
        base_diagnostics["train_side"] = train_side
    tagged_lines = ([(line, st, "paint") for line, st in colored_lines]
                    + [(line, st, "edge") for line, st in edge_lines]
                    + train_bound_lines)
    merged_pool = _merge_line_pool(tagged_lines, distance_px=distance_px)
    if down_scale < 1.0:
        pool = _cap_line_pool(merged_pool, min_y=h * 0.35, min_steep_y=h * 0.72)
    else:
        pool = [item for item in merged_pool if max(item[0].a[1], item[0].b[1]) >= h * 0.35]

    # Precompute fast per-line attributes so pair scoring avoids redundant work
    line_meta: list[dict[str, Any]] = []
    same_mask = color_space.startswith("hybrid")
    for line, stats, tag in pool:
        pf, pc = _paint_line_evidence(
            gray, line, mask_union,
            gray_box=gray_box, gray_p10=gray_p10, gray_p90=gray_p90, mask_dilated=mask_union_dil,
        )
        if same_mask:
            pf_method = pf
        else:
            pf_method, _ = _paint_line_evidence(
                gray, line, mask,
                gray_box=gray_box, gray_p10=gray_p10, gray_p90=gray_p90, mask_dilated=mask_dil,
            )
        direction = _direction(line)
        yf = _line_mask_fraction(line, yellow_dil)
        wf = _line_mask_fraction(line, white_dil)
        paint_total = yf + wf
        purity = yf / max(paint_total, 1e-9)
        dominant = ("yellow" if yf >= 0.20 and purity >= 0.70
                    else "white" if wf >= 0.20 and purity <= 0.30
                    else "mixed" if paint_total >= 0.20 else "none")
        dx = abs(line.b[0] - line.a[0])
        dy = abs(line.b[1] - line.a[1])
        line_meta.append({
            "line": line, "stats": stats, "tag": tag,
            "can_be_L": bool(stats.get("can_be_L", True)),
            "can_be_E": bool(stats.get("can_be_E", True)),
            "dir": direction, "dx": dx, "dy": dy,
            "max_y": max(line.a[1], line.b[1]),
            "paint_frac": max(pf, pf_method),
            "yellow_frac": yf, "white_frac": wf,
            "yellow_purity": purity, "dominant_paint": dominant,
            "paint_contrast": pc,
        })
    orientation_consensus = _yellow_orientation_consensus(line_meta, float(min(h, w)))
    base_diagnostics["yellow_orientation_consensus"] = orientation_consensus

    def _score_pair(l_meta: dict[str, Any], e_meta: dict[str, Any],
                    cand_L: Segment, cand_E: Segment) -> dict[str, Any]:
        band = _evaluate_paint_band(
            lab_l, mask_union, cand_L, cand_E, float(p10), contrast,
            lab_l_box=lab_l_box, mask_bool=mask_union_bool,
        )
        if _dark_inset_pass and band["passed"]:
            l_strip = float(band.get("l_strip_med", 1.0))
            flanked_dark = (
                (float(band["safe_bg_med"]) - l_strip) >= 0.14
                and (float(band["interior_bg_med"]) - l_strip) >= 0.14
            )
            if not flanked_dark or band.get("wide_coping_slab") or float(band.get("width_cv", 1.0)) > 0.80:
                band = {**band, "passed": False}
        if not band["passed"]:
            return {
                "passed": False,
                "semantic_quality": 0.0,
                "band": band,
                "trench": {"passed": False, "quality": 0.0},
                "geom_q": 0.0,
            }
        trench = _evaluate_trench_drop(
            lab_l, lab_b, mask_union, cand_L, cand_E, contrast,
            lab_l_box=lab_l_box, lab_b_box=lab_b_box, mask_dilated=mask_union_dil,
            p10=float(p10),
        )
        if (
            not trench["passed"]
            and train_context.get("valid")
            and not trench.get("dark_border_rebound", False)
        ):
            t_overlap = _train_overlap_metrics(train_context, cand_L, cand_E)
            if not t_overlap.get("rejected") and (
                t_overlap.get("train_boundary_confirmed")
                or (band.get("wide_coping_slab") and t_overlap.get("train_overhang_confirmed"))
            ):
                trench = {
                    **trench,
                    "passed": True,
                    "quality": max(float(trench.get("quality", 0.0)), 0.72),
                    "yolo_train_boundary_rescued": True,
                }
        if (
            not trench["passed"]
            and _dark_inset_pass
            and not trench.get("dark_border_rebound", False)
            and float(trench.get("pos_frac", 0.0)) >= 0.50
            and float(trench.get("med_drop_far", 0.0)) >= 0.10
            and float(trench.get("p75_drop_far", 0.0)) >= 0.30
            and (float(band["safe_bg_med"]) - float(band.get("l_strip_med", 1.0))) >= 0.16
            and (float(band["interior_bg_med"]) - float(band.get("l_strip_med", 1.0))) >= 0.16
        ):
            trench = {
                **trench,
                "passed": True,
                "quality": max(float(trench.get("quality", 0.0)), 0.68),
                "dark_inset_glare_rescued": True,
            }
        if not trench["passed"]:
            return {
                "passed": False,
                "semantic_quality": 0.0,
                "band": band,
                "trench": trench,
                "geom_q": 0.0,
            }
        safe_bg_med = float(band.get("safe_bg_med", 1.0))
        interior_bg_med = float(band.get("interior_bg_med", safe_bg_med))
        far_ext_med = trench.get("med_ext")
        glossy_reflection = (
            far_ext_med is not None
            and safe_bg_med <= 0.25
            and (float(far_ext_med) - safe_bg_med) >= 0.34
            and float(far_ext_med) >= interior_bg_med + 0.14
        )
        if glossy_reflection:
            return {
                "passed": False,
                "semantic_quality": 0.0,
                "band": band,
                "trench": trench,
                "geom_q": 0.0,
            }
        geom_ok = True
        p1x, p1y = cand_L.a
        p2x, p2y = cand_L.b
        q1x, q1y = cand_E.a
        q2x, q2y = cand_E.b
        dpx, dpy = p2x - p1x, p2y - p1y
        dqx, dqy = q2x - q1x, q2y - q1y
        det = dpx * (-dqy) - dpy * (-dqx)
        vp_bonus = 0.0
        if abs(det) > 1e-5:
            t_param = ((q1x - p1x) * (-dqy) - (q1y - p1y) * (-dqx)) / det
            vx = p1x + t_param * dpx
            vy = p1y + t_param * dpy
            if 0 <= vy < h * 0.70 and -w * 0.5 <= vx <= w * 1.5:
                vp_bonus = 0.12
            elif vy >= h:
                vp_bonus = -0.10
        vert_q = 0.12 if l_meta["dy"] >= l_meta["dx"] * 1.3 else (-0.08 if l_meta["dx"] >= l_meta["dy"] * 0.95 else 0.0)
        supp_q = min(float(l_meta["stats"]["support_length"]) / max(min(h, w), 1), 3.0) / 3.0
        geom_q = float(np.clip(0.55 * supp_q + 0.25 + vp_bonus + vert_q, 0.0, 1.0))
        sem_q = 0.42 * float(band["quality"]) + 0.38 * float(trench["quality"]) + 0.20 * geom_q
        curb_trench_ok = (
            float(band["e_paint_support"]) <= 0.72
            or (float(trench["med_drop_far"]) >= 0.35 and float(trench["pos_frac"]) >= 0.75)
            or bool(trench.get("yolo_train_boundary_rescued"))
            or bool(trench.get("dark_inset_glare_rescued"))
        )
        passed = bool(band["passed"] and trench["passed"] and geom_ok and curb_trench_ok and sem_q >= 0.56)
        return {
            "passed": passed,
            "semantic_quality": round(float(sem_q), 4),
            "band": band,
            "trench": trench,
            "geom_q": round(geom_q, 4),
        }

    raw_options: list[tuple[float, Segment, Segment, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    ambiguous_rejections = 0
    min_pair_gap = min(h, w) * max(min_gap_frac, .04)
    max_pair_gap = min(h, w) * max_gap_frac
    min_clip_len = min(h, w) * 0.22
    for i in range(len(line_meta)):
        mi = line_meta[i]
        if mi["max_y"] < h * 0.35:
            continue
        dix, diy = float(mi["dir"][0]), float(mi["dir"][1])
        for j in range(i + 1, len(line_meta)):
            mj = line_meta[j]
            if mj["max_y"] < h * 0.35:
                continue
            if mi["paint_frac"] < 0.20 and mj["paint_frac"] < 0.20:
                continue
            if (mi["dy"] > mi["dx"] * 1.3 and mj["dx"] > mj["dy"] * 1.3) or (
                    mj["dy"] > mj["dx"] * 1.3 and mi["dx"] > mi["dy"] * 1.3):
                continue
            djx, djy = float(mj["dir"][0]), float(mj["dir"][1])
            if abs(dix * djx + diy * djy) < _COS_35_DEG:
                continue

            la, lb = mi["line"], mj["line"]
            pre_clip, pre_vp_y = _clip_perspective_pair(la, lb, w, h)
            if pre_clip is None:
                continue
            c_la, c_lb = pre_clip.safety_line, pre_clip.platform_edge
            if min(c_la.length(), c_lb.length()) < min_clip_len:
                continue

            is_steep = max(mi["dy"], mj["dy"]) >= max(mi["dx"], mj["dx"]) * 0.7
            if is_steep:
                if min(max(c_la.a[1], c_la.b[1]), max(c_lb.a[1], c_lb.b[1])) < h * 0.72:
                    continue
                if min(abs(c_la.b[1] - c_la.a[1]), abs(c_lb.b[1] - c_lb.a[1])) < h * 0.48:
                    continue
                pa_b = _axis_point_xy(c_la, 1, max(c_la.a[1], c_la.b[1]))
                pb_b = _axis_point_xy(c_lb, 1, max(c_lb.a[1], c_lb.b[1]))
                pa_t = _axis_point_xy(c_la, 1, min(c_la.a[1], c_la.b[1]))
                pb_t = _axis_point_xy(c_lb, 1, min(c_lb.a[1], c_lb.b[1]))
                if pa_b is not None and pb_b is not None and pa_t is not None and pb_t is not None:
                    if abs(pa_b[0] - pb_b[0]) < abs(pa_t[0] - pb_t[0]) * 0.85:
                        continue
            ze = _zone_evidence(gray, c_la, c_lb, gray_box=gray_box, image_median=gray_median)
            if ze is None:
                continue
            gap, interior_brightness, interior_light_fraction = ze
            if interior_brightness < 35.0:
                continue
            if not (min_pair_gap <= gap <= max_pair_gap):
                continue

            if _has_interior_trench(lab_l, c_la, c_lb, float(p10), contrast, lab_l_box=lab_l_box):
                continue

            # Test both semantic role assignments symmetrically on the clipped visible span
            eval_ab = (
                _score_pair(mi, mj, c_la, c_lb)
                if mi["paint_frac"] >= 0.20 and mi.get("can_be_L", True) and mj.get("can_be_E", True)
                else None
            )
            eval_ba = (
                _score_pair(mj, mi, c_lb, c_la)
                if mj["paint_frac"] >= 0.20 and mj.get("can_be_L", True) and mi.get("can_be_E", True)
                else None
            )

            candidates_dir: list[tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any], dict[str, Any], Segment, Segment]] = []
            if eval_ab is not None and eval_ab["passed"]:
                candidates_dir.append((eval_ab, eval_ba, mi, mj, la, lb))
            if eval_ba is not None and eval_ba["passed"]:
                candidates_dir.append((eval_ba, eval_ab, mj, mi, lb, la))
            if not candidates_dir:
                continue

            best_eval, rev_eval, l_m, e_m, cand_L, cand_E = max(
                candidates_dir, key=lambda x: x[0]["semantic_quality"]
            )
            vp_profile = _classify_vp(c_la if cand_L is la else c_lb,
                                      c_lb if cand_E is lb else c_la, w, h)
            role_mask = (yellow_only if l_m["dominant_paint"] == "yellow"
                         else bright_white if l_m["dominant_paint"] == "white"
                         else mask_union)
            if _premature_color_crossing(role_mask, cand_L, vp_profile):
                continue
            if rev_eval is not None and rev_eval["passed"]:
                margin = best_eval["semantic_quality"] - rev_eval["semantic_quality"]
                if margin < 0.08:
                    ambiguous_rejections += 1
                    continue

            sw = float(best_eval["band"].get("stripe_width_px", 0.0))
            if sw > 0 and gap <= sw * 1.9:
                continue

            vp_adjustment = _VP_ADJUSTMENTS[str(vp_profile["vp_class"])] - _vp_horizon_penalty(pre_vp_y, cand_L, h, w)
            orientation_adjustment = _orientation_adjustment(cand_L, orientation_consensus)
            compact_q = float(np.clip((0.080 * min(h, w) - sw) / (0.050 * min(h, w)), 0.0, 1.0))
            color_compact_bonus = (12.0 * float(l_m["yellow_purity"]) * compact_q
                                   if l_m["dominant_paint"] == "yellow" else 0.0)
            score = best_eval["semantic_quality"] * 500.0 + (
                min(float(l_m["paint_contrast"]) / robust_range, 2.0) * 15.0
            ) + vp_adjustment + orientation_adjustment + color_compact_bonus + (
                _train_score_adjustment(
                    _train_overlap_metrics(train_context, cand_L, cand_E),
                    wide_coping_slab=bool(best_eval["band"].get("wide_coping_slab")),
                )
                if train_context.get("valid") else 0.0
            )
            aux = {
                "gap_px": round(gap, 3),
                "gap_normalized": round(gap / min(h, w), 5),
                "interior_brightness_descriptive": round(interior_brightness, 3),
                "interior_light_fraction_descriptive": round(interior_light_fraction, 4),
                "paint_support_fraction": round(l_m["paint_frac"], 4),
                "paint_contrast": round(l_m["paint_contrast"], 3),
                "paint_roles": {
                    "yellow_frac": round(float(l_m["yellow_frac"]), 4),
                    "white_frac": round(float(l_m["white_frac"]), 4),
                    "paint_frac": round(float(l_m["paint_frac"]), 4),
                    "yellow_purity": round(float(l_m["yellow_purity"]), 4),
                    "dominant_paint": l_m["dominant_paint"],
                },
                "vp_profile": vp_profile,
                "vp_adjustment": round(vp_adjustment, 4),
                "orientation_adjustment": round(orientation_adjustment, 4),
                "color_compact_bonus": round(color_compact_bonus, 4),
                "pre_eval": best_eval,
                "l_meta": l_m,
                "e_meta": e_m,
            }
            raw_options.append((score, cand_L, cand_E, aux, l_m["stats"], e_m["stats"], best_eval))

    raw_options.sort(key=lambda item: (
        -item[0],
        tuple(round(float(value), 6) for segment in (item[1], item[2])
              for point in (segment.a, segment.b) for value in point),
    ))
    ranked: list[dict[str, Any]] = []
    refined_options: list[tuple[float, HazardZone, dict[str, Any]]] = []
    seen_refined: list[tuple[int, Segment, Segment]] = []
    seen_rib_rejected: list[tuple[Segment, Segment]] = []
    rib_rejected_pairs: set[int] = set()
    structural_rib_rejections = 0
    train_core_rejections = 0

    variant_options: list[tuple[int, Segment, Segment, dict[str, Any], dict[str, Any], dict[str, Any], Segment, dict[str, Any]]] = []
    base_fit_by_pair: dict[int, tuple[float, float, float]] = {}
    seen_anchored_edges: list[tuple[Segment, int]] = []
    for pair_idx, (_, cand_L, cand_E, aux, safety_stats, edge_stats, _) in enumerate(raw_options[:24]):
        # Step 1 & Step 2 invariant: refine painted stroke BEFORE canonical VP clip & final rescoring
        cand_variants = _painted_stroke_candidates(mask_union, cand_L, cand_E, stripe_convention)
        if pair_idx < 8 and not any(m.get("stripe_refinement") == "applied_fg_angular" for _, m in cand_variants):
            e_side = 1 if (cand_L.midpoint[0] - cand_E.midpoint[0]) >= 0 else -1
            already_e = any(
                se_side == e_side
                and math.hypot(cand_E.a[0] - se.a[0], cand_E.a[1] - se.a[1]) <= 6.0
                and math.hypot(cand_E.b[0] - se.b[0], cand_E.b[1] - se.b[1]) <= 6.0
                for se, se_side in seen_anchored_edges
            )
            if not already_e:
                seen_anchored_edges.append((cand_E, e_side))
                anchored = _edge_anchored_stripe_candidate(mask_union, cand_L, cand_E, stripe_convention)
                if anchored is not None:
                    cand_variants.append(anchored)
        for refined_L, stripe_metadata in cand_variants:
            variant_options.append((pair_idx, cand_L, cand_E, aux, safety_stats, edge_stats, refined_L, stripe_metadata))

    for pair_idx, cand_L, cand_E, aux, safety_stats, edge_stats, refined_L, stripe_metadata in variant_options:
        clipped_zone, vp_y = _clip_perspective_pair(refined_L, cand_E, w, h)
        if clipped_zone is None:
            continue

        # Verify positive non-crossing separation at visible stations
        dir_z = _direction(clipped_zone.safety_line)
        axis_z = 0 if abs(dir_z[0]) >= abs(dir_z[1]) else 1
        lim_z = (w - 1) if axis_z == 0 else (h - 1)
        l_pts_ok = True
        seps: list[float] = []
        ref_sign = None
        for frac in (0.15, 0.5, 0.85):
            # Evaluate on canonical segment endpoints directly
            pL = np.asarray(clipped_zone.safety_line.a) * (1 - frac) + np.asarray(clipped_zone.safety_line.b) * frac
            pE = _axis_point(clipped_zone.platform_edge, axis_z, pL[axis_z])
            if pE is None:
                l_pts_ok = False
                break
            d_vec = pE - pL
            g_loc = float(np.linalg.norm(d_vec))
            if g_loc < 4.0:
                l_pts_ok = False
                break
            seps.append(g_loc)
            if ref_sign is None:
                ref_sign = d_vec
            elif float(np.dot(d_vec, ref_sign)) <= 0:
                l_pts_ok = False
                break
        if not l_pts_ok or not seps:
            continue

        min_dim_z = float(min(w, h))
        if (float(max(seps)) / min_dim_z) > 0.72:
            continue
        # Deduplicate clipped zones across distinct raw pairs, while allowing X-crossing perspective refits
        # (applied_fg_angular / edge_anchored) to compete against earlier tilted candidates at the same midpoint.
        other_seen = [(s, e) for p_i, s, e in seen_refined if p_i != pair_idx]
        if stripe_metadata.get("stripe_refinement") in ("applied_fg_angular", "edge_anchored"):
            dir_cur_L = _direction(clipped_zone.safety_line)
            dir_cur_E = _direction(clipped_zone.platform_edge)
            cos_dup_tol = math.cos(math.radians(1.2))
            other_check = [
                (s, e)
                for s, e in other_seen
                if abs(float(np.dot(dir_cur_L, _direction(s)))) >= cos_dup_tol
                and abs(float(np.dot(dir_cur_E, _direction(e)))) >= cos_dup_tol
            ]
        else:
            other_check = other_seen
        if _pair_in_list(other_check, clipped_zone.safety_line, clipped_zone.platform_edge, distance_px * 0.6) or any(
            _distinct_candidate_geometry(
                clipped_zone, HazardZone(s, e, clipped_zone.unsafe_side), w, h
            )[1] < 0.025
            for s, e in other_check
        ):
            continue

        # Final verification & evidence recomputation AFTER stroke shift & VP clip
        final_eval = _score_pair(aux["l_meta"], aux["e_meta"], clipped_zone.safety_line, clipped_zone.platform_edge)
        if not final_eval["passed"]:
            continue
        rev_final = _score_pair(aux["e_meta"], aux["l_meta"], clipped_zone.platform_edge, clipped_zone.safety_line)
        if rev_final["passed"] and (final_eval["semantic_quality"] - rev_final["semantic_quality"]) < 0.08:
            ambiguous_rejections += 1
            continue

        rib_diagnostics = _evaluate_repeated_ribs(
            lab_l, mask_union, clipped_zone.safety_line, clipped_zone.platform_edge,
            float(p10), contrast, final_eval["trench"], final_eval["band"],
        )
        if train_suppression != "off" and rib_diagnostics["roof_rib_reject"]:
            if pair_idx not in rib_rejected_pairs and not _pair_in_list(
                seen_rib_rejected, clipped_zone.safety_line, clipped_zone.platform_edge, distance_px * 0.6
            ):
                rib_rejected_pairs.add(pair_idx)
                seen_rib_rejected.append((clipped_zone.safety_line, clipped_zone.platform_edge))
                structural_rib_rejections += 1
            continue
        train_overlap = _train_overlap_metrics(
            train_context, clipped_zone.safety_line, clipped_zone.platform_edge)
        if train_overlap["rejected"]:
            train_core_rejections += 1
            continue

        edge_contrast, edge_darkcore = _edge_evidence(gray, clipped_zone.platform_edge)
        region = _two_sided_region_evidence(gray, clipped_zone.safety_line, clipped_zone.platform_edge)
        # Blend multi-offset persistent trench score into two_sided_region if close-curb train occlusion slightly lowered wide 4.5% sample
        if not region.get("reliable", False) and final_eval["trench"]["passed"]:
            region = {
                **region,
                "reliable": True,
                "brightness_drop_normalized": max(
                    float(region.get("brightness_drop_normalized", 0.0)),
                    float(final_eval["trench"]["med_drop_near"]),
                ),
                "drop_consistency": max(
                    float(region.get("drop_consistency", 0.0)),
                    float(final_eval["trench"]["pos_frac"]),
                ),
                "reason": None,
            }

        final_vp = _classify_vp(clipped_zone.safety_line, clipped_zone.platform_edge, w, h)
        vp_pen = _vp_horizon_penalty(vp_y, clipped_zone.safety_line, h, w)
        cur_bq = float(final_eval["band"].get("quality", 0.0))
        cur_lsup = float(final_eval["band"].get("l_support", 0.0))
        ref_kind = str(stripe_metadata.get("stripe_refinement", ""))
        if ref_kind in ("applied_fg_angular", "edge_anchored"):
            base_fit = base_fit_by_pair.get(pair_idx)
            # Do not let a higher-support stripe refit lose solely because a tilted base crossed E earlier.
            if (
                base_fit is not None
                and cur_bq >= base_fit[1] + 0.03
                and cur_lsup >= base_fit[2] + 0.05
                and (vp_y is None or vp_y <= 0.36 * h)
            ):
                vp_pen = min(vp_pen, base_fit[0])
        else:
            prev_base = base_fit_by_pair.get(pair_idx)
            if prev_base is None or cur_bq > prev_base[1]:
                base_fit_by_pair[pair_idx] = (vp_pen, cur_bq, cur_lsup)
        vp_adjustment = _VP_ADJUSTMENTS[str(final_vp["vp_class"])] - vp_pen
        orientation_adjustment = _orientation_adjustment(clipped_zone.safety_line, orientation_consensus)
        final_width = float(final_eval["band"].get("stripe_width_px", 0.0))
        compact_q = float(np.clip((0.080 * min(h, w) - final_width) / (0.050 * min(h, w)), 0.0, 1.0))
        color_compact_bonus = (12.0 * float(aux["l_meta"]["yellow_purity"]) * compact_q
                               if aux["l_meta"]["dominant_paint"] == "yellow" else 0.0)
        train_adjustment = (
            _train_score_adjustment(train_overlap, wide_coping_slab=bool(final_eval["band"].get("wide_coping_slab")))
            if train_context.get("valid") else 0.0
        )
        final_score = final_eval["semantic_quality"] * 500.0 + (
            min(float(aux["paint_contrast"]) / robust_range, 2.0) * 15.0
        ) + (
            min(float(edge_contrast) / robust_range, 2.0) * 18.0
        ) + (
            min(float(edge_darkcore) / robust_range, 1.5) * 16.0
        ) + vp_adjustment + orientation_adjustment + color_compact_bonus + train_adjustment
        if (
            (
                final_score < 395.0
                and edge_darkcore < 18.0
                and (final_eval["band"]["quality"] < 0.90 or final_eval["trench"]["quality"] < 0.90)
            )
            or (final_score < 365.0 and final_eval["band"]["quality"] < 0.90 and final_eval["trench"]["quality"] < 0.70)
        ):
            continue
        evidence = {
            "gap_px": aux["gap_px"],
            "gap_normalized": aux["gap_normalized"],
            "interior_brightness_descriptive": aux["interior_brightness_descriptive"],
            "interior_light_fraction_descriptive": aux["interior_light_fraction_descriptive"],
            "edge_contrast": round(edge_contrast, 3),
            "edge_darkcore": round(edge_darkcore, 3),
            "paint_support_fraction": aux["paint_support_fraction"],
            "paint_contrast": aux["paint_contrast"],
            "paint_roles": aux["paint_roles"],
            "vp": final_vp,
            "vp_adjustment": round(vp_adjustment, 4),
            "orientation_adjustment": round(orientation_adjustment, 4),
            "color_compact_bonus": round(color_compact_bonus, 4),
            "train_adjustment": round(train_adjustment, 4),
            "repeated_ribs": rib_diagnostics,
            "train_mask_overlap": train_overlap,
            "two_sided_region": region,
            "semantic_profile": {
                "semantic_quality": final_eval["semantic_quality"],
                "reverse_quality": rev_final["semantic_quality"],
                "band": final_eval["band"],
                "trench": final_eval["trench"],
                "vp_y": round(vp_y, 2) if vp_y is not None else None,
            },
            "edge_evidence_type": "semantic_region_transition",
        }
        scaled_L = (_rescale_segment(clipped_zone.safety_line, sx_orig, sy_orig, orig_w, orig_h)
                    if (w != orig_w or h != orig_h) else clipped_zone.safety_line)
        scaled_E = (_rescale_segment(clipped_zone.platform_edge, sx_orig, sy_orig, orig_w, orig_h)
                    if (w != orig_w or h != orig_h) else clipped_zone.platform_edge)
        out_zone = HazardZone(scaled_L, scaled_E, clipped_zone.unsafe_side)
        seen_refined.append((pair_idx, clipped_zone.safety_line, clipped_zone.platform_edge))
        record = {
            "rank": 0,
            "safety_line": _segment_payload(scaled_L, orig_w, orig_h),
            "platform_edge": _segment_payload(scaled_E, orig_w, orig_h),
            "coordinate_space": "pixels_and_normalized",
            "support": {"safety": safety_stats, "platform_edge": edge_stats},
            "evidence": evidence,
            "diagnostics": stripe_metadata,
            "score": round(float(final_score), 4),
            "uncertainty_reason": "heuristic_rgb_semantics_require_review",
            "estimated_review_required": True,
        }
        refined_options.append((final_score, out_zone, record))

    if train_suppression != "off" and structural_rib_rejections >= 5:
        # Only a widespread confirmed roof-rib field supplies scene context for
        # alternate line pairs through the same field; isolated single-pair rib
        # positives on normal concrete platforms must not cascade into valid pairs.
        kept_options: list[tuple[float, HazardZone, dict[str, Any]]] = []
        for option in refined_options:
            ribs = option[2]["evidence"]["repeated_ribs"]
            partial_same_field = (
                int(ribs.get("observable_stations", 0)) >= 8
                and float(ribs.get("rib_positive_fraction", 0.0)) >= 0.30
                and float(ribs.get("safe_window_dynamic_range", 0.0)) >= 0.30
                and float(ribs.get("median_transition_count", 0.0)) >= 4.0
                and ribs.get("cross_station_pitch_cv") is not None
                and float(ribs["cross_station_pitch_cv"]) <= 0.20
            )
            if partial_same_field:
                structural_rib_rejections += 1
            else:
                kept_options.append(option)
        refined_options = kept_options
    scene_rib_dominant = bool(train_suppression != "off" and structural_rib_rejections >= 6)
    if scene_rib_dominant:
        refined_options = []

    refined_options.sort(
        key=lambda item: (
            -item[0],
            tuple(round(float(value), 6) for point in (
                item[1].safety_line.a, item[1].safety_line.b,
                item[1].platform_edge.a, item[1].platform_edge.b) for value in point),
        )
    )
    deduped_options: list[tuple[float, HazardZone, dict[str, Any]]] = []
    for option in refined_options:
        if any(
            _distinct_candidate_geometry(option[1], kept[1], orig_w, orig_h)[1] < 0.025
            for kept in deduped_options
        ):
            continue
        deduped_options.append(option)
        if len(deduped_options) >= 3:
            break
    refined_options = deduped_options
    for rank_idx, (_, _, rec) in enumerate(refined_options, start=1):
        rec["rank"] = rank_idx
        ranked.append(rec)

    diagnostics = {
        **base_diagnostics,
        "candidate_pairs": len(raw_options),
        "ambiguous_orientation_rejections": ambiguous_rejections,
        "structural_rib_rejections": structural_rib_rejections,
        "scene_rib_dominant": scene_rib_dominant,
        "train_core_rejections": train_core_rejections,
        "top_candidates": ranked,
    }
    if not refined_options:
        if ambiguous_rejections > 0 and not raw_options:
            diagnostics["uncertainty_reason"] = "ambiguous_semantic_role_ordering"
            return LineResult(None, "uncertain_ambiguous_orientation", diagnostics, [])
        diagnostics["uncertainty_reason"] = "no_reliable_two_sided_platform_edge_evidence"
        if not scene_rib_dominant and not (ambiguous_rejections > 0 and not raw_options):
            fb_res = _run_dark_inset_fallback((edges, edge_lines))
            if fb_res is not None:
                return fb_res
        return LineResult(None, "uncertain_no_platform_edge", diagnostics, [])

    score, zone, record = refined_options[0]
    if len(refined_options) >= 2:
        second_score, second_zone, _ = refined_options[1]
        distinct, separation = _distinct_candidate_geometry(zone, second_zone, orig_w, orig_h)
        close_threshold = max(8.0, 0.02 * abs(score))
        score_margin = score - second_score
        diagnostics["competing_geometry"] = {
            "score_margin": round(score_margin, 4),
            "threshold": round(close_threshold, 4),
            "mean_normalized_separation": round(separation, 5),
            "distinct": distinct,
        }
        if distinct and score_margin < close_threshold:
            diagnostics["uncertainty_reason"] = "uncertain_competing_geometry"
            return LineResult(None, "uncertain_competing_geometry", diagnostics, ranked)
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
