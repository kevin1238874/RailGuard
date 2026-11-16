"""Optional local contact-point refinement with structured evidence and reflection rejection."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detection import Person, Point


@dataclass(frozen=True)
class FootEvidence:
    """Structured per-foot contact assessment returned by refinement."""

    point: Point
    seed: Point
    confidence: float
    status: str  # accepted, ambiguous, rejected_reflection, no_support
    reason: str
    candidate_margin: float

    def as_dict(self) -> dict:
        return {
            "point": [float(self.point[0]), float(self.point[1])],
            "seed": [float(self.seed[0]), float(self.seed[1])],
            "confidence": float(self.confidence),
            "status": self.status,
            "reason": self.reason,
            "candidate_margin": float(self.candidate_margin),
        }


def _seed_confidences(person: Person, clipped: list[Point]) -> list[float]:
    default_conf = float(person.confidence)
    ankles = person.ankles or []
    result: list[float] = []
    for sx, sy in clipped:
        best_score = default_conf
        best_dist = 1e9
        for ankle in ankles:
            if len(ankle) != 3:
                continue
            ax, ay, score = map(float, ankle)
            dist = float(np.hypot(ax - sx, ay - sy))
            if dist <= 12.0 and dist < best_dist:
                best_dist = dist
                best_score = float(np.clip(score, 0.0, 1.0))
        result.append(best_score)
    return result


def _skeleton_leg_incoherent(person: Person, idx: int, seed: Point, box_height: float) -> bool:
    """Return True when matched hip->knee->ankle hierarchy is anatomically inverted."""
    kpts = person.keypoints
    if not kpts or len(kpts) < 17:
        return False
    # Map foot index/position to closest COCO ankle (15=left, 16=right).
    d15 = np.hypot(kpts[15][0] - seed[0], kpts[15][1] - seed[1]) if kpts[15][1] > 0 else 1e9
    d16 = np.hypot(kpts[16][0] - seed[0], kpts[16][1] - seed[1]) if kpts[16][1] > 0 else 1e9
    if min(d15, d16) > 20.0:
        return False
    ankle_i, knee_i, hip_i = (15, 13, 11) if d15 <= d16 else (16, 14, 12)
    ax, ay, ac = kpts[ankle_i]
    kx, ky, kc = kpts[knee_i]
    hx, hy, hc = kpts[hip_i]
    tol = 0.05 * box_height
    if kc >= 0.35 and ky > 0 and ay > 0 and ay < ky - tol:
        return True
    if hc >= 0.35 and kc >= 0.35 and hy > 0 and ky > 0 and ky < hy - tol:
        return True
    return False


def _detect_glass_reflection_pair(gray: np.ndarray, person: Person,
                                  clipped: list[Point]) -> dict[int, str]:
    """Identify when one of two widely spaced feet is a train-glass reflection across a dark trench seam."""
    if len(clipped) != 2:
        return {}
    height, width = gray.shape[:2]
    x1, y1, x2, y2 = person.bbox
    box_height = max(y2 - y1, 1.0)
    (sx0, sy0), (sx1, sy1) = clipped
    span = abs(sx1 - sx0)
    if span < 0.14 * box_height:
        return {}

    kpts = person.keypoints
    if kpts is not None and len(kpts) >= 13 and kpts[11][1] > 0 and kpts[12][1] > 0:
        hip_y = int(round(0.5 * (kpts[11][1] + kpts[12][1])))
    else:
        hip_y = int(round(y1 + 0.45 * box_height))

    xa = int(max(0, np.floor(min(sx0, sx1))))
    xb = int(min(width - 1, np.ceil(max(sx0, sx1))))
    y_leg_lo = max(0, hip_y)
    y_leg_hi = min(height, int(round(max(sy0, sy1) + 0.12 * box_height)))
    y_below_lo = min(height - 1, int(round(max(sy0, sy1))))
    y_below_hi = min(height, int(round(max(sy0, sy1) + 0.28 * box_height)))
    if xb - xa < 6 or y_leg_hi - y_leg_lo < 8 or y_below_hi - y_below_lo < 6:
        return {}

    leg_strip = gray[y_leg_lo:y_leg_hi, xa:xb + 1]
    below_strip = gray[y_below_lo:y_below_hi, xa:xb + 1]
    dark_leg_cols = np.median(leg_strip, axis=0) < 55.0
    dark_below_cols = np.median(below_strip, axis=0) < 55.0
    continuous_seam = dark_leg_cols & dark_below_cols
    min_seam_cols = max(3, int(round(0.025 * box_height)))
    if int(np.sum(continuous_seam)) < min_seam_cols:
        return {}

    half_w = max(8, int(round(0.05 * box_height)))
    vert_scores: list[float] = []
    leg_vert_scores: list[float] = []
    out_lumas: list[float] = []
    seam_center_x = xa + float(np.mean(np.flatnonzero(continuous_seam)))

    for sx, sy in clipped:
        cx = int(np.clip(round(sx), 0, width - 1))
        ay = int(np.clip(round(sy), 0, height - 1))
        px0, px1 = max(0, cx - half_w), min(width, cx + half_w + 1)
        by_hi = min(height, max(ay + 12, int(round(ay + 0.35 * box_height))))
        below_patch = gray[ay:by_hi, px0:px1]
        leg_patch = gray[y_leg_lo:max(y_leg_lo + 4, ay), px0:px1]

        def _gx_gy_ratio(patch: np.ndarray) -> float:
            if patch.shape[0] < 3 or patch.shape[1] < 3:
                return 1.0
            gx = float(np.abs(cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)).mean())
            gy = float(np.abs(cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)).mean())
            return gx / (gy + 1e-3)

        vert_scores.append(_gx_gy_ratio(below_patch))
        leg_vert_scores.append(_gx_gy_ratio(leg_patch))

        out_span = max(12, int(round(0.11 * box_height)))
        if sx < seam_center_x:
            ox0, ox1 = max(0, cx - out_span), max(1, cx)
        else:
            ox0, ox1 = cx, min(width, cx + out_span)
        out_patch = gray[y_leg_lo:y_leg_hi, ox0:ox1]
        out_lumas.append(float(out_patch.mean()) if out_patch.size else 128.0)

    hi_idx = 0 if vert_scores[0] >= vert_scores[1] else 1
    lo_idx = 1 - hi_idx
    if (vert_scores[hi_idx] >= 2.0 and vert_scores[hi_idx] >= 1.8 * vert_scores[lo_idx]
            and (leg_vert_scores[hi_idx] >= 1.35 * leg_vert_scores[lo_idx]
                 or out_lumas[hi_idx] <= 0.85 * out_lumas[lo_idx])):
        return {hi_idx: "train_glass_reflection_across_seam"}
    # Seam clearly divides the two legs without confident asymmetry -> mark less-supported one ambiguous.
    weak_idx = 0 if out_lumas[0] <= out_lumas[1] else 1
    return {weak_idx: "seam_separated_ambiguous_foot"}


def assess_foot_evidence(image: np.ndarray, person: Person,
                         points: list[Point] | None = None, *, enabled: bool = True,
                         foreground_mask: np.ndarray | None = None,
                         roi_height_ratio: float = 0.18,
                         max_shift_ratio: float = 0.10,
                         gray: np.ndarray | None = None,
                         edges: np.ndarray | None = None) -> tuple[list[FootEvidence], str]:
    """Produce per-foot refined points with status, margin, and glass-reflection rejection."""
    if not isinstance(image, np.ndarray) or image.ndim not in {2, 3} or image.size == 0:
        raise ValueError("image must be a nonempty numpy array")
    if not 0 < roi_height_ratio <= 0.5 or not 0 <= max_shift_ratio <= 0.5:
        raise ValueError("invalid refinement bounds")
    height, width = image.shape[:2]
    if foreground_mask is not None and foreground_mask.shape[:2] != (height, width):
        raise ValueError("foreground_mask shape must match image")

    seeds = list(points) if points is not None else [person.bbox_bottom]
    clipped = [(float(np.clip(x, 0, width - 1)), float(np.clip(y, 0, height - 1)))
               for x, y in seeds]
    confidences = _seed_confidences(person, clipped)
    if not enabled:
        evidences = [
            FootEvidence(point=pt, seed=pt, confidence=conf, status="accepted",
                         reason="refinement_disabled", candidate_margin=1.0)
            for pt, conf in zip(clipped, confidences)
        ]
        return evidences, "refinement_disabled"

    if gray is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    elif gray.shape[:2] != (height, width):
        raise ValueError("gray shape must match image")
    if edges is None:
        edges = cv2.Canny(gray, 50, 150)
    elif edges.shape[:2] != (height, width):
        raise ValueError("edges shape must match image")
    reflection_flags = _detect_glass_reflection_pair(gray, person, clipped)

    x1, y1, x2, y2 = person.bbox
    box_height = max(y2 - y1, 1.0)
    radius = max(1, int(round(box_height * max_shift_ratio)))
    roi_height = max(2, int(round(box_height * roi_height_ratio)))
    sep_thresh = max(4.0, 0.03 * box_height)

    evidences: list[FootEvidence] = []
    changed = False
    for idx, ((seed_x, seed_y), seed_conf) in enumerate(zip(clipped, confidences)):
        flag_reason = reflection_flags.get(idx)
        if flag_reason == "train_glass_reflection_across_seam":
            evidences.append(FootEvidence(
                point=(seed_x, seed_y), seed=clipped[idx], confidence=seed_conf,
                status="rejected_reflection", reason=flag_reason, candidate_margin=0.0,
            ))
            continue

        left = max(0, int(np.floor(max(x1, seed_x - radius))))
        right = min(width - 1, int(np.ceil(min(x2, seed_x + radius))))
        top = max(0, int(np.floor(max(y1, seed_y - roi_height))))
        bottom = min(height - 1, int(np.ceil(min(y2, seed_y + radius))))
        if left > right or top > bottom:
            status = "ambiguous" if flag_reason else "no_support"
            reason = flag_reason or "no_local_edge_support"
            evidences.append(FootEvidence(
                point=(seed_x, seed_y), seed=clipped[idx], confidence=seed_conf,
                status=status, reason=reason, candidate_margin=0.0,
            ))
            continue

        skel_bad = _skeleton_leg_incoherent(person, idx, clipped[idx], box_height)
        support = edges[top:bottom + 1, left:right + 1] > 0
        if foreground_mask is not None:
            support &= foreground_mask[top:bottom + 1, left:right + 1] > 0
        ys, xs = np.nonzero(support)
        if len(xs) == 0:
            if flag_reason:
                status, reason = "ambiguous", flag_reason
            elif skel_bad:
                status, reason = "ambiguous", "incoherent_leg_skeleton_no_support"
            elif seed_conf < 0.35:
                status, reason = "ambiguous", "low_confidence_no_support"
            else:
                status, reason = "no_support", "no_local_edge_support"
            evidences.append(FootEvidence(
                point=(seed_x, seed_y), seed=clipped[idx], confidence=seed_conf,
                status=status, reason=reason, candidate_margin=0.0,
            ))
            continue

        # Prefer bottommost local edge closest to seed horizontally.
        cand_x = xs.astype(float) + left
        cand_y = ys.astype(float) + top
        scores = (cand_y - top) / max(bottom - top, 1) - 0.35 * np.abs(cand_x - seed_x) / max(radius, 1)
        order = np.argsort(-scores)
        best_idx = int(order[0])
        candidate = (float(cand_x[best_idx]), float(cand_y[best_idx]))

        dists = np.hypot(cand_x - candidate[0], cand_y - candidate[1])
        distant = np.flatnonzero(dists >= sep_thresh)
        if len(distant) == 0:
            margin = 1.0
        else:
            runner_score = float(np.max(scores[distant]))
            denom = abs(float(scores[best_idx])) + 1e-6
            margin = float(np.clip((float(scores[best_idx]) - runner_score) / denom, 0.0, 1.0))

        if np.hypot(candidate[0] - seed_x, candidate[1] - seed_y) <= roi_height + radius:
            chosen = candidate
            changed |= chosen != (seed_x, seed_y)
        else:
            chosen = (seed_x, seed_y)

        if flag_reason is not None:
            status, reason = "ambiguous", flag_reason
        elif skel_bad:
            status, reason = "ambiguous", "incoherent_leg_skeleton"
        elif seed_conf < 0.30 and len(xs) < 3 and margin < 0.10:
            status, reason = "ambiguous", "low_confidence_ambiguous_support"
        else:
            status, reason = "accepted", "refined_local_support"

        evidences.append(FootEvidence(
            point=chosen, seed=clipped[idx], confidence=seed_conf,
            status=status, reason=reason, candidate_margin=margin,
        ))

    wrapper_status = "refined_local_support" if changed else "refinement_no_support"
    return evidences, wrapper_status


def refine_foot_points(image: np.ndarray, person: Person,
                       points: list[Point] | None = None, *, enabled: bool = True,
                       foreground_mask: np.ndarray | None = None,
                       roi_height_ratio: float = 0.18,
                       max_shift_ratio: float = 0.10) -> tuple[list[Point], str]:
    """Move seeds only to nearby in-box edge/foreground support."""
    evidences, wrapper_status = assess_foot_evidence(
        image, person, points, enabled=enabled, foreground_mask=foreground_mask,
        roi_height_ratio=roi_height_ratio, max_shift_ratio=max_shift_ratio,
    )
    return [ev.point for ev in evidences], wrapper_status


class FootpointRefiner:
    def __init__(self, enabled: bool = False, **options):
        self.enabled = bool(enabled)
        self.options = options

    def refine(self, image: np.ndarray, person: Person, points: list[Point],
               foreground_mask: np.ndarray | None = None):
        return refine_foot_points(
            image, person, points, enabled=self.enabled,
            foreground_mask=foreground_mask, **self.options)


refine_contact_points = refine_foot_points
