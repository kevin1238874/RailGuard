"""Orientation-independent 2-D line fitting and Hough-segment clustering.

The RANSAC implementation in this module is deliberately small and explicit:
each hypothesis is made from two points, residuals are perpendicular distances,
the largest consensus wins, and the final model is re-fitted on all inliers.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .geometry import Segment


@dataclass(frozen=True)
class LineModel:
    """Infinite 2-D line represented by a point and a unit direction."""

    point: np.ndarray
    direction: np.ndarray

    def residuals(self, points: np.ndarray) -> np.ndarray:
        values = _points(points)
        normal = np.array([-self.direction[1], self.direction[0]], dtype=float)
        return np.abs((values - self.point) @ normal)


@dataclass(frozen=True)
class RansacLineResult:
    model: LineModel
    inlier_mask: np.ndarray
    residuals: np.ndarray
    trials: int
    inlier_ratio: float


@dataclass(frozen=True)
class LineCluster:
    segments: tuple[Segment, ...]
    model: LineModel
    support_length: float


def _points(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
        raise ValueError("points must be an Nx2 array with N >= 2")
    if not np.isfinite(values).all():
        raise ValueError("points must contain only finite coordinates")
    return values


def fit_line(points: np.ndarray | Sequence[Sequence[float]],
             weights: np.ndarray | Sequence[float] | None = None) -> LineModel:
    """Weighted least-squares orthogonal line fit (ưu tiên các điểm ở gần camera/góc nhìn lớn)."""
    values = _points(points)
    if weights is not None:
        w = np.asarray(weights, dtype=float).flatten()
        if len(w) != len(values) or (w <= 0).all():
            w = np.ones(len(values), dtype=float)
        else:
            w = np.clip(w, 1e-4, None)
            w = w / w.sum()
        point = np.average(values, axis=0, weights=w)
        centered = (values - point) * np.sqrt(w[:, np.newaxis])
    else:
        point = values.mean(axis=0)
        centered = values - point

    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    if not len(singular) or singular[0] <= np.finfo(float).eps:
        raise ValueError("cannot fit a line to coincident points")
    direction = vectors[0].astype(float)
    if direction[0] < -1e-12 or (abs(direction[0]) <= 1e-12 and direction[1] < 0):
        direction *= -1
    return LineModel(point=point, direction=direction / np.linalg.norm(direction))


def fit_line_ransac(
    points: np.ndarray | Sequence[Sequence[float]],
    *,
    max_trials: int = 200,
    residual_threshold: float = 2.0,
    min_inlier_ratio: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> RansacLineResult:
    """Fit a line with true two-point RANSAC and return its consensus mask.

    Consensus is ranked by inlier count, then by the median inlier residual.
    A deterministic integer ``random_state`` produces reproducible hypotheses.
    """
    values = _points(points)
    if isinstance(max_trials, bool) or int(max_trials) <= 0:
        raise ValueError("max_trials must be a positive integer")
    if not np.isfinite(residual_threshold) or residual_threshold <= 0:
        raise ValueError("residual_threshold must be positive and finite")
    if not np.isfinite(min_inlier_ratio) or not 0 < min_inlier_ratio <= 1:
        raise ValueError("min_inlier_ratio must be in (0, 1]")
    rng = (random_state if isinstance(random_state, np.random.Generator)
           else np.random.default_rng(random_state))
    required = max(2, int(math.ceil(len(values) * float(min_inlier_ratio))))
    best_mask: np.ndarray | None = None
    best_key: tuple[int, float] | None = None
    trials_run = 0
    for _ in range(int(max_trials)):
        trials_run += 1
        i, j = rng.choice(len(values), size=2, replace=False)
        vector = values[j] - values[i]
        norm = float(np.linalg.norm(vector))
        if norm <= np.finfo(float).eps:
            continue
        direction = vector / norm
        normal = np.array([-direction[1], direction[0]])
        residuals = np.abs((values - values[i]) @ normal)
        mask = residuals <= float(residual_threshold)
        count = int(mask.sum())
        if count < required:
            continue
        median = float(np.median(residuals[mask]))
        key = (count, -median)
        if best_key is None or key > best_key:
            best_key, best_mask = key, mask
    if best_mask is None:
        raise ValueError(
            f"RANSAC found no consensus meeting min_inlier_ratio={min_inlier_ratio}"
        )
    model = fit_line(values[best_mask])
    residuals = model.residuals(values)
    # Recompute once after the consensus refit, then refit the stable set.
    refined = residuals <= float(residual_threshold)
    if int(refined.sum()) >= required:
        best_mask = refined
        model = fit_line(values[best_mask])
        residuals = model.residuals(values)
    return RansacLineResult(
        model=model,
        inlier_mask=best_mask.copy(),
        residuals=residuals,
        trials=trials_run,
        inlier_ratio=float(best_mask.mean()),
    )


# A concise alias is useful in notebooks while keeping the explicit public name.
ransac_line = fit_line_ransac


def _segment_direction(segment: Segment) -> np.ndarray:
    direction = np.asarray(segment.b, dtype=float) - np.asarray(segment.a, dtype=float)
    length = float(np.linalg.norm(direction))
    if length <= np.finfo(float).eps:
        raise ValueError("segments must have distinct endpoints")
    direction /= length
    if direction[0] < -1e-12 or (abs(direction[0]) <= 1e-12 and direction[1] < 0):
        direction *= -1
    return direction


def line_clustering(
    segments: Sequence[Segment],
    *,
    angle_threshold_deg: float = 10.0,
    distance_threshold: float = 12.0,
    min_cluster_size: int = 1,
) -> list[LineCluster]:
    """Group locally co-oriented, near-collinear segments.

    Pairwise compatibility is symmetric and tolerates modest perspective
    convergence. It does not impose one global parallel direction on clusters.
    """
    items = tuple(segments)
    if not items:
        return []
    if not 0 <= angle_threshold_deg < 90:
        raise ValueError("angle_threshold_deg must be in [0, 90)")
    if distance_threshold <= 0:
        raise ValueError("distance_threshold must be positive")
    if min_cluster_size <= 0:
        raise ValueError("min_cluster_size must be positive")
    directions = np.asarray([_segment_direction(item) for item in items], dtype=float)
    a_pts = np.asarray([item.a for item in items], dtype=float)
    b_pts = np.asarray([item.b for item in items], dtype=float)
    centers = 0.5 * (a_pts + b_pts)
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    n_items = len(items)
    parent = list(range(n_items))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra

    cosine = math.cos(math.radians(float(angle_threshold_deg)))
    angle_ok = np.abs(directions @ directions.T) >= cosine
    max_long_gap = max(distance_threshold * 10.0, 160.0)
    for i in range(n_items - 1):
        cand_js = np.flatnonzero(angle_ok[i, i + 1:]) + (i + 1)
        if cand_js.size == 0:
            continue
        deltas = centers[cand_js] - centers[i]
        d_ni = np.abs(deltas @ normals[i])
        d_nj = np.abs(np.sum(deltas * normals[cand_js], axis=1))
        perp_ok = np.maximum(d_ni, d_nj) <= distance_threshold
        cand_js = cand_js[perp_ok]
        if cand_js.size == 0:
            continue
        di = directions[i]
        pi_a = float(a_pts[i] @ di)
        pi_b = float(b_pts[i] @ di)
        min_i, max_i = (pi_a, pi_b) if pi_a <= pi_b else (pi_b, pi_a)
        pj_a = a_pts[cand_js] @ di
        pj_b = b_pts[cand_js] @ di
        min_j = np.minimum(pj_a, pj_b)
        max_j = np.maximum(pj_a, pj_b)
        long_gaps = np.maximum(0.0, np.maximum(min_i, min_j) - np.minimum(max_i, max_j))
        for j in cand_js[long_gaps <= max_long_gap]:
            union(i, int(j))
    groups: dict[int, list[Segment]] = {}
    for index, segment in enumerate(items):
        groups.setdefault(root(index), []).append(segment)
    clusters = []
    for members in groups.values():
        if len(members) < min_cluster_size:
            continue
        endpoints = np.asarray([point for segment in members
                                for point in (segment.a, segment.b)], dtype=float)
        clusters.append(LineCluster(
            segments=tuple(members),
            model=fit_line(endpoints),
            support_length=float(sum(segment.length() for segment in members)),
        ))
    return sorted(clusters, key=lambda cluster: cluster.support_length, reverse=True)

