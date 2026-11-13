"""Temporal M1 utilities for ordered frames from one stationary camera."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _validate_frames(frames: Sequence[np.ndarray], camera_ids: Sequence[str]) -> list[np.ndarray]:
    if isinstance(frames, np.ndarray):
        raise ValueError("temporal median requires an ordered sequence, not a single image")
    frames = list(frames)
    camera_ids = list(camera_ids)
    if len(frames) < 2:
        raise ValueError("temporal median requires at least two consecutive frames")
    if len(camera_ids) != len(frames):
        raise ValueError("one camera_id is required for every frame")
    if len(set(camera_ids)) != 1:
        raise ValueError("temporal median cannot mix camera_id values")
    first = frames[0]
    if not isinstance(first, np.ndarray) or first.size == 0:
        raise ValueError("frames must be non-empty numpy arrays")
    for frame in frames:
        if not isinstance(frame, np.ndarray) or frame.shape != first.shape or frame.dtype != first.dtype:
            raise ValueError("all temporal frames must have the same shape and dtype")
    return frames


def temporal_median_background(frames: Sequence[np.ndarray], camera_ids: Sequence[str],
                               *, window_size: int | None = None,
                               sampling: int = 1) -> np.ndarray:
    """Return a same-shape/dtype median background from one camera's frames.

    Frames must already be in chronological order. ``sampling`` takes every
    Nth frame, and ``window_size`` limits computation to the most recent
    sampled frames. At least two selected frames are required. The pixel-wise
    median resists short-lived foreground objects but cannot guarantee their
    removal if foreground occupies half or more of the selected observations.
    """
    frames = _validate_frames(frames, camera_ids)
    if isinstance(sampling, bool) or int(sampling) != sampling or int(sampling) <= 0:
        raise ValueError("sampling must be a positive integer")
    selected = frames[::int(sampling)]
    if window_size is not None:
        if isinstance(window_size, bool) or int(window_size) != window_size or int(window_size) < 2:
            raise ValueError("window_size must be at least 2")
        selected = selected[-int(window_size):]
    if len(selected) < 2:
        raise ValueError("sampling/window selection must retain at least two frames")
    source_dtype = selected[0].dtype
    median = np.median(np.stack(selected, axis=0), axis=0)
    if np.issubdtype(source_dtype, np.integer):
        bounds = np.iinfo(source_dtype)
        median = np.clip(np.rint(median), bounds.min, bounds.max)
    return median.astype(source_dtype)


class TemporalMedianBackground:
    """Configured temporal-median estimator for ordered same-camera frames."""

    def __init__(self, window_size: int | None = None, sampling: int = 1):
        self.window_size = window_size
        self.sampling = sampling

    def compute(self, frames: Sequence[np.ndarray], camera_ids: Sequence[str]) -> np.ndarray:
        return temporal_median_background(
            frames, camera_ids, window_size=self.window_size, sampling=self.sampling)

    __call__ = compute


__all__ = ["TemporalMedianBackground", "temporal_median_background"]
