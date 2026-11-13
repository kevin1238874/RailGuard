"""Optional local contact-point refinement with conservative bounds."""
from __future__ import annotations

import cv2
import numpy as np

from .detection import Person, Point


def refine_foot_points(image: np.ndarray, person: Person,
                       points: list[Point] | None = None, *, enabled: bool = True,
                       foreground_mask: np.ndarray | None = None,
                       roi_height_ratio: float = 0.18,
                       max_shift_ratio: float = 0.10) -> tuple[list[Point], str]:
    """Move seeds only to nearby in-box edge/foreground support."""
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
    if not enabled:
        return clipped, "refinement_disabled"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    x1, y1, x2, y2 = person.bbox
    box_height = max(y2 - y1, 1.0)
    radius = max(1, int(round(box_height * max_shift_ratio)))
    roi_height = max(2, int(round(box_height * roi_height_ratio)))
    refined: list[Point] = []
    changed = False
    for seed_x, seed_y in clipped:
        left = max(0, int(np.floor(max(x1, seed_x - radius))))
        right = min(width - 1, int(np.ceil(min(x2, seed_x + radius))))
        top = max(0, int(np.floor(max(y1, seed_y - roi_height))))
        bottom = min(height - 1, int(np.ceil(min(y2, seed_y + radius))))
        if left > right or top > bottom:
            refined.append((seed_x, seed_y))
            continue
        support = edges[top:bottom + 1, left:right + 1] > 0
        if foreground_mask is not None:
            support &= foreground_mask[top:bottom + 1, left:right + 1] > 0
        ys, xs = np.nonzero(support)
        if len(xs) == 0:
            refined.append((seed_x, seed_y))
            continue
        max_y = int(ys.max())
        candidates = np.flatnonzero(ys == max_y)
        best = candidates[int(np.argmin(np.abs((xs[candidates] + left) - seed_x)))]
        candidate = (float(xs[best] + left), float(ys[best] + top))
        if np.hypot(candidate[0] - seed_x, candidate[1] - seed_y) <= roi_height + radius:
            refined.append(candidate)
            changed |= candidate != (seed_x, seed_y)
        else:
            refined.append((seed_x, seed_y))
    return refined, "refined_local_support" if changed else "refinement_no_support"


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
