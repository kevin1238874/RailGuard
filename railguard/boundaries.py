"""Classical CV to estimate safety paint and platform-edge lines.

Heuristic detection: calibration is recommended for reliable comparisons.
"""
from dataclasses import dataclass
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
        l = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)).apply(l)
        image = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (16, 55, 100), (42, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 165), (179, 68, 255))
    if color == "yellow":
        m = yellow
    elif color == "white":
        m = white
    elif color == "both":
        m = cv2.bitwise_or(yellow, white)
    else:
        raise ValueError("color must be yellow, white or both")
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    return m


def _candidate_segments(image: np.ndarray, mode: str, color: str, clahe: bool) -> list[Segment]:
    h, w = image.shape[:2]
    if mode == "color":
        edges = cv2.Canny(paint_mask(image, color, clahe), 35, 110)
    elif mode == "canny":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 65, 145)
    else:
        raise ValueError("mode must be color or canny")
    raw = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=max(28, w//45),
                          minLineLength=max(35,w//12), maxLineGap=max(12,w//35))
    if raw is None:
        return []
    out = []
    for row in raw[:,0,:]:
        x1,y1,x2,y2 = map(float,row)
        seg = Segment((x1,y1),(x2,y2))
        if abs(x2-x1) < w*.1 or seg.length() < w*.09:
            continue
        if (y1+y2)/2 < .30*h:
            continue
        if abs(y2-y1) > abs(x2-x1)*1.2:
            continue
        out.append(seg)
    return out


def _fit_segments(candidates: list[Segment], width: int, threshold: float = 12.0) -> list[Segment]:
    """Deterministic RANSAC-like candidate consensus using segment-midpoint inliers."""
    if not candidates:
        return []
    def slope(s):
        return (s.b[1]-s.a[1])/(s.b[0]-s.a[0])
    scored = []
    for seed in candidates:
        k = slope(seed)
        y0 = seed.y_at(width/2)
        support = [s for s in candidates if abs(slope(s)-k) <= .25 and
                   abs(s.y_at(width/2)-y0) <= threshold]
        score = sum(s.length() for s in support)
        scored.append((score, seed, support))
    scored.sort(key=lambda s:s[0], reverse=True)
    selected = []
    for _, seed, support in scored:
        if any(abs(seed.y_at(width/2)-s.y_at(width/2)) < 28 for s in selected):
            continue
        xs, ys = [], []
        for seg in support:
            # Longer segments contribute endpoints, consensus removes fragmented lines.
            xs.extend([seg.a[0], seg.b[0]])
            ys.extend([seg.a[1], seg.b[1]])
        if len(set(xs)) < 2:
            continue
        a,b = np.polyfit(xs, ys, 1)
        selected.append(Segment((0.,float(b)),(float(width-1),float(a*(width-1)+b))))
        if len(selected) >= 6:
            break
    return selected


def estimate_zone(image: np.ndarray, method: str = "color_hough", color: str = "both",
                  clahe: bool = False, min_gap_frac: float = .015, max_gap_frac: float = .26) -> LineResult:
    """Estimate two near-parallel lines; no semantic guarantee which is edge.

    FAIL-CLOSED: returns zone=None instead of inventing boundaries with insufficient evidence.
    For images where the true platform geometry is ambiguous, prefer explicit per-scene calibration.
    """
    if method not in {"color_hough", "canny_hough", "color_ransac", "hybrid"}:
        raise ValueError(f"Unknown method: {method}")
    h, w = image.shape[:2]
    seeds = _candidate_segments(image, "canny" if method == "canny_hough" else "color", color, clahe)
    if method == "hybrid":
        seeds += _candidate_segments(image, "canny", color, clahe)
    lines = _fit_segments(seeds, w, 15 if method in ("color_ransac","hybrid") else 9)
    options=[]
    for i, a in enumerate(lines):
        for b in lines[i+1:]:
            ya,yb = a.y_at(w*.5), b.y_at(w*.5)
            gap=abs(ya-yb)
            if not (h*min_gap_frac <= gap <= h*max_gap_frac):
                continue
            ka = (a.b[1]-a.a[1]) / (a.b[0]-a.a[0])
            kb = (b.b[1]-b.a[1]) / (b.b[0]-b.a[0])
            if abs(ka-kb) > .22:
                continue
            # Choose lower-half paired lines; cannot infer painted vs edge semantically.
            options.append(((ya+yb)/2 - gap*.02, a, b))
    if not options:
        return LineResult(None,"uncertain_no_pair",{"hough_segments":len(seeds),"lines":len(lines)})
    _, a, b=max(options,key=lambda v:v[0])
    # Under fixed-camera assumption, closer line is usually lower in the image.
    upper, lower=sorted((a,b),key=lambda s:s.y_at(w/2))
    return LineResult(HazardZone(upper,lower), "estimated_review_required",
                      {"hough_segments":len(seeds),"lines":len(lines),"method":method})
