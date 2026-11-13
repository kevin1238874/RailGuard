"""Testable UI state, undo/reset, debug-stage builders, and export helpers for RailGuard Studio."""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .boundaries import estimate_zone, paint_mask
from .classic_detectors import HOGSVMDetector, MOG2BlobDetector
from .detection import YOLODetector
from .foot_refinement import refine_foot_points
from .geometry import HazardZone, Segment
from .pipeline import aggregate_frame, analyze, draw
from .preprocessing import apply_pipeline, compute_quality_flags
from .temporal import temporal_median_background

DEBUG_TABS = (
    "Input",
    "Preprocessing",
    "Masks",
    "Canny-Hough",
    "RANSAC",
    "Final Zone",
    "People & Feet",
    "Results",
)


@dataclass
class CalibrationSessionState:
    """Tracks interactive L/E points, undo history, and confirmation state."""

    points: list[list[float] | None] = field(default_factory=lambda: [None, None, None, None])
    history: list[list[list[float] | None]] = field(default_factory=list)
    confirmed: bool = False
    zone_source: str = "manual"
    generation: int = 0

    def update_points(self, new_points: list[list[float] | None], *, source: str = "manually edited") -> bool:
        if new_points == self.points:
            return False
        self.history.append([list(p) if p is not None else None for p in self.points])
        self.points = [list(p) if p is not None else None for p in new_points]
        self.confirmed = False
        self.zone_source = source
        self.generation += 1
        return True

    def undo(self) -> bool:
        if not self.history:
            return False
        self.points = self.history.pop()
        self.confirmed = False
        self.zone_source = "undo previous edit"
        self.generation += 1
        return True

    def reset(self) -> None:
        self.history.append([list(p) if p is not None else None for p in self.points])
        self.points = [None, None, None, None]
        self.confirmed = False
        self.zone_source = "manual"
        self.generation += 1


def available_temporal_controls(num_frames: int) -> dict[str, bool]:
    """Disable sequence-only methods (B2 MOG2 and TemporalMedian) on single images."""
    multi = int(num_frames) >= 2
    return {"mog2_enabled": multi, "temporal_median_enabled": multi}


def can_run_inference(zone: HazardZone | None, confirmed: bool) -> tuple[bool, str]:
    """Block inference unless calibration geometry is valid AND visually confirmed."""
    if zone is None:
        return False, "Calibration geometry is incomplete or invalid."
    if not confirmed:
        return False, "Confirm the calibrated L/E geometry before running inference."
    return True, "ok"


def build_debug_stages(
    frames: list[np.ndarray],
    *,
    boundary_preproc: str = "raw",
    detector_preproc: str = "raw",
    use_temporal_median: bool = False,
    boundary_method: str = "hybrid_region_evidence",
    paint_color: str = "both",
    stripe_edge_mode: str = "safe_side_edge",
) -> dict[str, Any]:
    """Run M1 + M2 and collect available debug stage images and diagnostics."""
    if not frames:
        raise ValueError("at least one frame is required")
    controls = available_temporal_controls(len(frames))
    if use_temporal_median and not controls["temporal_median_enabled"]:
        raise ValueError("TemporalMedian requires at least 2 frames from the same camera")

    current = frames[-1]
    base_for_m2 = (
        temporal_median_background(frames, ["cam0"] * len(frames))
        if use_temporal_median
        else current
    )
    m2_img = apply_pipeline(base_for_m2, [boundary_preproc])
    m3_img = apply_pipeline(current, [detector_preproc])
    quality = compute_quality_flags(current)
    suggestion = estimate_zone(
        m2_img,
        method=boundary_method,
        color=paint_color,
        stripe_convention=stripe_edge_mode,
    )
    mask = paint_mask(m2_img, paint_color)
    gray = cv2.cvtColor(m2_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    hough_vis = m2_img.copy()
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=40, minLineLength=max(20, m2_img.shape[1] // 15), maxLineGap=15)
    if segs is not None:
        for seg in segs[:, 0]:
            cv2.line(hough_vis, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])), (0, 255, 0), 2)

    ransac_vis = m2_img.copy()
    for idx, cand in enumerate((suggestion.candidates or [])[:3]):
        s_pts = cand.get("safety_line", {}).get("pixels")
        e_pts = cand.get("platform_edge", {}).get("pixels")
        if s_pts and e_pts:
            thick = 3 if idx == 0 else 1
            cv2.line(ransac_vis, tuple(map(int, s_pts[0])), tuple(map(int, s_pts[1])), (0, 180, 255), thick)
            cv2.line(ransac_vis, tuple(map(int, e_pts[0])), tuple(map(int, e_pts[1])), (255, 160, 40), thick)

    stages: dict[str, Any] = {
        "Input": current,
        "Preprocessing": {"m2_image": m2_img, "m3_image": m3_img, "quality": quality},
        "Masks": mask,
        "Canny-Hough": cv2.cvtColor(hough_vis, cv2.COLOR_BGR2RGB),
        "RANSAC": cv2.cvtColor(ransac_vis, cv2.COLOR_BGR2RGB),
        "suggestion": suggestion,
    }
    return stages


def candidate_to_zone(candidate: dict[str, Any]) -> HazardZone | None:
    """Convert a LineResult candidate dictionary into a HazardZone."""
    s_pts = candidate.get("safety_line", {}).get("pixels")
    e_pts = candidate.get("platform_edge", {}).get("pixels")
    if not s_pts or not e_pts or len(s_pts) != 2 or len(e_pts) != 2:
        return None
    return HazardZone(
        Segment(tuple(map(float, s_pts[0])), tuple(map(float, s_pts[1]))),
        Segment(tuple(map(float, e_pts[0])), tuple(map(float, e_pts[1]))),
    )


def run_ui_inference(
    frames: list[np.ndarray],
    zone: HazardZone | None,
    confirmed: bool,
    *,
    detector_kind: str = "B1 HOG+SVM",
    detector_preproc: str = "raw",
    weights: str | None = None,
    confidence: float = 0.25,
    imgsz: int = 640,
    pose_fallback: str = "bbox",
    refine_feet: bool = False,
) -> dict[str, Any]:
    """Execute M3 + M4 inference with calibration guard and explicit error handling."""
    ok, reason = can_run_inference(zone, confirmed)
    if not ok:
        raise ValueError(reason)
    controls = available_temporal_controls(len(frames))
    if detector_kind == "B2 MOG2" and not controls["mog2_enabled"]:
        raise ValueError("B2 MOG2 is disabled for single-image input; upload >=2 frames.")

    current = frames[-1]
    proc_frames = [apply_pipeline(f, [detector_preproc]) for f in frames]
    target = proc_frames[-1]

    if detector_kind == "B1 HOG+SVM":
        det = HOGSVMDetector()
        people = det.predict(target)
        foot_method = "bbox"
    elif detector_kind == "B2 MOG2":
        det = MOG2BlobDetector(warmup=min(3, max(1, len(proc_frames) - 1)))
        people = det.predict_sequence(proc_frames, ["cam0"] * len(proc_frames))[-1]
        foot_method = "bbox"
    elif detector_kind == "B3 YOLO det":
        det = YOLODetector("bbox", weights=weights or "yolo11n.pt", confidence=confidence, imgsz=imgsz)
        people = det.predict(target)
        foot_method = "bbox"
    elif detector_kind == "A YOLO pose":
        det = YOLODetector("pose", weights=weights or "yolo11n-pose.pt", confidence=confidence, imgsz=imgsz)
        people = det.predict(target)
        foot_method = "pose"
    else:
        raise ValueError(f"Unknown detector_kind: {detector_kind!r}")

    if refine_feet and people:
        refined = []
        for p in people:
            pts = list(p.ankles) if (foot_method == "pose" and p.ankles) else [p.bbox_bottom]
            new_pts, _ = refine_foot_points(target, p, pts, enabled=True)
            refined.append(p)
            if new_pts:
                p.metadata["refined_foot_points"] = [list(pt) for pt in new_pts]
        people = refined

    results, zone_used, cal_status = analyze(
        target,
        people,
        zone,
        foot_method=foot_method,
        pose_fallback=pose_fallback,
    )
    frame = aggregate_frame(
        results,
        observation_status="no_detections" if not people else "detections_complete",
        calibration_status=cal_status,
        calibration_valid=zone_used is not None,
    )
    visual = draw(target, results, zone_used, f"frame={frame.status}; method={detector_kind}")
    png_ok, png = cv2.imencode(".png", visual)
    if not png_ok:
        raise RuntimeError("Cannot encode inference output PNG")
    return {
        "image_png": png.tobytes(),
        "results": [r.as_dict() for r in results],
        "frame": frame.as_dict(),
        "params": {
            "detector": detector_kind,
            "detector_preproc": detector_preproc,
            "confidence": confidence,
            "imgsz": imgsz,
            "pose_fallback": pose_fallback,
            "refine_feet": refine_feet,
        },
    }


def results_to_csv(results: list[dict[str, Any]]) -> str:
    """Serialize per-person M4 results to CSV."""
    buf = io.StringIO()
    fields = ["person_id", "status", "intrusion", "foot_source", "reason", "confidence", "bbox", "foot_points"]
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for row in results:
        writer.writerow(
            {
                "person_id": row.get("person_id", ""),
                "status": row.get("status", "UNKNOWN"),
                "intrusion": row.get("intrusion"),
                "foot_source": row.get("foot_source", ""),
                "reason": row.get("reason", ""),
                "confidence": row.get("confidence", ""),
                "bbox": json.dumps(row.get("bbox")),
                "foot_points": json.dumps(row.get("foot_points")),
            }
        )
    return buf.getvalue()
