"""Shared M4 person and frame decisions; ambiguous evidence never becomes SAFE."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import cv2
import numpy as np

from .boundaries import estimate_zone
from .detection import Person
from .geometry import HazardZone, PointAssessment, VALID_STATUSES


@dataclass
class Decision:
    bbox: tuple[float, float, float, float]
    confidence: float
    foot_points: list[tuple[float, float]]
    foot_source: str
    intrusion: bool | None
    object_id: str | None = None
    pose_outcome: str | None = None
    refinement_status: str | None = None
    status: str | None = None
    reason: str = "unclassified"
    reasons: list[str] = field(default_factory=list)
    point_assessments: list[PointAssessment] = field(default_factory=list)
    foot_evidence: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status is None:
            self.status = ("DANGER" if self.intrusion is True else
                           "SAFE" if self.intrusion is False else "UNKNOWN")
        if self.status not in VALID_STATUSES:
            raise ValueError("status must be SAFE, DANGER, or UNKNOWN")
        expected = True if self.status == "DANGER" else False if self.status == "SAFE" else None
        if self.intrusion is not expected:
            raise ValueError("intrusion must agree with status")

    def as_dict(self) -> dict:
        points = [list(p) for p in self.foot_points]
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "foot_points": points,  # backward-compatible name
            "observed_points": points,
            "footpoint_source": self.foot_source,
            "foot_source": self.foot_source,
            "intrusion": self.intrusion,
            "status": self.status,
            "reason": self.reason,
            "reasons": list(self.reasons),
            "point_assessments": [item.as_dict() for item in self.point_assessments],
            "object_id": self.object_id,
            "pose_outcome": self.pose_outcome,
            "refinement_status": self.refinement_status,
            "foot_evidence": list(self.foot_evidence),
        }


@dataclass(frozen=True)
class FrameDecision:
    status: str
    reason: str
    reasons: tuple[str, ...]
    observation_status: str
    calibration_status: str
    people: int
    danger: int
    unknown: int
    safe: int
    detector_partial: bool = False

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "reasons": list(self.reasons),
            "observation_status": self.observation_status,
            "calibration_status": self.calibration_status,
            "detector_partial": self.detector_partial,
            "people": self.people,
            "danger": self.danger,
            "unknown": self.unknown,
            "safe": self.safe,
        }


def aggregate_frame(decisions: list[Decision], *,
                    observation_status: str | None = None,
                    calibration_status: str = "manually_calibrated",
                    calibration_valid: bool = True,
                    detector_coverage: dict | None = None) -> FrameDecision:
    """Use one tri-state aggregation rule for CLI, Studio, and evaluation.

    Priority order: DANGER > person UNKNOWN > detector failure/partial >
    calibration failure > complete SAFE > confirmed-no-people SAFE >
    no-detections UNKNOWN.
    """
    allowed_observations = {
        "detections_complete", "detections_partial", "no_detections",
        "confirmed_no_people", "detector_failure",
    }
    if observation_status is None:
        observation_status = "detections_complete" if decisions else "no_detections"
    if observation_status not in allowed_observations:
        raise ValueError(f"Unknown observation_status: {observation_status}")
    if any(dec.status not in VALID_STATUSES for dec in decisions):
        raise ValueError("Every person decision must have SAFE, DANGER, or UNKNOWN status")

    detector_partial = bool(detector_coverage and detector_coverage.get("detector_partial"))
    if observation_status == "detections_partial":
        detector_partial = True
    elif detector_partial and observation_status != "detector_failure":
        observation_status = "detections_partial"

    counts = {name: sum(dec.status == name for dec in decisions)
              for name in ("DANGER", "UNKNOWN", "SAFE")}
    if counts["DANGER"]:
        status, reason, reasons = "DANGER", "person_danger", ("person_danger",)
    elif counts["UNKNOWN"]:
        status, reason, reasons = "UNKNOWN", "person_unknown", ("person_unknown",)
    elif observation_status == "detector_failure":
        status, reason, reasons = "UNKNOWN", "detector_failure", ("detector_failure",)
    elif observation_status == "detections_partial" or detector_partial:
        status, reason, reasons = "UNKNOWN", "detections_partial", ("detections_partial",)
    elif not calibration_valid:
        reason = ("invalid_calibration" if calibration_status == "invalid_calibration"
                  else "missing_calibration")
        status, reasons = "UNKNOWN", (reason,)
    elif decisions:
        status, reason, reasons = "SAFE", "all_observed_people_safe", ("all_observed_people_safe",)
    elif observation_status == "confirmed_no_people":
        status, reason, reasons = "SAFE", "confirmed_no_people", ("confirmed_no_people",)
    else:
        # ``detections_complete`` with zero decisions is still not proof that no
        # person exists; normalize the metadata to the explicit empty outcome.
        observation_status = "no_detections"
        status, reason, reasons = "UNKNOWN", "no_detections", ("no_detections",)

    return FrameDecision(
        status, reason, reasons, observation_status, calibration_status,
        len(decisions), counts["DANGER"], counts["UNKNOWN"], counts["SAFE"],
        detector_partial=detector_partial,
    )


def analyze(image: np.ndarray, people: list[Person], zone: HazardZone | None = None,
            foot_method: str = "bbox", auto_boundary: bool = False,
            boundary_method: str = "color_hough", paint_color: str = "both",
            tolerance_px: float = 0.0, keypoint_confidence: float = 0.35,
            pose_fallback: str = "bbox", refine_footpoints: bool = False,
            foreground_mask: np.ndarray | None = None,
            tolerance_normalized: float | None = None,
            image_edge_margin_px: float = 1.0) -> tuple[list[Decision], HazardZone | None, str]:
    if not isinstance(image, np.ndarray) or image.ndim < 2:
        raise ValueError("image must be an array with height and width")
    height, width = image.shape[:2]
    if width < 2 or height < 2:
        raise ValueError("image dimensions must be >= 2")
    if not math.isfinite(tolerance_px) or tolerance_px < 0:
        raise ValueError("tolerance_px must be finite and >= 0")
    if tolerance_normalized is not None:
        if tolerance_px != 0:
            raise ValueError("Specify tolerance_px or tolerance_normalized, not both")
        if not math.isfinite(tolerance_normalized) or tolerance_normalized < 0:
            raise ValueError("tolerance_normalized must be finite and >= 0")
        tolerance_px = tolerance_normalized * math.hypot(width - 1, height - 1)

    status = "manually_calibrated" if zone else "missing_calibration"
    if zone is None and auto_boundary:
        estimate = estimate_zone(image, method=boundary_method, color=paint_color)
        zone, status = estimate.zone, estimate.status
    if zone is not None:
        try:
            zone.polygon()
        except (TypeError, ValueError, cv2.error):
            zone, status = None, "invalid_calibration"

    cached_gray: np.ndarray | None = None
    cached_edges: np.ndarray | None = None
    if refine_footpoints and people:
        cached_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        cached_edges = cv2.Canny(cached_gray, 50, 150)

    decisions: list[Decision] = []
    for person in people:
        feet, source = person.foot_points(foot_method, keypoint_confidence, pose_fallback)
        refinement_status = None
        evidences = []
        if refine_footpoints and feet:
            from .foot_refinement import assess_foot_evidence
            evidences, refinement_status = assess_foot_evidence(
                image, person, feet, foreground_mask=foreground_mask,
                gray=cached_gray, edges=cached_edges)
            feet = [ev.point for ev in evidences if ev.status != "rejected_reflection"]
        rejected_count = sum(1 for ev in evidences if ev.status == "rejected_reflection")
        ambiguous_count = sum(1 for ev in evidences if ev.status == "ambiguous")
        pose_outcome = ("bbox_fallback" if source == "bbox_fallback" else
                        "abstain" if source == "abstain" else
                        "ankle" if source in {"one_ankle", "two_ankles"} else None)

        assessments: list[PointAssessment] = []
        if zone is None:
            decision_status = "UNKNOWN"
            reason = "invalid_calibration" if status == "invalid_calibration" else "missing_calibration"
            reasons = [reason]
        elif not feet:
            decision_status = "UNKNOWN"
            if rejected_count > 0:
                reason = "all_footpoints_rejected_as_reflection"
            else:
                reason = ("missing_or_low_confidence_keypoints" if foot_method in {"pose", "pose_ankle"}
                          else "missing_footpoints")
            reasons = [reason]
        else:
            assessments = [zone.assess_point(
                point, (height, width), tolerance_px=tolerance_px,
                image_edge_margin_px=image_edge_margin_px,
            ) for point in feet]
            if any(item.status == "DANGER" for item in assessments):
                decision_status = "DANGER"
                selected = [item for item in assessments if item.status == "DANGER"]
                reason = selected[0].reason
                reasons = _unique(code for item in selected for code in item.reasons)
            elif any(item.status == "UNKNOWN" for item in assessments):
                decision_status = "UNKNOWN"
                selected = [item for item in assessments if item.status == "UNKNOWN"]
                reason = selected[0].reason
                reasons = _unique(code for item in selected for code in item.reasons)
            elif ambiguous_count > 0 or rejected_count > 0:
                decision_status = "UNKNOWN"
                reason = "ambiguous_footpoint_evidence"
                reasons = _unique([reason, *(ev.reason for ev in evidences if ev.status in {"ambiguous", "rejected_reflection"})])
            else:
                decision_status = "SAFE"
                selected = [item for item in assessments if item.status == "SAFE"]
                reason = selected[0].reason
                reasons = _unique(code for item in selected for code in item.reasons)

        intrusion = True if decision_status == "DANGER" else False if decision_status == "SAFE" else None
        decisions.append(Decision(
            bbox=person.bbox, confidence=person.confidence,
            foot_points=feet, foot_source=source, intrusion=intrusion,
            object_id=person.object_id, pose_outcome=pose_outcome,
            refinement_status=refinement_status, status=decision_status,
            reason=reason, reasons=reasons, point_assessments=assessments,
            foot_evidence=[ev.as_dict() for ev in evidences],
        ))
    return decisions, zone, status


def _unique(values) -> list[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def draw(image: np.ndarray, decisions: list[Decision], zone: HazardZone | None,
         status: str = "") -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]
    if zone is not None:
        try:
            poly = zone.corridor_polygon(w, h).astype(np.int32)
        except Exception:
            poly = zone.polygon().astype(np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [poly], (0, 160, 240))
        cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, canvas)

        clipped_l, clipped_e = zone.clipped_boundary_segments(w, h)
        pL_a = (int(round(clipped_l.a[0])), int(round(clipped_l.a[1])))
        pL_b = (int(round(clipped_l.b[0])), int(round(clipped_l.b[1])))
        pE_a = (int(round(clipped_e.a[0])), int(round(clipped_e.a[1])))
        pE_b = (int(round(clipped_e.b[0])), int(round(clipped_e.b[1])))
        cv2.line(canvas, pL_a, pL_b, (0, 165, 255), 3)
        cv2.line(canvas, pE_a, pE_b, (255, 180, 0), 3)
    for dec in decisions:
        color = ((35, 55, 235) if dec.status == "DANGER" else
                 (50, 180, 65) if dec.status == "SAFE" else (180, 180, 180))
        x1, y1, x2, y2 = map(int, dec.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        for foot in dec.foot_points:
            cv2.circle(canvas, tuple(map(int, foot)), 5, color, -1)
        cv2.putText(canvas, dec.status, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(canvas, status[:90], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)
    return canvas
