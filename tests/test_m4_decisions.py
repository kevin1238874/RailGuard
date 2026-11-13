import numpy as np

from railguard.detection import Person
from railguard.geometry import HazardZone, Segment
from railguard.pipeline import Decision, aggregate_frame, analyze


def _horizontal(l_y=40, e_y=70):
    return HazardZone(
        Segment((10, l_y), (90, l_y)),
        Segment((10, e_y), (90, e_y)),
    )


def _decision(status):
    return Decision(
        (10, 10, 20, 20), 1.0, [(15, 20)], "bbox_bottom_center",
        True if status == "DANGER" else False if status == "SAFE" else None,
        status=status, reason=status.lower(), reasons=[status.lower()],
    )


def test_opposite_facing_cameras_use_calibrated_line_direction_not_y():
    down_unsafe = _horizontal(40, 70)
    up_unsafe = _horizontal(70, 40)

    assert down_unsafe.assess_point((50, 80), (100, 100)).reason == "beyond_platform_edge"
    assert down_unsafe.assess_point((50, 20), (100, 100)).status == "SAFE"
    assert up_unsafe.assess_point((50, 30), (100, 100)).reason == "beyond_platform_edge"
    assert up_unsafe.assess_point((50, 90), (100, 100)).status == "SAFE"


def test_vertical_and_steep_diagonal_lines_have_orientation_independent_rules():
    vertical = HazardZone(Segment((30, 10), (30, 90)), Segment((70, 10), (70, 90)))
    diagonal = HazardZone(Segment((20, 5), (25, 95)), Segment((60, 5), (65, 95)))

    assert vertical.assess_point((50, 50), (100, 100)).reason == "inside_hazard_zone"
    assert vertical.assess_point((80, 50), (100, 100)).reason == "beyond_platform_edge"
    assert vertical.assess_point((15, 50), (100, 100)).status == "SAFE"
    assert diagonal.assess_point((45, 50), (100, 100)).status == "DANGER"
    assert diagonal.assess_point((75, 50), (100, 100)).reason == "beyond_platform_edge"


def test_beyond_e_is_danger_even_though_outside_finite_z_polygon():
    image = np.zeros((100, 100, 3), np.uint8)
    decisions, _, _ = analyze(image, [Person((40, 10, 60, 80))], _horizontal())
    decision = decisions[0]

    assert not _horizontal().contains((50, 80))
    assert decision.status == "DANGER" and decision.intrusion is True
    assert decision.reason == "beyond_platform_edge"
    payload = decision.as_dict()
    assert payload["observed_points"] == [[50.0, 80]]
    assert payload["point_assessments"][0]["signed_distance_to_L"] > 0
    assert payload["point_assessments"][0]["signed_distance_to_E"] > 0


def test_on_l_with_pixel_or_normalized_tolerance_is_danger():
    zone = HazardZone(Segment((30, 10), (30, 90)), Segment((70, 10), (70, 90)))
    assert zone.assess_point((29.5, 50), (100, 100), tolerance_px=0.5).reason == "on_safety_line"

    image = np.zeros((100, 100, 3), np.uint8)
    result, _, _ = analyze(
        image, [Person((20, 20, 38, 50))], zone,
        tolerance_normalized=0.01,
    )
    assert result[0].status == "DANGER"
    assert result[0].reason == "on_safety_line"


def test_small_roi_and_safe_side_are_classified_without_scale_assumptions():
    zone = HazardZone(Segment((2, 2), (5, 2)), Segment((2, 5), (5, 5)))
    assert zone.assess_point((4, 4), (8, 8), image_edge_margin_px=0).status == "DANGER"
    assert zone.assess_point((4, 1), (8, 8), image_edge_margin_px=0).status == "SAFE"


def test_invalid_polygon_and_missing_calibration_produce_unknown():
    image = np.zeros((100, 100, 3), np.uint8)
    crossing = HazardZone(
        Segment((10, 10), (90, 90)),
        Segment((10, 90), (90, 10)),
    )
    invalid, zone, calibration = analyze(image, [Person((30, 20, 50, 60))], crossing)
    assert zone is None and calibration == "invalid_calibration"
    assert invalid[0].status == "UNKNOWN"
    assert invalid[0].reason == "invalid_calibration"

    missing, zone, calibration = analyze(image, [Person((30, 20, 50, 60))], None)
    assert zone is None and calibration == "missing_calibration"
    assert missing[0].status == "UNKNOWN"
    assert missing[0].reason == "missing_calibration"


def test_missing_or_low_confidence_ankles_are_unknown_not_safe():
    image = np.zeros((100, 100, 3), np.uint8)
    person = Person((30, 10, 50, 60), ankles=[(40, 60, 0.1)])
    decisions, _, _ = analyze(
        image, [person], _horizontal(), foot_method="pose", pose_fallback="abstain",
    )
    assert decisions[0].status == "UNKNOWN" and decisions[0].intrusion is None
    assert decisions[0].reason == "missing_or_low_confidence_keypoints"


def test_unobservable_points_are_unknown_and_near_edge_is_explicit():
    zone = _horizontal()
    assert zone.assess_point((-1, 20), (100, 100)).reason == "point_out_of_bounds"
    near = zone.assess_point((1, 20), (100, 100))
    assert near.status == "UNKNOWN" and near.reason == "near_image_edge"
    # Conclusive Beyond-E evidence remains DANGER while retaining the edge flag.
    beyond = zone.assess_point((1, 80), (100, 100))
    assert beyond.status == "DANGER"
    assert beyond.reasons == ("beyond_platform_edge", "near_image_edge")


def test_multi_person_aggregation_priority_is_danger_unknown_safe():
    danger = aggregate_frame([_decision("SAFE"), _decision("UNKNOWN"), _decision("DANGER")])
    unknown = aggregate_frame([_decision("SAFE"), _decision("UNKNOWN")])
    safe = aggregate_frame([_decision("SAFE"), _decision("SAFE")])
    assert danger.status == "DANGER"
    assert unknown.status == "UNKNOWN"
    assert safe.status == "SAFE"


def test_empty_observation_metadata_distinguishes_detection_from_no_people():
    empty = aggregate_frame([], calibration_valid=True)
    confirmed = aggregate_frame(
        [], observation_status="confirmed_no_people", calibration_valid=True,
    )
    missing_calibration = aggregate_frame(
        [], observation_status="confirmed_no_people",
        calibration_status="missing_calibration", calibration_valid=False,
    )
    failed = aggregate_frame(
        [_decision("SAFE")], observation_status="detector_failure",
        calibration_valid=True,
    )
    assert empty.status == "UNKNOWN" and empty.reason == "no_detections"
    assert empty.as_dict()["observation_status"] == "no_detections"
    assert confirmed.status == "SAFE" and confirmed.reason == "confirmed_no_people"
    assert missing_calibration.status == "UNKNOWN"
    assert missing_calibration.reason == "missing_calibration"
    assert failed.status == "UNKNOWN" and failed.reason == "detector_failure"
