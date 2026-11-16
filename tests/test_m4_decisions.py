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


def test_above_finite_vp_horizon_is_unknown_while_below_remains_decidable():
    # L: x=20 at y=99 -> x=46 at y=25; E: x=70 at y=99 -> x=53 at y=25 -> VP near (49.3, 15.5)
    zone = HazardZone(Segment((20, 99), (46, 25)), Segment((70, 99), (53, 25)))
    horizon_y = zone.observable_horizon_y(100, 100)
    assert horizon_y is not None and horizon_y > 15.0

    # Above VP horizon: infinite lines cross/invert, so footpoint must be UNKNOWN/outside_observable_zone
    above = zone.assess_point((35, 10), (100, 100))
    assert above.status == "UNKNOWN"
    assert above.reason == "outside_observable_zone"
    assert above.inside_z is False

    # Below VP horizon: M4 rules remain unchanged (SAFE left of L, DANGER beyond E)
    safe_pt = zone.assess_point((10, 80), (100, 100))
    danger_pt = zone.assess_point((85, 80), (100, 100))
    assert safe_pt.status == "SAFE"
    assert danger_pt.status == "DANGER" and danger_pt.reason == "beyond_platform_edge"


def test_structured_foot_evidence_rejects_glass_reflection_while_keeping_real_foot():
    import cv2
    from railguard.foot_refinement import FootEvidence, assess_foot_evidence

    image = np.full((240, 240, 3), 160, np.uint8)
    # Horizontal platform brick texture on real side (x < 98)
    for y in range(20, 235, 16):
        cv2.line(image, (0, y), (96, y), (110, 110, 110), 2)
    # Continuous dark vertical track trench/seam (x = 98..126) extending below foot height
    image[:, 98:127] = 18
    # Train side (x >= 127) with strong vertical window mullions/reflections
    image[:, 127:] = 85
    for x in range(132, 236, 12):
        cv2.line(image, (x, 0), (x, 239), (30, 30, 30), 2)

    kpts = [(112.0, 90.0, 0.9)] * 17
    kpts[11] = (106.0, 120.0, 0.95)
    kpts[12] = (114.0, 120.0, 0.95)
    kpts[13] = (86.0, 150.0, 0.95)
    kpts[14] = (140.0, 150.0, 0.95)
    kpts[15] = (70.0, 175.0, 0.95)
    kpts[16] = (160.0, 175.0, 0.95)
    person = Person((55, 40, 175, 185), 0.9, ankles=[kpts[15], kpts[16]], keypoints=kpts)

    evidences, wrapper_status = assess_foot_evidence(image, person, [(70.0, 175.0), (160.0, 175.0)])
    assert len(evidences) == 2 and all(isinstance(ev, FootEvidence) for ev in evidences)
    assert evidences[0].status == "accepted"
    assert evidences[1].status == "rejected_reflection"
    assert wrapper_status in {"refined_local_support", "refinement_no_support"}

    zone = HazardZone(Segment((45, 0), (45, 239)), Segment((96, 0), (96, 239)))
    decisions, _, _ = analyze(image, [person], zone, foot_method="pose", refine_footpoints=True)
    assert len(decisions) == 1
    assert decisions[0].status == "DANGER"
    assert decisions[0].reason == "inside_hazard_zone"
    assert len(decisions[0].foot_points) == 1
    assert decisions[0].foot_points[0][0] < 96.0


def test_safe_with_ambiguous_or_rejected_foot_evidence_becomes_unknown():
    image = np.full((120, 120, 3), 140, np.uint8)
    # Inverted hip->knee->ankle skeleton (ankle above knee) on safe side of L=(80,y)
    kpts = [(30.0, 20.0, 0.9)] * 17
    kpts[11] = (28.0, 50.0, 0.9)
    kpts[12] = (34.0, 50.0, 0.9)
    kpts[13] = (28.0, 82.0, 0.9)
    kpts[14] = (34.0, 82.0, 0.9)
    kpts[15] = (28.0, 65.0, 0.9)
    kpts[16] = (34.0, 95.0, 0.9)
    person = Person((15, 10, 48, 100), 0.9, ankles=[kpts[15], kpts[16]], keypoints=kpts)
    zone = HazardZone(Segment((80, 0), (80, 119)), Segment((100, 0), (100, 119)))

    decisions, _, _ = analyze(image, [person], zone, foot_method="pose", refine_footpoints=True)
    assert len(decisions) == 1
    assert decisions[0].status == "UNKNOWN"
    assert decisions[0].reason == "ambiguous_footpoint_evidence"
    assert decisions[0].intrusion is None
    assert decisions[0].foot_evidence[0]["status"] == "ambiguous"


def test_aggregate_frame_completeness_priority_order():
    safe_dec = Decision((1, 2, 3, 4), 0.9, [(1, 2)], "pose", False)
    danger_dec = Decision((1, 2, 3, 4), 0.9, [(1, 2)], "pose", True)
    unknown_dec = Decision((1, 2, 3, 4), 0.9, [], "abstain", None)

    partial_cov = {"proposals": 3, "pose_covered": 1, "uncovered": 2, "detector_partial": True}
    complete_cov = {"proposals": 3, "pose_covered": 3, "uncovered": 0, "detector_partial": False}

    # SAFE + partial detection -> UNKNOWN (prevents False SAFE when distant workers are missed)
    frame_partial = aggregate_frame([safe_dec], detector_coverage=partial_cov)
    assert frame_partial.status == "UNKNOWN"
    assert frame_partial.reason == "detections_partial"
    assert frame_partial.observation_status == "detections_partial"
    assert frame_partial.detector_partial is True

    # Priority order: DANGER > detector partial, and person UNKNOWN > detector partial
    assert aggregate_frame([safe_dec, danger_dec], detector_coverage=partial_cov).status == "DANGER"
    unk_partial = aggregate_frame([safe_dec, unknown_dec], detector_coverage=partial_cov)
    assert unk_partial.status == "UNKNOWN" and unk_partial.reason == "person_unknown"

    # detector partial > calibration failure
    cal_and_partial = aggregate_frame([safe_dec], calibration_valid=False, detector_coverage=partial_cov)
    assert cal_and_partial.status == "UNKNOWN" and cal_and_partial.reason == "detections_partial"

    # SAFE + complete coverage -> SAFE
    complete_safe = aggregate_frame([safe_dec], detector_coverage=complete_cov)
    assert complete_safe.status == "SAFE" and complete_safe.detector_partial is False


def test_assess_foot_evidence_cached_gray_and_edges_matches_uncached():
    import cv2
    from railguard.foot_refinement import assess_foot_evidence

    image = np.full((120, 120, 3), 140, np.uint8)
    cv2.rectangle(image, (30, 85), (75, 92), (35, 35, 35), -1)
    person = Person((25, 20, 80, 95), 0.9, ankles=[(40.0, 88.0, 0.9), (60.0, 89.0, 0.9)])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)

    seeds = [(40.0, 88.0), (60.0, 89.0)]
    ev_uncached = assess_foot_evidence(image, person, seeds)
    ev_cached = assess_foot_evidence(image, person, seeds, gray=gray, edges=edges)
    assert ev_cached == ev_uncached
