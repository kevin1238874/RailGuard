import numpy as np
import pytest
from railguard.geometry import HazardZone, Segment
from railguard.config import get_zone
from railguard.detection import Person
from railguard.pipeline import analyze


def test_normalized_to_pixels():
    cfg = {"coordinates": "normalized", "zone": {
        "safety_line": [[0, 0.6], [1, 0.6]],
        "platform_edge": [[0, 0.9], [1, 0.9]]}}
    zone = get_zone(cfg, 101, 101)
    assert zone.contains((40, 75))
    assert zone.contains((40, 60))  # boundary included
    assert not zone.contains((40, 30))
    assert not zone.contains((40, 95))


def test_reversed_segment_order():
    zone = HazardZone(Segment((100, 60), (0, 60)), Segment((100, 80), (0, 80)))
    assert zone.contains((50, 75))
    assert not zone.contains((50, 40))


def test_reject_degenerate_geometry():
    with pytest.raises(ValueError, match="Degenerate"):
        HazardZone(Segment((0, 0), (100, 0)), Segment((0, 0), (100, 0))).polygon()


def test_missing_calibration_abstains():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    decisions, zone, status = analyze(image, [Person((10, 10, 40, 80))])
    assert decisions == [] and zone is None and status == "missing_calibration"


def test_ankle_rule_checks_either_foot():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    zone = HazardZone(Segment((0, 60), (99, 60)), Segment((0, 90), (99, 90)))
    person = Person((10, 20, 30, 90), ankles=[(15, 50, .9), (25, 70, .9)])
    decisions, _, _ = analyze(image, [person], zone, foot_method="pose")
    assert decisions[0].intrusion is True
    assert decisions[0].foot_source == "two_ankles"


def test_missing_pose_keypoints_abstain_or_fallback():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    zone = HazardZone(Segment((0, 60), (99, 60)), Segment((0, 90), (99, 90)))
    person = Person((10, 20, 30, 80), ankles=[(0, 0, 0.9)])
    res, _, _ = analyze(image, [person], zone, foot_method="pose", pose_fallback="abstain")
    assert res[0].intrusion is None
    res, _, _ = analyze(image, [person], zone, foot_method="pose", pose_fallback="bbox")
    assert res[0].intrusion is True and res[0].foot_source == "bbox_fallback"
