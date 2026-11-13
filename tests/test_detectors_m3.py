from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

from railguard.classic_detectors import (
    AlwaysUnsafeBaseline, FrameLevelUnsafePrediction, HOGSVMDetector,
    MOG2BlobDetector,
)
from railguard.cli import main
from railguard.detection import Person, YOLODetector
from railguard.foot_refinement import refine_foot_points
from railguard.geometry import HazardZone, Segment
from railguard.pipeline import analyze


class _FakeHOG:
    def detectMultiScale(self, image, **kwargs):
        return np.array([[8, 10, 24, 70], [9, 11, 24, 70], [58, 12, 22, 68]]), np.array([2., 1., 2.])


class _Box:
    def __init__(self, bbox, confidence=.9, cls=0):
        self.xyxy = np.array([bbox], dtype=float)
        self.conf = np.array([confidence], dtype=float)
        self.cls = np.array([cls], dtype=float)


class _MockYOLO:
    def __init__(self, pose=False, missing=0):
        self.pose = pose
        self.missing = missing
        self.options = None

    def predict(self, **options):
        self.options = options
        keypoints = None
        if self.pose:
            xy = np.ones((1, 17, 2), dtype=float) * 20
            xy[0, 15] = (16, 70)
            xy[0, 16] = (28, 72)
            confidence = np.ones((1, 17), dtype=float) * .9
            for index in range(self.missing):
                xy[0, 16 - index] = (0, 0)
                confidence[0, 16 - index] = 0
            keypoints = SimpleNamespace(xy=xy, conf=confidence)
        return [SimpleNamespace(boxes=[_Box((10, 8, 34, 78))], keypoints=keypoints)]


def test_b0_is_frame_level_only_and_never_fabricates_bbox():
    result = AlwaysUnsafeBaseline().predict(np.zeros((32, 32, 3), np.uint8))
    assert isinstance(result, FrameLevelUnsafePrediction)
    assert result.unsafe and result.level == "frame"
    assert not isinstance(result, list) and not hasattr(result, "bbox")


def test_b1_empty_dark_bright_and_mocked_multi_person_are_safe_unified_outputs():
    detector = HOGSVMDetector()
    for value in (0, 255):
        people = detector.predict(np.full((160, 96, 3), value, np.uint8))
        assert all(isinstance(person, Person) for person in people)
    people = HOGSVMDetector(hog=_FakeHOG()).predict(np.zeros((160, 96, 3), np.uint8))
    assert len(people) == 2  # overlapping first pair is suppressed
    assert all(person.detector == "hog_svm" and person.runtime_ms >= 0 for person in people)


def test_b2_sequence_output_and_rejects_single_or_mixed_camera():
    frames = [np.zeros((100, 100, 3), np.uint8) for _ in range(6)]
    for index in range(3, 6):
        cv2.rectangle(frames[index], (40 + index, 30), (55 + index, 80), (255, 255, 255), -1)
    detector = MOG2BlobDetector(
        warmup=2, learning_rate=.03, min_area=30, max_area_ratio=.4,
        min_aspect=.1, max_aspect=1.2, large_train_ratio=.8)
    output = detector.predict_sequence(frames, ["cam-a"] * len(frames))
    assert len(output) == len(frames)
    assert all(isinstance(person, Person) for people in output for person in people)
    assert detector.last_diagnostics[0].warmup
    with pytest.raises(ValueError, match="single-image"):
        detector.predict(frames[0])
    with pytest.raises(ValueError, match="consistent camera_id"):
        MOG2BlobDetector().predict_sequence(frames, ["a", "b", "a", "a", "a", "a"])


def test_mock_b3_and_pose_share_person_format_and_log_inference_config():
    image = np.zeros((80, 80, 3), np.uint8)
    mock = _MockYOLO()
    person = YOLODetector(
        "bbox", weights="mock.pt", conf=.2, imgsz=320, iou=.5,
        device="cpu", model=mock).predict(image)[0]
    assert isinstance(person, Person) and person.detector == "yolo_detection"
    assert mock.options["conf"] == .2 and mock.options["imgsz"] == 320
    assert mock.options["iou"] == .5 and mock.options["device"] == "cpu"

    pose = YOLODetector("pose", weights="mock-pose.pt", model=_MockYOLO(pose=True)).predict(image)[0]
    assert isinstance(pose, Person) and len(pose.ankles) == 2
    assert pose.foot_points("bbox_bottom")[1] == "bbox_bottom_center"
    assert pose.foot_points("pose_ankle")[1] == "two_ankles"


@pytest.mark.parametrize("kwargs", [
    {"conf": -0.1}, {"conf": 1.1}, {"imgsz": 0}, {"iou": 0},
    {"iou": 1.1}, {"device": ""}, {"weights": ""},
])
def test_yolo_inference_config_validation(kwargs):
    with pytest.raises(ValueError):
        YOLODetector("bbox", model=_MockYOLO(), **kwargs)


@pytest.mark.parametrize("missing, expected", [(1, "one_ankle"), (2, "abstain")])
def test_pose_missing_one_or_two_ankles(missing, expected):
    image = np.zeros((80, 80, 3), np.uint8)
    person = YOLODetector(
        "pose", weights="mock.pt", model=_MockYOLO(pose=True, missing=missing)).predict(image)[0]
    _, source = person.foot_points("pose", fallback="abstain")
    assert source == expected
    if missing == 2:
        assert person.foot_points("pose", fallback="bbox")[1] == "bbox_fallback"
        zone = HazardZone(Segment((0, 50), (79, 50)), Segment((0, 79), (79, 79)))
        abstain, _, _ = analyze(image, [person], zone, foot_method="pose", pose_fallback="abstain")
        fallback, _, _ = analyze(image, [person], zone, foot_method="pose", pose_fallback="bbox")
        assert abstain[0].pose_outcome == "abstain" and abstain[0].intrusion is None
        assert fallback[0].pose_outcome == "bbox_fallback"


def test_footpoint_refinement_stays_in_image_bbox_and_local_roi():
    image = np.zeros((60, 70, 3), np.uint8)
    cv2.line(image, (22, 49), (42, 49), (255, 255, 255), 1)
    person = Person((20, 10, 45, 55))
    points, status = refine_foot_points(image, person, [(33, 55)])
    assert status in {"refined_local_support", "refinement_no_support"}
    assert all(0 <= x < 70 and 0 <= y < 60 for x, y in points)
    assert all(20 <= x <= 45 and 10 <= y <= 55 for x, y in points)


def test_benchmark_people_help_and_synthetic_artifacts(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["benchmark-people", "--help"])
    assert exc.value.code == 0
    assert "detector_metrics.csv" in capsys.readouterr().out

    output = tmp_path / "benchmark"
    main(["benchmark-people", "--synthetic", "--out", str(output)])
    payload = capsys.readouterr().out
    assert "DONE_SYNTHETIC" in payload
    metrics = pd.read_csv(output / "detector_metrics.csv")
    assert {"B0_always_unsafe", "B1_hog_svm", "B2_mog2_blob"}.issubset(set(metrics.detector))
    assert (output / "failure_examples" / "taxonomy.json").is_file()
