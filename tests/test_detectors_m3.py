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


def _make_mock_box(xyxy, conf=0.9, cls=0):
    return SimpleNamespace(xyxy=np.array([xyxy], dtype=float), conf=np.array([conf]), cls=np.array([cls]))


def _make_coco17(left_ankle, right_ankle):
    pts = [(1.0, 1.0)] * 17
    pts[15] = left_ankle
    pts[16] = right_ankle
    return pts, [0.95] * 17


class _MockResult:
    def __init__(self, boxes, kpts=None):
        self.boxes = boxes
        if kpts is not None:
            xy = np.array([k[0] for k in kpts], dtype=float)
            conf = np.array([k[1] for k in kpts], dtype=float)
            self.keypoints = SimpleNamespace(xy=xy, conf=conf)
        else:
            self.keypoints = None


def test_crop_pose_best_iou_one_to_one_matching_and_coverage():
    image = np.zeros((200, 200, 3), np.uint8)
    close_box = (20.0, 40.0, 70.0, 180.0)
    far_box = (110.0, 30.0, 140.0, 100.0)

    class _ProposalDet:
        def predict(self, **kwargs):
            return [_MockResult([_make_mock_box(close_box, 0.92), _make_mock_box(far_box, 0.78)])]

    class _PoseWithCrop:
        def predict(self, **kwargs):
            src = kwargs["source"]
            if src.shape[:2] == (200, 200):
                return [_MockResult([_make_mock_box(close_box, 0.90)], [_make_coco17((35.0, 172.0), (55.0, 174.0))])]
            # Cropped distant person: return local coordinates inside padded crop.
            ch, cw = src.shape[:2]
            local_box = (4.0, 4.0, max(6.0, cw - 4.0), max(8.0, ch - 4.0))
            return [_MockResult([_make_mock_box(local_box, 0.85)], [_make_coco17((12.0, 60.0), (22.0, 62.0))])]

    det_crop = YOLODetector("pose", weights="mock-pose.pt", model=_PoseWithCrop(),
                            det_model=_ProposalDet(), enable_crop_pose=True)
    people_crop = det_crop.predict(image)
    assert len(people_crop) == 2
    assert det_crop.last_proposals == 2
    assert det_crop.last_pose_covered == 2
    assert det_crop.last_coverage["detector_partial"] is False
    assert people_crop[0].metadata["pose_source"] == "direct_pose"
    assert people_crop[1].metadata["pose_source"] == "crop_pose"
    assert people_crop[1].keypoints is not None and len(people_crop[1].keypoints) == 17
    # Crop offset check: x > 90 and y > 15 -> shifted back to full image coordinates.
    assert all(px > 90 and py > 15 for px, py, _ in people_crop[1].ankles)

    det_nocrop = YOLODetector("pose", weights="mock-pose.pt", model=_PoseWithCrop(),
                              det_model=_ProposalDet(), enable_crop_pose=False)
    people_nocrop = det_nocrop.predict(image)
    assert len(people_nocrop) == 1
    assert det_nocrop.last_proposals == 2
    assert det_nocrop.last_pose_covered == 1
    assert det_nocrop.last_coverage["detector_partial"] is True


def test_pose_crop_matches_target_person_when_multiple_pose_candidates_in_crop():
    image = np.zeros((200, 200, 3), np.uint8)
    target_box = (80.0, 40.0, 120.0, 140.0)

    class _ProposalDet:
        def predict(self, **kwargs):
            return [_MockResult([_make_mock_box(target_box, 0.88)])]

    class _MultiCropPose:
        def predict(self, **kwargs):
            src = kwargs["source"]
            if src.shape[:2] == (200, 200):
                return [_MockResult([])]
            # Candidate 0 is a skinny sliver on far left of crop; candidate 1 matches target bbox.
            pad_x = max(4, int((120.0 - 80.0) * 0.15))
            pad_y = max(4, int((140.0 - 40.0) * 0.15))
            cx1, cy1 = max(0, int(80.0 - pad_x)), max(0, int(40.0 - pad_y))
            b0 = (0.0, 0.0, 5.0, 20.0)
            b1 = (80.0 - cx1, 40.0 - cy1, 120.0 - cx1, 140.0 - cy1)
            return [_MockResult(
                [_make_mock_box(b0, 0.70), _make_mock_box(b1, 0.90)],
                [_make_coco17((2.0, 18.0), (3.0, 19.0)), _make_coco17((20.0, 95.0), (30.0, 96.0))],
            )]

    det = YOLODetector("pose", weights="mock-pose.pt", model=_MultiCropPose(),
                       det_model=_ProposalDet(), enable_crop_pose=True)
    people = det.predict(image)
    assert len(people) == 1
    # Must choose candidate 1's ankles shifted by (cx1, cy1), NOT candidate 0's corner ankles.
    assert people[0].ankles[0][1] > 100.0


def test_weight_resolution_independent_of_cwd_and_mock_never_loads_yolo(tmp_path, monkeypatch):
    from pathlib import Path
    from railguard.detection import _resolve_weight_path

    det_mock = YOLODetector("pose", weights="mock.pt", model=_MockYOLO(pose=True))
    assert det_mock._det_model is None

    monkeypatch.chdir(tmp_path)
    resolved = Path(_resolve_weight_path("yolo11n.pt"))
    assert resolved.is_file()
    assert resolved.name == "yolo11n.pt"


def test_failed_crop_pose_marks_partial_and_rejects_zero_iou():
    from railguard.geometry import HazardZone, Segment
    from railguard.pipeline import aggregate_frame, analyze

    image = np.zeros((240, 240, 3), np.uint8)
    target_box = (80.0, 60.0, 120.0, 160.0)

    class _ProposalDet:
        def predict(self, **kwargs):
            return [_MockResult([_make_mock_box(target_box, 0.90)])]

    class _DisjointCropPose:
        def predict(self, **kwargs):
            src = kwargs["source"]
            if src.shape[:2] == (240, 240):
                return [_MockResult([])]
            # Only return a zero-IoU box inside the padded crop (outside target_box).
            return [_MockResult(
                [_make_mock_box((0.0, 0.0, 4.0, 10.0), 0.95)],
                [_make_coco17((2.0, 8.0), (3.0, 9.0))],
            )]

    det = YOLODetector("pose", weights="mock-pose.pt", model=_DisjointCropPose(),
                       det_model=_ProposalDet(), enable_crop_pose=True)
    people = det.predict(image)
    assert len(people) == 1
    assert people[0].metadata["pose_source"] == "det_only_no_pose"
    assert people[0].keypoints is None
    assert det.last_coverage["crop_pose_recovered"] == 0
    assert det.last_coverage["coverage_ratio"] == 0.0
    assert det.last_coverage["detector_partial"] is True

    # Even if fallback="bbox" yields a SAFE point, partial pose coverage must escalate frame -> UNKNOWN.
    zone = HazardZone(
        Segment((20.0, 20.0), (20.0, 220.0)),
        Segment((10.0, 20.0), (10.0, 220.0)),
    )
    decisions, _, _ = analyze(image, people, zone)
    assert decisions[0].status == "SAFE"
    frame = aggregate_frame(decisions, detector_coverage=det.last_coverage)
    assert frame.status == "UNKNOWN"
    assert frame.reason == "detections_partial"


def test_glare_recovery_recovers_sunlit_person_and_filters_unverified_sliver():
    # Scene with shadow half + bright blown-out glare patch (L >= 210) on right half
    image = np.full((160, 160, 3), 60, np.uint8)
    image[20:140, 80:155] = 248

    primary_box = (15.0, 40.0, 45.0, 130.0)
    sunlit_box = (95.0, 35.0, 130.0, 130.0)
    sliver_box = (82.0, 35.0, 92.0, 115.0)

    class _GlareMockPose:
        def __init__(self):
            self.calls = 0

        def predict(self, **kwargs):
            self.calls += 1
            src = kwargs["source"]
            # Full-frame glare image is visibly darker on the bright patch than raw image
            is_compressed = float(src[25:50, 90:150].mean()) < 240.0
            if src.shape[:2] == (160, 160) and not is_compressed:
                return [_MockResult(
                    [_make_mock_box(primary_box, 0.85)],
                    [_make_coco17((22.0, 124.0), (35.0, 125.0))],
                )]
            if src.shape[:2] == (160, 160) and is_compressed:
                return [_MockResult(
                    [
                        _make_mock_box(primary_box, 0.86),   # duplicate -> skipped
                        _make_mock_box(sunlit_box, 0.28),    # valid ankles inside box -> recovered
                        _make_mock_box(sliver_box, 0.22),    # low conf + ankles outside box -> rejected
                    ],
                    [
                        _make_coco17((22.0, 124.0), (35.0, 125.0)),
                        _make_coco17((104.0, 124.0), (118.0, 125.0)),
                        _make_coco17((0.0, 0.0), (0.0, 0.0)),
                    ],
                )]
            return [_MockResult([])]

    mock = _GlareMockPose()
    det = YOLODetector("pose", weights="mock-pose.pt", model=mock, glare_recovery=True)
    people = det.predict(image)
    assert len(people) == 2
    assert people[1].detector == "yolo_glare_recovered"
    assert people[1].metadata["glare_recovered"] is True
    assert people[1].metadata["glare_bright_frac"] >= 0.15
    assert det.last_coverage["glare_recovered"] == 1

    # On a dark scene without glare, glare_recovery skips extra passes
    dark = np.full((160, 160, 3), 40, np.uint8)
    mock_dark = _GlareMockPose()
    det_dark = YOLODetector("pose", weights="mock-pose.pt", model=mock_dark, glare_recovery=True)
    det_dark.predict(dark)
    assert mock_dark.calls == 1


def test_glare_exposure_variants_and_bbox_first_cascade():
    image = np.full((480, 360, 3), 75, np.uint8)
    # Create localized high-glare zone with real contrast near top-left; bottom stays darker
    image[40:180, 10:150] = 246
    cv2.rectangle(image, (40, 60), (95, 165), (50, 50, 50), -1)

    with pytest.raises(ValueError, match="glare_exposure"):
        YOLODetector("pose", weights="mock.pt", model=_MockYOLO(pose=True), glare_exposure="bad_mode")
    with pytest.raises(ValueError, match="cascade_mode"):
        YOLODetector("pose", weights="mock.pt", model=_MockYOLO(pose=True), cascade_mode="bad_mode")

    base_box = (240.0, 260.0, 305.0, 440.0)

    class _DetBBoxCascade:
        def __init__(self):
            self.calls = 0

        def predict(self, **kwargs):
            self.calls += 1
            src = kwargs["source"]
            # Full frame finds dark-region person; first darkened glare ROI pass recovers glare person
            if src.shape[:2] == (480, 360):
                return [_MockResult([_make_mock_box(base_box, 0.88)])]
            if self.calls == 2 and float(src.mean()) < 220.0:
                return [_MockResult([_make_mock_box((38.0, 16.0, 96.0, 168.0), 0.72)])]
            return [_MockResult([])]

    class _PoseBatchMock:
        def __init__(self):
            self.batch_sizes = []
            self.single_calls = 0

        def _pose_res(self, c):
            return _MockResult(
                [_make_mock_box((5.0, 5.0, c.shape[1] - 5.0, c.shape[0] - 5.0), 0.85)],
                [_make_coco17((20.0, c.shape[0] - 8.0), (45.0, c.shape[0] - 8.0))],
            )

        def predict(self, **kwargs):
            src = kwargs["source"]
            if isinstance(src, list):
                self.batch_sizes.append(len(src))
                return [self._pose_res(c) for c in src]
            self.single_calls += 1
            return [self._pose_res(src)]

    pose_mock = _PoseBatchMock()
    det_mock = _DetBBoxCascade()
    det = YOLODetector(
        "pose", weights="mock-pose.pt", model=pose_mock, det_model=det_mock,
        glare_recovery=True, glare_exposure="gamma06", cascade_mode="bbox_first",
    )
    people = det.predict(image)
    assert len(det.last_glare_rois) >= 1
    assert det_mock.calls >= 2
    assert pose_mock.batch_sizes == [2]
    assert len(people) == 2
    sources = {p.metadata["pose_source"] for p in people}
    assert sources == {"crop_pose", "glare_gamma06_crop_pose"}
    glare_people = [p for p in people if p.metadata.get("glare_recovered")]
    assert len(glare_people) == 1
    assert glare_people[0].detector == "yolo_glare_recovered"


def _bbox_iou_helper(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, (a[2] - a[0]) * (a[3] - a[1])) + max(0.0, (b[2] - b[0]) * (b[3] - b[1])) - inter
    return inter / area if area > 0 else 0.0
