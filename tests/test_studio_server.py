"""Studio server API handler parity tests (M1 preprocess, M2 horizon, M3 coverage, M4 completeness)."""
import base64

import cv2
import numpy as np
import pytest

from railguard.detection import Person
from railguard.server import handle_auto_zone, handle_run_pipeline


def _encode_png_data_uri(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def test_server_handle_auto_zone_accepts_preprocess_and_returns_horizon():
    image = np.full((120, 160, 3), 210, np.uint8)
    # Draw yellow warning stripe (BGR) and dark track trench
    image[:, 70:82] = (30, 210, 235)
    image[:, 100:] = 25
    b64 = _encode_png_data_uri(image)

    resp = handle_auto_zone({"image": b64, "method": "lab_canny_hough", "preprocess": "clahe"})
    assert resp["preprocess"] == "clahe"
    assert resp["preproc"] == "clahe"
    assert resp["method"] == "lab_canny_hough"
    assert "diagnostics" in resp
    if resp["points"] is not None:
        assert "horizon_y" in resp


def test_server_handle_run_pipeline_propagates_flags_and_partial_coverage():
    image = np.full((120, 120, 3), 180, np.uint8)
    b64 = _encode_png_data_uri(image)
    calls = {}

    class _StubDetector:
        def __init__(self, method, *, weights, confidence, imgsz, enable_crop_pose=True):
            calls["method"] = method
            calls["enable_crop_pose"] = enable_crop_pose
            self.last_coverage = {
                "proposals": 3,
                "pose_covered": 1,
                "uncovered": 2,
                "detector_partial": True,
            }

        def predict(self, proc_img):
            return [Person((10, 20, 40, 100), 0.9, ankles=[(20.0, 95.0, 0.9), (30.0, 95.0, 0.9)])]

    resp = handle_run_pipeline(
        {
            "image": b64,
            "detector": "A YOLO pose",
            "preprocess": "gamma",
            "enable_crop_pose": False,
            "refine_footpoints": False,
            "points": [[0.6, 1.0], [0.6, 0.0], [0.85, 1.0], [0.85, 0.0]],
        },
        detector_factory=_StubDetector,
    )
    assert calls == {"method": "pose", "enable_crop_pose": False}
    assert resp["preprocess"] == "gamma"
    assert resp["enable_crop_pose"] is False
    assert resp["refine_footpoints"] is False
    assert resp["frame"]["status"] == "UNKNOWN"
    assert resp["frame"]["reason"] == "detections_partial"
    assert resp["frame"]["detector_partial"] is True
    assert resp["detector_coverage"]["proposals"] == 3
    assert isinstance(resp["overlay_image"], str) and len(resp["overlay_image"]) > 100


def test_server_rejects_corrupt_image_payload():
    with pytest.raises(ValueError):
        handle_auto_zone({"image": "data:image/png;base64,bm90LWFuLWltYWdl"})


def test_server_auto_preproc_isolates_m2_and_enables_glare_recovery():
    image = np.full((120, 160, 3), 245, np.uint8)
    image[:, 70:82] = (30, 210, 235)
    image[:, 100:] = 25
    b64 = _encode_png_data_uri(image)

    auto_resp = handle_auto_zone({"image": b64, "method": "lab_canny_hough", "preproc": "auto"})
    assert auto_resp["preprocess"] == "auto"
    assert auto_resp["effective_preprocess"] == "raw"

    hl_resp = handle_auto_zone({"image": b64, "method": "lab_canny_hough", "preproc": "highlight_compress"})
    assert hl_resp["preprocess"] == "highlight_compress"
    assert hl_resp["effective_preprocess"] == "raw"

    for m1_only in ("gamma", "clahe"):
        m1_resp = handle_auto_zone({"image": b64, "method": "lab_canny_hough", "preproc": m1_only})
        assert m1_resp["preprocess"] == m1_only
        assert m1_resp["effective_preprocess"] == "raw"

    recorded = {}

    class _GlareAwareStubDetector:
        def __init__(self, method, *, weights, confidence, imgsz, enable_crop_pose=True, glare_recovery=False):
            recorded["glare_recovery"] = glare_recovery
            self.last_coverage = {"proposals": 1, "pose_covered": 1, "uncovered": 0, "detector_partial": False}

        def predict(self, proc_img):
            recorded["input_mean"] = float(proc_img.mean())
            return [Person((10, 20, 40, 100), 0.9, ankles=[(20.0, 95.0, 0.9), (30.0, 95.0, 0.9)])]

    run_auto = handle_run_pipeline(
        {
            "image": b64,
            "detector": "A YOLO pose",
            "preproc": "auto",
            "points": [[0.6, 1.0], [0.6, 0.0], [0.85, 1.0], [0.85, 0.0]],
        },
        detector_factory=_GlareAwareStubDetector,
    )
    assert recorded["glare_recovery"] is True
    assert run_auto["glare_recovery"] is True
    assert run_auto["preprocess"] == "auto"

    run_hl = handle_run_pipeline(
        {
            "image": b64,
            "detector": "A YOLO pose",
            "preproc": "highlight_compress",
            "points": [[0.6, 1.0], [0.6, 0.0], [0.85, 1.0], [0.85, 0.0]],
        },
        detector_factory=_GlareAwareStubDetector,
    )
    assert recorded["glare_recovery"] is False
    assert recorded["input_mean"] < float(image.mean())
    assert run_hl["preprocess"] == "highlight_compress"


def test_server_routes_auto_cascade_kwargs_and_detector_cache(monkeypatch):
    import railguard.server as srv

    image = np.full((120, 160, 3), 240, np.uint8)
    b64 = _encode_png_data_uri(image)
    calls = []

    class _FullKwargStubDetector:
        def __init__(
            self, method, *, weights, confidence, imgsz=640,
            enable_crop_pose=True, glare_recovery=False,
            glare_exposure="highlight_compress", cascade_mode="auto",
        ):
            calls.append((method, imgsz, glare_recovery, glare_exposure, cascade_mode))
            self.last_coverage = {"proposals": 1, "pose_covered": 1, "uncovered": 0, "detector_partial": False}
            self.last_glare_rois = [(0, 0, 80, 120)] if glare_recovery else []

        def predict(self, proc_img):
            return [Person((10, 20, 40, 100), 0.9, ankles=[(20.0, 95.0, 0.9), (30.0, 95.0, 0.9)])]

    resp = handle_run_pipeline(
        {
            "image": b64,
            "detector": "A YOLO pose",
            "preproc": "auto",
            "points": [[0.6, 1.0], [0.6, 0.0], [0.85, 1.0], [0.85, 0.0]],
        },
        detector_factory=_FullKwargStubDetector,
    )
    assert calls[-1] == ("pose", 640, True, "gamma06", "dual_full")
    assert resp["imgsz"] == 640
    assert resp["glare_exposure"] == "gamma06"
    assert resp["cascade_mode"] == "dual_full"
    assert resp["glare_rois"] == [[0, 0, 80, 120]]

    # High-resolution wide frame (>=1280px) routes to 960 + bbox_first cascade
    hd_b64 = _encode_png_data_uri(np.full((720, 1280, 3), 240, np.uint8))
    resp_hd = handle_run_pipeline(
        {
            "image": hd_b64,
            "detector": "A YOLO pose",
            "preproc": "auto",
            "points": [[0.6, 1.0], [0.6, 0.0], [0.85, 1.0], [0.85, 0.0]],
        },
        detector_factory=_FullKwargStubDetector,
    )
    assert calls[-1] == ("pose", 960, True, "gamma06", "bbox_first")
    assert resp_hd["imgsz"] == 960
    assert resp_hd["cascade_mode"] == "bbox_first"

    # Bare "gamma" in Studio must apply actual highlight gamma darkening (< 1.0), not identity 1.0
    gamma_seen = {}

    class _GammaCheckDetector:
        def __init__(self, *args, **kwargs):
            self.last_coverage = {"proposals": 1, "pose_covered": 1, "uncovered": 0, "detector_partial": False}

        def predict(self, proc_img):
            gamma_seen["mean"] = float(proc_img.mean())
            return []

    handle_run_pipeline(
        {
            "image": b64,
            "detector": "A YOLO pose",
            "preproc": "gamma",
            "points": [[0.6, 1.0], [0.6, 0.0], [0.85, 1.0], [0.85, 0.0]],
        },
        detector_factory=_GammaCheckDetector,
    )
    assert gamma_seen["mean"] < 235.0

    # Verify _DETECTOR_CACHE reuses real YOLODetector instances on repeated identical requests
    srv._DETECTOR_CACHE.clear()
    init_count = {"n": 0}

    class _CountingFactory(_FullKwargStubDetector):
        def __init__(self, *args, **kwargs):
            init_count["n"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(srv, "YOLODetector", _CountingFactory)
    payload = {
        "image": b64,
        "detector": "A YOLO pose",
        "preproc": "auto",
        "points": [[0.6, 1.0], [0.6, 0.0], [0.85, 1.0], [0.85, 0.0]],
    }
    handle_run_pipeline(payload, detector_factory=_CountingFactory)
    handle_run_pipeline(payload, detector_factory=_CountingFactory)
    assert init_count["n"] == 1
    srv._DETECTOR_CACHE.clear()


def test_server_auto_zone_train_suppression_default_and_lazy_challenger():
    import railguard.server as srv
    from railguard.train_mask import TrainMaskResult

    image = np.full((160, 200, 3), 190, np.uint8)
    image[:, 80:94] = (30, 210, 235)
    image[:, 130:] = 20
    payload = _encode_png_data_uri(image)
    calls = {"init": 0, "predict": 0}

    class StubProvider:
        def __init__(self, **kwargs):
            calls["init"] += 1
            calls["kwargs"] = kwargs

        def predict(self, source):
            calls["predict"] += 1
            mask = np.zeros(source.shape[:2], bool)
            mask[:, 150:195] = True
            return TrainMaskResult(mask, {"valid": True, "coverage": float(mask.mean())})

    srv._TRAIN_MASK_PROVIDER_CACHE.clear()
    structural = handle_auto_zone({"image": payload, "method": "lab_canny_hough"},
                                  train_mask_provider_factory=StubProvider)
    assert structural["train_suppression_requested"] == "structural"
    assert structural["train_suppression_effective"] == "structural"
    assert calls["init"] == 0 and calls["predict"] == 0

    request = {"image": payload, "method": "lab_canny_hough", "train_suppression": "yolo_seg"}
    first = handle_auto_zone(request, train_mask_provider_factory=StubProvider)
    second = handle_auto_zone(request, train_mask_provider_factory=StubProvider)
    assert first["train_prior_valid"] is True
    assert first["train_suppression_effective"] == "yolo_seg"
    assert second["train_prior_valid"] is True
    assert calls["init"] == 1 and calls["predict"] == 2
    srv._TRAIN_MASK_PROVIDER_CACHE.clear()
