"""Tests for P6 UI state, calibration guards, debug stages, and exports."""
from __future__ import annotations

import csv
import io

import numpy as np
import pytest

from railguard.geometry import HazardZone, Segment
from railguard.ui_state import (
    CalibrationSessionState,
    available_temporal_controls,
    build_debug_stages,
    can_run_inference,
    results_to_csv,
    run_ui_inference,
)


def _synthetic_frame(w: int = 160, h: int = 120) -> np.ndarray:
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    img[65:70, :] = (240, 240, 240)
    img[95:, :] = (25, 25, 25)
    return img


def test_calibration_session_undo_reset_and_reconfirm():
    state = CalibrationSessionState()
    p1 = [[0.1, 0.5], [0.9, 0.5], [0.1, 0.8], [0.9, 0.8]]
    assert state.update_points(p1, source="auto") is True
    state.confirmed = True

    p2 = [[0.2, 0.5], [0.9, 0.5], [0.1, 0.8], [0.9, 0.8]]
    assert state.update_points(p2, source="manual") is True
    assert state.confirmed is False  # editing invalidates confirmation

    assert state.undo() is True
    assert state.points == p1
    assert state.confirmed is False

    state.reset()
    assert state.points == [None, None, None, None]
    assert state.undo() is True
    assert state.points == p1


def test_single_image_disables_mog2_and_temporal_median():
    assert available_temporal_controls(1) == {
        "mog2_enabled": False,
        "temporal_median_enabled": False,
    }
    assert available_temporal_controls(2) == {
        "mog2_enabled": True,
        "temporal_median_enabled": True,
    }
    frame = _synthetic_frame()
    with pytest.raises(ValueError, match="at least 2 frames"):
        build_debug_stages([frame], use_temporal_median=True)


def test_invalid_or_unconfirmed_calibration_blocks_inference():
    zone = HazardZone(Segment((10.0, 60.0), (150.0, 60.0)), Segment((10.0, 95.0), (150.0, 95.0)))
    assert can_run_inference(None, True)[0] is False
    assert can_run_inference(zone, False)[0] is False
    assert can_run_inference(zone, True) == (True, "ok")

    frame = _synthetic_frame()
    with pytest.raises(ValueError, match="Confirm"):
        run_ui_inference([frame], zone, confirmed=False, detector_kind="B1 HOG+SVM")


def test_ui_inference_b1_and_csv_export():
    zone = HazardZone(Segment((10.0, 60.0), (150.0, 60.0)), Segment((10.0, 95.0), (150.0, 95.0)))
    frame = _synthetic_frame()
    stages = build_debug_stages([frame], boundary_preproc="clahe", detector_preproc="raw")
    assert "Input" in stages
    assert "Preprocessing" in stages

    out = run_ui_inference([frame], zone, confirmed=True, detector_kind="B1 HOG+SVM")
    assert out["image_png"].startswith(b"\x89PNG")
    assert "frame" in out and "params" in out
    csv_text = results_to_csv(out["results"])
    rows = list(csv.reader(io.StringIO(csv_text)))
    assert rows[0][:4] == ["person_id", "status", "intrusion", "foot_source"]
