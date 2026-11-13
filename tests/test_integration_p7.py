"""Offline stationary-camera synthetic integration test for P7 QA."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from railguard.boundaries import estimate_zone
from railguard.config import get_zone
from railguard.detection import Person
from railguard.experiments import run_experiment
from railguard.pipeline import aggregate_frame, analyze
from railguard.preprocessing import apply_pipeline
from railguard.temporal import temporal_median_background
from railguard.zone_editor import config_from_points, points_from_zone


def _make_stationary_sequence(tmp_path: Path) -> tuple[list[np.ndarray], list[Person]]:
    h, w = 180, 240
    frames: list[np.ndarray] = []
    for idx in range(4):
        img = np.full((h, w, 3), 135, dtype=np.uint8)
        img[95:103, :] = (235, 235, 235)  # L: painted white safety line
        img[135:, :] = (22, 22, 22)       # E: dark track void below platform edge
        if idx == 3:
            # Draw synthetic person standing inside Z on final frame
            cv2.rectangle(img, (100, 60), (130, 120), (40, 80, 180), -1)
        frames.append(img)
    mock_people = [
        Person(
            bbox=(100.0, 60.0, 130.0, 120.0),
            confidence=0.92,
            ankles=((110.0, 120.0), (120.0, 120.0)),
        )
    ]
    return frames, mock_people


def test_full_offline_pipeline_and_e1_to_e5_bundle(tmp_path: Path):
    frames, mock_people = _make_stationary_sequence(tmp_path)
    bg = temporal_median_background(frames, ["cam_a"] * len(frames))
    pre_m2 = apply_pipeline(bg, ["clahe"])
    suggestion = estimate_zone(pre_m2, method="hybrid_region_evidence", color="white")
    assert suggestion.zone is not None

    pts = points_from_zone(suggestion.zone, 240, 180)
    cfg = config_from_points(pts, 240, 180)
    zone = get_zone(cfg, 240, 180)

    results, zone_used, cal_status = analyze(frames[-1], mock_people, zone, foot_method="pose")
    frame = aggregate_frame(results, observation_status="detections_complete",
                            calibration_status=cal_status, calibration_valid=zone_used is not None)
    assert frame.status in {"DANGER", "SAFE", "UNKNOWN"}

    out_dir = tmp_path / "exp"
    bundle = run_experiment("E4", out_dir=out_dir, synthetic=True)
    assert (out_dir / "e4_ablation.csv").exists()
    metrics = json.loads((out_dir / "E4" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "SYNTHETIC_FIXTURE_ONLY"
    assert bundle["experiment"] == "E4"
