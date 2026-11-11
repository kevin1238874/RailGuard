"""Regression coverage for direct manipulation calibration and auto proposals."""
import cv2
import numpy as np
import pytest
import yaml

from railguard.boundaries import estimate_zone
from railguard.zone_editor import (
    empty_points, validate_points, config_from_points, points_from_config,
    points_from_zone, safe_camera_id, calibration_yaml, write_calibration,
)


def test_incomplete_zone_cannot_be_saved_or_approved():
    assert empty_points() == [None] * 4
    with pytest.raises(ValueError, match="L0"):
        config_from_points(empty_points(), 200, 100)
    partial = [[0.1, 0.2], None, None, None]
    assert validate_points(partial) == partial
    with pytest.raises(ValueError):
        config_from_points(partial, 200, 100)


@pytest.mark.parametrize("points", [
    [[0, 0], [1, 0], [0, 1]],
    [[float("nan"), 0], [1, 0], [0, 1], [1, 1]],
    [[1.1, 0], [1, 0], [0, 1], [1, 1]],
    [[0, 0], [1, 0], [0, -0.3], [1, 1]],
    [[True, 0], [1, 0], [0, 1], [1, 1]],
])
def test_reject_invalid_endpoint_payload(points):
    with pytest.raises(ValueError):
        validate_points(points)


def test_valid_zone_yaml_round_trip_and_save(tmp_path):
    # Near-vertical lines: representative of the user's station camera view.
    points = [[.25, .0], [.08, 1.0], [.57, 0], [.45, 1]]
    cfg = config_from_points(points, 800, 480)
    assert cfg["coordinates"] == "normalized"
    text = calibration_yaml(points, 800, 480)
    assert points_from_config(yaml.safe_load(text), 800, 480) == points
    p = write_calibration(points, 800, 480, "red-00-00-front-left", tmp_path)
    assert p.name == "red-00-00-front-left.yaml"
    assert yaml.safe_load(p.read_text()) == cfg
    with pytest.raises(FileExistsError):
        write_calibration(points, 800, 480, "red-00-00-front-left", tmp_path)
    write_calibration(points, 800, 480, "red-00-00-front-left", tmp_path, overwrite=True)


@pytest.mark.parametrize("bad", ["", "../oops", "a/b", "a\\b", "../", "a..b", "x" * 81])
def test_camera_names_safe(bad):
    with pytest.raises(ValueError):
        safe_camera_id(bad)


def _synthetic_pair(orientation: str) -> np.ndarray:
    canvas = np.full((450, 650, 3), 75, np.uint8)
    if orientation == "vertical":
        cv2.line(canvas, (150, 0), (100, 449), (245, 245, 245), 12)
        cv2.line(canvas, (355, 0), (305, 449), (15, 15, 15), 14)
    else:
        cv2.line(canvas, (0, 200), (649, 250), (245, 245, 245), 12)
        cv2.line(canvas, (0, 310), (649, 360), (15, 15, 15), 14)
    return canvas


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_auto_zone_handles_camera_orientation(orientation):
    canvas = _synthetic_pair(orientation)
    result = estimate_zone(canvas, method="hybrid")
    assert result.status == "estimated_review_required"
    assert result.zone is not None
    points = points_from_zone(result.zone, canvas.shape[1], canvas.shape[0])
    # For the sanity test the selected lines must form a legitimate region.
    assert config_from_points(points, canvas.shape[1], canvas.shape[0])
    if orientation == "vertical":
        assert abs(points[0][1] - points[1][1]) > .8
    else:
        assert abs(points[0][0] - points[1][0]) > .8


def test_no_paint_never_claims_geometry():
    blank = np.full((200, 320, 3), 60, np.uint8)
    result = estimate_zone(blank, method="hybrid")
    assert result.zone is None
    assert result.status.startswith("uncertain_")
