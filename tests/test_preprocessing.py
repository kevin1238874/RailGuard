import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from railguard.cli import main
from railguard.preprocessing import (
    PREPROCESSORS,
    apply_pipeline,
    clahe,
    compute_quality_flags,
    default_m1_config,
    gamma,
    gaussian_blur,
    hist_eq,
    identity,
    make_contact_sheet,
    median_blur,
    resolve_pipeline,
)
from railguard.temporal import TemporalMedianBackground, temporal_median_background


def _gradient(height=32, width=64):
    row = np.linspace(0, 255, width, dtype=np.uint8)
    return np.repeat(row[None, :, None], height, axis=0).repeat(3, axis=2)


@pytest.mark.parametrize("method,params", [
    (identity, {}),
    (gaussian_blur, {"kernel_size": 3}),
    (median_blur, {"kernel_size": 3}),
    (hist_eq, {}),
    (clahe, {"clip_limit": 2.0, "tile_grid_size": [4, 4]}),
    (gamma, {"gamma": 1.8}),
])
def test_all_methods_preserve_shape_dtype_and_have_no_nan(method, params):
    image = _gradient()
    result = method(image, **params)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.isfinite(result).all()


def test_identity_returns_independent_copy():
    image = _gradient(4, 5)
    result = identity(image)
    assert np.array_equal(result, image)
    assert not np.shares_memory(result, image)


@pytest.mark.parametrize("method", [gaussian_blur, median_blur])
@pytest.mark.parametrize("kernel", [0, 2, -3, 2.5])
def test_blurs_reject_invalid_kernel(method, kernel):
    with pytest.raises(ValueError, match="odd integer"):
        method(_gradient(), kernel_size=kernel)


def test_noise_is_reduced_by_blurs():
    rng = np.random.default_rng(7)
    noisy = np.full((64, 64, 3), 127, dtype=np.uint8)
    noisy = np.clip(noisy.astype(np.int16) + rng.normal(0, 30, noisy.shape), 0, 255).astype(np.uint8)
    assert gaussian_blur(noisy, 5).std() < noisy.std()
    assert median_blur(noisy, 5).std() < noisy.std()


def test_quality_flags_dark_bright_clipped_and_gradient():
    dark = np.zeros((8, 8, 3), dtype=np.uint8)
    clipped = np.full((8, 8, 3), 255, dtype=np.uint8)
    gradient = _gradient()
    dark_flags = compute_quality_flags(dark)
    clipped_flags = compute_quality_flags(clipped)
    gradient_flags = compute_quality_flags(gradient)
    assert dark_flags["dark_pixel_ratio"] == 1.0
    assert clipped_flags["bright_pixel_ratio"] == 1.0
    assert clipped_flags["clipped_highlights_ratio"] == 1.0
    assert clipped_flags["metadata"]["highlight_recovery_claimed"] is False
    assert gradient_flags["dynamic_range"] > 200
    assert gradient_flags["contrast_std"] > 0
    assert all(np.isfinite(value) for key, value in gradient_flags.items()
               if key != "metadata")


@pytest.mark.parametrize("enhancer", [hist_eq, clahe])
def test_luminance_enhancement_preserves_neutrals_and_color_hue(enhancer):
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    image[:, :20] = (80, 80, 80)       # neutral patch
    image[:, 20:40] = (20, 80, 180)    # orange patch
    image[:, 40:60] = (150, 70, 30)    # blue patch
    image[:, 60:] = (40, 150, 60)      # green patch
    result = enhancer(image)

    neutral = result[:, :20].astype(np.int16)
    assert np.max(np.ptp(neutral, axis=2)) <= 2
    before = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    after = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    for x in (30, 50, 70):
        hue_delta = abs(int(before[20, x, 0]) - int(after[20, x, 0]))
        hue_delta = min(hue_delta, 180 - hue_delta)
        # Luminance remapping can move saturated colors near the BGR gamut
        # boundary slightly, but must not produce the large hue rotations seen
        # with independent per-channel equalization.
        assert hue_delta <= 12


def test_gamma_convention_brightens_above_one_and_clamps():
    image = np.full((5, 5, 3), 64, dtype=np.uint8)
    brightened = gamma(image, 2.0)
    darkened = gamma(image, 0.5)
    assert brightened.mean() > image.mean() > darkened.mean()
    endpoints = gamma(np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8), 2.0)
    assert endpoints.min() == 0 and endpoints.max() == 255
    with pytest.raises(ValueError):
        gamma(image, 0)


def test_small_image_and_composable_pipeline():
    image = np.array([[[20, 40, 60], [80, 100, 120]],
                      [[140, 160, 180], [200, 220, 240]]], dtype=np.uint8)
    result, metadata = apply_pipeline(
        image, [{"method": "clahe", "params": {"tile_grid_size": [2, 2]}},
                {"method": "gaussian", "params": {"kernel_size": 3}}],
        return_metadata=True)
    assert result.shape == image.shape and result.dtype == image.dtype
    assert [step["method"] for step in metadata["pipeline"]] == ["clahe", "gaussian"]
    assert metadata["runtime_ms"] >= 0
    assert set(("raw", "identity", "gaussian", "median", "hist_eq", "clahe", "gamma")) <= PREPROCESSORS.keys()


def test_reproducible_config_has_separate_m2_and_m3_branches():
    config = default_m1_config()
    assert config["version"] == 1 and config["seed"] == 0
    assert resolve_pipeline(config, "m2_boundary") != resolve_pipeline(config, "m3_person")
    assert "gamma>1 brightens" in config["gamma_convention"]
    assert json.loads(json.dumps(config)) == config
    assert yaml.safe_load(yaml.safe_dump(config)) == config


def test_temporal_median_removes_transient_foreground_and_preserves_type():
    background = np.full((20, 30, 3), 40, dtype=np.uint8)
    frames = [background.copy() for _ in range(5)]
    frames[2][5:15, 10:20] = 230
    result = TemporalMedianBackground(window_size=5)(frames, ["cam-a"] * 5)
    assert np.array_equal(result, background)
    assert result.shape == background.shape and result.dtype == background.dtype


def test_temporal_sampling_and_recent_window():
    frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (0, 20, 40, 60, 80, 100)]
    result = temporal_median_background(frames, ["camera-1"] * 6, sampling=2, window_size=2)
    assert np.all(result == 60)  # median of sampled recent frames with values 40 and 80


def test_temporal_rejects_single_image_single_frame_and_mixed_cameras():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="single image"):
        temporal_median_background(image, ["cam"])
    with pytest.raises(ValueError, match="at least two"):
        temporal_median_background([image], ["cam"])
    with pytest.raises(ValueError, match="mix"):
        temporal_median_background([image, image], ["cam-a", "cam-b"])


def test_contact_sheet_generation_and_cli(tmp_path: Path):
    image = _gradient(24, 32)
    processed, metadata = apply_pipeline(image, "clahe+gaussian", return_metadata=True)
    sheet = make_contact_sheet(image, processed, metadata["runtime_ms"], panel_size=(120, 80))
    assert sheet.shape == (220, 240, 3)
    assert sheet.dtype == np.uint8

    source = tmp_path / "input.png"
    output = tmp_path / "preview.png"
    assert cv2.imwrite(str(source), image)
    main(["preprocess", "--input", str(source), "--pipeline", "clahe+gaussian",
          "--out", str(output), "--synthetic"])
    assert output.is_file()
    assert output.with_name("preview_processed.png").is_file()
    record = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert record["status"] == "DONE_SYNTHETIC"
    assert record["records"][0]["pipeline"][0]["method"] == "clahe"
