"""Synthetic Phase-P2 coverage; these tests do not validate RailEye3D data."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from railguard.boundaries import M2_METHODS, estimate_zone
from railguard.cli import main
from railguard.geometry import Segment
from railguard.line_models import fit_line_ransac, line_clustering


@pytest.mark.parametrize("slope", [0.0, .65, 12.0])
def test_true_ransac_handles_orientations_outliers_and_seed(slope):
    rng = np.random.default_rng(42)
    if slope < 10:
        x = np.linspace(0, 120, 60)
        clean = np.column_stack((x, 15 + slope*x + rng.normal(0, .35, len(x))))
    else:  # near vertical, represented without slope instability
        y = np.linspace(0, 120, 60)
        clean = np.column_stack((35 + .025*y + rng.normal(0, .35, len(y)), y))
    outliers = rng.uniform((-20, -30), (140, 150), size=(40, 2))
    points = np.vstack((clean, outliers))
    first = fit_line_ransac(points, max_trials=500, residual_threshold=1.5,
                            min_inlier_ratio=.5, random_state=9)
    second = fit_line_ransac(points, max_trials=500, residual_threshold=1.5,
                             min_inlier_ratio=.5, random_state=9)
    assert np.array_equal(first.inlier_mask, second.inlier_mask)
    assert np.allclose(first.model.point, second.model.point)
    assert first.inlier_ratio >= .58
    assert np.median(first.residuals[:len(clean)]) < .7


def test_line_clustering_groups_collinear_segments_without_global_parallel_rule():
    segments = [
        Segment((0, 10), (40, 12)), Segment((45, 12.2), (90, 15)),
        Segment((5, 70), (45, 55)), Segment((50, 53), (90, 35)),
        Segment((25, 0), (25, 30)),
    ]
    clusters = line_clustering(segments, angle_threshold_deg=12, distance_threshold=7)
    assert sorted(len(cluster.segments) for cluster in clusters) == [1, 2, 2]


def _platform_scene(scale=1.0, platform=135, track=25, broken=False):
    h, w = round(300*scale), round(500*scale)
    image = np.full((h, w, 3), platform, np.uint8)
    stripe = [(0, round(90*scale)), (w-1, round(120*scale))]
    edge = [(0, round(190*scale)), (w-1, round(220*scale))]
    if broken:
        for start, end in ((0, .28), (.38, .65), (.74, 1)):
            p0 = (round((w-1)*start), round(stripe[0][1] + 30*scale*start))
            p1 = (round((w-1)*end), round(stripe[0][1] + 30*scale*end))
            cv2.line(image, p0, p1, (245, 245, 245), max(7, round(16*scale)))
    else:
        cv2.line(image, *stripe, (245, 245, 245), max(7, round(16*scale)))
    polygon = np.array([[0, edge[0][1]], [w-1, edge[1][1]], [w-1, h-1], [0, h-1]])
    cv2.fillPoly(image, [polygon.astype(np.int32)], (track, track, track))
    return image


def test_thick_stripe_safe_side_edge_and_center_are_explicit():
    image = _platform_scene()
    safe = estimate_zone(image, method="hybrid_region_evidence",
                         stripe_convention="safe_side_edge")
    center = estimate_zone(image, method="hybrid_region_evidence",
                           stripe_convention="stripe_center")
    assert safe.zone is not None and center.zone is not None
    safe_y = np.mean([safe.zone.safety_line.a[1], safe.zone.safety_line.b[1]])
    center_y = np.mean([center.zone.safety_line.a[1], center.zone.safety_line.b[1]])
    assert safe_y < center_y - 5
    assert safe.diagnostics["stripe_convention"] == "safe_side_edge"
    assert safe.diagnostics["paint_mask_nonzero"] > 0


def test_two_edges_of_one_white_stripe_do_not_fabricate_platform_edge():
    image = np.full((260, 460, 3), 130, np.uint8)
    cv2.line(image, (0, 100), (459, 120), (250, 250, 250), 20)
    result = estimate_zone(image, method="hybrid_region_evidence")
    assert result.zone is None
    assert result.status == "uncertain_no_platform_edge"


@pytest.mark.parametrize("platform,track,stripe", [(190, 45, 250), (72, 12, 180)])
def test_relative_region_evidence_handles_bright_and_dark_lighting(platform, track, stripe):
    image = _platform_scene(platform=platform, track=track)
    # Repaint so the low-light case still has a detectable but non-clipped stripe.
    cv2.line(image, (0, 90), (499, 120), (stripe, stripe, stripe), 16)
    result = estimate_zone(image, method="hybrid_region_evidence")
    assert result.status == "estimated_review_required"
    assert result.diagnostics["two_sided_region"]["reliable"] is True


def test_broken_occluded_line_and_resolution_scaling():
    small = estimate_zone(_platform_scene(.75, broken=True), method="hsv_hough_ransac")
    large = estimate_zone(_platform_scene(1.5, broken=True), method="hsv_hough_ransac")
    assert small.zone is not None and large.zone is not None
    sn = small.candidates[0]["safety_line"]["normalized"]
    ln = large.candidates[0]["safety_line"]["normalized"]
    assert np.max(np.abs(np.asarray(sn) - np.asarray(ln))) < .08
    assert small.candidates[0]["support"]["safety"]["fit"] == "ransac"


def test_method_matrix_is_available_and_has_no_probability_confidence():
    image = _platform_scene()
    for method in M2_METHODS:
        result = estimate_zone(image, method=method, preprocess="raw")
        for candidate in result.candidates or []:
            assert "score" in candidate
            assert "confidence" not in candidate
            assert candidate["estimated_review_required"] is True


def test_inspect_and_benchmark_lines_cli(tmp_path, capsys):
    image_path = tmp_path / "scene.png"
    cv2.imwrite(str(image_path), _platform_scene())
    inspect_path = tmp_path / "inspect.json"
    main(["inspect-lines", "--image", str(image_path), "--method",
          "hybrid_region_evidence", "--out", str(inspect_path), "--synthetic"])
    inspected = json.loads(inspect_path.read_text(encoding="utf-8"))
    assert inspected["validation_status"] == "DONE_SYNTHETIC"
    assert len(inspected["candidates"]) <= 3
    fixtures = tmp_path / "fixtures.json"
    fixtures.write_text(json.dumps([{
        "image": image_path.name, "coordinates": "pixels",
        "safety_line": [[0, 90], [499, 120]],
        "platform_edge": [[0, 190], [499, 220]],
    }]), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.json"
    main(["benchmark-lines", "--fixtures", str(fixtures), "--method",
          "hybrid_region_evidence", "--preprocess", "raw", "--out",
          str(benchmark_path), "--synthetic"])
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    assert benchmark["validation_status"] == "DONE_SYNTHETIC"
    assert benchmark["results"][0]["abstain_rate"] == 0
    assert benchmark["results"][0]["mean_angle_error_deg"] is not None
    capsys.readouterr()

