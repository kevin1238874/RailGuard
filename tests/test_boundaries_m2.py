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



def _make_platform_scene(
    width: int = 360,
    height: int = 360,
    mirror: bool = False,
    stripe_dark_border: bool = False,
    with_train_across_trench: bool = False,
    lime_train: bool = False,
) -> "np.ndarray":
    import cv2
    import numpy as np

    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Base: concrete platform (x in [0, 220]) and dark track trench (x in [220, 360])
    img[:, :220] = (175, 175, 175)
    img[:, 220:] = (26, 26, 26)
    # Tactile yellow stripe at x in [100, 118]
    img[:, 100:118] = (45, 205, 235)
    if stripe_dark_border:
        img[:, 118:126] = (22, 22, 22)
    if with_train_across_trench:
        if lime_train:
            # Lime-yellow-green car body (H ~ 35 in OpenCV HSV) beyond dark trench
            img[:, 270:] = (50, 210, 185)
        else:
            img[:, 270:] = (205, 205, 205)
    if mirror:
        img = cv2.flip(img, 1)
    return img


def test_narrow_dark_stripe_border_does_not_steal_edge() -> None:
    import numpy as np
    from railguard.boundaries import estimate_zone

    img = _make_platform_scene(stripe_dark_border=True)
    res = estimate_zone(img, "lab_canny_hough", preprocess="raw")
    assert res.zone is not None, f"Unexpected status {res.status}"
    mid_y = 180.0
    l_x = np.interp(mid_y, [res.zone.safety_line.a[1], res.zone.safety_line.b[1]],
                    [res.zone.safety_line.a[0], res.zone.safety_line.b[0]])
    e_x = np.interp(mid_y, [res.zone.platform_edge.a[1], res.zone.platform_edge.b[1]],
                    [res.zone.platform_edge.a[0], res.zone.platform_edge.b[0]])
    assert 92.0 <= l_x <= 122.0
    assert 205.0 <= e_x <= 235.0


def test_mirrored_platform_orientation_right_side_platform() -> None:
    from railguard.boundaries import estimate_zone

    img = _make_platform_scene(mirror=True)
    res = estimate_zone(img, "lab_canny_hough", preprocess="raw")
    assert res.zone is not None, f"Unexpected status {res.status}"
    l_mid_x = 0.5 * (res.zone.safety_line.a[0] + res.zone.safety_line.b[0])
    e_mid_x = 0.5 * (res.zone.platform_edge.a[0] + res.zone.platform_edge.b[0])
    assert l_mid_x > e_mid_x
    assert 235.0 <= l_mid_x <= 268.0
    assert 130.0 <= e_mid_x <= 152.0


def test_lime_train_distractor_and_paint_mask_exclusion() -> None:
    import cv2
    import numpy as np
    from railguard.boundaries import estimate_zone, paint_mask

    img = _make_platform_scene(mirror=True, with_train_across_trench=True, lime_train=True)
    # Lime train is mirrored onto left side x in [0, 90); real stripe is x in [242, 260)
    mask = paint_mask(img, "both", False, "hsv")
    assert float(np.mean(mask[:, :80] > 0)) < 0.05
    res = estimate_zone(img, "hsv_canny_hough", preprocess="raw")
    assert res.zone is not None, f"Unexpected status {res.status}"
    l_mid_x = 0.5 * (res.zone.safety_line.a[0] + res.zone.safety_line.b[0])
    assert l_mid_x >= 230.0


def test_interior_trench_rejects_cross_trench_corridor() -> None:
    import cv2
    import numpy as np
    from railguard.boundaries import _has_interior_trench
    from railguard.geometry import Segment

    img = _make_platform_scene(with_train_across_trench=True, lime_train=False)
    lab_l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(float)
    p10, p95 = np.percentile(lab_l, (10, 95))
    contrast = max(float(p95 - p10), 20.0)
    l_seg = Segment((100.0, 20.0), (100.0, 340.0))
    e_true = Segment((220.0, 20.0), (220.0, 340.0))
    e_across = Segment((285.0, 20.0), (285.0, 340.0))
    assert not _has_interior_trench(lab_l, l_seg, e_true, float(p10), contrast)
    assert _has_interior_trench(lab_l, l_seg, e_across, float(p10), contrast)


def test_zone_evidence_respects_vp_clipped_span() -> None:
    import numpy as np
    from railguard.boundaries import _zone_evidence
    from railguard.geometry import Segment

    # Top half [0..140] is pitch dark with crossed extrapolated lines;
    # overlapping VP-clipped span [150..299] encloses bright platform interior (gray=200).
    gray = np.zeros((300, 300), dtype=np.uint8)
    gray[150:, :] = 200
    l_clipped = Segment((120.0, 150.0), (80.0, 299.0))
    e_clipped = Segment((140.0, 150.0), (220.0, 299.0))
    ze = _zone_evidence(gray, l_clipped, e_clipped, image_median=100.0)
    assert ze is not None
    gap, med_bright, light_frac = ze
    assert gap > 20.0
    assert med_bright >= 190.0
    assert light_frac == 1.0


def test_asymmetric_cross_trench_trap_and_trench_ext_metrics() -> None:
    import numpy as np
    from railguard.boundaries import _evaluate_trench_drop, _has_interior_trench
    from railguard.geometry import Segment

    # Platform bright on left [0..111], deep dark track gap [112..145], dark train wall [146..240]
    lab_l = np.full((240, 240), 185.0, dtype=np.float64)
    lab_b = np.full((240, 240), 128.0, dtype=np.float64)
    lab_l[:, 112:146] = 20.0
    lab_l[:, 146:] = 75.0
    mask = np.zeros((240, 240), dtype=np.uint8)

    l_curb = Segment((86.0, 20.0), (86.0, 220.0))
    e_train = Segment((185.0, 20.0), (185.0, 220.0))
    assert _has_interior_trench(lab_l, l_curb, e_train, 20.0, 165.0)

    # Opposite ordering across track trench rebounds onto the bright platform at 0.16*d (=38.4px) past E (x~107.6)
    l_win = Segment((180.0, 20.0), (180.0, 220.0))
    e_wall = Segment((146.0, 20.0), (146.0, 220.0))
    tr = _evaluate_trench_drop(lab_l, lab_b, mask, l_win, e_wall, 165.0, p10=20.0)
    assert tr["med_ext"] is not None and tr["med_ext"] >= 0.85


def test_left_train_warm_stone_coping_and_dark_floor_m2() -> None:
    import numpy as np
    from railguard.boundaries import estimate_zone, paint_mask

    h, w = 360, 240
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Left train underframe/body, dark track gap, warm pinkish-beige granite coping, bright stripe, dark grey floor (L~70)
    ys = np.arange(h, dtype=np.float64)[:, None]
    xs = np.arange(w, dtype=np.float64)[None, :]
    x_gap_l = 6.0 + (92.0 - 6.0) * (1.0 - ys / (h - 1.0))
    x_gap_r = 32.0 + (108.0 - 32.0) * (1.0 - ys / (h - 1.0))
    x_str_l = 166.0 + (120.0 - 166.0) * (1.0 - ys / (h - 1.0))
    x_str_r = x_str_l + 14.0

    # Safe side dark-grey concourse floor (~BGR 65, 68, 70 -> LAB L~70)
    img[:, :] = (65, 68, 70)
    # Train lower body/shadow on far left
    train_mask = xs < x_gap_l
    img[np.broadcast_to(train_mask, (h, w))] = (48, 54, 62)
    # Deep dark gap E
    gap_mask = (xs >= x_gap_l) & (xs < x_gap_r)
    img[np.broadcast_to(gap_mask, (h, w))] = (8, 10, 14)
    # Warm beige granite coping between gap and stripe
    coping_mask = (xs >= x_gap_r) & (xs < x_str_l)
    img[np.broadcast_to(coping_mask, (h, w))] = (100, 122, 155)
    # Bright white-yellow safety stripe L
    stripe_mask = (xs >= x_str_l) & (xs < x_str_r)
    img[np.broadcast_to(stripe_mask, (h, w))] = (150, 200, 232)

    ml = paint_mask(img, "yellow", False, "lab")
    assert float((ml > 0).mean()) < 0.20

    res = estimate_zone(img, method="lab_canny_hough")
    assert res.zone is not None
    # Platform edge E must be on the left (train gap), safety line L on the right (stripe), never inverted
    l_mid_x = res.zone.safety_line.midpoint[0]
    e_mid_x = res.zone.platform_edge.midpoint[0]
    assert e_mid_x < l_mid_x
    assert abs(e_mid_x - 62.0) < 20.0
    assert abs(l_mid_x - 144.0) < 20.0


def test_left_train_white_ceiling_above_vp_jpeg_m2() -> None:
    """Bright white ceiling/wall above VP separated by a dark gap must not trigger premature-crossing drop."""
    import numpy as np
    from railguard.boundaries import estimate_zone

    h, w = 360, 240
    img = np.full((h, w, 3), (65, 68, 70), dtype=np.uint8)
    ys = np.arange(h, dtype=np.float64)[:, None]
    xs = np.arange(w, dtype=np.float64)[None, :]
    x_gap_l = 6.0 + (92.0 - 6.0) * (1.0 - ys / (h - 1.0))
    x_gap_r = 32.0 + (108.0 - 32.0) * (1.0 - ys / (h - 1.0))
    x_str_l = 166.0 + (120.0 - 166.0) * (1.0 - ys / (h - 1.0))
    x_str_r = x_str_l + 14.0

    rows = ys >= 82
    img[np.broadcast_to((xs < x_gap_l) & rows, (h, w))] = (48, 54, 62)
    img[np.broadcast_to((xs >= x_gap_l) & (xs < x_gap_r) & rows, (h, w))] = (8, 10, 14)
    img[np.broadcast_to((xs >= x_gap_r) & (xs < x_str_l) & rows, (h, w))] = (100, 122, 155)
    img[np.broadcast_to((xs >= x_str_l) & (xs < x_str_r) & rows, (h, w))] = (150, 200, 232)
    # Dark canopy/crowd gap immediately above VP, bright white ceiling above that
    img[58:82, :] = (28, 30, 34)
    img[10:55, 85:175] = (232, 232, 236)

    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert decoded is not None

    for method in ("lab_canny_hough", "hybrid_region_evidence"):
        res = estimate_zone(decoded, method=method)
        assert res.zone is not None, f"{method} abstained: {res.status}"
        assert res.zone.platform_edge.midpoint[0] < res.zone.safety_line.midpoint[0]


def test_train_suppression_contract_and_mask_shape_validation() -> None:
    import pytest
    from railguard.boundaries import estimate_zone

    image = _make_platform_scene()
    default = estimate_zone(image, method="lab_canny_hough")
    assert default.diagnostics["train_suppression"] == "structural"
    with pytest.raises(ValueError, match="same height and width"):
        estimate_zone(image, method="lab_canny_hough", train_suppression="yolo_seg",
                      exclusion_mask=np.zeros((20, 20), np.uint8))
    with pytest.raises(ValueError, match="train_suppression"):
        estimate_zone(image, method="lab_canny_hough", train_suppression="bbox")


def test_train_core_rejects_l_not_platform_edge() -> None:
    from railguard.boundaries import _prepare_train_exclusion, _train_overlap_metrics
    from railguard.geometry import Segment

    mask = np.zeros((240, 320), np.uint8)
    mask[:, 35:145] = 1
    context = _prepare_train_exclusion(mask, 240, 320, "yolo_seg")
    l_inside = Segment((90.0, 10.0), (90.0, 230.0))
    edge = Segment((220.0, 10.0), (220.0, 230.0))
    metrics = _train_overlap_metrics(context, l_inside, edge)
    assert metrics["rejected"] is True
    # E overlap remains diagnostic and cannot independently reject a valid L.
    l_clear = Segment((180.0, 10.0), (180.0, 230.0))
    e_inside = Segment((90.0, 10.0), (90.0, 230.0))
    edge_metrics = _train_overlap_metrics(context, l_clear, e_inside)
    assert edge_metrics["e_train_core_overlap"] > 0.5
    assert edge_metrics["rejected"] is False


def test_train_overlap_metrics_bounds_and_boundary_confirmation() -> None:
    from railguard.boundaries import _prepare_train_exclusion, _train_overlap_metrics, _train_score_adjustment
    from railguard.geometry import Segment

    mask = np.zeros((240, 320), np.uint8)
    mask[:, 215:] = 1
    ctx = _prepare_train_exclusion(mask, 240, 320, "yolo_seg")
    assert ctx["valid"] is True

    # Out-of-bounds extension should clip safely and confirm E along the train boundary
    l_seg = Segment((120.0, -20.0), (120.0, 260.0))
    e_edge = Segment((212.0, -20.0), (212.0, 260.0))
    ov_edge = _train_overlap_metrics(ctx, l_seg, e_edge)
    assert ov_edge["valid_stations"] >= 8
    assert ov_edge["outside_train_overlap"] >= 0.90
    assert ov_edge["train_boundary_confirmed"] is True
    assert ov_edge["rejected"] is False
    assert _train_score_adjustment(ov_edge) > 0.0

    # E sitting deep inside the train core must receive a negative adjustment
    e_deep = Segment((265.0, 10.0), (265.0, 230.0))
    ov_deep = _train_overlap_metrics(ctx, l_seg, e_deep)
    assert ov_deep["e_train_core_overlap"] >= 0.90
    assert _train_score_adjustment(ov_deep) < 0.0


def test_brick_pavers_do_not_trigger_false_rib_cascade() -> None:
    from railguard.boundaries import estimate_zone

    img = _make_platform_scene(width=360, height=360)
    # Add light horizontal stone/brick joint lines on the safe platform interior
    for y in range(18, 360, 22):
        cv2.line(img, (0, y), (98, y), (145, 145, 145), 2)
    res = estimate_zone(img, method="lab_canny_hough", train_suppression="structural")
    assert res.zone is not None, f"Unexpected status {res.status}"
    assert res.diagnostics["scene_rib_dominant"] is False
    assert int(res.diagnostics["structural_rib_rejections"]) < 5


def test_hybrid_yolo_seg_lazy_fallback_and_trench_rescue(monkeypatch) -> None:
    import railguard.train_mask as tm
    from railguard.boundaries import estimate_zone

    # Scene with a yellow safety stripe and a bright train body flush against E (no dark drop)
    h, w = 360, 360
    img = np.full((h, w, 3), (175, 175, 175), dtype=np.uint8)
    img[:, 100:118] = (45, 205, 235)
    img[:, 215:] = (182, 184, 182)
    img[:110, 220:] = (95, 95, 95)
    cv2.line(img, (215, 0), (215, h - 1), (115, 115, 115), 2)

    # Without YOLO prior, negligible drop outside E should abstain
    no_yolo = estimate_zone(img, method="lab_canny_hough", train_suppression="structural")
    assert no_yolo.zone is None

    train_mask = np.zeros((h, w), dtype=bool)
    train_mask[:, 220:] = True

    class _StubProvider:
        def predict(self, _image):
            return tm.TrainMaskResult(
                mask=train_mask,
                metadata={"provider": "yolo_seg", "valid": True, "warm_inference_ms": 12.3, "fallback_reason": None},
            )

    monkeypatch.setattr(tm, "get_default_train_mask_provider", lambda: _StubProvider())
    rescued = estimate_zone(img, method="lab_canny_hough", train_suppression="hybrid_yolo_seg")
    assert rescued.zone is not None, f"Expected YOLO train boundary rescue, got {rescued.status}"
    assert rescued.diagnostics["train_prior_valid"] is True
    assert rescued.diagnostics["exclusion_metadata"]["warm_inference_ms"] == 12.3
    assert rescued.diagnostics["semantic_profile"]["trench"].get("yolo_train_boundary_rescued") is True


def test_dark_inset_fallback_and_structural_never_imports_torch() -> None:
    import subprocess
    import sys
    from railguard.boundaries import estimate_zone

    # Low-chroma dark granite inset strip flanked by bright floor on both sides + crisp pit drop
    h, w = 360, 360
    img = np.full((h, w, 3), (175, 175, 175), dtype=np.uint8)
    img[:, 115:131] = (88, 88, 88)
    img[:, 225:] = (24, 24, 24)
    res = estimate_zone(img, method="lab_canny_hough", train_suppression="structural")
    assert res.zone is not None, f"Expected dark_inset fallback accept, got {res.status}"
    assert res.diagnostics.get("fallback_pass") == "dark_inset"
    l_mid_x = res.zone.safety_line.midpoint[0]
    e_mid_x = res.zone.platform_edge.midpoint[0]
    assert 105.0 <= l_mid_x <= 140.0
    assert 215.0 <= e_mid_x <= 238.0

    # Verify fresh subprocess in default structural mode never imports torch or ultralytics
    proc = subprocess.run(
        [
            sys.executable, "-c",
            "import sys, numpy as np; from railguard.boundaries import estimate_zone; "
            "estimate_zone(np.full((64, 64, 3), 128, np.uint8), train_suppression='structural'); "
            "assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules",
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_yolo_train_core_presuppression_and_train_boundary_proposal() -> None:
    from railguard.boundaries import _prepare_train_exclusion, _train_overlap_metrics, estimate_zone
    from railguard.geometry import Segment

    # Left train (x <= 125) with strong internal bright orange sill & door frame edges;
    # flush low-contrast train/platform seam at x = 125 and yellow safety line on right.
    h, w = 480, 360
    img = np.full((h, w, 3), (170, 170, 170), dtype=np.uint8)
    # Yellow safety stripe from (195, 60) -> (250, 479)
    cv2.line(img, (195, 60), (250, 479), (40, 200, 235), 14)
    # Dark pit seam adjacent to left train edge: (165, 60) -> (110, 479)
    pts_train = np.array([[0, 0], [175, 0], [110, h - 1], [0, h - 1]], dtype=np.int32)
    cv2.fillPoly(img, [pts_train], (155, 155, 155))
    # Bright door frame and orange interior sill strictly inside train_core
    cv2.line(img, (48, 0), (48, h - 1), (245, 245, 245), 6)
    cv2.line(img, (78, 120), (35, h - 1), (35, 165, 245), 10)
    # Subtle curb shadow along train perimeter
    cv2.line(img, (175, 0), (110, h - 1), (95, 95, 95), 4)

    train_mask = np.zeros((h, w), dtype=bool)
    cv2.fillPoly(train_mask.view(np.uint8), [pts_train], 1)
    res = estimate_zone(
        img,
        method="hybrid_region_evidence",
        train_suppression="yolo_seg",
        exclusion_mask=train_mask,
    )
    assert res.zone is not None, f"Expected zone recovery, got {res.status}"
    assert res.diagnostics.get("train_side") == "left"
    # Platform edge E must align near slanted seam (mid ~142), not vertical interior door x=48
    e_mid_x = res.zone.platform_edge.midpoint[0]
    assert 120.0 <= e_mid_x <= 165.0, f"E landed on wrong edge: {e_mid_x}"

    # Verify slanted-mask crossing vertical line does NOT get train_boundary_confirmed bonus
    ctx = _prepare_train_exclusion(train_mask, h, w, "yolo_seg")
    l_vert = Segment((250.0, 20.0), (250.0, h - 1.0))
    e_cross = Segment((140.0, 20.0), (140.0, h - 1.0))
    ov_cross = _train_overlap_metrics(ctx, l_vert, e_cross)
    assert ov_cross["train_boundary_confirmed"] is False


def test_two_zone_foreground_anchor_resists_far_horizon_outlier() -> None:
    from railguard.boundaries import estimate_zone

    # True stripe runs from (190, 60) -> (250, 479); true edge (175, 60) -> (110, 479)
    h, w = 480, 360
    img = np.full((h, w, 3), (175, 175, 175), dtype=np.uint8)
    pit = np.array([[0, 60], [175, 60], [110, h - 1], [0, h - 1]], dtype=np.int32)
    cv2.fillPoly(img, [pit], (28, 28, 28))
    cv2.line(img, (190, 60), (250, h - 1), (45, 205, 238), 14)
    # Inject noisy bright yellow/white reflection in far upper region only (y < 0.35h)
    far_noise = np.array([[205, 0], [238, 0], [218, int(0.30 * h)], [188, int(0.30 * h)]], dtype=np.int32)
    cv2.fillPoly(img, [far_noise], (55, 215, 245))

    res = estimate_zone(img, method="hybrid_region_evidence", train_suppression="structural")
    assert res.zone is not None, f"Unexpected status {res.status}"
    # Check bottom foreground x of safety line stays true (~243-252, safe_side_edge shifted slightly right)
    pts_l = sorted([res.zone.safety_line.a, res.zone.safety_line.b], key=lambda p: p[1])
    x_bot = pts_l[-1][0]
    assert 238.0 <= x_bot <= 265.0, f"Foreground L pulled off track at bottom: {x_bot}"


def test_vp_horizon_penalty_adapts_to_landscape_aspect_ratio() -> None:
    from railguard.boundaries import Segment, _vp_horizon_penalty

    line = Segment((490.0, 140.0), (526.0, 407.0))
    assert _vp_horizon_penalty(0.34 * 408.0, line, 408, width=612) == 0.0
    assert _vp_horizon_penalty(0.34 * 408.0, line, 408, width=300) > 65.0


def test_landscape_ribbed_tactile_fg_tracking() -> None:
    from railguard.boundaries import estimate_zone

    # Landscape scene with true horizon around y~139 (0.34*h) and tapered, ribbed yellow tactile tiles
    h, w = 408, 612
    img = np.full((h, w, 3), (168, 168, 168), dtype=np.uint8)
    pit = np.array([[0, 142], [486, 142], [348, h - 1], [0, h - 1]], dtype=np.int32)
    cv2.fillPoly(img, [pit], (28, 28, 28))
    # Tapered yellow tactile block from (492..499) at y=145 to (472..526) at y=407, split by dark grooves
    tactile = np.array([[492, 145], [499, 145], [526, h - 1], [472, h - 1]], dtype=np.int32)
    cv2.fillPoly(img, [tactile], (45, 205, 238))
    for frac in (0.20, 0.42, 0.64, 0.84):
        x_top = int(round(492 + frac * 7))
        x_bot = int(round(472 + frac * 54))
        cv2.line(img, (x_top, 145), (x_bot, h - 1), (95, 130, 145), 2)

    for method in ("hybrid_region_evidence", "lab_canny_hough"):
        res = estimate_zone(img, method=method, preprocess="raw", train_suppression="structural")
        assert res.zone is not None, f"{method} returned {res.status}"
        pts_l = sorted([res.zone.safety_line.a, res.zone.safety_line.b], key=lambda p: p[1])
        pts_e = sorted([res.zone.platform_edge.a, res.zone.platform_edge.b], key=lambda p: p[1])
        # Platform edge E must stay left of safety line L (platform right of track)
        assert pts_e[-1][0] < pts_l[-1][0]
        # Bottom foreground x_L must follow the tapered outer tactile border (~520..532), not tilt left (<505)
        assert 505.0 <= pts_l[-1][0] <= 540.0, f"{method} foreground L tilted off outer border: {pts_l}"
        assert 482.0 <= pts_l[0][0] <= 515.0, f"{method} upper L drifted onto gray floor: {pts_l}"


def test_edge_anchored_stripe_recovers_closest_stripe_from_skewed_ref() -> None:
    from railguard.boundaries import Segment, _edge_anchored_stripe_candidate

    h, w = 600, 340
    mask = np.zeros((h, w), dtype=np.uint8)
    edge = Segment((175.0, 160.0), (28.0, 595.0))
    # Closest safety stripe on right (platform) side of edge: (182, 175) -> (245, 595), ~14 px wide
    cv2.line(mask, (182, 175), (245, 595), 255, 14)
    # Decoy secondary floor stripe deeper inside platform: (215, 175) -> (315, 595)
    cv2.line(mask, (215, 175), (315, 595), 255, 14)
    # Strongly skewed >14 deg ref_line whose only guarantee is indicating the platform side of `edge`
    skewed_ref = Segment((120.0, 0.0), (295.0, 595.0))

    res = _edge_anchored_stripe_candidate(mask, skewed_ref, edge, "safe_side_edge")
    assert res is not None
    anc_seg, meta = res
    assert meta["stripe_refinement"] == "edge_anchored"
    pts = sorted([anc_seg.a, anc_seg.b], key=lambda p: p[1])
    # Interpolate at y=595 (bottom) and y=200 (upper) -> outer safe-side edge around ~252 and ~192
    x_bot = pts[0][0] + (pts[1][0] - pts[0][0]) * ((595.0 - pts[0][1]) / (pts[1][1] - pts[0][1]))
    x_top = pts[0][0] + (pts[1][0] - pts[0][0]) * ((200.0 - pts[0][1]) / (pts[1][1] - pts[0][1]))
    assert 240.0 <= x_bot <= 262.0, f"Anchored bottom drifted from closest stripe: {x_bot}"
    assert 182.0 <= x_top <= 202.0, f"Anchored top drifted from closest stripe: {x_top}"


def test_portrait_fg_angular_beats_tilted_base_at_midhorizon() -> None:
    from railguard.boundaries import estimate_zone

    # Reproduce moscow_1 geometry: portrait 340x602, train right, horizon vp_y~204 (0.34*h)
    h, w = 602, 340
    img = np.full((h, w, 3), (155, 155, 155), dtype=np.uint8)
    pit = np.array([[185, 204], [w - 1, 204], [w - 1, h - 1], [310, 590]], dtype=np.int32)
    cv2.fillPoly(img, [pit], (25, 25, 25))
    # Coping border + yellow safety stripe: bottom ~(172..192, 601), top ~(182..188, 210)
    stripe = np.array([[182, 212], [188, 212], [192, h - 1], [172, h - 1]], dtype=np.int32)
    cv2.fillPoly(img, [stripe], (45, 205, 238))

    for method in ("lab_canny_hough", "hybrid_region_evidence"):
        res = estimate_zone(img, method=method, preprocess="raw", train_suppression="structural")
        assert res.zone is not None, f"{method} returned {res.status}"
        pts_l = sorted([res.zone.safety_line.a, res.zone.safety_line.b], key=lambda p: p[1])
        # Upper L must stay with the yellow stripe (~180..192), not slant left across the floor (<173)
        assert 175.0 <= pts_l[0][0] <= 195.0, f"{method} upper L skewed left: {pts_l}"
        assert 165.0 <= pts_l[-1][0] <= 190.0, f"{method} foreground L drifted: {pts_l}"


def test_painted_stroke_candidates_tracks_foreshortened_tapered_stripe() -> None:
    from railguard.boundaries import Segment, _painted_stroke_candidates

    h, w = 600, 340
    mask = np.zeros((h, w), dtype=np.uint8)
    # Tapered stripe widening from ~5 px at top to ~24 px at bottom; left/safe border slopes 184 -> 174
    poly = np.array([[184, 90], [189, 90], [198, h - 1], [174, h - 1]], dtype=np.int32)
    cv2.fillPoly(mask, [poly], 255)
    edge = Segment((192.0, 90.0), (338.0, 510.0))
    # Initial parallel-shifted line ~10 px right of outer border and tilted like stripe center/inner edge
    tilted_ref = Segment((194.0, 90.0), (206.0, h - 1))

    cands = _painted_stroke_candidates(mask, tilted_ref, edge, "safe_side_edge")
    fg_cands = [seg for seg, meta in cands if meta.get("stripe_refinement") == "applied_fg_angular"]
    assert fg_cands, f"Expected applied_fg_angular candidate, got {[m for _, m in cands]}"
    fg = fg_cands[0]
    pts = sorted([fg.a, fg.b], key=lambda p: p[1])
    x_bot = pts[0][0] + (pts[1][0] - pts[0][0]) * ((560.0 - pts[0][1]) / (pts[1][1] - pts[0][1]))
    x_mid = pts[0][0] + (pts[1][0] - pts[0][0]) * ((280.0 - pts[0][1]) / (pts[1][1] - pts[0][1]))
    assert 170.0 <= x_bot <= 180.0, f"Foreground angular fit missed tapered near border: x_bot={x_bot}"
    assert 176.0 <= x_mid <= 186.0, f"Foreground angular fit drifted at mid: x_mid={x_mid}"


def test_edge_anchored_ignores_upper_edge_glare_and_reaches_wide_bottom_stripe() -> None:
    from railguard.boundaries import Segment, _edge_anchored_stripe_candidate

    h, w = 600, 340
    mask = np.zeros((h, w), dtype=np.uint8)
    edge = Segment((120.0, 0.0), (339.0, 510.0))
    # Stripe sits up to ~164 px from edge at bottom (> 0.44*340=149.6 px)
    poly = np.array([[182, 220], [188, 220], [192, 560], [175, 560]], dtype=np.int32)
    cv2.fillPoly(mask, [poly], 255)
    # Upper platform-edge glare false positives right beside E
    cv2.line(mask, (152, 85), (178, 165), 255, 6)
    skewed_ref = Segment((166.0, 0.0), (194.0, 599.0))

    res = _edge_anchored_stripe_candidate(mask, skewed_ref, edge, "safe_side_edge")
    assert res is not None
    anc, meta = res
    assert meta["stripe_refinement"] == "edge_anchored"
    pts = sorted([anc.a, anc.b], key=lambda p: p[1])
    x_bot = pts[0][0] + (pts[1][0] - pts[0][0]) * ((550.0 - pts[0][1]) / (pts[1][1] - pts[0][1]))
    x_top = pts[0][0] + (pts[1][0] - pts[0][0]) * ((220.0 - pts[0][1]) / (pts[1][1] - pts[0][1]))
    assert 170.0 <= x_bot <= 181.0, f"Wide-offset stripe truncated at bottom: x_bot={x_bot}"
    assert 178.0 <= x_top <= 188.0, f"Upper edge glare pulled top left: x_top={x_top}"
