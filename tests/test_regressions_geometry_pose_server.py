"""Regression tests for the geometry corridor, explicit Crop-Pose and Studio server fixes."""
from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from railguard.detection import YOLODetector
from railguard.geometry import HazardZone, Segment
from railguard.pipeline import draw

SHAPE = (300, 300)


def _rotate(point, quarter_turns, size=299.0):
    x, y = point
    for _ in range(quarter_turns % 4):
        x, y = size - y, x
    return (x, y)


def _zone(quarter_turns):
    # Base layout: horizontal L at y=100, E at y=150; safe side is above L.
    l_line = [_rotate(p, quarter_turns) for p in ((10, 100), (200, 100))]
    e_line = [_rotate(p, quarter_turns) for p in ((190, 150), (5, 150))]
    return HazardZone(Segment(*l_line), Segment(*e_line))


# (point in the base layout, expected status, expected reason)
CASES = [
    ((50, 30), "SAFE", "safe_side_of_safety_line"),     # far on the safe side
    ((150, 60), "SAFE", "safe_side_of_safety_line"),
    ((120, 120), "DANGER", "inside_hazard_zone"),       # inside finite Z
    ((250, 125), "DANGER", "inside_hazard_zone"),       # between the lines, past the clicked endpoints
    ((125, 250), "DANGER", "beyond_platform_edge"),
]


@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3])
@pytest.mark.parametrize("point, status, reason", CASES)
def test_corridor_decisions_do_not_depend_on_camera_orientation(quarter_turns, point, status, reason):
    zone = _zone(quarter_turns)
    result = zone.assess_point(_rotate(point, quarter_turns), SHAPE)
    assert (result.status, result.reason) == (status, reason)


def test_horizontal_safe_side_point_is_never_danger():
    """Before the fix, (50, 30) with L/E horizontal was DANGER/inside_hazard_zone."""
    zone = HazardZone(Segment((10, 100), (200, 100)), Segment((190, 150), (5, 150)))
    result = zone.assess_point((50, 30), SHAPE)
    assert result.status == "SAFE" and result.inside_z is False


@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3])
def test_corridor_polygon_agrees_with_half_plane_test(quarter_turns):
    zone = _zone(quarter_turns)
    poly = zone.corridor_polygon(300, 300)
    assert len(poly) >= 3
    rng = np.random.default_rng(0)
    for x, y in rng.uniform(1, 298, size=(400, 2)):
        d_l, d_e = zone.signed_distances((x, y))
        if min(abs(d_l), abs(d_e)) < 1.0:
            continue  # skip boundary rounding
        inside_poly = cv2.pointPolygonTest(poly, (float(x), float(y)), False) > 0
        assert inside_poly == zone.in_corridor((x, y))


def test_converging_perspective_corridor_stops_at_vanishing_point():
    # L and E converge to a vanishing point at (150, 40).
    zone = HazardZone(Segment((150, 40), (60, 299)), Segment((150, 40), (200, 299)))
    assert zone.in_corridor((130, 250))
    assert not zone.in_corridor((150, 10))  # beyond the vanishing point
    assert zone.assess_point((30, 250), SHAPE).status == "SAFE"


def test_draw_handles_horizontal_zone():
    image = np.zeros((300, 300, 3), np.uint8)
    out = draw(image, [], _zone(0))
    assert out.shape == image.shape
    assert out[30, 50].sum() == 0          # safe side is not painted as hazard
    assert out[125, 120].sum() > 0         # corridor is painted


# ---------------------------------------------------------------- Crop-Pose

class _Box:
    def __init__(self, bbox, confidence=.9, cls=0):
        self.xyxy = np.array([bbox], dtype=float)
        self.conf = np.array([confidence], dtype=float)
        self.cls = np.array([cls], dtype=float)


def _kp(*people_ankles):
    xy = np.ones((len(people_ankles), 17, 2), dtype=float) * 5
    conf = np.ones((len(people_ankles), 17), dtype=float) * .9
    for index, (left, right) in enumerate(people_ankles):
        xy[index, 15], xy[index, 16] = left, right
    return SimpleNamespace(xy=xy, conf=conf)


class _PoseModel:
    """Full image: one pose-only person. Crops: a bystander first, then the target."""

    def __init__(self):
        self.calls = []

    def predict(self, **opts):
        image = opts["source"]
        self.calls.append(image.shape[:2])
        if image.shape[:2] == (200, 200):
            return [SimpleNamespace(boxes=[_Box((150, 20, 170, 80))],
                                    keypoints=_kp(((155, 75), (165, 76))))]
        h, w = image.shape[:2]
        # Bystander at the crop's left edge listed first; target fills the crop.
        return [SimpleNamespace(
            boxes=[_Box((0, 0, 4, 10)), _Box((3, 3, w - 3, h - 3))],
            keypoints=_kp(((1, 8), (2, 9)), ((w / 2 - 2, h - 20), (w / 2 + 2, h - 20))))]


class _DetModel:
    def predict(self, **opts):
        return [SimpleNamespace(boxes=[_Box((40, 60, 70, 140)), _Box((151, 21, 169, 80))],
                                keypoints=None)]


def test_crop_pose_is_opt_in_and_requires_explicit_det_model():
    plain = YOLODetector("pose", weights="mock.pt", model=_PoseModel())
    people = plain.predict(np.zeros((200, 200, 3), np.uint8))
    assert len(people) == 1 and plain.last_config["crop_pose"] is False
    with pytest.raises(ValueError):
        YOLODetector("pose", weights="mock.pt", model=_PoseModel(), enable_crop_pose=True)


def test_crop_pose_keeps_pose_only_people_and_picks_matching_crop_person():
    detector = YOLODetector("pose", weights="mock.pt", model=_PoseModel(),
                            enable_crop_pose=True, det_model=_DetModel(),
                            crop_pose_det_weights="det.pt")
    people = detector.predict(np.zeros((200, 200, 3), np.uint8))
    # The direct-pose person (matched to the 2nd det box) is kept, not replaced.
    direct = [p for p in people if p.detector == "yolo_pose"]
    added = [p for p in people if p.detector == "yolo_pose_crop"]
    assert len(direct) == 1 and len(added) == 1
    assert direct[0].bbox == (150, 20, 170, 80)
    # The added person's ankles come from the target, not the first (bystander) result.
    target = added[0]
    assert target.bbox == (40, 60, 70, 140)
    points, source = target.foot_points("pose")
    assert source == "two_ankles"
    assert all(40 <= x <= 70 and 120 <= y <= 140 for x, y in points)
    assert target.metadata["crop_pose"] is True
    assert target.metadata["crop_pose_det_weights"] == "det.pt"


# ---------------------------------------------------------------- Server

@pytest.fixture
def studio_url():
    from railguard.server import RailGuardHandler
    server = ThreadingHTTPServer(("127.0.0.1", 0), RailGuardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _post(url, payload, content_type="application/json"):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": content_type}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, dict(response.headers), json.loads(response.read())
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), json.loads(err.read() or b"{}")


def test_server_serves_studio_without_wildcard_cors(studio_url):
    with urllib.request.urlopen(studio_url + "/", timeout=10) as response:
        assert response.status == 200
        assert "Access-Control-Allow-Origin" not in response.headers
    image = np.zeros((40, 40, 3), np.uint8)
    b64 = base64.b64encode(cv2.imencode(".png", image)[1].tobytes()).decode()
    status, headers, body = _post(studio_url + "/api/auto-zone", {"image": b64})
    assert status == 200 and "Access-Control-Allow-Origin" not in headers
    assert "status" in body


def test_server_rejects_bad_images_and_non_json(studio_url):
    status, _, body = _post(studio_url + "/api/auto-zone", {"image": base64.b64encode(b"nope").decode()})
    assert status == 500 and "decode" in body["error"]
    status, _, _ = _post(studio_url + "/api/auto-zone", {"image": "x"}, content_type="text/plain")
    assert status == 415
