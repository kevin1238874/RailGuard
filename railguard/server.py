"""Modern lightweight web server for RailGuard Studio Pro (HTML5/Canvas)."""
from __future__ import annotations

import base64
import inspect
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from railguard.boundaries import estimate_zone
from railguard.config import get_zone
from railguard.detection import YOLODetector
from railguard.pipeline import aggregate_frame, analyze, draw
from railguard.preprocessing import apply_pipeline
from railguard.train_mask import YoloSegTrainMaskProvider
from railguard.zone_editor import config_from_points, points_from_zone

STUDIO_DIR = Path(__file__).parent / "studio"


def _decode_b64_image(data_uri: str) -> np.ndarray:
    if not isinstance(data_uri, str) or not data_uri.strip():
        raise ValueError("image payload must be a nonempty base64 string")
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    raw = base64.b64decode(data_uri)
    arr = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("unable to decode image bytes")
    return decoded


_M3_GLARE_ONLY_PREPROCS = {
    "auto",
    "highlight_compress",
    "glare_suppress",
    "exposure_fusion",
    "mertens_fusion",
    "gamma",
    "clahe",
}


def _resolve_preproc(req: dict[str, Any]) -> str:
    value = req.get("preprocess")
    if value is None:
        value = req.get("preproc", "raw")
    return str(value or "raw").strip() or "raw"


def _pipeline_steps(preproc: str) -> list[str | dict[str, Any]]:
    """Expand bare Studio preprocessor names into effective parameterized pipeline steps."""
    if preproc == "gamma":
        return [{"method": "gamma", "params": {"gamma": 0.65}}]
    return [preproc]


def _supports_kwarg(fn, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


_TRAIN_MASK_PROVIDER_CACHE: dict[tuple[Any, ...], YoloSegTrainMaskProvider] = {}
_TRAIN_MASK_PROVIDER_LOCK = threading.RLock()


def _get_train_mask_provider(factory, weights: str, confidence: float,
                             imgsz: int, device: str | None):
    key = (weights, round(confidence, 6), imgsz, device)
    with _TRAIN_MASK_PROVIDER_LOCK:
        provider = _TRAIN_MASK_PROVIDER_CACHE.get(key)
        if provider is None:
            provider = factory(weights=weights, confidence=confidence,
                               imgsz=imgsz, device=device)
            _TRAIN_MASK_PROVIDER_CACHE[key] = provider
        return provider


def handle_auto_zone(
    req: dict[str, Any],
    *,
    train_mask_provider_factory=YoloSegTrainMaskProvider,
) -> dict[str, Any]:
    """Run Studio automatic boundary estimation with unified preprocess & horizon diagnostics."""
    img = _decode_b64_image(req["image"])
    preproc = _resolve_preproc(req)
    method = req.get("method", "hybrid_region_evidence")
    train_suppression = str(req.get("train_suppression", "structural"))
    if train_suppression not in {"off", "structural", "yolo_seg", "hybrid_yolo_seg"}:
        raise ValueError("train_suppression must be off, structural, yolo_seg, or hybrid_yolo_seg")
    # Keep M2 boundary fitting isolated from M3 highlight compression / auto glare modes.
    m2_preproc = "raw" if preproc in _M3_GLARE_ONLY_PREPROCS else preproc
    proc_img = apply_pipeline(img, _pipeline_steps(m2_preproc)) if m2_preproc != "raw" else img
    exclusion_mask = None
    train_prior_metadata: dict[str, Any] = {}
    train_prior_valid = False
    fallback_reason = None
    effective_suppression = train_suppression
    if train_suppression in {"yolo_seg", "hybrid_yolo_seg"}:
        weights = str(req.get("train_weights", "yolo11n-seg.pt"))
        train_confidence = float(req.get("train_confidence", 0.25))
        train_imgsz = int(req.get("train_imgsz", 640))
        train_device = req.get("train_device")
        try:
            provider = _get_train_mask_provider(
                train_mask_provider_factory, weights, train_confidence,
                train_imgsz, train_device,
            )
            with _TRAIN_MASK_PROVIDER_LOCK:
                prediction = provider.predict(img)
            train_prior_metadata = dict(prediction.metadata or {})
            train_prior_valid = bool(train_prior_metadata.get("valid"))
            if train_prior_valid:
                exclusion_mask = prediction.mask
            else:
                fallback_reason = str(train_prior_metadata.get("fallback_reason") or "invalid_train_prior")
                effective_suppression = "structural"
        except Exception as err:
            fallback_reason = f"provider_error:{type(err).__name__}"
            train_prior_metadata = {"valid": False, "error": str(err)}
            effective_suppression = "structural"
    res = estimate_zone(
        proc_img, method=method, train_suppression=effective_suppression,
        exclusion_mask=exclusion_mask,
        exclusion_metadata=train_prior_metadata,
    )
    if res.zone:
        pts = points_from_zone(res.zone, proc_img.shape[1], proc_img.shape[0])
        horizon_y = res.zone.observable_horizon_y(proc_img.shape[1], proc_img.shape[0])
    else:
        pts = None
        horizon_y = None
    return {
        "status": res.status,
        "method": method,
        "preprocess": preproc,
        "preproc": preproc,
        "effective_preprocess": m2_preproc,
        "train_suppression_requested": train_suppression,
        "train_suppression_effective": effective_suppression,
        "train_prior_valid": train_prior_valid,
        "train_prior_metadata": train_prior_metadata,
        "train_prior_fallback_reason": fallback_reason,
        "points": pts,
        "horizon_y": horizon_y,
        "diagnostics": dict(res.diagnostics or {}),
    }


_DETECTOR_CACHE: dict[tuple[Any, ...], YOLODetector] = {}


def _get_detector(detector_factory, method: str, det_kwargs: dict[str, Any]):
    """Reuse loaded YOLO models across Studio requests without bypassing test stubs."""
    if detector_factory is not YOLODetector:
        return detector_factory(method, **det_kwargs)
    key = (
        method,
        str(det_kwargs.get("weights", "")),
        round(float(det_kwargs.get("confidence", 0.25)), 4),
        det_kwargs.get("imgsz", 640),
        bool(det_kwargs.get("enable_crop_pose", True)),
        bool(det_kwargs.get("glare_recovery", False)),
        str(det_kwargs.get("glare_exposure", "highlight_compress")),
        str(det_kwargs.get("cascade_mode", "auto")),
    )
    det = _DETECTOR_CACHE.get(key)
    if det is None:
        det = detector_factory(method, **det_kwargs)
        _DETECTOR_CACHE[key] = det
    return det


def handle_run_pipeline(req: dict[str, Any], *, detector_factory=YOLODetector) -> dict[str, Any]:
    """Run M1->M3->M4 analysis cleanly with crop-pose, foot refinement, glare recovery, and completeness."""
    img = _decode_b64_image(req["image"])
    h, w = img.shape[:2]
    points = req["points"]
    cfg = config_from_points(points, w, h)
    zone = get_zone(cfg, w, h)

    det_type = str(req.get("detector", "A YOLO pose"))
    conf = float(req.get("confidence", 0.25))
    preproc = _resolve_preproc(req)
    enable_crop_pose = bool(req.get("enable_crop_pose", True))
    refine_footpoints = bool(req.get("refine_footpoints", True))
    train_suppression = str(req.get("train_suppression", "structural"))
    if train_suppression not in {"off", "structural", "yolo_seg", "hybrid_yolo_seg"}:
        raise ValueError("train_suppression must be off, structural, yolo_seg, or hybrid_yolo_seg")
    glare_recovery = bool(req.get("glare_recovery", preproc == "auto"))
    high_res_frame = max(h, w) >= 1280
    imgsz = int(req.get("imgsz", 960 if (high_res_frame and (preproc == "auto" or glare_recovery)) else 640))
    glare_exposure = str(req.get("glare_exposure", "gamma06"))
    default_cascade = (
        ("bbox_first" if high_res_frame else "dual_full")
        if glare_recovery else "auto"
    )
    cascade_mode = str(req.get("cascade_mode", default_cascade))
    proc_img = apply_pipeline(img, _pipeline_steps(preproc)) if preproc not in {"raw", "auto"} else img

    if "pose" in det_type.lower():
        method = "pose"
        det_kwargs: dict[str, Any] = {
            "weights": "yolo11n-pose.pt",
            "confidence": conf,
            "imgsz": imgsz,
            "enable_crop_pose": enable_crop_pose,
        }
        foot_m = "pose"
    else:
        method = "bbox"
        det_kwargs = {
            "weights": "yolo11n.pt",
            "confidence": conf,
            "imgsz": imgsz,
        }
        foot_m = "bbox"

    for opt_key, opt_val in (
        ("glare_recovery", glare_recovery),
        ("glare_exposure", glare_exposure),
        ("cascade_mode", cascade_mode),
    ):
        if _supports_kwarg(detector_factory, opt_key):
            det_kwargs[opt_key] = opt_val

    det = _get_detector(detector_factory, method, det_kwargs)
    people = det.predict(proc_img)
    coverage = dict(getattr(det, "last_coverage", {}) or {})
    glare_rois = [list(r) for r in (getattr(det, "last_glare_rois", []) or [])]
    decisions, zone_used, cal_status = analyze(
        proc_img, people, zone, foot_method=foot_m, refine_footpoints=refine_footpoints,
    )
    obs_status = ("detections_partial" if coverage.get("detector_partial")
                  else ("detections_complete" if people else "no_detections"))
    frame = aggregate_frame(
        decisions,
        observation_status=obs_status,
        calibration_status=cal_status,
        calibration_valid=zone_used is not None,
        detector_coverage=coverage or None,
    )
    vis = draw(img, decisions, zone_used, f"RailGuard Studio · Frame: {frame.status}")
    _, png = cv2.imencode(".png", vis)
    b64_out = base64.b64encode(png.tobytes()).decode("ascii")

    return {
        "frame": frame.as_dict(),
        "results": [d.as_dict() for d in decisions],
        "detector_coverage": coverage,
        "glare_rois": glare_rois,
        "imgsz": imgsz,
        "preprocess": preproc,
        "preproc": preproc,
        "glare_recovery": glare_recovery,
        "glare_exposure": glare_exposure,
        "cascade_mode": cascade_mode,
        "enable_crop_pose": enable_crop_pose,
        "refine_footpoints": refine_footpoints,
        "train_suppression_requested": train_suppression,
        "train_suppression_effective": train_suppression,
        "overlay_image": b64_out,
    }


class RailGuardHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict[str, Any], status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index.html"):
            index_file = STUDIO_DIR / "index.html"
            content = index_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")
        try:
            req = json.loads(body)
        except Exception as err:
            self._send_json({"error": f"Invalid JSON: {err}"}, 400)
            return

        if self.path == "/api/auto-zone":
            try:
                self._send_json(handle_auto_zone(req))
            except ValueError as err:
                self._send_json({"error": str(err)}, 400)
            except Exception as err:
                self._send_json({"error": str(err)}, 500)

        elif self.path == "/api/run-pipeline":
            try:
                self._send_json(handle_run_pipeline(req))
            except ValueError as err:
                self._send_json({"error": str(err)}, 400)
            except Exception as err:
                self._send_json({"error": str(err)}, 500)
        else:
            self.send_response(404)
            self.end_headers()


def run_studio_server(port: int = 8502, host: str = "127.0.0.1"):
    server = HTTPServer((host, port), RailGuardHandler)
    print(f"🚀 RailGuard Studio Pro đang chạy tại: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server.")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8502
    run_studio_server(port=port)
