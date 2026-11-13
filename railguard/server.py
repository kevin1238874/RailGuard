"""Modern lightweight web server for RailGuard Studio Pro (HTML5/Canvas)."""
from __future__ import annotations

import base64
import json
import os
import sys
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
from railguard.zone_editor import config_from_points, points_from_zone

STUDIO_DIR = Path(__file__).parent / "studio"


def _decode_b64_image(data_uri: str) -> np.ndarray:
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    raw = base64.b64decode(data_uri)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


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
                img = _decode_b64_image(req["image"])
                preproc = req.get("preproc", "raw")
                method = req.get("method", "hybrid_region_evidence")
                if preproc != "raw":
                    img = apply_pipeline(img, [preproc])
                res = estimate_zone(img, method=method)
                if res.zone:
                    pts = points_from_zone(res.zone, img.shape[1], img.shape[0])
                    self._send_json({"status": res.status, "points": pts})
                else:
                    self._send_json({"status": res.status, "points": None})
            except Exception as err:
                self._send_json({"error": str(err)}, 500)

        elif self.path == "/api/run-pipeline":
            try:
                img = _decode_b64_image(req["image"])
                h, w = img.shape[:2]
                points = req["points"]
                cfg = config_from_points(points, w, h)
                zone = get_zone(cfg, w, h)

                det_type = req.get("detector", "A YOLO pose")
                conf = float(req.get("confidence", 0.25))
                preproc = req.get("preproc", "raw")
                proc_img = apply_pipeline(img, [preproc]) if preproc != "raw" else img

                if "pose" in det_type.lower():
                    det = YOLODetector("pose", weights="yolo11n-pose.pt", confidence=conf, imgsz=640)
                    people = det.predict(proc_img)
                    foot_m = "pose"
                else:
                    det = YOLODetector("bbox", weights="yolo11n.pt", confidence=conf, imgsz=640)
                    people = det.predict(proc_img)
                    foot_m = "bbox"

                decisions, zone_used, cal_status = analyze(proc_img, people, zone, foot_method=foot_m)
                frame = aggregate_frame(
                    decisions,
                    observation_status="detections_complete" if people else "no_detections",
                    calibration_status=cal_status,
                    calibration_valid=zone_used is not None,
                )
                vis = draw(img, decisions, zone_used, f"RailGuard Studio · Frame: {frame.status}")
                _, png = cv2.imencode(".png", vis)
                b64_out = base64.b64encode(png.tobytes()).decode("ascii")

                self._send_json({
                    "frame": frame.as_dict(),
                    "results": [d.as_dict() for d in decisions],
                    "overlay_image": b64_out,
                })
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
