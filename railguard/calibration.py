"""Optional desktop interactive calibration: click L0, L1, E0, E1 and save YAML.

Requires a graphical desktop for OpenCV highgui. For headless servers, write
normalized points to a YAML file based on a measured sample image instead.
"""
from __future__ import annotations
from pathlib import Path
import cv2
import yaml
from .geometry import HazardZone, Segment, UNSAFE_SIDE_CONVENTION


def calibrate(image_file: str, output: str) -> Path:
    image = cv2.imread(image_file)
    if image is None:
        raise FileNotFoundError(image_file)
    h, w = image.shape[:2]
    clicks: list[tuple[int, int]] = []
    window = "RailGuard - L0 L1 E0 E1 (unsafe is beyond E); S=save, U=undo, R=reset, ESC=cancel"

    def on_click(event, x, y, _flags, _params):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_click)
    try:
        while True:
            canvas = image.copy()
            for i, point in enumerate(clicks):
                cv2.circle(canvas, point, 6, (40, 40, 240), -1)
                cv2.putText(canvas, ("L0", "L1", "E0", "E1")[i],
                            (point[0] + 8, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (50, 240, 255), 2)
            if len(clicks) >= 2:
                cv2.line(canvas, clicks[0], clicks[1], (0, 255, 255), 2)
            if len(clicks) == 4:
                cv2.line(canvas, clicks[2], clicks[3], (255, 140, 0), 2)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                raise KeyboardInterrupt("Calibration cancelled")
            if key == ord("r"):
                clicks.clear()
            if key == ord("u") and clicks:
                clicks.pop()
            if key == ord("s") and len(clicks) == 4:
                zone = HazardZone(Segment(clicks[0], clicks[1]), Segment(clicks[2], clicks[3]))
                zone.polygon()
                cfg = {"coordinates": "normalized", "zone": {
                    "safety_line": [[round(p[0]/(w-1), 6), round(p[1]/(h-1), 6)] for p in clicks[:2]],
                    "platform_edge": [[round(p[0]/(w-1), 6), round(p[1]/(h-1), 6)] for p in clicks[2:]],
                    "unsafe_side": UNSAFE_SIDE_CONVENTION}}
                target = Path(output)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
                return target
    finally:
        cv2.destroyWindow(window)
