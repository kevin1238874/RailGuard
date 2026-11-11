"""Generate an artificial THREE-SEQUENCE smoke test, never research ground truth."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np

SEQUENCES = ("red-00-00-front-left", "red-01-00-front-left", "red-04-00-front-left")


def build(root: Path) -> None:
    for sequence in SEQUENCES:
        boxes_dir = root / "annotations" / "anno" / "unified" / "bboxes" / sequence
        params_dir = root / "annotations" / "anno" / "unified" / "params" / sequence
        img_dir = root / "images" / sequence
        cfg_dir = root / "configs"
        for d in (boxes_dir, params_dir, img_dir, cfg_dir):
            d.mkdir(parents=True, exist_ok=True)
        (cfg_dir / f"{sequence}.yaml").write_text(
            "# SYNTHETIC GEOMETRY, NOT REAL RAILWAY ANNOTATION\n"
            "coordinates: pixels\nzone:\n"
            "  safety_line: [[0, 60], [159, 60]]\n"
            "  platform_edge: [[0, 95], [159, 95]]\n", encoding="utf-8")
        for frame in (0, 15):
            canvas = np.zeros((128, 160, 3), np.uint8)
            canvas[:] = (60, 60, 60)
            cv2.line(canvas, (0, 60), (159, 60), (0, 230, 230), 2)
            cv2.line(canvas, (0, 95), (159, 95), (230, 160, 0), 2)
            if frame == 0:
                # Deliberately unsafe (person's box bottom-center at y=82).
                cv2.rectangle(canvas, (45, 30), (70, 82), (50, 120, 200), 2)
                (boxes_dir / "00000.csv").write_text("45,30,70,82,adult,101\n", encoding="utf-8")
                (params_dir / "00000.csv").write_text("0,True\n", encoding="utf-8")
            else:
                # Safe but visible person in frame 15.
                cv2.rectangle(canvas, (95, 20), (120, 50), (50, 120, 200), 2)
                (boxes_dir / "00015.csv").write_text("95,20,120,50,adult,102\n", encoding="utf-8")
                (params_dir / "00015.csv").write_text("0,False\n", encoding="utf-8")
            cv2.imwrite(str(img_dir / f"{frame:05d}.png"), canvas)
    print(f"Generated artificial demo in {root} (DO NOT treat as real RailEye3D data)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    build(parser.parse_args().out)
