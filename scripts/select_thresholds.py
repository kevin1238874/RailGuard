"""Select thresholds on val, freeze them, then score test once per foot-point method.

    python scripts/select_thresholds.py val    # sweep -> val_sweep.csv, selection.json
    python scripts/select_thresholds.py test   # scores only the frozen selection.json

Scoring is railguard.evaluate.evaluate_manifest, unchanged. Zones and detections are
only cached between runs, so every run sees the same L/E for a given image.

Predeclared rule: highest end-to-end unsafe F1 on val. Ties: fewest parameters changed
from the defaults, then closest imgsz, conf and keypoint threshold to the defaults.
NMS IoU (0.7), matching IoU (0.5) and the boundary method stay fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "threshold_selection"
CACHE = OUT / "cache"
MANIFEST = ROOT / "data" / "processed" / "splits.csv"
BOUNDARY = "hybrid_region_evidence"
NMS = 0.7
# 0.60-0.80 were added (val only, before any test run) because 0.50 won at the grid edge.
CONFS = [0.10, 0.15, 0.25, 0.35, 0.50, 0.60, 0.70, 0.80]
SIZES = [640, 1280]
KPTS = [0.20, 0.35, 0.50]
DEFAULT = {"conf": 0.25, "imgsz": 640, "kpt": 0.35}
METHODS = {"bbox": ("yolo", "bbox"), "ankle_abstain": ("pose", "abstain"),
           "ankle_fallback": ("pose", "bbox")}
WORKERS, THREADS = 4, 4


def _init():
    import cv2
    import torch
    cv2.setNumThreads(THREADS)
    torch.set_num_threads(THREADS)


def _key(image) -> str:
    return hashlib.sha1(image.tobytes()).hexdigest()


def _store(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{time.time_ns()}.tmp")
    tmp.write_bytes(pickle.dumps(value))
    tmp.replace(path)
    return value


def _frames(split: str) -> list[str]:
    import pandas as pd
    from railguard.splitting import evaluation_subset
    df = evaluation_subset(pd.read_csv(MANIFEST, dtype={"object_id": str}), split)
    return [rows.iloc[0]["image"] for _, rows in df.groupby(["sequence", "frame"], sort=True)]


def zone_job(image_file: str):
    import cv2
    from railguard.boundaries import estimate_zone
    image = cv2.imread(image_file)
    path = CACHE / "zones" / f"{_key(image)}.pkl"
    if not path.exists():
        est = estimate_zone(image, method=BOUNDARY, color="both")
        _store(path, (est.zone, est.status))


def detect_job(args):
    detector, imgsz, conf, image_files = args
    import cv2
    from railguard.detection import YOLODetector
    weights = "yolo11n-pose.pt" if detector == "pose" else "yolo11n.pt"
    folder = CACHE / f"{detector}_{imgsz}_{conf}"
    model = None
    for image_file in image_files:
        image = cv2.imread(image_file)
        path = folder / f"{_key(image)}.pkl"
        if path.exists():
            continue
        if model is None:
            model = YOLODetector("pose" if detector == "pose" else "bbox", weights, conf,
                                 None, imgsz=imgsz, iou=NMS)
        _store(path, model.predict(image))
    return folder.name


def score(split: str, method: str, conf: float, imgsz: int, kpt: float, out: Path) -> dict:
    import railguard.evaluate as ev
    import railguard.pipeline as pl
    detector, fallback = METHODS[method]
    folder = CACHE / f"{detector}_{imgsz}_{conf}"

    class CachedDetector:
        def __init__(self, *a, **k):
            pass

        def predict(self, image):
            return pickle.loads((folder / f"{_key(image)}.pkl").read_bytes())

    def cached_zone(image, **_):
        zone, status = pickle.loads((CACHE / "zones" / f"{_key(image)}.pkl").read_bytes())
        return SimpleNamespace(zone=zone, status=status)

    def analyze(*a, **k):
        return pl_analyze(*a, keypoint_confidence=kpt, **k)

    pl_analyze, old_zone, old_det = pl.analyze, pl.estimate_zone, ev.YOLODetector
    pl.estimate_zone, ev.YOLODetector, ev.analyze = cached_zone, CachedDetector, analyze
    try:
        result = ev.evaluate_manifest(
            MANIFEST, out, ROOT / "configs", split, detector,
            "yolo11n-pose.pt" if detector == "pose" else "yolo11n.pt", BOUNDARY, True,
            fallback, conf, 0.5, sample_visualizations=0, imgsz=imgsz, nms_iou=NMS)
    finally:
        pl.estimate_zone, ev.YOLODetector, ev.analyze = old_zone, old_det, pl_analyze
    (out / "config.json").write_text(json.dumps(
        {"method": method, "conf": conf, "imgsz": imgsz, "kpt": kpt, "nms_iou": NMS,
         "boundary": BOUNDARY, "split": split}, indent=2))
    return result


def prepare(pool, split: str, dets: list[tuple]):
    images = _frames(split)
    t = time.time()
    list(pool.map(zone_job, images, chunksize=4))
    print(f"[{split}] zones ready for {len(images)} frames: {time.time() - t:.0f}s", flush=True)
    # Several chunks per detector setting so that all workers stay busy.
    n = 1 if len(images) < 100 else WORKERS
    jobs = [(d, s, c, images[i::n]) for d, s, c in dets for i in range(n)]
    t = time.time()
    for name in pool.map(detect_job, jobs):
        print(f"[{split}] {name} done at {time.time() - t:.0f}s", flush=True)


def distance(row: dict) -> tuple:
    changed = sum(row[k] != DEFAULT[k] for k in DEFAULT)
    return (changed, abs(row["imgsz"] - 640), abs(row["conf"] - 0.25), abs(row["kpt"] - 0.35))


def run_val(pool):
    import pandas as pd
    prepare(pool, "val", list(itertools.product(["yolo", "pose"], SIZES, CONFS)))
    rows = []
    for method in METHODS:
        for imgsz, conf, kpt in itertools.product(SIZES, CONFS, KPTS):
            if method == "bbox" and kpt != DEFAULT["kpt"]:
                continue  # keypoints are not used by the bbox rule
            r = score("val", method, conf, imgsz, kpt,
                      OUT / "val" / f"{method}_{imgsz}_{conf}_{kpt}")
            rows.append({"method": method, "imgsz": imgsz, "conf": conf, "kpt": kpt,
                         **{k: r[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1",
                                              "det_recall", "extra_boxes",
                                              "unknown_pose_instances")}})
    pd.DataFrame(rows).to_csv(OUT / "val_sweep.csv", index=False)
    selection = {}
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        best_f1 = max(r["f1"] for r in group)
        tied = [r for r in group if abs(r["f1"] - best_f1) < 1e-12]
        best = min(tied, key=distance)
        default = next(r for r in group if all(r[k] == DEFAULT[k] for k in DEFAULT))
        selection[method] = {"selected": best, "default": default, "tied_configs": len(tied)}
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection, indent=2))


def run_test(pool):
    selection = json.loads((OUT / "selection.json").read_text())
    chosen = {m: s["selected"] for m, s in selection.items()}
    dets = sorted({(METHODS[m][0], c["imgsz"], c["conf"]) for m, c in chosen.items()})
    prepare(pool, "test", dets)
    for m, c in chosen.items():
        r = score("test", m, c["conf"], c["imgsz"], c["kpt"], OUT / "test" / m)
        print(m, {k: c[k] for k in DEFAULT}, {k: r[k] for k in ("tp", "fp", "fn", "precision",
                                                                "recall", "f1")},
              "frame_f1", r["frame_level_alert_metrics"]["f1"],
              "unknown_frames", r["unknown_frames"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["val", "test"])
    stage = ap.parse_args().stage
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with ProcessPoolExecutor(WORKERS, initializer=_init) as pool:
        (run_val if stage == "val" else run_test)(pool)
    print(f"{stage} finished in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
