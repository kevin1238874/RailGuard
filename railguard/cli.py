"""RailGuard CLI: prepare, inspect, split, calibrate, predict, evaluate, export, train."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import pandas as pd
from .calibration import calibrate
from .config import load_config, get_zone
from .detection import Person, YOLODetector
from .evaluate import evaluate_manifest
from .export_yolo import export_yolo, train_yolo
from .pipeline import analyze, draw
from .raileye import make_manifest, read_frame, summarize_manifest
from .splitting import assert_no_shared_sequences, split_manifest


def _save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Cannot write image to {path}")


def _parse(argv=None):
    parser = argparse.ArgumentParser(prog="railguard", description="CS406 railway platform intrusion monitoring")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Strictly parse RailEye3D unified CSV annotations")
    p.add_argument("--annotations", required=True)
    p.add_argument("--images", help="Directory containing <sequence>/<frame>.jpg")
    p.add_argument("--sequences", nargs="+")
    p.add_argument("--allow-missing-images", action="store_true", help="Annotation audit only")
    p.add_argument("--out", default="data/processed/manifest.csv")

    p = sub.add_parser("audit", help="Summarize class counts, unsafe ratios and frames")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="outputs/data_audit.json")

    p = sub.add_parser("split", help="Train/validation/test by distinct entire sequences")
    p.add_argument("--manifest", required=True)
    p.add_argument("--test-sequence", required=True)
    p.add_argument("--val-sequence", help="Default: deterministic holdout of another whole sequence")
    p.add_argument("--out", default="data/processed/splits.csv")

    p = sub.add_parser("calibrate", help="Click 4 line endpoints in a desktop window")
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True, help="configs/<sequence>.yaml")

    p = sub.add_parser("predict", help="Analyze an image with model or GT boxes")
    p.add_argument("--image", required=True)
    p.add_argument("--config", help="A calibrated per-camera YAML file")
    p.add_argument("--detector", choices=["gt", "yolo", "pose"], default="yolo")
    p.add_argument("--bbox-file", help="Ground-truth bbox CSV for detector=gt")
    p.add_argument("--weights", help="YOLO weights path or official model name")
    p.add_argument("--confidence", type=float, default=0.25)
    p.add_argument("--pose-fallback", choices=["bbox", "abstain"], default="bbox")
    p.add_argument("--auto-boundary", action="store_true", help="EXPERIMENTAL image-line heuristic")
    p.add_argument("--boundary-method", choices=["color_hough", "canny_hough", "color_ransac", "hybrid"],
                   default="color_hough")
    p.add_argument("--out", default="outputs/prediction.jpg")

    p = sub.add_parser("evaluate", help="Evaluate on held-out split, export errors")
    p.add_argument("--manifest", required=True)
    p.add_argument("--configs", default="configs")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--detector", choices=["gt", "yolo", "pose"], default="gt")
    p.add_argument("--weights")
    p.add_argument("--confidence", type=float, default=0.25)
    p.add_argument("--pose-fallback", choices=["bbox", "abstain"], default="bbox")
    p.add_argument("--auto-boundary", action="store_true")
    p.add_argument("--boundary-method", choices=["color_hough", "canny_hough", "color_ransac", "hybrid"], default="color_hough")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--out", default="outputs/evaluation")

    p = sub.add_parser("export-yolo", help="Optional: generate detector train/val data")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="data/processed/yolo")

    p = sub.add_parser("train", help="Optional supervised detector fine-tuning")
    p.add_argument("--data", required=True)
    p.add_argument("--weights", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    if args.command == "prepare":
        df = make_manifest(args.annotations, args.images, args.sequences,
                           require_images=not args.allow_missing_images)
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(json.dumps({**summarize_manifest(df), "out": str(out)}, indent=2))
    elif args.command == "audit":
        df = pd.read_csv(args.manifest)
        summary = summarize_manifest(df)
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    elif args.command == "split":
        df = pd.read_csv(args.manifest, dtype={"object_id": str})
        df = split_manifest(df, args.test_sequence, args.val_sequence)
        assert_no_shared_sequences(df)
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        summary = df.groupby("split").agg(instances=("frame", "size"), sequences=("sequence", "nunique"))
        print(summary.to_string()); print(f"Saved {out}")
    elif args.command == "calibrate":
        print(f"Saved {calibrate(args.image, args.out)}")
    elif args.command == "predict":
        image = cv2.imread(args.image)
        if image is None:
            raise FileNotFoundError(args.image)
        cfg = load_config(args.config) if args.config else {}
        zone = get_zone(cfg, image.shape[1], image.shape[0])
        if zone is None and not args.auto_boundary:
            raise ValueError("Provide --config with calibrated geometry or explicitly --auto-boundary")
        if args.detector == "gt":
            if not args.bbox_file:
                raise ValueError("--bbox-file is required for --detector gt")
            people = [Person(tuple(r[k] for k in ("x1", "y1", "x2", "y2")), object_id=r["object_id"])
                      for r in read_frame(args.bbox_file)]
        else:
            people = YOLODetector("pose" if args.detector == "pose" else "bbox", args.weights,
                                  args.confidence).predict(image)
        decisions, zone, status = analyze(
            image, people, zone, foot_method="pose" if args.detector == "pose" else "bbox",
            auto_boundary=args.auto_boundary, boundary_method=args.boundary_method,
            paint_color=cfg.get("paint_color", "both"), pose_fallback=args.pose_fallback)
        _save_image(args.out, draw(image, decisions, zone, status))
        result = {"status": status, "people": len(people),
                  "violations": sum(d.intrusion is True for d in decisions),
                  "unknown": sum(d.intrusion is None for d in decisions),
                  "detections": [d.as_dict() for d in decisions]}
        Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    elif args.command == "evaluate":
        result = evaluate_manifest(args.manifest, args.out, args.configs, args.split,
                                   args.detector, args.weights, args.boundary_method,
                                   args.auto_boundary, args.pose_fallback, args.confidence,
                                   args.iou_threshold)
        print(json.dumps(result, indent=2))
    elif args.command == "export-yolo":
        print(export_yolo(args.manifest, args.out))
    elif args.command == "train":
        train_yolo(args.data, args.weights, args.epochs, args.imgsz, args.seed)


if __name__ == "__main__":
    main()
