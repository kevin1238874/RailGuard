"""RailGuard CLI: prepare, inspect, split, calibrate, predict, evaluate, export, train."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import pandas as pd
from .boundaries import M1_METHODS, M2_METHODS, estimate_zone, paint_mask
from .calibration import calibrate
from .classic_detectors import AlwaysUnsafeBaseline, HOGSVMDetector, MOG2BlobDetector
from .config import load_config, get_zone
from .detection import Person, YOLODetector
from .evaluate import evaluate_manifest
from .experiments import run_experiment
from .export_yolo import export_yolo, train_yolo
from .geometry import Segment
from .metrics import FAILURE_CATEGORIES
from .pipeline import aggregate_frame, analyze, draw
from .preprocessing import (PREPROCESSORS, apply_pipeline, compute_quality_flags,
                            make_contact_sheet, resolve_pipeline)
from .raileye import (audit_manifest, blocked_data_audit, make_manifest, read_frame,
                      summarize_manifest, write_data_audit)
from .splitting import (PRETRAINED_VAL_TEST, TRAIN_VAL_TEST,
                        assert_no_shared_sequences, split_manifest)


_BOUNDARY_METHOD_CHOICES = tuple(dict.fromkeys(
    ("color_hough", "canny_hough", "color_ransac", "hybrid", *M2_METHODS)))


def _save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Cannot write image to {path}")


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _line_angle_degrees(first: Segment, second: Segment) -> float:
    a = np.asarray(first.b) - np.asarray(first.a)
    b = np.asarray(second.b) - np.asarray(second.a)
    cosine = np.clip(abs(float(np.dot(a, b))) / (np.linalg.norm(a) * np.linalg.norm(b)), 0, 1)
    return float(np.degrees(np.arccos(cosine)))


def _line_distance(first: Segment, second: Segment) -> float:
    def residuals(source: Segment, target: Segment) -> list[float]:
        direction = np.asarray(target.b) - np.asarray(target.a)
        direction = direction / np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]])
        return [abs(float((np.asarray(point) - np.asarray(target.a)) @ normal))
                for point in (source.a, source.b)]
    return float(np.mean(residuals(first, second) + residuals(second, first)))


def _line_overlap(first: Segment, second: Segment) -> float:
    direction = np.asarray(second.b) - np.asarray(second.a)
    target_length = float(np.linalg.norm(direction))
    direction /= target_length
    projected = sorted(float((np.asarray(p) - np.asarray(second.a)) @ direction)
                       for p in (first.a, first.b))
    intersection = max(0.0, min(projected[1], target_length) - max(projected[0], 0.0))
    return intersection / target_length


def _draw_line_candidates(image: np.ndarray, candidates: list[dict]) -> np.ndarray:
    output = image.copy()
    colors = [(0, 255, 0), (0, 200, 255), (255, 100, 0)]
    for candidate, color in zip(candidates, colors):
        for key, thickness in (("safety_line", 3), ("platform_edge", 2)):
            points = candidate[key]["pixels"]
            a, b = (tuple(np.rint(point).astype(int)) for point in points)
            cv2.line(output, a, b, color, thickness, cv2.LINE_AA)
        origin = tuple(np.rint(candidate["safety_line"]["pixels"][0]).astype(int))
        cv2.putText(output, f"#{candidate['rank']}", origin, cv2.FONT_HERSHEY_SIMPLEX,
                    .55, color, 2, cv2.LINE_AA)
    return output


def _run_inspect_lines(args) -> None:
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    started = perf_counter()
    result = estimate_zone(
        image, method=args.method, color=args.paint_color,
        preprocess=args.preprocess, stripe_convention=args.stripe_convention,
        random_state=args.random_state)
    payload = {
        "status": result.status,
        "validation_status": "DONE_SYNTHETIC" if args.synthetic else "IMPLEMENTED_NOT_VALIDATED_ON_REAL_DATA",
        "runtime_ms": round((perf_counter() - started) * 1000, 3),
        "method": args.method, "preprocess": args.preprocess,
        "candidates": result.candidates or [], "diagnostics": result.diagnostics,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.overlay:
        _save_image(args.overlay, _draw_line_candidates(image, result.candidates or []))
    if args.debug_mask:
        space = "lab" if args.method.startswith("lab_") else "hsv"
        _save_image(args.debug_mask, paint_mask(image, args.paint_color, color_space=space))
    print(json.dumps(payload, indent=2))


def _fixture_segment(raw, coordinates: str, width: int, height: int) -> Segment:
    if (isinstance(raw, bool) or not isinstance(raw, (list, tuple))
            or any(isinstance(value, (bool, np.bool_))
                   for point in raw if isinstance(point, (list, tuple)) for value in point)):
        raise ValueError("manual L/E fixtures must contain endpoints, not boolean labels")
    line = Segment.from_list(raw)
    if coordinates == "normalized":
        line = Segment((line.a[0] * (width - 1), line.a[1] * (height - 1)),
                       (line.b[0] * (width - 1), line.b[1] * (height - 1)))
    elif coordinates != "pixels":
        raise ValueError("fixture coordinates must be 'pixels' or 'normalized'")
    return line


def _load_line_fixtures(path: str | Path) -> tuple[list[dict], Path]:
    fixture_path = Path(path)
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    records = raw.get("fixtures", raw) if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ValueError("fixtures JSON must be a list or an object containing 'fixtures'")
    return records, fixture_path.parent


def _run_benchmark_lines(args) -> None:
    fixtures, fixture_root = _load_line_fixtures(args.fixtures)
    methods = args.method or list(M2_METHODS)
    preprocessors = args.preprocess or list(M1_METHODS)
    records = []
    for method in methods:
        for preprocess in preprocessors:
            measurements = []
            abstained = 0
            runtime = 0.0
            for fixture in fixtures:
                image_path = Path(fixture["image"])
                if not image_path.is_absolute():
                    image_path = (Path(args.images) if args.images else fixture_root) / image_path
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(image_path)
                started = perf_counter()
                result = estimate_zone(
                    image, method=method, preprocess=preprocess,
                    stripe_convention=args.stripe_convention,
                    random_state=args.random_state)
                elapsed = (perf_counter() - started) * 1000
                runtime += elapsed
                if result.zone is None:
                    abstained += 1
                    continue
                h, w = image.shape[:2]
                coordinates = fixture.get("coordinates", "pixels")
                manual_l = _fixture_segment(fixture["safety_line"], coordinates, w, h)
                manual_e = _fixture_segment(fixture["platform_edge"], coordinates, w, h)
                measurements.append({
                    "line_distance_px": float(np.mean([
                        _line_distance(result.zone.safety_line, manual_l),
                        _line_distance(result.zone.platform_edge, manual_e)])),
                    "angle_error_deg": float(np.mean([
                        _line_angle_degrees(result.zone.safety_line, manual_l),
                        _line_angle_degrees(result.zone.platform_edge, manual_e)])),
                    "overlap": float(np.mean([
                        _line_overlap(result.zone.safety_line, manual_l),
                        _line_overlap(result.zone.platform_edge, manual_e)])),
                    "runtime_ms": elapsed,
                })
            def mean(key: str):
                return (round(float(np.mean([item[key] for item in measurements])), 5)
                        if measurements else None)
            records.append({
                "method": method, "preprocess": preprocess,
                "fixtures": len(fixtures), "evaluated": len(measurements),
                "abstained": abstained,
                "abstain_rate": round(abstained / len(fixtures), 5) if fixtures else None,
                "mean_line_distance_px": mean("line_distance_px"),
                "mean_angle_error_deg": mean("angle_error_deg"),
                "mean_overlap": mean("overlap"),
                "mean_runtime_ms": round(runtime / len(fixtures), 5) if fixtures else None,
            })
    payload = {
        "validation_status": "DONE_SYNTHETIC" if args.synthetic else "IMPLEMENTED_NOT_VALIDATED_ON_REAL_DATA",
        "fixture_count": len(fixtures), "results": records,
        "note": "Scores compare against independent manual L/E endpoints; boolean GT is never geometry.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _preprocess_inputs(path: str | Path) -> list[Path]:
    source = Path(path)
    if source.is_file():
        return [source]
    if source.is_dir():
        images = sorted(p for p in source.iterdir()
                        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"No supported images found in {source}")
        return images
    raise FileNotFoundError(source)


def _run_preprocess(args) -> None:
    if args.config:
        if args.params:
            raise ValueError("--params cannot be combined with --config")
        cfg = load_config(args.config)
        steps = resolve_pipeline(cfg, args.branch)
    else:
        selected = args.pipeline or args.method
        steps = selected
        if args.params:
            if args.pipeline:
                raise ValueError("--params is only valid with --method")
            try:
                params = json.loads(args.params)
            except json.JSONDecodeError as exc:
                raise ValueError("--params must be a JSON object") from exc
            if not isinstance(params, dict):
                raise ValueError("--params must be a JSON object")
            steps = [{"method": args.method, "params": params}]

    inputs = _preprocess_inputs(args.input)
    output = Path(args.out)
    folder_mode = Path(args.input).is_dir()
    if folder_mode:
        output.mkdir(parents=True, exist_ok=True)
    elif output.suffix.lower() not in _IMAGE_SUFFIXES:
        raise ValueError("For one input image, --out must be an image contact-sheet path")

    records = []
    for image_path in inputs:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {image_path}")
        processed, metadata = apply_pipeline(image, steps, return_metadata=True)
        sheet = make_contact_sheet(image, processed, metadata["runtime_ms"])
        if folder_mode:
            sheet_path = output / f"{image_path.stem}_contact_sheet.png"
            processed_path = output / f"{image_path.stem}_processed.png"
        else:
            sheet_path = output
            processed_path = output.with_name(f"{output.stem}_processed.png")
        _save_image(sheet_path, sheet)
        _save_image(processed_path, processed)
        records.append({
            "input": str(image_path), "processed": str(processed_path),
            "contact_sheet": str(sheet_path), **metadata,
            "quality_original": compute_quality_flags(image),
            "quality_processed": compute_quality_flags(processed),
        })

    metadata_path = ((output / "preprocessing.json") if folder_mode
                     else output.with_suffix(".json"))
    metadata_path.write_text(json.dumps({
        "status": "DONE_SYNTHETIC" if args.synthetic else "IMPLEMENTED_NOT_VALIDATED_ON_REAL_DATA",
        "branch": args.branch if args.config else None,
        "records": records,
        "limitation": "Quality flags are descriptive and do not imply recovery of clipped highlights.",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"images": len(records), "out": str(output),
                      "metadata": str(metadata_path)}, indent=2))


def _synthetic_people_benchmark() -> list[dict]:
    """Small deterministic protocol fixture; it is not an accuracy claim."""
    frames = [np.zeros((160, 96, 3), dtype=np.uint8) for _ in range(7)]
    for index in range(4, 7):
        x = 24 + (index - 4) * 3
        cv2.rectangle(frames[index], (x, 55), (x + 22, 135), (230, 230, 230), -1)

    b0 = AlwaysUnsafeBaseline().predict(frames[-1])
    started = perf_counter()
    b1_people = HOGSVMDetector().predict(frames[-1])
    b1_runtime = (perf_counter() - started) * 1000
    b2 = MOG2BlobDetector(
        warmup=3, learning_rate=0.05, min_area=80, min_aspect=0.1,
        max_aspect=1.0, large_train_ratio=0.5)
    b2_frames = b2.predict_sequence(frames, ["synthetic-camera"] * len(frames))
    b2_people = sum(len(items) for items in b2_frames)
    b2_runtime = sum(item.runtime_ms for item in b2.last_diagnostics)

    # B3/A are explicit mock outputs: no model or network access occurs.
    mock_pose = Person(
        (30, 40, 56, 138), .91, ankles=[(36, 134, .88), (50, 135, .9)],
        runtime_ms=.1, detector="yolo_pose_mock",
        metadata={"weights": "mock-pose", "conf": .25, "imgsz": 640, "iou": .7})
    _, ankle_source = mock_pose.foot_points("pose_ankle")
    rows = [
        {"detector": "B0_always_unsafe", "output_level": "frame",
         "footpoint_method": "not_applicable", "comparison": "frame_baseline",
         "frames": 1, "detections": "", "runtime_ms": b0.runtime_ms},
        {"detector": "B1_hog_svm", "output_level": "instance",
         "footpoint_method": "bbox_bottom", "comparison": "classical",
         "frames": 1, "detections": len(b1_people), "runtime_ms": b1_runtime},
        {"detector": "B2_mog2_blob", "output_level": "instance",
         "footpoint_method": "bbox_bottom", "comparison": "ordered_fixed_camera_only",
         "frames": len(frames), "detections": b2_people, "runtime_ms": b2_runtime},
        {"detector": "B3_yolo_detection_mock", "output_level": "instance",
         "footpoint_method": "bbox_bottom", "comparison": "end_to_end_cross_weight",
         "frames": 1, "detections": 1, "runtime_ms": .1},
        {"detector": "A_yolo_pose_mock", "output_level": "instance",
         "footpoint_method": "bbox_bottom", "comparison": "same_pose_weights",
         "frames": 1, "detections": 1, "runtime_ms": mock_pose.runtime_ms},
        {"detector": "A_yolo_pose_mock", "output_level": "instance",
         "footpoint_method": ankle_source, "comparison": "same_pose_weights",
         "frames": 1, "detections": 1, "runtime_ms": mock_pose.runtime_ms},
    ]
    for row in rows:
        row.update({"status": "DONE_SYNTHETIC", "conf": .25, "imgsz": 640,
                    "iou": .7, "precision": "", "recall": "", "map50": ""})
    return rows


def _run_benchmark_people(args) -> None:
    if args.imgsz <= 0 or not 0 <= args.conf <= 1 or not 0 < args.iou <= 1:
        raise ValueError("benchmark imgsz must be positive, conf in [0,1], and iou in (0,1]")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    failures = output / "failure_examples"
    failures.mkdir(parents=True, exist_ok=True)
    (failures / "taxonomy.json").write_text(json.dumps({
        "categories": list(FAILURE_CATEGORIES),
        "note": "Populate with matched/unmatched reviewed examples; none are inferred by filename.",
    }, indent=2), encoding="utf-8")

    if args.synthetic or args.mock or not args.manifest:
        records = _synthetic_people_benchmark()
        status = "DONE_SYNTHETIC"
        note = "Synthetic arrays and mocked YOLO outputs only; no real-data validation."
    else:
        manifest = Path(args.manifest)
        status = "BLOCKED_REAL_DATA"
        note = "RailEye3D RGB manifest or readable image files are absent."
        usable = False
        if manifest.is_file():
            try:
                table = pd.read_csv(manifest)
                usable = ("image" in table.columns and not table.empty
                          and any(Path(str(path)).is_file() for path in table["image"].dropna()))
            except (pd.errors.EmptyDataError, OSError):
                usable = False
        if usable:
            status = "IMPLEMENTED_NOT_VALIDATED_ON_REAL_DATA"
            note = ("Manifest discovered. Run the frozen detector matrix only with authorized local "
                    "weights; this command did not invent scores or download weights.")
        records = [{
            "detector": "real_manifest_protocol", "output_level": "mixed",
            "footpoint_method": "bbox_bottom_vs_pose_ankle",
            "comparison": "same_pose_weights_and_cross_weight", "frames": 0,
            "detections": "", "runtime_ms": "", "status": status,
            "conf": args.conf, "imgsz": args.imgsz, "iou": args.iou,
            "precision": "", "recall": "", "map50": "",
        }]
    metrics_path = output / "detector_metrics.csv"
    pd.DataFrame.from_records(records).to_csv(metrics_path, index=False)
    payload = {
        "status": status, "detector_metrics": str(metrics_path),
        "failure_examples": str(failures), "note": note,
    }
    (output / "benchmark_people.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


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
    p.add_argument("--manifest", default="data/processed/manifest.csv")
    p.add_argument("--out", default="outputs/data_audit.json")
    p.add_argument("--csv-out", help="Default: same path as --out with .csv suffix")

    p = sub.add_parser("split", help="Leakage-safe whole-sequence protocol split")
    p.add_argument("--manifest", required=True)
    p.add_argument("--split-mode", choices=[PRETRAINED_VAL_TEST, TRAIN_VAL_TEST],
                   default=PRETRAINED_VAL_TEST,
                   help="Default: no-train Stockerau validation / Fels test protocol")
    p.add_argument("--test-sequence", help="Required only for train_val_test")
    p.add_argument("--val-sequence", help="Default: deterministic holdout of another whole sequence")
    p.add_argument("--calibration-frame", action="append", default=[], metavar="SEQUENCE/FRAME",
                   help="Predeclared calibration-only frame; repeat as needed. Excluded from evaluation.")
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
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    p.add_argument("--device", help="Ultralytics device, for example cpu, 0, or cuda:0")
    p.add_argument("--pose-fallback", choices=["bbox", "abstain"], default="bbox")
    p.add_argument("--auto-boundary", action="store_true", help="EXPERIMENTAL image-line heuristic")
    p.add_argument("--boundary-method", choices=_BOUNDARY_METHOD_CHOICES,
                   default="color_hough")
    p.add_argument("--out", default="outputs/prediction.jpg")

    p = sub.add_parser("evaluate", help="Evaluate on held-out split, export errors")
    p.add_argument("--manifest", required=True)
    p.add_argument("--configs", default="configs")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--detector", choices=["gt", "yolo", "pose"], default="gt")
    p.add_argument("--weights")
    p.add_argument("--confidence", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    p.add_argument("--device", help="Ultralytics device, for example cpu, 0, or cuda:0")
    p.add_argument("--pose-fallback", choices=["bbox", "abstain"], default="bbox")
    p.add_argument("--auto-boundary", action="store_true")
    p.add_argument("--boundary-method", choices=_BOUNDARY_METHOD_CHOICES, default="color_hough")
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

    p = sub.add_parser(
        "preprocess", help="Run an M1 image pipeline and export contact-sheet previews",
        description="Preprocess one BGR image or every supported image in a frame folder.",
        epilog=("Examples: railguard preprocess --input frame.jpg --method clahe "
                "--params '{\"clip_limit\": 2.0}' --out outputs/clahe.png; "
                "railguard preprocess --input frames --pipeline clahe+gaussian "
                "--out outputs/previews; railguard preprocess --input frame.jpg "
                "--config configs/m1_preprocessing.yaml --branch m2_boundary "
                "--out outputs/m2.png"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Input image or folder of ordered frame images")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--method", choices=sorted(PREPROCESSORS), default="raw",
                           help="One registered method (default: raw)")
    selection.add_argument("--pipeline", help="Ordered methods separated by '+' or ','")
    selection.add_argument("--config", help="YAML/JSON-friendly M1 config containing branches")
    p.add_argument("--branch", choices=["m2_boundary", "m3_person"], default="m2_boundary",
                   help="Config branch used with --config (default: m2_boundary)")
    p.add_argument("--params", help="JSON object of parameters for --method")
    p.add_argument("--out", default="outputs/preprocessing.png",
                   help="Contact-sheet image for one input, or output directory for a folder")
    p.add_argument("--synthetic", action="store_true",
                   help="Label metadata DONE_SYNTHETIC (only for genuinely synthetic inputs)")

    p = sub.add_parser(
        "inspect-lines", help="Inspect ranked Phase-P2 safety-line/platform-edge candidates",
        description="Run one M2/M1 combination and export Top-3 review-only L/E proposals.",
        epilog=("Examples:\n  railguard inspect-lines --image frame.jpg --method hsv_hough_ransac "
                "--preprocess clahe --out outputs/lines.json --overlay outputs/lines.png\n"
                "  railguard inspect-lines --image synthetic.png --method hybrid_region_evidence "
                "--stripe-convention stripe_center --synthetic"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--method", choices=M2_METHODS, default="hybrid_region_evidence")
    p.add_argument("--preprocess", choices=M1_METHODS, default="raw")
    p.add_argument("--paint-color", choices=["white", "yellow", "both"], default="both")
    p.add_argument("--stripe-convention", choices=["safe_side_edge", "stripe_center"],
                   default="safe_side_edge")
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--out", help="Optional JSON output path")
    p.add_argument("--overlay", help="Optional ranked-candidate overlay image")
    p.add_argument("--debug-mask", help="Optional HSV/LAB paint-mask image")
    p.add_argument("--synthetic", action="store_true",
                   help="Label outputs DONE_SYNTHETIC")

    p = sub.add_parser(
        "benchmark-lines", help="Benchmark M2/M1 combinations against manual L/E fixtures",
        description=("Compute line distance, angle error, overlap, abstain rate, and runtime "
                     "against independent endpoint fixtures."),
        epilog=("Example:\n  railguard benchmark-lines --fixtures fixtures/lines.json "
                "--images fixtures/images --method hsv_hough_ransac --preprocess raw "
                "--out outputs/line_benchmark.json --synthetic"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fixtures", required=True,
                   help="JSON list with image, safety_line, platform_edge, and coordinates")
    p.add_argument("--images", help="Base directory for relative fixture image paths")
    p.add_argument("--method", action="append", choices=M2_METHODS,
                   help="Repeat to select methods; default: complete M2 matrix")
    p.add_argument("--preprocess", action="append", choices=M1_METHODS,
                   help="Repeat to select M1 variants; default: raw, clahe, gamma")
    p.add_argument("--stripe-convention", choices=["safe_side_edge", "stripe_center"],
                   default="safe_side_edge")
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--out", default="outputs/line_benchmark.json")
    p.add_argument("--synthetic", action="store_true",
                   help="Label outputs DONE_SYNTHETIC")

    p = sub.add_parser(
        "benchmark-people", help="Benchmark Phase-P3 person and footpoint baselines",
        description=("Export detector_metrics.csv plus a reviewed failure_examples directory. "
                     "Synthetic/mock mode never loads YOLO weights."),
        epilog=("Examples:\n"
                "  railguard benchmark-people --synthetic --out outputs/people_benchmark\n"
                "  railguard benchmark-people --mock --out outputs/people_mock\n"
                "  railguard benchmark-people --manifest data/processed/manifest.csv "
                "--out outputs/people_real"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true",
                      help="Use generated frames and mocked YOLO outputs; status DONE_SYNTHETIC")
    mode.add_argument("--mock", action="store_true",
                      help="Alias protocol mode using mocked YOLO outputs and no downloads")
    mode.add_argument("--manifest", help="Real prepared manifest; absent RGB yields BLOCKED_REAL_DATA")
    p.add_argument("--weights", help="Local detection weights for a future authorized real run")
    p.add_argument("--pose-weights", help="Local pose weights for same/cross-weight comparisons")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="outputs/people_benchmark")

    p = sub.add_parser(
        "run-experiment", help="Run the guarded Phase-P5 E1--E5 experiment protocol",
        description=("Write a complete, auditable P5 experiment bundle. Real runs never invent "
                     "predictions; --synthetic is an explicitly labelled plumbing fixture."),
        epilog=("Examples:\n"
                "  railguard run-experiment --experiment all --synthetic "
                "--out-dir outputs/experiments\n"
                "  railguard run-experiment --experiment E5 "
                "--manifest data/processed/splits_with_predictions.csv "
                "--out-dir outputs/experiments"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", choices=["E1", "E2", "E3", "E4", "E5", "all"],
                   required=True, help="Experiment family or the complete E1--E5 matrix")
    p.add_argument("--manifest", help="Frozen split manifest, optionally with GT/prediction columns")
    p.add_argument("--synthetic", action="store_true",
                   help="Run only the labelled SYNTHETIC_FIXTURE_ONLY smoke fixture")
    p.add_argument("--out-dir", default="outputs/experiments",
                   help="Bundle root (default: outputs/experiments)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--configs", default="configs",
                   help="Frozen camera YAML directory hashed for E5")

    p = sub.add_parser("studio", help="Launch RailGuard Studio HTML5/Canvas web UI")
    p.add_argument("--port", type=int, default=8502, help="Server port (default: 8502)")
    p.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
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
        manifest = Path(args.manifest)
        if manifest.is_file():
            try:
                df = pd.read_csv(manifest, dtype={"object_id": str})
            except pd.errors.EmptyDataError:
                summary = blocked_data_audit(
                    "Manifest is empty; no RailEye3D data is available.", manifest)
            else:
                summary = audit_manifest(df, manifest)
        else:
            summary = blocked_data_audit(
                "Manifest does not exist; no RailEye3D dataset is available.", manifest)
        json_out, csv_out = write_data_audit(summary, args.out, args.csv_out)
        summary["artifacts"] = {"json": str(json_out), "csv": str(csv_out)}
        print(json.dumps(summary, indent=2))
    elif args.command == "split":
        df = pd.read_csv(args.manifest, dtype={"object_id": str})
        df = split_manifest(df, args.test_sequence, args.val_sequence,
                            split_mode=args.split_mode,
                            calibration_frames=args.calibration_frame)
        assert_no_shared_sequences(df)
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        summary = df.groupby("split").agg(instances=("frame", "size"), sequences=("sequence", "nunique"))
        calibration_count = df.loc[df["calibration_only"], ["sequence", "frame"]].drop_duplicates().shape[0]
        print(summary.to_string()); print(f"Calibration-only frames: {calibration_count}"); print(f"Saved {out}")
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
            people = YOLODetector(
                "pose" if args.detector == "pose" else "bbox", args.weights,
                args.confidence, args.device, imgsz=args.imgsz, iou=args.iou).predict(image)
        decisions, zone, status = analyze(
            image, people, zone, foot_method="pose" if args.detector == "pose" else "bbox",
            auto_boundary=args.auto_boundary, boundary_method=args.boundary_method,
            paint_color=cfg.get("paint_color", "both"), pose_fallback=args.pose_fallback)
        observation_status = ("confirmed_no_people" if args.detector == "gt" and not people
                              else "no_detections" if not people else "detections_complete")
        frame = aggregate_frame(
            decisions, observation_status=observation_status,
            calibration_status=status, calibration_valid=zone is not None)
        _save_image(args.out, draw(
            image, decisions, zone, f"frame={frame.status}; calibration={status}"))
        result = {"status": frame.status, "reason": frame.reason,
                  "calibration_status": status, "frame": frame.as_dict(),
                  "people": len(people),
                  "violations": sum(d.intrusion is True for d in decisions),
                  "unknown": sum(d.intrusion is None for d in decisions),
                  "detections": [d.as_dict() for d in decisions]}
        Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    elif args.command == "evaluate":
        result = evaluate_manifest(args.manifest, args.out, args.configs, args.split,
                                   args.detector, args.weights, args.boundary_method,
                                   args.auto_boundary, args.pose_fallback, args.confidence,
                                   args.iou_threshold, imgsz=args.imgsz,
                                   nms_iou=args.iou, device=args.device)
        print(json.dumps(result, indent=2))
    elif args.command == "export-yolo":
        print(export_yolo(args.manifest, args.out))
    elif args.command == "train":
        train_yolo(args.data, args.weights, args.epochs, args.imgsz, args.seed)
    elif args.command == "preprocess":
        _run_preprocess(args)
    elif args.command == "inspect-lines":
        _run_inspect_lines(args)
    elif args.command == "benchmark-lines":
        _run_benchmark_lines(args)
    elif args.command == "benchmark-people":
        _run_benchmark_people(args)
    elif args.command == "run-experiment":
        result = run_experiment(
            args.experiment, args.manifest, synthetic=args.synthetic,
            out_dir=args.out_dir, seed=args.seed, config_dir=args.configs)
        printable = result
        if args.experiment == "all":
            printable = {
                name: {key: value for key, value in item.items() if key != "metrics"}
                for name, item in result.items()
            }
        print(json.dumps(printable, indent=2))
    elif args.command == "studio":
        from .server import run_studio_server
        run_studio_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
