"""Phase P5 scientific experiment protocol and artifact runners.

The runner deliberately separates *protocol execution* from scientific claims.
``--synthetic`` uses a small, explicit prediction fixture to exercise metric and
artifact plumbing.  A normal run never invents predictions: if a manifest does
not contain externally produced ``GT`` and ``prediction`` columns, the bundle is
written with ``BLOCKED_REAL_DATA`` status and empty score fields.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import yaml

from .metrics import binary_metrics
from .splitting import assert_no_shared_sequences, evaluation_subset


EXPERIMENT_FILES = {
    "E1": "e1_geometry.csv",
    "E2": "e2_detectors.csv",
    "E3": "e3_footpoints.csv",
    "E4": "e4_ablation.csv",
    "E5": "e5_cross_station.csv",
}

EXPERIMENT_METHODS = {
    "E1": (
        "raw_hsv_hough", "clahe_hsv_hough", "gamma_lab_hough",
        "temporal_median_hsv_hough", "raw_hsv_ransac", "manual_frozen",
    ),
    "E2": ("B0_frame_only", "B1_hog_svm", "B2_mog2_video", "B3_yolo", "A_yolo_pose"),
    "E3": (
        "bbox_bottom", "pose_two_ankles", "pose_one_ankle", "bbox_fallback",
        "pose_abstain", "bounded_refinement",
    ),
    "E4": (
        "final_pipeline", "without_roi", "without_preprocessing",
        "without_candidate_geometry", "without_pose_ankle", "without_refinement",
    ),
    "E5": ("freeze_on_red-00-00", "transfer_to_red-04-01"),
}

_PREDICTION_COLUMNS = [
    "sequence", "frame", "instance", "GT", "prediction", "confidence", "reason",
    "split", "calibration_only", "iou", "has_gt_box",
]
_FAILURE_COLUMNS = _PREDICTION_COLUMNS + ["failure_type"]
_VALID_PREDICTIONS = {"DANGER", "SAFE", "UNKNOWN", "MISSED"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)) and not pd.isna(value) and value in (0, 1):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _unsafe(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"unsafe", "danger", "true", "1", "yes"}:
        return True
    if text in {"safe", "false", "0", "no"}:
        return False
    raise ValueError(f"GT must be SAFE/UNSAFE or boolean, got {value!r}")


def _prediction(value: Any) -> str:
    text = str(value).strip().upper()
    aliases = {"1": "DANGER", "TRUE": "DANGER", "0": "SAFE", "FALSE": "SAFE",
               "MISSED_DETECTION": "MISSED", "": "UNKNOWN", "NAN": "UNKNOWN"}
    text = aliases.get(text, text)
    if text not in _VALID_PREDICTIONS:
        raise ValueError(f"prediction must be one of {sorted(_VALID_PREDICTIONS)}, got {value!r}")
    return text


def _synthetic_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return an explicit protocol fixture; values are not RailEye3D results."""
    manifest = pd.DataFrame([
        {"sequence": "red-00-00-fixture", "frame": 0, "split": "val", "calibration_only": False},
        {"sequence": "red-04-01-fixture", "frame": 0, "split": "test", "calibration_only": True},
        {"sequence": "red-04-01-fixture", "frame": 1, "split": "test", "calibration_only": False},
        {"sequence": "red-04-01-fixture", "frame": 2, "split": "test", "calibration_only": False},
        {"sequence": "red-04-01-fixture", "frame": 3, "split": "test", "calibration_only": False},
        {"sequence": "red-04-01-fixture", "frame": 4, "split": "test", "calibration_only": False},
    ])
    records = [
        # This deliberate error proves calibration-only rows cannot affect scores.
        ("red-04-01-fixture", 0, "calibration-person", "UNSAFE", "SAFE", .99,
         "calibration_fixture_excluded", .95),
        ("red-04-01-fixture", 1, "person-tp", "UNSAFE", "DANGER", .91,
         "fixture_true_positive", .90),
        ("red-04-01-fixture", 1, "person-tn", "SAFE", "SAFE", .88,
         "fixture_true_negative", .85),
        ("red-04-01-fixture", 2, "person-unknown", "UNSAFE", "UNKNOWN", .20,
         "fixture_missing_footpoint", .70),
        ("red-04-01-fixture", 3, "person-missed", "UNSAFE", "MISSED", 0.0,
         "fixture_missed_detection", 0.0),
        ("red-04-01-fixture", 4, "person-fp", "SAFE", "DANGER", .65,
         "fixture_false_alarm", .60),
    ]
    predictions = pd.DataFrame(records, columns=[
        "sequence", "frame", "instance", "GT", "prediction", "confidence", "reason", "iou",
    ])
    predictions = predictions.merge(manifest, on=["sequence", "frame"], how="left")
    predictions["has_gt_box"] = True
    return manifest, predictions[_PREDICTION_COLUMNS]


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=_PREDICTION_COLUMNS)


def _load_real_tables(manifest: str | Path | None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if manifest is None or not Path(manifest).is_file():
        return (pd.DataFrame(columns=["sequence", "frame", "split", "calibration_only"]),
                _empty_predictions(), "Manifest is missing; no real-data run was performed.")
    try:
        source = pd.read_csv(manifest, dtype={"object_id": str})
    except (OSError, pd.errors.EmptyDataError) as exc:
        return (pd.DataFrame(columns=["sequence", "frame", "split", "calibration_only"]),
                _empty_predictions(), f"Manifest could not be read: {exc}")
    required_identity = {"sequence", "frame", "split"}
    missing = required_identity - set(source.columns)
    if missing:
        raise ValueError(f"Experiment manifest lacks required columns: {sorted(missing)}")
    if "calibration_only" not in source:
        source["calibration_only"] = False
    split_manifest = source[["sequence", "frame", "split", "calibration_only"]].drop_duplicates().copy()
    split_manifest["calibration_only"] = split_manifest["calibration_only"].map(_truthy)

    # Only score predictions explicitly supplied by an upstream frozen run.
    gt_column = "GT" if "GT" in source else "gt" if "gt" in source else None
    if gt_column is None and "beyond_safety_line" in source:
        gt_column = "beyond_safety_line"
    prediction_column = "prediction" if "prediction" in source else None
    if gt_column is None or prediction_column is None:
        return split_manifest, _empty_predictions(), (
            "Manifest contains no externally produced GT/prediction pair; predictions were not fabricated."
        )
    scored = pd.DataFrame({
        "sequence": source["sequence"], "frame": source["frame"],
        "instance": source.get("instance", source.get("object_id", source.index.astype(str))),
        "GT": source[gt_column], "prediction": source[prediction_column],
        "confidence": source.get("confidence", pd.Series([np.nan] * len(source))),
        "reason": source.get("reason", pd.Series(["supplied_prediction"] * len(source))),
        "split": source["split"], "calibration_only": source["calibration_only"].map(_truthy),
        "iou": source.get("iou", pd.Series([np.nan] * len(source))),
        "has_gt_box": source.get("has_gt_box", pd.Series([False] * len(source))),
    })
    return split_manifest, scored[_PREDICTION_COLUMNS], "Scored only predictions explicitly supplied in the manifest."


def _score_predictions(predictions: pd.DataFrame, runtime_seconds: float) -> tuple[dict[str, Any], pd.DataFrame]:
    if predictions.empty:
        raise ValueError("Cannot score an empty prediction table")
    scored = predictions.copy()
    scored["GT"] = scored["GT"].map(lambda value: "UNSAFE" if _unsafe(value) else "SAFE")
    scored["prediction"] = scored["prediction"].map(_prediction)
    truth, predicted = [], []
    failures: list[dict[str, Any]] = []
    unknown_unsafe = missed_detection_unsafe = 0
    for row in scored.to_dict("records"):
        unsafe = row["GT"] == "UNSAFE"
        prediction = row["prediction"]
        if unsafe:
            truth.append(1)
            predicted.append(int(prediction == "DANGER"))
            if prediction != "DANGER":
                kind = "FN_UNKNOWN" if prediction == "UNKNOWN" else (
                    "FN_MISSED" if prediction == "MISSED" else "FN_SAFE_PREDICTION")
                unknown_unsafe += int(prediction == "UNKNOWN")
                missed_detection_unsafe += int(prediction == "MISSED")
                failures.append({**row, "failure_type": kind})
        elif prediction in {"DANGER", "SAFE"}:
            truth.append(0)
            predicted.append(int(prediction == "DANGER"))
            if prediction == "DANGER":
                failures.append({**row, "failure_type": "FP"})
        elif prediction == "UNKNOWN":
            failures.append({**row, "failure_type": "UNKNOWN_SAFE_GT"})
    end_to_end = binary_metrics(truth, predicted)

    determined = scored[scored["prediction"].isin({"DANGER", "SAFE"})]
    conditional = binary_metrics(
        determined["GT"].eq("UNSAFE").astype(int),
        determined["prediction"].eq("DANGER").astype(int),
    ) if not determined.empty else None

    frame_truth, frame_pred = [], []
    for _, rows in scored.groupby(["sequence", "frame"], sort=True):
        is_unsafe = bool(rows["GT"].eq("UNSAFE").any())
        has_danger = bool(rows["prediction"].eq("DANGER").any())
        has_determined = bool(rows["prediction"].isin({"DANGER", "SAFE"}).any())
        if is_unsafe or has_determined:
            frame_truth.append(int(is_unsafe))
            frame_pred.append(int(has_danger))
    frame_metrics = binary_metrics(frame_truth, frame_pred) if frame_truth else None

    gt_box_mask = scored["has_gt_box"].map(_truthy)
    numeric_iou = pd.to_numeric(scored.loc[gt_box_mask, "iou"], errors="coerce").dropna()
    bbox_metrics = None
    if not numeric_iou.empty:
        bbox_metrics = {
            "mean_iou": float(numeric_iou.mean()),
            "map50": float((numeric_iou >= .5).mean()),
            "mAP@0.5": float((numeric_iou >= .5).mean()),
            "gt_boxes": int(gt_box_mask.sum()),
            "iou_observations": int(len(numeric_iou)),
            "note": "Fixture/supplied IoU hit rate at 0.5; full detector AP requires ranked detections.",
        }
    total = len(scored)
    unknown = int(scored["prediction"].eq("UNKNOWN").sum())
    # Explicit UNKNOWN observations still count as detections; MISSED does not.
    detected = int((scored["prediction"] != "MISSED").sum())
    frames = scored[["sequence", "frame"]].drop_duplicates().shape[0]
    fps = float(frames / runtime_seconds) if runtime_seconds > 0 else None
    metrics = {
        "TP": end_to_end["tp"], "FP": end_to_end["fp"],
        "FN": end_to_end["fn"], "TN": end_to_end["tn"],
        "tp": end_to_end["tp"], "fp": end_to_end["fp"],
        "fn": end_to_end["fn"], "tn": end_to_end["tn"],
        "unsafe_class": {key: end_to_end[key] for key in ("precision", "recall", "f1")},
        "unsafe_precision": end_to_end["precision"],
        "unsafe_recall": end_to_end["recall"],
        "unsafe_f1": end_to_end["f1"],
        "end_to_end_unsafe_recall": end_to_end["recall"],
        "frame_level_alert": frame_metrics,
        "frame_level_alert_precision": None if frame_metrics is None else frame_metrics["precision"],
        "frame_level_alert_recall": None if frame_metrics is None else frame_metrics["recall"],
        "frame_level_alert_f1": None if frame_metrics is None else frame_metrics["f1"],
        "false_alarms": end_to_end["fp"], "missed_unsafe": end_to_end["fn"],
        "unknown_unsafe": unknown_unsafe, "missed_detection_unsafe": missed_detection_unsafe,
        "unknown_rate": float(unknown / total),
        "detection_coverage": float(detected / total),
        "scored_instances": total, "scored_frames": frames,
        "runtime_seconds": runtime_seconds, "runtime_ms": runtime_seconds * 1000.0, "fps": fps,
        "bbox": bbox_metrics,
        "iou_mean": None if bbox_metrics is None else bbox_metrics["mean_iou"],
        "map50": None if bbox_metrics is None else bbox_metrics["map50"],
        "mAP@0.5": None if bbox_metrics is None else bbox_metrics["map50"],
        "conditional_matched_determined": conditional,
        "scope_note_b0_mog2": (
            "B0 is frame-level only and must not be interpreted as instance detection. "
            "MOG2 is valid only for ordered fixed-camera video and has a different output scope."
        ),
        "protocol_note": (
            "End-to-end unsafe metrics count missed and UNKNOWN unsafe persons as FN. "
            "Conditional matched/determined metrics are reported separately and never replace end-to-end recall."
        ),
    }
    failure_table = pd.DataFrame(failures, columns=_FAILURE_COLUMNS)
    return metrics, failure_table


def _camera_hashes(config_dir: str | Path | None) -> dict[str, str]:
    if config_dir is None:
        return {}
    root = Path(config_dir)
    if not root.is_dir():
        return {}
    return {str(path): _sha256(path) for path in sorted(root.glob("*.yaml")) if path.is_file()}


def _environment(seed: int, weight_files: Iterable[str | Path] | None = None) -> dict[str, Any]:
    versions = {}
    for package in ("railguard-cv", "numpy", "pandas", "opencv-python", "PyYAML", "ultralytics", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    weights = {}
    for item in weight_files or ():
        path = Path(item)
        weights[str(path)] = _sha256(path) if path.is_file() else None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(), "gpu": "not_probed_no_model_run",
        "package_versions": versions, "seed": seed, "weight_hashes": weights,
    }


def _write_figures(directory: Path, synthetic: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if not synthetic:
        (directory / "README.md").write_text(
            "No figures were generated because no scoreable real-data predictions were available.\n",
            encoding="utf-8",
        )
        return
    original = np.zeros((96, 160, 3), dtype=np.uint8)
    original[:] = (35, 35, 35)
    cv2.line(original, (0, 54), (159, 54), (0, 220, 255), 3)
    cv2.line(original, (0, 78), (159, 78), (200, 200, 200), 2)
    cv2.rectangle(original, (65, 26), (88, 72), (255, 180, 80), 2)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(original, (0, 140, 140), (80, 255, 255))
    edges = cv2.Canny(gray, 50, 150)
    zone = original.copy()
    overlay = zone.copy()
    cv2.fillPoly(overlay, [np.array([[0, 54], [159, 54], [159, 78], [0, 78]])], (0, 0, 180))
    zone = cv2.addWeighted(overlay, .25, zone, .75, 0)
    figures = {
        "original.png": original,
        "preprocessed.png": cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR),
        "mask.png": mask, "edges.png": edges,
        "lines.png": original, "Z_overlay.png": zone, "detections_overlay.png": original,
    }
    for name, image in figures.items():
        if not cv2.imwrite(str(directory / name), image):
            raise IOError(f"Could not write synthetic figure {directory / name}")


def _experiment_summary_rows(experiment: str, metrics: dict[str, Any], status: str) -> pd.DataFrame:
    rows = []
    for method in EXPERIMENT_METHODS[experiment]:
        row = {
            "experiment": experiment, "method": method, "data_status": status,
            "measurement_scope": "protocol_fixture" if status == "SYNTHETIC_FIXTURE_ONLY" else "supplied_predictions",
            "TP": metrics.get("TP"), "FP": metrics.get("FP"),
            "FN": metrics.get("FN"), "TN": metrics.get("TN"),
            "unsafe_precision": metrics.get("unsafe_precision"),
            "unsafe_recall": metrics.get("unsafe_recall"), "unsafe_f1": metrics.get("unsafe_f1"),
            "unknown_rate": metrics.get("unknown_rate"),
            "detection_coverage": metrics.get("detection_coverage"),
            "runtime_ms": metrics.get("runtime_ms"), "fps": metrics.get("fps"),
            "map50": metrics.get("map50"),
        }
        if experiment == "E2" and method == "B0_frame_only":
            row["scope_note"] = "Frame-level alert baseline only; instance metrics are not comparable."
        elif experiment == "E2" and method == "B2_mog2_video":
            row["scope_note"] = "Ordered fixed-camera video only; not valid on isolated frames."
        else:
            row["scope_note"] = "No method ranking is inferred from a shared protocol fixture."
        rows.append(row)
    return pd.DataFrame(rows)


def _blocked_metrics(status: str, note: str) -> dict[str, Any]:
    return {
        "status": status, "data_status": status, "TP": None, "FP": None, "FN": None, "TN": None,
        "unsafe_class": {"precision": None, "recall": None, "f1": None},
        "frame_level_alert": None, "false_alarms": None, "missed_unsafe": None,
        "unknown_rate": None, "detection_coverage": None, "fps": None,
        "runtime_seconds": None, "runtime_ms": None, "bbox": None,
        "iou_mean": None, "map50": None, "mAP@0.5": None,
        "conditional_matched_determined": None,
        "scope_note_b0_mog2": (
            "B0 is frame-level only; MOG2 requires ordered fixed-camera video. Neither was scored."
        ),
        "note": note,
    }


def run_experiment(experiment: str, manifest: str | Path | None = None, *,
                   synthetic: bool = False, out_dir: str | Path = "outputs/experiments",
                   seed: int = 42, config_dir: str | Path | None = "configs",
                   weight_files: Iterable[str | Path] | None = None) -> dict[str, Any]:
    """Run one P5 experiment or all experiments and write complete bundles.

    Real manifests are scoreable only when they carry frozen upstream ``GT`` and
    ``prediction`` columns.  This guard prevents this orchestration layer from
    silently tuning or manufacturing model outputs on the test set.
    """
    requested = str(experiment).upper()
    if requested == "ALL":
        return {name: run_experiment(
            name, manifest, synthetic=synthetic, out_dir=out_dir, seed=seed,
            config_dir=config_dir, weight_files=weight_files,
        ) for name in EXPERIMENT_FILES}
    if requested not in EXPERIMENT_FILES:
        raise ValueError(f"experiment must be one of {[*EXPERIMENT_FILES, 'all']}")

    random.seed(seed)
    np.random.seed(seed)
    root = Path(out_dir)
    bundle = root / requested
    bundle.mkdir(parents=True, exist_ok=True)
    # Snapshot E5 camera configuration before any score is calculated.
    frozen_hashes = _camera_hashes(config_dir) if requested == "E5" else {}
    started = perf_counter()
    if synthetic:
        split_table, predictions = _synthetic_tables()
        status = "SYNTHETIC_FIXTURE_ONLY"
        source_note = "Generated protocol fixture only; it is not RailEye3D and has no scientific validity."
    else:
        split_table, predictions, source_note = _load_real_tables(manifest)
        status = "SCORED_MANIFEST_PREDICTIONS" if not predictions.empty else "BLOCKED_REAL_DATA"

    if not split_table.empty:
        assert_no_shared_sequences(split_table)

    split_table.to_csv(bundle / "split_manifest.csv", index=False)
    scoreable = predictions.copy()
    if not scoreable.empty:
        scoreable = evaluation_subset(scoreable, "test")
    elapsed = max(perf_counter() - started, 1e-12)
    if scoreable.empty:
        metrics = _blocked_metrics(status, source_note)
        failures = pd.DataFrame(columns=_FAILURE_COLUMNS)
    else:
        metrics, failures = _score_predictions(scoreable, elapsed)
        metrics.update({"status": status, "data_status": status, "source_note": source_note})
    metrics["calibration_only_frames_excluded"] = int(
        split_table.loc[split_table["calibration_only"].map(_truthy), ["sequence", "frame"]]
        .drop_duplicates().shape[0]
    ) if not split_table.empty else 0

    protocol_config: dict[str, Any] = {
        "experiment": requested, "method": list(EXPERIMENT_METHODS[requested]),
        "params": {
            "scoring_split": "test", "selection_split": "val",
            "calibration_frames_scored": False,
            "unknown_or_missed_unsafe_count_as_fn": True,
        },
        "seed": seed, "data_status": status,
        "manifest": None if manifest is None else str(manifest),
        "synthetic": synthetic, "test_set_tuning": False,
        "protocol_guard": "Parameters must be frozen on validation before test scoring.",
        "frozen_camera_yaml_hashes": frozen_hashes,
    }
    protocol_config["config_hash"] = _config_hash(protocol_config)
    metrics["config_hash"] = protocol_config["config_hash"]
    metrics["frozen_camera_yaml_hashes"] = frozen_hashes
    (bundle / "config.yaml").write_text(
        yaml.safe_dump(_jsonable(protocol_config), sort_keys=False), encoding="utf-8")
    (bundle / "environment.json").write_text(
        json.dumps(_environment(seed, weight_files), indent=2), encoding="utf-8")
    (bundle / "metrics.json").write_text(
        json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")
    scoreable.reindex(columns=_PREDICTION_COLUMNS).to_csv(bundle / "predictions.csv", index=False)
    failures.reindex(columns=_FAILURE_COLUMNS).to_csv(bundle / "failures.csv", index=False)
    _write_figures(bundle / "figures", synthetic=synthetic)

    summary_table = _experiment_summary_rows(requested, metrics, status)
    summary_name = EXPERIMENT_FILES[requested]
    summary_table.to_csv(bundle / summary_name, index=False)
    root.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(root / summary_name, index=False)
    if status == "BLOCKED_REAL_DATA":
        finding = "No quantitative finding is reported because scoreable real-data predictions are absent."
    elif status == "SYNTHETIC_FIXTURE_ONLY":
        finding = (
            f"The fixture exercised {metrics['scored_instances']} scored instances; its values test the "
            "runner and metric definitions only and must not be treated as method performance."
        )
    else:
        finding = (
            f"The supplied frozen predictions yielded unsafe recall {metrics['unsafe_recall']:.4f}; "
            "the runner did not select or tune parameters on test data."
        )
    (bundle / "summary.md").write_text(
        f"# {requested} experiment summary\n\n"
        f"Data status: `{status}`\n\n"
        f"## Finding\n\n{finding}\n\n"
        f"## Protocol guards\n\n"
        f"Calibration-only frames excluded: {metrics['calibration_only_frames_excluded']}. "
        "Configuration selection is restricted to validation and `test_set_tuning` is false.\n\n"
        f"## Limitations\n\n{source_note} B0 and MOG2 have different evaluation scopes as recorded in metrics.json.\n",
        encoding="utf-8",
    )
    return {
        "experiment": requested, "status": status, "bundle": str(bundle),
        "summary_csv": str(bundle / summary_name), "config_hash": protocol_config["config_hash"],
        "metrics": metrics,
    }


def run_experiments(experiment: str = "all", **kwargs: Any) -> dict[str, Any]:
    """Plural convenience alias for callers orchestrating the E1--E5 matrix."""
    return run_experiment(experiment, **kwargs)
