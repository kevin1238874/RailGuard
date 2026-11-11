"""Held-out evaluation against RailEye3D beyond_safety_line supervision.

A geometry-only GT-box oracle isolates zone/foot heuristic errors.
End-to-end YOLO and Pose include box matching + unknown / missed positives.
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import pandas as pd
from .config import get_zone, load_config
from .detection import Person, YOLODetector
from .metrics import binary_metrics, greedily_match
from .pipeline import analyze, draw
from .raileye import PERSON_LABELS
from .splitting import assert_no_shared_sequences


def _gt_instances(rows: pd.DataFrame) -> tuple[list[Person], list[bool], list[float | None]]:
    people, truth, occlusions = [], [], []
    for row in rows.itertuples(index=False):
        if not bool(row.object_present) or str(row.label).lower() not in PERSON_LABELS:
            continue
        if pd.isna(row.beyond_safety_line):
            raise ValueError(f"Missing beyond_safety_line for {row.sequence}/{row.frame}/{row.object_id}")
        people.append(Person((float(row.x1), float(row.y1), float(row.x2), float(row.y2)),
                             object_id=str(row.object_id)))
        val = row.beyond_safety_line
        truth.append(val.strip().lower() in {"true", "1"} if isinstance(val, str) else bool(val))
        occlusions.append(None if pd.isna(row.occlusion) else float(row.occlusion))
    return people, truth, occlusions


def evaluate_manifest(manifest: str | Path, output_dir: str | Path, config_dir: str | Path,
                      split: str = "test", detector: str = "gt", weights: str | None = None,
                      boundary_method: str = "color_hough", auto_boundary: bool = False,
                      pose_fallback: str = "bbox", confidence: float = 0.25,
                      iou_threshold: float = 0.5, sample_visualizations: int = 12) -> dict:
    df = pd.read_csv(manifest, dtype={"object_id": str})
    if "object_present" not in df.columns:
        raise ValueError("Missing 'object_present' column; recreate the manifest with prepare")
    if "split" not in df.columns:
        raise ValueError("Run split first; evaluation requires an explicit held-out split")
    assert_no_shared_sequences(df)
    df = df[df["split"] == split]
    if df.empty:
        raise ValueError(f"No rows in split={split!r}")
    if detector not in {"gt", "yolo", "pose"}:
        raise ValueError("Detector must be gt, yolo or pose")
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must lie in (0,1]")
    model = None if detector == "gt" else YOLODetector(
        "pose" if detector == "pose" else "bbox", weights, confidence)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scored_y, scored_pred, conditional_y, conditional_pred, records = [], [], [], [], []
    frame_y, frame_pred = [], []
    matched_count = missed_count = extra_count = abstained_count = unknown_count = 0
    scored_frames = no_zone_frames = 0
    for (sequence, frame), rows in df.groupby(["sequence", "frame"], sort=True):
        image_file = rows.iloc[0]["image"]
        if not isinstance(image_file, str) or not image_file.strip():
            raise FileNotFoundError(f"Images required for geometric evaluation: {sequence}/{frame}")
        image = cv2.imread(image_file)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_file}")
        height, width = image.shape[:2]
        config_file = Path(config_dir) / f"{sequence}.yaml"
        config = load_config(config_file) if config_file.exists() else {}
        if "zone" not in config and not auto_boundary:
            raise FileNotFoundError(f"No per-sequence zone calibration: {config_file}; "
                                    "run calibrate or pass --auto-boundary (experimental)")
        zone = get_zone(config, width, height)
        gt, labels, occlusion = _gt_instances(rows)
        observed = gt if detector == "gt" else model.predict(image)
        decisions, zone, status = analyze(
            image, observed, zone, foot_method="pose" if detector == "pose" else "bbox",
            auto_boundary=auto_boundary, boundary_method=boundary_method,
            paint_color=config.get("paint_color", "both"), pose_fallback=pose_fallback)
        frame_y.append(int(any(labels)))
        frame_pred.append(int(any(dec.intrusion is True for dec in decisions)))
        if zone is None:
            # Undefined geometry is not a negative result. It impacts coverage.
            no_zone_frames += 1
            abstained_count += len(gt)
            for i, person in enumerate(gt):
                if labels[i]:
                    scored_y.append(1); scored_pred.append(0)
                records.append({"sequence": sequence, "frame": frame, "object_id": person.object_id,
                                "truth": int(labels[i]), "prediction": "unknown_zone",
                                "error": bool(labels[i]), "foot_source": "none", "iou": 0,
                                "occlusion": occlusion[i], "zone_status": status})
            continue
        scored_frames += 1
        matches, missed, extra = greedily_match(gt, observed, threshold=iou_threshold)
        matched_count += len(matches); missed_count += len(missed); extra_count += len(extra)
        for gt_idx, pred_idx, overlap in matches:
            label = int(labels[gt_idx])
            decision = decisions[pred_idx]
            if decision.intrusion is None:
                unknown_count += 1
                if label:
                    scored_y.append(1); scored_pred.append(0)
            else:
                value = int(decision.intrusion)
                scored_y.append(label); scored_pred.append(value)
                conditional_y.append(label); conditional_pred.append(value)
            records.append({"sequence": sequence, "frame": frame, "object_id": gt[gt_idx].object_id,
                            "truth": label, "prediction": "unknown" if decision.intrusion is None else int(decision.intrusion),
                            "error": decision.intrusion is None or label != decision.intrusion,
                            "foot_source": decision.foot_source, "iou": overlap,
                            "occlusion": occlusion[gt_idx], "zone_status": status})
        for gt_idx in missed:
            label = int(labels[gt_idx])
            if label:
                scored_y.append(1); scored_pred.append(0)
            records.append({"sequence": sequence, "frame": frame, "object_id": gt[gt_idx].object_id,
                            "truth": label, "prediction": "missed_detection", "error": bool(label),
                            "foot_source": "none", "iou": 0, "occlusion": occlusion[gt_idx],
                            "zone_status": status})
        for pred_idx in extra:
            decision = decisions[pred_idx]
            if decision.intrusion is True:
                scored_y.append(0); scored_pred.append(1)
            records.append({"sequence": sequence, "frame": frame, "object_id": "<unmatched_prediction>",
                            "truth": 0, "prediction": "unknown" if decision.intrusion is None else int(decision.intrusion),
                            "error": bool(decision.intrusion), "foot_source": decision.foot_source,
                            "iou": 0, "occlusion": None, "zone_status": status})
        if scored_frames <= sample_visualizations:
            cv2.imwrite(str(output / f"{sequence}_{int(frame):05d}.jpg"),
                        draw(image, decisions, zone, status))
    if not scored_y:
        raise RuntimeError("No scored positives/negatives; check input and calibration")
    result = binary_metrics(scored_y, scored_pred)
    div = lambda a, b: a / b if b else 0.0
    result.update({"split": split, "detector": detector, "iou_threshold": iou_threshold,
                   "frame_level_alert_metrics": binary_metrics(frame_y, frame_pred),
                   "matched_boxes": matched_count, "missed_boxes": missed_count,
                   "extra_boxes": extra_count,
                   "det_precision": div(matched_count, matched_count + extra_count),
                   "det_recall": div(matched_count, matched_count + missed_count),
                   "scored_frames": scored_frames, "abstained_frames": no_zone_frames,
                   "abstained_gt_instances": abstained_count, "unknown_pose_instances": unknown_count,
                   "conditional_metrics": binary_metrics(conditional_y, conditional_pred) if conditional_y else None,
                   "note": "End-to-end unsafe F1: unmatched/unknown positive GT count as FN. "
                           "Unmatched unsafe predictions count as FP. Undetected negatives excluded from TN; "
                           "interpret accuracy with caution. "
                           "RailEye3D beyond_safety_line is an annotation label, not a line coordinate."})
    (output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    predictions = pd.DataFrame.from_records(records)
    predictions.to_csv(output / "predictions.csv", index=False)
    if records:
        failures = predictions[predictions["error"].astype(bool)]
    else:
        failures = predictions.copy()
    failures.to_csv(output / "failure_cases.csv", index=False)
    return result
