import json
from pathlib import Path
import pandas as pd
from railguard.raileye import make_manifest
from railguard.splitting import split_manifest
from railguard.evaluate import evaluate_manifest
from railguard.export_yolo import export_yolo
from railguard.cli import main


def _prepared(tiny_raileye, tmp_path, calibration_frames=None):
    root, images, sequences = tiny_raileye
    manifest = make_manifest(root, images)
    split = split_manifest(manifest, sequences[2], sequences[1],
                           calibration_frames=calibration_frames)
    csv = tmp_path / "splits.csv"
    split.to_csv(csv, index=False)
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / f"{sequences[2]}.yaml").write_text("coordinates: pixels\nzone:\n  safety_line: [[0, 60], [99, 60]]\n  platform_edge: [[0, 90], [99, 90]]\n")
    return csv, cfg_dir


def test_evaluate_oracle(tiny_raileye, tmp_path):
    csv, cfg_dir = _prepared(tiny_raileye, tmp_path)
    out = evaluate_manifest(csv, tmp_path / "metrics", cfg_dir, split="test", detector="gt")
    assert out["tp"] == 1
    assert out["fn"] == out["fp"] == 0
    assert out["precision"] == out["recall"] == out["f1"] == 1.0
    assert out["matched_boxes"] == 2
    assert (tmp_path / "metrics" / "failure_cases.csv").exists()


def test_missing_calibration_fails_early(tiny_raileye, tmp_path):
    csv, _ = _prepared(tiny_raileye, tmp_path)
    try:
        evaluate_manifest(csv, tmp_path / "scores", tmp_path / "missing")
    except FileNotFoundError as exc:
        assert "calibration" in str(exc)
    else:
        raise AssertionError("Evaluation must fail without calibration")


def test_export_yolo_keeps_zero_person_frame(tiny_raileye, tmp_path):
    csv, _ = _prepared(tiny_raileye, tmp_path)
    out = tmp_path / "yolo"
    counts = export_yolo(csv, out)
    assert counts == {"train": 2, "val": 2, "test": 2}
    assert len(list((out / "images" / "test").glob("*.png"))) == 2
    assert (out / "labels" / "test" / "red-04-00-front-left_00015.txt").read_text() == ""


def test_cli_prepare_audit_split(tiny_raileye, tmp_path, capsys):
    root, images, seq = tiny_raileye
    manifest = tmp_path / "manifest.csv"
    main(["prepare", "--annotations", str(root), "--images", str(images), "--out", str(manifest)])
    assert manifest.exists()
    main(["audit", "--manifest", str(manifest), "--out", str(tmp_path / "audit.json")])
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["frames"] == 6
    assert (tmp_path / "audit.csv").exists()
    main(["split", "--manifest", str(manifest), "--split-mode", "train_val_test",
          "--test-sequence", seq[2],
          "--val-sequence", seq[1], "--out", str(tmp_path / "splits.csv")])
    assert set(pd.read_csv(tmp_path / "splits.csv")["split"]) == {"train", "val", "test"}


def test_cli_audit_empty_data_writes_blocked_json_and_csv(tmp_path):
    json_out = tmp_path / "data_audit.json"
    csv_out = tmp_path / "data_audit.csv"
    main(["audit", "--manifest", str(tmp_path / "missing.csv"),
          "--out", str(json_out), "--csv-out", str(csv_out)])
    audit = json.loads(json_out.read_text(encoding="utf-8"))
    assert audit["status"] == "BLOCKED_REAL_DATA"
    assert audit["frames"] == audit["person_instances"] == 0
    assert csv_out.exists()
    flat = pd.read_csv(csv_out).set_index("metric")["value"]
    assert flat["status"] == "BLOCKED_REAL_DATA"


def test_evaluate_reports_frame_level_metrics(tiny_raileye, tmp_path):
    csv, cfg_dir = _prepared(tiny_raileye, tmp_path)
    out = evaluate_manifest(csv, tmp_path / "metrics", cfg_dir, split="test", detector="gt")
    assert out["frame_level_alert_metrics"]["n"] == 2
    assert out["frame_level_alert_metrics"]["tp"] == 1
    assert out["frame_level_alert_metrics"]["tn"] == 1


def test_evaluate_excludes_calibration_only_frame(tiny_raileye, tmp_path):
    _, _, sequences = tiny_raileye
    csv, cfg_dir = _prepared(
        tiny_raileye,
        tmp_path,
        calibration_frames=[f"{sequences[2]}/00015"],
    )
    out = evaluate_manifest(csv, tmp_path / "metrics", cfg_dir, split="test", detector="gt")
    assert out["frame_level_alert_metrics"]["n"] == 1
    predictions = pd.read_csv(tmp_path / "metrics" / "predictions.csv")
    assert set(predictions["frame"]) == {0}
