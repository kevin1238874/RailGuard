import json
from pathlib import Path
import pandas as pd
from railguard.raileye import make_manifest
from railguard.splitting import split_manifest
from railguard.evaluate import evaluate_manifest
from railguard.export_yolo import export_yolo
from railguard.cli import main


def _prepared(tiny_raileye, tmp_path):
    root, images, sequences = tiny_raileye
    manifest = make_manifest(root, images)
    split = split_manifest(manifest, sequences[2], sequences[1])
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
    assert json.loads((tmp_path / "audit.json").read_text())["frames"] == 6
    main(["split", "--manifest", str(manifest), "--test-sequence", seq[2],
          "--val-sequence", seq[1], "--out", str(tmp_path / "splits.csv")])
    assert set(pd.read_csv(tmp_path / "splits.csv")["split"]) == {"train", "val", "test"}


def test_evaluate_reports_frame_level_metrics(tiny_raileye, tmp_path):
    csv, cfg_dir = _prepared(tiny_raileye, tmp_path)
    out = evaluate_manifest(csv, tmp_path / "metrics", cfg_dir, split="test", detector="gt")
    assert out["frame_level_alert_metrics"]["n"] == 2
    assert out["frame_level_alert_metrics"]["tp"] == 1
    assert out["frame_level_alert_metrics"]["tn"] == 1
