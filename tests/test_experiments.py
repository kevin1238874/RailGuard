import json
from pathlib import Path

import pandas as pd
import yaml

from railguard.cli import main
from railguard.experiments import EXPERIMENT_FILES, run_experiment


CORE_BUNDLE = {
    "config.yaml", "environment.json", "split_manifest.csv", "metrics.json",
    "predictions.csv", "failures.csv", "figures", "summary.md",
}


def test_e1_to_e5_synthetic_bundles_are_complete(tmp_path):
    out = tmp_path / "experiments"
    result = run_experiment("all", synthetic=True, out_dir=out, config_dir=tmp_path / "configs")
    assert set(result) == set(EXPERIMENT_FILES)
    for experiment, csv_name in EXPERIMENT_FILES.items():
        bundle = out / experiment
        assert CORE_BUNDLE <= {path.name for path in bundle.iterdir()}
        assert (bundle / csv_name).is_file()
        assert (out / csv_name).is_file()
        assert any((bundle / "figures").glob("*.png"))
        assert result[experiment]["status"] == "SYNTHETIC_FIXTURE_ONLY"
        assert "SYNTHETIC_FIXTURE_ONLY" in (bundle / "summary.md").read_text(encoding="utf-8")


def test_metrics_count_unknown_and_missed_unsafe_as_end_to_end_fn(tmp_path):
    out = tmp_path / "experiments"
    run_experiment("E2", synthetic=True, out_dir=out)
    metrics = json.loads((out / "E2" / "metrics.json").read_text(encoding="utf-8"))
    assert (metrics["TP"], metrics["FP"], metrics["FN"], metrics["TN"]) == (1, 1, 2, 1)
    assert metrics["unknown_unsafe"] == 1
    assert metrics["missed_detection_unsafe"] == 1
    assert metrics["end_to_end_unsafe_recall"] == 1 / 3
    assert metrics["conditional_matched_determined"]["recall"] == 1.0
    assert metrics["unknown_rate"] == 1 / 5
    assert metrics["detection_coverage"] == 4 / 5


def test_calibration_only_frame_is_listed_but_excluded_from_predictions_and_scores(tmp_path):
    out = tmp_path / "experiments"
    run_experiment("E1", synthetic=True, out_dir=out)
    bundle = out / "E1"
    split = pd.read_csv(bundle / "split_manifest.csv")
    predictions = pd.read_csv(bundle / "predictions.csv")
    metrics = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))
    calibration = split[split["calibration_only"]]
    assert set(calibration["frame"]) == {0}
    assert "calibration-person" not in set(predictions["instance"])
    assert metrics["calibration_only_frames_excluded"] == 1
    assert metrics["scored_instances"] == 5


def test_missing_real_manifest_writes_blocked_bundle(tmp_path):
    out = tmp_path / "experiments"
    result = run_experiment("E5", tmp_path / "absent.csv", out_dir=out,
                            config_dir=tmp_path / "configs")
    bundle = out / "E5"
    assert result["status"] == "BLOCKED_REAL_DATA"
    assert CORE_BUNDLE <= {path.name for path in bundle.iterdir()}
    metrics = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((bundle / "config.yaml").read_text(encoding="utf-8"))
    assert metrics["status"] == "BLOCKED_REAL_DATA"
    assert metrics["TP"] is None
    assert config["test_set_tuning"] is False
    assert config["frozen_camera_yaml_hashes"] == {}


def test_e5_records_frozen_camera_yaml_hash_and_cli_runs(tmp_path, capsys):
    configs = tmp_path / "configs"
    configs.mkdir()
    camera = configs / "red-04-01-fixture.yaml"
    camera.write_text("zone: {}\n", encoding="utf-8")
    out = tmp_path / "cli-experiments"
    main(["run-experiment", "--experiment", "E5", "--synthetic",
          "--configs", str(configs), "--out-dir", str(out)])
    printed = json.loads(capsys.readouterr().out)
    config = yaml.safe_load((out / "E5" / "config.yaml").read_text(encoding="utf-8"))
    assert printed["status"] == "SYNTHETIC_FIXTURE_ONLY"
    assert str(camera) in config["frozen_camera_yaml_hashes"]
    assert len(config["frozen_camera_yaml_hashes"][str(camera)]) == 64
