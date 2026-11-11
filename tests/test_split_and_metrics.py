import pandas as pd
import pytest
from railguard.metrics import binary_metrics, bbox_iou, greedily_match
from railguard.splitting import assert_no_shared_sequences, split_manifest
from railguard.detection import Person


def test_split_entire_sequences():
    df = pd.DataFrame({"sequence": ["a", "a", "b", "b", "c"], "frame": [0, 15, 0, 15, 0]})
    out = split_manifest(df, "c", "b")
    assert out.groupby("sequence").split.nunique().max() == 1
    assert out.groupby("sequence").split.first().to_dict() == {"a": "train", "b": "val", "c": "test"}
    assert_no_shared_sequences(out)


def test_split_two_sequences_fails():
    df = pd.DataFrame({"sequence": ["a", "b"], "frame": [0, 0]})
    with pytest.raises(ValueError, match=">=3 sequences"):
        split_manifest(df, "b")


def test_sequence_leakage_detected():
    df = pd.DataFrame({"sequence": ["a", "a"], "frame": [0, 15], "split": ["train", "test"]})
    with pytest.raises(ValueError, match="Data leakage"):
        assert_no_shared_sequences(df)


def test_binary_metrics():
    out = binary_metrics([1, 0, 1, 0], [1, 1, 0, 0])
    assert (out["tp"], out["tn"], out["fp"], out["fn"]) == (1, 1, 1, 1)
    assert out["f1"] == 0.5


def test_matching_and_iou():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    gt = [Person((0, 0, 10, 10)), Person((30, 30, 40, 40))]
    det = [Person((1, 1, 11, 11)), Person((50, 50, 60, 60))]
    matched, miss, extra = greedily_match(gt, det, threshold=.5)
    assert len(matched) == len(miss) == len(extra) == 1
