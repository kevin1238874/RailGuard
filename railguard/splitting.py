"""Leakage-safe whole-sequence splits: never randomize frames across splits."""
from __future__ import annotations
from collections.abc import Iterable
from typing import Any

import pandas as pd

TRAIN_VAL_TEST = "train_val_test"
PRETRAINED_VAL_TEST = "pretrained_val_test"


def _frame_identity(value: str | tuple[str, int]) -> tuple[str, int]:
    if isinstance(value, tuple) and len(value) == 2:
        sequence, frame = value
    elif isinstance(value, str) and "/" in value:
        sequence, frame = value.rsplit("/", 1)
    else:
        raise ValueError("Calibration frames must use SEQUENCE/FRAME or (sequence, frame)")
    sequence = str(sequence).strip()
    if not sequence:
        raise ValueError("Calibration frame sequence cannot be empty")
    try:
        frame_number = int(frame)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid calibration frame number: {frame!r}") from exc
    if frame_number < 0:
        raise ValueError("Calibration frame number cannot be negative")
    return sequence, frame_number


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0", "", "nan"}:
        return False
    raise ValueError(f"Invalid calibration_only boolean: {value!r}")


def _mark_calibration_frames(df: pd.DataFrame,
                             calibration_frames: Iterable[str | tuple[str, int]] | None) -> pd.DataFrame:
    out = df.copy()
    identities = {_frame_identity(value) for value in (calibration_frames or [])}
    available = set(zip(out["sequence"].astype(str), pd.to_numeric(out["frame"]).astype(int)))
    unknown = sorted(identities - available)
    if unknown:
        raise ValueError(f"Calibration frames are not present in the manifest: {unknown}")
    row_ids = list(zip(out["sequence"].astype(str), pd.to_numeric(out["frame"]).astype(int)))
    out["calibration_only"] = [identity in identities for identity in row_ids]
    out["evaluation_eligible"] = ~out["calibration_only"] & out["split"].isin({"train", "val", "test"})
    return out


def split_manifest(df: pd.DataFrame, test_sequence: str | None = None,
                   val_sequence: str | None = None, split_mode: str = TRAIN_VAL_TEST,
                   calibration_frames: Iterable[str | tuple[str, int]] | None = None) -> pd.DataFrame:
    """Assign whole-sequence protocol splits and flag calibration-only frames.

    ``pretrained_val_test`` is the primary no-training protocol: every
    ``red-00-00*`` sequence is validation and every ``red-04-01*`` sequence is
    test. Other sequences are explicitly excluded. The legacy train/val/test
    behavior remains available for optional fine-tuning.
    """
    sequences = set(df["sequence"].unique())
    if split_mode not in {TRAIN_VAL_TEST, PRETRAINED_VAL_TEST}:
        raise ValueError(f"Unknown split_mode {split_mode!r}")
    if split_mode == PRETRAINED_VAL_TEST:
        if test_sequence is not None or val_sequence is not None:
            raise ValueError("pretrained_val_test uses fixed red-00-00*/red-04-01* prefixes; "
                             "do not pass explicit sequence holdouts")
        sequence_values = df["sequence"].astype(str)
        val_mask = sequence_values.str.startswith("red-00-00")
        test_mask = sequence_values.str.startswith("red-04-01")
        if not val_mask.any():
            raise ValueError("pretrained_val_test requires at least one red-00-00* validation sequence")
        if not test_mask.any():
            raise ValueError("pretrained_val_test requires at least one red-04-01* test sequence")
        out = df.copy()
        out["split"] = "excluded"
        out.loc[val_mask, "split"] = "val"
        out.loc[test_mask, "split"] = "test"
        out["split_mode"] = PRETRAINED_VAL_TEST
        out = _mark_calibration_frames(out, calibration_frames)
        assert_no_shared_sequences(out)
        return out

    if test_sequence is None:
        raise ValueError("test_sequence is required for train_val_test mode")
    if test_sequence not in sequences:
        raise ValueError(f"Unknown test sequence {test_sequence!r}; available: {sorted(sequences)}")
    if val_sequence is None:
        candidates = sorted(sequences - {test_sequence})
        if len(candidates) < 2:
            raise ValueError("A leakage-safe train/validation/test split requires >=3 sequences. "
                             "Provide additional sequences; do not randomly split frames of one sequence.")
        val_sequence = candidates[-1]
    if val_sequence not in sequences or val_sequence == test_sequence:
        raise ValueError("Validation sequence must exist and differ from test sequence")
    if not sequences - {test_sequence, val_sequence}:
        raise ValueError("No train sequence left after validation and test holdout")
    out = df.copy()
    out["split"] = "train"
    out.loc[out.sequence == val_sequence, "split"] = "val"
    out.loc[out.sequence == test_sequence, "split"] = "test"
    out["split_mode"] = TRAIN_VAL_TEST
    out = _mark_calibration_frames(out, calibration_frames)
    assert_no_shared_sequences(out)
    return out


def evaluation_subset(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Select scoreable rows, always excluding predeclared calibration frames."""
    if "split" not in df.columns:
        raise ValueError("Manifest lacks a split column")
    out = df[df["split"] == split]
    if "calibration_only" in out.columns:
        out = out[~out["calibration_only"].map(_truthy)]
    return out.copy()


def assert_no_shared_sequences(df: pd.DataFrame) -> None:
    if "split" not in df.columns:
        raise ValueError("Manifest lacks a split column")
    counts = df.groupby("sequence")["split"].nunique()
    if not counts.empty and counts.max() > 1:
        raise ValueError("Data leakage: one sequence appears in multiple splits")
    counts = df.groupby(["sequence", "frame"])["split"].nunique()
    if not counts.empty and counts.max() > 1:
        raise ValueError("Data leakage: one frame appears in multiple splits")
