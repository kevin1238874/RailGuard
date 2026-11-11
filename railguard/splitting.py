"""Leakage-safe whole-sequence splits: never randomize frames across splits."""
from __future__ import annotations
import pandas as pd


def split_manifest(df: pd.DataFrame, test_sequence: str, val_sequence: str | None = None) -> pd.DataFrame:
    sequences = set(df["sequence"].unique())
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
    assert_no_shared_sequences(out)
    return out


def assert_no_shared_sequences(df: pd.DataFrame) -> None:
    if "split" not in df.columns:
        raise ValueError("Manifest lacks a split column")
    counts = df.groupby("sequence")["split"].nunique()
    if not counts.empty and counts.max() > 1:
        raise ValueError("Data leakage: one sequence appears in multiple splits")
    counts = df.groupby(["sequence", "frame"])["split"].nunique()
    if not counts.empty and counts.max() > 1:
        raise ValueError("Data leakage: one frame appears in multiple splits")
