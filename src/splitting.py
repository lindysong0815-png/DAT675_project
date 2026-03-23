"""Scaffold-based splitting helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd


def build_scaffold_groups(df: pd.DataFrame, scaffold_col: str = "Scaffold") -> list[list[int]]:
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, scaffold in enumerate(df[scaffold_col]):
        scaffold_to_indices[scaffold].append(idx)
    return sorted(scaffold_to_indices.values(), key=len, reverse=True)


def make_scaffold_split(
    df: pd.DataFrame,
    scaffold_groups: list[list[int]],
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> dict[str, list[int]]:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train/val/test fractions must sum to 1")

    rng = np.random.RandomState(seed)
    groups = list(scaffold_groups)
    rng.shuffle(groups)

    n_total = len(df)
    train_target = int(round(train_frac * n_total))
    val_target = int(round(val_frac * n_total))

    train_idx, val_idx, test_idx = [], [], []

    for group in groups:
        if len(train_idx) + len(group) <= train_target:
            train_idx.extend(group)
        elif len(val_idx) + len(group) <= val_target:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    return {
        "train_idx": sorted(train_idx),
        "val_idx": sorted(val_idx),
        "test_idx": sorted(test_idx),
    }


def build_split_dataframes(df: pd.DataFrame, split: dict[str, list[int]]) -> dict[str, pd.DataFrame]:
    return {
        "train": df.loc[split["train_idx"]].reset_index(drop=True).copy(),
        "val": df.loc[split["val_idx"]].reset_index(drop=True).copy(),
        "test": df.loc[split["test_idx"]].reset_index(drop=True).copy(),
    }


def scaffold_leakage_counts(split_pack: dict[str, pd.DataFrame], scaffold_col: str = "Scaffold") -> dict[str, int]:
    train_scaffolds = set(split_pack["train"][scaffold_col])
    val_scaffolds = set(split_pack["val"][scaffold_col])
    test_scaffolds = set(split_pack["test"][scaffold_col])
    return {
        "Train ∩ Val": len(train_scaffolds & val_scaffolds),
        "Train ∩ Test": len(train_scaffolds & test_scaffolds),
        "Val ∩ Test": len(val_scaffolds & test_scaffolds),
    }
