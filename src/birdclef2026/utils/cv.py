"""CV-fold helpers: read fold assignment csv, derive (train_idx, val_idx)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

FOLD_COL = "fold"


def read_fold_assignment(path) -> pd.DataFrame:
    """Load the csv produced by ``scripts/03_make_folds.py``.

    Required columns: ``sample_id`` (string-coerced filename) and ``fold`` (int 0..K-1).
    Extra columns (``primary_label``, ``filename``, ``path``) are preserved.
    """
    df = pd.read_csv(path)
    required = {"sample_id", FOLD_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df[FOLD_COL] = df[FOLD_COL].astype(int)
    return df


def fold_indices(df: pd.DataFrame, fold: int) -> Tuple[Iterable[int], Iterable[int]]:
    """Return (train_idx, val_idx) — positional ``iloc`` indices, not index labels.

    Use with ``torch.utils.data.Subset(dataset, train_idx)`` or with
    ``df.iloc[train_idx]``. (We deliberately use ``np.where`` instead of
    ``df.index[mask]`` so callers don't need to ``reset_index`` first.)
    """
    fold_arr = df[FOLD_COL].to_numpy()
    train_idx = np.where(fold_arr != fold)[0].tolist()
    val_idx = np.where(fold_arr == fold)[0].tolist()
    return train_idx, val_idx
