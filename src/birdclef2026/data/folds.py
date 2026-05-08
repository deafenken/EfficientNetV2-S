"""StratifiedGroupKFold by recording id (filename).

Fixes the legacy CV leak: ``metadata.split_train_val`` used
``train_test_split(stratify=primary_label)`` which puts different 5-second
chunks of the *same* recording across train and val, making OOF AUC a
meaningless overestimate. We split by ``filename`` so each recording lives
in exactly one fold.

Edge cases:
  * Singleton classes (only 1 recording) — StratifiedGroupKFold raises;
    we fall back to plain GroupKFold which sacrifices stratification.
  * Any leftover unassigned rows are dropped into the smallest fold (so they
    end up *in training* for 4 of 5 folds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def make_folds(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
    stratify_col: str = "primary_label",
    group_col: str = "filename",
) -> pd.Series:
    """Return an int fold index per row in ``df`` (0..n_splits-1).

    Args:
        df: dataframe with at least ``stratify_col`` and ``group_col``.
        n_splits: number of folds (5 in plans).
        seed: random_state for StratifiedGroupKFold's shuffle.
        stratify_col: column whose value distribution we try to preserve.
        group_col: column whose values must not cross folds (recording id).

    Returns:
        pd.Series of length ``len(df)`` with the same index as ``df``.
    """
    if df.empty:
        return pd.Series([], dtype=int, name="fold")
    if stratify_col not in df.columns or group_col not in df.columns:
        raise KeyError(
            f"need columns '{stratify_col}' and '{group_col}'; got {list(df.columns)}"
        )

    y = df[stratify_col].astype(str).to_numpy()
    groups = df[group_col].astype(str).to_numpy()
    folds = np.full(len(df), -1, dtype=np.int64)

    try:
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (_, val_idx) in enumerate(skf.split(np.zeros(len(df)), y, groups=groups)):
            folds[val_idx] = fold_idx
    except ValueError:
        gkf = GroupKFold(n_splits=n_splits)
        for fold_idx, (_, val_idx) in enumerate(gkf.split(np.zeros(len(df)), y, groups=groups)):
            folds[val_idx] = fold_idx

    if (folds < 0).any():
        # Defensive: drop unassigned rows into the smallest fold so they stay in training in 4/5 folds.
        n_unassigned = int((folds < 0).sum())
        sizes = np.bincount(folds[folds >= 0], minlength=n_splits)
        smallest = int(sizes.argmin())
        print(
            f"[folds] WARN: {n_unassigned} rows could not be assigned by "
            f"StratifiedGroupKFold (likely singleton classes); routed to fold {smallest}."
        )
        folds[folds < 0] = smallest

    return pd.Series(folds, index=df.index, name="fold")
