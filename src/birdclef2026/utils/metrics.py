"""Macro-averaged ROC-AUC + per-class breakdown.

BirdCLEF official metric is **macro AUC across the species columns**, with
classes that contain only one label value (all-positive / all-negative) being
undefined and dropped from the average.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def macro_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    return_per_class: bool = False,
):
    """Macro AUC, skipping classes with only a single label value.

    Args:
        y_true:  shape (N, C), 0/1 multi-label targets.
        y_score: shape (N, C), real-valued scores (sigmoid probabilities or logits).
        return_per_class: if True, return ``(macro, per_class)`` where ``per_class[c]``
            is np.nan for skipped classes.

    Returns:
        float or (float, np.ndarray)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}")
    # ROC-AUC needs binary labels. Our dataset emits soft secondary labels
    # (e.g. 0.3) for the secondary_labels column — treat any positive value as
    # a positive class for evaluation. Hard 0/1 labels pass through unchanged.
    y_true_bin = (y_true > 0).astype(np.int8)
    n_classes = y_true_bin.shape[1]
    per_class = np.full(n_classes, np.nan, dtype=np.float64)
    for c in range(n_classes):
        col = y_true_bin[:, c]
        if col.min() != col.max():
            per_class[c] = roc_auc_score(col, y_score[:, c])
    valid = per_class[~np.isnan(per_class)]
    macro = float(valid.mean()) if valid.size else float("nan")
    if return_per_class:
        return macro, per_class
    return macro


def labelwise_auc(y_true, y_score) -> float:
    """Backward-compatible alias for legacy callers (``birdclef2026.metrics``)."""
    return macro_auc(y_true, y_score)
