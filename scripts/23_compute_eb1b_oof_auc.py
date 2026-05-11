#!/usr/bin/env python3
"""Pre-compute per-class OOF ROC-AUC for the eB1b fold-0 SWA model.

The c0 deep-ensemble kernel (b07) uses this per-class AUC as a trust
weight when blending the NFNet branch into A34's 3-way rank fusion:

    w_d_class = clip((auc - 0.5) / 0.4, 0.0, 1.25)

Classes where eB1b is no better than random (AUC <= 0.5) contribute 0
to the blend; classes where it is strong (AUC >= 0.9) contribute fully.
Pre-computing here avoids shipping fold labels into the Kaggle kernel
and removes ~30 s of sklearn work from the scoring run.

The output ``nfnet_oof_auc.npy`` is then packaged alongside ``swa.pt``
into the ``winbeaux/birdclef2026-eb1b-pkg-ckpt`` dataset by
``scripts/22_kaggle_submit.py``.

Strict-fail: any inconsistency between the OOF tensor and the
checkpoint's target_columns aborts the script — we want a clean signal
in the kernel, not silent weight scrambling.

Usage:
    cd birdclef-2026
    uv run python scripts/23_compute_eb1b_oof_auc.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
FOLD_DIR = ROOT / "outputs/exp/eB1b_nfnet_bgnoise/fold_0"
OOF_NPZ = FOLD_DIR / "swa_oof.npz"
CKPT = FOLD_DIR / "swa.pt"
OUT_NPY = FOLD_DIR / "nfnet_oof_auc.npy"


def main() -> None:
    assert OOF_NPZ.is_file(), f"missing OOF: {OOF_NPZ}"
    assert CKPT.is_file(), f"missing ckpt: {CKPT}"

    d = np.load(OOF_NPZ)
    y = d["labels"]          # (N, C) float (soft for secondaries)
    logits = d["logits"]     # (N, C)
    N, C = y.shape
    print(f"[oof] loaded {OOF_NPZ.name}: N={N} C={C}")

    # Sanity: ckpt's target_columns should match the OOF column count, and
    # be the same ordering the inference path will use. We don't need the
    # names here (the kernel will reorder by target_cols → PRIMARY_LABELS),
    # but mismatching widths is an immediate bug.
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    target_cols = list(state.get("target_columns") or [])
    assert len(target_cols) == C, (
        f"target_columns len {len(target_cols)} != OOF cols {C}"
    )

    yhat = 1.0 / (1.0 + np.exp(-logits))  # sigmoid; AUC is monotone so OK
    y_bin = (y >= 0.5).astype(np.int8)
    pos = y_bin.sum(axis=0)

    aucs = np.full(C, 0.5, dtype=np.float64)
    n_skipped = 0
    for c in range(C):
        if pos[c] == 0 or pos[c] == N:
            n_skipped += 1
            continue
        aucs[c] = roc_auc_score(y_bin[:, c], yhat[:, c])

    aucs = aucs.astype(np.float32)
    np.save(OUT_NPY, aucs)

    print(f"[oof] saved {OUT_NPY}")
    print(
        f"[oof] AUC stats: mean={aucs.mean():.4f}"
        f"  min={aucs.min():.3f}"
        f"  median={np.median(aucs):.3f}"
        f"  max={aucs.max():.3f}"
    )
    print(
        f"[oof] classes >0.9: {int((aucs > 0.9).sum())}/{C}"
        f"  >0.95: {int((aucs > 0.95).sum())}/{C}"
        f"  <0.5: {int((aucs < 0.5).sum())}/{C}"
        f"  =0.5 (skipped degenerate): {n_skipped}/{C}"
    )


if __name__ == "__main__":
    main()
