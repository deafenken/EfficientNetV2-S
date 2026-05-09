"""Assemble per-fold OOF arrays into a single cross-validated OOF prediction.

Each fold writes ``outputs/exp/<exp_id>/fold_K/{oof,swa_oof}.npz`` with
``(sample_ids, logits, labels)`` over its val rows. Across 5 folds these
union to the full train_audio set (each row predicted exactly once,
by the fold where it lives in the val split).

This script concatenates the per-fold arrays, dedups any overlap (rare
— happens only if folds.csv was rebuilt mid-training), recomputes the
macro AUC over the combined set, and writes
``outputs/exp/<exp_id>/oof.npz`` + ``oof_auc.txt``.

Usage:
    PYTHONPATH=src uv run python -m birdclef2026.inference.assemble_oof \\
        --exp-dir outputs/exp/e01_v2s_5fold \\
        --kind swa  # 'best' | 'swa'

The combined OOF is the input to:
- 5-fold ensemble stacking
- pseudo-label generation
- per-class threshold calibration
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..utils.metrics import macro_auc


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def assemble(exp_dir: Path, kind: str = "swa") -> tuple[Path, float]:
    """Concatenate fold_*/{kind}_oof.npz (or oof.npz) → exp_dir/{kind}_oof.npz."""
    if kind == "best":
        fold_glob = "fold_*/oof.npz"
        out_name = "oof.npz"
        auc_name = "oof_auc.txt"
    elif kind == "swa":
        fold_glob = "fold_*/swa_oof.npz"
        out_name = "swa_oof.npz"
        auc_name = "swa_oof_auc.txt"
    else:
        raise ValueError(f"kind must be 'best' | 'swa', got {kind!r}")

    fold_paths = sorted(exp_dir.glob(fold_glob))
    if not fold_paths:
        raise FileNotFoundError(
            f"no {fold_glob} under {exp_dir} — train all folds first"
        )
    print(f"[assemble] kind={kind}  {len(fold_paths)} fold(s):")
    for p in fold_paths:
        print(f"  {p}")

    all_ids: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for p in fold_paths:
        d = np.load(p, allow_pickle=False)
        all_ids.append(np.asarray(d["sample_ids"]))
        all_logits.append(np.asarray(d["logits"]))
        all_labels.append(np.asarray(d["labels"]))
    ids = np.concatenate(all_ids)
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Dedup by sample_id — keep the first occurrence (folds shouldn't
    # share val rows; if they do, log it).
    _, uniq_idx, counts = np.unique(ids, return_index=True, return_counts=True)
    if (counts > 1).any():
        n_dup = int((counts > 1).sum())
        print(f"[assemble] WARN: {n_dup} sample_ids appear in multiple folds; keeping first")
    ids = ids[uniq_idx]
    logits = logits[uniq_idx]
    labels = labels[uniq_idx]

    order = np.argsort(ids)
    ids = ids[order]
    logits = logits[order]
    labels = labels[order]

    probs = _sigmoid(logits)
    auc = macro_auc(labels, probs)

    out_path = exp_dir / out_name
    np.savez(out_path, sample_ids=ids, logits=logits, labels=labels)
    (exp_dir / auc_name).write_text(f"{auc:.6f}\n")
    print(
        f"[done] wrote {out_path} ({len(ids)} rows, "
        f"{labels.shape[1]} classes), macro_auc={auc:.4f}"
    )
    return out_path, float(auc)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", type=Path, required=True)
    p.add_argument("--kind", choices=("best", "swa"), default="swa")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    assemble(args.exp_dir, kind=args.kind)


if __name__ == "__main__":
    main()
