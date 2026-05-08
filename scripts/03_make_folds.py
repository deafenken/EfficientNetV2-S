#!/usr/bin/env python3
"""Build 5-fold StratifiedGroupKFold csv from train.csv.

Replaces the broken ``birdclef2026.metadata.split_train_val`` (random split,
same recording across train/val). Output csv has columns
``[sample_id, fold, primary_label, filename, path]`` and is consumed by
``birdclef2026.training.train_ddp``.

Usage:
    PYTHONPATH=src uv run python scripts/03_make_folds.py \\
        --config configs/data.yaml \\
        --out outputs/folds/folds.csv \\
        --n-splits 5 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from birdclef2026.data.folds import make_folds
from birdclef2026.metadata import build_train_metadata, get_target_columns
from birdclef2026.utils.config import load_yaml, resolve_data_root
from birdclef2026.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out", default="outputs/folds/folds.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-soundscapes",
        action="store_true",
        help="Also include rows from train_soundscapes_labels.csv (for R1+ pseudo runs).",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.include_soundscapes:
        cfg.setdefault("data", {})["include_train_soundscapes"] = True

    data_root = resolve_data_root(cfg)
    target_columns = get_target_columns(data_root, cfg)
    df = build_train_metadata(data_root, cfg, target_columns)
    if df.empty:
        raise SystemExit(f"[ERR] no rows resolved from {data_root}; check data download")
    print(f"[info] resolved {len(df):,} training rows from {data_root}")

    folds = make_folds(
        df, n_splits=int(args.n_splits), seed=int(args.seed),
        stratify_col="primary_label", group_col="filename",
    )
    df = df.assign(fold=folds.values)
    out = pd.DataFrame({
        "sample_id": df["filename"].astype(str),
        "fold": df["fold"].astype(int),
        "primary_label": df["primary_label"].astype(str),
        "filename": df["filename"].astype(str),
        "path": df["path"].astype(str),
        "source": df["source"].astype(str),
    })

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    out.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path}  rows={len(out):,}  classes={len(target_columns)}")
    counts = out["fold"].value_counts().sort_index()
    print(f"[info] fold sizes: {dict(counts)}")


if __name__ == "__main__":
    main()
