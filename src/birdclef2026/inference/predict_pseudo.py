"""Pseudo-label R1: sliding-window predict on train_soundscapes → soft labels.

Used to bootstrap a second training round on top of the labeled train_audio
+ labeled train_soundscapes. The +0.01-0.02 LB lift documented in BirdCLEF
2024 1st place and 2025 2nd place both run this iteration step.

Pipeline
--------
1. Load N fold checkpoints (V2-S 5-fold ensemble, by default).
2. Walk every audio file under ``train_soundscapes/`` (unlabeled — the
   labeled subset listed in ``train_soundscapes_labels.csv`` is excluded
   so we don't overwrite hard labels with softer pseudo labels).
3. Slide a non-overlapping 5s window over each file, forward through the
   ensemble, average sigmoid probabilities across folds.
4. For each window, keep classes with ``prob >= prob_min`` (soft) and the
   top class always (primary). Write a pseudo-label CSV that the metadata
   builder picks up via ``data.pseudo_label_csv``.

Output CSV columns (one row per (filename, end_time) window kept):
    filename, end_time, primary_label, secondary_labels, prob_primary

`secondary_labels` is a space-separated string (matching the
train.csv format) of all classes above ``prob_min`` other than primary.

Usage:
    PYTHONPATH=src uv run python -m birdclef2026.inference.predict_pseudo \\
        --exp-dir outputs/exp/e01_v2s_5fold \\
        --ckpt-name swa.pt \\
        --defaults configs/data.yaml \\
        --out outputs/pseudo/r1.csv \\
        --prob-min 0.5
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .predict import (
    AUDIO_SUFFIXES,
    discover_checkpoints,
    load_sed_checkpoint,
)
from ..utils.audio import crop_or_pad, load_audio
from ..utils.config import load_yaml, resolve_data_root


def _list_unlabeled_soundscapes(
    soundscape_dir: Path, labeled_csv: Path | None,
) -> list[Path]:
    """All audio files under soundscape_dir minus those listed in labeled_csv."""
    files: list[Path] = []
    if soundscape_dir.exists():
        for suffix in AUDIO_SUFFIXES:
            files.extend(soundscape_dir.rglob(f"*{suffix}"))
    files = sorted(set(files))
    if labeled_csv is None or not labeled_csv.exists():
        return files
    df = pd.read_csv(labeled_csv)
    if "filename" not in df.columns:
        return files
    labeled_names = set(df["filename"].astype(str).tolist())
    return [p for p in files if p.name not in labeled_names]


@torch.no_grad()
def predict_file(
    models: Sequence[torch.nn.Module],
    path: Path,
    sample_rate: int,
    clip_seconds: float,
    device: torch.device,
    batch_size: int,
    amp_dtype: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a non-overlapping window over the file → (n_windows, n_classes) probs.

    Returns (end_times_sec, probs).
    """
    waveform = load_audio(path, sample_rate)
    length = int(round(sample_rate * clip_seconds))
    n = waveform.numel()
    if n < length:
        windows = [crop_or_pad(waveform, length, random_crop=False)]
        end_times = np.array([clip_seconds], dtype=np.float64)
    else:
        n_windows = n // length
        windows = []
        end_times = []
        for k in range(n_windows):
            start = k * length
            end_sample = start + length
            windows.append(waveform[start:end_sample].contiguous())
            end_times.append((end_sample) / sample_rate)
        end_times = np.array(end_times, dtype=np.float64)

    if amp_dtype == "bf16":
        ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    else:
        ctx = torch.autocast(device_type=device.type, enabled=False)

    n_models = len(models)
    all_probs: list[np.ndarray] = []
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        wave_batch = torch.stack(chunk).to(device, non_blocking=True)
        ensemble: torch.Tensor | None = None
        with ctx:
            for model in models:
                out = model(wave_batch)
                p = torch.sigmoid(out["clipwise_logits"].float())
                ensemble = p if ensemble is None else ensemble + p
        ensemble = (ensemble / n_models).cpu().numpy()
        all_probs.append(ensemble)
    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 0))
    return end_times, probs


def main() -> None:
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=False)
    grp.add_argument("--exp-dir", type=Path, default=None)
    grp.add_argument("--checkpoints", nargs="+", type=Path, default=None)
    p.add_argument("--ckpt-name", default="swa.pt")
    p.add_argument("--defaults", default="configs/data.yaml")
    p.add_argument("--out", type=Path, default=Path("outputs/pseudo/r1.csv"))
    p.add_argument(
        "--prob-min", type=float, default=0.5,
        help="Soft-label threshold. Classes with avg prob >= prob_min become positive.",
    )
    p.add_argument(
        "--keep-only-positive", action="store_true",
        help="Drop windows where no class clears prob_min (treat as nocall).",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--amp-dtype", default="bf16", choices=("bf16", "fp32"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.amp_dtype == "bf16":
        args.amp_dtype = "fp32"

    if args.checkpoints:
        ckpt_paths = [Path(x) for x in args.checkpoints]
    elif args.exp_dir:
        ckpt_paths = discover_checkpoints(args.exp_dir, args.ckpt_name)
    else:
        raise SystemExit("must pass --exp-dir or --checkpoints")
    print(f"[pseudo] {len(ckpt_paths)} checkpoint(s):")
    for cp in ckpt_paths:
        print(f"  {cp}")

    fallback_cfg = load_yaml(args.defaults)
    models: list[torch.nn.Module] = []
    target_columns: list[str] | None = None
    audio_cfg: dict | None = None
    for cp in ckpt_paths:
        m, t_cols, a_cfg = load_sed_checkpoint(cp, device, fallback_cfg=fallback_cfg)
        if target_columns is None:
            target_columns, audio_cfg = t_cols, a_cfg
        elif t_cols != target_columns:
            raise RuntimeError(f"{cp} target_columns differ from first checkpoint")
        models.append(m)

    assert target_columns is not None and audio_cfg is not None
    sample_rate = int(audio_cfg.get("sample_rate", 32000))
    clip_seconds = float(audio_cfg.get("clip_seconds", 5.0))

    data_root = resolve_data_root(fallback_cfg)
    paths_cfg = fallback_cfg.get("paths", {})
    soundscape_dir = data_root / paths_cfg.get("train_soundscapes", "train_soundscapes")
    labeled_csv = data_root / paths_cfg.get(
        "soundscape_labels_csv", "train_soundscapes_labels.csv"
    )
    files = _list_unlabeled_soundscapes(soundscape_dir, labeled_csv)
    print(
        f"[pseudo] {len(files)} unlabeled soundscape file(s) under {soundscape_dir}; "
        f"prob_min={args.prob_min}, keep_only_positive={args.keep_only_positive}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cls_arr = np.array(target_columns)

    n_kept = 0
    n_dropped = 0
    with args.out.open("w") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["filename", "end_time", "primary_label", "secondary_labels", "prob_primary"]
        )
        for path in tqdm(files, desc="pseudo"):
            end_times, probs = predict_file(
                models, path,
                sample_rate=sample_rate, clip_seconds=clip_seconds,
                device=device, batch_size=args.batch_size, amp_dtype=args.amp_dtype,
            )
            for win_idx, end_time in enumerate(end_times):
                row_probs = probs[win_idx]
                top_idx = int(np.argmax(row_probs))
                top_prob = float(row_probs[top_idx])
                pos_mask = row_probs >= args.prob_min
                if args.keep_only_positive and not pos_mask.any():
                    n_dropped += 1
                    continue
                pos_classes = cls_arr[pos_mask].tolist()
                primary = cls_arr[top_idx] if top_prob >= args.prob_min or not args.keep_only_positive else "nocall"
                secondary = [c for c in pos_classes if c != primary]
                writer.writerow([
                    path.name,
                    f"{end_time:.3f}",
                    primary,
                    " ".join(secondary),
                    f"{top_prob:.4f}",
                ])
                n_kept += 1

    print(f"[done] wrote {args.out}: kept={n_kept}, dropped={n_dropped}")


if __name__ == "__main__":
    main()
