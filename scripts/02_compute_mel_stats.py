#!/usr/bin/env python3
"""Compute global log-mel mean/std over the training set.

Replaces the legacy per-sample normalization (which destroys absolute energy
information that distinguishes faint insect calls from loud bird songs).

Streams files in random order, accumulates Welford-style running stats on the
log-mel spectrograms, and writes a small JSON to ``outputs/stats/mel_stats.json``.

Usage:
    PYTHONPATH=src uv run python scripts/02_compute_mel_stats.py \\
        --config configs/data.yaml \\
        --out outputs/stats/mel_stats.json \\
        --max-files 4000   # 4k files is enough for stable stats; default scans all
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from tqdm import tqdm

from birdclef2026.metadata import build_train_metadata, get_target_columns
from birdclef2026.utils.audio import LogMel, crop_or_pad, load_audio
from birdclef2026.utils.config import load_yaml, resolve_data_root
from birdclef2026.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out", default="outputs/stats/mel_stats.json")
    parser.add_argument("--max-files", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    audio_cfg = cfg.get("audio", {}) or {}
    sr = int(audio_cfg.get("sample_rate", 32000))
    sec = float(audio_cfg.get("clip_seconds", 5.0))
    length = int(round(sr * sec))

    data_root = resolve_data_root(cfg)
    target_columns = get_target_columns(data_root, cfg)
    df = build_train_metadata(data_root, cfg, target_columns)
    if df.empty:
        raise SystemExit(f"[ERR] no rows resolved from {data_root}; check data download")

    random.seed(args.seed)
    rows = df.to_dict(orient="records")
    random.shuffle(rows)
    if args.max_files > 0:
        rows = rows[: args.max_files]
    print(f"[info] computing mel stats over {len(rows):,} files (sr={sr}, clip={sec}s)")

    # CRITICAL: pass normalize=False so we get raw log-mel values out of
    # LogMel.forward. With normalize=True (default) and no global stats,
    # LogMel falls back to per-tensor (mel - mel.mean()) / mel.std(), which
    # makes every file mean=0/std=1 — accumulating those gives the bogus
    # mean≈0/std≈1 stats we observed before this fix.
    mel = LogMel(
        sample_rate=sr,
        n_fft=int(audio_cfg.get("n_fft", 2048)),
        hop_length=int(audio_cfg.get("hop_length", 512)),
        n_mels=int(audio_cfg.get("n_mels", 128)),
        f_min=float(audio_cfg.get("f_min", 20.0)),
        f_max=float(audio_cfg.get("f_max", sr / 2)),
        top_db=float(audio_cfg.get("top_db", 80.0)),
        global_mean=None,
        global_std=None,
        normalize=False,
    )

    # Vectorized two-pass: accumulate sum / sum_sq in float64.
    # log-mel values live in roughly [-80, 80] dB; ~150M samples is well within float64 precision.
    total = 0
    sum_x = 0.0
    sum_x2 = 0.0
    for row in tqdm(rows):
        try:
            wav = load_audio(row["path"], sr)
        except Exception:
            continue
        wav = crop_or_pad(wav, length, random_crop=True)
        spec = mel(wav).double().reshape(-1)
        total += int(spec.numel())
        sum_x += float(spec.sum().item())
        sum_x2 += float((spec * spec).sum().item())

    if total < 2:
        raise SystemExit("[ERR] not enough samples accumulated")
    mean = sum_x / total
    var = max(sum_x2 / total - mean * mean, 1e-12)
    std = math.sqrt(var)
    n = total

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    payload = {
        "n_values": int(n),
        "n_files": int(len(rows)),
        "mean": float(mean),
        "std": float(std),
        "audio_config": {
            "sample_rate": sr,
            "clip_seconds": sec,
            "n_fft": int(audio_cfg.get("n_fft", 2048)),
            "hop_length": int(audio_cfg.get("hop_length", 512)),
            "n_mels": int(audio_cfg.get("n_mels", 128)),
            "f_min": float(audio_cfg.get("f_min", 20.0)),
            "f_max": float(audio_cfg.get("f_max", sr / 2)),
            "top_db": float(audio_cfg.get("top_db", 80.0)),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(
        f"[done] {out_path}  mean={mean:.4f}  std={std:.4f}  "
        f"(n_values={n:,}, n_files={len(rows):,})"
    )
    print(
        "[next] paste mean/std into configs/data.yaml under `audio.global_mean` / `audio.global_std`"
    )


if __name__ == "__main__":
    main()
