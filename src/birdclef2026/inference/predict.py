"""Test-soundscape inference with sliding-window averaging + checkpoint ensemble.

For each test audio file, run a non-overlapping sliding window of
``clip_seconds`` (default 5 s) windows, forward each batch through every
loaded checkpoint, average sigmoid probabilities across windows AND
checkpoints, and write a Kaggle-format submission.csv whose row order
matches ``sample_submission.csv`` exactly.

Usage:
    PYTHONPATH=src uv run python -m birdclef2026.inference.predict \\
        --exp-dir outputs/exp/e01_v2s_5fold \\
        --ckpt-name swa.pt \\
        --defaults configs/data.yaml \\
        --output submission.csv

    # or pass explicit checkpoints:
    PYTHONPATH=src uv run python -m birdclef2026.inference.predict \\
        --checkpoints outputs/exp/e01_v2s_5fold/fold_0/swa.pt \\
                      outputs/exp/e01_v2s_5fold/fold_1/swa.pt \\
        --defaults configs/data.yaml \\
        --output submission.csv

Notes
-----
* Equal-weight ensemble across checkpoints. Use ``ensemble.py`` (separate
  script) for weighted / rank-blended fusion later.
* Each checkpoint is loaded fresh on CPU and moved to GPU once; held in
  memory simultaneously, so V2-S × 5 ≈ 110 MB is fine.
* Sliding-window average is in **probability space** (post-sigmoid).
  This matches what ``training/train_ddp.py::validate`` does for OOF
  aggregation, so OOF metric == test metric pattern.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..metadata import get_target_columns, parse_row_id
from ..models.sed import build_model_from_config
from ..utils.audio import crop_or_pad, load_audio
from ..utils.config import load_experiment_config, load_yaml, resolve_data_root
from ..utils.io import ensure_dir

AUDIO_SUFFIXES = (".ogg", ".wav", ".flac", ".mp3", ".m4a")


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #


def load_sed_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    *,
    fallback_cfg: dict,
) -> tuple[torch.nn.Module, list[str], dict]:
    """Load one fold's checkpoint; return (model, target_columns, audio_cfg).

    Checkpoints written by ``training/train_ddp.py`` carry a copy of the
    full experiment config under ``ckpt['config']`` — we use that to
    rebuild the SEDModel so backbone / mel / head choices match training.
    Falls back to ``fallback_cfg`` if a checkpoint pre-dates the convention.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("config") or fallback_cfg
    target_columns = list(state.get("target_columns") or get_target_columns(
        resolve_data_root(cfg), cfg
    ))
    # Saved cfg has training-time pretrained=True / pretrained_backbone_path=...,
    # but at load time the state_dict overwrites everything. Skip both so the
    # model builds offline (no timm download, no missing .pth lookup).
    cfg = dict(cfg)
    cfg["model"] = dict(cfg.get("model") or {})
    cfg["model"]["pretrained"] = False
    cfg["model"]["pretrained_backbone_path"] = None
    model = build_model_from_config(cfg, num_classes=len(target_columns))
    raw_state = state.get("model")
    if raw_state is None:
        raise RuntimeError(f"{ckpt_path} has no 'model' key — wrong checkpoint format")
    # Strip any DDP 'module.' prefix that snuck through.
    cleaned = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in raw_state.items()
    }
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    audio_cfg = (cfg.get("audio") or {}).copy()
    return model, target_columns, audio_cfg


def discover_checkpoints(exp_dir: Path, ckpt_name: str) -> list[Path]:
    """Find all ``fold_*/<ckpt_name>`` under ``exp_dir`` (sorted by fold idx)."""
    fold_dirs = sorted(
        (p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")),
        key=lambda p: int(p.name.split("_", 1)[1]),
    )
    found = [d / ckpt_name for d in fold_dirs if (d / ckpt_name).exists()]
    if not found:
        raise FileNotFoundError(
            f"no {ckpt_name!r} files under {exp_dir}/fold_*/ — train first or pass --checkpoints"
        )
    return found


# --------------------------------------------------------------------------- #
# Test rows: build from sample_submission.csv (preferred) or scan dir.
# --------------------------------------------------------------------------- #


def _find_audio_file(test_dir: Path, audio_id: str) -> Path | None:
    for suffix in AUDIO_SUFFIXES:
        candidate = test_dir / f"{audio_id}{suffix}"
        if candidate.exists():
            return candidate
    if test_dir.exists():
        for suffix in AUDIO_SUFFIXES:
            matches = list(test_dir.rglob(f"{audio_id}{suffix}"))
            if matches:
                return matches[0]
    return None


def build_test_rows(
    sample_submission_path: Path,
    test_dir: Path,
    clip_seconds: float,
    segments_per_file: int = 12,
) -> tuple[list[dict], list[str]]:
    """Returns (rows, row_id_order).

    Each row: {row_id, audio_id, end_time, path}. row_id_order keeps the
    sample-submission row sequence so the output CSV stays in the exact
    order Kaggle expects.
    """
    rows: list[dict] = []
    row_id_order: list[str] = []

    if sample_submission_path.exists():
        sample = pd.read_csv(sample_submission_path)
        for row_id in sample["row_id"].astype(str):
            audio_id, end_time = parse_row_id(row_id)
            audio_path = _find_audio_file(test_dir, audio_id)
            rows.append({
                "row_id": row_id,
                "audio_id": audio_id,
                "end_time": end_time,
                "path": audio_path,
            })
            row_id_order.append(row_id)
        return rows, row_id_order

    files: list[Path] = []
    if test_dir.exists():
        for suffix in AUDIO_SUFFIXES:
            files.extend(test_dir.rglob(f"*{suffix}"))
    if not files:
        raise RuntimeError(f"no sample_submission.csv and no audio in {test_dir}")
    for path in sorted(files):
        audio_id = path.stem
        for idx in range(1, segments_per_file + 1):
            end_time = float(idx * clip_seconds)
            row_id = f"{audio_id}_{int(end_time)}"
            rows.append({
                "row_id": row_id,
                "audio_id": audio_id,
                "end_time": end_time,
                "path": path,
            })
            row_id_order.append(row_id)
    return rows, row_id_order


# --------------------------------------------------------------------------- #
# Inference: per-file batched forward over all rows assigned to that file.
# --------------------------------------------------------------------------- #


def _extract_window(
    waveform: torch.Tensor,
    sample_rate: int,
    clip_seconds: float,
    end_time: float,
) -> torch.Tensor:
    """Extract a fixed-length window ending at ``end_time`` seconds."""
    length = int(round(sample_rate * clip_seconds))
    end_sample = int(round(float(end_time) * sample_rate))
    start_sample = end_sample - length
    return crop_or_pad(waveform, length, random_crop=False, start_sample=start_sample)


@torch.no_grad()
def predict_for_models(
    models: Sequence[torch.nn.Module],
    rows: Sequence[dict],
    *,
    sample_rate: int,
    clip_seconds: float,
    device: torch.device,
    batch_size: int = 32,
    amp_dtype: str = "bf16",
) -> dict[str, np.ndarray]:
    """Equal-weight probability ensemble across ``models``.

    For each unique audio file: load once, build all the row windows
    in one go, forward each batch through every model, accumulate
    sigmoid probs, divide by ``len(models)``. Skips rows whose path
    is None (missing file) — caller fills those with zeros.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    missing: list[str] = []
    for row in rows:
        if row["path"] is None:
            missing.append(row["row_id"])
        else:
            grouped[str(row["path"])].append(row)
    if missing:
        print(f"[infer] WARN: {len(missing)} rows have no audio path; filling zeros")

    if amp_dtype == "bf16":
        autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    elif amp_dtype == "fp32":
        autocast_ctx = torch.autocast(device_type=device.type, enabled=False)
    else:
        raise ValueError(f"unsupported amp_dtype {amp_dtype!r}")

    n_models = len(models)
    if n_models == 0:
        raise ValueError("no models passed to predict_for_models")
    n_classes = None
    predictions: dict[str, np.ndarray] = {}

    for path, group_rows in tqdm(grouped.items(), desc="infer files"):
        waveform = load_audio(path, sample_rate)
        windows = [
            _extract_window(waveform, sample_rate, clip_seconds, row["end_time"])
            for row in group_rows
        ]
        # batch-forward over the file's windows
        for start in range(0, len(windows), batch_size):
            chunk = windows[start : start + batch_size]
            batch_rows = group_rows[start : start + batch_size]
            wave_batch = torch.stack(chunk).to(device, non_blocking=True)
            ensemble_probs: torch.Tensor | None = None
            with autocast_ctx:
                for model in models:
                    out = model(wave_batch)
                    p = torch.sigmoid(out["clipwise_logits"].float())
                    if ensemble_probs is None:
                        ensemble_probs = p
                    else:
                        ensemble_probs = ensemble_probs + p
            ensemble_probs = (ensemble_probs / n_models).cpu().numpy()
            if n_classes is None:
                n_classes = ensemble_probs.shape[1]
            for row, prob in zip(batch_rows, ensemble_probs):
                predictions[row["row_id"]] = prob

    return predictions


# --------------------------------------------------------------------------- #
# Submission writer
# --------------------------------------------------------------------------- #


def predictions_to_submission(
    rows: Sequence[dict],
    row_id_order: Sequence[str],
    predictions: dict[str, np.ndarray],
    model_columns: Sequence[str],
    output_columns: Sequence[str],
) -> pd.DataFrame:
    """Build the Kaggle-format submission DataFrame.

    Reindexes per-row probability vectors (in ``model_columns`` order)
    into ``output_columns`` order so the column layout matches whatever
    sample_submission.csv expects, even if training used a different
    species ordering.
    """
    model_to_idx = {label: idx for idx, label in enumerate(model_columns)}
    n_classes = len(model_columns)
    zeros = np.zeros(n_classes, dtype=np.float32)

    data: dict[str, list] = {"row_id": list(row_id_order)}
    for col in output_columns:
        data[col] = []
    # row_id_order may contain the same row_id only once; iterate it.
    for row_id in row_id_order:
        probs = predictions.get(row_id, zeros)
        for col in output_columns:
            idx = model_to_idx.get(col)
            data[col].append(float(probs[idx]) if idx is not None else 0.0)
    return pd.DataFrame(data)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=False)
    grp.add_argument(
        "--exp-dir", type=Path, default=None,
        help="Parent dir holding fold_0/, fold_1/, ... — auto-discovers <ckpt-name>.",
    )
    grp.add_argument(
        "--checkpoints", nargs="+", type=Path, default=None,
        help="Explicit checkpoint paths (overrides --exp-dir).",
    )
    p.add_argument(
        "--ckpt-name", default="swa.pt",
        help="Filename to look for under each fold dir (default: swa.pt; fall back to best.pt).",
    )
    p.add_argument(
        "--defaults", default="configs/data.yaml",
        help="Shared data yaml; only used when checkpoint metadata is missing.",
    )
    p.add_argument("--output", type=Path, default=Path("submission.csv"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--amp-dtype", default="bf16", choices=("bf16", "fp32"),
        help="bf16 on Ada/L40 is fastest; fp32 for CPU-only / Kaggle non-GPU.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.amp_dtype == "bf16":
        # bf16 on CPU is fine on Sapphire Rapids but Kaggle CPU is older —
        # auto-fallback to fp32 to avoid a confusing slowdown / crash.
        print("[infer] cpu device — forcing amp_dtype=fp32")
        args.amp_dtype = "fp32"

    # Resolve checkpoint list.
    if args.checkpoints:
        ckpt_paths = [Path(p) for p in args.checkpoints]
    elif args.exp_dir:
        ckpt_paths = discover_checkpoints(args.exp_dir, args.ckpt_name)
        if not ckpt_paths:
            print(f"[infer] no {args.ckpt_name!r} found, falling back to best.pt")
            ckpt_paths = discover_checkpoints(args.exp_dir, "best.pt")
    else:
        raise SystemExit("must pass --exp-dir or --checkpoints")
    print(f"[infer] loading {len(ckpt_paths)} checkpoint(s):")
    for p in ckpt_paths:
        print(f"  {p}")

    # Load all checkpoints. First one defines target_columns / audio_cfg; we
    # require the rest to match.
    fallback_cfg = load_yaml(args.defaults)
    models: list[torch.nn.Module] = []
    target_columns: list[str] | None = None
    audio_cfg: dict | None = None
    for path in ckpt_paths:
        model, t_cols, a_cfg = load_sed_checkpoint(path, device, fallback_cfg=fallback_cfg)
        if target_columns is None:
            target_columns = t_cols
            audio_cfg = a_cfg
        else:
            if t_cols != target_columns:
                raise RuntimeError(
                    f"{path} target_columns differ from first checkpoint — "
                    f"can't blend across class orderings"
                )
        models.append(model)

    assert target_columns is not None and audio_cfg is not None
    sample_rate = int(audio_cfg.get("sample_rate", 32000))
    clip_seconds = float(audio_cfg.get("clip_seconds", 5.0))

    # Build test rows.
    data_root = resolve_data_root(fallback_cfg)
    paths_cfg = fallback_cfg.get("paths", {})
    sample_submission_path = data_root / paths_cfg.get("sample_submission", "sample_submission.csv")
    test_dir = data_root / paths_cfg.get("test_soundscapes", "test_soundscapes")
    rows, row_id_order = build_test_rows(
        sample_submission_path, test_dir, clip_seconds=clip_seconds,
    )
    print(
        f"[infer] {len(rows)} rows over {len(set(r['audio_id'] for r in rows))} "
        f"audio files; sample_rate={sample_rate} clip={clip_seconds}s"
    )

    predictions = predict_for_models(
        models, rows,
        sample_rate=sample_rate, clip_seconds=clip_seconds,
        device=device, batch_size=args.batch_size, amp_dtype=args.amp_dtype,
    )

    # Output column order from sample_submission (or the model's order if missing).
    output_columns = get_target_columns(data_root, fallback_cfg)
    submission = predictions_to_submission(
        rows, row_id_order, predictions,
        model_columns=target_columns,
        output_columns=output_columns,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True) if args.output.parent != Path(".") else None
    submission.to_csv(args.output, index=False, float_format="%.8f")
    print(f"[done] wrote {args.output} shape={submission.shape}")


if __name__ == "__main__":
    main()
