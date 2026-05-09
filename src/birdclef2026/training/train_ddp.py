"""DDP training entrypoint for one fold of one experiment.

Usage (4×L40):
    torchrun --standalone --nproc_per_node=4 \\
        -m birdclef2026.training.train_ddp \\
        --config configs/exp/e01_v2s_5fold.yaml \\
        --fold 0

What the script does:
    1. Init DDP (NCCL); seed; resolve data root.
    2. Read fold csv (``outputs/folds/<exp.fold_csv>``); split into train_df / val_df.
    3. Build SEDDataset(s) + DataLoaders.
       Train uses DistributedSampler. Val runs on rank-0 only (no padding/dedup mess).
    4. Build SEDModel (timm backbone + LogMel + SpecAugment + AttHead + RandomFiltering).
    5. Build optimizer (AdamW or RAdam) + SequentialLR(LinearLR warmup + CosineAnnealingLR).
    6. Build FocalBCEWithLogits loss + EMA + bf16 autocast.
    7. Training loop with grad clip, EMA update, per-step scheduler.
    8. Rank-0 validation each epoch (uses EMA weights), CSV log, save best.
    9. After the last epoch, rank-0 saves OOF logits to ``outputs/exp/<exp_id>/fold_K/oof.npz``.

Outputs (per fold):
    outputs/exp/<exp_id>/fold_K/best.pt        # state_dict of model + ema
    outputs/exp/<exp_id>/fold_K/last.pt
    outputs/exp/<exp_id>/fold_K/log.csv        # epoch / loss / auc / lr
    outputs/exp/<exp_id>/fold_K/oof.npz        # sample_ids / logits / labels (post-EMA)
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Sequence

# Silence the torchaudio 2.8→2.9 `load_with_torchcodec` deprecation. The
# previous `PYTHONWARNINGS=ignore::UserWarning:torchaudio` env trick failed
# because Python's warning filter `module` field matches the *caller's*
# __name__ (here: birdclef2026.data.dataset), not torchaudio. Match by message
# regex instead, applied before torchaudio is imported transitively. fork-mode
# DataLoader workers inherit this filter automatically.
warnings.filterwarnings(
    "ignore",
    message=r"In 2\.9, this function's implementation will be changed",
    category=UserWarning,
)

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..data.dataset import SEDDataset
from ..data.transforms import BackgroundNoise
from ..metadata import build_train_metadata, get_target_columns
from ..models.losses import FocalBCEWithLogits
from ..models.sed import build_model_from_config
from ..utils.audio import LogMel  # noqa: F401  (used indirectly via SEDModel; kept for typing reference)
from ..utils.config import load_experiment_config, resolve_data_root
from ..utils.cv import read_fold_assignment
from ..utils.io import append_csv, ensure_dir, save_checkpoint, save_oof
from ..utils.metrics import macro_auc
from ..utils.seed import fold_seed, seed_everything
from .ema import ModelEMA


# --------------------------------------------------------------------------- #
# DDP helpers
# --------------------------------------------------------------------------- #


def _ddp_setup() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if dist.is_initialized() else 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_main(rank: int) -> bool:
    return rank == 0


# --------------------------------------------------------------------------- #
# Data construction
# --------------------------------------------------------------------------- #


def _materialize_metadata(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Use legacy ``birdclef2026.metadata`` to build the train_audio + soundscape frame.

    Mapping logic / taxonomy / label parsing is correct in the legacy module — we
    only avoid its broken ``split_train_val``.
    """
    data_root = resolve_data_root(cfg)
    target_columns = get_target_columns(data_root, cfg)
    df = build_train_metadata(data_root, cfg, target_columns)
    if df.empty:
        raise RuntimeError(f"No train rows resolved from {data_root}")
    return df, list(target_columns)


def _attach_fold(df: pd.DataFrame, fold_csv: Path) -> pd.DataFrame:
    """Left-join the fold csv (sample_id, fold) into the metadata frame.

    ``sample_id`` matches ``filename`` (with file extension). Rows missing from
    the fold csv are dropped (e.g., soundscape rows when fold csv only covers
    train_audio).
    """
    folds = read_fold_assignment(fold_csv)
    folds = folds[["sample_id", "fold"]].rename(columns={"sample_id": "filename"})
    merged = df.merge(folds, on="filename", how="inner")
    if merged.empty:
        raise RuntimeError(
            f"No rows matched between metadata.filename and fold csv at {fold_csv}"
        )
    return merged.reset_index(drop=True)


def _collect_noise_paths(cfg: dict) -> list[Path]:
    aug_cfg = cfg.get("augment", {}).get("background_noise", {}) or {}
    roots: Sequence[str] = aug_cfg.get("dirs", []) or []
    suffixes = tuple(aug_cfg.get("suffixes", [".wav", ".ogg", ".flac", ".mp3"]))
    paths: list[Path] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for p in root_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in suffixes:
                paths.append(p)
    return paths


def _build_loaders(
    cfg: dict,
    df: pd.DataFrame,
    target_columns: Sequence[str],
    fold: int,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DataLoader, SEDDataset, SEDDataset]:
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise RuntimeError(
            f"Empty split for fold {fold}: train={len(train_df)} val={len(val_df)}"
        )

    audio_cfg = cfg.get("audio", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    aug_cfg = cfg.get("augment", {}) or {}
    bg_cfg = aug_cfg.get("background_noise", {}) or {}

    bg_noise = None
    noise_paths = _collect_noise_paths(cfg)
    if bg_cfg.get("enabled", False) and noise_paths:
        bg_noise = BackgroundNoise(
            noise_paths=noise_paths,
            sample_rate=int(audio_cfg.get("sample_rate", 32000)),
            clip_seconds=float(audio_cfg.get("clip_seconds", 5.0)),
            min_amp=float(bg_cfg.get("min_amp", 0.25)),
            max_amp=float(bg_cfg.get("max_amp", 0.75)),
            p=float(bg_cfg.get("p", 0.5)),
            normalize=bool(bg_cfg.get("normalize", True)),
        )
    elif bg_cfg.get("enabled", False) and _is_main(rank):
        print(f"[warn] background_noise.enabled=True but no noise files found in {bg_cfg.get('dirs')}")

    mixup_cfg = aug_cfg.get("mixup", {}) or {}
    sample_rate = int(audio_cfg.get("sample_rate", 32000))
    clip_seconds = float(audio_cfg.get("clip_seconds", 5.0))

    train_ds = SEDDataset(
        df=train_df,
        target_columns=target_columns,
        sample_rate=sample_rate,
        clip_seconds=clip_seconds,
        training=True,
        mixup_p=float(mixup_cfg.get("p", 0.5)),
        mixup_alpha=mixup_cfg.get("alpha"),
        mixup_target_aggregation=str(mixup_cfg.get("target_aggregation", "sum")),
        secondary_weight=float(cfg.get("data", {}).get("secondary_weight", 0.3)),
        background_noise=bg_noise,
    )
    val_ds = SEDDataset(
        df=val_df,
        target_columns=target_columns,
        sample_rate=sample_rate,
        clip_seconds=clip_seconds,
        training=False,
        mixup_p=0.0,
        secondary_weight=float(cfg.get("data", {}).get("secondary_weight", 0.3)),
        background_noise=None,
    )

    bs = int(train_cfg.get("batch_size_per_gpu", 64))
    workers = int(train_cfg.get("num_workers", 8))

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
        prefetch_factor=int(train_cfg.get("prefetch_factor", 2)) if workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("val_batch_size", bs)),
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=workers > 0,
    )
    return train_loader, val_loader, train_ds, val_ds


# --------------------------------------------------------------------------- #
# Train / val loops
# --------------------------------------------------------------------------- #


def _autocast_ctx(amp_dtype: str):
    """Pick the autocast context. Only bf16 / fp32 are supported.

    fp16 was removed (H3) because the train loop has no GradScaler — fp16
    without a scaler diverges at the first step. bf16 is what L40 wants
    anyway and does not need a scaler.
    """
    if amp_dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if amp_dtype == "fp32":
        return torch.autocast(device_type="cuda", enabled=False)
    raise ValueError(
        f"unsupported amp_dtype {amp_dtype!r}; use 'bf16' (recommended on L40) or 'fp32'"
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    loss_fn,
    *,
    device,
    amp_dtype: str,
    grad_clip: float,
    ema: ModelEMA | None,
    epoch: int,
    rank: int,
):
    model.train()
    loader.sampler.set_epoch(epoch)
    running_loss = 0.0
    n_batches = 0
    t0 = time.time()
    for it, batch in enumerate(loader):
        wave, target, _meta = batch
        wave = wave.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(amp_dtype):
            out = model(wave)
            logits = out["clipwise_logits"]
            loss = loss_fn(logits, target)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        running_loss += float(loss.detach().cpu().item())
        n_batches += 1
        if rank == 0 and (it % 50 == 0):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  e{epoch} it={it}/{len(loader)} loss={loss.item():.4f} lr={lr:.2e} "
                f"elapsed={time.time() - t0:.0f}s"
            )
    return running_loss / max(1, n_batches)


@torch.no_grad()
def validate(
    eval_module: torch.nn.Module,
    loader,
    loss_fn,
    *,
    device,
    amp_dtype: str,
    n_classes: int,
):
    """Rank-0 only. Runs one full pass on val_loader and returns OOF arrays + AUC."""
    eval_module.eval()
    all_logits = []
    all_labels = []
    all_ids: list[str] = []
    running_loss = 0.0
    n_batches = 0
    for batch in loader:
        wave, target, meta = batch
        wave = wave.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with _autocast_ctx(amp_dtype):
            out = eval_module(wave)
            logits = out["clipwise_logits"]
            loss = loss_fn(logits, target)
        all_logits.append(logits.float().detach().cpu().numpy())
        all_labels.append(target.detach().cpu().numpy())
        all_ids.extend(list(meta["sample_id"]))
        running_loss += float(loss.detach().cpu().item())
        n_batches += 1
    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    val_loss = running_loss / max(1, n_batches)
    val_auc = macro_auc(labels_np, _sigmoid(logits_np))
    return val_loss, val_auc, np.asarray(all_ids), logits_np, labels_np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment yaml.")
    parser.add_argument("--fold", type=int, required=True, help="Fold index in [0, n_splits-1].")
    parser.add_argument(
        "--defaults",
        default="configs/data.yaml",
        help="Shared data config; merged under the experiment yaml.",
    )
    parser.add_argument(
        "--exp-id",
        default=None,
        help="Override experiment id (defaults to the exp yaml stem).",
    )
    args = parser.parse_args()

    rank, world_size, local_rank = _ddp_setup()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.benchmark = True

    cfg = load_experiment_config(args.config, defaults_path=args.defaults)
    seed_everything(fold_seed(int(cfg.get("seed", 42)), int(args.fold)))

    # H1: auto-load global mel mean/std if user didn't paste them into yaml.
    audio_cfg = cfg.setdefault("audio", {})
    if audio_cfg.get("global_mean") is None or audio_cfg.get("global_std") is None:
        mel_stats_path = Path(cfg.get("paths", {}).get("mel_stats", "outputs/stats/mel_stats.json"))
        if mel_stats_path.exists():
            stats = json.loads(mel_stats_path.read_text())
            audio_cfg["global_mean"] = float(stats["mean"])
            audio_cfg["global_std"] = float(stats["std"])
            if _is_main(rank):
                print(
                    f"[mel] auto-loaded global stats from {mel_stats_path}: "
                    f"mean={audio_cfg['global_mean']:.4f} std={audio_cfg['global_std']:.4f}"
                )
        else:
            if _is_main(rank):
                print(
                    f"[mel] WARN: {mel_stats_path} not found and yaml has no audio.global_mean/std. "
                    f"Falling back to per-sample mel norm — run `make mel-stats` for full quality."
                )

    exp_id = args.exp_id or Path(args.config).stem
    out_dir = ensure_dir(Path(cfg.get("paths", {}).get("output_dir", "outputs/exp")) / exp_id / f"fold_{args.fold}")
    if _is_main(rank):
        print(f"[exp] id={exp_id}  fold={args.fold}  world_size={world_size}  out_dir={out_dir}")

    # ---- data ----
    df, target_columns = _materialize_metadata(cfg)
    fold_csv = Path(cfg.get("paths", {}).get("fold_csv", "outputs/folds/folds.csv"))
    df = _attach_fold(df, fold_csv)
    train_loader, val_loader, _train_ds, _val_ds = _build_loaders(
        cfg, df, target_columns, args.fold, rank, world_size
    )

    # ---- model ----
    n_classes = len(target_columns)
    # Avoid 4 ranks racing to download timm pretrained weights from HF (or the
    # hf-mirror.com CN mirror): non-zero ranks wait at the first barrier while
    # rank 0 builds + populates ~/.cache/huggingface/hub; then everyone else
    # builds, hitting the on-disk cache (no network).
    if world_size > 1 and rank != 0:
        dist.barrier()
    model = build_model_from_config(cfg, num_classes=n_classes).to(device)
    if world_size > 1 and rank == 0:
        dist.barrier()

    if world_size > 1:
        # SyncBatchNorm gives all ranks a consistent BN running mean/var.
        # Required so EMA-on-rank0 (which copies BN buffers verbatim from the
        # rank-0 model) reflects global stats. (review_commit2.md H5)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ---- loss / optimizer / scheduler ----
    loss_cfg = cfg.get("loss", {}) or {}
    loss_fn = FocalBCEWithLogits(
        alpha=float(loss_cfg.get("alpha", 0.25)),
        gamma=float(loss_cfg.get("gamma", 2.0)),
        bce_weight=float(loss_cfg.get("bce_weight", 1.0)),
        focal_weight=float(loss_cfg.get("focal_weight", 1.0)),
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
        ls_mode=str(loss_cfg.get("ls_mode", "standard")),
    ).to(device)

    train_cfg = cfg.get("train", {}) or {}
    opt_cfg = cfg.get("optim", {}) or {}
    opt_name = str(opt_cfg.get("name", "adamw")).lower()
    lr = float(opt_cfg.get("lr", 1e-4))
    wd = float(opt_cfg.get("weight_decay", 1e-2))
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, eps=1e-8)
    elif opt_name == "radam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"unknown optim.name: {opt_name!r}")

    epochs = int(train_cfg.get("epochs", 50))
    warmup_epochs = float(train_cfg.get("warmup_epochs", 2.0))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(warmup_epochs * steps_per_epoch))
    lr_min = float(opt_cfg.get("lr_min", 1e-6))
    warm = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cos = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=lr_min)
    scheduler = SequentialLR(optimizer, schedulers=[warm, cos], milestones=[warmup_steps])

    # ---- EMA ----
    ema_decay = float(train_cfg.get("ema_decay", 0.999))
    ema: ModelEMA | None = ModelEMA(model, decay=ema_decay) if ema_decay > 0 else None
    if ema is not None:
        ema.module = ema.module.to(device)

    # ---- loop ----
    amp_dtype = str(train_cfg.get("amp_dtype", "bf16")).lower()
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_csv = out_dir / "log.csv"
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    oof_path = out_dir / "oof.npz"

    best_auc = -1.0
    final_oof = None  # (ids, logits, labels)

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn,
            device=device, amp_dtype=amp_dtype, grad_clip=grad_clip,
            ema=ema, epoch=epoch, rank=rank,
        )

        # validation on rank0 only — simple, no all_gather padding to dedup.
        if _is_main(rank):
            eval_module = ema.module if ema is not None else (
                model.module if isinstance(model, DDP) else model
            )
            val_loss, val_auc, ids, val_logits, val_labels = validate(
                eval_module, val_loader, loss_fn,
                device=device, amp_dtype=amp_dtype, n_classes=n_classes,
            )
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[fold {args.fold} e{epoch}] train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} lr={lr_now:.2e}"
            )
            append_csv(log_csv, {
                "fold": args.fold, "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_auc": round(val_auc, 6),
                "lr": float(lr_now),
            })
            ckpt = {
                "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "config": cfg,
                "fold": int(args.fold),
                "epoch": int(epoch),
                "val_auc": float(val_auc),
                "target_columns": list(target_columns),
            }
            save_checkpoint(ckpt, last_path)
            if val_auc > best_auc:
                best_auc = float(val_auc)
                save_checkpoint(ckpt, best_path)
                final_oof = (ids, val_logits, val_labels)

        if dist.is_initialized():
            dist.barrier()

    if _is_main(rank) and final_oof is not None:
        ids, val_logits, val_labels = final_oof
        # M5: sort by sample_id for deterministic cross-fold alignment downstream.
        order = np.argsort(ids)
        ids = ids[order]
        val_logits = val_logits[order]
        val_labels = val_labels[order]
        save_oof(oof_path, sample_ids=ids, logits=val_logits, labels=val_labels)
        # also write a one-line summary file for orchestration scripts
        (out_dir / "oof_auc.txt").write_text(f"{best_auc:.6f}\n")
        print(f"[done] best val_auc={best_auc:.4f}; wrote {oof_path}")

    _ddp_cleanup()


if __name__ == "__main__":
    main()
