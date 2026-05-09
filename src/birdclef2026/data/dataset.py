"""Multi-label SED Dataset returning raw waveforms.

Returns ``(waveform, target, sample_id)`` per item. Mel-spectrogram extraction
and on-GPU augmentations (SpecAugment, RandomFiltering) live inside the model
so this dataset stays CPU-side and lightweight.

Args:
    df:               metadata frame with columns
                      ['path', 'filename', 'labels', 'primary_label', 'end_time', 'source'].
    target_columns:   list of 234 species names (column order in submission CSV).
    sample_rate:      32000.
    clip_seconds:     5.0 — fixed crop length.
    training:         True for train fold (random crop + augmentations).
    mixup_p:          probability of MixUp; 0 disables.
    mixup_alpha:      None for 50/50 wave avg (BirdCLEF 25 2nd default).
    mixup_target_aggregation: 'sum' (clamp 0..1, probabilistic OR), 'max', or 'beta'.
    secondary_weight: 0.3 by default (soft-label for ``secondary_labels``).
    background_noise: optional ``BackgroundNoise`` augmentor (sample-level).
"""

from __future__ import annotations

import math
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils.audio import crop_or_pad, extract_window, load_audio
from .transforms import BackgroundNoise, mixup_pair


def _is_nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


class SEDDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_columns: Sequence[str],
        *,
        sample_rate: int = 32000,
        clip_seconds: float = 5.0,
        training: bool = True,
        mixup_p: float = 0.5,
        mixup_alpha: float | None = None,
        mixup_target_aggregation: str = "sum",
        secondary_weight: float = 0.3,
        background_noise: BackgroundNoise | None = None,
        primary_only_for_soundscape: bool = False,
        val_n_crops: int = 1,
    ):
        if df.empty:
            raise ValueError("SEDDataset got an empty dataframe")
        for col in ("path", "labels", "primary_label", "end_time"):
            if col not in df.columns:
                raise KeyError(f"SEDDataset df is missing column '{col}'")

        self.df = df.reset_index(drop=True)
        self.target_columns = list(target_columns)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.target_columns)}
        self.sample_rate = int(sample_rate)
        self.clip_seconds = float(clip_seconds)
        self.length_samples = int(round(self.sample_rate * self.clip_seconds))

        self.training = bool(training)
        self.mixup_p = float(mixup_p) if training else 0.0
        self.mixup_alpha = mixup_alpha
        self.mixup_target_aggregation = mixup_target_aggregation
        self.secondary_weight = float(secondary_weight)
        self.background_noise = background_noise if training else None
        self.primary_only_for_soundscape = bool(primary_only_for_soundscape)
        # Val multi-crop: emit K uniformly-spaced 5s windows per recording so
        # downstream val averaging matches the inference sliding-window pattern.
        # Forced to 1 in train mode (random crop already gives diverse windows).
        self.val_n_crops = max(1, int(val_n_crops)) if not self.training else 1

    def __len__(self) -> int:
        return len(self.df) * self.val_n_crops

    # ------------------------------------------------------------------ #
    # core: load + crop + multi-hot target.
    # Bad-file resilient: a single corrupt .ogg should not kill a worker mid-epoch.
    # ``_load_raw`` skips background-noise so MixUp can apply noise *once*
    # after the two waves are combined (M1).
    # ------------------------------------------------------------------ #
    _MAX_LOAD_RETRIES = 3

    def _load_raw(
        self, idx: int, *, _retries_left: int | None = None,
        val_crop_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if _retries_left is None:
            _retries_left = self._MAX_LOAD_RETRIES
        row = self.df.iloc[idx]
        try:
            waveform = load_audio(row["path"], self.sample_rate)
        except IOError:
            if _retries_left > 0:
                new_idx = int(torch.randint(0, len(self.df), (1,)).item())
                if new_idx == idx:
                    new_idx = (idx + 1) % len(self.df)
                return self._load_raw(
                    new_idx, _retries_left=_retries_left - 1,
                    val_crop_idx=val_crop_idx,
                )
            waveform = torch.zeros(self.length_samples, dtype=torch.float32)

        end_time = row["end_time"]
        if not _is_nan(end_time):
            waveform = extract_window(
                waveform, self.sample_rate, self.clip_seconds,
                end_time=float(end_time), random_crop=False,
            )
        elif self.training:
            # Train: random crop somewhere along the clip — gives the model
            # diverse windows including the bird call.
            waveform = crop_or_pad(
                waveform, self.length_samples, random_crop=True,
            )
        else:
            # Val: train_audio rows have no end_time annotation. With
            # val_n_crops=K>1 we deterministically pick the val_crop_idx-th
            # of K uniformly-spaced 5s windows along the recording, so the
            # outer loop covers the full clip and validate() can average per
            # recording (matches inference sliding-window). Falls back to
            # center crop when K=1.
            n = waveform.numel()
            length = self.length_samples
            if n <= length:
                start = 0
            elif self.val_n_crops > 1 and val_crop_idx is not None:
                k = self.val_n_crops
                # k offsets in [0, n - length]; for k=1 collapse to center.
                if k == 1:
                    start = max(0, (n - length) // 2)
                else:
                    span = n - length
                    start = int(round(val_crop_idx * span / (k - 1)))
                start = max(0, min(start, n - length))
            else:
                start = max(0, (n - length) // 2)
            waveform = crop_or_pad(
                waveform, self.length_samples, random_crop=False,
                start_sample=start,
            )

        target = self._multi_hot(row)
        meta = {
            "sample_id": str(row.get("filename", row.get("path", idx))),
            "primary_label": str(row.get("primary_label", "")),
            "source": str(row.get("source", "")),
        }
        return waveform, target, meta

    def _load_one(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Single-sample load WITH background noise (used when no MixUp partner)."""
        waveform, target, meta = self._load_raw(idx)
        if self.training and self.background_noise is not None:
            waveform = self.background_noise(waveform)
        return waveform, target, meta

    def _multi_hot(self, row) -> torch.Tensor:
        target = torch.zeros(len(self.target_columns), dtype=torch.float32)
        labels = row["labels"] if isinstance(row["labels"], (list, tuple, set)) else []
        primary = str(row.get("primary_label", "")).strip()

        if self.primary_only_for_soundscape and str(row.get("source", "")) == "train_soundscapes":
            # Soundscape labels are window-level and already curated; treat them as hard.
            for label in labels:
                idx = self.label_to_idx.get(str(label))
                if idx is not None:
                    target[idx] = 1.0
            return target

        # train_audio: primary = 1.0, secondary = secondary_weight
        for label in labels:
            label = str(label).strip()
            idx = self.label_to_idx.get(label)
            if idx is None:
                continue
            if label == primary:
                target[idx] = 1.0
            else:
                target[idx] = max(float(target[idx].item()), self.secondary_weight)
        if primary and primary in self.label_to_idx and target[self.label_to_idx[primary]] == 0:
            target[self.label_to_idx[primary]] = 1.0
        return target

    # ------------------------------------------------------------------ #
    # public: __getitem__ wraps load + optional MixUp
    # MixUp branch loads partners with _load_raw (no noise) and applies
    # background_noise once at the end so we don't double-stack noise.
    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        # Val multi-crop: outer index is row * val_n_crops + crop_idx.
        if not self.training and self.val_n_crops > 1:
            row_idx = idx // self.val_n_crops
            crop_idx = idx % self.val_n_crops
            return self._load_raw(row_idx, val_crop_idx=crop_idx)

        do_mixup = (
            self.training
            and self.mixup_p > 0
            and torch.rand(1).item() < self.mixup_p
        )
        if do_mixup:
            partner_idx = int(torch.randint(0, len(self.df), (1,)).item())
            if partner_idx == idx:
                partner_idx = (idx + 1) % len(self.df)
            wave_a, target_a, meta = self._load_raw(idx)
            wave_b, target_b, _ = self._load_raw(partner_idx)
            waveform, target = mixup_pair(
                wave_a, target_a, wave_b, target_b,
                alpha=self.mixup_alpha,
                target_aggregation=self.mixup_target_aggregation,
                normalize=True,
            )
            if self.background_noise is not None:
                waveform = self.background_noise(waveform)
            return waveform, target, meta

        return self._load_one(idx)
