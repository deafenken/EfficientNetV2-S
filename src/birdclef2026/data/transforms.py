"""Audio augmentation primitives.

Two flavors:
  * **Sample-level** (NumPy / 1-D torch tensors, runs in Dataset workers):
      ``BackgroundNoise`` — amp-domain mix with a noise sample (BirdCLEF 25 2nd default 0.25–0.75)
      ``MixUp``           — wave-level avg + ``clamp(t1 + t2, 0, 1)`` target (probabilistic OR)
  * **Batch-level** (``nn.Module``, runs on GPU inside the model):
      ``SpecAugment``     — freq + time masking after mel
      ``RandomFiltering`` — 4-point random freq EQ via STFT round-trip

Defaults mirror the public BirdCLEF 25 2nd Place repo
(``VSydorskyy/BirdCLEF_2025_2nd_place``); see plans for citations.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torchaudio


# --------------------------------------------------------------------------- #
# Sample-level (CPU, Dataset.__getitem__)
# --------------------------------------------------------------------------- #


class BackgroundNoise:
    """Amplitude-domain mix with a randomly sampled noise clip.

    augmented = (1 - α) · wave + α · noise,   α ~ U(min_amp, max_amp)

    Args:
        noise_paths: pool of background-noise file paths (e.g., ESC-50 non-bird + soundscape-nocall).
        sample_rate: target sample rate; clips are resampled if needed.
        clip_seconds: window length to crop / loop the noise to.
        min_amp, max_amp: uniform range for the noise amplitude (default 0.25–0.75).
        p: probability of applying.
        normalize: peak-normalize the mixed waveform after combining.
    """

    def __init__(
        self,
        noise_paths: Sequence[Path] | None,
        sample_rate: int = 32000,
        clip_seconds: float = 5.0,
        min_amp: float = 0.25,
        max_amp: float = 0.75,
        p: float = 0.5,
        normalize: bool = True,
    ):
        self.noise_paths = [Path(p) for p in (noise_paths or [])]
        self.sample_rate = int(sample_rate)
        self.length_samples = int(round(self.sample_rate * float(clip_seconds)))
        self.min_amp = float(min_amp)
        self.max_amp = float(max_amp)
        self.p = float(p)
        self.normalize = bool(normalize)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.noise_paths or random.random() >= self.p:
            return waveform
        noise = _load_noise_clip(
            random.choice(self.noise_paths), self.sample_rate, self.length_samples
        )
        if noise is None:
            return waveform
        amp = random.uniform(self.min_amp, self.max_amp)
        mixed = (1.0 - amp) * waveform + amp * noise
        if self.normalize:
            peak = mixed.abs().max().clamp_min(1e-6)
            mixed = mixed / peak
        return mixed


def _load_noise_clip(path: Path, sample_rate: int, length_samples: int) -> torch.Tensor | None:
    """Load a noise file and force-crop/pad to ``length_samples``. Returns None on failure."""
    try:
        wav, sr = torchaudio.load(str(path))
    except Exception:
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), always_2d=True, dtype="float32")
            wav = torch.from_numpy(data.T)
        except Exception:
            return None
    wav = wav.float()
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.squeeze(0)
    n = wav.numel()
    if n >= length_samples:
        start = random.randint(0, n - length_samples)
        return wav[start : start + length_samples].contiguous()
    # tile-pad to fill the window
    reps = (length_samples + n - 1) // n
    return wav.repeat(reps)[:length_samples].contiguous()


def mixup_pair(
    wave_a: torch.Tensor,
    target_a: torch.Tensor,
    wave_b: torch.Tensor,
    target_b: torch.Tensor,
    *,
    alpha: float | None = None,
    target_aggregation: str = "sum",
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BirdCLEF 25 2nd-style wave-level MixUp.

    Args:
        alpha: if None → 50/50 mix; if float → Beta(alpha, alpha) mixing weight.
        target_aggregation:
            'sum'   target = clamp(target_a + target_b, 0, 1)   (probabilistic OR; default)
            'max'   target = max(target_a, target_b)
            'beta'  target = w · target_a + (1 - w) · target_b   (only meaningful with alpha != None)
    """
    if alpha is None:
        wave = (wave_a + wave_b) * 0.5
    else:
        w = float(np.random.beta(alpha, alpha))
        wave = w * wave_a + (1.0 - w) * wave_b

    if target_aggregation == "sum":
        target = (target_a + target_b).clamp_(0.0, 1.0)
    elif target_aggregation == "max":
        target = torch.max(target_a, target_b)
    elif target_aggregation == "beta":
        if alpha is None:
            target = (target_a + target_b) * 0.5
        else:
            target = w * target_a + (1.0 - w) * target_b
    else:
        raise ValueError(f"unknown target_aggregation: {target_aggregation!r}")

    if normalize:
        peak = wave.abs().max().clamp_min(1e-6)
        wave = wave / peak

    return wave, target


# --------------------------------------------------------------------------- #
# Batch-level (GPU, nn.Module living inside the model)
# --------------------------------------------------------------------------- #


class SpecAugment(nn.Module):
    """Per-sample frequency + time masking. No-op when ``self.training`` is False.

    Defaults match BirdCLEF 25 2nd Place (``best_solo_ebs.py``):
        freq_mask: max width 10 bins, up to 3 masks, p=0.3
        time_mask: max width 20 frames, up to 3 masks, p=0.3
    """

    def __init__(
        self,
        freq_mask_param: int = 10,
        freq_mask_max: int = 3,
        freq_mask_p: float = 0.3,
        time_mask_param: int = 20,
        time_mask_max: int = 3,
        time_mask_p: float = 0.3,
    ):
        super().__init__()
        self.freq = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.freq_mask_max = int(freq_mask_max)
        self.time_mask_max = int(time_mask_max)
        self.freq_mask_p = float(freq_mask_p)
        self.time_mask_p = float(time_mask_p)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel shape: (B, 1, n_mels, T)  or  (B, n_mels, T)
        if not self.training:
            return mel
        B = mel.shape[0]
        for i in range(B):
            if torch.rand(1).item() < self.freq_mask_p:
                n = int(torch.randint(1, self.freq_mask_max + 1, (1,)).item())
                for _ in range(n):
                    mel[i : i + 1] = self.freq(mel[i : i + 1])
            if torch.rand(1).item() < self.time_mask_p:
                n = int(torch.randint(1, self.time_mask_max + 1, (1,)).item())
                for _ in range(n):
                    mel[i : i + 1] = self.time(mel[i : i + 1])
        return mel


class RandomFiltering(nn.Module):
    """Random per-sample EQ via STFT round-trip.

    Picks ``n_bands`` random gain control points in [min_db, 0], linearly
    interpolates to the full freq axis, multiplies the magnitude spectrogram,
    and inverts back to the waveform. Mirrors BirdCLEF 25 2nd's
    ``RandomFiltering(min_db=-20, n_bands=4)``.
    """

    def __init__(
        self,
        n_bands: int = 4,
        min_db: float = -20.0,
        n_fft: int = 1024,
        hop_length: int = 512,
        normalize_wave: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.n_bands = int(n_bands)
        self.min_db = float(min_db)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.normalize_wave = bool(normalize_wave)
        self.eps = float(eps)
        self.register_buffer("hann_window", torch.hann_window(self.n_fft))

    def _shape_filter(self, n_samples: int, n_bins: int, device, dtype) -> torch.Tensor:
        # control points in dB ∈ [min_db, 0]
        ctrl = torch.rand(n_samples, self.n_bands, device=device, dtype=dtype) * self.min_db
        # linear interp to n_bins along last dim
        coeffs = nn.functional.interpolate(
            ctrl.unsqueeze(1), mode="linear", size=n_bins, align_corners=True
        ).squeeze(1)
        return torch.pow(10.0, coeffs / 20.0)  # dB → amp

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return wave
        # Force fp32 for STFT round-trip — bf16 autocast on torch.stft / istft
        # is fragile across PyTorch versions and silently degrades phase. (H2)
        with torch.autocast(device_type=wave.device.type, enabled=False):
            wave_f = wave.float()
            spec = torch.stft(
                wave_f,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.hann_window,
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )  # (B, F, T_spec)
            gain = self._shape_filter(spec.shape[0], spec.shape[1], spec.device, wave_f.dtype)
            spec = spec * gain.unsqueeze(-1)
            out = torch.istft(
                spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.hann_window,
                center=True,
                length=wave_f.shape[-1],
                return_complex=False,
            )
            if self.normalize_wave:
                peak = out.abs().amax(dim=-1, keepdim=True).clamp_min(self.eps)
                out = out / peak
        return out
