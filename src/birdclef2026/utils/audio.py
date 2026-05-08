"""Audio I/O and log-mel spectrogram with global-statistic normalization.

The legacy module ``birdclef2026.audio`` per-sample-normalizes mel which
destroys absolute energy information that is class-discriminative for bird
calls (e.g., insect chirps vs frog calls have very different SNR). Here we
use dataset-level mean/std estimated by ``scripts/02_compute_mel_stats.py``.

Public API:
    load_audio(path, sample_rate)       -> 1-D float tensor (raises IOError on failure)
    crop_or_pad(wav, length, ...)       -> fixed-length 1-D tensor
    extract_window(wav, sr, sec, ...)   -> fixed-length 1-D tensor (with end_time anchor)
    LogMel(n_fft=2048, ...)             -> nn.Module(waveform) -> (B, 1, n_mels, T)

Implementation notes
--------------------
* ``LogMel`` computes power-mel → ``10 * log10(clamp_min(amin))``.
  We do **not** use ``torchaudio.transforms.AmplitudeToDB(top_db=...)`` because
  its top-dB clipping is *relative to the batch maximum*, which makes the same
  mel value depend on what other items happen to be in the batch — a subtle
  but real source of train/inference distribution drift. (See review_commit2.md
  H6.) The fixed ``amin=1e-10`` floor gives a deterministic dB scale.
* ``LogMel.forward`` and downstream STFT-based augmentations run inside an
  explicit ``torch.autocast(enabled=False)`` block to force fp32 for the FFT
  and log10 — bf16 makes both numerically unstable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio


def load_audio(path, sample_rate: int) -> torch.Tensor:
    """Load mono audio at ``sample_rate``.

    Raises:
        IOError: when both torchaudio and soundfile fail to decode the file.
                 Callers (``SEDDataset._load_one``) catch this and retry on a
                 different sample so a single corrupt .ogg does not kill a
                 DataLoader worker mid-epoch.
    """
    path = Path(path)
    try:
        waveform, sr = torchaudio.load(str(path))
    except Exception as torchaudio_err:
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), always_2d=True, dtype="float32")
            waveform = torch.from_numpy(data.T)
        except Exception as soundfile_err:
            raise IOError(
                f"Failed to decode {path}: torchaudio={torchaudio_err}; "
                f"soundfile={soundfile_err}"
            ) from soundfile_err
    waveform = waveform.float()
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform.squeeze(0).contiguous()


def crop_or_pad(
    waveform: torch.Tensor,
    length_samples: int,
    *,
    random_crop: bool = False,
    start_sample: Optional[int] = None,
    pad_mode: str = "constant",
) -> torch.Tensor:
    n = waveform.numel()
    if n >= length_samples:
        if start_sample is None:
            start_sample = (
                int(torch.randint(0, n - length_samples + 1, (1,)).item())
                if random_crop and n > length_samples
                else 0
            )
        start_sample = max(0, min(int(start_sample), n - length_samples))
        return waveform[start_sample : start_sample + length_samples].contiguous()
    pad = length_samples - n
    return F.pad(waveform, (0, pad), mode=pad_mode)


def extract_window(
    waveform: torch.Tensor,
    sample_rate: int,
    clip_seconds: float,
    *,
    end_time: Optional[float] = None,
    random_crop: bool = False,
) -> torch.Tensor:
    """Extract a fixed-length window. If ``end_time`` is given, anchor the window so it ends there."""
    length = int(round(sample_rate * clip_seconds))
    if end_time is None:
        return crop_or_pad(waveform, length, random_crop=random_crop)
    end_sample = int(round(float(end_time) * sample_rate))
    start_sample = end_sample - length
    return crop_or_pad(waveform, length, random_crop=False, start_sample=start_sample)


class LogMel(torch.nn.Module):
    """Log-mel spectrogram with optional global-statistic normalization.

    Output shape: ``(B, 1, n_mels, T_mel)`` for batched waveform input
    ``(B, T_audio)``; ``(1, 1, n_mels, T_mel)`` if a 1-D waveform is given.

    Args:
        sample_rate, n_fft, hop_length, n_mels, f_min, f_max: standard mel params.
        amin: floor for the power spectrogram before ``log10``. ``1e-10`` matches
            BirdCLEF 25 2nd Place.
        global_mean / global_std: dataset-level normalization stats from
            ``scripts/02_compute_mel_stats.py``. If either is ``None``, falls
            back to per-sample normalization (only intended for smoke tests).
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 20.0,
        f_max: float = 14000.0,
        amin: float = 1e-10,
        # ``top_db`` is accepted for backward compatibility with older configs
        # but is intentionally ignored — see module docstring (H6).
        top_db: float = 80.0,  # noqa: ARG002
        global_mean: Optional[float] = None,
        global_std: Optional[float] = None,
    ):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            normalized=False,
        )
        self.amin = float(amin)
        self.has_global = global_mean is not None and global_std is not None
        if self.has_global:
            self.register_buffer("global_mean", torch.tensor(float(global_mean)))
            self.register_buffer("global_std", torch.tensor(max(float(global_std), 1e-6)))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Force fp32 for STFT + log10 (autocast bf16 makes both unstable; H2).
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            wav = waveform.float()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)  # (1, T)
            mel = self.mel(wav)                              # power-mel, ≥ 0
            mel = 10.0 * torch.log10(mel.clamp_min(self.amin))
            if self.has_global:
                mel = (mel - self.global_mean) / self.global_std
            else:
                mel = (mel - mel.mean()) / mel.std().clamp_min(1e-6)
        return mel
