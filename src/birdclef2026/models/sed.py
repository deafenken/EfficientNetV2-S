"""SED head + end-to-end SEDModel (mel + augs + backbone + head).

The ``AttHead`` mirrors BirdCLEF 2025 2nd Place's ``code_base/models/blocks.py::AttHead``:

    feat (B, C, F', T')
      → GeMFreq pool over freq → (B, C, 1, T') → squeeze → (B, C, T') → (B, T', C)
      → Dropout → Linear(C → 512) → ReLU → Dropout → (B, 512, T')
      → attention = tanh(Conv1d(512, n_class))
      → fix_scale =       Conv1d(512, n_class)
      → clipwise_logits = sum( fix_scale * softmax(attention, dim=-1), dim=-1 )

The ``SEDModel`` wraps:
    waveform → MelExtractor → SpecAugment(train) → CNN backbone → AttHead

Training uses ``clipwise_logits``. Inference can also expose ``framewise_logits``
for time-axis ensembling / TopN postprocessing in Commit 3.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.transforms import RandomFiltering, SpecAugment
from ..utils.audio import LogMel
from .backbones import CNNBackbone, create_backbone


class GeMFreq(nn.Module):
    """Generalized-mean pooling along the frequency axis.

    Input  (B, C, F, T)
    Output (B, C, 1, T)
    """

    def __init__(self, p: float = 3.0, learnable: bool = True, eps: float = 1e-6):
        super().__init__()
        if learnable:
            self.p = nn.Parameter(torch.full((1,), float(p)))
        else:
            self.register_buffer("p", torch.tensor(float(p)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = x.shape[-2]
        return F.avg_pool2d(x.clamp_min(self.eps).pow(self.p), kernel_size=(f, 1)).pow(
            1.0 / self.p
        )


class AttHead(nn.Module):
    """Attention pooling head producing clipwise logits + framewise logits."""

    def __init__(
        self,
        in_chans: int,
        num_classes: int,
        hidden: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.pool = GeMFreq()
        self.dense = nn.Sequential(
            nn.Dropout(dropout / 2),
            nn.Linear(in_chans, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attention = nn.Conv1d(hidden, num_classes, kernel_size=1, bias=True)
        self.fix_scale = nn.Conv1d(hidden, num_classes, kernel_size=1, bias=True)

    def forward(self, feat: torch.Tensor) -> dict:
        # feat: (B, C, F', T')
        feat = self.pool(feat).squeeze(-2)        # (B, C, T')
        feat = feat.permute(0, 2, 1)              # (B, T', C)
        feat = self.dense(feat).permute(0, 2, 1)  # (B, hidden, T')

        time_att = torch.tanh(self.attention(feat))   # (B, n_class, T')
        feat_v = self.fix_scale(feat)                  # (B, n_class, T')
        weights = torch.softmax(time_att, dim=-1)

        clipwise_logits = (feat_v * weights).sum(dim=-1)               # (B, n_class)
        clipwise_pred = (torch.sigmoid(feat_v) * weights).sum(dim=-1)  # (B, n_class)
        return {
            "clipwise_logits": clipwise_logits,
            "clipwise_pred": clipwise_pred,
            "framewise_logits": feat_v.permute(0, 2, 1),               # (B, T', n_class)
        }


class SEDModel(nn.Module):
    """End-to-end SED model: waveform → mel → augs → backbone → head.

    Args:
        backbone: timm-name string OR a pre-built ``CNNBackbone``.
        num_classes: number of species (234 for 2026).
        mel_kwargs: dict forwarded to ``LogMel`` (sample_rate, n_mels, ..., global_mean/std).
        spec_augment_kwargs: dict forwarded to ``SpecAugment`` (active only in training mode).
        random_filter_kwargs: dict forwarded to ``RandomFiltering`` (training mode).
        head_dropout: passed to the AttHead.
    """

    def __init__(
        self,
        backbone: str | CNNBackbone = "tf_efficientnetv2_s.in21k_ft_in1k",
        *,
        num_classes: int,
        mel_kwargs: Optional[dict] = None,
        spec_augment_kwargs: Optional[dict] = None,
        random_filter_kwargs: Optional[dict] = None,
        head_dropout: float = 0.5,
        pretrained: bool = True,
        pretrained_backbone_path: Optional[str] = None,
        backbone_in_chans: int = 1,
        warmstart_strict: bool = False,
        features_only: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)

        # Body
        if isinstance(backbone, str):
            self.backbone = create_backbone(
                backbone,
                pretrained=pretrained,
                in_chans=backbone_in_chans,
                pretrained_state_dict_path=pretrained_backbone_path,
                warmstart_strict=warmstart_strict,
                features_only=features_only,
            )
        else:
            self.backbone = backbone

        # Wave-domain aug (GPU). ``enabled`` defaults to True, matches BirdCLEF 25 2nd's behavior.
        rf_kwargs = dict(random_filter_kwargs or {})
        rf_enabled = bool(rf_kwargs.pop("enabled", True))
        self.random_filter: Optional[RandomFiltering] = (
            RandomFiltering(**rf_kwargs) if rf_enabled else None
        )

        # Mel
        self.mel = LogMel(**(mel_kwargs or {}))

        # Spec-domain aug (GPU)
        self.spec_augment = SpecAugment(**(spec_augment_kwargs or {}))

        # Head
        self.head = AttHead(
            in_chans=self.backbone.num_features,
            num_classes=self.num_classes,
            dropout=head_dropout,
        )

    def forward(self, waveform: torch.Tensor) -> dict:
        # waveform: (B, T)
        if self.training and self.random_filter is not None:
            waveform = self.random_filter(waveform)
        mel = self.mel(waveform)                   # (B, 1, n_mels, T') after our LogMel
        if mel.ndim == 3:
            mel = mel.unsqueeze(1)
        mel = self.spec_augment(mel)
        feat = self.backbone(mel)                  # (B, C, F', T')
        return self.head(feat)


def build_model_from_config(cfg: dict, num_classes: int) -> SEDModel:
    """Build an SEDModel from a flat config dict (e.g., the merged exp yaml)."""
    model_cfg = cfg.get("model", {}) or {}
    audio_cfg = cfg.get("audio", {}) or {}
    aug_cfg = cfg.get("augment", {}) or {}

    mel_kwargs = {
        "sample_rate": int(audio_cfg.get("sample_rate", 32000)),
        "n_fft": int(audio_cfg.get("n_fft", 2048)),
        "hop_length": int(audio_cfg.get("hop_length", 512)),
        "n_mels": int(audio_cfg.get("n_mels", 128)),
        "f_min": float(audio_cfg.get("f_min", 20.0)),
        "f_max": float(audio_cfg.get("f_max", audio_cfg.get("sample_rate", 32000) // 2)),
        "top_db": float(audio_cfg.get("top_db", 80.0)),
        "global_mean": audio_cfg.get("global_mean"),
        "global_std": audio_cfg.get("global_std"),
    }

    spec_aug_kwargs = aug_cfg.get("spec_augment", {}) or {}
    rand_filt_kwargs = aug_cfg.get("random_filter", {}) or {}

    return SEDModel(
        backbone=str(model_cfg.get("name", "tf_efficientnetv2_s.in21k_ft_in1k")),
        num_classes=num_classes,
        mel_kwargs=mel_kwargs,
        spec_augment_kwargs=spec_aug_kwargs,
        random_filter_kwargs=rand_filt_kwargs,
        head_dropout=float(model_cfg.get("head_dropout", 0.5)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        pretrained_backbone_path=model_cfg.get("pretrained_backbone_path"),
        backbone_in_chans=int(model_cfg.get("backbone_in_chans", 1)),
        warmstart_strict=bool(model_cfg.get("warmstart_strict", False)),
        features_only=bool(model_cfg.get("features_only", False)),
    )
