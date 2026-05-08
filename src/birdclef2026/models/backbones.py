"""timm-based CNN backbones for BirdCLEF SED.

Wraps a timm model so that:
  * input is (B, 1, n_mels, T) — 1-channel log-mel
  * 1→3 channel conversion happens inside the wrapper (repeat) to use ImageNet-pretrained weights
  * output is the **last spatial feature map** ``(B, C, F', T')`` where
    ``F'`` corresponds to the frequency axis and ``T'`` to the time axis,
    ready for the SED ``AttHead`` to GeMFreq-pool over ``F'``.

Supported names are anything timm understands. Plans default:
  * ``tf_efficientnetv2_s.in21k_ft_in1k``  — main workhorse
  * ``eca_nfnet_l0``                       — diversity
  * ``convnext_tiny.in12k_ft_in1k``        — optional 3rd path
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "timm is required for birdclef2026.models.backbones; install via uv sync"
    ) from e


@dataclass
class BackboneOut:
    feat_dim: int  # C in (B, C, F', T')


class CNNBackbone(nn.Module):
    """Thin timm wrapper with 1ch→3ch and feature-map output.

    Use ``num_features`` to wire up downstream SED head.
    """

    def __init__(
        self,
        name: str = "tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained: bool = True,
        in_chans: int = 1,
    ):
        super().__init__()
        self.name = name
        # We do 1→3 expansion ourselves (repeat) to keep ImageNet-pretrained conv1 weights.
        # Setting in_chans=3 lets timm load the original first-conv unchanged.
        self.in_chans = int(in_chans)
        self.body = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,        # drop classifier
            global_pool="",       # keep the spatial feature map
        )
        # Determine the channel count of the last feature map.
        if hasattr(self.body, "num_features"):
            self.num_features = int(self.body.num_features)
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 128, 128)
                feat = self.body(dummy)
                self.num_features = int(feat.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:                        # (B, n_mels, T) → (B, 1, n_mels, T)
            x = x.unsqueeze(1)
        if x.shape[1] == 1:                    # 1ch → 3ch by repeat (use ImageNet conv1 weights)
            x = x.expand(-1, 3, -1, -1).contiguous()
        return self.body(x)                    # (B, C, F', T')


def create_backbone(
    name: str = "tf_efficientnetv2_s.in21k_ft_in1k",
    pretrained: bool = True,
    in_chans: int = 1,
) -> CNNBackbone:
    return CNNBackbone(name=name, pretrained=pretrained, in_chans=in_chans)
