"""timm-based CNN backbones for BirdCLEF SED.

Wraps a timm model so that:
  * input is (B, 1, n_mels, T) — 1-channel log-mel (no 1→3 repeat)
  * output is the **last spatial feature map** ``(B, C, F', T')`` where
    ``F'`` corresponds to the frequency axis and ``T'`` to the time axis,
    ready for the SED ``AttHead`` to GeMFreq-pool over ``F'``.

Supported names are anything timm understands. Plans default:
  * ``tf_efficientnetv2_s.in21k_ft_in1k``  — main workhorse
  * ``eca_nfnet_l0``                       — diversity
  * ``convnext_tiny.in12k_ft_in1k``        — optional 3rd path

Pretrained backbone hot-swap
----------------------------
``pretrained_state_dict_path`` lets us replace timm's ImageNet warm-start
with a bird-domain pretrained backbone — e.g. BirdCLEF 2025 2nd Place's
public ``tf_efficientnetv2_s_in21k_Pretrainversion1.pth`` (~80 MB,
pretrained on BC 2021-2024 + Xeno-Canto + iNaturalist, multi-year). With
that checkpoint loaded, the model starts well outside the trivial-zero
attractor that haunts cold-ImageNet starts on 234-way multi-label SED,
which is the difference between val_auc≈0.5 plateaus and a clean cosine
climb to ~0.92 single fold.

The public release used ``features_only=True``, so the last few keys
(``conv_head.weight`` + ``bn2.*``, ~6% of params) aren't in the file. Two
ways to deal with that:

  * ``features_only=False`` (default, legacy) — build a regular timm model
    with conv_head + bn2 and load with ``strict=False``. The 5 missing
    keys stay at random timm init, sitting between the warm backbone and
    the SED head. This silently caps the model: e01 V2-S LB 0.43 was
    caused by exactly this footgun (conv_head + bn2 random-init).

  * ``features_only=True`` (new, recommended) — build the timm model with
    ``features_only=True, out_indices=(-1,)``, which stops at the last
    block output (V2-S = 256 ch, not 1280 after conv_head). Now every
    weight in the state_dict matches a parameter in the model, and the
    SED head's GeMFreq + Linear(256 → 512) replaces conv_head.

``warmstart_strict`` is a separate footgun guard: when True, any non-empty
``missing_keys`` after ``load_state_dict(strict=False)`` raises so we
don't silently ship a half-initialized backbone again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    """Thin timm wrapper with optional bird-domain pretrained warm-start."""

    def __init__(
        self,
        name: str = "tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained: bool = True,
        in_chans: int = 1,
        pretrained_state_dict_path: str | Path | None = None,
        warmstart_strict: bool = False,
        features_only: bool = False,
    ):
        super().__init__()
        self.name = name
        self.in_chans = int(in_chans)
        self.warmstart_strict = bool(warmstart_strict)
        self.features_only = bool(features_only)
        # Build with the channel count we'll actually feed at forward. timm
        # averages ImageNet's 3-channel conv1 weights into our 1-channel stem
        # (when pretrained=True), giving a sane warm-start that the bird-
        # domain pretrained checkpoint then overwrites.
        if self.features_only:
            # features_only=True returns a list of intermediate feature maps;
            # ``out_indices=(-1,)`` keeps only the last block output. timm
            # rejects num_classes / global_pool when features_only=True.
            self.body = timm.create_model(
                name,
                pretrained=pretrained,
                features_only=True,
                in_chans=self.in_chans,
                out_indices=(-1,),
            )
            # feature_info.channels() returns a list (one per out_indices entry).
            ch = self.body.feature_info.channels()
            self.num_features = int(ch[-1])
        else:
            self.body = timm.create_model(
                name,
                pretrained=pretrained,
                num_classes=0,        # drop classifier
                global_pool="",       # keep the spatial feature map
                in_chans=self.in_chans,
            )
            if hasattr(self.body, "num_features"):
                self.num_features = int(self.body.num_features)
            else:
                with torch.no_grad():
                    dummy = torch.zeros(1, self.in_chans, 128, 128)
                    feat = self.body(dummy)
                    self.num_features = int(feat.shape[1])

        if pretrained_state_dict_path is not None:
            self._load_external_pretrained(
                Path(pretrained_state_dict_path),
                strict=self.warmstart_strict,
            )

    def _load_external_pretrained(self, path: Path, strict: bool = False) -> None:
        """Load a bird-domain pretrained backbone state_dict.

        Accepts:
          - bare timm state_dicts (e.g. Sydorskyy's ``Pretrainversion1.pth``)
          - Lightning-wrapped ``{'state_dict': {...}}`` dicts
          - dicts with a ``backbone.*`` / ``body.*`` / ``model.*`` prefix
        Always loads with ``strict=False`` underneath; if ``strict=True`` is
        passed, raises after the load when ``missing_keys`` is non-empty.
        That's the footgun guard for "ckpt architecture vs model architecture
        silently disagree" — see e01 V2-S LB 0.43 post-mortem.
        """
        if not path.exists():
            raise FileNotFoundError(f"pretrained backbone not found: {path}")
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            for prefix in ("backbone.body.", "backbone.", "body.", "model."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            cleaned[k] = v
        load = self.body.load_state_dict(cleaned, strict=False)
        n_total = len(self.body.state_dict())
        n_in = len(cleaned)
        n_unexpected = len(load.unexpected_keys)
        n_missing = len(load.missing_keys)
        print(
            f"[backbone] {path.name}: loaded {n_in - n_unexpected}/{n_total} keys "
            f"(missing={n_missing}, unexpected={n_unexpected})"
        )
        if n_missing > 0:
            sample = ", ".join(load.missing_keys[:3]) + (
                ", ..." if n_missing > 3 else ""
            )
            print(f"[backbone]   missing (kept timm-init): {sample}")
        if n_unexpected > 0:
            sample = ", ".join(load.unexpected_keys[:3]) + (
                ", ..." if n_unexpected > 3 else ""
            )
            print(f"[backbone]   unexpected (ignored): {sample}")
        if strict and n_missing > 0:
            preview = "\n  - ".join(load.missing_keys[:10])
            extra = (
                f"\n  ... ({n_missing - 10} more)" if n_missing > 10 else ""
            )
            raise RuntimeError(
                f"warmstart_strict=True: {n_missing} missing keys after loading "
                f"{path.name}. First {min(n_missing, 10)}:\n  - {preview}{extra}\n"
                f"Either set features_only=True (to drop conv_head + bn2 in V2-S "
                f"SED-style ckpts) or pick a checkpoint that matches the model."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:                        # (B, n_mels, T) → (B, 1, n_mels, T)
            x = x.unsqueeze(1)
        if x.shape[1] != self.in_chans:
            if x.shape[1] == 1 and self.in_chans == 3:
                # Legacy fallback for 3ch backbones — repeat once to fit stem.
                x = x.expand(-1, 3, -1, -1).contiguous()
            else:
                raise ValueError(
                    f"backbone expects in_chans={self.in_chans}, got {x.shape[1]}"
                )
        out = self.body(x)
        if self.features_only:
            # features_only=True returns list[Tensor]; with out_indices=(-1,)
            # there is exactly one entry — the last block's (B, C, F', T').
            return out[-1] if isinstance(out, (list, tuple)) else out
        return out                             # (B, C, F', T')


def create_backbone(
    name: str = "tf_efficientnetv2_s.in21k_ft_in1k",
    pretrained: bool = True,
    in_chans: int = 1,
    pretrained_state_dict_path: str | Path | None = None,
    warmstart_strict: bool = False,
    features_only: bool = False,
) -> CNNBackbone:
    return CNNBackbone(
        name=name,
        pretrained=pretrained,
        in_chans=in_chans,
        pretrained_state_dict_path=pretrained_state_dict_path,
        warmstart_strict=warmstart_strict,
        features_only=features_only,
    )
