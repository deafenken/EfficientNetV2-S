"""Exponential moving average of model parameters.

Pattern (training loop):
    ema = ModelEMA(model, decay=0.999)
    ...
    optimizer.step()
    ema.update(model)               # call AFTER step
    ...
    # Validation: use ema.module instead of model
    val_auc = evaluate(ema.module, val_loader)

Implementation note (review_commit2.md H5)
------------------------------------------
We EMA only **parameters**; buffers (BatchNorm running mean/var,
``num_batches_tracked``, etc.) are *copied* from the latest model rather than
EMA'd. EMA-ing BN buffers under DDP — which by default does NOT synchronize
BN statistics across ranks — accumulates per-rank drift that biases validation
on rank 0. The same convention is used by ``timm.utils.ModelEmaV2``.

For best results combine this EMA with ``nn.SyncBatchNorm.convert_sync_batchnorm``
applied to the model *before* DDP wrapping, so the buffers we copy are
already a global estimate.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


def _unwrap_ddp(model: nn.Module) -> nn.Module:
    inner = getattr(model, "module", None)
    if isinstance(inner, nn.Module):
        return inner
    return model


class ModelEMA:
    """Track an EMA shadow of the (DDP-unwrapped) model parameters.

    Buffers are copied from the source model verbatim (no decay).

    Decay warmup
    ------------
    With a fixed ``decay=0.999`` the EMA stays ~80% **initial weights** for
    the first 200 steps (`0.999**222 ≈ 0.80`), which makes early-epoch
    validation look like random because we're effectively evaluating a
    near-untrained model. ``warmup=True`` (default) ramps the effective
    decay as ``min(decay, (1 + n_updates) / (10 + n_updates))`` — the same
    schedule used by ``timm.utils.ModelEmaV2`` and the original Inception
    EMA. The schedule starts near 0 (EMA tracks the live model closely),
    crosses 0.9 around step 100, and converges to ``decay`` after a few
    hundred steps so the long-term averaging behavior is preserved.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: bool = True):
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.num_updates = 0
        base = _unwrap_ddp(model)
        self.module = deepcopy(base).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def _current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        # +1 so step 0 gives a non-zero decay; +10 controls how fast the
        # ramp approaches the asymptote.
        ramped = (1 + self.num_updates) / (10 + self.num_updates)
        return min(self.decay, ramped)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        d = self._current_decay()
        src = _unwrap_ddp(model)
        # EMA on parameters
        src_params = dict(src.named_parameters())
        for name, dst in self.module.named_parameters():
            s = src_params[name].detach().to(dst.device, dtype=dst.dtype)
            dst.mul_(d).add_(s, alpha=1.0 - d)
        # Copy buffers (BN running mean/var, num_batches_tracked, etc.)
        src_buffers = dict(src.named_buffers())
        for name, dst in self.module.named_buffers():
            s = src_buffers[name].detach().to(dst.device, dtype=dst.dtype)
            dst.copy_(s)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state)
