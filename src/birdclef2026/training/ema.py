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
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        base = _unwrap_ddp(model)
        self.module = deepcopy(base).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        src = _unwrap_ddp(model)
        # EMA on parameters
        src_params = dict(src.named_parameters())
        for name, dst in self.module.named_parameters():
            s = src_params[name].detach().to(dst.device, dtype=dst.dtype)
            dst.mul_(self.decay).add_(s, alpha=1.0 - self.decay)
        # Copy buffers (BN running mean/var, num_batches_tracked, etc.)
        src_buffers = dict(src.named_buffers())
        for name, dst in self.module.named_buffers():
            s = src_buffers[name].detach().to(dst.device, dtype=dst.dtype)
            dst.copy_(s)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state)
