"""Focal BCE losses with optional label smoothing.

Default config mirrors BirdCLEF 2025 2nd Place's ``FocalLossBCE``:
    α=0.25, γ=2.0, BCE-Focal weighted 1:1.

Label-smoothing convention follows the same writeup: instead of the usual
``(1-ε)·y + ε/2`` floor, we offer two modes:
    'standard'   y' = (1-ε) y + ε · 0.5       (binary symmetric smoothing)
    'positive'   y' = (1-ε) y + ε · y.sum()/C (BirdCLEF 25 2nd's "LSF1005")
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

LSMode = Literal["standard", "positive", "off"]


def smooth_targets(targets: torch.Tensor, eps: float, mode: LSMode = "standard") -> torch.Tensor:
    if eps <= 0 or mode == "off":
        return targets
    if mode == "standard":
        return targets * (1.0 - eps) + 0.5 * eps
    if mode == "positive":
        # Per-row scalar: average positive count divided by the number of classes.
        if targets.ndim != 2:
            raise ValueError("positive label-smoothing expects (B, C) targets")
        n_classes = targets.shape[-1]
        per_row_mean = targets.sum(dim=-1, keepdim=True) / float(n_classes)
        return targets * (1.0 - eps) + eps * per_row_mean
    raise ValueError(f"unknown label_smoothing mode: {mode!r}")


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Same recipe as ``torchvision.ops.focal_loss.sigmoid_focal_loss`` (vendored to drop the dep)."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class FocalBCEWithLogits(nn.Module):
    """``bce_weight·BCE + focal_weight·Focal`` — BirdCLEF 25 2nd's default.

    Args:
        alpha:        focal positive weighting (0.25 default).
        gamma:        focal focusing exponent (2.0 default).
        bce_weight:   weight on plain BCE (1.0 default).
        focal_weight: weight on focal BCE (1.0 default).
        label_smoothing: ε in (0, 1).
        ls_mode:      'standard' | 'positive' | 'off'.
        pos_weight:   optional per-class positive-sample weight, shape (C,).
        reduction:    'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        bce_weight: float = 1.0,
        focal_weight: float = 1.0,
        label_smoothing: float = 0.0,
        ls_mode: LSMode = "standard",
        pos_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.bce_weight = float(bce_weight)
        self.focal_weight = float(focal_weight)
        self.label_smoothing = float(label_smoothing)
        self.ls_mode: LSMode = ls_mode
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = smooth_targets(targets, self.label_smoothing, self.ls_mode)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction=self.reduction
        )
        focal = sigmoid_focal_loss(
            logits, targets, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction
        )
        return self.bce_weight * bce + self.focal_weight * focal
