"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python / numpy / torch (incl. CUDA all-devices) and PYTHONHASHSEED."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def fold_seed(base_seed: int, fold_idx: int) -> int:
    """Deterministic per-fold seed; keeps cross-fold variance low while still mixing."""
    return int(base_seed) * 1000 + int(fold_idx)
