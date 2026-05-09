"""Class-balanced samplers for SED training.

The plain ``DistributedSampler(shuffle=True)`` draws every sample with
equal probability. With BirdCLEF's long-tailed species distribution
(~1000:1 between most/least-frequent species in 234 classes) this means
rare classes barely show up per batch — the model never gets enough
gradient on them and macro-AUC stays low.

``DistributedWeightedSampler`` replaces it with a ``multinomial`` over
per-sample weights computed from class frequencies. With sqrt-balanced
weighting (``weight_i ∝ 1 / sqrt(class_count)``) plus a rare-class floor
(``min_count=30`` capping the denominator), the per-epoch expected count
for class ``c`` becomes ``∝ sqrt(c)`` — popular species still appear
more often, but the ratio compresses from ~1000:1 to ~30:1, which
matches BirdCLEF 2025 2nd Place's ``class_weights_path="sqrt"`` recipe.

DDP-aware: each rank computes its own slice via rank-aware seeding so
the four processes don't draw identical batches each epoch (``epoch *
1_000_003 + rank``). ``set_epoch(epoch)`` advances the RNG, mirroring
the standard ``DistributedSampler`` API so ``train_one_epoch`` keeps
calling ``loader.sampler.set_epoch(epoch)`` unchanged.
"""

from __future__ import annotations

import math
from typing import Iterator, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class DistributedWeightedSampler(Sampler[int]):
    """Per-rank multinomial sampler over per-sample weights."""

    def __init__(
        self,
        weights: Sequence[float],
        *,
        num_samples: int | None = None,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
        replacement: bool = True,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank {rank} out of range [0, {num_replicas})")

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if self.weights.ndim != 1:
            raise ValueError(f"weights must be 1-D, got shape {tuple(self.weights.shape)}")
        if num_samples is None:
            num_samples = int(self.weights.numel())
        # Each rank yields num_samples // world_size indices per epoch.
        self.num_samples = max(1, int(num_samples) // int(num_replicas))
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.replacement = bool(replacement)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        # Large prime multiplier makes per-(epoch, rank) seeds well separated.
        g.manual_seed(self.seed + self.epoch * 1_000_003 + self.rank)
        idx = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement,
            generator=g,
        )
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.num_samples


def sqrt_balanced_weights(
    primary_labels: Sequence[str],
    *,
    min_count: int = 30,
) -> list[float]:
    """Return one weight per row: ``1 / sqrt(max(count, min_count))``.

    ``min_count`` caps the denominator for rare classes — without this,
    a 5-sample species would be 14× more sampled than a 1000-sample one,
    which over-weights noisy rare-class examples. ``min_count=30``
    treats anything ≤30 as if it had 30 samples (so the rarest class
    sits at most ``sqrt(30/1000) ≈ 5.5×`` more sampled than the densest).
    """
    counts: dict[str, int] = {}
    for label in primary_labels:
        s = str(label)
        counts[s] = counts.get(s, 0) + 1
    floor = max(1, int(min_count))
    return [
        1.0 / math.sqrt(max(counts.get(str(label), 1), floor))
        for label in primary_labels
    ]
