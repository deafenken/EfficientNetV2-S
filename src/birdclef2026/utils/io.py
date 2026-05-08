"""IO helpers: directory creation, atomic checkpoint save, OOF logits writers, CSV logger."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(state: Mapping[str, Any], path) -> None:
    """Atomic torch.save: write to .tmp then rename, so crashes never leave half-written .pt files."""
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(state), tmp)
    tmp.replace(path)


def load_checkpoint(path, map_location: str = "cpu") -> dict:
    return torch.load(str(path), map_location=map_location)


def save_oof(
    path,
    *,
    sample_ids,
    logits,
    labels,
    extra: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Save per-sample (sample_id, logits, labels) as a compressed npz.

    Stored as float32 so downstream sklearn metrics don't have to upcast.
    Use the package's `inference.ensemble` helpers to re-read.
    """
    path = Path(path)
    ensure_dir(path.parent)
    payload: dict[str, np.ndarray] = {
        "sample_ids": np.asarray(sample_ids),
        "logits": np.asarray(logits, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.float32),
    }
    if extra:
        for k, v in extra.items():
            payload[k] = np.asarray(v)
    np.savez_compressed(path, **payload)


def append_csv(path, row: Mapping[str, Any]) -> None:
    """Append a row to a CSV, creating it (with header) on first write.

    DDP callers must guard with ``if rank == 0`` to avoid 4 processes racing.
    """
    path = Path(path)
    ensure_dir(path.parent)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
