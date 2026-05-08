"""YAML config loader with experiment-config layering.

Resolution order (last wins):
    1. ``configs/data.yaml``      — shared data + audio + path defaults
    2. ``configs/exp/eNN_*.yaml`` — per-experiment overrides

Use ``load_experiment_config('configs/exp/e01_v2s_5fold.yaml')`` to get the
merged dict.  Standalone-yaml callers (e.g. legacy code) can use ``load_yaml``.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_DATA_YAML = "configs/data.yaml"


def load_yaml(path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a dict")
    return data


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict:
    """Recursively merge ``overlay`` into ``base``. Overlay wins on leaf values."""
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_experiment_config(
    exp_path,
    *,
    defaults_path: str | None = DEFAULT_DATA_YAML,
) -> dict:
    """Load an experiment yaml and overlay it on the shared data defaults."""
    cfg: dict = {}
    if defaults_path is not None and Path(defaults_path).exists():
        cfg = load_yaml(defaults_path)
    cfg = deep_merge(cfg, load_yaml(exp_path))
    cfg["_exp_path"] = str(exp_path)
    return cfg


def resolve_data_root(config: Mapping[str, Any]) -> Path:
    """Pick the first existing data root from env / config / common Kaggle mounts."""
    paths = config.get("paths", {}) if isinstance(config, dict) else {}
    candidates: list[Path] = []
    env_root = os.environ.get("BIRDCLEF_DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    if paths.get("data_root"):
        candidates.append(Path(paths["data_root"]))
    candidates.extend(
        [
            Path("/kaggle/input/birdclef-2026"),
            Path("/kaggle/input/competitions/birdclef-2026"),
            Path("/kaggle/input/birdclef-2026-repack"),
        ]
    )
    train_csv_name = paths.get("train_csv", "train.csv")
    sample_name = paths.get("sample_submission", "sample_submission.csv")
    for cand in candidates:
        if (cand / train_csv_name).exists() or (cand / sample_name).exists():
            return cand
    return candidates[0] if candidates else Path("data/birdclef-2026")
