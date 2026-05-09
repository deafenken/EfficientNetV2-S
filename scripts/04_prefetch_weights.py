#!/usr/bin/env python3
"""scripts/04_prefetch_weights.py — single-process pretrained weight prefetch.

Downloads the timm backbone(s) referenced by an experiment yaml into the
HuggingFace cache (``~/.cache/huggingface/hub``) so that subsequent
``torchrun --nproc_per_node=N`` runs hit the disk cache instead of having
N ranks race to the network and exhaust the mirror's connection budget
(SSL EOF / "Cannot send a request, as the client has been closed.").

Usage:
    PYTHONPATH=src uv run python scripts/04_prefetch_weights.py \
        --config configs/exp/e01_v2s_5fold.yaml

    # multiple configs at once:
    PYTHONPATH=src uv run python scripts/04_prefetch_weights.py \
        --config configs/exp/e01_v2s_5fold.yaml configs/exp/e02_nfnet.yaml

Tip: if you're behind a flaky mirror, just re-run — kagglehub-style cache
makes it idempotent (timm/HF hub skip files already on disk).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from birdclef2026.utils.config import load_experiment_config


def _backbone_name(cfg: dict) -> str:
    return str((cfg.get("model", {}) or {}).get("name", "tf_efficientnetv2_s.in21k_ft_in1k"))


def prefetch(name: str) -> None:
    import timm

    print(f"[prefetch] timm.create_model({name!r}, pretrained=True)")
    model = timm.create_model(name, pretrained=True, num_classes=0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[prefetch] ok — {name} loaded ({n_params/1e6:.1f}M params)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", nargs="+", required=True,
        help="One or more experiment yamls to read backbones from.",
    )
    parser.add_argument(
        "--defaults", default="configs/data.yaml",
        help="Shared data config (merged under each exp yaml).",
    )
    args = parser.parse_args()

    if os.environ.get("HF_ENDPOINT"):
        print(f"[env] HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    if os.environ.get("HF_HOME"):
        print(f"[env] HF_HOME={os.environ['HF_HOME']}")

    seen: set[str] = set()
    for cfg_path in args.config:
        cfg = load_experiment_config(cfg_path, defaults_path=args.defaults)
        name = _backbone_name(cfg)
        if name in seen:
            print(f"[skip] {name} already prefetched in this run")
            continue
        seen.add(name)
        prefetch(name)

    print("\n✅ all weights cached. Now run training:")
    print("    make train-ddp FOLD=0")


if __name__ == "__main__":
    main()
