#!/usr/bin/env python3
"""scripts/05_prep_esc50_ambient.py — filter ESC-50 to ambient-only subset.

The full ESC-50 dataset (~600 MB, 6000 wav files via the
``mmoreaux/environmental-sound-classification-50`` kagglehub dataset) covers
50 sound categories including bird-like ones (``chirping_birds``, ``crow``,
``hen``, ``rooster``). Mixing those into bird recordings would create label
noise. This script symlinks ONLY the ambient categories matching BirdCLEF
2025 2nd Place's published BG-noise list — rain, wind, insects, thunderstorm,
frog, crickets — into ``data/aux/esc50_ambient/`` for the SED dataset's
``BackgroundNoise`` augmentor to pick up.

Idempotent: re-running just refreshes symlinks; ``--force`` rebuilds.

Usage:
    PYTHONPATH=src uv run python scripts/05_prep_esc50_ambient.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Sydorskyy's ESC-50 ambient categories (from BC2025 2nd Place's BG-mix
# `OneOf(BackgroundNoise(esc50_categories=...))` config).
AMBIENT_CATEGORIES = ("rain", "wind", "insects", "thunderstorm", "frog", "crickets")


def find_esc50_root() -> Path:
    """Locate the kagglehub-cached ESC-50 root containing esc50.csv."""
    candidates = [
        Path.home() / ".cache/kagglehub/datasets/mmoreaux/environmental-sound-classification-50",
        Path("/root/.cache/kagglehub/datasets/mmoreaux/environmental-sound-classification-50"),
    ]
    for cand in candidates:
        if not cand.exists():
            continue
        for p in cand.rglob("esc50.csv"):
            return p.parent
    raise FileNotFoundError(
        "ESC-50 not in kagglehub cache. Run:\n"
        "  python -c 'import kagglehub; kagglehub.dataset_download(\"mmoreaux/environmental-sound-classification-50\")'"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("data/aux/esc50_ambient"),
        help="Where to write symlinks to ambient-category wav files.",
    )
    parser.add_argument(
        "--categories", nargs="+", default=list(AMBIENT_CATEGORIES),
        help="ESC-50 category names to include (default: BC2025 2nd ambient list).",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild from scratch.")
    args = parser.parse_args()

    root = find_esc50_root()
    csv_path = root / "esc50.csv"
    audio_dir_candidates = [root / "audio", root / "ESC-50-master/audio"]
    audio_dir = next((d for d in audio_dir_candidates if d.exists()), None)
    if audio_dir is None:
        raise FileNotFoundError(
            f"ESC-50 audio dir not found under {root} (tried: {audio_dir_candidates})"
        )

    if args.force and args.out.exists():
        for p in args.out.iterdir():
            p.unlink() if p.is_symlink() or p.is_file() else None
    args.out.mkdir(parents=True, exist_ok=True)

    cat_set = set(args.categories)
    n_total = 0
    n_per_cat: dict[str, int] = {c: 0 for c in cat_set}
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cat = row["category"]
            if cat not in cat_set:
                continue
            src = audio_dir / row["filename"]
            if not src.exists():
                # some uploads scatter audio into nested dirs; rglob fallback
                matches = list(audio_dir.rglob(row["filename"]))
                if not matches:
                    continue
                src = matches[0]
            dst = args.out / row["filename"]
            if dst.exists():
                if dst.is_symlink() and dst.resolve() == src.resolve():
                    n_total += 1
                    n_per_cat[cat] += 1
                    continue
                dst.unlink()
            os.symlink(src, dst)
            n_total += 1
            n_per_cat[cat] += 1

    print(f"[esc50_ambient] {n_total} files in {args.out}")
    for cat in sorted(cat_set):
        print(f"  {cat:<14} {n_per_cat.get(cat, 0)}")
    if n_total < 200:
        print(
            "[esc50_ambient] WARN: very few files — check the ESC-50 cache "
            "or the --categories list",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
