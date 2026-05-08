#!/usr/bin/env python3
"""scripts/01_download_data.py — kagglehub-based dataset bootstrap.

Modern replacement for the bash-around-kaggle-CLI flow. ``kagglehub``:
  * reads ``KAGGLE_API_TOKEN`` env var (new flow) **or** ``~/.kaggle/kaggle.json``
    (legacy)
  * downloads + auto-unzips into ``~/.cache/kagglehub/competitions/<slug>/<ver>/``
  * caches by version, so re-runs are no-ops

After the cache path is materialized we symlink it to ``data/birdclef-2026/``
(override with ``--data-dir``) so existing configs keep working unchanged
(``configs/data.yaml :: paths.data_root``).

Usage:
    uv run python scripts/01_download_data.py            # main competition only
    uv run python scripts/01_download_data.py --aux      # + Perch-meta
    uv run python scripts/01_download_data.py --force    # bypass cache
    uv run python scripts/01_download_data.py --no-main --aux   # only aux
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

COMPETITION_SLUG = "birdclef-2026"
DEFAULT_DATA_DIR = Path("data/birdclef-2026")
DEFAULT_AUX_DIR = Path("data/aux")

# Auxiliary public datasets used by various plans. Add slugs here when needed.
AUX_DATASET_SLUGS: list[tuple[str, str]] = [
    # (slug, friendly_dirname)
    ("jaejohn/perch-meta", "perch-meta"),
]


def _check_credentials() -> None:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    home_json = Path.home() / ".kaggle" / "kaggle.json"
    repo_json = Path(".kaggle/kaggle.json")
    if home_json.exists() or repo_json.exists():
        return
    raise SystemExit(
        "ERR: missing Kaggle credentials.\n"
        "  Run scripts/00_setup_env.sh first, or set up:\n"
        "    export KAGGLE_API_TOKEN='KGAT_...'  (modern, recommended)\n"
        "  OR\n"
        "    mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
    )


def _link_or_copy(src: Path, dst: Path) -> None:
    """Materialize ``src`` (kagglehub cache directory) at ``dst``.

    Idempotent rules:
      * if ``dst`` is a symlink to ``src`` already → no-op
      * if ``dst`` is a real non-empty directory → respect it, do nothing
      * if ``dst`` is empty / a stale symlink → replace with a symlink to ``src``
    Falls back to ``shutil.copytree`` on filesystems that reject symlinks.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_real = Path(os.path.realpath(src))

    if dst.is_symlink():
        if Path(os.path.realpath(dst)) == src_real:
            print(f"[ok]   {dst} already linked to {src_real}")
            return
        dst.unlink()
    elif dst.is_dir():
        if any(dst.iterdir()):
            print(f"[skip] {dst} already populated; not overwriting (delete it manually if you want a re-link)")
            return
        dst.rmdir()

    try:
        os.symlink(src_real, dst, target_is_directory=True)
        print(f"[link] {dst} -> {src_real}")
    except OSError as e:
        print(f"[warn] symlink failed ({e}); falling back to copytree (slow, uses disk)")
        shutil.copytree(src_real, dst)
        print(f"[copy] {dst}")


def download_competition(slug: str, target: Path, force: bool) -> Path:
    import kagglehub

    print(f"[hub] competition_download('{slug}', force={force})")
    cache = Path(kagglehub.competition_download(slug, force_download=force))
    print(f"[hub] cache: {cache}")
    _link_or_copy(cache, target)
    return target


def download_dataset(slug: str, friendly: str, aux_root: Path, force: bool) -> Path:
    import kagglehub

    print(f"[hub] dataset_download('{slug}', force={force})")
    cache = Path(kagglehub.dataset_download(slug, force_download=force))
    print(f"[hub] cache: {cache}")
    target = aux_root / friendly
    _link_or_copy(cache, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"Where to expose the competition data (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--aux-dir", type=Path, default=DEFAULT_AUX_DIR,
        help=f"Where to expose auxiliary datasets (default: {DEFAULT_AUX_DIR})",
    )
    parser.add_argument(
        "--aux", action="store_true",
        help="Also pull auxiliary datasets (Perch-meta, etc.)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass the kagglehub cache and re-download from Kaggle",
    )
    parser.add_argument(
        "--no-main", action="store_true",
        help="Skip the main competition (use with --aux to fetch only aux)",
    )
    args = parser.parse_args()

    _check_credentials()

    if not args.no_main:
        download_competition(COMPETITION_SLUG, args.data_dir, args.force)

    if args.aux:
        for slug, friendly in AUX_DATASET_SLUGS:
            download_dataset(slug, friendly, args.aux_dir, args.force)

    print("\n✅ data ready. Verify with:")
    print("    PYTHONPATH=src uv run python -m birdclef2026.inspect_data --config configs/baseline.yaml")


if __name__ == "__main__":
    main()
