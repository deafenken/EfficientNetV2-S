#!/usr/bin/env python3
"""Manifest-driven Kaggle submission driver for BirdCLEF-2026.

Single, consolidated replacement for the old 17/18/22 scripts. Variants are
defined in ``submissions/manifest.yaml`` (the source of truth), not hardcoded.

Usage (run from repo root, on the box that has the checkpoints — the L40):
  uv run python scripts/kaggle/submit.py b03_eB1b_nfnet_bgnoise
  uv run python scripts/kaggle/submit.py b09_eB1b_R2_clean --kernel-only
  uv run python scripts/kaggle/submit.py b03_eB1b_nfnet_bgnoise --dry-run
  uv run python scripts/kaggle/submit.py --list

Pipeline (identical to the proven old 22): stage payload → upload dataset
(kagglehub, IPv4-only) → poll until ready → convert .py→.ipynb → kaggle
kernels push. ``--dry-run`` does stage + convert only (NO Kaggle network calls)
— use it to verify a new variant before spending a real submission.

Auth: split-stack (PAT for kagglehub upload, legacy ~/.kaggle/kaggle.json for
kernel push). See scripts/kaggle/auth.py and docs/SUBMISSION_WORKFLOW.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # allow `import auth/lib/manifest` as siblings
ROOT = HERE.parents[1]                 # scripts/kaggle → repo root

import auth      # noqa: E402
import lib       # noqa: E402
import manifest  # noqa: E402


def main() -> None:
    variants = manifest.load_b_variants(ROOT)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "variant", nargs="?", choices=sorted(variants),
        help="variant id from submissions/manifest.yaml (e.g. b03_eB1b_nfnet_bgnoise)",
    )
    parser.add_argument(
        "--variant", dest="variant_opt", choices=sorted(variants),
        help="(back-compat alias for the positional arg, matching old 22 CLI)",
    )
    parser.add_argument(
        "--kernel-only", action="store_true",
        help="skip stage + dataset upload; only convert .py→.ipynb and push the kernel",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="stage + convert only; make NO Kaggle network calls (safe verification)",
    )
    parser.add_argument(
        "--list", action="store_true", help="list submittable variants and exit",
    )
    args = parser.parse_args()

    if args.list:
        for vid in sorted(variants):
            v = variants[vid]
            tag = "kernel-only" if v.kernel_only else (v.checkpoint or "")
            print(f"  {vid:32s} {v.kernel_id or '':48s} {tag}")
        return

    vid = args.variant or args.variant_opt
    if not vid:
        parser.error("variant id required (e.g. b03_eB1b_nfnet_bgnoise); use --list to see all")

    v = variants[vid]
    print(f"[main] variant: {v.id}")
    print(f"[main] dataset: {v.dataset_handle or '(none — blend-only kernel)'}")
    print(f"[main] kernel : {v.kernel_dir.relative_to(ROOT)}")

    kernel_only = args.kernel_only or v.kernel_only

    if args.dry_run:
        if not kernel_only:
            lib.stage_payload(v, ROOT)
        lib.convert_notebook(v, ROOT)
        print("[dry-run] OK — staged + converted; skipped dataset upload + kernel push.")
        return

    if kernel_only:
        if v.kernel_only:
            print("[main] kernel-only variant: skipping stage + dataset upload")
        else:
            print("[main] --kernel-only: skipping stage + dataset upload")
        lib.convert_notebook(v, ROOT)
        lib.push_kernel(v, ROOT)
        return

    auth.ensure_pat_token()
    lib.stage_payload(v, ROOT)
    lib.push_dataset(v)
    lib.convert_notebook(v, ROOT)
    lib.push_kernel(v, ROOT)


if __name__ == "__main__":
    main()
