#!/usr/bin/env python3
"""DEPRECATED shim → scripts/kaggle/submit.py (manifest-driven).

The submit logic moved into the ``scripts/kaggle/`` package; variants now live
in ``submissions/manifest.yaml`` instead of a hardcoded dict. This shim is kept
so existing muscle-memory / Makefile targets keep working:

    uv run python scripts/22_kaggle_submit.py --variant b03_eB1b_nfnet_bgnoise

New canonical entry point:

    uv run python scripts/kaggle/submit.py b03_eB1b_nfnet_bgnoise

``submit.main()`` accepts both the positional id and the ``--variant`` alias.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "kaggle"))

import submit  # noqa: E402

if __name__ == "__main__":
    main = submit.main
    main()
