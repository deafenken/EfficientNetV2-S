#!/usr/bin/env python3
"""DEPRECATED → use: uv run python scripts/kaggle/submit.py b02_e01_v2s_baseline

This single-purpose script (e01 V2-S baseline fold-0) was superseded by the
manifest-driven scripts/kaggle/ package. It also contained a bug — it staged
the eb1-pkg-ckpt dataset-metadata.json for the e01 dataset, corrupting e01's
metadata on re-run. The new package derives the metadata path from the dataset
handle, so that bug is gone. Forwarding for back-compat.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "kaggle"))
import submit  # noqa: E402

if __name__ == "__main__":
    print(
        "[deprecated] scripts/18_kaggle_e01_submit.py → "
        "scripts/kaggle/submit.py b02_e01_v2s_baseline",
        file=sys.stderr,
    )
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    sys.argv = [sys.argv[0], "b02_e01_v2s_baseline", *flags]
    submit.main()
