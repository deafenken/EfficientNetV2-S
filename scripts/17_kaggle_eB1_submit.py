#!/usr/bin/env python3
"""DEPRECATED → use: uv run python scripts/kaggle/submit.py b01_eB1_nfnet_l0

This single-purpose script (eB1 NFNet-L0 fold-0) was superseded by the
manifest-driven scripts/kaggle/ package. Forwarding for back-compat.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "kaggle"))
import submit  # noqa: E402

if __name__ == "__main__":
    print(
        "[deprecated] scripts/17_kaggle_eB1_submit.py → "
        "scripts/kaggle/submit.py b01_eB1_nfnet_l0",
        file=sys.stderr,
    )
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    sys.argv = [sys.argv[0], "b01_eB1_nfnet_l0", *flags]
    submit.main()
