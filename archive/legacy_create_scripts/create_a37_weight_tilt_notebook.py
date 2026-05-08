#!/usr/bin/env python3
"""Build A37: clone of A36 with the final blend tilted toward A33.

A33 alone scored 0.942, A35 alone 0.941. A36 (50/50) hit 0.943.
A37 nudges weights toward the stronger branch: W_A33 = 0.55, W_v181 = 0.45.

Single-line patch over A36's final blend cell.
"""
from __future__ import annotations

import json
from pathlib import Path


BASE_NB = Path(
    "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/"
    "a36_a33_a35_blend.ipynb"
)
OUT_NB = Path(
    "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/"
    "a37_a33_a35_tilt_55_45.ipynb"
)


PATCHES = [
    ("W_V181 = 0.50  # equal weights initially; both branches scored within 0.001 of each other",
     "W_V181 = 0.45  # A37: tilt toward A33 (which scored 0.001 higher solo)"),
    ("W_A33  = 0.50",
     "W_A33  = 0.55  # A37: was 0.50 in A36"),
]


def main() -> None:
    if not BASE_NB.exists():
        raise FileNotFoundError(BASE_NB)

    nb = json.loads(BASE_NB.read_text())
    applied = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        if "A36 FINAL ENSEMBLE" not in src:
            continue
        for old, new in PATCHES:
            if old not in src:
                raise RuntimeError(f"A37 patch not found:\n  {old!r}")
            src = src.replace(old, new)
        cell["source"] = src.splitlines(keepends=True)
        applied = len(PATCHES)
        break

    if applied != len(PATCHES):
        raise RuntimeError(f"A37: expected {len(PATCHES)} patches, applied {applied}")

    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT_NB} ({len(nb['cells'])} cells, {applied} patches applied)")


if __name__ == "__main__":
    main()
