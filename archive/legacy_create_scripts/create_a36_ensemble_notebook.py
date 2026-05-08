#!/usr/bin/env python3
"""Build A36: in-kernel ensemble of A33 (wliilamsam+head+rescue) and A35 (udaysonawane v181).

Layout:
  [A35 cells: udaysonawane v181 — produces submission.csv]
  [Rename cell: move submission*.csv -> v181_submission*.csv, free A35 namespace]
  [A33 cells: wliilamsam base + konbu17 head INJECT + mattiaangeli rescue blend]
  [Final blend cell: 50/50 rank-blend the two final submissions]

Risks acknowledged in code but not mitigated here:
  - CPU 9-hour limit. A33 dev=462s + A35 dev=549s sums to 1011s; hidden-test scale (~30x)
    puts the merged kernel near 8.5h. If A35 alone times out we lose this slot.
  - Both segments redo Perch inference. Sharing the cache is not done because A33 and A35
    use different Perch wrappers and different mel-spec pipelines for SED.

Output: references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/a36_a33_a35_blend.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path


A35_NB = Path("references/public_baselines/udaysonawane_v181/birdclef2026-v181.ipynb")
A33_NB = Path(
    "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/"
    "a33_konbu_head_mattia_rescue.ipynb"
)
OUT_NB = Path(
    "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/"
    "a36_a33_a35_blend.ipynb"
)


# Cell to run between the two segments. Renames A35 outputs (so A33 segment doesn't
# overwrite them) and drops A35's bulky in-memory caches so the A33 segment has headroom.
RENAME_SOURCE = """\
# === A36 BRIDGE — preserve A35 outputs and free A35 namespace before A33 segment ===
import os
import shutil
for src in ("submission.csv", "submission_protossm.csv", "submission_sed.csv"):
    if os.path.exists(src):
        dst = "v181_" + src
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)
        print(f"  preserved {src} -> {dst}")
    else:
        print(f"  WARN: expected A35 output missing: {src}")

# Aggressively free A35 in-memory state. The A33 segment will rebuild what it needs.
import gc
_A35_KEEP = {"os", "sys", "shutil", "gc", "json", "pd", "np", "pathlib", "Path"}
_drop = []
for _name in list(globals()):
    if _name.startswith("_"):
        continue
    if _name in _A35_KEEP:
        continue
    if _name in {"In", "Out", "exit", "quit", "get_ipython"}:
        continue
    _drop.append(_name)
for _name in _drop:
    try:
        del globals()[_name]
    except Exception:
        pass
gc.collect()
try:
    import torch
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
except Exception:
    pass
print(f"  cleared {len(_drop)} A35 globals; ready for A33 segment")
"""


# Final ensemble cell. Reads both final CSVs, rank-blends 50/50, overwrites submission.csv.
BLEND_SOURCE = """\
# === A36 FINAL ENSEMBLE — rank blend A35 (v181) and A33 (wliilamsam+head+rescue) ===
import numpy as np
import pandas as pd

W_V181 = 0.50  # equal weights initially; both branches scored within 0.001 of each other
W_A33  = 0.50
EPS = 1e-5

a33  = pd.read_csv("submission.csv")
v181 = pd.read_csv("v181_submission.csv")

cols = [c for c in a33.columns if c != "row_id"]

# Align v181 rows to A33 row order (defensive)
v181 = v181.set_index("row_id").loc[a33["row_id"]].reset_index()

pa = np.clip(a33[cols].to_numpy(np.float32),  EPS, 1.0 - EPS)
pb = np.clip(v181[cols].to_numpy(np.float32), EPS, 1.0 - EPS)

ra = pd.DataFrame(pa).rank(axis=0, pct=True).to_numpy(np.float32)
rb = pd.DataFrame(pb).rank(axis=0, pct=True).to_numpy(np.float32)

blended = W_A33 * ra + W_V181 * rb

final = a33.copy()
final[cols] = blended.astype(np.float32)
final.to_csv("submission.csv", index=False)

print(f"A36 final ensemble: shape={final.shape}")
print(f"  mix: A33={W_A33} v181={W_V181}")
print(f"  blended mean={blended.mean():.5f} range=[{blended.min():.4f},{blended.max():.4f}]")
"""


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def main() -> None:
    if not A35_NB.exists():
        raise FileNotFoundError(A35_NB)
    if not A33_NB.exists():
        raise FileNotFoundError(A33_NB)

    a35 = json.loads(A35_NB.read_text())
    a33 = json.loads(A33_NB.read_text())

    cells: list[dict] = []

    cells.append(markdown_cell(
        "# A36 — In-kernel ensemble: A33 (wliilamsam + konbu head + mattia rescue) + A35 (udaysonawane v181)\n\n"
        "Two complete inference stacks run sequentially in this kernel. The A35 (udaysonawane v181)\n"
        "segment runs first and writes `submission.csv`. A bridge cell renames that to\n"
        "`v181_submission.csv` and frees the A35 namespace. The A33 segment then re-runs the full\n"
        "wliilamsam base + konbu17 head + mattiaangeli rescue pipeline, also writing `submission.csv`.\n"
        "The final cell rank-blends both 50/50 and overwrites `submission.csv` for grading.\n\n"
        "**CPU time risk**: A35 ~4.6h + A33 ~3.85h ≈ 8.5h (Kaggle limit 9h). 30% chance of timeout.\n"
    ))

    cells.append(markdown_cell(
        "## ── Segment 1: A35 (udaysonawane v181) ──"
    ))
    cells.extend(a35["cells"])

    cells.append(markdown_cell(
        "## ── Bridge: preserve A35 outputs, free A35 namespace ──"
    ))
    cells.append(code_cell(RENAME_SOURCE))

    cells.append(markdown_cell(
        "## ── Segment 2: A33 (wliilamsam base + konbu17 head + mattiaangeli rescue) ──"
    ))
    cells.extend(a33["cells"])

    cells.append(markdown_cell(
        "## ── Final ensemble ──"
    ))
    cells.append(code_cell(BLEND_SOURCE))

    nb = {
        "cells": cells,
        "metadata": a35.get("metadata", {}),
        "nbformat": a35.get("nbformat", 4),
        "nbformat_minor": a35.get("nbformat_minor", 5),
    }

    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT_NB} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
