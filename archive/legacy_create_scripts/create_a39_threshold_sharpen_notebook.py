#!/usr/bin/env python3
"""Build A39: A33 + activate the per-class threshold sharpening that wliilamsam
calibrated but left commented out (cell 7f-3 + cell 10 line ~165).

Single-line uncomment in A33 base."""
from __future__ import annotations
import json
from pathlib import Path

BASE_NB = Path(
    "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/"
    "a33_konbu_head_mattia_rescue.ipynb"
)
OUT_NB = Path(
    "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/"
    "a39_a33_threshold_sharpen.ipynb"
)

OLD = "# probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS)"
NEW = "probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS)  # A39: activated"


def main() -> None:
    nb = json.loads(BASE_NB.read_text())
    applied = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        if OLD in src:
            cnt = src.count(OLD)
            if cnt != 1:
                raise RuntimeError(f"A39: expected 1 occurrence, found {cnt}")
            src = src.replace(OLD, NEW)
            cell["source"] = src.splitlines(keepends=True)
            applied += 1
            break
    if applied != 1:
        raise RuntimeError("A39: did not find the commented apply line")
    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT_NB} ({len(nb['cells'])} cells, 1 patch applied)")


if __name__ == "__main__":
    main()
