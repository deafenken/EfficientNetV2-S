#!/usr/bin/env python3
"""Build A41: A33 with mattiaangeli's rescue boost coefficients amplified.

Rationale: A33 = 0.942, A36 = 0.943 (A33 + ensemble). The +0.001 came partly from rescue
gates catching marginal cases. If we amplify the boost, more samples that match the gate
conditions get a stronger nudge. Could go either way: stronger rescues catch more correct
positives but also more false positives.

Three boost values changed:
  FAKE_ONLY_BLEND  0.12 -> 0.18  (stronger Proto-only rescue)
  PROTO_CONT_BLEND 0.15 -> 0.22  (stronger temporal-continuity rescue)
  SED_ONLY_BLEND   0.12 -> 0.18  (stronger SED-spike rescue)

Thresholds (which decide *which* samples get rescued) are unchanged — only the magnitude
of the rescue is amplified.
"""
from __future__ import annotations
import json
from pathlib import Path

BASE_NB = Path(
    "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/"
    "a33_konbu_head_mattia_rescue.ipynb"
)
OUT_NB = Path(
    "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/"
    "a41_a33_amplified_rescue.ipynb"
)

PATCHES = [
    ("FAKE_ONLY_BLEND = 0.12",
     "FAKE_ONLY_BLEND = 0.18  # A41: amplified from 0.12"),
    ("PROTO_CONT_BLEND     = 0.15",
     "PROTO_CONT_BLEND     = 0.22  # A41: amplified from 0.15"),
    ("SED_ONLY_BLEND    = 0.12",
     "SED_ONLY_BLEND    = 0.18  # A41: amplified from 0.12"),
]


def main() -> None:
    nb = json.loads(BASE_NB.read_text())
    target_idx = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c.get("source", []))
        if "Cell 12 (A33)" in s and "FAKE_ONLY_BLEND" in s:
            target_idx = i
            break
    if target_idx is None:
        raise RuntimeError("A41: could not find A33 final blend cell")

    src = "".join(nb["cells"][target_idx].get("source", []))
    for old, new in PATCHES:
        if old not in src:
            raise RuntimeError(f"A41 patch not found:\n  {old!r}")
        if src.count(old) > 1:
            raise RuntimeError(f"A41 patch not unique ({src.count(old)}x):\n  {old!r}")
        src = src.replace(old, new)
    nb["cells"][target_idx]["source"] = src.splitlines(keepends=True)

    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT_NB} ({len(nb['cells'])} cells, {len(PATCHES)} patches in cell #{target_idx})")


if __name__ == "__main__":
    main()
