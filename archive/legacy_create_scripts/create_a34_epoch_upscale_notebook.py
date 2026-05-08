#!/usr/bin/env python3
"""Build A34: A33 base + limprog v0.9999 epoch upscale + mattiaangeli SED_W=0.40.

Two patches over A33:

1. Cell #4 (`CFG`): bump training schedule to limprog v0.9999's values
   - ProtoSSM:     n_epochs 80/40 -> 100/60, patience 20/8 -> 25/12
   - ResidualSSM:  n_epochs 40/20 -> 60/40,  patience 12/6 -> 25/8
   - MLP probes:   max_iter 500/200 -> 700/300

2. Cell #40 (final blend): SED_W 0.35 -> 0.40 (mattiaangeli's calibrated value),
   PROTO_W = 1 - SED_W - HEAD_W = 0.45.

Output: references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/a34_a33_epoch_upscale.ipynb
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
    "a34_a33_epoch_upscale.ipynb"
)


# All replacements are exact-string. They will fail loudly if A33's source drifts.
EPOCH_PATCHES = [
    # ProtoSSM
    (
        '"n_epochs":        80  if MODE == "train" else 40,',
        '"n_epochs":        100 if MODE == "train" else 60,',
    ),
    (
        '"patience":        20  if MODE == "train" else 8,',
        '"patience":        25 if MODE == "train" else 12,',
    ),
    # ResidualSSM
    (
        '"n_epochs": 40  if MODE == "train" else 20,',
        '"n_epochs": 60  if MODE == "train" else 40,',
    ),
    (
        '"patience": 12  if MODE == "train" else 6,',
        '"patience": 25  if MODE == "train" else 8,',
    ),
    # MLP probes
    (
        '"max_iter": 500  if MODE == "train" else 200,',
        '"max_iter": 700  if MODE == "train" else 300,',
    ),
]

# Final blend: bump SED_W to mattiaangeli's calibrated 0.40.
MIX_PATCHES = [
    (
        "HEAD_W  = 0.15\nSED_W   = 0.35\nPROTO_W = 1.0 - SED_W - HEAD_W  # 0.50",
        "HEAD_W  = 0.15\nSED_W   = 0.40  # A34: mattiaangeli's calibrated value\nPROTO_W = 1.0 - SED_W - HEAD_W  # 0.45",
    ),
]


def patch_cell_source(src: str, patches: list[tuple[str, str]]) -> tuple[str, int]:
    n_applied = 0
    for old, new in patches:
        if old not in src:
            raise RuntimeError(f"A34 patch not found in source:\n  expected: {old!r}")
        if src.count(old) > 1:
            raise RuntimeError(
                f"A34 patch is not unique in source ({src.count(old)} occurrences):\n  {old!r}"
            )
        src = src.replace(old, new)
        n_applied += 1
    return src, n_applied


def main() -> None:
    if not BASE_NB.exists():
        raise FileNotFoundError(BASE_NB)

    nb = json.loads(BASE_NB.read_text())

    epoch_done = mix_done = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        if not epoch_done and '"n_epochs"' in src and '"patience"' in src and '"max_iter"' in src:
            new_src, n = patch_cell_source(src, EPOCH_PATCHES)
            cell["source"] = new_src.splitlines(keepends=True)
            epoch_done = True
            print(f"  applied {n} epoch patches to cell")
            continue
        if (
            not mix_done
            and "HEAD_W  = 0.15" in src
            and "SED_W   = 0.35" in src
            and "PROTO_W = 1.0 - SED_W - HEAD_W" in src
        ):
            new_src, n = patch_cell_source(src, MIX_PATCHES)
            cell["source"] = new_src.splitlines(keepends=True)
            mix_done = True
            print(f"  applied {n} mix patches to cell")
            continue

    if not epoch_done:
        raise RuntimeError("A34: did not find epoch/patience config cell to patch")
    if not mix_done:
        raise RuntimeError("A34: did not find mix-weight cell to patch")

    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT_NB} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
