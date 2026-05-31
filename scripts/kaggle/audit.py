#!/usr/bin/env python3
"""Validate submissions/manifest.yaml against on-disk reality.

  uv run python scripts/kaggle/audit.py        # or: make audit-submissions

Checks, for every ``b_variants`` row:
  * kaggle/variants/<id>/ exists and has exactly one .py jupytext source
  * kernel-metadata.json exists and its ``id`` matches the manifest ``kernel_id``
  * dataset-metadata.json exists for the declared dataset_handle slug
  * checkpoint exists locally (WARN only — normally lives on the GPU box)
  * dataset_version / kernel_version are recorded (WARN if null)
  * variant-id numbering has no unexplained gaps (b07→b08 rename is whitelisted)

Exits non-zero if any hard PROBLEM is found; warnings never fail the build.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

import manifest  # noqa: E402

# Intentional gaps in the b-line numbering (documented lineage, not missing).
KNOWN_GAPS = {7}  # b07 was renamed to b08 (commit dc9d398)


def main() -> int:
    data = manifest.load_manifest(ROOT)
    rows = data.get("b_variants", [])
    problems: list[str] = []
    warnings: list[str] = []

    for row in rows:
        vid = row["id"]
        vdir = ROOT / "kaggle/variants" / vid
        meta = vdir / "kernel-metadata.json"

        if not vdir.is_dir():
            problems.append(f"{vid}: kaggle/variants/{vid}/ missing")
            continue

        if not meta.is_file():
            problems.append(f"{vid}: kernel-metadata.json missing")
        else:
            kid = json.loads(meta.read_text()).get("id")
            if row.get("kernel_id") and kid != row["kernel_id"]:
                problems.append(
                    f"{vid}: manifest kernel_id '{row['kernel_id']}' != metadata id '{kid}'"
                )

        pys = [p for p in vdir.glob("*.py") if not p.name.startswith("_")]
        if len(pys) != 1:
            problems.append(
                f"{vid}: expected exactly one .py source, found {[p.name for p in pys]}"
            )

        handle = row.get("dataset_handle")
        if handle:
            slug = handle.split("/", 1)[1]
            dm = ROOT / "kaggle/datasets" / slug / "dataset-metadata.json"
            if not dm.is_file():
                problems.append(f"{vid}: dataset-metadata.json missing for {slug}")

        if not row.get("kernel_only") and row.get("status") != "not_trained":
            ck = row.get("checkpoint")
            if ck and not (ROOT / ck).is_file():
                warnings.append(f"{vid}: checkpoint {ck} not present locally (expected on the GPU box)")
            if row.get("dataset_version") is None:
                warnings.append(f"{vid}: dataset_version not recorded")
            if row.get("kernel_version") is None:
                warnings.append(f"{vid}: kernel_version not recorded")

    # Numbering-gap detection on the bNN prefix.
    nums = sorted(
        int(m.group(1))
        for r in rows
        if (m := re.match(r"b0*(\d+)_", r["id"]))
    )
    if nums:
        for n in range(min(nums), max(nums) + 1):
            if n not in nums and n not in KNOWN_GAPS:
                problems.append(f"b-line numbering gap: b{n:02d} missing (and not a known rename)")

    print(f"[audit] {len(rows)} b-variants checked")
    for w in warnings:
        print(f"  WARN  {w}")
    for p in problems:
        print(f"  FAIL  {p}")
    if problems:
        print(f"[audit] {len(problems)} problem(s).")
        return 1
    print(f"[audit] OK ({len(warnings)} warning(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
