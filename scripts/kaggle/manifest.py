"""Load submissions/manifest.yaml → Variant objects for the submit tool.

The manifest is the SINGLE SOURCE OF TRUTH (see submissions/manifest.yaml and
docs/NAMING.md). This module turns each ``b_variants`` row into a frozen
``Variant`` the pipeline consumes, deriving the two paths that used to be
hand-written in the old VARIANTS dict:

  * payload_dir = /tmp/<dataset slug>            (staging dir, wiped per run)
  * kernel_dir  = kaggle/variants/<id>           (kernel-metadata.json + .py)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

MANIFEST_REL = "submissions/manifest.yaml"


@dataclass(frozen=True)
class Variant:
    """One submittable b-line variant."""

    id: str
    kernel_dir: Path
    kernel_id: str | None = None
    checkpoint: str | None = None          # relative to repo root
    dataset_handle: str | None = None      # <owner>/<slug>
    payload_dir: Path | None = None        # /tmp/<slug>; None for kernel-only
    notes_label: str = ""
    kernel_only: bool = False
    extra_payload: tuple[tuple[str, str], ...] = ()


def manifest_path(root: Path) -> Path:
    return root / MANIFEST_REL


def load_manifest(root: Path) -> dict:
    p = manifest_path(root)
    if not p.is_file():
        sys.exit(f"[FATAL] manifest not found: {p}")
    return yaml.safe_load(p.read_text())


def load_b_variants(root: Path) -> dict[str, Variant]:
    """Return ``{id: Variant}`` for every row under ``b_variants``."""
    data = load_manifest(root)
    out: dict[str, Variant] = {}
    for row in data.get("b_variants", []):
        vid = row["id"]
        handle = row.get("dataset_handle")
        slug = handle.split("/", 1)[1] if handle else None
        out[vid] = Variant(
            id=vid,
            kernel_dir=root / "kaggle/variants" / vid,
            kernel_id=row.get("kernel_id"),
            checkpoint=row.get("checkpoint"),
            dataset_handle=handle,
            payload_dir=(Path("/tmp") / slug) if slug else None,
            notes_label=row.get("notes_label", ""),
            kernel_only=bool(row.get("kernel_only", False)),
            extra_payload=tuple(tuple(e) for e in row.get("extra_payload", [])),
        )
    return out
