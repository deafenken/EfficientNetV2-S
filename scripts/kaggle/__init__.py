"""BirdCLEF-2026 Kaggle submission tooling.

Consolidated, manifest-driven replacement for the old triplicated submit
scripts (17/18/22). Modules:

  auth.py      single credential resolver (PAT for kagglehub, legacy json for
               `kaggle kernels push`)
  manifest.py  loads submissions/manifest.yaml → Variant objects
  lib.py       the four pipeline steps (stage → dataset → convert → push)
  submit.py    CLI: `uv run python scripts/kaggle/submit.py <variant_id>`
  audit.py     `make audit-submissions` — validate manifest ↔ on-disk reality

Run from the repo root. See docs/SUBMISSION_WORKFLOW.md and docs/NAMING.md.
"""
