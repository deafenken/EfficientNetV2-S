#!/usr/bin/env bash
# scripts/01_download_data.sh — thin wrapper around scripts/01_download_data.py.
#
# Pulls BirdCLEF+ 2026 (and optional aux datasets) via kagglehub, which works
# with both the new ``KAGGLE_API_TOKEN`` env var and the legacy ``kaggle.json``.
# Idempotent (kagglehub caches by version; we link the cache into ./data/).
#
# Examples:
#     bash scripts/01_download_data.sh                  # main competition only
#     bash scripts/01_download_data.sh --aux            # + Perch-meta
#     bash scripts/01_download_data.sh --force          # bypass cache
#     bash scripts/01_download_data.sh --no-main --aux  # only aux
#
# All args are forwarded to scripts/01_download_data.py — see ``--help``.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHONPATH=src exec uv run python scripts/01_download_data.py "$@"
