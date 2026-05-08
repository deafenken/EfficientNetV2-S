#!/usr/bin/env bash
# scripts/01_download_data.sh — idempotent dataset bootstrap.
#
# Wraps:
#   - scripts/download_data.sh       (BirdCLEF+ 2026 main competition data, ~30 GB)
#   - scripts/download_aux_data.sh   (Perch meta + future aux datasets)
#
# Skips downloads when data is already on disk. Override locations with:
#   BIRDCLEF_DATA_DIR=/path/to/data/birdclef-2026
#   BIRDCLEF_AUX_DIR=/path/to/data/aux

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${BIRDCLEF_DATA_DIR:-$ROOT/data/birdclef-2026}"
AUX_ROOT="${BIRDCLEF_AUX_DIR:-$ROOT/data/aux}"

# Main competition data
if [[ -f "$DATA_ROOT/sample_submission.csv" \
      && -f "$DATA_ROOT/train.csv" \
      && -d "$DATA_ROOT/train_audio" ]]; then
    echo "[skip] main data already at $DATA_ROOT"
else
    echo "[download] main competition data → $DATA_ROOT"
    bash "$ROOT/scripts/download_data.sh" "$DATA_ROOT"
fi

# Aux datasets (Perch meta etc., used by Plan A/C Perch branches)
if [[ -d "$AUX_ROOT/perch-meta" \
      && -n "$(ls -A "$AUX_ROOT/perch-meta" 2>/dev/null || true)" ]]; then
    echo "[skip] aux data already at $AUX_ROOT"
else
    echo "[download] aux datasets → $AUX_ROOT"
    bash "$ROOT/scripts/download_aux_data.sh" "$AUX_ROOT" || \
        echo "  WARN: aux download failed; continuing (Perch branch will need it later)"
fi

echo
echo "✅ data ready. Verify with:"
echo "    PYTHONPATH=src uv run python -m birdclef2026.inspect_data --config configs/baseline.yaml"
