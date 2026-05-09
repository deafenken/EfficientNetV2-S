#!/usr/bin/env bash
# scripts/10_train.sh — torchrun wrapper for one fold of one experiment.
#
# Usage:
#   bash scripts/10_train.sh <exp_yaml> <fold>
#   bash scripts/10_train.sh configs/exp/e01_v2s_5fold.yaml 0
#
# Override GPU count or per-rank batch size via env vars:
#   NPROC=2 bash scripts/10_train.sh configs/exp/e01_v2s_5fold.yaml 0
#   CUDA_VISIBLE_DEVICES=0,1 NPROC=2 bash scripts/10_train.sh ... 0
#
# Logs/checkpoints land under outputs/exp/<exp_id>/fold_<fold>/.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <exp_yaml> <fold> [extra args ...]" >&2
    exit 2
fi

EXP_YAML="$1"; shift
FOLD="$1"; shift

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NPROC="${NPROC:-1}"
DEFAULTS="${DEFAULTS:-configs/data.yaml}"

# Silence the torchaudio 2.8→2.9 `load_with_torchcodec` deprecation spam.
# It fires once per torchaudio.load() call → workers × ranks → screenfuls of noise.
# The message is informational only; remove this line if you want to track the
# deprecation again.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning:torchaudio}"

echo "[train] exp=$EXP_YAML fold=$FOLD nproc=$NPROC defaults=$DEFAULTS"

PYTHONPATH=src exec uv run torchrun \
    --standalone \
    --nproc_per_node="$NPROC" \
    -m birdclef2026.training.train_ddp \
    --config "$EXP_YAML" \
    --fold "$FOLD" \
    --defaults "$DEFAULTS" \
    "$@"
