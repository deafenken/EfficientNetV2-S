#!/usr/bin/env bash
# scripts/19_parallel_fix.sh — launch the two domain-adapter fix experiments
# in parallel on a 4-GPU box (2 GPUs each):
#
#   e03  V2-S fixed (features_only + warmstart_strict + BG noise)  → CUDA 0,1
#   eB1b NFNet + BG noise + cross-species mixup                     → CUDA 2,3
#
# Both target fold 0 (override with arg). Designed for the L40x4 training box.
#
# Usage:
#   bash scripts/19_parallel_fix.sh                  # fold 0
#   bash scripts/19_parallel_fix.sh 1                # fold 1
#
# Logs:
#   /tmp/e03_fold<F>.log
#   /tmp/eB1b_fold<F>.log
#
# Monitor:
#   tail -f /tmp/e03_fold0.log /tmp/eB1b_fold0.log
#
# Wait for both to finish (in another shell):
#   wait <e03_pid> <eB1b_pid>          # PIDs printed at launch
#   # or simply:
#   while pgrep -f "train_ddp.*e03_v2s_fixed\|train_ddp.*eB1b_nfnet_bgnoise" >/dev/null; do sleep 30; done

set -euo pipefail

FOLD="${1:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---- Sanity checks (cheap, fail fast) -------------------------------------
for cfg in configs/exp/e03_v2s_fixed.yaml configs/exp/eB1b_nfnet_bgnoise.yaml; do
    [[ -f "$cfg" ]] || { echo "[ERR] missing config: $cfg" >&2; exit 1; }
done

if [[ ! -f outputs/stats/mel_stats.json ]]; then
    echo "[ERR] outputs/stats/mel_stats.json missing — run 'make mel-stats' first" >&2
    exit 1
fi

# Both yamls reference /workspace/birdclef-2026/data/aux/esc50_ambient (absolute).
# Warn (not fail) if it's not there — train_ddp.py will print its own warn line
# and proceed without BG noise, which kills the whole point of this run.
ESC_DIR="/workspace/birdclef-2026/data/aux/esc50_ambient"
if [[ ! -d "$ESC_DIR" ]]; then
    echo "[WARN] $ESC_DIR not found — BG noise will be silently disabled."
    echo "[WARN] If you intend to run on a non-L40 box, edit the yamls' background_noise.dirs first."
fi

# Need 4 GPUs (the whole reason this script exists)
N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
if [[ "$N_GPUS" -lt 4 ]]; then
    echo "[ERR] need >=4 GPUs, got $N_GPUS" >&2
    exit 1
fi

# Pre-existing fold dirs would clobber on re-run; warn explicitly.
for exp in e03_v2s_fixed eB1b_nfnet_bgnoise; do
    out="outputs/exp/${exp}/fold_${FOLD}"
    if [[ -d "$out" ]] && find "$out" -name '*.pt' -print -quit | grep -q .; then
        echo "[WARN] $out already has ckpts; re-running will overwrite."
    fi
done

# ---- Launch ---------------------------------------------------------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

E03_LOG="/tmp/e03_fold${FOLD}.log"
EB1B_LOG="/tmp/eB1b_fold${FOLD}.log"

echo "[launch] e03  V2-S fixed  → CUDA 0,1 → $E03_LOG"
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0,1 nohup \
    uv run torchrun \
        --nproc_per_node=2 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:29501 \
        -m birdclef2026.training.train_ddp \
        --config configs/exp/e03_v2s_fixed.yaml \
        --fold "$FOLD" \
        --defaults configs/data.yaml \
    > "$E03_LOG" 2>&1 &
E03_PID=$!

# Tiny stagger so the two torchruns don't race on rdzv socket bind.
sleep 2

echo "[launch] eB1b NFNet+BG    → CUDA 2,3 → $EB1B_LOG"
PYTHONPATH=src CUDA_VISIBLE_DEVICES=2,3 nohup \
    uv run torchrun \
        --nproc_per_node=2 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:29502 \
        -m birdclef2026.training.train_ddp \
        --config configs/exp/eB1b_nfnet_bgnoise.yaml \
        --fold "$FOLD" \
        --defaults configs/data.yaml \
    > "$EB1B_LOG" 2>&1 &
EB1B_PID=$!

echo
echo "[pid]    e03=$E03_PID  eB1b=$EB1B_PID"
echo
echo "Monitor:"
echo "  tail -f $E03_LOG $EB1B_LOG"
echo
echo "Wait for both to finish (this shell):"
echo "  wait $E03_PID $EB1B_PID"
