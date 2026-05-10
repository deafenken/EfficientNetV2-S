#!/usr/bin/env bash
# scripts/21_parallel_pseudo_train.sh — launch the two R1 pseudo-label
# experiments in parallel on a 4-GPU box (2 GPUs each):
#
#   e04  V2-S + pseudo-label R1   (fork of e03)  → CUDA 0,1
#   eB1c NFNet + pseudo-label R1  (fork of eB1b) → CUDA 2,3
#
# Both target fold 0 (override with arg). Designed for the L40x4 training box.
#
# Usage:
#   bash scripts/21_parallel_pseudo_train.sh                  # fold 0
#   bash scripts/21_parallel_pseudo_train.sh 1                # fold 1
#
# Logs:
#   /tmp/e04_fold<F>.log
#   /tmp/eB1c_fold<F>.log
#
# Monitor:
#   tail -f /tmp/e04_fold0.log /tmp/eB1c_fold0.log
#
# Wait for both to finish (in another shell):
#   wait <e04_pid> <eB1c_pid>          # PIDs printed at launch
#   # or simply:
#   while pgrep -f "train_ddp.*e04_v2s_pseudo\|train_ddp.*eB1c_nfnet_pseudo" >/dev/null; do sleep 30; done

set -euo pipefail

FOLD="${1:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PSEUDO_CSV="outputs/pseudo/r1_eB1b_e03.csv"

# ---- Sanity checks (cheap, fail fast) -------------------------------------
for cfg in configs/exp/e04_v2s_pseudo.yaml configs/exp/eB1c_nfnet_pseudo.yaml; do
    [[ -f "$cfg" ]] || { echo "[ERR] missing config: $cfg" >&2; exit 1; }
done

if [[ ! -f outputs/stats/mel_stats.json ]]; then
    echo "[ERR] outputs/stats/mel_stats.json missing — run 'make mel-stats' first" >&2
    exit 1
fi

# Pseudo csv must exist AND have meaningful content.
# A failed pseudo run can leave behind a header-only or zero-byte csv; both
# would silently train without the R1 signal we're trying to inject.
if [[ ! -f "$PSEUDO_CSV" ]]; then
    echo "[ERR] missing pseudo-label csv: $PSEUDO_CSV" >&2
    echo "[ERR] run scripts/20_run_pseudo_r1.sh first" >&2
    exit 1
fi
PSEUDO_ROWS=$(wc -l < "$PSEUDO_CSV")
if [[ "$PSEUDO_ROWS" -le 1000 ]]; then
    echo "[ERR] $PSEUDO_CSV has only $PSEUDO_ROWS rows (<= 1000) — looks broken" >&2
    echo "[ERR] re-run scripts/20_run_pseudo_r1.sh and verify before proceeding" >&2
    exit 1
fi
echo "[ok] pseudo csv: $PSEUDO_CSV  rows=$PSEUDO_ROWS"

# Both yamls reference /workspace/birdclef-2026/data/aux/esc50_ambient (absolute).
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
for exp in e04_v2s_pseudo eB1c_nfnet_pseudo; do
    out="outputs/exp/${exp}/fold_${FOLD}"
    if [[ -d "$out" ]] && find "$out" -name '*.pt' -print -quit | grep -q .; then
        echo "[WARN] $out already has ckpts; re-running will overwrite."
    fi
done

# ---- Launch ---------------------------------------------------------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

E04_LOG="/tmp/e04_fold${FOLD}.log"
EB1C_LOG="/tmp/eB1c_fold${FOLD}.log"

echo "[launch] e04  V2-S + R1 pseudo  → CUDA 0,1 → $E04_LOG"
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0,1 nohup \
    uv run torchrun \
        --nproc_per_node=2 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:29503 \
        -m birdclef2026.training.train_ddp \
        --config configs/exp/e04_v2s_pseudo.yaml \
        --fold "$FOLD" \
        --defaults configs/data.yaml \
    > "$E04_LOG" 2>&1 &
E04_PID=$!

# Tiny stagger so the two torchruns don't race on rdzv socket bind.
sleep 2

echo "[launch] eB1c NFNet + R1 pseudo → CUDA 2,3 → $EB1C_LOG"
PYTHONPATH=src CUDA_VISIBLE_DEVICES=2,3 nohup \
    uv run torchrun \
        --nproc_per_node=2 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:29504 \
        -m birdclef2026.training.train_ddp \
        --config configs/exp/eB1c_nfnet_pseudo.yaml \
        --fold "$FOLD" \
        --defaults configs/data.yaml \
    > "$EB1C_LOG" 2>&1 &
EB1C_PID=$!

echo
echo "[pid]    e04=$E04_PID  eB1c=$EB1C_PID"
echo
echo "Monitor:"
echo "  tail -f $E04_LOG $EB1C_LOG"
echo
echo "Wait for both to finish (this shell):"
echo "  wait $E04_PID $EB1C_PID"
