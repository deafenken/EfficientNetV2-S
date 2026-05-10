#!/usr/bin/env bash
# scripts/20_run_pseudo_r1.sh — generate Round-1 pseudo-label CSV from the
# eB1b + e03 fold-0 ensemble.
#
# Pipeline context:
#   eB1b (NFNet + BG noise + cross-species mixup, fold-0)   LB ~0.78–0.80
#   e03  (V2-S features_only + warmstart + BG noise, fold-0) LB target >= 0.7
#   We average their sigmoid predictions over a sliding non-overlapping 5 s
#   window across every unlabeled file in train_soundscapes/, then keep
#   classes with prob >= prob_min as soft labels. This CSV feeds eB1c / e04
#   (R1 pseudo-label experiments) via `data.pseudo_label_csv`.
#
# Designed for the L40x4 training box (cwd=/workspace/birdclef-2026). Single-
# GPU inference is fine — bottleneck is disk read + soundscape count, not GEMM.
#
# Usage (on L40 box):
#   bash scripts/20_run_pseudo_r1.sh
#
# Output:
#   outputs/pseudo/r1_eB1b_e03.csv
#
# Pre-reqs:
#   * outputs/exp/eB1b_nfnet_bgnoise/fold_0/swa.pt   (parallel run #1 done)
#   * outputs/exp/e03_v2s_fixed/fold_0/swa.pt        (parallel run #2 done)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EB1B_CKPT="outputs/exp/eB1b_nfnet_bgnoise/fold_0/swa.pt"
E03_CKPT="outputs/exp/e03_v2s_fixed/fold_0/swa.pt"
OUT_CSV="outputs/pseudo/r1_eB1b_e03.csv"

# ---- Sanity checks (cheap, fail fast) -------------------------------------
# Refuse to run before the parallel training round actually finished.
for ckpt in "$EB1B_CKPT" "$E03_CKPT"; do
    if [[ ! -f "$ckpt" ]]; then
        echo "[ERR] missing checkpoint: $ckpt" >&2
        echo "[ERR] run scripts/19_parallel_fix.sh and wait for swa.pt first" >&2
        exit 1
    fi
done

if [[ ! -f configs/data.yaml ]]; then
    echo "[ERR] configs/data.yaml missing" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT_CSV")"

echo "[pseudo] checkpoints:"
echo "  $EB1B_CKPT"
echo "  $E03_CKPT"
echo "[pseudo] out: $OUT_CSV"
echo

# ---- Run inference --------------------------------------------------------
# `time` so the user sees wall-clock duration in the log.
time PYTHONPATH=src uv run python -m birdclef2026.inference.predict_pseudo \
    --checkpoints "$EB1B_CKPT" "$E03_CKPT" \
    --defaults configs/data.yaml \
    --out "$OUT_CSV" \
    --prob-min 0.5 \
    --batch-size 32

# ---- Report ---------------------------------------------------------------
if [[ ! -s "$OUT_CSV" ]]; then
    echo "[ERR] $OUT_CSV is empty after run" >&2
    exit 1
fi

ROWS=$(wc -l < "$OUT_CSV")
SIZE=$(du -h "$OUT_CSV" | cut -f1)
echo
echo "[done] $OUT_CSV"
echo "  rows (incl. header): $ROWS"
echo "  size:                $SIZE"
