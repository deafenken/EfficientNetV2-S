#!/usr/bin/env bash
# scripts/15_parallel_explore.sh — run B (architecture) and C (single-knob
# ablation) trial pipelines in parallel on disjoint 2-GPU subsets.
#
# Each pipeline runs 3 trials sequentially in its own GPU slot:
#   B: GPUs 0,1, MASTER_PORT 29400, configs/exp/eB1..eB3
#   C: GPUs 2,3, MASTER_PORT 29401, configs/exp/eC1..eC3
#
# ~5h end-to-end on 4×L40 bf16 (each trial fold 0 × 50 ep ≈ 100 min on 2 GPU).
#
# Usage:
#   bash scripts/15_parallel_explore.sh
#
# Resumable: skips trials whose fold_0/swa.pt already exists. Single-trial
# failure (rc != 0) does NOT abort the pipeline — next trial still runs.
#
# Output:
#   /tmp/explore_B.log, /tmp/explore_C.log — per-pipeline aggregate logs
#   outputs/exp/<exp_id>/fold_0/{best.pt,last.pt,swa.pt,log.csv,...}
#   /tmp/explore_summary.txt — comparison table (auto-written when both finish)

set -uo pipefail   # NOTE: no -e — keep going past trial-level failures

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

run_pipeline() {
    local label="$1" gpus="$2" port="$3"
    local log="/tmp/explore_${label}.log"
    shift 3
    : > "$log"
    echo "[$(date +'%F %T')] [$label] pipeline start (gpus=$gpus port=$port)" | tee -a "$log"
    for cfg in "$@"; do
        local exp_id; exp_id=$(basename "$cfg" .yaml)
        local out_dir="outputs/exp/$exp_id/fold_0"
        if [[ -f "$out_dir/swa.pt" ]]; then
            echo "[$(date +'%F %T')] [$label] skip $exp_id (swa.pt already exists)" | tee -a "$log"
            continue
        fi
        echo "[$(date +'%F %T')] [$label] start $exp_id" | tee -a "$log"
        MASTER_PORT="$port" CUDA_VISIBLE_DEVICES="$gpus" NPROC=2 \
            bash scripts/10_train.sh "$cfg" 0 >> "$log" 2>&1
        local rc=$?
        echo "[$(date +'%F %T')] [$label] done $exp_id rc=$rc" | tee -a "$log"
    done
    echo "[$(date +'%F %T')] [$label] PIPELINE COMPLETE" | tee -a "$log"
}

run_pipeline B "0,1" 29400 \
    configs/exp/eB1_nfnet_l0.yaml \
    configs/exp/eB2_convnextv2_tiny.yaml \
    configs/exp/eB3_efficientnetv2_b2.yaml &
B_PID=$!

run_pipeline C "2,3" 29401 \
    configs/exp/eC1_head_lr_mul1.yaml \
    configs/exp/eC2_pos_weight2.yaml \
    configs/exp/eC3_head_dropout02.yaml &
C_PID=$!

wait "$B_PID" "$C_PID"
echo "[$(date +'%F %T')] both pipelines done; writing summary" | tee -a /tmp/explore_master.log
bash scripts/16_summarize_explore.sh > /tmp/explore_summary.txt 2>&1
cat /tmp/explore_summary.txt | tee -a /tmp/explore_master.log
