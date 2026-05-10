#!/usr/bin/env bash
# scripts/16_summarize_explore.sh — print B+C trial vs e01 baseline.
#
# Reads outputs/exp/eB*_*/fold_0/{swa,oof}_auc.txt and outputs/exp/eC*_*/fold_0/...
# and outputs a comparison table sorted by Δ vs e01 SWA.
#
# Usage:
#   bash scripts/16_summarize_explore.sh           # print to stdout
#   bash scripts/16_summarize_explore.sh > file    # capture

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

E01_SWA=$(cat outputs/exp/e01_v2s_5fold/fold_0/swa_auc.txt 2>/dev/null || echo "0.8553")
E02_SWA=$(cat outputs/exp/e02_v2s_5fold_v2/fold_0/swa_auc.txt 2>/dev/null || echo "n/a")

echo "BirdCLEF-2026 explore summary — fold 0 SWA val_auc"
echo "Generated: $(date +'%F %T')"
echo ""
printf "%-32s %-9s %-9s %-10s %s\n" "exp_id" "swa" "best" "Δ_vs_e01" "notes"
printf "%-32s %-9s %-9s %-10s %s\n" "--------------------------------" "--------" "--------" "---------" "-----"

# Baselines first
printf "%-32s %-9s %-9s %-10s %s\n" "e01_v2s_5fold (baseline)" "$E01_SWA" \
    "$(cat outputs/exp/e01_v2s_5fold/fold_0/oof_auc.txt 2>/dev/null || echo n/a)" \
    "0.0000" "BC2024 warm + bf16 + sampler+ swa+head_lr3"
if [[ "$E02_SWA" != "n/a" ]]; then
    delta=$(awk -v a="$E02_SWA" -v b="$E01_SWA" 'BEGIN{printf "%+.4f", a-b}')
    printf "%-32s %-9s %-9s %-10s %s\n" "e02_v2s_5fold_v2 (ablation)" "$E02_SWA" \
        "$(cat outputs/exp/e02_v2s_5fold_v2/fold_0/oof_auc.txt 2>/dev/null || echo n/a)" \
        "$delta" "BC2025 SWA bb + fp32 + ESC-50 BG"
fi

# Then B and C trials
for d in outputs/exp/eB*_*/fold_0 outputs/exp/eC*_*/fold_0; do
    [[ -d "$d" ]] || continue
    exp=$(basename "$(dirname "$d")")
    swa=$(cat "$d/swa_auc.txt" 2>/dev/null || echo "n/a")
    best=$(cat "$d/oof_auc.txt" 2>/dev/null || echo "n/a")
    if [[ "$swa" == "n/a" ]]; then
        delta="n/a"
    else
        delta=$(awk -v a="$swa" -v b="$E01_SWA" 'BEGIN{printf "%+.4f", a-b}')
    fi
    printf "%-32s %-9s %-9s %-10s\n" "$exp" "$swa" "$best" "$delta"
done

echo ""
echo "Decision criteria:"
echo "  Δ ≥ +0.010 → significant winner; consider 5-fold scaling"
echo "  Δ ∈ [-0.005, +0.005] → noise band; no signal"
echo "  Δ ≤ -0.005 → definitively negative; skip"
