#!/usr/bin/env bash
# scripts/17_kaggle_eB1_submit.sh — Kaggle submission for eB1 NFNet-L0 fold-0.
#
# What it does (~5 min e2e once creds are in place):
#   1. Stage src/birdclef2026/ + configs/data.yaml + outputs/exp/eB1_nfnet_l0/fold_0/swa.pt
#      into a Kaggle dataset payload dir
#   2. Push (or version) the dataset:
#        longkunshicandyman/birdclef2026-eb1-pkg-ckpt
#   3. Convert kaggle/variants/b01_eB1_nfnet_l0/*.py → *.ipynb
#   4. Push the kernel:
#        longkunshicandyman/birdclef-2026-b01-eb1-nfnet-l0
#   5. Tail the kernel run status until it finishes (or ~9h cap)
#
# What you still do by hand AFTER this script finishes:
#   - Open the kernel run on Kaggle and click "Submit to Competition"
#     (BirdCLEF 2026 is a code competition; final submission is per-run).
#
# Prereqs (must satisfy before invoking):
#   1. ``pip install kaggle`` (or ``uv tool install kaggle``) somewhere on PATH
#   2. ``~/.kaggle/kaggle.json`` (or ``$KAGGLE_CONFIG_DIR/kaggle.json``) with
#      {"username": "...", "key": "..."} — chmod 600
#   3. Run from L40 box (intern user) inside birdclef-2026/ — needs the swa.pt
#      checkpoint that lives there
#
# Usage:
#   bash scripts/17_kaggle_eB1_submit.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --------------------------------------------------------------------------- #
# 0. Preflight: kaggle CLI + creds
# --------------------------------------------------------------------------- #

if ! command -v kaggle >/dev/null 2>&1; then
    echo "[FATAL] kaggle CLI not found. Install via:"
    echo "        pip install kaggle    (or)    uv tool install kaggle"
    exit 1
fi

if [[ -z "${KAGGLE_CONFIG_DIR:-}" ]]; then
    KAGGLE_CONFIG_DIR="$HOME/.kaggle"
fi
if [[ ! -f "$KAGGLE_CONFIG_DIR/kaggle.json" ]]; then
    echo "[FATAL] no kaggle.json at $KAGGLE_CONFIG_DIR/kaggle.json"
    echo "        Drop your token there (chmod 600) and re-run."
    exit 1
fi
chmod 600 "$KAGGLE_CONFIG_DIR/kaggle.json" 2>/dev/null || true

CKPT="outputs/exp/eB1_nfnet_l0/fold_0/swa.pt"
if [[ ! -f "$CKPT" ]]; then
    echo "[FATAL] checkpoint not found: $CKPT"
    echo "        Run this script from the L40 box where eB1 trained."
    exit 1
fi

# --------------------------------------------------------------------------- #
# 1. Stage dataset payload
# --------------------------------------------------------------------------- #

PAYLOAD="/tmp/birdclef2026-eb1-pkg-ckpt"
echo "[stage] payload at $PAYLOAD"
rm -rf "$PAYLOAD"
mkdir -p "$PAYLOAD/src/birdclef2026"
cp -r src/birdclef2026/. "$PAYLOAD/src/birdclef2026/"
# Strip __pycache__ everywhere — Kaggle doesn't need them and they bloat the upload.
find "$PAYLOAD" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
cp configs/data.yaml "$PAYLOAD/data.yaml"
cp "$CKPT" "$PAYLOAD/swa.pt"
cp kaggle/datasets/birdclef2026-eb1-pkg-ckpt/dataset-metadata.json "$PAYLOAD/"
echo "[stage] payload contents:"
( cd "$PAYLOAD" && find . -maxdepth 3 -type f | sort | head -40 )
echo "[stage] total size: $(du -sh "$PAYLOAD" | cut -f1)"

# --------------------------------------------------------------------------- #
# 2. Push (or version) the dataset
# --------------------------------------------------------------------------- #

DS_ID="longkunshicandyman/birdclef2026-eb1-pkg-ckpt"
DS_EXISTS="$(kaggle datasets status "$DS_ID" 2>&1 | tr '[:upper:]' '[:lower:]' || true)"
if echo "$DS_EXISTS" | grep -q "ready\|complete\|pending\|processing"; then
    echo "[ds] $DS_ID exists — pushing new version"
    kaggle datasets version -p "$PAYLOAD" -m "eB1 fold-0 SWA + package $(date +%Y-%m-%d_%H%M)" --dir-mode zip
else
    echo "[ds] $DS_ID not found — creating"
    kaggle datasets create -p "$PAYLOAD" --dir-mode zip
fi

# --------------------------------------------------------------------------- #
# 3. Convert .py → .ipynb
# --------------------------------------------------------------------------- #

KERNEL_DIR="kaggle/variants/b01_eB1_nfnet_l0"
PY_SRC="$KERNEL_DIR/birdclef-2026-b01-eb1-nfnet-l0.py"
IPYNB="$KERNEL_DIR/birdclef-2026-b01-eb1-nfnet-l0.ipynb"
echo "[nb] converting $PY_SRC → $IPYNB"
python3 scripts/_py_to_ipynb.py "$PY_SRC" "$IPYNB"

# --------------------------------------------------------------------------- #
# 4. Push the kernel
# --------------------------------------------------------------------------- #

echo "[kernel] pushing $KERNEL_DIR"
kaggle kernels push -p "$KERNEL_DIR"

# --------------------------------------------------------------------------- #
# 5. Tail status
# --------------------------------------------------------------------------- #

KERNEL_ID="longkunshicandyman/birdclef-2026-b01-eb1-nfnet-l0"
echo "[kernel] watching status (Ctrl-C to detach; kernel keeps running):"
kaggle kernels status "$KERNEL_ID" || true
echo ""
echo "[done] Once the kernel finishes successfully, open it on Kaggle and"
echo "       click 'Submit to Competition' to submit the run to BirdCLEF 2026."
echo "       Kernel URL: https://www.kaggle.com/code/$KERNEL_ID"
