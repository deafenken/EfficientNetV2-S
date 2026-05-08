#!/usr/bin/env bash
# scripts/00_setup_env.sh — idempotent dev / training-machine bootstrap.
#
# Run on the 4×L40 box (or your local dev box) once after `git clone`,
# and again whenever pyproject.toml / uv.lock changes.
#
# Steps:
#   1. Verify tooling (uv, git)
#   2. `uv sync` to install pinned deps into ./.venv
#   3. Report GPU presence (nvidia-smi -L)
#   4. Verify Kaggle credentials (~/.kaggle/kaggle.json or ./.kaggle/kaggle.json)
#
# Safe to re-run.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

step() { printf '\n[%s] %s\n' "$1" "$2"; }

step 1/4 "Tooling check"
command -v git >/dev/null || { echo "ERROR: git not found"; exit 1; }
if ! command -v uv >/dev/null; then
    echo "ERROR: uv not found."
    echo "  Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "  uv:  $(uv --version)"
echo "  git: $(git --version)"

step 2/4 "Install dependencies (uv sync)"
uv sync

step 3/4 "GPU check"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || echo "  WARN: nvidia-smi present but no GPU listed"
else
    echo "  WARN: nvidia-smi not found — training will fall back to CPU."
fi

step 4/4 "Kaggle credentials"
if [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
    echo "  found  ~/.kaggle/kaggle.json"
elif [[ -f "$ROOT/.kaggle/kaggle.json" ]]; then
    echo "  found  ./.kaggle/kaggle.json (repo-local)"
else
    cat <<'EOF'
  MISSING kaggle.json — needed for `bash scripts/01_download_data.sh`.
  Steps:
    1) https://www.kaggle.com/settings/account → 'Create New Token'
    2) mkdir -p "$HOME/.kaggle"
    3) mv ~/Downloads/kaggle.json "$HOME/.kaggle/kaggle.json"
    4) chmod 600 "$HOME/.kaggle/kaggle.json"
EOF
fi

echo
echo "✅ setup_env complete. Next: bash scripts/01_download_data.sh"
