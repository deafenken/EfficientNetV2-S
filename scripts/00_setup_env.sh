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
# Kaggle changed its API auth UI in 2025. Three sources of truth, in priority order:
#   1) KAGGLE_API_TOKEN env var (new-style, recommended for headless servers)
#   2) ~/.kaggle/kaggle.json   (legacy, still fully supported)
#   3) ./.kaggle/kaggle.json   (repo-local legacy fallback used by the Makefile)
if [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
    echo "  found  \$KAGGLE_API_TOKEN (new-style env var)"
elif [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
    echo "  found  ~/.kaggle/kaggle.json (legacy)"
elif [[ -f "$ROOT/.kaggle/kaggle.json" ]]; then
    echo "  found  ./.kaggle/kaggle.json (repo-local legacy)"
elif [[ -f "$HOME/.kaggle/access_token" ]]; then
    cat <<'EOF'
  found  ~/.kaggle/access_token  — BUT recent Kaggle CLI versions do NOT read
  this file directly. Export it as an env var before running download scripts:
    export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
  (Add the export line to ~/.bashrc / ~/.zshrc to make it persistent.)
EOF
else
    cat <<'EOF'
  MISSING Kaggle credentials — needed for `bash scripts/01_download_data.sh`.

  Kaggle now offers two auth flows at https://www.kaggle.com/settings/api :

  Option A — Legacy kaggle.json (RECOMMENDED on first install, still supported)
    1) Scroll past the new "Generate New Token" button.
    2) In the "Legacy API Credentials" block, click "Create Legacy API Key".
    3) Browser downloads kaggle.json.
       mkdir -p "$HOME/.kaggle"
       mv ~/Downloads/kaggle.json "$HOME/.kaggle/kaggle.json"
       chmod 600 "$HOME/.kaggle/kaggle.json"

  Option B — New-style token string (best for headless servers)
    1) Click the top "Generate New Token" button → a popup shows a long
       KGAT_... string (NO file is downloaded).
    2) Copy it and add to your shell rc (~/.bashrc or ~/.zshrc):
         export KAGGLE_API_TOKEN="paste-the-KGAT_...-string-here"
    3) source ~/.bashrc   (or open a fresh shell).

  Either flow requires you to first ACCEPT the competition rules at
  https://www.kaggle.com/competitions/birdclef-2026/rules — otherwise the
  download API returns 403.
EOF
fi

echo
echo "✅ setup_env complete. Next: bash scripts/01_download_data.sh"
