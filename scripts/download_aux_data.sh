#!/usr/bin/env bash
set -euo pipefail

export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-${PWD}/.kaggle}"
KAGGLE_BIN="${KAGGLE_BIN:-${PWD}/.venv/bin/kaggle}"
OUT_DIR="${1:-data/aux}"

mkdir -p "${OUT_DIR}"
mkdir -p "${KAGGLE_CONFIG_DIR}"
chmod 700 "${KAGGLE_CONFIG_DIR}"

# Public cache used by the Perch + ProtoSSM/LGBM notebooks.
"${KAGGLE_BIN}" datasets download -d jaejohn/perch-meta -p "${OUT_DIR}/perch-meta"
unzip -n "${OUT_DIR}/perch-meta/perch-meta.zip" -d "${OUT_DIR}/perch-meta"

echo "Auxiliary data ready at ${OUT_DIR}"

