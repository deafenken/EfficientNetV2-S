#!/usr/bin/env bash
set -euo pipefail

COMPETITION="birdclef-2026"
OUT_DIR="${1:-data/birdclef-2026}"
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-${PWD}/.kaggle}"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"

mkdir -p "${OUT_DIR}"
mkdir -p "${KAGGLE_CONFIG_DIR}"
chmod 700 "${KAGGLE_CONFIG_DIR}"
"${KAGGLE_BIN}" competitions download -c "${COMPETITION}" -p "${OUT_DIR}"
unzip -n "${OUT_DIR}/${COMPETITION}.zip" -d "${OUT_DIR}"

echo "Data ready at ${OUT_DIR}"
