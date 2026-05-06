#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-kaggle/version_3_lgbm_plus}"
NOTEBOOK_SRC="references/public_baselines/youssefmo942009_version_3_lgbm/version_3_lgbm_plus.ipynb"
KERNEL_SLUG="${KERNEL_SLUG:-birdclef-2026-perch-lgbm-plus}"

mkdir -p "${OUT_DIR}"
cp "${NOTEBOOK_SRC}" "${OUT_DIR}/version_3_lgbm_plus.ipynb"
cp "kaggle/kernel-metadata.json" "${OUT_DIR}/kernel-metadata.json"

if [[ -n "${KAGGLE_USERNAME:-}" ]]; then
  python -c "import json, pathlib; p=pathlib.Path('${OUT_DIR}/kernel-metadata.json'); d=json.loads(p.read_text()); d['id']='${KAGGLE_USERNAME}/${KERNEL_SLUG}'; p.write_text(json.dumps(d, indent=2) + '\n')"
fi

echo "Kaggle kernel package ready at ${OUT_DIR}"
echo "Edit ${OUT_DIR}/kernel-metadata.json and replace REPLACE_WITH_KAGGLE_USERNAME if it is still present."
echo "After adding .kaggle/kaggle.json, push with:"
echo "  KAGGLE_CONFIG_DIR=.kaggle .venv/bin/kaggle kernels push -p ${OUT_DIR}"
