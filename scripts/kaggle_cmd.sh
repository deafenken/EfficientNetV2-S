#!/usr/bin/env bash
set -euo pipefail

export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-${PWD}/.kaggle}"
KAGGLE_BIN="${KAGGLE_BIN:-${PWD}/.venv/bin/kaggle}"

if [[ -z "${KAGGLE_API_TOKEN:-}" && -f "${KAGGLE_CONFIG_DIR}/kaggle.json" ]]; then
  token="$("${PWD}/.venv/bin/python" - <<'PY'
import json
import pathlib

path = pathlib.Path(".kaggle/kaggle.json")
data = json.loads(path.read_text())
key = data.get("key", "")
print(key if key.startswith("KGAT_") else "")
PY
)"
  if [[ -n "${token}" ]]; then
    export KAGGLE_API_TOKEN="${token}"
  fi
fi

exec "${KAGGLE_BIN}" "$@"

