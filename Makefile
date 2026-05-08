PYTHON ?= python
CONFIG ?= configs/baseline.yaml
CHECKPOINT ?= outputs/baseline/best.pt
OUTPUT ?= submission.csv
KAGGLE_CONFIG_DIR ?= $(CURDIR)/.kaggle
KAGGLE_BIN ?= $(CURDIR)/.venv/bin/kaggle
KERNEL_DIR ?= kaggle/version_3_lgbm_plus
KERNEL_REF ?= longkunshicandyman/birdclef-2026-perch-lgbm-plus

.PHONY: help setup-env data setup setup-pip lock download download-aux \
	inspect debug-train train infer-fallback infer smoke \
	prepare-kaggle-kernel push-kaggle-kernel kernel-status

help:
	@echo "Common targets:"
	@echo "  setup-env             — bash scripts/00_setup_env.sh (uv sync + GPU/Kaggle check)"
	@echo "  data                  — bash scripts/01_download_data.sh (idempotent)"
	@echo "  inspect               — PYTHONPATH=src python -m birdclef2026.inspect_data"
	@echo "  debug-train / train   — legacy single-process baseline (will be replaced in Commit 2)"
	@echo "  infer / infer-fallback— legacy single-checkpoint inference"
	@echo "  smoke                 — scripts/smoke_checks.py (postprocess sanity)"
	@echo "  prepare-kaggle-kernel — sync kaggle/<dir> for submission notebook push"
	@echo "  push-kaggle-kernel    — kaggle kernels push -p kaggle/<dir>"
	@echo "  kernel-status         — kaggle kernels status <ref>"

# ---- bootstrap (idempotent) ----
setup-env:
	bash scripts/00_setup_env.sh

data:
	bash scripts/01_download_data.sh

# ---- environment management ----
setup:
	uv sync

setup-pip:
	$(PYTHON) -m pip install -r requirements.txt

lock:
	uv lock
	uv export --no-hashes --no-emit-project --format requirements-txt -o requirements.txt

# ---- raw download wrappers (called by `make data`; kept for explicit use) ----
download:
	KAGGLE_CONFIG_DIR="$(KAGGLE_CONFIG_DIR)" KAGGLE_BIN="$(KAGGLE_BIN)" bash scripts/download_data.sh

download-aux:
	KAGGLE_CONFIG_DIR="$(KAGGLE_CONFIG_DIR)" KAGGLE_BIN="$(KAGGLE_BIN)" bash scripts/download_aux_data.sh

# ---- legacy single-process baseline (replaced by Commit 2 DDP entry) ----
inspect:
	PYTHONPATH=src $(PYTHON) -m birdclef2026.inspect_data --config $(CONFIG)

debug-train:
	PYTHONPATH=src $(PYTHON) -m birdclef2026.train --config $(CONFIG) --debug

train:
	PYTHONPATH=src $(PYTHON) -m birdclef2026.train --config $(CONFIG)

infer-fallback:
	PYTHONPATH=src $(PYTHON) -m birdclef2026.infer --config $(CONFIG) --output $(OUTPUT) --fallback zeros

infer:
	PYTHONPATH=src $(PYTHON) -m birdclef2026.infer --config $(CONFIG) --checkpoint $(CHECKPOINT) --output $(OUTPUT)

smoke:
	PYTHONPATH=src $(PYTHON) scripts/smoke_checks.py

# ---- Kaggle kernel push (still useful for the future inference notebook) ----
prepare-kaggle-kernel:
	bash scripts/prepare_kaggle_kernel.sh $(KERNEL_DIR)

push-kaggle-kernel:
	bash scripts/kaggle_cmd.sh kernels push -p $(KERNEL_DIR)

kernel-status:
	bash scripts/kaggle_cmd.sh kernels status $(KERNEL_REF)
