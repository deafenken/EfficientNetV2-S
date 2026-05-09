PYTHON ?= python
CONFIG ?= configs/baseline.yaml
CHECKPOINT ?= outputs/baseline/best.pt
OUTPUT ?= submission.csv
KAGGLE_CONFIG_DIR ?= $(CURDIR)/.kaggle
KAGGLE_BIN ?= $(CURDIR)/.venv/bin/kaggle
KERNEL_DIR ?= kaggle/version_3_lgbm_plus
KERNEL_REF ?= longkunshicandyman/birdclef-2026-perch-lgbm-plus

# Commit-2 training pipeline knobs.
DATA_CONFIG ?= configs/data.yaml
EXP ?= configs/exp/e01_v2s_5fold.yaml
FOLD ?= 0
FOLD_CSV ?= outputs/folds/folds.csv
MEL_STATS ?= outputs/stats/mel_stats.json
MEL_STATS_FILES ?= 4000

.PHONY: help setup-env data setup setup-pip lock download download-aux \
	inspect debug-train train infer-fallback infer smoke \
	folds mel-stats prefetch-weights train-ddp \
	predict assemble-oof \
	prepare-kaggle-kernel push-kaggle-kernel kernel-status

help:
	@echo "Common targets:"
	@echo "  setup-env             — bash scripts/00_setup_env.sh (uv sync + GPU/Kaggle check)"
	@echo "  data                  — bash scripts/01_download_data.sh (idempotent download)"
	@echo "  inspect               — PYTHONPATH=src python -m birdclef2026.inspect_data"
	@echo "  smoke                 — scripts/smoke_checks.py (postprocess sanity)"
	@echo ""
	@echo "Commit 2 training pipeline:"
	@echo "  make folds            — build outputs/folds/folds.csv (StratifiedGroupKFold by filename)"
	@echo "  make mel-stats        — compute outputs/stats/mel_stats.json (paste mean/std into data.yaml)"
	@echo "  make prefetch-weights EXP=configs/exp/e01_v2s_5fold.yaml"
	@echo "                        — single-process timm pretrained download (run before train-ddp)"
	@echo "  make train-ddp EXP=configs/exp/e01_v2s_5fold.yaml FOLD=0"
	@echo "                        — torchrun --nproc_per_node=\$$N for one fold"
	@echo ""
	@echo "Commit 3 inference / submission:"
	@echo "  make assemble-oof EXP_DIR=outputs/exp/e01_v2s_5fold KIND=swa"
	@echo "                        — concat fold_*/swa_oof.npz → exp_dir/swa_oof.npz + AUC"
	@echo "  make predict EXP_DIR=outputs/exp/e01_v2s_5fold CKPT=swa.pt OUTPUT=submission.csv"
	@echo "                        — sliding-window test inference + N-fold ensemble"
	@echo ""
	@echo "Legacy single-process baseline (kept for now):"
	@echo "  debug-train / train   — python -m birdclef2026.train"
	@echo "  infer / infer-fallback— python -m birdclef2026.infer"
	@echo ""
	@echo "Kaggle kernel push (for the future submission notebook):"
	@echo "  prepare-kaggle-kernel — sync kaggle/<dir>"
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

# ---- Commit 2 training pipeline ----
folds:
	PYTHONPATH=src uv run python scripts/03_make_folds.py --config $(DATA_CONFIG) --out $(FOLD_CSV)

mel-stats:
	PYTHONPATH=src uv run python scripts/02_compute_mel_stats.py \
		--config $(DATA_CONFIG) --out $(MEL_STATS) --max-files $(MEL_STATS_FILES)

prefetch-weights:
	PYTHONPATH=src uv run python scripts/04_prefetch_weights.py \
		--config $(EXP) --defaults $(DATA_CONFIG)

train-ddp:
	bash scripts/10_train.sh $(EXP) $(FOLD)

# ---- Commit 3 inference + submission ----
EXP_DIR ?= outputs/exp/e01_v2s_5fold
CKPT ?= swa.pt
KIND ?= swa
OUTPUT ?= submission.csv

assemble-oof:
	PYTHONPATH=src uv run python -m birdclef2026.inference.assemble_oof \
		--exp-dir $(EXP_DIR) --kind $(KIND)

predict:
	PYTHONPATH=src uv run python -m birdclef2026.inference.predict \
		--exp-dir $(EXP_DIR) --ckpt-name $(CKPT) \
		--defaults $(DATA_CONFIG) --output $(OUTPUT)

# ---- legacy single-process baseline (kept until Commit 2 supersedes) ----
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
