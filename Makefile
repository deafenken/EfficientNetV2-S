PYTHON ?= python
CONFIG ?= configs/baseline.yaml
CHECKPOINT ?= outputs/baseline/best.pt
OUTPUT ?= submission.csv
KAGGLE_CONFIG_DIR ?= $(CURDIR)/.kaggle
KAGGLE_BIN ?= $(CURDIR)/.venv/bin/kaggle
KERNEL_DIR ?= kaggle/version_3_lgbm_plus
ABLATION_DIR ?= references/public_baselines/youssefmo942009_version_3_lgbm/ablations
SCORE_ABLATION_DIR ?= references/public_baselines/youssefmo942009_version_3_lgbm/score_ablations
KERNEL_REF ?= longkunshicandyman/birdclef-2026-perch-lgbm-plus

.PHONY: setup setup-pip lock download download-aux inspect debug-train train infer-fallback infer smoke prepare-kaggle-kernel prepare-kaggle-variants push-kaggle-kernel kernel-status submit-completed-variants ablations score-ablations

setup:
	uv sync

setup-pip:
	$(PYTHON) -m pip install -r requirements.txt

lock:
	uv lock
	uv export --no-hashes --no-emit-project --format requirements-txt -o requirements.txt

download:
	KAGGLE_CONFIG_DIR="$(KAGGLE_CONFIG_DIR)" KAGGLE_BIN="$(KAGGLE_BIN)" bash scripts/download_data.sh

download-aux:
	KAGGLE_CONFIG_DIR="$(KAGGLE_CONFIG_DIR)" KAGGLE_BIN="$(KAGGLE_BIN)" bash scripts/download_aux_data.sh

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

prepare-kaggle-kernel:
	bash scripts/prepare_kaggle_kernel.sh $(KERNEL_DIR)

prepare-kaggle-variants:
	$(PYTHON) scripts/prepare_kaggle_variants.py

push-kaggle-kernel:
	bash scripts/kaggle_cmd.sh kernels push -p $(KERNEL_DIR)

kernel-status:
	bash scripts/kaggle_cmd.sh kernels status $(KERNEL_REF)

submit-completed-variants:
	$(PYTHON) scripts/submit_completed_variants.py

ablations:
	$(PYTHON) scripts/create_lgbm_ablation_notebooks.py references/public_baselines/youssefmo942009_version_3_lgbm/version_3_lgbm_plus.ipynb $(ABLATION_DIR)

score-ablations:
	$(PYTHON) scripts/create_v3_score_ablation_notebooks.py references/public_baselines/youssefmo942009_version_3_lgbm/version_3_lgbm.ipynb $(SCORE_ABLATION_DIR)
