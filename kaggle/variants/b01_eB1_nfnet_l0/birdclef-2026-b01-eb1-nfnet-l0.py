# %% [markdown]
# # BirdCLEF 2026 — b01: eB1 NFNet-L0 fold-0 (LB verification)
#
# Single-fold inference using `outputs/exp/eB1_nfnet_l0/fold_0/swa.pt`
# (val_auc=0.9790, SWA averaged over last 10 epochs of a 50-epoch run).
# This is a **verification submission** — confirm the high local val_auc
# is reproduced on Kaggle's held-out test set before scaling to 5-fold.
#
# Pipeline: drop our `birdclef2026` package onto sys.path, call the
# library's own `predict.py` end-to-end. No model code is duplicated here;
# the notebook is a thin shim around our existing inference module.

# %% cell 1 — install timm + torch wheels offline
# Kaggle mounts a `kernel_sources` either at /kaggle/input/<slug>/ or
# /kaggle/input/notebooks/<owner>/<slug>/ depending on the source type.
# Locate `requirements.txt` by globbing both spots so we don't care.
import glob
import subprocess
_req = glob.glob("/kaggle/input/**/birdclef-2026-download-wheels/requirements.txt", recursive=True)
assert _req, "no birdclef-2026-download-wheels found under /kaggle/input/"
_REQ = _req[0]
_WHEELS = _REQ.replace("/requirements.txt", "/wheels")
print("wheels at:", _WHEELS)
subprocess.check_call([
    "pip", "install", "-q", "--no-index",
    "-r", _REQ,
    "--find-links", _WHEELS,
])

# %% cell 2 — wire up paths
import os
import sys
import shutil
from pathlib import Path

PKG_INPUT = Path("/kaggle/input/birdclef2026-eb1-pkg-ckpt")
PKG_SRC = PKG_INPUT / "src"
CKPT = PKG_INPUT / "swa.pt"
DATA_YAML = PKG_INPUT / "data.yaml"

# Make our package importable.
sys.path.insert(0, str(PKG_SRC))

# `predict.py` resolves `data_root` via `paths.data_root` in the yaml AND a
# Kaggle fallback (/kaggle/input/birdclef-2026). Prefer the Kaggle fallback,
# so we drop the local-dev `data_root` line from the yaml before loading.
import yaml as _yaml
_cfg = _yaml.safe_load(DATA_YAML.read_text())
_cfg.setdefault("paths", {}).pop("data_root", None)

# Stage the trimmed yaml under the working dir so `predict.main()` can read it
# at the default relative path.
WORK_CFG = Path("configs"); WORK_CFG.mkdir(exist_ok=True)
(WORK_CFG / "data.yaml").write_text(_yaml.safe_dump(_cfg, sort_keys=False))

print("ckpt size :", CKPT.stat().st_size // (1024 * 1024), "MB")
print("ckpt path :", CKPT)

# %% cell 3 — sanity: confirm Kaggle data root + sample_submission rows
from birdclef2026.utils.config import resolve_data_root, load_yaml
DATA_ROOT = resolve_data_root(load_yaml(WORK_CFG / "data.yaml"))
print("DATA_ROOT :", DATA_ROOT)
print("test_dir  :", DATA_ROOT / "test_soundscapes")

import pandas as pd
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"
sample_sub = pd.read_csv(SAMPLE_SUB)
print(f"sample_submission rows: {len(sample_sub)}")

# %% cell 4 — run inference end-to-end via the library
# Equivalent to:
#   python -m birdclef2026.inference.predict \
#     --checkpoints /kaggle/input/.../swa.pt \
#     --defaults configs/data.yaml \
#     --output /kaggle/working/submission.csv \
#     --batch-size 32 --amp-dtype bf16
import torch
print("CUDA available:", torch.cuda.is_available(),
      "device count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)

from birdclef2026.inference import predict as _predict_mod

sys.argv = [
    "predict",
    "--checkpoints", str(CKPT),
    "--defaults", str(WORK_CFG / "data.yaml"),
    "--output", "/kaggle/working/submission.csv",
    "--batch-size", "32",
    "--amp-dtype", "bf16" if torch.cuda.is_available() else "fp32",
]
_predict_mod.main()

# %% cell 5 — quick QC: row count + non-trivial probability spread
sub = pd.read_csv("/kaggle/working/submission.csv")
print("submission shape:", sub.shape)
print("columns[:6]:", list(sub.columns[:6]))
prob_cols = sub.columns[1:]
prob_mat = sub[prob_cols].to_numpy()
print(f"prob min/mean/max: {prob_mat.min():.4f} / {prob_mat.mean():.4f} / {prob_mat.max():.4f}")
print(f"rows where max prob < 0.05 (cold): {(prob_mat.max(axis=1) < 0.05).sum()}")
sub.head(3)
