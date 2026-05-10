# %% [markdown]
# # BirdCLEF 2026 — b01: eB1 NFNet-L0 fold-0 (LB verification)
#
# Single-fold inference using `outputs/exp/eB1_nfnet_l0/fold_0/swa.pt`
# (val_auc=0.9790, SWA averaged over last 10 epochs of a 50-epoch run).
# This is a **verification submission** — confirm the high local val_auc
# is reproduced on Kaggle's held-out test set before scaling to 5-fold.
#
# Per Kaggle's code-comp debugging guidance, this notebook is fail-soft:
# we always emit a valid submission.csv first, then try to overwrite with
# the real model output. Any exception in the predict step is caught and
# logged; the kernel exits 0 so the submission status is COMPLETE (with
# baseline ~0.5 AUC if predict broke) instead of "Notebook Threw Exception"
# (which would burn a daily submission slot for no signal).

# %% cell 1 — write zero-fallback submission.csv FIRST (before anything else)
# pandas + the competition data are in Kaggle's base image, so this works
# even before our wheels install runs.
from pathlib import Path
import pandas as _pd

_KAGGLE_INPUT = Path("/kaggle/input/competitions/birdclef-2026")
if not _KAGGLE_INPUT.exists():
    _KAGGLE_INPUT = Path("/kaggle/input/birdclef-2026")
_SAMPLE_SUB = _KAGGLE_INPUT / "sample_submission.csv"

_OUT = Path("/kaggle/working/submission.csv")
_stub = _pd.read_csv(_SAMPLE_SUB)
for _c in _stub.columns[1:]:
    _stub[_c] = 0.0
_stub.to_csv(_OUT, index=False)
print(f"[stub] wrote zero-fallback submission.csv ({len(_stub)} rows × {len(_stub.columns)} cols)")

# %% cell 2 — install timm + torch wheels offline
# Kaggle mounts a `kernel_sources` either at /kaggle/input/<slug>/ or
# /kaggle/input/notebooks/<owner>/<slug>/ depending on the source type.
import glob
import subprocess
try:
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
    print("[pip] OK")
except BaseException as _e:
    print(f"[FATAL] pip install failed: {type(_e).__name__}: {_e}")
    print("[FATAL] keeping zero-fallback submission.csv; exiting cleanly")
    import sys as _sys; _sys.exit(0)

# %% cell 3 — wire up paths + import our package
import os
import sys
import traceback

PKG_INPUT = Path("/kaggle/input/birdclef2026-eb1-pkg-ckpt")
PKG_SRC = PKG_INPUT / "src"
CKPT = PKG_INPUT / "swa.pt"
DATA_YAML = PKG_INPUT / "data.yaml"

try:
    sys.path.insert(0, str(PKG_SRC))
    import yaml as _yaml
    _cfg = _yaml.safe_load(DATA_YAML.read_text())
    _cfg.setdefault("paths", {}).pop("data_root", None)
    WORK_CFG = Path("configs"); WORK_CFG.mkdir(exist_ok=True)
    (WORK_CFG / "data.yaml").write_text(_yaml.safe_dump(_cfg, sort_keys=False))
    print(f"ckpt size : {CKPT.stat().st_size // (1024 * 1024)} MB")

    from birdclef2026.utils.config import resolve_data_root, load_yaml
    DATA_ROOT = resolve_data_root(load_yaml(WORK_CFG / "data.yaml"))
    print(f"DATA_ROOT : {DATA_ROOT}")
    print(f"test_dir  : {DATA_ROOT / 'test_soundscapes'}")

    import pandas as pd
    sample_sub = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    print(f"sample_submission rows: {len(sample_sub)}")
    print("[setup] OK")
except BaseException as _e:
    print(f"[FATAL] setup failed: {type(_e).__name__}: {_e}")
    traceback.print_exc()
    print("[FATAL] keeping zero-fallback submission.csv; exiting cleanly")
    sys.exit(0)

# %% cell 4 — run inference end-to-end via the library (fail-soft)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    import torch
    print(f"CUDA visible: {torch.cuda.is_available()} | torch={torch.__version__} | threads={torch.get_num_threads()}")

    from birdclef2026.inference import predict as _predict_mod
    sys.argv = [
        "predict",
        "--checkpoints", str(CKPT),
        "--defaults", str(WORK_CFG / "data.yaml"),
        "--output", str(_OUT),
        "--batch-size", "4",
        "--amp-dtype", "fp32",
    ]
    _predict_mod.main()
    print("[predict] OK", flush=True)
except BaseException as _e:
    print(f"[FATAL] predict.main() raised: {type(_e).__name__}: {_e}", flush=True)
    traceback.print_exc()
    print("[FATAL] keeping zero-fallback submission.csv; exiting cleanly", flush=True)

# %% cell 5 — quick QC: row count + non-trivial probability spread
import pandas as pd
try:
    sub = pd.read_csv(_OUT)
    print(f"submission shape: {sub.shape}")
    print(f"columns[:6]: {list(sub.columns[:6])}")
    prob_mat = sub.iloc[:, 1:].to_numpy()
    print(f"prob min/mean/max: {prob_mat.min():.4f} / {prob_mat.mean():.4f} / {prob_mat.max():.4f}")
    print(f"rows where max prob < 0.05 (cold): {(prob_mat.max(axis=1) < 0.05).sum()}")
    print(sub.head(3).to_string())
except BaseException as _e:
    print(f"[QC fail] {type(_e).__name__}: {_e}")
