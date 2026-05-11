# %% [markdown]
# # BirdCLEF 2026 — b09: eB1b R2 clean (single NFNet, no PL)
#
# Single-fold inference using `outputs/exp/eB1b_R2_clean/fold_0/swa.pt`.
# This is the b03 baseline retrained with the three "actually move the
# domain gap" knobs from the BirdCLEF 2024/2025 winner writeups, minus the
# pseudo-label round (which needs a multi-arch teacher we don't have yet):
#   - **P1 target-domain BG**: `train_soundscapes/` replaces ESC-50 as the
#     background-noise pool, plus log-uniform `RandomAmplitude` modelling
#     the near/far SNR seen on passive soundscape recordings.
#   - **P3 sharper targets**: `label_smoothing` 0.05 → 0.02.
#   - **P2 wide TTA at inference**: ±2.5 s offset crops with chunk smoothing
#     `[0.1, 0.2, 0.4, 0.2, 0.1]` across each file's 12 windows
#     (2025-1st: +0.012 LB; 2024-3rd: chunk smoothing).
#
# **Why we submit this**: validate the P1+P2+P3 stack vs b03 (LB 0.798). If
# LB ≥ 0.83, we unlock the proper noisy-student round (need to train a
# second arch — V2-S clean / EffNet-B3 — first, then 3-model teacher
# ensemble, then R2 PL with PowerTransform).
#
# Same fail-soft pattern as b03: write a zero-fallback submission.csv
# first, then attempt to overwrite via the library inference path.

# %% cell 1 — write zero-fallback submission.csv FIRST
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

# %% cell 2 — install timm + torch wheels offline (same pattern as b03)
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

# Multi-dataset kernels can mount under /kaggle/input/datasets/<owner>/<slug>/
# or /kaggle/input/<slug>/ depending on Kaggle's plumbing; discover the eB1b R2
# package via glob so we don't care which path it landed in.
_PKG_HITS = sorted(glob.glob(
    "/kaggle/input/**/birdclef2026-ebr2-pkg-ckpt/swa.pt",
    recursive=True,
))
assert _PKG_HITS, "could not find birdclef2026-ebr2-pkg-ckpt/swa.pt under /kaggle/input/"
PKG_INPUT = Path(_PKG_HITS[0]).parent
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
    print(f"PKG_INPUT : {PKG_INPUT}")
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

# %% cell 4 — run inference end-to-end via the library (fail-soft).
# CLI flags: --tta-offsets gives an explicit ±2.5 s spread (vs the default
# 0.5 s ladder produced by --n-tta-crops 5), and --chunk-smooth applies the
# 2024 3rd-place [0.1, 0.2, 0.4, 0.2, 0.1] temporal smoothing across each
# file's 12 windows. batch-size 32 is the upper-bound that fits in Kaggle
# CPU memory for NFNet-L0; 8x faster than b03's batch 4.
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
        "--batch-size", "32",
        "--amp-dtype", "fp32",
        "--tta-offsets", "-2.5", "-1.25", "0", "1.25", "2.5",
        "--chunk-smooth",
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
