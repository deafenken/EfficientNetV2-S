# %% cell 1
# Cell 0 — Install ONNX Runtime + TF 2.20
from pathlib import Path
import subprocess
import sys

INPUT_ROOT = Path("/kaggle/input")

# Keep this broad: the public ONNX dataset usually contains both perch_v2.onnx
# and onnxruntime-*.whl in a path with "perch" in its name.
PERCH_ONNX_HINT = "perch-onnx-for-birdclef-2026"
REQUIRE_ONNX_PERCH = True
ONNX_INTRA_OP_THREADS = 4
IO_PREFETCH_WORKERS = 4
FAST_TTA_MAX_BATCH_SIZE = 512

def _matches_hint(path, hint):
    return hint is None or hint.lower() in str(path).lower()

def _find_input_file(pattern, hint=None, required=False):
    candidates = []
    if INPUT_ROOT.exists():
        candidates = [p for p in INPUT_ROOT.rglob(pattern) if _matches_hint(p, hint)]
        if not candidates and hint is not None:
            candidates = list(INPUT_ROOT.rglob(pattern))
    candidates = sorted(set(candidates))
    if not candidates:
        if required:
            raise FileNotFoundError(f"Could not find {pattern!r} under /kaggle/input")
        return None
    return candidates[0]

onnxruntime_wheel = _find_input_file("onnxruntime-*.whl", PERCH_ONNX_HINT, required=REQUIRE_ONNX_PERCH)
if onnxruntime_wheel is not None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(onnxruntime_wheel)], check=True)
    print("Installed ONNX Runtime:", onnxruntime_wheel)

# Original TensorFlow wheels required by the reference notebook.
!pip install -q --no-deps /kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels/tensorboard-2.20.0-py3-none-any.whl
!pip install -q --no-deps /kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels/tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


# %% cell 2
# Cell 1 — Mode switch
MODE = "submit" 

assert MODE in {"train", "submit"}

print("MODE =", MODE)

# %% cell 3
# Cell 2 — Imports and run config
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gc
import json
import re
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    ort = None
    _ONNX_AVAILABLE = False

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
tf.experimental.numpy.experimental_enable_numpy_behavior()

_WALL_START = time.time()

BASE = Path("/kaggle/input/competitions/birdclef-2026")
MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12

DEVICE = torch.device("cpu")  # Competition constraint

LOGS = {}  # Comprehensive logging dict

CFG = {
    "mode": MODE,
    "verbose": MODE == "train",

    # expensive research blocks
    "run_oof_baseline": MODE == "train",
    "run_probe_check": False,
    "run_probe_grid": False,

    # inference
    "batch_files": 16,
    "proxy_reduce_grid": ["max", "mean"],
    "proxy_reduce": "max",
    "run_proxy_reduce_grid": False,
    "dryrun_n_files": 50 if MODE == "train" else 20,

    # cache behavior
    "require_full_cache_in_submit": False,
    "full_cache_input_dir": Path("/kaggle/input/perch-meta"),
    "full_cache_work_dir": Path("/kaggle/working/perch_cache"),

    # frozen baseline fusion params
    "best_fusion": {
        "lambda_event": 0.4,
        "lambda_texture": 1.0,
        "lambda_proxy_texture": 0.8,
        "smooth_texture": 0.35,
        "smooth_event": 0.15,
    },

    # V17: ProtoSSM v5 — LARGER model
    "proto_ssm": {
        "d_model": 256,               # V17: increased from 128→256
        "d_state": 16,
        "n_ssm_layers": 3,            # V17: increased from 2→3
        "dropout": 0.15,
        "n_prototypes": 1,
        "n_sites": 20,
        "meta_dim": 16,
        "use_cross_attn": True,
        "cross_attn_heads": 4,
    },

    # ProtoSSM v5 training
    "proto_ssm_train": {
        "n_epochs": 60 if MODE == "train" else 40,   # ← was always 60,
        "lr": 1e-3,
        "weight_decay": 2e-3,
        "val_ratio": 0.15,
        "patience": 15  if MODE == "train" else 8,    # ← was always 15
        "pos_weight_cap": 30.0,
        "distill_weight": 0.1,
        "proto_margin": 0.1,
        "label_smoothing": 0.02,
        "oof_n_splits": 3,
        "mixup_alpha": 0.3,
        "focal_gamma": 2.0,
        "swa_start_frac": 0.7,
        "swa_lr": 5e-4,
    },

    # frozen probe params
    "frozen_best_probe": {
        "pca_dim": 64,
        "min_pos": 8,
        "C": 0.50,
        "alpha": 0.40,
    },

    # Residual SSM
    "residual_ssm": {
        "d_model": 64,
        "d_state": 8,
        "n_ssm_layers": 1,
        "dropout": 0.1,
        "correction_weight": 0.3,
        "n_epochs": 30,
        "lr": 1e-3,
        "patience": 8,
    },

    # Per-taxon temperature
    "temperature": {
        "aves": 1.10,
        "texture": 0.95,
    },

    # V17: Post-processing parameters
    "file_level_top_k": 2,
    "tta_shifts": [0, 1, -1],
    
    # V17 NEW: Rank-aware post-processing
    "rank_aware_scale": True,
    "rank_aware_power": 0.5,  # Power transform on file max
    
    # V17 NEW: Delta shift smoothing
    "delta_shift_alpha": 0.15,
    
    # V17 NEW: Per-class thresholds (grid search range)
    "threshold_grid": [0.3, 0.4, 0.5, 0.6, 0.7],

    "probe_backend": "mlp",
    "mlp_params": {
        "hidden_layer_sizes": (128,),
        "activation": "relu",
        "max_iter": 300,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 15,
        "random_state": 42,
        "learning_rate_init": 0.001,
        "alpha": 0.01,
    },
}

def resolve_full_cache_input_dir(default_dir):
    default_dir = Path(default_dir)
    required = ("full_perch_meta.parquet", "full_perch_arrays.npz")
    if all((default_dir / name).exists() for name in required):
        return default_dir
    if INPUT_ROOT.exists():
        for meta_path in sorted(INPUT_ROOT.rglob("full_perch_meta.parquet")):
            candidate = meta_path.parent
            if all((candidate / name).exists() for name in required):
                return candidate
    return default_dir

CFG["full_cache_input_dir"] = resolve_full_cache_input_dir(CFG["full_cache_input_dir"])
CFG["full_cache_work_dir"].mkdir(parents=True, exist_ok=True)
print("Resolved full_cache_input_dir:", CFG["full_cache_input_dir"])

print("TensorFlow:", tf.__version__)
print("PyTorch:", torch.__version__)
print("onnxruntime available:", _ONNX_AVAILABLE)
print("Competition dir exists:", BASE.exists())
print("Model dir exists:", MODEL_DIR.exists())
print("V17 CFG: d_model=256, n_ssm_layers=3")
print(json.dumps(
    {k: (str(v) if isinstance(v, Path) else v) for k, v in CFG.items()},
    indent=2
))


# %% cell 4
# ── V18 CFG UPGRADES ──────────────────────
CFG["proto_ssm"] = {
    "d_model": 320, "d_state": 32, "n_ssm_layers": 4,
    "dropout": 0.12, "n_prototypes": 2, "n_sites": 20,
    "meta_dim": 24, "use_cross_attn": True, "cross_attn_heads": 8,
}
CFG["proto_ssm_train"] = {
    "n_epochs": 80, "lr": 8e-4, "weight_decay": 1e-3,
    "val_ratio": 0.15, "patience": 20, "pos_weight_cap": 25.0,
    "distill_weight": 0.15, "proto_margin": 0.15,
    "label_smoothing": 0.03, "oof_n_splits": 5,
    "mixup_alpha": 0.4, "focal_gamma": 2.5,
    "swa_start_frac": 0.65, "swa_lr": 4e-4,
    "use_cosine_restart": True, "restart_period": 20,
}
CFG["residual_ssm"] = {
    "d_model": 128, "d_state": 16, "n_ssm_layers": 2,
    "dropout": 0.1, "correction_weight": 0.35,
    "n_epochs": 40, "lr": 8e-4, "patience": 12,
}
CFG["best_fusion"]["lambda_event"]         = 0.45
CFG["best_fusion"]["lambda_texture"]       = 1.1
CFG["best_fusion"]["lambda_proxy_texture"] = 0.9
CFG["threshold_grid"] = [0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]
CFG["tta_shifts"]        = [0, 1, -1, 2, -2]
CFG["rank_aware_power"]  = 0.4
CFG["delta_shift_alpha"] = 0.20
CFG["mlp_params"] = {
    "hidden_layer_sizes": (256, 128), "activation": "relu",
    "max_iter": 500, "early_stopping": True,
    "validation_fraction": 0.15, "n_iter_no_change": 20,
    "random_state": 42, "learning_rate_init": 5e-4, "alpha": 0.005,
}
CFG["frozen_best_probe"] = {
    "pca_dim": 128, "min_pos": 5, "C": 0.75, "alpha": 0.45
}
print("✅ V18 CFG loaded")

# %% cell 5

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def get_cosine_restart_scheduler(optimizer, restart_period=20):
    return CosineAnnealingWarmRestarts(
        optimizer, T_0=restart_period, T_mult=1, eta_min=1e-5
    )

print("✅ Cosine Restart Scheduler defined")

# %% cell 6
# ── STEP 3: Mixup + CutMix Hybrid ─
def mixup_cutmix(emb, logits, labels, alpha=0.4, cutmix_prob=0.3):
    B, T, D = emb.shape
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(B)

    if np.random.rand() < cutmix_prob:
        # CutMix on time dimension
        cut_len = max(1, int(T * (1 - lam)))
        cut_start = np.random.randint(0, T - cut_len + 1)
        new_emb = emb.clone()
        new_emb[:, cut_start:cut_start+cut_len, :] = emb[idx, cut_start:cut_start+cut_len, :]
        new_logits = logits.clone()
        new_logits[:, cut_start:cut_start+cut_len, :] = logits[idx, cut_start:cut_start+cut_len, :]
        lam_actual = 1.0 - cut_len / T
        new_labels = lam_actual * labels + (1-lam_actual) * labels[idx]
    else:
        # Standard Mixup
        new_emb    = lam * emb    + (1-lam) * emb[idx]
        new_logits = lam * logits + (1-lam) * logits[idx]
        new_labels = lam * labels + (1-lam) * labels[idx]

    return new_emb, new_logits, new_labels

print("✅ Mixup+CutMix defined")

# %% cell 7
# ── STEP 4: Species-Frequency Aware Focal Loss ──
def build_class_freq_weights(Y_FULL, cap=10.0):
    pos_count = Y_FULL.sum(axis=0).astype(np.float32) + 1.0
    total     = Y_FULL.shape[0]
    freq      = pos_count / total
    weights   = 1.0 / (freq ** 0.5)
    weights   = np.clip(weights, 1.0, cap)
    weights   = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

def species_focal_loss(logits, targets, class_weights, 
                       gamma=2.5, label_smoothing=0.03):
    targets_smooth = targets * (1 - label_smoothing) + label_smoothing / 2.0
    bce    = F.binary_cross_entropy_with_logits(
                 logits, targets_smooth, reduction="none")
    pt     = torch.exp(-bce)
    focal  = ((1 - pt) ** gamma) * bce
    w      = class_weights.to(logits.device).unsqueeze(0)
    return (focal * w).mean()

print("✅ Species Focal Loss defined")

# %% cell 8
taxonomy = pd.read_csv(BASE / "taxonomy.csv")
sample_sub = pd.read_csv(BASE / "sample_submission.csv")
soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES = len(PRIMARY_LABELS)

taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
soundscape_labels["primary_label"] = soundscape_labels["primary_label"].astype(str)

def parse_soundscape_labels(x):
    if pd.isna(x):
        return []
    return [t.strip() for t in str(x).split(";") if t.strip()]

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

def parse_soundscape_filename(name):
    m = FNAME_RE.match(name)
    if not m:
        return {
            "file_id": None,
            "site": None,
            "date": pd.NaT,
            "time_utc": None,
            "hour_utc": -1,
            "month": -1,
        }
    file_id, site, ymd, hms = m.groups()
    dt = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
    return {
        "file_id": file_id,
        "site": site,
        "date": dt,
        "time_utc": hms,
        "hour_utc": int(hms[:2]),
        "month": int(dt.month) if pd.notna(dt) else -1,
    }

def union_labels(series):
    return sorted(set(lbl for x in series for lbl in parse_soundscape_labels(x)))

# Deduplicate duplicated rows and aggregate labels per 5s window
sc_clean = (
    soundscape_labels
    .groupby(["filename", "start", "end"])["primary_label"]
    .apply(union_labels)
    .reset_index(name="label_list")
)

sc_clean["start_sec"] = pd.to_timedelta(sc_clean["start"]).dt.total_seconds().astype(int)
sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
sc_clean["row_id"] = sc_clean["filename"].str.replace(".ogg", "", regex=False) + "_" + sc_clean["end_sec"].astype(str)

meta = sc_clean["filename"].apply(parse_soundscape_filename).apply(pd.Series)
sc_clean = pd.concat([sc_clean, meta], axis=1)

# Fully-labeled files
windows_per_file = sc_clean.groupby("filename").size()
full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
sc_clean["file_fully_labeled"] = sc_clean["filename"].isin(full_files)

# Multi-hot label matrix aligned with sc_clean row order
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}
Y_SC = np.zeros((len(sc_clean), N_CLASSES), dtype=np.uint8)

for i, labels in enumerate(sc_clean["label_list"]):
    idxs = [label_to_idx[lbl] for lbl in labels if lbl in label_to_idx]
    if idxs:
        Y_SC[i, idxs] = 1

full_truth = (
    sc_clean[sc_clean["file_fully_labeled"]]
    .sort_values(["filename", "end_sec"])
    .reset_index(drop=False)
)

Y_FULL_TRUTH = Y_SC[full_truth["index"].to_numpy()]

print("sc_clean:", sc_clean.shape)
print("Y_SC:", Y_SC.shape, Y_SC.dtype)
print("Full files:", len(full_files))
print("Trusted full windows:", len(full_truth))
print("Active classes in full windows:", int((Y_FULL_TRUTH.sum(axis=0) > 0).sum()))

# %% cell 9
CLASS_WEIGHTS = build_class_freq_weights(Y_FULL_TRUTH)
print("✅ Class weights built")

# %% cell 10
# ── STEP 5: Isotonic Calibration + Threshold Optimization ──
from sklearn.isotonic import IsotonicRegression

def calibrate_and_optimize_thresholds(oof_probs, Y_FULL, 
                                       threshold_grid, n_windows=12):
    n_samples, n_cls = oof_probs.shape
    thresholds = np.full(n_cls, 0.5, dtype=np.float32)
    n_files  = n_samples // n_windows
    file_oof = oof_probs.reshape(n_files, n_windows, n_cls).max(axis=1)
    file_y   = Y_FULL.reshape(n_files, n_windows, n_cls).max(axis=1)

    for c in range(n_cls):
        y_true, y_prob = file_y[:, c], file_oof[:, c]
        if y_true.sum() < 3:
            continue
        try:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(y_prob, y_true)
            y_cal = ir.transform(y_prob)
        except:
            y_cal = y_prob

        best_f1, best_t = 0.0, 0.5
        for t in threshold_grid:
            pred = (y_cal >= t).astype(int)
            tp = ((pred==1)&(y_true==1)).sum()
            fp = ((pred==1)&(y_true==0)).sum()
            fn = ((pred==0)&(y_true==1)).sum()
            prec = tp/(tp+fp+1e-8)
            rec  = tp/(tp+fn+1e-8)
            f1   = 2*prec*rec/(prec+rec+1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t

    print(f"Mean threshold: {thresholds.mean():.3f}")
    print(f"Range: [{thresholds.min():.2f}, {thresholds.max():.2f}]")
    return thresholds

print("✅ Calibration + Threshold function defined")

# %% cell 11
# ── STEP 6: Ensemble Weight Sweep ──
def sweep_ensemble_weight(oof_proto, oof_mlp, Y_FULL, 
                          n_windows=12,
                          candidates=np.arange(0.3, 0.8, 0.05)):
    n_files = oof_proto.shape[0] // n_windows
    file_y  = Y_FULL.reshape(n_files, n_windows, -1).max(axis=1)
    best_auc, best_w = 0.0, 0.6

    for w in candidates:
        blended   = w * oof_proto + (1-w) * oof_mlp
        file_pred = blended.reshape(n_files, n_windows, -1).max(axis=1)
        try:
            auc = macro_auc_skip_empty(file_y, file_pred)
        except:
            continue
        if auc > best_auc:
            best_auc, best_w = auc, w

    print(f"Best ensemble weight (proto): {best_w:.2f}")
    print(f"Best AUC: {best_auc:.5f}")
    return best_w

print("✅ Ensemble Weight Sweep defined")

# %% cell 12
# Cell 3 — Load Perch, mapping, and selective frog proxies
BEST = CFG["best_fusion"]

def resolve_perch_onnx_path():
    candidates = []
    if INPUT_ROOT.exists():
        for p in INPUT_ROOT.rglob("*.onnx"):
            if "perch" not in str(p).lower():
                continue
            if _matches_hint(p, PERCH_ONNX_HINT):
                candidates.append(p)
        if not candidates and PERCH_ONNX_HINT is not None:
            candidates = [p for p in INPUT_ROOT.rglob("*.onnx") if "perch" in str(p).lower()]
    return sorted(set(candidates))[0] if candidates else None

PERCH_ONNX_PATH = resolve_perch_onnx_path()
USE_ONNX_PERCH = bool(_ONNX_AVAILABLE and PERCH_ONNX_PATH is not None and PERCH_ONNX_PATH.exists())
if REQUIRE_ONNX_PERCH and not USE_ONNX_PERCH:
    raise RuntimeError("ONNX Perch is required but `perch_v2.onnx` or `onnxruntime` was not resolved.")

if USE_ONNX_PERCH:
    print("Using ONNX Perch for test inference:", PERCH_ONNX_PATH)
    _so = ort.SessionOptions()
    _so.intra_op_num_threads = int(ONNX_INTRA_OP_THREADS)
    ONNX_SESSION = ort.InferenceSession(str(PERCH_ONNX_PATH), sess_options=_so, providers=["CPUExecutionProvider"])
    ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
    ONNX_INPUT_SHAPE = list(ONNX_SESSION.get_inputs()[0].shape)
    ONNX_INPUT_RANK = len(ONNX_INPUT_SHAPE)
    ONNX_OUTPUT_MAP = {o.name: i for i, o in enumerate(ONNX_SESSION.get_outputs())}
    print("ONNX input:", ONNX_INPUT_NAME, ONNX_INPUT_SHAPE)
    print("ONNX outputs:", ONNX_OUTPUT_MAP)
else:
    ONNX_SESSION = None
    ONNX_INPUT_NAME = None
    ONNX_INPUT_SHAPE = None
    ONNX_INPUT_RANK = None
    ONNX_OUTPUT_MAP = {}

# TensorFlow SavedModel is heavy; do not load it when ONNX is active.
infer_fn = None
if not USE_ONNX_PERCH:
    birdclassifier = tf.saved_model.load(str(MODEL_DIR))
    infer_fn = birdclassifier.signatures["serving_default"]

bc_labels = (
    pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
    .reset_index()
    .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
)

NO_LABEL_INDEX = len(bc_labels)

MANUAL_SCIENTIFIC_NAME_MAP = {
    # Optional future synonym fixes (add manual name corrections here)
}

taxonomy = taxonomy.copy()
taxonomy["scientific_name_lookup"] = taxonomy["scientific_name"].replace(MANUAL_SCIENTIFIC_NAME_MAP)

bc_lookup = bc_labels.rename(columns={"scientific_name": "scientific_name_lookup"})

mapping = taxonomy.merge(
    bc_lookup[["scientific_name_lookup", "bc_index"]],
    on="scientific_name_lookup",
    how="left"
)

mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL_INDEX).astype(int)

label_to_bc_index = mapping.set_index("primary_label")["bc_index"]
BC_INDICES = np.array([int(label_to_bc_index.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)

MAPPED_MASK = BC_INDICES != NO_LABEL_INDEX
MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
UNMAPPED_POS = np.where(~MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_INDICES = BC_INDICES[MAPPED_MASK].astype(np.int32)

CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA = {"Amphibia", "Insecta"}

ACTIVE_CLASSES = [PRIMARY_LABELS[i] for i in np.where(Y_SC.sum(axis=0) > 0)[0]]

idx_active_texture = np.array(
    [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) in TEXTURE_TAXA],
    dtype=np.int32
)
idx_active_event = np.array(
    [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) not in TEXTURE_TAXA],
    dtype=np.int32
)

idx_mapped_active_texture = idx_active_texture[MAPPED_MASK[idx_active_texture]]
idx_mapped_active_event = idx_active_event[MAPPED_MASK[idx_active_event]]

idx_unmapped_active_texture = idx_active_texture[~MAPPED_MASK[idx_active_texture]]
idx_unmapped_active_event = idx_active_event[~MAPPED_MASK[idx_active_event]]

idx_unmapped_inactive = np.array(
    [i for i in UNMAPPED_POS if PRIMARY_LABELS[i] not in ACTIVE_CLASSES],
    dtype=np.int32
)

# Build automatic genus proxies for unmapped non-sonotypes
unmapped_df = mapping[mapping["bc_index"] == NO_LABEL_INDEX].copy()
unmapped_non_sonotype = unmapped_df[
    ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
].copy()

def get_genus_hits(scientific_name):
    genus = str(scientific_name).split()[0]
    hits = bc_labels[
        bc_labels["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)
    ].copy()
    return genus, hits

proxy_map = {}
for _, row in unmapped_non_sonotype.iterrows():
    target = row["primary_label"]
    sci = row["scientific_name"]
    genus, hits = get_genus_hits(sci)
    if len(hits) > 0:
        proxy_map[target] = {
            "target_scientific_name": sci,
            "genus": genus,
            "bc_indices": hits["bc_index"].astype(int).tolist(),
            "proxy_scientific_names": hits["scientific_name"].tolist(),
        }

# Enable genus proxies for Amphibia, Insecta, and Aves (unmapped species)
PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
SELECTED_PROXY_TARGETS = sorted([
    t for t in proxy_map.keys()
    if CLASS_NAME_MAP.get(t) in PROXY_TAXA
])
print(f"Proxy targets by class: { {cls: sum(1 for t in SELECTED_PROXY_TARGETS if CLASS_NAME_MAP.get(t)==cls) for cls in PROXY_TAXA} }")

selected_proxy_pos = np.array([label_to_idx[c] for c in SELECTED_PROXY_TARGETS], dtype=np.int32)

selected_proxy_pos_to_bc = {
    label_to_idx[target]: np.array(proxy_map[target]["bc_indices"], dtype=np.int32)
    for target in SELECTED_PROXY_TARGETS
}

idx_selected_proxy_active_texture = np.intersect1d(selected_proxy_pos, idx_active_texture)
idx_selected_prioronly_active_texture = np.setdiff1d(idx_unmapped_active_texture, selected_proxy_pos)
idx_selected_prioronly_active_event = np.setdiff1d(idx_unmapped_active_event, selected_proxy_pos)

print(f"Mapped classes: {MAPPED_MASK.sum()} / {N_CLASSES}")
print(f"Unmapped classes: {(~MAPPED_MASK).sum()}")
print("Selected frog proxy targets:", SELECTED_PROXY_TARGETS)
print("Active texture classes:", len(idx_active_texture))
print("Selected proxy active texture:", len(idx_selected_proxy_active_texture))
print("Prior-only active texture:", len(idx_selected_prioronly_active_texture))
print("Prior-only active event:", len(idx_selected_prioronly_active_event))


# %% cell 13
# Cell 4 — Metrics and helper utilities
def macro_auc_skip_empty(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")

def smooth_cols_fixed12(scores, cols, alpha=0.35):
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()

    s = scores.copy()
    assert len(s) % N_WINDOWS == 0, "Expected full-file blocks of 12 windows"
    view = s.reshape(-1, N_WINDOWS, s.shape[1])

    x = view[:, :, cols]
    prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)

    view[:, :, cols] = (1.0 - alpha) * x + 0.5 * alpha * (prev_x + next_x)
    return s

def smooth_events_fixed12(scores, cols, alpha=0.15):
    """Soft max-pool context for event birds (Aves).
    Uses local_max instead of average neighbor, preserving transient call detection."""
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()
    s = scores.copy()
    assert len(s) % N_WINDOWS == 0
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    local_max = np.maximum(x, np.maximum(prev_x, next_x))
    view[:, :, cols] = (1.0 - alpha) * x + alpha * local_max
    return s

def seq_features_1d(v):
    """
    v: shape (n_rows,), ordered as full-file blocks of 12 windows
    Extended: tambah std_v untuk capture variance temporal dalam file
    """
    assert len(v) % N_WINDOWS == 0, "Expected full-file blocks of 12 windows"
    x = v.reshape(-1, N_WINDOWS)

    prev_v = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
    next_v = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
    mean_v = np.repeat(x.mean(axis=1), N_WINDOWS)
    max_v  = np.repeat(x.max(axis=1),  N_WINDOWS)
    std_v  = np.repeat(x.std(axis=1),  N_WINDOWS)

    return prev_v, next_v, mean_v, max_v, std_v

# %% cell 14
# V16/V17 NEW: Focal loss, file-level scaling, TTA, rank-aware, delta shift, per-class thresholds

def focal_bce_with_logits(logits, targets, gamma=2.0, pos_weight=None, reduction="mean"):
    """Focal loss for multi-label classification.
    Reduces contribution of easy examples, focuses on hard ones."""
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    
    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    focal_weight = (1 - pt) ** gamma
    loss = focal_weight * bce
    
    if reduction == "mean":
        return loss.mean()
    return loss


def file_level_confidence_scale(preds, n_windows=12, top_k=2):
    """Rank 1/2 technique: scale each window's predictions by the file's top-K mean confidence."""
    N, C = preds.shape
    assert N % n_windows == 0
    view = preds.reshape(-1, n_windows, C)
    sorted_view = np.sort(view, axis=1)
    top_k_mean = sorted_view[:, -top_k:, :].mean(axis=1, keepdims=True)
    scaled = view * top_k_mean
    return scaled.reshape(N, C)


def temporal_shift_tta(emb_files, logits_files, model, site_ids, hours, shifts=[0, 1, -1], max_batch_size=None):
    """TTA by circular-shifting the 12-window sequence, batched for speed."""
    n_files = emb_files.shape[0]
    n_shifts = len(shifts)
    if n_shifts == 0:
        return np.zeros((n_files, emb_files.shape[1], logits_files.shape[2]), dtype=np.float32)

    e_list, l_list = [], []
    for shift in shifts:
        if shift == 0:
            e_list.append(emb_files)
            l_list.append(logits_files)
        else:
            e_list.append(np.roll(emb_files, shift, axis=1))
            l_list.append(np.roll(logits_files, shift, axis=1))

    e_batch = np.concatenate(e_list, axis=0)
    l_batch = np.concatenate(l_list, axis=0)
    site_batch = np.tile(site_ids, n_shifts)
    hour_batch = np.tile(hours, n_shifts)

    if max_batch_size is None:
        max_batch_size = int(FAST_TTA_MAX_BATCH_SIZE)

    model.eval()
    pred_chunks = []
    with torch.no_grad():
        total = e_batch.shape[0]
        for start_idx in range(0, total, max_batch_size):
            end_idx = min(start_idx + max_batch_size, total)
            out, _, _ = model(
                torch.tensor(e_batch[start_idx:end_idx], dtype=torch.float32),
                torch.tensor(l_batch[start_idx:end_idx], dtype=torch.float32),
                site_ids=torch.tensor(site_batch[start_idx:end_idx], dtype=torch.long),
                hours=torch.tensor(hour_batch[start_idx:end_idx], dtype=torch.long),
            )
            pred_chunks.append(out.numpy())

    pred_batch = np.concatenate(pred_chunks, axis=0)
    pred_batch = pred_batch.reshape(n_shifts, n_files, pred_batch.shape[1], pred_batch.shape[2])

    all_preds = []
    for i, shift in enumerate(shifts):
        pred_i = pred_batch[i]
        if shift != 0:
            pred_i = np.roll(pred_i, -shift, axis=1)
        all_preds.append(pred_i)
    return np.mean(all_preds, axis=0).astype(np.float32, copy=False)

# V17: Post-processing utilities

def rank_aware_scaling(scores, n_windows=12, power=0.5):
    """V17: 2025 Rank 3 technique. Scale each window by (file_max)^power.
    Suppresses predictions in uncertain files, boosts confident files."""
    N, C = scores.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    
    view = scores.reshape(n_files, n_windows, C)
    file_max = view.max(axis=1, keepdims=True)  # (F, 1, C)
    
    # Apply power transform to file max
    scale = np.power(file_max, power)
    
    # Scale each window
    scaled = view * scale
    return scaled.reshape(N, C)


def delta_shift_smooth(scores, n_windows=12, alpha=0.15):
    """V17: 2025 Rank 1 technique. Temporal smoothing across windows.
    new[t] = (1-alpha)*old[t] + 0.5*alpha*(old[t-1] + old[t+1])"""
    N, C = scores.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    
    view = scores.reshape(n_files, n_windows, C)
    
    # Create shifted versions
    prev_view = np.concatenate([view[:, :1, :], view[:, :-1, :]], axis=1)
    next_view = np.concatenate([view[:, 1:, :], view[:, -1:, :]], axis=1)
    
    # Delta shift smoothing
    smoothed = (1 - alpha) * view + 0.5 * alpha * (prev_view + next_view)
    
    return smoothed.reshape(N, C)


def optimize_per_class_thresholds(oof_scores, y_true, n_windows=12, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """V17: Find optimal decision threshold per class from OOF predictions.
    Optimizes F1-like metric (precision-recall balance) for each species."""
    n_classes = oof_scores.shape[1]
    best_thresholds = np.zeros(n_classes)
    best_scores = np.zeros(n_classes)
    
    for c in range(n_classes):
        y_c = y_true[:, c]
        scores_c = oof_scores[:, c]
        
        # Skip classes with no positive samples
        if y_c.sum() == 0:
            best_thresholds[c] = 0.5
            continue
            
        # Find best threshold
        best_f1 = 0
        best_t = 0.5
        
        for t in thresholds:
            pred_c = (scores_c > t).astype(int)
            tp = ((pred_c == 1) & (y_c == 1)).sum()
            fp = ((pred_c == 1) & (y_c == 0)).sum()
            fn = ((pred_c == 0) & (y_c == 1)).sum()
            
            if tp + fp == 0 or tp + fn == 0:
                continue
                
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        
        best_thresholds[c] = best_t
        best_scores[c] = best_f1
    
    return best_thresholds, best_scores


def apply_per_class_thresholds(scores, thresholds, n_windows=12):
    """V17: Apply per-class thresholds to convert scores to binary predictions."""
    N, C = scores.shape
    assert C == len(thresholds)
    
    # For competition, we submit probabilities but threshold for metrics
    # Apply threshold as a scaling factor that sharpens confident predictions
    scaled = np.copy(scores)
    
    for c in range(C):
        t = thresholds[c]
        # Sharpen: push above-threshold scores higher, below-threshold lower
        mask_above = scores[:, c] > t
        scaled[mask_above, c] = 0.5 + 0.5 * (scores[mask_above, c] - t) / (1 - t + 1e-8)
        scaled[~mask_above, c] = 0.5 * scores[~mask_above, c] / (t + 1e-8)
    
    return np.clip(scaled, 0, 1)


print("V17 utilities defined: focal_bce_with_logits, file_level_confidence_scale, temporal_shift_tta,")
print("  rank_aware_scaling, delta_shift_smooth, optimize_per_class_thresholds, apply_per_class_thresholds")


# %% cell 15
# Cell 5 — Perch inference with embeddings + selective proxies
import concurrent.futures

def read_soundscape_60s(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        raise ValueError(f"Unexpected sample rate {sr} in {path}; expected {SR}")
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    elif len(y) > FILE_SAMPLES:
        y = y[:FILE_SAMPLES]
    return y

def _run_perch_batch(x):
    if USE_ONNX_PERCH:
        onnx_outs = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: _format_onnx_input(x)})
        logits = onnx_outs[ONNX_OUTPUT_MAP["label"]].astype(np.float32, copy=False)
        emb = onnx_outs[ONNX_OUTPUT_MAP["embedding"]].astype(np.float32, copy=False)
        return logits, emb
    outputs = infer_fn(inputs=tf.convert_to_tensor(x))
    logits = outputs["label"].numpy().astype(np.float32, copy=False)
    emb = outputs["embedding"].numpy().astype(np.float32, copy=False)
    return logits, emb

def _dim_matches(dim, value):
    try:
        return int(dim) == int(value)
    except Exception:
        return False

def _format_onnx_input(x):
    # Adapt waveform batches to ONNX exports that require rank-2/3/4 input.
    x = x.astype(np.float32, copy=False)
    if ONNX_INPUT_RANK == 2:
        return x
    if ONNX_INPUT_RANK == 3:
        shape = list(ONNX_INPUT_SHAPE or [])
        tail = shape[1:] if len(shape) == 3 else []
        if len(tail) == 2 and _dim_matches(tail[0], WINDOW_SAMPLES):
            return x[:, :, None]
        if len(tail) == 2 and _dim_matches(tail[1], WINDOW_SAMPLES):
            return x[:, None, :]
        return x[:, :, None]
    if ONNX_INPUT_RANK == 4:
        shape = list(ONNX_INPUT_SHAPE or [])
        tail = shape[1:] if len(shape) == 4 else []
        if len(tail) == 3 and _dim_matches(tail[0], WINDOW_SAMPLES):
            return x[:, :, None, None]
        if len(tail) == 3 and _dim_matches(tail[1], WINDOW_SAMPLES):
            return x[:, None, :, None]
        if len(tail) == 3 and _dim_matches(tail[2], WINDOW_SAMPLES):
            return x[:, None, None, :]
        # The public Perch ONNX often reports a dynamic 4D input but starts by
        # reshaping back to waveform. Preserve row-major waveform order.
        return x[:, :, None, None]
    raise ValueError(f"Unsupported ONNX input rank: {ONNX_INPUT_RANK}, shape={ONNX_INPUT_SHAPE}")

def infer_perch_with_embeddings(paths, batch_files=16, verbose=True, proxy_reduce="max"):
    paths = [Path(p) for p in paths]
    n_files = len(paths)
    n_rows = n_files * N_WINDOWS

    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)

    scores = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embeddings = np.zeros((n_rows, 1536), dtype=np.float32)

    write_row = 0
    iterator = range(0, n_files, batch_files)
    if verbose:
        iterator = tqdm(iterator, total=(n_files + batch_files - 1) // batch_files, desc="Perch batches")

    workers = max(1, int(IO_PREFETCH_WORKERS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as io_executor:
        next_paths = paths[:batch_files]
        future_audio = [io_executor.submit(read_soundscape_60s, p) for p in next_paths]

        for start in iterator:
            batch_paths = next_paths
            batch_n = len(batch_paths)
            batch_audio = [f.result() for f in future_audio]

            next_start = start + batch_files
            if next_start < n_files:
                next_paths = paths[next_start:next_start + batch_files]
                future_audio = [io_executor.submit(read_soundscape_60s, p) for p in next_paths]

            x = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            batch_row_start = write_row
            x_pos = 0

            for i, path in enumerate(batch_paths):
                y = batch_audio[i]
                x[x_pos:x_pos + N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)

                meta = parse_soundscape_filename(path.name)
                stem = path.stem

                row_ids[write_row:write_row + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
                filenames[write_row:write_row + N_WINDOWS] = path.name
                sites[write_row:write_row + N_WINDOWS] = meta["site"]
                hours[write_row:write_row + N_WINDOWS] = int(meta["hour_utc"])

                x_pos += N_WINDOWS
                write_row += N_WINDOWS

            logits, emb = _run_perch_batch(x)

            scores[batch_row_start:write_row, MAPPED_POS] = logits[:, MAPPED_BC_INDICES]
            embeddings[batch_row_start:write_row] = emb

            for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
                sub = logits[:, bc_idx_arr]
                if proxy_reduce == "max":
                    proxy_score = sub.max(axis=1)
                elif proxy_reduce == "mean":
                    proxy_score = sub.mean(axis=1)
                else:
                    raise ValueError("proxy_reduce must be 'max' or 'mean'")
                scores[batch_row_start:write_row, pos] = proxy_score.astype(np.float32)

            del x, logits, emb, batch_audio
            gc.collect()

    meta_df = pd.DataFrame({
        "row_id": row_ids,
        "filename": filenames,
        "site": sites,
        "hour_utc": hours,
    })

    return meta_df, scores, embeddings


# %% cell 16
# Cell 6 — Load or compute full-file Perch cache
def resolve_full_cache_paths():
    candidates = []

    # Working dir cache
    candidates.append((
        CFG["full_cache_work_dir"] / "full_perch_meta.parquet",
        CFG["full_cache_work_dir"] / "full_perch_arrays.npz"
    ))

    # Legacy working paths
    candidates.append((
        Path("/kaggle/working/full_perch_meta.parquet"),
        Path("/kaggle/working/full_perch_arrays.npz")
    ))

    # Attached input dataset
    if CFG["full_cache_input_dir"].exists():
        candidates.append((
            CFG["full_cache_input_dir"] / "full_perch_meta.parquet",
            CFG["full_cache_input_dir"] / "full_perch_arrays.npz"
        ))

    for meta_path, npz_path in candidates:
        if meta_path.exists() and npz_path.exists():
            return meta_path, npz_path

    return None, None

cache_meta, cache_npz = resolve_full_cache_paths()

if cache_meta is not None and cache_npz is not None:
    print("Loading cached full-file Perch outputs from:")
    print("  ", cache_meta)
    print("  ", cache_npz)

    meta_full = pd.read_parquet(cache_meta)
    arr = np.load(cache_npz)
    scores_full_raw = arr["scores_full_raw"].astype(np.float32)
    emb_full = arr["emb_full"].astype(np.float32)

else:
    if CFG["mode"] == "submit" and CFG["require_full_cache_in_submit"]:
        raise FileNotFoundError(
            "Submit mode requires cached full-file Perch outputs. "
            "Attach the cache dataset or place full_perch_meta.parquet/full_perch_arrays.npz in working dir."
        )

    print("No cache found. Running Perch on trusted full files...")
    full_paths = [BASE / "train_soundscapes" / fn for fn in full_files]

    # Use CFG["proxy_reduce"] for consistency with grid search
    meta_full, scores_full_raw, emb_full = infer_perch_with_embeddings(
        full_paths,
        batch_files=CFG["batch_files"],
        verbose=CFG["verbose"],
        proxy_reduce=CFG["proxy_reduce"],
    )

    out_meta = CFG["full_cache_work_dir"] / "full_perch_meta.parquet"
    out_npz = CFG["full_cache_work_dir"] / "full_perch_arrays.npz"

    meta_full.to_parquet(out_meta, index=False)
    np.savez_compressed(
        out_npz,
        scores_full_raw=scores_full_raw,
        emb_full=emb_full,
    )

    print("Saved cache to:")
    print("  ", out_meta)
    print("  ", out_npz)

# Align truth to cached order
full_truth_aligned = full_truth.set_index("row_id").loc[meta_full["row_id"]].reset_index()
Y_FULL = Y_SC[full_truth_aligned["index"].to_numpy()]

assert np.all(full_truth_aligned["filename"].values == meta_full["filename"].values)
assert np.all(full_truth_aligned["row_id"].values == meta_full["row_id"].values)

print("meta_full:", meta_full.shape)
print("scores_full_raw:", scores_full_raw.shape, scores_full_raw.dtype)
print("emb_full:", emb_full.shape, emb_full.dtype)
print("Y_FULL:", Y_FULL.shape, Y_FULL.dtype)

# [MODIFIED - Opsi 3] Grid search proxy_reduce: evaluasi "max" vs "mean" via OOF AUC
# Dilakukan hanya saat train mode; hasilnya di-freeze ke CFG["proxy_reduce"] untuk submit
PROXY_REDUCE_CACHE = CFG["full_cache_work_dir"] / "proxy_reduce_grid.json"

if CFG.get("run_proxy_reduce_grid", False):
    print("\n[Opsi 3] Running proxy_reduce grid search: max vs mean...")
    proxy_reduce_results = {}

    for pr in CFG["proxy_reduce_grid"]:
        full_paths = [BASE / "train_soundscapes" / fn for fn in full_files]
        _meta, _scores, _emb = infer_perch_with_embeddings(
            full_paths,
            batch_files=CFG["batch_files"],
            verbose=False,
            proxy_reduce=pr,
        )

        # OOF baseline AUC untuk proxy_reduce ini (tanpa probe)
        _oof_b, _oof_p, _ = build_oof_base_prior(
            scores_full_raw=_scores,
            meta_full=_meta,
            sc_clean=sc_clean,
            Y_SC=Y_SC,
            n_splits=5,
            verbose=False,
        )
        auc = macro_auc_skip_empty(Y_FULL, _oof_b)
        proxy_reduce_results[pr] = float(auc)
        print(f"  proxy_reduce={pr!r:6s} → OOF baseline AUC = {auc:.6f}")

    best_pr = max(proxy_reduce_results, key=proxy_reduce_results.get)
    CFG["proxy_reduce"] = best_pr
    print(f"\n  Best proxy_reduce = {best_pr!r} (AUC={proxy_reduce_results[best_pr]:.6f})")

    PROXY_REDUCE_CACHE.write_text(json.dumps({
        "results": proxy_reduce_results,
        "best_proxy_reduce": best_pr,
    }, indent=2))
    print("  Saved to:", PROXY_REDUCE_CACHE)

elif PROXY_REDUCE_CACHE.exists():
    _pr_data = json.loads(PROXY_REDUCE_CACHE.read_text())
    CFG["proxy_reduce"] = _pr_data["best_proxy_reduce"]
    print(f"[Opsi 3] Loaded proxy_reduce from cache: {CFG['proxy_reduce']!r}")
    print("  Grid results:", _pr_data["results"])

else:
    print(f"[Opsi 3] Using default proxy_reduce={CFG['proxy_reduce']!r} (submit mode or no cache)")

# %% cell 17
# Cell 7 — Fold-safe metadata prior tables
def fit_prior_tables(prior_df, Y_prior):
    prior_df = prior_df.reset_index(drop=True)

    global_p = Y_prior.mean(axis=0).astype(np.float32)

    # Site
    site_keys = sorted(prior_df["site"].dropna().astype(str).unique().tolist())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_n = np.zeros(len(site_keys), dtype=np.float32)
    site_p = np.zeros((len(site_keys), Y_prior.shape[1]), dtype=np.float32)

    for s in site_keys:
        i = site_to_i[s]
        mask = prior_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        site_p[i] = Y_prior[mask].mean(axis=0)

    # Hour
    hour_keys = sorted(prior_df["hour_utc"].dropna().astype(int).unique().tolist())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_n = np.zeros(len(hour_keys), dtype=np.float32)
    hour_p = np.zeros((len(hour_keys), Y_prior.shape[1]), dtype=np.float32)

    for h in hour_keys:
        i = hour_to_i[h]
        mask = prior_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        hour_p[i] = Y_prior[mask].mean(axis=0)

    # Site-hour
    sh_to_i = {}
    sh_n_list = []
    sh_p_list = []

    for (s, h), idx in prior_df.groupby(["site", "hour_utc"]).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n_list)
        idx = np.array(list(idx))
        sh_n_list.append(len(idx))
        sh_p_list.append(Y_prior[idx].mean(axis=0))

    sh_n = np.array(sh_n_list, dtype=np.float32)
    sh_p = np.stack(sh_p_list).astype(np.float32) if len(sh_p_list) else np.zeros((0, Y_prior.shape[1]), dtype=np.float32)

    return {
        "global_p": global_p,
        "site_to_i": site_to_i,
        "site_n": site_n,
        "site_p": site_p,
        "hour_to_i": hour_to_i,
        "hour_n": hour_n,
        "hour_p": hour_p,
        "sh_to_i": sh_to_i,
        "sh_n": sh_n,
        "sh_p": sh_p,
    }

def prior_logits_from_tables(sites, hours, tables, eps=1e-4):
    n = len(sites)
    p = np.repeat(tables["global_p"][None, :], n, axis=0).astype(np.float32, copy=True)

    site_idx = np.fromiter(
        (tables["site_to_i"].get(str(s), -1) for s in sites),
        dtype=np.int32,
        count=n
    )
    hour_idx = np.fromiter(
        (tables["hour_to_i"].get(int(h), -1) if int(h) >= 0 else -1 for h in hours),
        dtype=np.int32,
        count=n
    )
    sh_idx = np.fromiter(
        (tables["sh_to_i"].get((str(s), int(h)), -1) if int(h) >= 0 else -1 for s, h in zip(sites, hours)),
        dtype=np.int32,
        count=n
    )

    valid = hour_idx >= 0
    if valid.any():
        nh = tables["hour_n"][hour_idx[valid]][:, None]
        wh = nh / (nh + 8.0)
        p[valid] = wh * tables["hour_p"][hour_idx[valid]] + (1.0 - wh) * p[valid]

    valid = site_idx >= 0
    if valid.any():
        ns = tables["site_n"][site_idx[valid]][:, None]
        ws = ns / (ns + 8.0)
        p[valid] = ws * tables["site_p"][site_idx[valid]] + (1.0 - ws) * p[valid]

    valid = sh_idx >= 0
    if valid.any():
        nsh = tables["sh_n"][sh_idx[valid]][:, None]
        wsh = nsh / (nsh + 4.0)
        p[valid] = wsh * tables["sh_p"][sh_idx[valid]] + (1.0 - wsh) * p[valid]

    np.clip(p, eps, 1.0 - eps, out=p)
    return (np.log(p) - np.log1p(-p)).astype(np.float32, copy=False)

def fuse_scores_with_tables(base_scores, sites, hours, tables,
                            lambda_event=BEST["lambda_event"],
                            lambda_texture=BEST["lambda_texture"],
                            lambda_proxy_texture=BEST["lambda_proxy_texture"],
                            smooth_texture=BEST["smooth_texture"],
                            smooth_event=BEST["smooth_event"]):
    scores = base_scores.copy()
    prior = prior_logits_from_tables(sites, hours, tables)

    # mapped active
    if len(idx_mapped_active_event):
        scores[:, idx_mapped_active_event] += lambda_event * prior[:, idx_mapped_active_event]

    if len(idx_mapped_active_texture):
        scores[:, idx_mapped_active_texture] += lambda_texture * prior[:, idx_mapped_active_texture]

    # selected frog proxies
    if len(idx_selected_proxy_active_texture):
        scores[:, idx_selected_proxy_active_texture] += lambda_proxy_texture * prior[:, idx_selected_proxy_active_texture]

    # prior-only active unmapped
    if len(idx_selected_prioronly_active_event):
        scores[:, idx_selected_prioronly_active_event] = lambda_event * prior[:, idx_selected_prioronly_active_event]

    if len(idx_selected_prioronly_active_texture):
        scores[:, idx_selected_prioronly_active_texture] = lambda_texture * prior[:, idx_selected_prioronly_active_texture]

    # inactive unmapped
    if len(idx_unmapped_inactive):
        scores[:, idx_unmapped_inactive] = -8.0

    scores = smooth_cols_fixed12(scores, idx_active_texture, alpha=smooth_texture)
    scores = smooth_events_fixed12(scores, idx_active_event, alpha=smooth_event)
    return scores.astype(np.float32, copy=False), prior

# %% cell 18
# Cell 8 — Honest OOF base/prior meta-features (required for final stacker fit)
def build_oof_base_prior(scores_full_raw, meta_full, sc_clean, Y_SC, n_splits=5, verbose=True):
    groups_full = meta_full["filename"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    oof_base = np.zeros_like(scores_full_raw, dtype=np.float32)
    oof_prior = np.zeros_like(scores_full_raw, dtype=np.float32)
    fold_id = np.full(len(meta_full), -1, dtype=np.int16)

    splits = list(gkf.split(scores_full_raw, groups=groups_full))
    iterator = tqdm(splits, desc="OOF base/prior folds", disable=not verbose)

    for fold, (tr_idx, va_idx) in enumerate(iterator, 1):
        tr_idx = np.sort(tr_idx)
        va_idx = np.sort(va_idx)

        val_files = set(meta_full.iloc[va_idx]["filename"].tolist())

        # Fold-safe prior tables: exclude all validation files
        prior_mask = ~sc_clean["filename"].isin(val_files).values
        prior_df_fold = sc_clean.loc[prior_mask].reset_index(drop=True)
        Y_prior_fold = Y_SC[prior_mask]

        tables = fit_prior_tables(prior_df_fold, Y_prior_fold)

        va_base, va_prior = fuse_scores_with_tables(
            scores_full_raw[va_idx],
            sites=meta_full.iloc[va_idx]["site"].to_numpy(),
            hours=meta_full.iloc[va_idx]["hour_utc"].to_numpy(),
            tables=tables,
        )

        oof_base[va_idx] = va_base
        oof_prior[va_idx] = va_prior
        fold_id[va_idx] = fold

    assert (fold_id >= 0).all()
    return oof_base, oof_prior, fold_id

OOF_META_CACHE = CFG["full_cache_work_dir"] / "full_oof_meta_features.npz"

if OOF_META_CACHE.exists():
    print("Loading cached OOF meta-features from:", OOF_META_CACHE)
    arr = np.load(OOF_META_CACHE)
    oof_base = arr["oof_base"].astype(np.float32)
    oof_prior = arr["oof_prior"].astype(np.float32)
    oof_fold_id = arr["fold_id"].astype(np.int16)
else:
    print("Building OOF meta-features...")
    oof_base, oof_prior, oof_fold_id = build_oof_base_prior(
        scores_full_raw=scores_full_raw,
        meta_full=meta_full,
        sc_clean=sc_clean,
        Y_SC=Y_SC,
        n_splits=5,
        verbose=CFG["verbose"],
    )

    np.savez_compressed(
        OOF_META_CACHE,
        oof_base=oof_base,
        oof_prior=oof_prior,
        fold_id=oof_fold_id,
    )
    print("Saved OOF meta-features to:", OOF_META_CACHE)

baseline_oof_auc = macro_auc_skip_empty(Y_FULL, oof_base)

if MODE == "train":
    raw_local_auc = macro_auc_skip_empty(Y_FULL, scores_full_raw)
    print(f"Raw local AUC (not OOF-dependent): {raw_local_auc:.6f}")
    print(f"Honest OOF baseline AUC: {baseline_oof_auc:.6f}")

# %% cell 19
# Cell 9 — Classwise embedding-probe helpers
def build_class_features(emb_proj, raw_col, prior_col, base_col):
    """
    emb_proj: (n, d)
    raw_col, prior_col, base_col: (n,)
    returns: (n, d + 13)

    Fitur: embedding + 7 sequential + 3 interaction + std + 3 diff
    """
    prev_base, next_base, mean_base, max_base, std_base = seq_features_1d(base_col)

    # Diff features: posisi window relatif terhadap konteks file
    diff_mean = base_col - mean_base   # apakah window ini lebih tinggi dari rata2 file?
    diff_prev = base_col - prev_base   # onset: naik dari window sebelumnya?
    diff_next = base_col - next_base   # offset: turun ke window berikutnya?

    feats = np.concatenate([
        emb_proj,
        raw_col[:, None],
        prior_col[:, None],
        base_col[:, None],
        prev_base[:, None],
        next_base[:, None],
        mean_base[:, None],
        max_base[:, None],
        std_base[:, None],             # variance temporal dalam file
        diff_mean[:, None],            # deviasi dari mean file
        diff_prev[:, None],            # deteksi onset
        diff_next[:, None],            # deteksi offset
        # interaction terms
        (raw_col * prior_col)[:, None],
        (raw_col * base_col)[:, None],
        (prior_col * base_col)[:, None],
    ], axis=1)

    return feats.astype(np.float32, copy=False)

def run_oof_embedding_probe(
    scores_raw,
    emb,
    meta_df,
    y_true,
    pca_dim=64,
    min_pos=8,
    C=0.25,
    alpha=0.5,
):
    groups = meta_df["filename"].to_numpy()
    gkf = GroupKFold(n_splits=5)

    oof_base_local = np.zeros_like(scores_raw, dtype=np.float32)
    oof_final = np.zeros_like(scores_raw, dtype=np.float32)

    modeled_counts = np.zeros(scores_raw.shape[1], dtype=np.int32)

    split_list = list(gkf.split(scores_raw, groups=groups))

    for fold, (tr_idx, va_idx) in enumerate(tqdm(split_list, desc="Embedding-probe folds", disable=not CFG["verbose"]), 1):
    # for fold, (tr_idx, va_idx) in enumerate(tqdm(split_list, desc="Embedding-probe folds"), 1):
        tr_idx = np.sort(tr_idx)
        va_idx = np.sort(va_idx)

        val_files = set(meta_df.iloc[va_idx]["filename"].tolist())

        # Fold-safe priors
        prior_mask = ~sc_clean["filename"].isin(val_files).values
        prior_df_fold = sc_clean.loc[prior_mask].reset_index(drop=True)
        Y_prior_fold = Y_SC[prior_mask]
        tables = fit_prior_tables(prior_df_fold, Y_prior_fold)

        base_tr, prior_tr = fuse_scores_with_tables(
            scores_raw[tr_idx],
            sites=meta_df.iloc[tr_idx]["site"].to_numpy(),
            hours=meta_df.iloc[tr_idx]["hour_utc"].to_numpy(),
            tables=tables,
        )
        base_va, prior_va = fuse_scores_with_tables(
            scores_raw[va_idx],
            sites=meta_df.iloc[va_idx]["site"].to_numpy(),
            hours=meta_df.iloc[va_idx]["hour_utc"].to_numpy(),
            tables=tables,
        )

        oof_base_local[va_idx] = base_va
        oof_final[va_idx] = base_va

        # Embedding preprocessing on train fold only
        scaler = StandardScaler()
        emb_tr_s = scaler.fit_transform(emb[tr_idx])
        emb_va_s = scaler.transform(emb[va_idx])

        n_comp = min(pca_dim, emb_tr_s.shape[0] - 1, emb_tr_s.shape[1])
        pca = PCA(n_components=n_comp)
        Z_tr = pca.fit_transform(emb_tr_s).astype(np.float32)
        Z_va = pca.transform(emb_va_s).astype(np.float32)

        class_iterator = np.where(y_true[tr_idx].sum(axis=0) >= min_pos)[0].tolist()

        for cls_idx in tqdm(class_iterator, desc=f"Fold {fold} classes", leave=False, disable=not CFG["verbose"]):
        # for cls_idx in tqdm(class_iterator, desc=f"Fold {fold} classes", leave=False):
            y_tr = y_true[tr_idx, cls_idx]

            if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
                continue

            X_tr_cls = build_class_features(
                Z_tr,
                raw_col=scores_raw[tr_idx, cls_idx],
                prior_col=prior_tr[:, cls_idx],
                base_col=base_tr[:, cls_idx],
            )
            X_va_cls = build_class_features(
                Z_va,
                raw_col=scores_raw[va_idx, cls_idx],
                prior_col=prior_va[:, cls_idx],
                base_col=base_va[:, cls_idx],
            )

            # Pilih backend probe: mlp | lgbm | logreg
            backend = CFG.get("probe_backend", "mlp")
            n_pos = int(y_tr.sum())
            n_neg = len(y_tr) - n_pos

            if backend == "mlp":
                # MLPClassifier tidak support sample_weight
                # Gunakan oversampling: duplikasi positif agar balance
                if n_pos > 0 and n_neg > n_pos:
                    repeat = max(1, n_neg // n_pos)
                    pos_idx = np.where(y_tr == 1)[0]
                    X_bal = np.vstack([X_tr_cls, np.tile(X_tr_cls[pos_idx], (repeat, 1))])
                    y_bal = np.concatenate([y_tr, np.ones(len(pos_idx) * repeat, dtype=y_tr.dtype)])
                else:
                    X_bal, y_bal = X_tr_cls, y_tr
                clf = MLPClassifier(**CFG["mlp_params"])
                clf.fit(X_bal, y_bal)
                pred_va = clf.predict_proba(X_va_cls)[:, 1].astype(np.float32)
                pred_va = np.log(pred_va + 1e-7) - np.log(1 - pred_va + 1e-7)
            elif backend == "lgbm" and _LGBM_AVAILABLE:
                scale_pos = max(1.0, n_neg / max(n_pos, 1))
                clf = LGBMClassifier(
                    **CFG["lgbm_params"],
                    scale_pos_weight=scale_pos,
                )
                clf.fit(X_tr_cls, y_tr)
                pred_va = clf.predict_proba(X_va_cls)[:, 1].astype(np.float32)
                pred_va = np.log(pred_va + 1e-7) - np.log(1 - pred_va + 1e-7)
            else:
                clf = LogisticRegression(
                    C=C, max_iter=400, solver="liblinear",
                    class_weight="balanced",
                )
                clf.fit(X_tr_cls, y_tr)
                pred_va = clf.decision_function(X_va_cls).astype(np.float32)

            oof_final[va_idx, cls_idx] = (
                (1.0 - alpha) * base_va[:, cls_idx] +
                alpha * pred_va
            )

            modeled_counts[cls_idx] += 1

    score_base = macro_auc_skip_empty(y_true, oof_base_local)
    score_final = macro_auc_skip_empty(y_true, oof_final)

    return {
        "oof_base": oof_base_local,
        "oof_final": oof_final,
        "modeled_counts": modeled_counts,
        "score_base": score_base,
        "score_final": score_final,
    }

# %% cell 20
# ProtoSSM v4 — Enhanced with Cross-Attention Layer

class SelectiveSSM(nn.Module):
    # Simplified Mamba-style selective state space model.
    # Input-dependent (selective) discretization of continuous-time SSM.
    # For T=12 bioacoustic windows, the sequential scan is efficient on CPU.

    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model, d_conv,
            padding=d_conv - 1, groups=d_model
        )
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B_size, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)

        h = torch.zeros(B_size, D, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dt_t = dt[:, t, :]
            dA = torch.exp(A[None, :, :] * dt_t[:, :, None])
            dB = dt_t[:, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            y_t = (h * C[:, t, None, :]).sum(-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


class TemporalCrossAttention(nn.Module):
    """Multi-head cross-attention between temporal windows.
    Captures non-local patterns (e.g., dawn chorus onset, counter-singing)
    that sequential SSM may miss."""
    
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (B, T, D)
        residual = x
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out
        
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x


class ProtoSSMv2(nn.Module):
    # Prototypical State Space Model v4 with cross-attention and metadata awareness.
    #
    # V16 additions:
    # - Cross-attention layer after SSM for non-local temporal patterns
    # - All other v2 features preserved (metadata, prototypes, gated fusion)
    
    def __init__(self, d_input=1536, d_model=192, d_state=16,
                 n_ssm_layers=2, n_classes=234, n_windows=12,
                 dropout=0.2, n_sites=20, meta_dim=16,
                 use_cross_attn=True, cross_attn_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_windows = n_windows

        # 1. Feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Learnable positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        # 3. Metadata embeddings
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        # 4. Bidirectional SSM layers
        self.ssm_fwd = nn.ModuleList()
        self.ssm_bwd = nn.ModuleList()
        self.ssm_merge = nn.ModuleList()
        self.ssm_norm = nn.ModuleList()
        for _ in range(n_ssm_layers):
            self.ssm_fwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_bwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_merge.append(nn.Linear(2 * d_model, d_model))
            self.ssm_norm.append(nn.LayerNorm(d_model))
        self.ssm_drop = nn.Dropout(dropout)

        # 4b. NEW: Cross-attention after SSM
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = TemporalCrossAttention(d_model, n_heads=cross_attn_heads, dropout=dropout)

        # 5. Learnable class prototypes
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))

        # 6. Per-class calibration bias
        self.class_bias = nn.Parameter(torch.zeros(n_classes))

        # 7. Per-class gated fusion with Perch logits
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

        # 8. Taxonomic auxiliary head
        self.n_families = 0
        self.family_head = None

    def init_prototypes_from_data(self, embeddings, labels):
        with torch.no_grad():
            h = self.input_proj(embeddings)
            for c in range(self.n_classes):
                mask = labels[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)

    def init_family_head(self, n_families, class_to_family):
        self.n_families = n_families
        self.family_head = nn.Linear(self.d_model, n_families)
        self.register_buffer('class_to_family', torch.tensor(class_to_family, dtype=torch.long))

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape

        # Project embeddings
        h = self.input_proj(emb)
        h = h + self.pos_enc[:, :T, :]

        # Add metadata embeddings
        if site_ids is not None and hours is not None:
            s_emb = self.site_emb(site_ids)
            h_emb = self.hour_emb(hours)
            meta = self.meta_proj(torch.cat([s_emb, h_emb], dim=-1))
            h = h + meta[:, None, :]

        # Bidirectional SSM
        for fwd, bwd, merge, norm in zip(
            self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm
        ):
            residual = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h = merge(torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = norm(h + residual)

        # NEW: Cross-attention for non-local temporal patterns
        if self.use_cross_attn:
            h = self.cross_attn(h)

        h_temporal = h

        # Prototypical cosine similarity + class bias
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        temp = F.softplus(self.proto_temp)
        sim = torch.matmul(h_norm, p_norm.T) * temp + self.class_bias[None, None, :]

        # Gated fusion with Perch logits
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            species_logits = alpha * sim + (1 - alpha) * perch_logits
        else:
            species_logits = sim

        # Taxonomic auxiliary prediction
        family_logits = None
        if self.family_head is not None:
            h_pool = h.mean(dim=1)
            family_logits = self.family_head(h_pool)

        return species_logits, family_logits, h_temporal

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

ssm_cfg = CFG["proto_ssm"]
print("ProtoSSMv4 architecture defined (with cross-attention).")
test_model = ProtoSSMv2(
    d_model=ssm_cfg["d_model"], n_ssm_layers=2,
    n_sites=ssm_cfg["n_sites"], meta_dim=ssm_cfg["meta_dim"],
    use_cross_attn=ssm_cfg.get("use_cross_attn", True),
    cross_attn_heads=ssm_cfg.get("cross_attn_heads", 4),
)
print(f"Parameter count: {test_model.count_parameters():,}")
del test_model

# %% cell 21
# ProtoSSM v4 Training Loop — with Mixup, Focal Loss, SWA

def build_taxonomy_groups(taxonomy_df, primary_labels):
    for col in ["family", "order", "class_name"]:
        if col in taxonomy_df.columns:
            group_map = taxonomy_df.set_index("primary_label")[col].to_dict()
            break
    else:
        group_map = {label: "Unknown" for label in primary_labels}

    groups = sorted(set(group_map.values()))
    grp_to_idx = {g: i for i, g in enumerate(groups)}
    class_to_group = []
    for label in primary_labels:
        grp = group_map.get(label, "Unknown")
        class_to_group.append(grp_to_idx.get(grp, 0))
    return len(groups), class_to_group, grp_to_idx


def build_site_mapping(meta_df):
    sites = meta_df["site"].unique().tolist()
    site_to_idx = {s: i + 1 for i, s in enumerate(sites)}
    n_sites = len(sites) + 1
    return site_to_idx, n_sites


def reshape_to_files(flat_array, meta_df, n_windows=N_WINDOWS):
    filenames = meta_df["filename"].to_numpy()
    unique_files = []
    seen = set()
    for f in filenames:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)

    n_files = len(unique_files)
    assert len(flat_array) == n_files * n_windows, \
        f"Expected {n_files * n_windows} rows, got {len(flat_array)}"

    new_shape = (n_files, n_windows) + flat_array.shape[1:]
    return flat_array.reshape(new_shape), unique_files


def get_file_metadata(meta_df, file_list, site_to_idx, n_sites_max):
    file_to_row = {}
    filenames = meta_df["filename"].to_numpy()
    sites = meta_df["site"].to_numpy()
    hours = meta_df["hour_utc"].to_numpy()
    for i, f in enumerate(filenames):
        if f not in file_to_row:
            file_to_row[f] = i

    site_ids = np.zeros(len(file_list), dtype=np.int64)
    hour_ids = np.zeros(len(file_list), dtype=np.int64)
    for fi, fname in enumerate(file_list):
        row = file_to_row.get(fname)
        if row is not None:
            sid = site_to_idx.get(sites[row], 0)
            site_ids[fi] = min(sid, n_sites_max - 1)
            hour_ids[fi] = int(hours[row]) % 24
    return site_ids, hour_ids


def mixup_files(emb, logits, labels, site_ids, hours, families, alpha=0.3):
    """File-level mixup augmentation for ProtoSSM training.
    Mixes pairs of files with random lambda from Beta(alpha, alpha).
    Returns augmented versions of all inputs."""
    n = len(emb)
    if alpha <= 0 or n < 2:
        return emb, logits, labels, site_ids, hours, families
    
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)  # Ensure lam >= 0.5 (dominant sample stays dominant)
    
    perm = np.random.permutation(n)
    
    emb_mix = lam * emb + (1 - lam) * emb[perm]
    logits_mix = lam * logits + (1 - lam) * logits[perm]
    labels_mix = lam * labels + (1 - lam) * labels[perm]
    
    # For discrete features (site, hour), keep the dominant sample's values
    families_mix = lam * families + (1 - lam) * families[perm] if families is not None else None
    
    return emb_mix, logits_mix, labels_mix, site_ids, hours, families_mix


def train_proto_ssm_single(model, emb_train, logits_train, labels_train,
                           site_ids_train=None, hours_train=None,
                           emb_val=None, logits_val=None, labels_val=None,
                           site_ids_val=None, hours_val=None,
                           file_families_train=None, file_families_val=None,
                           cfg=None, verbose=True):
    """Train a single ProtoSSM v4 model with mixup, focal loss, and SWA."""
    if cfg is None:
        cfg = CFG["proto_ssm_train"]

    label_smoothing = cfg.get("label_smoothing", 0.0)
    mixup_alpha = cfg.get("mixup_alpha", 0.0)
    focal_gamma = cfg.get("focal_gamma", 0.0)
    swa_start_frac = cfg.get("swa_start_frac", 1.0)  # 1.0 = disabled
    n_epochs = cfg["n_epochs"]
    swa_start_epoch = int(n_epochs * swa_start_frac)

    # Convert to tensors (base — unmixed)
    labels_np = labels_train.copy()
    
    # Apply label smoothing
    if label_smoothing > 0:
        labels_np = labels_np * (1.0 - label_smoothing) + label_smoothing / 2.0

    has_val = emb_val is not None
    if has_val:
        emb_v = torch.tensor(emb_val, dtype=torch.float32)
        logits_v = torch.tensor(logits_val, dtype=torch.float32)
        labels_v = torch.tensor(labels_val, dtype=torch.float32)
        site_v = torch.tensor(site_ids_val, dtype=torch.long) if site_ids_val is not None else None
        hour_v = torch.tensor(hours_val, dtype=torch.long) if hours_val is not None else None

    fam_v = torch.tensor(file_families_val, dtype=torch.float32) if (has_val and file_families_val is not None) else None

    # Class weights for imbalanced data
    labels_tr_t = torch.tensor(labels_np, dtype=torch.float32)
    pos_counts = labels_tr_t.sum(dim=(0, 1))
    total = labels_tr_t.shape[0] * labels_tr_t.shape[1]
    pos_weight = ((total - pos_counts) / (pos_counts + 1)).clamp(max=cfg["pos_weight_cap"])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"],
        epochs=n_epochs, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy='cos'
    )

    best_val_loss = float('inf')
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    # SWA state accumulator
    swa_state = None
    swa_count = 0

    for epoch in range(n_epochs):
        # === Mixup augmentation (per-epoch re-sampling) ===
        if mixup_alpha > 0 and epoch > 5:  # Skip mixup for first 5 epochs (warmup)
            emb_mix, logits_mix, labels_mix, _, _, fam_mix = mixup_files(
                emb_train, logits_train, labels_np,
                site_ids_train, hours_train, file_families_train,
                alpha=mixup_alpha,
            )
        else:
            emb_mix, logits_mix, labels_mix = emb_train, logits_train, labels_np
            fam_mix = file_families_train

        emb_tr = torch.tensor(emb_mix, dtype=torch.float32)
        logits_tr = torch.tensor(logits_mix, dtype=torch.float32)
        labels_tr = torch.tensor(labels_mix, dtype=torch.float32)
        site_tr = torch.tensor(site_ids_train, dtype=torch.long) if site_ids_train is not None else None
        hour_tr = torch.tensor(hours_train, dtype=torch.long) if hours_train is not None else None
        fam_tr = torch.tensor(fam_mix, dtype=torch.float32) if fam_mix is not None else None

        # === Train ===
        model.train()
        species_out, family_out, _ = model(emb_tr, logits_tr, site_ids=site_tr, hours=hour_tr)

        # Primary loss: focal BCE or weighted BCE
        if focal_gamma > 0:
            loss_main = focal_bce_with_logits(
                species_out, labels_tr,
                gamma=focal_gamma,
                pos_weight=pos_weight[None, None, :],
            )
        else:
            loss_main = F.binary_cross_entropy_with_logits(
                species_out, labels_tr,
                pos_weight=pos_weight[None, None, :]
            )

        # Knowledge distillation loss
        loss_distill = F.mse_loss(species_out, logits_tr)

        # Total loss
        loss = loss_main + cfg["distill_weight"] * loss_distill

        # Taxonomic auxiliary loss
        if family_out is not None and fam_tr is not None:
            loss_family = F.binary_cross_entropy_with_logits(family_out, fam_tr)
            loss = loss + 0.1 * loss_family

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # === SWA accumulation ===
        if epoch >= swa_start_epoch:
            if swa_state is None:
                swa_state = {k: v.clone() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                for k in swa_state:
                    swa_state[k] += model.state_dict()[k]
                swa_count += 1

        # === Validate ===
        model.eval()
        with torch.no_grad():
            if has_val:
                val_out, val_fam, _ = model(emb_v, logits_v, site_ids=site_v, hours=hour_v)
                val_loss = F.binary_cross_entropy_with_logits(
                    val_out, labels_v,
                    pos_weight=pos_weight[None, None, :]
                )

                val_pred = val_out.reshape(-1, val_out.shape[-1]).numpy()
                val_true = labels_v.reshape(-1, labels_v.shape[-1]).numpy()
                try:
                    val_auc = macro_auc_skip_empty(val_true, val_pred)
                except Exception:
                    val_auc = 0.0
            else:
                val_loss = loss
                val_auc = 0.0

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["val_auc"].append(val_auc)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and (epoch + 1) % 20 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            swa_info = f" swa={swa_count}" if swa_count > 0 else ""
            print(f"  Epoch {epoch+1:3d}: train={loss.item():.4f} val={val_loss.item():.4f} "
                  f"auc={val_auc:.4f} lr={lr_now:.6f} wait={wait}{swa_info}")

        if wait >= cfg["patience"]:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
            break

    # Apply SWA if we accumulated enough checkpoints
    if swa_state is not None and swa_count >= 3:
        if verbose:
            print(f"  Applying SWA (averaged {swa_count} checkpoints)")
        avg_state = {k: v / swa_count for k, v in swa_state.items()}
        model.load_state_dict(avg_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"  Training complete. Best val_loss={best_val_loss:.4f}")
        with torch.no_grad():
            alphas = torch.sigmoid(model.fusion_alpha).numpy()
            print(f"  Fusion alpha: mean={alphas.mean():.3f} min={alphas.min():.3f} max={alphas.max():.3f}")
            print(f"  Proto temperature: {F.softplus(model.proto_temp).item():.3f}")

    return model, history


def run_proto_ssm_oof(emb_files, logits_files, labels_files,
                      site_ids_all, hours_all,
                      file_families, file_groups,
                      n_families, class_to_family,
                      cfg=None, verbose=True):
    """Run GroupKFold OOF cross-validation for ProtoSSM v4."""
    if cfg is None:
        cfg = CFG["proto_ssm_train"]

    n_splits = cfg.get("oof_n_splits", 5)
    n_files = len(emb_files)
    ssm_cfg = CFG["proto_ssm"]

    oof_preds = np.zeros((n_files, N_WINDOWS, N_CLASSES), dtype=np.float32)
    fold_histories = []
    fold_alphas = []

    n_unique_groups = len(set(file_groups))
    if n_unique_groups < n_splits:
        print(f"  WARNING: Only {n_unique_groups} groups, reducing n_splits from {n_splits} to {n_unique_groups}")
        n_splits = n_unique_groups
    gkf = GroupKFold(n_splits=n_splits)
    dummy_y = np.zeros(n_files)

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(dummy_y, dummy_y, file_groups)):
        if verbose:
            print(f"\n--- Fold {fold_i+1}/{n_splits} (train={len(train_idx)}, val={len(val_idx)}) ---")

        fold_model = ProtoSSMv2(
            d_input=emb_files.shape[2],
            d_model=ssm_cfg["d_model"],
            d_state=ssm_cfg["d_state"],
            n_ssm_layers=ssm_cfg["n_ssm_layers"],
            n_classes=N_CLASSES,
            n_windows=N_WINDOWS,
            dropout=ssm_cfg["dropout"],
            n_sites=ssm_cfg["n_sites"],
            meta_dim=ssm_cfg["meta_dim"],
            use_cross_attn=ssm_cfg.get("use_cross_attn", True),
            cross_attn_heads=ssm_cfg.get("cross_attn_heads", 4),
        ).to(DEVICE)

        # Initialize prototypes
        emb_flat_fold = emb_files[train_idx].reshape(-1, emb_files.shape[2])
        labels_flat_fold = labels_files[train_idx].reshape(-1, N_CLASSES)
        fold_model.init_prototypes_from_data(
            torch.tensor(emb_flat_fold, dtype=torch.float32),
            torch.tensor(labels_flat_fold, dtype=torch.float32)
        )
        fold_model.init_family_head(n_families, class_to_family)

        # Train on fold
        fold_model, fold_hist = train_proto_ssm_single(
            fold_model,
            emb_files[train_idx], logits_files[train_idx], labels_files[train_idx].astype(np.float32),
            site_ids_train=site_ids_all[train_idx], hours_train=hours_all[train_idx],
            emb_val=emb_files[val_idx], logits_val=logits_files[val_idx],
            labels_val=labels_files[val_idx].astype(np.float32),
            site_ids_val=site_ids_all[val_idx], hours_val=hours_all[val_idx],
            file_families_train=file_families[train_idx],
            file_families_val=file_families[val_idx],
            cfg=cfg, verbose=verbose,
        )

        # OOF predictions with TTA
        fold_model.eval()
        tta_shifts = CFG.get("tta_shifts", [0])
        if len(tta_shifts) > 1:
            oof_preds[val_idx] = temporal_shift_tta(
                emb_files[val_idx], logits_files[val_idx], fold_model,
                site_ids_all[val_idx], hours_all[val_idx], shifts=tta_shifts
            )
        else:
            with torch.no_grad():
                val_emb = torch.tensor(emb_files[val_idx], dtype=torch.float32)
                val_logits = torch.tensor(logits_files[val_idx], dtype=torch.float32)
                val_sites = torch.tensor(site_ids_all[val_idx], dtype=torch.long)
                val_hours = torch.tensor(hours_all[val_idx], dtype=torch.long)
                val_out, _, _ = fold_model(val_emb, val_logits, site_ids=val_sites, hours=val_hours)
                oof_preds[val_idx] = val_out.numpy()

        fold_alphas.append(torch.sigmoid(fold_model.fusion_alpha).detach().numpy().copy())
        fold_histories.append(fold_hist)

    return oof_preds, fold_histories, fold_alphas


def optimize_ensemble_weight(oof_proto_flat, oof_mlp_flat, y_true_flat):
    """Grid search over blend weights to find optimal ProtoSSM ensemble weight."""
    weights = np.arange(0.0, 1.05, 0.05)
    results = []

    for w in weights:
        blended = w * oof_proto_flat + (1.0 - w) * oof_mlp_flat
        try:
            auc = macro_auc_skip_empty(y_true_flat, blended)
        except Exception:
            auc = 0.0
        results.append((w, auc))

    best_w, best_auc = max(results, key=lambda x: x[1])
    return best_w, best_auc, results


print("ProtoSSM v4 training functions defined (with mixup, focal loss, SWA, TTA).")

# %% cell 22
# Cell 10 — Probe tuning (train mode only)
grid_results = None
BEST_PROBE = None

if CFG["run_probe_check"]:
    probe_result = run_oof_embedding_probe(
        scores_raw=scores_full_raw,
        emb=emb_full,
        meta_df=meta_full,
        y_true=Y_FULL,
        pca_dim=64,
        min_pos=8,
        C=0.25,
        alpha=0.5,
    )

    print(f"Honest OOF baseline AUC: {probe_result['score_base']:.6f}")
    print(f"Honest OOF embedding-probe AUC: {probe_result['score_final']:.6f}")
    print(f"Delta: {probe_result['score_final'] - probe_result['score_base']:.6f}")

    modeled_classes = np.where(probe_result["modeled_counts"] > 0)[0]
    print("Modeled classes:", len(modeled_classes))
    print([PRIMARY_LABELS[i] for i in modeled_classes[:20]])

if CFG["run_probe_grid"]:
    param_grid = [
        {"pca_dim": 32, "min_pos": 8,  "C": 0.25, "alpha": 0.4},
        {"pca_dim": 64, "min_pos": 8,  "C": 0.25, "alpha": 0.4},
        {"pca_dim": 64, "min_pos": 8,  "C": 0.25, "alpha": 0.5},
        {"pca_dim": 64, "min_pos": 12, "C": 0.25, "alpha": 0.4},
        {"pca_dim": 96, "min_pos": 8,  "C": 0.25, "alpha": 0.4},
        {"pca_dim": 64, "min_pos": 8,  "C": 0.50, "alpha": 0.4},
    ]

    results = []
    for params in tqdm(param_grid, desc="Probe grid", disable=not CFG["verbose"]):
        out = run_oof_embedding_probe(
            scores_raw=scores_full_raw,
            emb=emb_full,
            meta_df=meta_full,
            y_true=Y_FULL,
            pca_dim=params["pca_dim"],
            min_pos=params["min_pos"],
            C=params["C"],
            alpha=params["alpha"],
        )
        results.append({
            **params,
            "baseline_oof_auc": out["score_base"],
            "probe_oof_auc": out["score_final"],
            "delta": out["score_final"] - out["score_base"],
            "n_modeled_classes": int((out["modeled_counts"] > 0).sum()),
        })

    grid_results = pd.DataFrame(results).sort_values("probe_oof_auc", ascending=False).reset_index(drop=True)
    display(grid_results)

    BEST_PROBE = {
        "pca_dim": int(grid_results.iloc[0]["pca_dim"]),
        "min_pos": int(grid_results.iloc[0]["min_pos"]),
        "C": float(grid_results.iloc[0]["C"]),
        "alpha": float(grid_results.iloc[0]["alpha"]),
    }

    # Save best params for future freezing
    best_probe_path = CFG["full_cache_work_dir"] / "best_probe_params.json"
    best_probe_path.write_text(json.dumps(BEST_PROBE, indent=2))
    print("Saved best probe params to:", best_probe_path)

else:
    BEST_PROBE = CFG["frozen_best_probe"]
    print("Using frozen BEST_PROBE in submit mode:")
    print(BEST_PROBE)

if grid_results is not None:
    grid_results.to_csv(CFG["full_cache_work_dir"] / "probe_grid_results.csv", index=False)

# %% cell 23
# Cell 11 — Freeze final probe params
if BEST_PROBE is None:
    BEST_PROBE = CFG["frozen_best_probe"]

print("Final BEST_PROBE =", BEST_PROBE)

# Optional — rerun best OOF probe once for diagnostics / caching
BEST_OOF_RESULT = None

if MODE == "train":
    BEST_OOF_RESULT = run_oof_embedding_probe(
        scores_raw=scores_full_raw,
        emb=emb_full,
        meta_df=meta_full,
        y_true=Y_FULL,
        pca_dim=int(BEST_PROBE["pca_dim"]),
        min_pos=int(BEST_PROBE["min_pos"]),
        C=float(BEST_PROBE["C"]),
        alpha=float(BEST_PROBE["alpha"]),
    )

    print(f"Honest OOF baseline AUC (BEST_PROBE rerun): {BEST_OOF_RESULT['score_base']:.6f}")
    print(f"Honest OOF probe AUC   (BEST_PROBE rerun): {BEST_OOF_RESULT['score_final']:.6f}")

# %% cell 24
# Cell 12 — Fit final prior tables on all labeled soundscapes
final_prior_tables = fit_prior_tables(sc_clean.reset_index(drop=True), Y_SC)

print("Built final prior tables for inference.")
print("OOF baseline AUC used for stacker training:", baseline_oof_auc)

# %% cell 25
# Cell 13 — Fit embedding scaler + PCA on all trusted full windows
emb_scaler = StandardScaler()
emb_full_scaled = emb_scaler.fit_transform(emb_full)

n_comp = min(
    int(BEST_PROBE["pca_dim"]),
    emb_full_scaled.shape[0] - 1,
    emb_full_scaled.shape[1]
)

emb_pca = PCA(n_components=n_comp)
Z_FULL = emb_pca.fit_transform(emb_full_scaled).astype(np.float32)

print("emb_full:", emb_full.shape)
print("Z_FULL:", Z_FULL.shape)
print("Explained variance ratio sum:", emb_pca.explained_variance_ratio_.sum())

# %% cell 26
# Instantiate and train ProtoSSM v4

# --- Step 1: Reshape to file-level ---
emb_files, file_list = reshape_to_files(emb_full, meta_full)
logits_files, _ = reshape_to_files(scores_full_raw, meta_full)
labels_files, _ = reshape_to_files(Y_FULL, meta_full)

print(f"Reshaped to file-level: emb={emb_files.shape}, logits={logits_files.shape}, labels={labels_files.shape}")
print(f"Files: {len(file_list)}")

# --- Step 2: Build taxonomy groups, site mapping, file metadata ---
n_families, class_to_family, fam_to_idx = build_taxonomy_groups(taxonomy, PRIMARY_LABELS)
print(f"Taxonomic groups: {n_families}")

site_to_idx, n_sites_mapped = build_site_mapping(meta_full)
n_sites_cfg = CFG["proto_ssm"]["n_sites"]
print(f"Sites mapped: {n_sites_mapped} (capped to {n_sites_cfg})")

site_ids_all, hours_all = get_file_metadata(meta_full, file_list, site_to_idx, n_sites_cfg)

# Build per-file family labels (multi-hot)
file_families = np.zeros((len(file_list), n_families), dtype=np.float32)
for fi in range(len(file_list)):
    active_classes = np.where(labels_files[fi].sum(axis=0) > 0)[0]
    for ci in active_classes:
        file_families[fi, class_to_family[ci]] = 1.0

# --- OOF Cross-Validation (TRAIN MODE ONLY) ---
ENSEMBLE_WEIGHT_PROTO = 0.5  # default, overridden by OOF in train mode
oof_proto_flat = None
fold_alphas = []

if MODE == "train":
    file_groups = np.array([f.split("_")[3] if len(f.split("_")) > 3 else f for f in file_list])
    print(f"File groups for OOF: {len(set(file_groups))} unique groups: {sorted(set(file_groups))}")

    t0_oof = time.time()
    oof_proto_preds, fold_histories, fold_alphas = run_proto_ssm_oof(
        emb_files, logits_files, labels_files,
        site_ids_all, hours_all,
        file_families, file_groups,
        n_families, class_to_family,
        cfg=CFG["proto_ssm_train"],
        verbose=CFG["verbose"],
    )
    oof_time = time.time() - t0_oof
    print(f"\nOOF cross-validation time: {oof_time:.1f}s")

    oof_proto_flat = oof_proto_preds.reshape(-1, N_CLASSES)
    y_flat = labels_files.reshape(-1, N_CLASSES).astype(np.float32)

    per_class_auc_proto = {}
    for ci in range(N_CLASSES):
        if y_flat[:, ci].sum() > 0 and y_flat[:, ci].sum() < len(y_flat):
            try:
                per_class_auc_proto[ci] = roc_auc_score(y_flat[:, ci], oof_proto_flat[:, ci])
            except Exception:
                pass

    overall_oof_auc_proto = macro_auc_skip_empty(y_flat, oof_proto_flat)
    print(f"ProtoSSM OOF macro AUC: {overall_oof_auc_proto:.4f}")

    LOGS["oof_auc_proto"] = overall_oof_auc_proto
    LOGS["per_class_auc_proto"] = {PRIMARY_LABELS[k]: v for k, v in per_class_auc_proto.items()}
    LOGS["oof_time"] = oof_time
else:
    print("Submit mode: skipping OOF cross-validation")

# --- Train final model on ALL data ---
ssm_cfg = CFG["proto_ssm"]
model = ProtoSSMv2(
    d_input=emb_full.shape[1],
    d_model=ssm_cfg["d_model"],
    d_state=ssm_cfg["d_state"],
    n_ssm_layers=ssm_cfg["n_ssm_layers"],
    n_classes=N_CLASSES,
    n_windows=N_WINDOWS,
    dropout=ssm_cfg["dropout"],
    n_sites=ssm_cfg["n_sites"],
    meta_dim=ssm_cfg["meta_dim"],
    use_cross_attn=ssm_cfg.get("use_cross_attn", True),
    cross_attn_heads=ssm_cfg.get("cross_attn_heads", 4),
).to(DEVICE)

emb_flat_tensor = torch.tensor(emb_full, dtype=torch.float32)
labels_flat_tensor = torch.tensor(Y_FULL, dtype=torch.float32)
model.init_prototypes_from_data(emb_flat_tensor, labels_flat_tensor)
model.init_family_head(n_families, class_to_family)

print(f"\nProtoSSM v4 parameters: {model.count_parameters():,}")

t0_final = time.time()
model, train_history = train_proto_ssm_single(
    model,
    emb_files, logits_files, labels_files.astype(np.float32),
    site_ids_train=site_ids_all, hours_train=hours_all,
    cfg=CFG["proto_ssm_train"],
    verbose=True,
)
train_time = time.time() - t0_final
print(f"Final model training time: {train_time:.1f}s")

with torch.no_grad():
    final_alphas = torch.sigmoid(model.fusion_alpha).numpy()
    print(f"Fusion alpha: mean={final_alphas.mean():.4f} min={final_alphas.min():.4f} max={final_alphas.max():.4f}")

# --- Train MLP probes ---
PROBE_CLASS_IDX = np.where(Y_FULL.sum(axis=0) >= int(CFG["frozen_best_probe"]["min_pos"]))[0].astype(np.int32)

probe_models = {}
for cls_idx in tqdm(PROBE_CLASS_IDX, desc="Training MLP probes", disable=not CFG["verbose"]):
    y = Y_FULL[:, cls_idx]
    if y.sum() == 0 or y.sum() == len(y):
        continue
    X_cls = build_class_features(
        Z_FULL,
        raw_col=scores_full_raw[:, cls_idx],
        prior_col=oof_prior[:, cls_idx],
        base_col=oof_base[:, cls_idx],
    )
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos > 0 and n_neg > n_pos:
        repeat = max(1, n_neg // n_pos)
        pos_idx = np.where(y == 1)[0]
        X_bal = np.vstack([X_cls, np.tile(X_cls[pos_idx], (repeat, 1))])
        y_bal = np.concatenate([y, np.ones(len(pos_idx) * repeat, dtype=y.dtype)])
    else:
        X_bal, y_bal = X_cls, y
    clf = MLPClassifier(**CFG["mlp_params"])
    clf.fit(X_bal, y_bal)
    probe_models[cls_idx] = clf

print(f"MLP probes trained: {len(probe_models)}")

# --- Optimize ensemble weight (TRAIN MODE ONLY) ---
if MODE == "train" and oof_proto_flat is not None:
    oof_mlp_flat = oof_base.copy()
    for cls_idx, clf in probe_models.items():
        X_cls = build_class_features(
            Z_FULL,
            raw_col=scores_full_raw[:, cls_idx],
            prior_col=oof_prior[:, cls_idx],
            base_col=oof_base[:, cls_idx],
        )
        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X_cls)[:, 1].astype(np.float32)
            pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
        else:
            pred = clf.decision_function(X_cls).astype(np.float32)
        alpha_probe = float(CFG["frozen_best_probe"]["alpha"])
        oof_mlp_flat[:, cls_idx] = (1.0 - alpha_probe) * oof_base[:, cls_idx] + alpha_probe * pred

    y_flat = labels_files.reshape(-1, N_CLASSES).astype(np.float32)
    best_w, best_auc, weight_results = optimize_ensemble_weight(oof_proto_flat, oof_mlp_flat, y_flat)
    ENSEMBLE_WEIGHT_PROTO = best_w

    mlp_only_auc = macro_auc_skip_empty(y_flat, oof_mlp_flat)
    print(f"\n=== Ensemble Optimization ===")
    print(f"Best ProtoSSM weight: {ENSEMBLE_WEIGHT_PROTO:.2f}")
    print(f"Best ensemble OOF AUC: {best_auc:.4f}")
    print(f"MLP-only OOF AUC: {mlp_only_auc:.4f}")

    for w, auc in weight_results:
        marker = " <-- best" if abs(w - best_w) < 0.01 else ""
        print(f"  w={w:.2f}: AUC={auc:.4f}{marker}")

    LOGS["ensemble_weight"] = ENSEMBLE_WEIGHT_PROTO
    LOGS["ensemble_auc"] = best_auc
    LOGS["mlp_only_auc"] = mlp_only_auc
else:
    print(f"\nUsing default ensemble weight: ProtoSSM={ENSEMBLE_WEIGHT_PROTO:.2f}")

LOGS["train_time_final"] = train_time
LOGS["n_probe_models"] = len(probe_models)

if fold_alphas:
    mean_alphas = np.stack(fold_alphas).mean(axis=0)
    print(f"\nFusion alpha (mean across folds):")
    print(f"  ProtoSSM-dominant (alpha>0.5): {(mean_alphas > 0.5).sum()} classes")
    print(f"  Perch-dominant (alpha<=0.5): {(mean_alphas <= 0.5).sum()} classes")

# %% cell 27
# Residual SSM: second-pass boosting on first-pass errors
# Wall-time safety: skip if > 4 min elapsed (leave max time for test inference)
_wall_min = (time.time() - _WALL_START) / 60.0
print(f"Wall time: {_wall_min:.1f} min")

res_model = None
CORRECTION_WEIGHT = 0.0

if _wall_min < 4.0:
    print("Training ResidualSSM...")
    
    class ResidualSSM(nn.Module):
        # Lightweight SSM that takes first-pass scores + embeddings and predicts corrections.
        # Architecture: project(concat(emb, first_pass)) -> 1-layer BiSSM -> linear head
    
        def __init__(self, d_input=1536, d_scores=234, d_model=64, d_state=8,
                     n_classes=234, n_windows=12, dropout=0.1, n_sites=20, meta_dim=8):
            super().__init__()
            self.d_model = d_model
            self.n_classes = n_classes
    
            # Project embeddings + first-pass scores
            self.input_proj = nn.Sequential(
                nn.Linear(d_input + d_scores, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
    
            # Metadata
            self.site_emb = nn.Embedding(n_sites, meta_dim)
            self.hour_emb = nn.Embedding(24, meta_dim)
            self.meta_proj = nn.Linear(2 * meta_dim, d_model)
    
            # Positional encoding
            self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
    
            # Single bidirectional SSM layer (lightweight)
            self.ssm_fwd = SelectiveSSM(d_model, d_state)
            self.ssm_bwd = SelectiveSSM(d_model, d_state)
            self.ssm_merge = nn.Linear(2 * d_model, d_model)
            self.ssm_norm = nn.LayerNorm(d_model)
            self.ssm_drop = nn.Dropout(dropout)
    
            # Output: per-class correction (additive)
            self.output_head = nn.Linear(d_model, n_classes)
    
            # Initialize output near zero (corrections start small)
            nn.init.zeros_(self.output_head.weight)
            nn.init.zeros_(self.output_head.bias)
    
        def forward(self, emb, first_pass_scores, site_ids=None, hours=None):
            # emb: (B, T, d_input), first_pass_scores: (B, T, n_classes)
            B, T, _ = emb.shape
    
            # Concatenate embeddings with first-pass scores
            x = torch.cat([emb, first_pass_scores], dim=-1)  # (B, T, d_input + d_scores)
            h = self.input_proj(x)
    
            # Add metadata
            if site_ids is not None and hours is not None:
                site_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
                hour_e = self.hour_emb(hours.clamp(0, 23))
                meta = self.meta_proj(torch.cat([site_e, hour_e], dim=-1))
                h = h + meta.unsqueeze(1)
    
            h = h + self.pos_enc[:, :T, :]
    
            # Bidirectional SSM
            residual = h
            h_f = self.ssm_fwd(h)
            h_b = self.ssm_bwd(h.flip(1)).flip(1)
            h = self.ssm_merge(torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = self.ssm_norm(h + residual)
    
            # Output correction
            correction = self.output_head(h)  # (B, T, n_classes)
            return correction
    
        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    # --- Train ResidualSSM on first-pass errors ---
    
    # Step 1: Compute first-pass scores on training data
    model.eval()
    with torch.no_grad():
        emb_train_t = torch.tensor(emb_files, dtype=torch.float32)
        logits_train_t = torch.tensor(logits_files, dtype=torch.float32)
        site_train_t = torch.tensor(site_ids_all, dtype=torch.long)
        hour_train_t = torch.tensor(hours_all, dtype=torch.long)
    
        proto_train_out, _, _ = model(emb_train_t, logits_train_t,
                                       site_ids=site_train_t, hours=hour_train_t)
        proto_train_scores = proto_train_out.numpy()  # (n_files, 12, 234)
    
    # MLP probe scores on training data (flat)
    mlp_train_scores_flat = np.zeros_like(scores_full_raw, dtype=np.float32)
    
    # Get prior-fused base for MLP
    train_base_scores, train_prior_scores = fuse_scores_with_tables(
        scores_full_raw,
        sites=meta_full["site"].to_numpy(),
        hours=meta_full["hour_utc"].to_numpy(),
        tables=final_prior_tables,
    )
    mlp_train_scores_flat = train_base_scores.copy()
    
    for cls_idx, clf in probe_models.items():
        X_cls = build_class_features(
            Z_FULL,
            raw_col=scores_full_raw[:, cls_idx],
            prior_col=train_prior_scores[:, cls_idx],
            base_col=train_base_scores[:, cls_idx],
        )
        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X_cls)[:, 1].astype(np.float32)
            pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
        else:
            pred = clf.decision_function(X_cls).astype(np.float32)
        alpha_p = float(CFG["frozen_best_probe"]["alpha"])
        mlp_train_scores_flat[:, cls_idx] = (1 - alpha_p) * train_base_scores[:, cls_idx] + alpha_p * pred
    
    # Reshape MLP scores to file-level
    mlp_train_scores_files, _ = reshape_to_files(mlp_train_scores_flat, meta_full)
    
    # First-pass ensemble (same formula as test-time)
    first_pass_files = (
        ENSEMBLE_WEIGHT_PROTO * proto_train_scores +
        (1 - ENSEMBLE_WEIGHT_PROTO) * mlp_train_scores_files
    ).astype(np.float32)
    
    # Step 2: Compute residuals (what the first pass got wrong)
    # Target: Y_FULL reshaped to files. Residual = target - sigmoid(first_pass)
    labels_float = labels_files.astype(np.float32)
    first_pass_probs = 1.0 / (1.0 + np.exp(-first_pass_files))
    residuals = labels_float - first_pass_probs  # in [-1, 1]
    
    print(f"First-pass training scores: {first_pass_files.shape}")
    print(f"Residuals: mean={residuals.mean():.4f}, std={residuals.std():.4f}, "
          f"abs_mean={np.abs(residuals).mean():.4f}")
    
    # Step 3: Train ResidualSSM
    res_cfg = CFG["residual_ssm"]
    res_model = ResidualSSM(
        d_input=emb_full.shape[1],
        d_scores=N_CLASSES,
        d_model=res_cfg["d_model"],
        d_state=res_cfg["d_state"],
        n_classes=N_CLASSES,
        n_windows=N_WINDOWS,
        dropout=res_cfg["dropout"],
        n_sites=CFG["proto_ssm"]["n_sites"],
        meta_dim=8,
    ).to(DEVICE)
    
    print(f"ResidualSSM parameters: {res_model.count_parameters():,}")
    
    # Train with MSE loss on residuals
    n_files = len(file_list)
    n_val = max(1, int(n_files * 0.15))
    perm = torch.randperm(n_files, generator=torch.Generator().manual_seed(123))
    val_i = perm[:n_val].numpy()
    train_i = perm[n_val:].numpy()
    
    emb_tr = torch.tensor(emb_files[train_i], dtype=torch.float32)
    fp_tr = torch.tensor(first_pass_files[train_i], dtype=torch.float32)
    res_tr = torch.tensor(residuals[train_i], dtype=torch.float32)
    site_tr = torch.tensor(site_ids_all[train_i], dtype=torch.long)
    hour_tr = torch.tensor(hours_all[train_i], dtype=torch.long)
    
    emb_va = torch.tensor(emb_files[val_i], dtype=torch.float32)
    fp_va = torch.tensor(first_pass_files[val_i], dtype=torch.float32)
    res_va = torch.tensor(residuals[val_i], dtype=torch.float32)
    site_va = torch.tensor(site_ids_all[val_i], dtype=torch.long)
    hour_va = torch.tensor(hours_all[val_i], dtype=torch.long)
    
    optimizer = torch.optim.AdamW(res_model.parameters(), lr=res_cfg["lr"], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=res_cfg["lr"],
        epochs=res_cfg["n_epochs"], steps_per_epoch=1,
        pct_start=0.1, anneal_strategy='cos'
    )
    
    best_val_loss = float('inf')
    best_state = None
    wait = 0
    
    t0_res = time.time()
    for epoch in range(res_cfg["n_epochs"]):
        res_model.train()
        correction = res_model(emb_tr, fp_tr, site_ids=site_tr, hours=hour_tr)
        loss = F.mse_loss(correction, res_tr)
    
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(res_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
        res_model.eval()
        with torch.no_grad():
            val_corr = res_model(emb_va, fp_va, site_ids=site_va, hours=hour_va)
            val_loss = F.mse_loss(val_corr, res_va)
    
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in res_model.state_dict().items()}
            wait = 0
        else:
            wait += 1
    
        if (epoch + 1) % 20 == 0:
            print(f"  ResidualSSM epoch {epoch+1}: train={loss.item():.6f} val={val_loss.item():.6f} wait={wait}")
    
        if wait >= res_cfg["patience"]:
            print(f"  ResidualSSM early stop at epoch {epoch+1}")
            break
    
    if best_state is not None:
        res_model.load_state_dict(best_state)
    
    res_time = time.time() - t0_res
    print(f"ResidualSSM training time: {res_time:.1f}s")
    print(f"Best val MSE: {best_val_loss:.6f}")
    
    # Verify correction magnitude
    res_model.eval()
    with torch.no_grad():
        all_corr = res_model(emb_train_t, torch.tensor(first_pass_files, dtype=torch.float32),
                             site_ids=site_train_t, hours=hour_train_t)
        corr_np = all_corr.numpy()
        print(f"Correction magnitude: mean_abs={np.abs(corr_np).mean():.4f}, max={np.abs(corr_np).max():.4f}")
    
    CORRECTION_WEIGHT = res_cfg["correction_weight"]
    print(f"Correction weight: {CORRECTION_WEIGHT}")
    LOGS["residual_ssm"] = {
        "params": res_model.count_parameters(),
        "train_time": res_time,
        "best_val_mse": best_val_loss,
        "correction_mean_abs": float(np.abs(corr_np).mean()),
        "correction_weight": CORRECTION_WEIGHT,
    }
    
else:
    print("SKIPPED ResidualSSM (wall time safety)")
    LOGS["residual_ssm"] = {"skipped": True, "wall_min": _wall_min}


# %% cell 28
# Cell 15 — Diagnostics
if MODE == "train":
    if grid_results is not None:
        best_row = grid_results.iloc[0]
        print(f"Best honest OOF probe AUC: {best_row['probe_oof_auc']:.6f}")
        print(f"Delta over honest OOF baseline: {best_row['delta']:.6f}")
else:
    print("Skipping train diagnostics in submit mode.")

# %% cell 29
# Cell 16 — Infer Perch on hidden test with embeddings
test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))

if len(test_paths) == 0:
    print(f"Hidden test not mounted. Dry-run on first {CFG['dryrun_n_files']} train soundscapes.")
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:CFG["dryrun_n_files"]]
else:
    print(f"Hidden test files: {len(test_paths)}")

# [MODIFIED - Opsi 3] Gunakan proxy_reduce terbaik dari grid search (bukan hardcode "max")
meta_test, scores_test_raw, emb_test = infer_perch_with_embeddings(
    test_paths,
    batch_files=CFG["batch_files"],
    verbose=CFG["verbose"],
    proxy_reduce=CFG["proxy_reduce"],  # hasil grid search, default "max"
)
print(f"proxy_reduce used for test inference: {CFG['proxy_reduce']!r}")

print("meta_test:", meta_test.shape)
print("scores_test_raw:", scores_test_raw.shape)
print("emb_test:", emb_test.shape)


# %% cell 30
# Score Fusion: ProtoSSM v4 + MLP Probes + Priors + TTA (OOF-optimized weight)

# --- Step 1: ProtoSSM v4 inference on test with TTA ---
emb_test_files, test_file_list = reshape_to_files(emb_test, meta_test)
logits_test_files, _ = reshape_to_files(scores_test_raw, meta_test)

# Build test metadata
test_site_ids, test_hours = get_file_metadata(meta_test, test_file_list, site_to_idx, CFG["proto_ssm"]["n_sites"])

emb_test_tensor = torch.tensor(emb_test_files, dtype=torch.float32)
logits_test_tensor = torch.tensor(logits_test_files, dtype=torch.float32)
test_site_tensor = torch.tensor(test_site_ids, dtype=torch.long)
test_hour_tensor = torch.tensor(test_hours, dtype=torch.long)

# V16: TTA — average predictions from shifted temporal sequences
model.eval()
tta_shifts = CFG.get("tta_shifts", [0])
if len(tta_shifts) > 1:
    print(f"Running TTA with shifts: {tta_shifts}")
    proto_scores = temporal_shift_tta(
        emb_test_files, logits_test_files, model,
        test_site_ids, test_hours, shifts=tta_shifts
    )
else:
    with torch.no_grad():
        proto_out, _, h_test = model(emb_test_tensor, logits_test_tensor,
                                      site_ids=test_site_tensor, hours=test_hour_tensor)
        proto_scores = proto_out.numpy()

# Flatten back to (n_rows, n_classes)
proto_scores_flat = proto_scores.reshape(-1, N_CLASSES).astype(np.float32)

print(f"ProtoSSM v4 test scores: {proto_scores_flat.shape}")
print(f"Score range: {proto_scores_flat.min():.3f} to {proto_scores_flat.max():.3f}")

# --- Step 2: Prior-fused base scores ---
test_base_scores, test_prior_scores = fuse_scores_with_tables(
    scores_test_raw,
    sites=meta_test["site"].to_numpy(),
    hours=meta_test["hour_utc"].to_numpy(),
    tables=final_prior_tables,
)

# --- Step 3: MLP probe scores ---
emb_test_scaled = emb_scaler.transform(emb_test)
Z_TEST = emb_pca.transform(emb_test_scaled).astype(np.float32)

mlp_scores = test_base_scores.copy()

for cls_idx, clf in probe_models.items():
    X_cls_test = build_class_features(
        Z_TEST,
        raw_col=scores_test_raw[:, cls_idx],
        prior_col=test_prior_scores[:, cls_idx],
        base_col=test_base_scores[:, cls_idx],
    )

    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X_cls_test)[:, 1].astype(np.float32)
        pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
    else:
        pred = clf.decision_function(X_cls_test).astype(np.float32)

    alpha = float(CFG["frozen_best_probe"]["alpha"])
    mlp_scores[:, cls_idx] = (1.0 - alpha) * test_base_scores[:, cls_idx] + alpha * pred

# --- Step 4: Ensemble fusion with OOF-optimized weight ---
print(f"\nUsing OOF-optimized ensemble weight: {ENSEMBLE_WEIGHT_PROTO:.2f}")

final_test_scores = (
    ENSEMBLE_WEIGHT_PROTO * proto_scores_flat +
    (1.0 - ENSEMBLE_WEIGHT_PROTO) * mlp_scores
).astype(np.float32)

# --- Step 5: Residual SSM correction (second pass) ---
if res_model is not None and CORRECTION_WEIGHT > 0:
    first_pass_test_files, _ = reshape_to_files(final_test_scores, meta_test)
    first_pass_test_t = torch.tensor(first_pass_test_files, dtype=torch.float32)

    res_model.eval()
    with torch.no_grad():
        test_correction = res_model(
            emb_test_tensor, first_pass_test_t,
            site_ids=test_site_tensor, hours=test_hour_tensor
        ).numpy()

    test_correction_flat = test_correction.reshape(-1, N_CLASSES).astype(np.float32)

    print(f"\nResidual correction: mean_abs={np.abs(test_correction_flat).mean():.4f}, "
          f"max={np.abs(test_correction_flat).max():.4f}")

    final_test_scores = final_test_scores + CORRECTION_WEIGHT * test_correction_flat
    print(f"Final scores (after residual): range [{final_test_scores.min():.3f}, {final_test_scores.max():.3f}]")
else:
    print("\nResidual correction: SKIPPED")

print(f"Final scores: {final_test_scores.shape}")

# --- Logging ---
test_logs = {}
window_scores = proto_scores.reshape(-1, N_WINDOWS, N_CLASSES).mean(axis=(0, 2))
test_logs["window_position_scores"] = window_scores.tolist()
print(f"\nWindow position mean scores: {[f'{s:.3f}' for s in window_scores]}")

if hasattr(model, 'class_to_family'):
    taxon_scores = defaultdict(list)
    idx_to_fam = {v: k for k, v in fam_to_idx.items()}
    for ci in range(N_CLASSES):
        fam_idx = class_to_family[ci]
        fam_name = idx_to_fam.get(fam_idx, f"group_{fam_idx}")
        taxon_scores[fam_name].append(float(proto_scores_flat[:, ci].mean()))

    test_logs["taxon_mean_scores"] = {k: float(np.mean(v)) for k, v in taxon_scores.items()}
    for k, v in sorted(taxon_scores.items(), key=lambda x: -np.mean(x[1]))[:5]:
        print(f"  {k}: mean_score={np.mean(v):.4f} (n_classes={len(v)})")

with torch.no_grad():
    p_norm = F.normalize(model.prototypes, dim=-1)
    cos_sim = torch.matmul(p_norm, p_norm.T)
    cos_sim.fill_diagonal_(0)
    top_sims = cos_sim.max(dim=1)[0].numpy()
    test_logs["prototype_max_similarity"] = {
        "mean": float(top_sims.mean()),
        "max": float(top_sims.max()),
        "min": float(top_sims.min()),
    }
    print(f"\nPrototype nearest-neighbor similarity: mean={top_sims.mean():.3f}, max={top_sims.max():.3f}")


LOGS["test_inference"] = test_logs

# %% cell 31
# Cell 18 — V17: Full post-processing pipeline

# V17: Optimize per-class thresholds from OOF (train mode only)
PER_CLASS_THRESHOLDS = np.full(N_CLASSES, 0.5, dtype=np.float32)
if MODE == "train" and oof_proto_flat is not None:
    print("Optimizing per-class thresholds from OOF...")
    best_thresholds, best_scores = optimize_per_class_thresholds(
        oof_proto_flat, Y_FULL, n_windows=N_WINDOWS, thresholds=CFG["threshold_grid"]
    )
    PER_CLASS_THRESHOLDS = best_thresholds.astype(np.float32)
    print(f"  Mean threshold: {best_thresholds.mean():.3f}")
    print(f"  Threshold range: [{best_thresholds.min():.2f}, {best_thresholds.max():.2f}]")
    print(f"  Mean F1 (proxy): {best_scores.mean():.3f}")
    
    # Show classes with extreme thresholds
    high_t = np.where(best_thresholds > 0.6)[0]
    low_t = np.where(best_thresholds < 0.4)[0]
    if len(high_t) > 0:
        print(f"  High threshold classes (>0.6): {len(high_t)}")
    if len(low_t) > 0:
        print(f"  Low threshold classes (<0.4): {len(low_t)}")
else:
    # Submit mode: use default 0.5 thresholds for all classes
    print("Using default per-class thresholds (0.5) for submit mode")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# --- Step 1: Per-taxon temperature scaling ---
temp_cfg = CFG["temperature"]
T_AVES = temp_cfg["aves"]
T_TEXTURE = temp_cfg["texture"]

class_temperatures = np.ones(N_CLASSES, dtype=np.float32) * T_AVES
for ci, label in enumerate(PRIMARY_LABELS):
    cn = CLASS_NAME_MAP.get(label, "Aves")
    if cn in TEXTURE_TAXA:
        class_temperatures[ci] = T_TEXTURE

print(f"\nPer-taxon temperature: Aves={T_AVES}, Texture={T_TEXTURE}")

scaled_scores = final_test_scores / class_temperatures[None, :]
probs = sigmoid(scaled_scores)

# --- Step 2: File-level confidence scaling ---
top_k = CFG.get("file_level_top_k", 0)
if top_k > 0:
    print(f"Applying file-level confidence scaling (top_k={top_k})")
    probs = file_level_confidence_scale(probs, n_windows=N_WINDOWS, top_k=top_k)
    probs = np.clip(probs, 0.0, 1.0)

# --- Step 3: V17 Rank-aware post-processing ---
if CFG.get("rank_aware_scale", False):
    power = CFG.get("rank_aware_power", 0.5)
    print(f"Applying rank-aware scaling (power={power})")
    probs = rank_aware_scaling(probs, n_windows=N_WINDOWS, power=power)
    probs = np.clip(probs, 0.0, 1.0)

# --- Step 4: V17 Delta shift smoothing ---
def adaptive_delta_smooth(probs, n_windows, base_alpha=0.20):
    n_files = probs.shape[0] // n_windows
    result = probs.copy()
    view = result.reshape(n_files, n_windows, -1)
    p_view = probs.reshape(n_files, n_windows, -1)
    for i in range(1, n_windows - 1):
        conf = p_view[:, i, :].max(axis=-1, keepdims=True)
        a = base_alpha * (1.0 - conf)
        neighbor_avg = (p_view[:, i-1, :] + p_view[:, i+1, :]) / 2.0
        view[:, i, :] = (1.0 - a) * p_view[:, i, :] + a * neighbor_avg
    return result.reshape(probs.shape)

alpha = CFG.get("delta_shift_alpha", 0.0)
if alpha > 0:
    print(f"Applying delta shift smoothing (alpha={alpha})")
    probs = adaptive_delta_smooth(probs, n_windows=N_WINDOWS, base_alpha=alpha)
    probs = np.clip(probs, 0.0, 1.0)
# --- Step 5: V17 Per-class threshold sharpening ---
print(f"Applying per-class threshold sharpening...")
probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS, n_windows=N_WINDOWS)

# --- Build submission ---
submission = pd.DataFrame(probs, columns=PRIMARY_LABELS)
submission.insert(0, "row_id", meta_test["row_id"].values)
submission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].astype(np.float32)

expected_rows = len(test_paths) * N_WINDOWS
assert len(submission) == expected_rows, f"Expected {expected_rows}, got {len(submission)}"
assert submission.columns.tolist() == ["row_id"] + PRIMARY_LABELS
assert not submission.isna().any().any()

submission.to_csv("submission.csv", index=False)

print("\nSaved submission.csv")
print("Submission shape:", submission.shape)
print(f"Final score range: {probs.min():.6f} to {probs.max():.6f}")
print(f"Final mean: {probs.mean():.4f}")
print(submission.iloc[:3, :8])

# %% cell 32
# Cell 19 — Final Diagnostics and Logging

# Save comprehensive logs
wall_time = time.time() - _WALL_START
LOGS["wall_time_seconds"] = wall_time
LOGS["experiment_id"] = "exp_038"
LOGS["experiment_name"] = "pantanal_onnx_fast_noalign"
LOGS["source_reference"] = "pantanal-distill-birdclef2026-onnx-0.93.ipynb"
LOGS["onnx_backend_active"] = bool(USE_ONNX_PERCH)
LOGS["onnx_path"] = str(PERCH_ONNX_PATH) if PERCH_ONNX_PATH is not None else None
LOGS["onnx_intra_op_threads"] = int(ONNX_INTRA_OP_THREADS)
LOGS["io_prefetch_workers"] = int(IO_PREFETCH_WORKERS)
LOGS["fast_tta_max_batch_size"] = int(FAST_TTA_MAX_BATCH_SIZE)
LOGS["onnx_tf_alignment"] = "disabled_for_submit_runtime"
LOGS["temperature"] = CFG["temperature"]
LOGS["ensemble_weight_proto"] = ENSEMBLE_WEIGHT_PROTO
LOGS["n_classes"] = N_CLASSES
LOGS["n_windows"] = N_WINDOWS
LOGS["cfg_proto_ssm"] = CFG["proto_ssm"]
LOGS["cfg_proto_ssm_train"] = {k: v for k, v in CFG["proto_ssm_train"].items() if not isinstance(v, (np.ndarray,))}
LOGS["v17_improvements"] = [
    "d_model_256", "n_ssm_layers_3", "cross_attention", "mixup", "focal_loss", "swa",
    "per_taxon_temperature", "file_level_scaling", "tta", "rank_aware_scaling",
    "delta_shift_smooth", "per_class_thresholds"
]
LOGS["per_class_thresholds"] = PER_CLASS_THRESHOLDS.tolist()

try:
    with open("/kaggle/working/exp_038_pantanal_onnx_fast_noalign_logs.json", "w") as f:
        json.dump(LOGS, f, indent=2, default=str)
    print("Saved /kaggle/working/exp_038_pantanal_onnx_fast_noalign_logs.json")
except Exception as e:
    print(f"Warning: could not save logs: {e}")

if MODE == "train":
    print("=== ProtoSSM v5 Training Summary ===")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"d_model: {CFG['proto_ssm']['d_model']}, n_ssm_layers: {CFG['proto_ssm']['n_ssm_layers']}")
    print(f"Wall time: {wall_time:.1f}s")
    print(f"OOF CV time: {LOGS.get('oof_time', 0):.1f}s")
    print(f"Final model training time: {LOGS.get('train_time_final', 0):.1f}s")
    print(f"Final train loss: {train_history['train_loss'][-1]:.4f}")
    print(f"Best val loss: {min(train_history['val_loss']):.4f}")
    print(f"Best val AUC: {max(train_history['val_auc']):.4f}")

    print(f"\n=== OOF Results ===")
    print(f"ProtoSSM OOF AUC: {LOGS.get('oof_auc_proto', 0):.4f}")
    print(f"MLP-only OOF AUC: {LOGS.get('mlp_only_auc', 0):.4f}")
    print(f"Ensemble OOF AUC: {LOGS.get('ensemble_auc', 0):.4f}")
    print(f"Optimized ProtoSSM weight: {ENSEMBLE_WEIGHT_PROTO:.2f}")

    with torch.no_grad():
        alphas = torch.sigmoid(model.fusion_alpha).numpy()
        high_proto = (alphas > 0.5).sum()
        high_perch = (alphas <= 0.5).sum()
        print(f"\nFusion alpha distribution (final model):")
        print(f"  ProtoSSM-dominant (alpha>0.5): {high_proto} classes")
        print(f"  Perch-dominant (alpha<=0.5): {high_perch} classes")

    print(f"\nPer-class calibration bias stats:")
    with torch.no_grad():
        cb = model.class_bias.numpy()
        print(f"  mean={cb.mean():.4f} std={cb.std():.4f} min={cb.min():.4f} max={cb.max():.4f}")

    print(f"\nMLP probes: {len(probe_models)} classes")

    if "per_class_auc_proto" in LOGS and LOGS["per_class_auc_proto"]:
        sorted_aucs = sorted(LOGS["per_class_auc_proto"].items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 10 classes by ProtoSSM OOF AUC:")
        for label, auc in sorted_aucs[:10]:
            print(f"  {label}: {auc:.4f}")
        print(f"\nBottom 10 classes by ProtoSSM OOF AUC:")
        for label, auc in sorted_aucs[-10:]:
            print(f"  {label}: {auc:.4f}")

    print("\nSubmission probability stats:")
    print(submission.iloc[:, 1:].stack().describe())
else:
    print("Submit mode completed.")
    print(f"ProtoSSM v5 parameters: {model.count_parameters():,}")
    print(f"Ensemble weight: {ENSEMBLE_WEIGHT_PROTO:.2f}")
    print(f"Wall time: {wall_time:.1f}s")
    print(f"ONNX backend active: {LOGS['onnx_backend_active']}")
    print(f"ONNX->TF alignment: {LOGS['onnx_tf_alignment']}")
    print(f"V17 improvements: {LOGS['v17_improvements']}")

