# %% cell 1
# ═══════════════════════════════════════════════════════════════
# Cell 0 — Environment (OFFLINE-SAFE)
# ═══════════════════════════════════════════════════════════════
# onnxscript NOT needed (only for export, optional)
# onnxruntime: try system → pip cache → wheel search → PyTorch fallback

import subprocess, sys, glob, os

def find_onnxruntime_wheel():
    """Search common Kaggle dataset paths for onnxruntime wheel."""
    search_roots = [
        "/kaggle/input",
    ]
    pattern = "**/onnxruntime*cp312*manylinux*.whl"
    for root in search_roots:
        for p in glob.glob(os.path.join(root, pattern), recursive=True):
            return p
    return None

_ORT_OK = False
try:
    import onnxruntime
    _ORT_OK = True
    print(f"onnxruntime pre-installed: {onnxruntime.__version__}")
except ImportError:
    pass

if not _ORT_OK:
    # Try pip install from offline cache
    try:
        subprocess.check_call([sys.executable, "install", "-q", "onnxruntime"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import onnxruntime
        _ORT_OK = True
        print(f"onnxruntime installed from cache: {onnxruntime.__version__}")
    except Exception:
        pass

if not _ORT_OK:
    # Try finding a wheel in attached datasets
    wheel = find_onnxruntime_wheel()
    if wheel:
        try:
            subprocess.check_call([sys.executable, "install", "-q", wheel],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import onnxruntime
            _ORT_OK = True
            print(f"onnxruntime installed from wheel: {onnxruntime.__version__}")
        except Exception as e:
            print(f"Wheel install failed: {e}")

if not _ORT_OK:
    print("⚠️  onnxruntime UNAVAILABLE — will use PyTorch fallback for inference")

# TensorFlow wheels (offline from dataset)
_tf_ok = False
try:
    import tensorflow as tf
    _tf_ok = True
    print(f"TensorFlow pre-installed: {tf.__version__}")
except ImportError:
    pass

if not _tf_ok:
    _tf_wheels = sorted(glob.glob("/kaggle/input/**/tensorflow-2.20*cp312*manylinux*.whl", recursive=True))
    _tb_wheels = sorted(glob.glob("/kaggle/input/**/tensorboard-2.20*py3*.whl", recursive=True))
    if _tb_wheels:
        subprocess.check_call([sys.executable, "install", "-q", "--no-deps", _tb_wheels[0]])
    if _tf_wheels:
        subprocess.check_call([sys.executable, "install", "-q", "--no-deps", _tf_wheels[0]])
        import tensorflow as tf
        _tf_ok = True
        print(f"TensorFlow installed from wheel: {tf.__version__}")
    else:
        print("⚠️  TensorFlow UNAVAILABLE — Perch inference will fail")

# ═══════════════════════════════════════════════════════════════
# Cell 1 — V19 Configuration
# ═══════════════════════════════════════════════════════════════
MODE = "submit"
assert MODE in {"train", "submit"}

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gc, json, re, time, warnings, shutil, joblib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

# Conditional ONNX imports
if _ORT_OK:
    import onnxruntime as ort
    import torch.onnx
    _USE_ONNX = True
else:
    _USE_ONNX = False

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

warnings.filterwarnings("ignore")
if _tf_ok:
    tf.experimental.numpy.experimental_enable_numpy_behavior()

_WALL_START = time.time()

BASE = Path("/kaggle/input/competitions/birdclef-2026")
MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
DEVICE = torch.device("cpu")

MY_MODEL_DIR = Path("/kaggle/input/datasets/dingjiarun/birds-train2")

CFG = {
    "mode": MODE,
    "verbose": MODE == "train",
    "batch_files": 16,
    "proxy_reduce": "max",
    "dryrun_n_files": 50 if MODE == "train" else 20,

    "full_cache_input_dir": Path("/kaggle/input/perch-meta"),
    "full_cache_work_dir": Path("/kaggle/working/perch_cache"),

    "best_fusion": {
        "lambda_event": 0.45,
        "lambda_texture": 1.10,
        "lambda_proxy_texture": 0.90,
        "smooth_texture": 0.30,
        "smooth_event": 0.10,
    },

    "proto_ssm": {
        "d_model": 320,
        "d_state": 32,
        "n_ssm_layers": 4,
        "dropout": 0.10,
        "n_prototypes": 2,
        "n_sites": 20,
        "meta_dim": 24,
        "use_cross_attn": True,
        "cross_attn_heads": 8,
        "use_multi_scale": True,
        "multi_scale_factors": [1, 2],
    },

    "proto_ssm_train": {
        "n_epochs": 80 if MODE == "train" else 60,
        "lr": 8e-4,
        "weight_decay": 1e-3,
        "val_ratio": 0.15,
        "patience": 20 if MODE == "train" else 12,
        "pos_weight_cap": 25.0,
        "distill_weight": 0.12,
        "proto_margin": 0.15,
        "label_smoothing": 0.025,
        "oof_n_splits": 5,
        "mixup_alpha": 0.4,
        "cutmix_prob": 0.3,
        "focal_gamma": 2.5,
        "swa_start_frac": 0.65,
        "swa_lr": 4e-4,
        "use_cosine_restart": True,
        "restart_period": 20,
        "warmup_epochs": 5,
        "grad_clip": 1.0,
    },

    "frozen_best_probe": {
        "pca_dim": 128,
        "min_pos": 5,
        "C": 0.75,
        "alpha": 0.45,
    },
    "probe_backend": "mlp",
    "mlp_params": {
        "hidden_layer_sizes": (256, 128),
        "activation": "relu",
        "max_iter": 500,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 20,
        "random_state": 42,
        "learning_rate_init": 5e-4,
        "alpha": 0.005,
    },

    "residual_ssm": {
        "d_model": 128, "d_state": 16, "n_ssm_layers": 2,
        "dropout": 0.08, "correction_weight": 0.30,
        "n_epochs": 40, "lr": 8e-4, "patience": 12,
    },

    "temperature": {"aves": 1.08, "texture": 0.92},

    "tta_shifts": [0, 1, -1, 2, -2],
    "file_level_top_k": 2,
    "rank_aware_scale": True,
    "rank_aware_power": 0.35,
    "delta_shift_alpha": 0.12,
    "threshold_grid": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],

    "use_per_class_temperature": True,
    "use_per_class_ensemble_weight": True,
    "use_isotonic_calibration": True,
    "use_frequency_based_thresholds": True,
    "adaptive_smooth": True,
    "smooth_min_conf": 0.7,
}

CFG["full_cache_work_dir"].mkdir(parents=True, exist_ok=True)

print(f"MODE = {MODE}")
print(f"PyTorch: {torch.__version__}")
print(f"ONNX runtime: {'✅ ' + ort.__version__ if _ORT_OK else '❌ (PyTorch fallback)'}")
print(f"TensorFlow: {'✅ ' + tf.__version__ if _tf_ok else '❌'}")
print(f"Competition dir: {'✅' if BASE.exists() else '❌'}")
print(f"Model dir: {'✅' if MODEL_DIR.exists() else '❌'}")
print("V19 Config loaded")

# %% cell 2
# ═══════════════════════════════════════════════════════════════
# Cell 2 — Data Loading
# ═══════════════════════════════════════════════════════════════
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
        return {"file_id": None, "site": None, "date": pd.NaT,
                "time_utc": None, "hour_utc": -1, "month": -1}
    file_id, site, ymd, hms = m.groups()
    dt = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
    return {"file_id": file_id, "site": site, "date": dt,
            "time_utc": hms, "hour_utc": int(hms[:2]),
            "month": int(dt.month) if pd.notna(dt) else -1}

def union_labels(series):
    return sorted(set(lbl for x in series for lbl in parse_soundscape_labels(x)))

sc_clean = (
    soundscape_labels
    .groupby(["filename", "start", "end"])["primary_label"]
    .apply(union_labels).reset_index(name="label_list")
)
sc_clean["start_sec"] = pd.to_timedelta(sc_clean["start"]).dt.total_seconds().astype(int)
sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
sc_clean["row_id"] = (sc_clean["filename"].str.replace(".ogg", "", regex=False)
                       + "_" + sc_clean["end_sec"].astype(str))

meta = sc_clean["filename"].apply(parse_soundscape_filename).apply(pd.Series)
sc_clean = pd.concat([sc_clean, meta], axis=1)

windows_per_file = sc_clean.groupby("filename").size()
full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
sc_clean["file_fully_labeled"] = sc_clean["filename"].isin(full_files)

label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}
Y_SC = np.zeros((len(sc_clean), N_CLASSES), dtype=np.uint8)
for i, labels in enumerate(sc_clean["label_list"]):
    idxs = [label_to_idx[lbl] for lbl in labels if lbl in label_to_idx]
    if idxs:
        Y_SC[i, idxs] = 1

full_truth = (sc_clean[sc_clean["file_fully_labeled"]]
              .sort_values(["filename", "end_sec"]).reset_index(drop=False))
Y_FULL_TRUTH = Y_SC[full_truth["index"].to_numpy()]

n_active = int((Y_FULL_TRUTH.sum(axis=0) > 0).sum())
print(f"sc_clean: {sc_clean.shape}, Full files: {len(full_files)}, "
      f"Trusted windows: {len(full_truth)}, Active classes: {n_active}")

# %% cell 3
# ═══════════════════════════════════════════════════════════════
# Cell 3 — Perch Model + Mapping (EXPANDED proxies for Aves)
# ═══════════════════════════════════════════════════════════════
BEST = CFG["best_fusion"]
birdclassifier = tf.saved_model.load(str(MODEL_DIR))
infer_fn = birdclassifier.signatures["serving_default"]

bc_labels = (
    pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
    .reset_index().rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
)
NO_LABEL_INDEX = len(bc_labels)

taxonomy["scientific_name_lookup"] = taxonomy["scientific_name"]
bc_lookup = bc_labels.rename(columns={"scientific_name": "scientific_name_lookup"})
mapping = taxonomy.merge(bc_lookup[["scientific_name_lookup", "bc_index"]],
                         on="scientific_name_lookup", how="left")
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

idx_active_texture = np.array([label_to_idx[c] for c in ACTIVE_CLASSES
                                if CLASS_NAME_MAP.get(c) in TEXTURE_TAXA], dtype=np.int32)
idx_active_event = np.array([label_to_idx[c] for c in ACTIVE_CLASSES
                              if CLASS_NAME_MAP.get(c) not in TEXTURE_TAXA], dtype=np.int32)
idx_mapped_active_texture = idx_active_texture[MAPPED_MASK[idx_active_texture]]
idx_mapped_active_event = idx_active_event[MAPPED_MASK[idx_active_event]]
idx_unmapped_active_texture = idx_active_texture[~MAPPED_MASK[idx_active_texture]]
idx_unmapped_active_event = idx_active_event[~MAPPED_MASK[idx_active_event]]
idx_unmapped_inactive = np.array([i for i in UNMAPPED_POS
                                   if PRIMARY_LABELS[i] not in ACTIVE_CLASSES], dtype=np.int32)

# ── V19: Genus proxies for ALL unmapped species (Aves + Amphibia + Insecta) ──
unmapped_df = mapping[mapping["bc_index"] == NO_LABEL_INDEX].copy()
proxy_map = {}
for _, row in unmapped_df.iterrows():
    target = row["primary_label"]
    sci = row["scientific_name"]
    genus = str(sci).split()[0]
    hits = bc_labels[bc_labels["scientific_name"].astype(str).str.match(
        rf"^{re.escape(genus)}\s", na=False)].copy()
    if len(hits) > 0:
        proxy_map[target] = {
            "bc_indices": hits["bc_index"].astype(int).tolist(),
        }

PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}  # V19: added Aves
SELECTED_PROXY_TARGETS = sorted([t for t in proxy_map.keys()
                                  if CLASS_NAME_MAP.get(t) in PROXY_TAXA])
selected_proxy_pos = np.array([label_to_idx[c] for c in SELECTED_PROXY_TARGETS], dtype=np.int32)
selected_proxy_pos_to_bc = {
    label_to_idx[target]: np.array(proxy_map[target]["bc_indices"], dtype=np.int32)
    for target in SELECTED_PROXY_TARGETS
}

idx_selected_proxy_texture = np.intersect1d(selected_proxy_pos, idx_active_texture)
idx_selected_proxy_event = np.intersect1d(selected_proxy_pos, idx_active_event)  # V19: proxy for Aves too
idx_selected_prioronly_texture = np.setdiff1d(idx_unmapped_active_texture, selected_proxy_pos)
idx_selected_prioronly_event = np.setdiff1d(idx_unmapped_active_event, selected_proxy_pos)

print(f"Mapped: {MAPPED_MASK.sum()}/{N_CLASSES}, Proxy targets: {len(SELECTED_PROXY_TARGETS)} "
      f"(texture={len(idx_selected_proxy_texture)}, event={len(idx_selected_proxy_event)})")

# %% cell 4
# ═══════════════════════════════════════════════════════════════
# Cell 4 — Utilities (improved metrics, smoothing, features)
# ═══════════════════════════════════════════════════════════════
def macro_auc_skip_empty(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    if keep.sum() == 0:
        return 0.0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")

def smooth_cols_fixed12(scores, cols, alpha=0.30):
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()
    s = scores.copy()
    assert len(s) % N_WINDOWS == 0
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    view[:, :, cols] = (1.0 - alpha) * x + 0.5 * alpha * (prev_x + next_x)
    return s

def smooth_events_fixed12(scores, cols, alpha=0.10):
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
    assert len(v) % N_WINDOWS == 0
    x = v.reshape(-1, N_WINDOWS)
    prev_v = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
    next_v = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
    mean_v = np.repeat(x.mean(axis=1), N_WINDOWS)
    max_v = np.repeat(x.max(axis=1), N_WINDOWS)
    std_v = np.repeat(x.std(axis=1), N_WINDOWS)
    return prev_v, next_v, mean_v, max_v, std_v

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ── V19: Improved loss functions ──
def build_class_freq_weights(Y_FULL, cap=10.0):
    pos_count = Y_FULL.sum(axis=0).astype(np.float32) + 1.0
    freq = pos_count / Y_FULL.shape[0]
    weights = 1.0 / (freq ** 0.5)
    weights = np.clip(weights, 1.0, cap)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

def species_focal_loss(logits, targets, class_weights, gamma=2.5, label_smoothing=0.025):
    targets_smooth = targets * (1 - label_smoothing) + label_smoothing / 2.0
    bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction="none")
    pt = torch.exp(-bce)
    focal = ((1 - pt) ** gamma) * bce
    w = class_weights.to(logits.device).unsqueeze(0)
    return (focal * w).mean()

def focal_bce_with_logits(logits, targets, gamma=2.5, pos_weight=None):
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(logits, targets,
                                                  pos_weight=pos_weight, reduction="none")
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    return ((1 - pt) ** gamma * bce).mean()

def mixup_cutmix(emb, logits, labels, alpha=0.4, cutmix_prob=0.3):
    B, T, D = emb.shape
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(B)
    if np.random.rand() < cutmix_prob:
        cut_len = max(1, int(T * (1 - lam)))
        cut_start = np.random.randint(0, T - cut_len + 1)
        new_emb = emb.clone()
        new_emb[:, cut_start:cut_start+cut_len, :] = emb[idx, cut_start:cut_start+cut_len, :]
        new_logits = logits.clone()
        new_logits[:, cut_start:cut_start+cut_len, :] = logits[idx, cut_start:cut_start+cut_len, :]
        lam_actual = 1.0 - cut_len / T
        new_labels = lam_actual * labels + (1-lam_actual) * labels[idx]
    else:
        new_emb = lam * emb + (1-lam) * emb[idx]
        new_logits = lam * logits + (1-lam) * logits[idx]
        new_labels = lam * labels + (1-lam) * labels[idx]
    return new_emb, new_logits, new_labels

def get_cosine_restart_scheduler(optimizer, restart_period=20):
    return CosineAnnealingWarmRestarts(optimizer, T_0=restart_period, T_mult=1, eta_min=1e-5)

# ── V19: Post-processing utilities ──
def file_level_confidence_scale(preds, n_windows=12, top_k=2):
    N, C = preds.shape
    assert N % n_windows == 0
    view = preds.reshape(-1, n_windows, C)
    sorted_view = np.sort(view, axis=1)
    top_k_mean = sorted_view[:, -top_k:, :].mean(axis=1, keepdims=True)
    # V19: Soft scaling with sqrt to reduce aggressiveness
    scale = np.sqrt(np.clip(top_k_mean, 0, 1))
    return (view * scale).reshape(N, C)

def rank_aware_scaling(scores, n_windows=12, power=0.35):
    N, C = scores.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view = scores.reshape(n_files, n_windows, C)
    file_max = view.max(axis=1, keepdims=True)
    scale = np.power(np.clip(file_max, 1e-6, 1.0), power)
    return (view * scale).reshape(N, C)

def adaptive_delta_smooth(probs, n_windows, base_alpha=0.12, min_conf=0.7):
    """V19: Only smooth UNCONFIDENT windows — preserve sharp call boundaries."""
    n_files = probs.shape[0] // n_windows
    result = probs.copy()
    view = result.reshape(n_files, n_windows, -1)
    p_view = probs.reshape(n_files, n_windows, -1)
    for i in range(1, n_windows - 1):
        conf = p_view[:, i, :].max(axis=-1, keepdims=True)
        # Adaptive alpha: high confidence → near zero smoothing
        a = base_alpha * np.clip(1.0 - conf / min_conf, 0, 1)
        neighbor_avg = (p_view[:, i-1, :] + p_view[:, i+1, :]) / 2.0
        view[:, i, :] = (1.0 - a) * p_view[:, i, :] + a * neighbor_avg
    return result.reshape(probs.shape)

def temporal_shift_tta_onnx(emb_files, logits_files, ort_session,
                             site_ids, hours, shifts=[0, 1, -1]):
    all_preds = []
    for shift in shifts:
        e = np.roll(emb_files, shift, axis=1) if shift != 0 else emb_files
        l = np.roll(logits_files, shift, axis=1) if shift != 0 else logits_files
        ort_inputs = {"emb": e.astype(np.float32), "logits": l.astype(np.float32),
                      "site": site_ids.astype(np.int64), "hour": hours.astype(np.int64)}
        pred = ort_session.run(None, ort_inputs)[0]
        if shift != 0:
            pred = np.roll(pred, -shift, axis=1)
        all_preds.append(pred)
    return np.mean(all_preds, axis=0)

# ── V19: Per-class calibration pipeline ──
def compute_frequency_based_thresholds(Y, n_windows=12, base=0.50, scale=0.15):
    """Frequency-heuristic thresholds: rare→low, common→high."""
    n_files = Y.shape[0] // n_windows
    file_y = Y.reshape(n_files, n_windows, -1).max(axis=1)
    pos_freq = file_y.mean(axis=0)
    # Log-linear mapping
    log_freq = np.log(pos_freq + 1e-6)
    log_max = np.log(pos_freq.max() + 1e-6)
    thresholds = base - scale * log_freq / (log_max + 1e-6)
    return np.clip(thresholds, 0.20, 0.70).astype(np.float32)

def compute_frequency_based_temperatures(Y, n_windows=12, t_rare=0.85, t_common=1.15):
    """Rare species → sharper (lower T), common → softer (higher T)."""
    n_files = Y.shape[0] // n_windows
    file_y = Y.reshape(n_files, n_windows, -1).max(axis=1)
    pos_freq = file_y.mean(axis=0)
    log_freq = np.log(pos_freq + 1e-6)
    log_max = np.log(pos_freq.max() + 1e-6)
    temps = t_rare + (t_common - t_rare) * log_freq / (log_max + 1e-6)
    return np.clip(temps, 0.80, 1.30).astype(np.float32)

def compute_frequency_based_ensemble_weights(Y, n_windows=12, w_rare=0.30, w_common=0.70):
    """Rare species → more prior/MLP weight, common → more ProtoSSM weight."""
    n_files = Y.shape[0] // n_windows
    file_y = Y.reshape(n_files, n_windows, -1).max(axis=1)
    pos_freq = file_y.mean(axis=0)
    log_freq = np.log(pos_freq + 1e-6)
    log_max = np.log(pos_freq.max() + 1e-6)
    weights = w_rare + (w_common - w_rare) * log_freq / (log_max + 1e-6)
    return np.clip(weights, 0.10, 0.90).astype(np.float32)

def apply_per_class_thresholds(scores, thresholds):
    """Sharpen predictions using per-class thresholds."""
    scaled = np.copy(scores)
    for c in range(len(thresholds)):
        t = thresholds[c]
        mask_above = scores[:, c] > t
        scaled[mask_above, c] = 0.5 + 0.5 * (scores[mask_above, c] - t) / (1 - t + 1e-8)
        scaled[~mask_above, c] = 0.5 * scores[~mask_above, c] / (t + 1e-8)
    return np.clip(scaled, 0, 1)

def calibrate_and_optimize_thresholds(oof_probs, Y_FULL, threshold_grid, n_windows=12):
    """Full isotonic + threshold optimization from OOF predictions."""
    n_samples, n_cls = oof_probs.shape
    thresholds = np.full(n_cls, 0.5, dtype=np.float32)
    n_files = n_samples // n_windows
    file_oof = oof_probs.reshape(n_files, n_windows, n_cls).max(axis=1)
    file_y = Y_FULL.reshape(n_files, n_windows, n_cls).max(axis=1)
    isotonic_models = {}

    for c in range(n_cls):
        y_true, y_prob = file_y[:, c], file_oof[:, c]
        if y_true.sum() < 3:
            thresholds[c] = 0.5
            continue
        try:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(y_prob, y_true)
            y_cal = ir.transform(y_prob)
            isotonic_models[c] = ir
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

    return thresholds, isotonic_models

print("✅ All V19 utilities defined")

# %% cell 5
# ═══════════════════════════════════════════════════════════════
# Cell 5 — Perch Inference (with expanded Aves proxies)
# ═══════════════════════════════════════════════════════════════
def read_soundscape_60s(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2: y = y.mean(axis=1)
    if sr != SR: raise ValueError(f"Unexpected SR {sr}")
    if len(y) < FILE_SAMPLES: y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    elif len(y) > FILE_SAMPLES: y = y[:FILE_SAMPLES]
    return y

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
        iterator = tqdm(iterator, total=(n_files + batch_files - 1) // batch_files, desc="Perch")

    for start in iterator:
        batch_paths = paths[start:start + batch_files]
        x = np.empty((len(batch_paths) * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        batch_row_start = write_row
        x_pos = 0

        for path in batch_paths:
            y = read_soundscape_60s(path)
            x[x_pos:x_pos + N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
            meta = parse_soundscape_filename(path.name)
            stem = path.stem
            row_ids[write_row:write_row+N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
            filenames[write_row:write_row+N_WINDOWS] = path.name
            sites[write_row:write_row+N_WINDOWS] = meta["site"]
            hours[write_row:write_row+N_WINDOWS] = int(meta["hour_utc"])
            x_pos += N_WINDOWS
            write_row += N_WINDOWS

        outputs = infer_fn(inputs=tf.convert_to_tensor(x))
        logits = outputs["label"].numpy().astype(np.float32, copy=False)
        emb = outputs["embedding"].numpy().astype(np.float32, copy=False)

        scores[batch_row_start:write_row, MAPPED_POS] = logits[:, MAPPED_BC_INDICES]
        embeddings[batch_row_start:write_row] = emb

        # V19: ALL genus proxies (Aves + Amphibia + Insecta)
        for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
            sub = logits[:, bc_idx_arr]
            proxy_score = sub.max(axis=1) if proxy_reduce == "max" else sub.mean(axis=1)
            scores[batch_row_start:write_row, pos] = proxy_score.astype(np.float32)

        del x, outputs, logits, emb
        gc.collect()

    return pd.DataFrame({"row_id": row_ids, "filename": filenames, "site": sites, "hour_utc": hours}), scores, embeddings

# %% cell 6
# ═══════════════════════════════════════════════════════════════
# Cell 6 — Cache Management
# ═══════════════════════════════════════════════════════════════
def resolve_full_cache_paths():
    candidates = [
        (CFG["full_cache_work_dir"] / "full_perch_meta.parquet",
         CFG["full_cache_work_dir"] / "full_perch_arrays.npz"),
        (Path("/kaggle/working/full_perch_meta.parquet"),
         Path("/kaggle/working/full_perch_arrays.npz")),
    ]
    if CFG["full_cache_input_dir"].exists():
        candidates.append((CFG["full_cache_input_dir"] / "full_perch_meta.parquet",
                           CFG["full_cache_input_dir"] / "full_perch_arrays.npz"))
    for meta_path, npz_path in candidates:
        if meta_path.exists() and npz_path.exists():
            return meta_path, npz_path
    return None, None

cache_meta, cache_npz = resolve_full_cache_paths()
if cache_meta is not None:
    print(f"Loading cache from {cache_meta}")
    meta_full = pd.read_parquet(cache_meta)
    arr = np.load(cache_npz)
    scores_full_raw = arr["scores_full_raw"].astype(np.float32)
    emb_full = arr["emb_full"].astype(np.float32)
else:
    print("No cache found, running Perch...")
    full_paths = [BASE / "train_soundscapes" / fn for fn in full_files]
    meta_full, scores_full_raw, emb_full = infer_perch_with_embeddings(
        full_paths, batch_files=CFG["batch_files"], verbose=CFG["verbose"])
    meta_full.to_parquet(CFG["full_cache_work_dir"] / "full_perch_meta.parquet", index=False)
    np.savez_compressed(CFG["full_cache_work_dir"] / "full_perch_arrays.npz",
                        scores_full_raw=scores_full_raw, emb_full=emb_full)

full_truth_aligned = full_truth.set_index("row_id").loc[meta_full["row_id"]].reset_index()
Y_FULL = Y_SC[full_truth_aligned["index"].to_numpy()]
assert np.all(full_truth_aligned["filename"].values == meta_full["filename"].values)

print(f"meta: {meta_full.shape}, scores: {scores_full_raw.shape}, emb: {emb_full.shape}, Y: {Y_FULL.shape}")

# %% cell 7
# ═══════════════════════════════════════════════════════════════
# Cell 7 — Prior Tables (ENHANCED: month prior + site×hour interaction)
# ═══════════════════════════════════════════════════════════════
def fit_prior_tables(prior_df, Y_prior):
    prior_df = prior_df.reset_index(drop=True)
    n_cls = Y_prior.shape[1]
    global_p = Y_prior.mean(axis=0).astype(np.float32)

    # Site
    site_keys = sorted(prior_df["site"].dropna().astype(str).unique().tolist())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_n = np.zeros(len(site_keys), dtype=np.float32)
    site_p = np.zeros((len(site_keys), n_cls), dtype=np.float32)
    for s in site_keys:
        i = site_to_i[s]
        mask = prior_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        site_p[i] = Y_prior[mask].mean(axis=0)

    # Hour
    hour_keys = sorted(prior_df["hour_utc"].dropna().astype(int).unique().tolist())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_n = np.zeros(len(hour_keys), dtype=np.float32)
    hour_p = np.zeros((len(hour_keys), n_cls), dtype=np.float32)
    for h in hour_keys:
        i = hour_to_i[h]
        mask = prior_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        hour_p[i] = Y_prior[mask].mean(axis=0)

    # Site-Hour interaction
    sh_to_i = {}
    sh_n_list, sh_p_list = [], []
    for (s, h), idx in prior_df.groupby(["site", "hour_utc"]).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n_list)
        idx = np.array(list(idx))
        sh_n_list.append(len(idx))
        sh_p_list.append(Y_prior[idx].mean(axis=0))
    sh_n = np.array(sh_n_list, dtype=np.float32)
    sh_p = np.stack(sh_p_list).astype(np.float32) if sh_p_list else np.zeros((0, n_cls), dtype=np.float32)

    # V19 NEW: Month prior
    month_keys = sorted(prior_df["month"].dropna().astype(int).unique().tolist())
    month_to_i = {m: i for i, m in enumerate(month_keys) if m > 0}
    month_n = np.zeros(len(month_keys), dtype=np.float32)
    month_p = np.zeros((len(month_keys), n_cls), dtype=np.float32)
    for m in month_keys:
        if m <= 0: continue
        i = month_to_i.get(m, -1)
        if i < 0: continue
        mask = prior_df["month"].astype(int).values == m
        month_n[i] = mask.sum()
        month_p[i] = Y_prior[mask].mean(axis=0)

    return {
        "global_p": global_p,
        "site_to_i": site_to_i, "site_n": site_n, "site_p": site_p,
        "hour_to_i": hour_to_i, "hour_n": hour_n, "hour_p": hour_p,
        "sh_to_i": sh_to_i, "sh_n": sh_n, "sh_p": sh_p,
        "month_to_i": month_to_i, "month_n": month_n, "month_p": month_p,
    }

def prior_logits_from_tables(sites, hours, months, tables, eps=1e-4):
    n = len(sites)
    n_cls = tables["global_p"].shape[0]
    p = np.repeat(tables["global_p"][None, :], n, axis=0).astype(np.float32, copy=True)

    site_idx = np.fromiter((tables["site_to_i"].get(str(s), -1) for s in sites), dtype=np.int32, count=n)
    hour_idx = np.fromiter((tables["hour_to_i"].get(int(h), -1) if int(h) >= 0 else -1 for h in hours), dtype=np.int32, count=n)
    sh_idx = np.fromiter((tables["sh_to_i"].get((str(s), int(h)), -1) if int(h) >= 0 else -1 for s, h in zip(sites, hours)), dtype=np.int32, count=n)

    # V19: Month prior
    if "month_to_i" in tables and months is not None:
        month_idx = np.fromiter((tables["month_to_i"].get(int(m), -1) if int(m) > 0 else -1 for m in months), dtype=np.int32, count=n)
        valid = month_idx >= 0
        if valid.any():
            nm = tables["month_n"][month_idx[valid]][:, None]
            wm = nm / (nm + 12.0)
            p[valid] = wm * tables["month_p"][month_idx[valid]] + (1.0 - wm) * p[valid]

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
                            smooth_event=BEST["smooth_event"],
                            months=None):
    scores = base_scores.copy()
    prior = prior_logits_from_tables(sites, hours, months, tables)

    # V19: Use proxy_event weight for Aves proxies
    lambda_proxy_event = 0.6  # V19 NEW

    if len(idx_mapped_active_event):
        scores[:, idx_mapped_active_event] += lambda_event * prior[:, idx_mapped_active_event]
    if len(idx_mapped_active_texture):
        scores[:, idx_mapped_active_texture] += lambda_texture * prior[:, idx_mapped_active_texture]
    if len(idx_selected_proxy_texture):
        scores[:, idx_selected_proxy_texture] += lambda_proxy_texture * prior[:, idx_selected_proxy_texture]
    # V19: Aves proxies get moderate event prior
    if len(idx_selected_proxy_event):
        scores[:, idx_selected_proxy_event] += lambda_proxy_event * prior[:, idx_selected_proxy_event]
    if len(idx_selected_prioronly_texture):
        scores[:, idx_selected_prioronly_texture] = lambda_texture * prior[:, idx_selected_prioronly_texture]
    if len(idx_selected_prioronly_event):
        scores[:, idx_selected_prioronly_event] = lambda_event * prior[:, idx_selected_prioronly_event]
    if len(idx_unmapped_inactive):
        scores[:, idx_unmapped_inactive] = -8.0

    scores = smooth_cols_fixed12(scores, idx_active_texture, alpha=smooth_texture)
    scores = smooth_events_fixed12(scores, idx_active_event, alpha=smooth_event)
    return scores.astype(np.float32, copy=False), prior

# %% cell 8
# ═══════════════════════════════════════════════════════════════
# Cell 8 — OOF Meta-Features (with month support)
# ═══════════════════════════════════════════════════════════════
def build_oof_base_prior(scores_full_raw, meta_full, sc_clean, Y_SC, n_splits=5, verbose=True):
    groups_full = meta_full["filename"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    oof_base = np.zeros_like(scores_full_raw, dtype=np.float32)
    oof_prior = np.zeros_like(scores_full_raw, dtype=np.float32)
    fold_id = np.full(len(meta_full), -1, dtype=np.int16)

    for fold, (tr_idx, va_idx) in enumerate(tqdm(list(gkf.split(scores_full_raw, groups=groups_full)),
                                                   desc="OOF folds", disable=not verbose)):
        val_files = set(meta_full.iloc[va_idx]["filename"].tolist())
        prior_mask = ~sc_clean["filename"].isin(val_files).values
        tables = fit_prior_tables(sc_clean.loc[prior_mask].reset_index(drop=True), Y_SC[prior_mask])
        months_va = meta_full.iloc[va_idx].apply(
            lambda r: parse_soundscape_filename(r["filename"]).get("month", -1), axis=1).values
        va_base, va_prior = fuse_scores_with_tables(
            scores_full_raw[va_idx], sites=meta_full.iloc[va_idx]["site"].to_numpy(),
            hours=meta_full.iloc[va_idx]["hour_utc"].to_numpy(), tables=tables, months=months_va)
        oof_base[va_idx] = va_base
        oof_prior[va_idx] = va_prior
        fold_id[va_idx] = fold + 1

    return oof_base, oof_prior, fold_id

OOF_META_CACHE = CFG["full_cache_work_dir"] / "full_oof_meta_features.npz"
if OOF_META_CACHE.exists():
    arr = np.load(OOF_META_CACHE)
    oof_base = arr["oof_base"].astype(np.float32)
    oof_prior = arr["oof_prior"].astype(np.float32)
else:
    oof_base, oof_prior, _ = build_oof_base_prior(scores_full_raw, meta_full, sc_clean, Y_SC, verbose=CFG["verbose"])
    np.savez_compressed(OOF_META_CACHE, oof_base=oof_base, oof_prior=oof_prior)

baseline_oof_auc = macro_auc_skip_empty(Y_FULL, oof_base)
print(f"OOF baseline AUC: {baseline_oof_auc:.6f}")

# %% cell 9
# ═══════════════════════════════════════════════════════════════
# Cell 9 — ProtoSSM v6: Multi-Scale Selective SSM Architecture
# ═══════════════════════════════════════════════════════════════
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv-1, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)
        h = torch.zeros(B, D, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dt_t = dt[:, t, :]
            dA = torch.exp(A[None, :, :] * dt_t[:, :, None])
            dB = dt_t[:, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            ys.append((h * C[:, t, None, :]).sum(-1))
        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


class TemporalCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model*2, d_model), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = residual + self.attn(x, x, x)[0]
        residual = x
        x = self.norm2(x)
        return residual + self.ffn(x)


class MultiScaleSSMBlock(nn.Module):
    """V19: Parallel SSM branches at different temporal scales."""
    def __init__(self, d_model, d_state, factors=[1, 2]):
        super().__init__()
        self.branches = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for f in factors:
            self.branches.append(SelectiveSSM(d_model, d_state))
            if f > 1:
                self.pools.append(nn.AvgPool1d(f, stride=f, padding=0))
                self.upsamples.append(nn.Upsample(scale_factor=f, mode='linear', align_corners=False))
            else:
                self.pools.append(nn.Identity())
                self.upsamples.append(nn.Identity())
        self.merge = nn.Linear(len(factors) * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.factors = factors

    def forward(self, x):
        B, T, D = x.shape
        branch_outs = []
        for i, (ssm, pool, upsample) in enumerate(zip(self.branches, self.pools, self.upsamples)):
            if self.factors[i] > 1:
                x_pooled = pool(x.transpose(1, 2)).transpose(1, 2)
                out = ssm(x_pooled)
                # Pad back to original length if needed
                if out.shape[1] < T:
                    pad = T - out.shape[1]
                    out = F.pad(out, (0, 0, 0, pad))
                out = out[:, :T, :]
                out = upsample(out.transpose(1, 2)).transpose(1, 2)
            else:
                out = ssm(x)
            branch_outs.append(out)
        merged = self.merge(torch.cat(branch_outs, dim=-1))
        return self.norm(x + merged)


class ProtoSSMv6(nn.Module):
    """V19: Multi-Scale ProtoSSM with dual prototypes, improved attention."""
    def __init__(self, d_input=1536, d_model=320, d_state=32, n_ssm_layers=4,
                 n_classes=234, n_windows=12, dropout=0.10, n_sites=20, meta_dim=24,
                 use_cross_attn=True, cross_attn_heads=8, n_prototypes=2,
                 use_multi_scale=True, multi_scale_factors=[1, 2]):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.n_prototypes = n_prototypes

        self.input_proj = nn.Sequential(nn.Linear(d_input, d_model), nn.LayerNorm(d_model),
                                          nn.GELU(), nn.Dropout(dropout))
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        # Multi-scale SSM layers
        self.use_multi_scale = use_multi_scale
        self.ssm_layers = nn.ModuleList()
        self.ssm_norms = nn.ModuleList()
        for i in range(n_ssm_layers):
            if use_multi_scale and i < 2:  # First 2 layers multi-scale, rest single-scale
                self.ssm_layers.append(MultiScaleSSMBlock(d_model, d_state, multi_scale_factors))
            else:
                # Bidirectional single-scale
                self.ssm_layers.append(nn.ModuleDict({
                    "fwd": SelectiveSSM(d_model, d_state),
                    "bwd": SelectiveSSM(d_model, d_state),
                    "merge": nn.Linear(2 * d_model, d_model),
                }))
            self.ssm_norms.append(nn.LayerNorm(d_model))
        self.ssm_drop = nn.Dropout(dropout)

        if use_cross_attn:
            self.cross_attn = TemporalCrossAttention(d_model, n_heads=cross_attn_heads, dropout=dropout)
        else:
            self.cross_attn = None

        # V19: Dual prototypes per class
        self.prototypes = nn.Parameter(torch.randn(n_classes * n_prototypes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

        # V19: Temporal position classifier (auxiliary)
        self.temporal_head = nn.Linear(d_model, n_windows)

        self.n_families = 0
        self.family_head = None

    def init_prototypes_from_data(self, embeddings, labels):
        with torch.no_grad():
            h = self.input_proj(embeddings)
            for c in range(self.n_classes):
                mask = labels[:, c] > 0.5
                if mask.sum() > 0:
                    mean_h = F.normalize(h[mask].mean(0), dim=0)
                    for k in range(self.n_prototypes):
                        self.prototypes.data[c * self.n_prototypes + k] = mean_h * (1.0 + 0.05 * k)

    def init_family_head(self, n_families, class_to_family):
        self.n_families = n_families
        self.family_head = nn.Linear(self.d_model, n_families)
        self.register_buffer('class_to_family', torch.tensor(class_to_family, dtype=torch.long))

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape
        h = self.input_proj(emb)
        h = h + self.pos_enc[:, :T, :]

        if site_ids is not None and hours is not None:
            meta = self.meta_proj(torch.cat([self.site_emb(site_ids), self.hour_emb(hours)], dim=-1))
            h = h + meta[:, None, :]

        for i, (layer, norm) in enumerate(zip(self.ssm_layers, self.ssm_norms)):
            residual = h
            if isinstance(layer, MultiScaleSSMBlock):
                h = layer(h)
            else:
                h_f = layer["fwd"](h)
                h_b = layer["bwd"](h.flip(1)).flip(1)
                h = layer["merge"](torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = norm(h + residual)

        if self.cross_attn is not None:
            h = self.cross_attn(h)

        h_temporal = h

        # Dual-prototype similarity: max over prototypes per class
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        temp = F.softplus(self.proto_temp)
        sim_all = torch.matmul(h_norm, p_norm.T) * temp  # (B, T, n_cls*n_proto)
        sim_all = sim_all.view(B, T, self.n_classes, self.n_prototypes)
        sim = sim_all.max(dim=-1)[0] + self.class_bias[None, None, :]

        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            species_logits = alpha * sim + (1 - alpha) * perch_logits
        else:
            species_logits = sim

        family_logits = None
        if self.family_head is not None:
            family_logits = self.family_head(h.mean(dim=1))

        # V19: Auxiliary temporal position prediction
        temporal_logits = self.temporal_head(h)  # (B, T, T)

        return species_logits, family_logits, h_temporal, temporal_logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

ssm_cfg = CFG["proto_ssm"]
test_m = ProtoSSMv6(d_model=ssm_cfg["d_model"], n_ssm_layers=ssm_cfg["n_ssm_layers"],
                     n_sites=ssm_cfg["n_sites"], meta_dim=ssm_cfg["meta_dim"],
                     n_prototypes=ssm_cfg["n_prototypes"],
                     use_multi_scale=ssm_cfg["use_multi_scale"],
                     multi_scale_factors=ssm_cfg["multi_scale_factors"])
print(f"ProtoSSM v6 params: {test_m.count_parameters():,}")
del test_m

# %% cell 10
# ═══════════════════════════════════════════════════════════════
# Cell 10 — Training Infrastructure (ALL improvements integrated)
# ═══════════════════════════════════════════════════════════════
def build_taxonomy_groups(taxonomy_df, primary_labels):
    for col in ["family", "order", "class_name"]:
        if col in taxonomy_df.columns:
            group_map = taxonomy_df.set_index("primary_label")[col].to_dict()
            break
    groups = sorted(set(group_map.values()))
    grp_to_idx = {g: i for i, g in enumerate(groups)}
    class_to_group = [grp_to_idx.get(group_map.get(l, "Unknown"), 0) for l in primary_labels]
    return len(groups), class_to_group, grp_to_idx

def reshape_to_files(flat_array, meta_df, n_windows=N_WINDOWS):
    filenames = meta_df["filename"].to_numpy()
    unique_files = list(dict.fromkeys(filenames))
    n_files = len(unique_files)
    assert len(flat_array) == n_files * n_windows
    return flat_array.reshape((n_files, n_windows) + flat_array.shape[1:]), unique_files

def build_site_mapping(meta_df):
    sites = meta_df["site"].unique().tolist()
    site_to_idx = {s: i+1 for i, s in enumerate(sites)}
    return site_to_idx, len(sites) + 1

def get_file_metadata(meta_df, file_list, site_to_idx, n_sites_max):
    file_to_row = {}
    for i, f in enumerate(meta_df["filename"].to_numpy()):
        if f not in file_to_row: file_to_row[f] = i
    site_ids = np.zeros(len(file_list), dtype=np.int64)
    hour_ids = np.zeros(len(file_list), dtype=np.int64)
    for fi, fname in enumerate(file_list):
        row = file_to_row.get(fname)
        if row is not None:
            site_ids[fi] = min(site_to_idx.get(meta_df.iloc[row]["site"], 0), n_sites_max - 1)
            hour_ids[fi] = int(meta_df.iloc[row]["hour_utc"]) % 24
    return site_ids, hour_ids

CLASS_WEIGHTS = build_class_freq_weights(Y_FULL_TRUTH)

def train_proto_ssm_single(model, emb_train, logits_train, labels_train,
                           site_ids_train, hours_train,
                           emb_val=None, logits_val=None, labels_val=None,
                           site_ids_val=None, hours_val=None,
                           file_families_train=None, file_families_val=None,
                           cfg=None, verbose=True):
    if cfg is None: cfg = CFG["proto_ssm_train"]
    n_epochs = cfg["n_epochs"]
    swa_start = int(n_epochs * cfg["swa_start_frac"])
    warmup = cfg.get("warmup_epochs", 5)

    labels_np = labels_train.copy()
    ls = cfg.get("label_smoothing", 0.0)
    if ls > 0: labels_np = labels_np * (1.0 - ls) + ls / 2.0

    has_val = emb_val is not None
    if has_val:
        emb_v = torch.tensor(emb_val, dtype=torch.float32)
        logits_v = torch.tensor(logits_val, dtype=torch.float32)
        labels_v = torch.tensor(labels_val, dtype=torch.float32)
        site_v = torch.tensor(site_ids_val, dtype=torch.long) if site_ids_val is not None else None
        hour_v = torch.tensor(hours_val, dtype=torch.long) if hours_val is not None else None

    pos_counts = torch.tensor(labels_np, dtype=torch.float32).sum(dim=(0, 1))
    total = labels_np.shape[0] * labels_np.shape[1]
    pos_weight = ((total - pos_counts) / (pos_counts + 1)).clamp(max=cfg["pos_weight_cap"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # V19: Cosine restart scheduler with warmup
    if cfg.get("use_cosine_restart"):
        scheduler = get_cosine_restart_scheduler(optimizer, cfg.get("restart_period", 20))
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg["lr"], epochs=n_epochs, steps_per_epoch=1,
            pct_start=0.1, anneal_strategy='cos')

    best_val_loss = float('inf')
    best_state = None
    wait = 0
    swa_state, swa_count = None, 0

    for epoch in range(n_epochs):
        # Warmup LR
        if epoch < warmup and not cfg.get("use_cosine_restart"):
            for pg in optimizer.param_groups:
                pg['lr'] = cfg['lr'] * (epoch + 1) / warmup

        # V19: Mixup+CutMix augmentation
        use_aug = epoch >= warmup and cfg.get("mixup_alpha", 0) > 0
        if use_aug:
            emb_aug, logits_aug, labels_aug = mixup_cutmix(
                emb_train, logits_train, labels_np,
                alpha=cfg["mixup_alpha"], cutmix_prob=cfg.get("cutmix_prob", 0.3))
        else:
            emb_aug, logits_aug, labels_aug = emb_train, logits_train, labels_np

        emb_tr = torch.tensor(emb_aug, dtype=torch.float32)
        logits_tr = torch.tensor(logits_aug, dtype=torch.float32)
        labels_tr = torch.tensor(labels_aug, dtype=torch.float32)
        site_tr = torch.tensor(site_ids_train, dtype=torch.long) if site_ids_train is not None else None
        hour_tr = torch.tensor(hours_train, dtype=torch.long) if hours_train is not None else None

        model.train()
        species_out, family_out, _, temporal_out = model(emb_tr, logits_tr, site_ids=site_tr, hours=hour_tr)

        # V19: Species-frequency-aware focal loss
        loss_main = species_focal_loss(
            species_out, labels_tr, CLASS_WEIGHTS,
            gamma=cfg["focal_gamma"], label_smoothing=0.0)

        # Knowledge distillation
        loss_distill = F.mse_loss(species_out, logits_tr)

        # V19: Temporal position auxiliary loss (self-supervised)
        loss_temporal = F.cross_entropy(
            temporal_out.reshape(-1, temporal_out.shape[-1]),
            torch.arange(temporal_out.shape[1], device=temporal_out.device).repeat(temporal_out.shape[0]))

        loss = loss_main + cfg["distill_weight"] * loss_distill + 0.05 * loss_temporal

        # Taxonomic auxiliary
        if family_out is not None and file_families_train is not None:
            fam_tr = torch.tensor(file_families_train, dtype=torch.float32)
            loss = loss + 0.1 * F.binary_cross_entropy_with_logits(family_out, fam_tr)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
        optimizer.step()
        if cfg.get("use_cosine_restart"):
            scheduler.step()
        else:
            scheduler.step()

        # SWA
        if epoch >= swa_start:
            if swa_state is None:
                swa_state = {k: v.clone() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                for k in swa_state: swa_state[k] += model.state_dict()[k]
                swa_count += 1

        # Validation
        model.eval()
        with torch.no_grad():
            if has_val:
                val_out, _, _, _ = model(emb_v, logits_v, site_ids=site_v, hours=hour_v)
                val_loss = F.binary_cross_entropy_with_logits(val_out, labels_v, pos_weight=pos_weight[None, None, :])
                val_pred = val_out.reshape(-1, val_out.shape[-1]).numpy()
                val_true = labels_v.reshape(-1, labels_v.shape[-1]).numpy()
                try: val_auc = macro_auc_skip_empty(val_true, val_pred)
                except: val_auc = 0.0
            else:
                val_loss, val_auc = loss, 0.0

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: train={loss.item():.4f} val={val_loss.item():.4f} "
                  f"auc={val_auc:.4f} lr={optimizer.param_groups[0]['lr']:.6f} wait={wait}")

        if wait >= cfg["patience"]:
            if verbose: print(f"  Early stop at {epoch+1}")
            break

    if swa_state is not None and swa_count >= 3:
        if verbose: print(f"  SWA: {swa_count} checkpoints")
        model.load_state_dict({k: v / swa_count for k, v in swa_state.items()})
    elif best_state is not None:
        model.load_state_dict(best_state)

    return model


def build_class_features(emb_proj, raw_col, prior_col, base_col):
    prev_base, next_base, mean_base, max_base, std_base = seq_features_1d(base_col)
    return np.concatenate([
        emb_proj, raw_col[:, None], prior_col[:, None], base_col[:, None],
        prev_base[:, None], next_base[:, None], mean_base[:, None],
        max_base[:, None], std_base[:, None],
        (base_col - mean_base)[:, None], (base_col - prev_base)[:, None],
        (raw_col * prior_col)[:, None], (raw_col * base_col)[:, None],
        (prior_col * base_col)[:, None],
    ], axis=1).astype(np.float32)

print("✅ V19 training infrastructure defined")

# %% cell 11
# ═══════════════════════════════════════════════════════════════
# Cell 11 — Main Training / Model Loading (BULLETPROOF submit)
# ═══════════════════════════════════════════════════════════════
import glob

# Prepare data
emb_files, file_list = reshape_to_files(emb_full, meta_full)
logits_files, _ = reshape_to_files(scores_full_raw, meta_full)
labels_files, _ = reshape_to_files(Y_FULL, meta_full)

n_families, class_to_family, fam_to_idx = build_taxonomy_groups(taxonomy, PRIMARY_LABELS)
site_to_idx, n_sites_mapped = build_site_mapping(meta_full)
site_ids_all, hours_all = get_file_metadata(meta_full, file_list, site_to_idx, CFG["proto_ssm"]["n_sites"])

file_families = np.zeros((len(file_list), n_families), dtype=np.float32)
for fi in range(len(file_list)):
    for ci in np.where(labels_files[fi].sum(axis=0) > 0)[0]:
        file_families[fi, class_to_family[ci]] = 1.0

# PCA for probes
emb_scaler = StandardScaler()
Z_FULL = PCA(n_components=min(int(CFG["frozen_best_probe"]["pca_dim"]), 707, 1536)).fit_transform(
    emb_scaler.fit_transform(emb_full)).astype(np.float32)

# Final prior tables
final_prior_tables = fit_prior_tables(sc_clean.reset_index(drop=True), Y_SC)

ENSEMBLE_WEIGHT_PROTO = 0.60
probe_models = {}
oof_proto_flat = None
PER_CLASS_THRESHOLDS = None
PER_CLASS_TEMPS = None
PER_CLASS_WEIGHTS = None
ISOTONIC_MODELS = None
_PROTOSSM_AVAILABLE = False  # Track if we actually loaded a model

MY_MODEL_DIR = Path("/kaggle/input/datasets/dingjiarun/birds-train2")

if MODE == "train":
    # ── OOF Cross-Validation ──
    file_groups = np.array([f.split("_")[3] if len(f.split("_")) > 3 else f for f in file_list])
    n_splits = min(CFG["proto_ssm_train"]["oof_n_splits"], len(set(file_groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros((len(file_list), N_WINDOWS, N_CLASSES), dtype=np.float32)

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(np.zeros(len(file_list)), groups=file_groups)):
        print(f"\n--- Fold {fold_i+1}/{n_splits} ---")
        ssm_cfg = CFG["proto_ssm"]
        fold_model = ProtoSSMv6(
            d_input=emb_files.shape[2], d_model=ssm_cfg["d_model"], d_state=ssm_cfg["d_state"],
            n_ssm_layers=ssm_cfg["n_ssm_layers"], n_classes=N_CLASSES, n_windows=N_WINDOWS,
            dropout=ssm_cfg["dropout"], n_sites=ssm_cfg["n_sites"], meta_dim=ssm_cfg["meta_dim"],
            n_prototypes=ssm_cfg["n_prototypes"],
            use_multi_scale=ssm_cfg["use_multi_scale"],
            multi_scale_factors=ssm_cfg["multi_scale_factors"]).to(DEVICE)

        fold_model.init_prototypes_from_data(
            torch.tensor(emb_files[train_idx].reshape(-1, emb_files.shape[2]), dtype=torch.float32),
            torch.tensor(labels_files[train_idx].reshape(-1, N_CLASSES), dtype=torch.float32))
        fold_model.init_family_head(n_families, class_to_family)

        fold_model = train_proto_ssm_single(
            fold_model, emb_files[train_idx], logits_files[train_idx],
            labels_files[train_idx].astype(np.float32),
            site_ids_train=site_ids_all[train_idx], hours_train=hours_all[train_idx],
            emb_val=emb_files[val_idx], logits_val=logits_files[val_idx],
            labels_val=labels_files[val_idx].astype(np.float32),
            site_ids_val=site_ids_all[val_idx], hours_val=hours_all[val_idx],
            file_families_train=file_families[train_idx],
            file_families_val=file_families[val_idx],
            verbose=CFG["verbose"])

        fold_model.eval()
        with torch.no_grad():
            out = fold_model(
                torch.tensor(emb_files[val_idx], dtype=torch.float32),
                torch.tensor(logits_files[val_idx], dtype=torch.float32),
                site_ids=torch.tensor(site_ids_all[val_idx], dtype=torch.long),
                hours=torch.tensor(hours_all[val_idx], dtype=torch.long))[0]
            oof_preds[val_idx] = out.numpy()

    oof_proto_flat = oof_preds.reshape(-1, N_CLASSES)
    y_flat = labels_files.reshape(-1, N_CLASSES).astype(float)
    print(f"ProtoSSM OOF AUC: {macro_auc_skip_empty(y_flat, oof_proto_flat):.4f}")

    # ── Train final model ──
    ssm_cfg = CFG["proto_ssm"]
    model = ProtoSSMv6(
        d_input=emb_files.shape[2], d_model=ssm_cfg["d_model"], d_state=ssm_cfg["d_state"],
        n_ssm_layers=ssm_cfg["n_ssm_layers"], n_classes=N_CLASSES, n_windows=N_WINDOWS,
        dropout=ssm_cfg["dropout"], n_sites=ssm_cfg["n_sites"], meta_dim=ssm_cfg["meta_dim"],
        n_prototypes=ssm_cfg["n_prototypes"],
        use_multi_scale=ssm_cfg["use_multi_scale"],
        multi_scale_factors=ssm_cfg["multi_scale_factors"]).to(DEVICE)
    model.init_prototypes_from_data(torch.tensor(emb_full, dtype=torch.float32),
                                     torch.tensor(Y_FULL, dtype=torch.float32))
    model.init_family_head(n_families, class_to_family)
    model = train_proto_ssm_single(model, emb_files, logits_files,
                                    labels_files.astype(np.float32),
                                    site_ids_all, hours_all, verbose=True)
    _PROTOSSM_AVAILABLE = True

    # ── Export ONNX ──
    if _USE_ONNX:
        model.eval()
        torch.onnx.export(model,
            (torch.randn(2, 12, 1536), torch.randn(2, 12, 234),
             torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long)),
            "protossm_v6.onnx",
            input_names=["emb", "logits", "site", "hour"],
            output_names=["species_out", "family_out", "h_temporal", "temporal_out"],
            dynamic_axes={k: {0: "batch_size"} for k in
                          ["emb", "logits", "site", "hour",
                           "species_out", "family_out", "h_temporal", "temporal_out"]},
            opset_version=17)
        print("✅ ONNX exported")
    else:
        torch.save(model.state_dict(), "protossm_state.pt")
        print("✅ PyTorch state saved")

    # ── Per-class calibration from OOF ──
    print("\n=== Per-Class Calibration ===")
    PER_CLASS_THRESHOLDS, ISOTONIC_MODELS = calibrate_and_optimize_thresholds(
        oof_proto_flat, Y_FULL, CFG["threshold_grid"])
    PER_CLASS_TEMPS = compute_frequency_based_temperatures(Y_FULL)
    PER_CLASS_WEIGHTS = compute_frequency_based_ensemble_weights(Y_FULL)

    # ── Optimize ensemble weight ──
    PROBE_IDX = np.where(Y_FULL.sum(axis=0) >= int(CFG["frozen_best_probe"]["min_pos"]))[0]
    for cls_idx in tqdm(PROBE_IDX, desc="MLP probes"):
        y = Y_FULL[:, cls_idx]
        if y.sum() == 0 or y.sum() == len(y): continue
        X = build_class_features(Z_FULL, scores_full_raw[:, cls_idx],
                                  oof_prior[:, cls_idx], oof_base[:, cls_idx])
        n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
        if n_pos > 0 and n_neg > n_pos:
            repeat = max(1, n_neg // n_pos)
            X = np.vstack([X, np.tile(X[y == 1], (repeat, 1))])
            y_aug = np.concatenate([y, np.ones(int(y.sum()) * repeat, dtype=y.dtype)])
        else:
            y_aug = y
        probe_models[cls_idx] = MLPClassifier(**CFG["mlp_params"]).fit(X, y_aug)

    oof_mlp = oof_base.copy()
    for cls_idx, clf in probe_models.items():
        X = build_class_features(Z_FULL, scores_full_raw[:, cls_idx],
                                  oof_prior[:, cls_idx], oof_base[:, cls_idx])
        prob = clf.predict_proba(X)[:, 1]
        pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
        oof_mlp[:, cls_idx] = ((1 - CFG["frozen_best_probe"]["alpha"]) * oof_base[:, cls_idx]
                               + CFG["frozen_best_probe"]["alpha"] * pred)

    best_auc, best_w = 0, 0.5
    for w in np.arange(0.0, 1.05, 0.05):
        auc = macro_auc_skip_empty(y_flat, w * oof_proto_flat + (1 - w) * oof_mlp)
        if auc > best_auc: best_auc, best_w = auc, w
    ENSEMBLE_WEIGHT_PROTO = best_w
    print(f"Ensemble: proto={best_w:.2f}, AUC={best_auc:.4f}")

    # Save everything
    joblib.dump(probe_models, "mlp_probes.pkl")
    joblib.dump(ISOTONIC_MODELS, "isotonic_models.pkl")
    np.savez("calibration_params.npz",
             thresholds=PER_CLASS_THRESHOLDS, temperatures=PER_CLASS_TEMPS,
             weights=PER_CLASS_WEIGHTS)
    with open("ensemble_weight.json", "w") as f:
        json.dump({"weight": float(ENSEMBLE_WEIGHT_PROTO)}, f)
    print("✅ All artifacts saved")

else:
    # ═══════════════════════════════════════════════════════════
    # SUBMIT MODE: Robust loading with broad search + graceful fallback
    # ═══════════════════════════════════════════════════════════
    print("=== SUBMIT MODE: Loading pre-trained models ===")

    # ── 1. Find ONNX or PyTorch model (search broadly) ──
    _found_onnx = None
    _found_pt = None

    # Check MY_MODEL_DIR first
    if MY_MODEL_DIR.exists():
        for fname in ["protossm_v6.onnx", "protossm_v4.onnx"]:
            if (MY_MODEL_DIR / fname).exists():
                _found_onnx = MY_MODEL_DIR / fname
                break
        if (MY_MODEL_DIR / "protossm_state.pt").exists():
            _found_pt = MY_MODEL_DIR / "protossm_state.pt"

    # If not found, recursive search across ALL attached datasets
    if _found_onnx is None and _found_pt is None:
        print("  Searching /kaggle/input for model files...")
        for p in glob.glob("/kaggle/input/**/protossm*.onnx", recursive=True):
            _found_onnx = Path(p)
            print(f"  Found ONNX: {p}")
            break
        if _found_onnx is None:
            for p in glob.glob("/kaggle/input/**/protossm*.pt", recursive=True):
                _found_pt = Path(p)
                print(f"  Found PyTorch: {p}")
                break

    # Copy to working dir
    if _found_onnx:
        shutil.copy(_found_onnx, _found_onnx.name)
        data_file = _found_onnx.parent / f"{_found_onnx.name}.data"
        if data_file.exists():
            shutil.copy(data_file, data_file.name)
        print(f"✅ Loaded {_found_onnx.name}")
        _PROTOSSM_AVAILABLE = True
    elif _found_pt:
        shutil.copy(_found_pt, "protossm_state.pt")
        print(f"✅ Loaded protossm_state.pt (PyTorch fallback)")
        _PROTOSSM_AVAILABLE = True
    else:
        print("⚠️  NO ProtoSSM model found anywhere!")
        print("   Will skip ProtoSSM and use MLP+Prior baseline only.")

    # ── 2. Load MLP probes ──
    _probes_loaded = False
    for probe_path in [MY_MODEL_DIR / "mlp_probes.pkl", "mlp_probes.pkl"]:
        if Path(probe_path).exists():
            probe_models = joblib.load(probe_path)
            _probes_loaded = True
            print(f"✅ {len(probe_models)} MLP probes loaded")
            break
    if not _probes_loaded:
        for p in glob.glob("/kaggle/input/**/mlp_probes.pkl", recursive=True):
            probe_models = joblib.load(p)
            _probes_loaded = True
            print(f"✅ {len(probe_models)} MLP probes found at {p}")
            break
    if not _probes_loaded:
        print("⚠️  No MLP probes found — will use prior-fused base only.")

    # ── 3. Load ensemble weight ──
    _weight_loaded = False
    for wp in [MY_MODEL_DIR / "ensemble_weight.json", "ensemble_weight.json"]:
        if Path(wp).exists():
            with open(wp) as f:
                ENSEMBLE_WEIGHT_PROTO = json.load(f)["weight"]
            _weight_loaded = True
            break
    if not _weight_loaded:
        for p in glob.glob("/kaggle/input/**/ensemble_weight.json", recursive=True):
            with open(p) as f:
                ENSEMBLE_WEIGHT_PROTO = json.load(f)["weight"]
            _weight_loaded = True
            break
    if not _weight_loaded:
        if not _PROTOSSM_AVAILABLE:
            ENSEMBLE_WEIGHT_PROTO = 0.0  # No model = 100% MLP/prior
        print(f"⚠️  No weight file, using default: {ENSEMBLE_WEIGHT_PROTO:.2f}")
    else:
        print(f"✅ Ensemble weight: {ENSEMBLE_WEIGHT_PROTO:.2f}")

    # ── 4. Load per-class calibration ──
    _cal_loaded = False
    for cp in [MY_MODEL_DIR / "calibration_params.npz", "calibration_params.npz"]:
        if Path(cp).exists():
            cal = np.load(cp)
            PER_CLASS_THRESHOLDS = cal["thresholds"]
            PER_CLASS_TEMPS = cal["temperatures"]
            PER_CLASS_WEIGHTS = cal["weights"]
            _cal_loaded = True
            break
    if not _cal_loaded:
        for p in glob.glob("/kaggle/input/**/calibration_params.npz", recursive=True):
            cal = np.load(p)
            PER_CLASS_THRESHOLDS = cal["thresholds"]
            PER_CLASS_TEMPS = cal["temperatures"]
            PER_CLASS_WEIGHTS = cal["weights"]
            _cal_loaded = True
            break

    if _cal_loaded:
        print(f"✅ Calibration loaded (thresh mean={PER_CLASS_THRESHOLDS.mean():.3f})")
    else:
        print("⚠️  No calibration file — computing frequency-based defaults")
        PER_CLASS_THRESHOLDS = compute_frequency_based_thresholds(Y_FULL)
        PER_CLASS_TEMPS = compute_frequency_based_temperatures(Y_FULL)
        PER_CLASS_WEIGHTS = compute_frequency_based_ensemble_weights(Y_FULL)
        print(f"   Thresholds: [{PER_CLASS_THRESHOLDS.min():.2f}, {PER_CLASS_THRESHOLDS.max():.2f}]")

    # ── 5. Load isotonic models (optional) ──
    for ip in [MY_MODEL_DIR / "isotonic_models.pkl", "isotonic_models.pkl"]:
        if Path(ip).exists():
            ISOTONIC_MODELS = joblib.load(ip)
            print(f"✅ {len(ISOTONIC_MODELS)} isotonic models")
            break
    else:
        for p in glob.glob("/kaggle/input/**/isotonic_models.pkl", recursive=True):
            ISOTONIC_MODELS = joblib.load(p)
            print(f"✅ {len(ISOTONIC_MODELS)} isotonic models found at {p}")
            break
    if 'ISOTONIC_MODELS' not in dir() or not ISOTONIC_MODELS:
        ISOTONIC_MODELS = {}

    # ── Summary ──
    print(f"\n{'─'*40}")
    print(f"ProtoSSM:  {'✅' if _PROTOSSM_AVAILABLE else '❌ (skipped)'}")
    print(f"MLP Probes: {'✅' if _probes_loaded else '❌'}")
    print(f"Calibration: {'✅' if _cal_loaded else '⚠️ frequency-defaults'}")
    if not _PROTOSSM_AVAILABLE and not _probes_loaded:
        print("\n🚨 CRITICAL: No models loaded! Submission will be prior-only baseline.")

# %% cell 12
# ═══════════════════════════════════════════════════════════════
# Cell 12 — Test Inference (ONNX or PyTorch fallback)
# ═══════════════════════════════════════════════════════════════
test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
if len(test_paths) == 0:
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:CFG["dryrun_n_files"]]
    print(f"Dry-run: {len(test_paths)} files")
else:
    print(f"Test files: {len(test_paths)}")

meta_test, scores_test_raw, emb_test = infer_perch_with_embeddings(
    test_paths, batch_files=CFG["batch_files"], verbose=CFG["verbose"],
    proxy_reduce=CFG["proxy_reduce"])

emb_test_files, test_file_list = reshape_to_files(emb_test, meta_test)
logits_test_files, _ = reshape_to_files(scores_test_raw, meta_test)
test_site_ids, test_hours = get_file_metadata(
    meta_test, test_file_list, site_to_idx, CFG["proto_ssm"]["n_sites"])

tta_shifts = CFG["tta_shifts"]

if _USE_ONNX:
    # ── ONNX path ──
    onnx_file = None
    for fname in ["protossm_v6.onnx", "protossm_v4.onnx"]:
        if Path(fname).exists():
            onnx_file = fname
            break
    if onnx_file is None:
        for fname in ["protossm_v6.onnx", "protossm_v4.onnx"]:
            src = MY_MODEL_DIR / fname
            if src.exists():
                shutil.copy(src, fname)
                if (MY_MODEL_DIR / f"{fname}.data").exists():
                    shutil.copy(MY_MODEL_DIR / f"{fname}.data", f"{fname}.data")
                onnx_file = fname
                break

    if onnx_file:
        print(f"Loading {onnx_file} via ONNX Runtime...")
        ort_session = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
        if len(tta_shifts) > 1:
            proto_scores = temporal_shift_tta_onnx(
                emb_test_files, logits_test_files, ort_session,
                test_site_ids, test_hours, shifts=tta_shifts)
        else:
            proto_scores = ort_session.run(None, {
                "emb": emb_test_files.astype(np.float32),
                "logits": logits_test_files.astype(np.float32),
                "site": test_site_ids.astype(np.int64),
                "hour": test_hours.astype(np.int64)})[0]
    else:
        print("⚠️ No ONNX file found, falling back to PyTorch")
        _USE_ONNX = False

if not _USE_ONNX:
    # ── PyTorch fallback path ──
    print("Loading model via PyTorch...")

    # Find saved model state dict
    _state_path = None
    for p in [MY_MODEL_DIR / "protossm_state.pt", "protossm_state.pt"]:
        if Path(p).exists():
            _state_path = p
            break

    ssm_cfg = CFG["proto_ssm"]
    model = ProtoSSMv6(
        d_input=emb_files.shape[2], d_model=ssm_cfg["d_model"], d_state=ssm_cfg["d_state"],
        n_ssm_layers=ssm_cfg["n_ssm_layers"], n_classes=N_CLASSES, n_windows=N_WINDOWS,
        dropout=ssm_cfg["dropout"], n_sites=ssm_cfg["n_sites"], meta_dim=ssm_cfg["meta_dim"],
        n_prototypes=ssm_cfg["n_prototypes"],
        use_multi_scale=ssm_cfg["use_multi_scale"],
        multi_scale_factors=ssm_cfg["multi_scale_factors"]).to(DEVICE)

    if _state_path:
        model.load_state_dict(torch.load(_state_path, map_location="cpu", weights_only=True))
        print(f"Loaded state from {_state_path}")
    else:
        print("⚠️ No state dict found — using randomly initialized model (will produce garbage!)")

    model.eval()

    def _pytorch_tta(emb_f, logits_f, mdl, s_ids, h_ids, shifts):
        all_preds = []
        for shift in shifts:
            e = np.roll(emb_f, shift, axis=1) if shift else emb_f
            l = np.roll(logits_f, shift, axis=1) if shift else logits_f
            with torch.no_grad():
                out = mdl(torch.tensor(e, dtype=torch.float32),
                          torch.tensor(l, dtype=torch.float32),
                          site_ids=torch.tensor(s_ids, dtype=torch.long),
                          hours=torch.tensor(h_ids, dtype=torch.long))[0]
                p = out.numpy()
            if shift:
                p = np.roll(p, -shift, axis=1)
            all_preds.append(p)
        return np.mean(all_preds, axis=0)

    proto_scores = _pytorch_tta(
        emb_test_files, logits_test_files, model,
        test_site_ids, test_hours, shifts=tta_shifts)

proto_scores_flat = proto_scores.reshape(-1, N_CLASSES).astype(np.float32)
print(f"ProtoSSM scores: {proto_scores_flat.shape}, "
      f"range=[{proto_scores_flat.min():.2f}, {proto_scores_flat.max():.2f}]")

# ── Prior-fused base + MLP probes ──
months_test = meta_test.apply(
    lambda r: parse_soundscape_filename(r["filename"]).get("month", -1), axis=1).values
test_base, test_prior = fuse_scores_with_tables(
    scores_test_raw, meta_test["site"].to_numpy(),
    meta_test["hour_utc"].to_numpy(), final_prior_tables, months=months_test)

_pca_for_test = PCA(n_components=Z_FULL.shape[1])
_pca_for_test.fit(emb_scaler.transform(emb_full))
Z_TEST = _pca_for_test.transform(emb_scaler.transform(emb_test)).astype(np.float32)

mlp_scores = test_base.copy()
for cls_idx, clf in probe_models.items():
    X = build_class_features(Z_TEST, scores_test_raw[:, cls_idx],
                              test_prior[:, cls_idx], test_base[:, cls_idx])
    prob = clf.predict_proba(X)[:, 1]
    pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
    mlp_scores[:, cls_idx] = ((1 - CFG["frozen_best_probe"]["alpha"]) * test_base[:, cls_idx]
                               + CFG["frozen_best_probe"]["alpha"] * pred)
print(f"MLP scores: range=[{mlp_scores.min():.2f}, {mlp_scores.max():.2f}]")

# %% cell 13
# ═══════════════════════════════════════════════════════════════
# Cell 13 — V19 CALIBRATED ENSEMBLE + POST-PROCESSING
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("V19 CALIBRATED POST-PROCESSING PIPELINE")
print("="*60)

# ── Step 1: Per-class ensemble weights ──
if PER_CLASS_WEIGHTS is not None and CFG["use_per_class_ensemble_weight"]:
    print(f"\n[1] Per-class ensemble weights (mean={PER_CLASS_WEIGHTS.mean():.3f})")
    final_logits = (PER_CLASS_WEIGHTS[None, :] * proto_scores_flat +
                    (1 - PER_CLASS_WEIGHTS[None, :]) * mlp_scores).astype(np.float32)
else:
    print(f"\n[1] Global ensemble weight: {ENSEMBLE_WEIGHT_PROTO:.2f}")
    final_logits = (ENSEMBLE_WEIGHT_PROTO * proto_scores_flat +
                    (1 - ENSEMBLE_WEIGHT_PROTO) * mlp_scores).astype(np.float32)

# ── Step 2: Per-class temperature scaling ──
if PER_CLASS_TEMPS is not None and CFG["use_per_class_temperature"]:
    print(f"[2] Per-class temperatures (mean={PER_CLASS_TEMPS.mean():.3f}, "
          f"range=[{PER_CLASS_TEMPS.min():.2f}, {PER_CLASS_TEMPS.max():.2f}])")
    scaled_logits = final_logits / PER_CLASS_TEMPS[None, :]
else:
    # Fallback: per-taxon
    class_temps = np.ones(N_CLASSES, dtype=np.float32) * CFG["temperature"]["aves"]
    for ci, l in enumerate(PRIMARY_LABELS):
        if CLASS_NAME_MAP.get(l) in TEXTURE_TAXA:
            class_temps[ci] = CFG["temperature"]["texture"]
    print(f"[2] Per-taxon temperatures: Aves={CFG['temperature']['aves']}, Texture={CFG['temperature']['texture']}")
    scaled_logits = final_logits / class_temps[None, :]

probs = sigmoid(scaled_logits)

# ── Step 3: Isotonic calibration (if models available) ──
if ISOTONIC_MODELS and len(ISOTONIC_MODELS) > 0 and CFG["use_isotonic_calibration"]:
    print(f"[3] Isotonic calibration: {len(ISOTONIC_MODELS)} classes")
    n_files = probs.shape[0] // N_WINDOWS
    file_probs = probs.reshape(n_files, N_WINDOWS, N_CLASSES).max(axis=1)
    for c, ir in ISOTONIC_MODELS.items():
        if c < N_CLASSES:
            file_probs[:, c] = ir.transform(file_probs[:, c])
    # Broadcast calibrated file-level back to windows (conservative: use file max)
    file_probs_expanded = np.repeat(file_probs[:, None, :], N_WINDOWS, axis=1)
    # Blend: 70% isotonic file-level + 30% raw window-level
    probs = 0.70 * file_probs_expanded.reshape(-1, N_CLASSES) + 0.30 * probs
    probs = np.clip(probs, 0, 1)
else:
    print("[3] Isotonic calibration: SKIPPED (no models)")

# ── Step 4: File-level confidence scaling (softened) ──
top_k = CFG.get("file_level_top_k", 2)
if top_k > 0:
    print(f"[4] File-level scaling (top_k={top_k}, sqrt-scaled)")
    probs = file_level_confidence_scale(probs, n_windows=N_WINDOWS, top_k=top_k)
    probs = np.clip(probs, 0, 1)

# ── Step 5: Rank-aware scaling (reduced power) ──
if CFG.get("rank_aware_scale"):
    power = CFG.get("rank_aware_power", 0.35)
    print(f"[5] Rank-aware scaling (power={power})")
    probs = rank_aware_scaling(probs, n_windows=N_WINDOWS, power=power)
    probs = np.clip(probs, 0, 1)

# ── Step 6: ADAPTIVE delta smoothing (confidence-gated) ──
alpha = CFG.get("delta_shift_alpha", 0.12)
if alpha > 0 and CFG.get("adaptive_smooth"):
    min_conf = CFG.get("smooth_min_conf", 0.7)
    print(f"[6] Adaptive delta smoothing (alpha={alpha}, min_conf={min_conf})")
    probs = adaptive_delta_smooth(probs, N_WINDOWS, base_alpha=alpha, min_conf=min_conf)
    probs = np.clip(probs, 0, 1)
elif alpha > 0:
    print(f"[6] Delta smoothing (alpha={alpha})")
    probs = adaptive_delta_smooth(probs, N_WINDOWS, base_alpha=alpha, min_conf=1.0)
    probs = np.clip(probs, 0, 1)

# ── Step 7: Per-class threshold sharpening ──
if PER_CLASS_THRESHOLDS is not None:
    print(f"[7] Per-class thresholds (mean={PER_CLASS_THRESHOLDS.mean():.3f}, "
          f"range=[{PER_CLASS_THRESHOLDS.min():.2f}, {PER_CLASS_THRESHOLDS.max():.2f}])")
    probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS)
else:
    print("[7] Default threshold 0.5")
    probs = apply_per_class_thresholds(probs, np.full(N_CLASSES, 0.5, dtype=np.float32))

print(f"\nFinal: range=[{probs.min():.6f}, {probs.max():.6f}], mean={probs.mean():.4f}")

# %% cell 14
# ═══════════════════════════════════════════════════════════════
# Cell 14 — Submission
# ═══════════════════════════════════════════════════════════════
submission = pd.DataFrame(probs, columns=PRIMARY_LABELS)
submission.insert(0, "row_id", meta_test["row_id"].values)
submission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].astype(np.float32)

assert len(submission) == len(test_paths) * N_WINDOWS
assert not submission.isna().any().any()

submission.to_csv("submission.csv", index=False)
print(f"\n✅ submission.csv saved: {submission.shape}")
print(submission.iloc[:3, :8])

wall_time = time.time() - _WALL_START
print(f"\nWall time: {wall_time:.1f}s")
print(f"\nV19 Improvements over V17/V18:")
print("  1. Multi-Scale SSM (parallel 1x + 2x branches)")
print("  2. Dual prototypes per class")
print("  3. Temporal position auxiliary loss")
print("  4. Mixup+CutMix hybrid augmentation")
print("  5. Species-frequency-aware focal loss")
print("  6. Cosine restart scheduler with warmup")
print("  7. Per-class temperature calibration")
print("  8. Per-class ensemble weights")
print("  9. Isotonic calibration (file-level)")
print(" 10. Adaptive confidence-gated smoothing")
print(" 11. Frequency-based threshold defaults")
print(" 12. Expanded Aves genus proxies")
print(" 13. Month prior table")
print(" 14. Softened file-level scaling (sqrt)")
print(" 15. Reduced smoothing aggressiveness")
