# %% cell 1
# Cell 0 -- v72 (Perch+PretrainedProtoSSM10K+EffNet+GlobalMLP): Install ONNX Runtime for Perch v2 (replaces TensorFlow)
# v48 = v45 + determinism fix (threads=1) + parallel file loading + spatial embeddings
# Perch ONNX is 9x faster than TF SavedModel on CPU.
import os, glob
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Install onnxruntime from bundled wheel datasets
_whl_candidates = glob.glob("/kaggle/input/**/onnxruntime*cp312*x86_64*.whl", recursive=True)
if _whl_candidates:
    # Sort to get latest version
    _whl_candidates.sort(reverse=True)
    print(f"Installing onnxruntime from: {_whl_candidates[0]}")
    os.system(f"pip install -q --no-deps '{_whl_candidates[0]}'")
else:
    print("ERROR: No onnxruntime wheel found in mounted datasets!")
    # List all input dirs for debugging
    import subprocess
    subprocess.run(["find", "/kaggle/input", "-maxdepth", "3", "-name", "*.whl"], capture_output=False)

# %% cell 2
# Cell 1 -- Mode switch
MODE = "submit"

assert MODE in {"train", "submit"}
print("MODE =", MODE)


# %% cell 3
# Cell 2 -- v48: Imports and run config
# v48 = v45 base + determinism fix + parallel loading + spatial embeddings
# Changes from v45: intra_op_num_threads=1 (fix 0.004 variance),
# ThreadPoolExecutor for parallel file loading, spatial_embedding extraction
from concurrent.futures import ThreadPoolExecutor

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
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

_WALL_START = time.time()

BASE = Path("/kaggle/input/competitions/birdclef-2026")
MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")  # For labels.csv mapping
# v40: ONNX model path (search multiple mount points)
_onnx_candidates = [
    Path("/kaggle/input/birdclef26-perch-onnx/perch_v2.onnx"),
    Path("/kaggle/input/perch-onnx-for-birdclef-2026/perch_v2.onnx"),
    Path("/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/perch_v2.onnx"),
    Path("/kaggle/input/datasets/yuriygreben/birdclef26-perch-onnx/perch_v2_no_dft.onnx"),
    Path("/kaggle/input/datasets/yuriygreben/birdclef26-perch-onnx/perch_v2.onnx"),
    Path("/kaggle/input/perch-onnx/perch_v2.onnx"),
    Path("/kaggle/input/datasets/mlnjsh/perch-onnx/perch_v2.onnx"),
]
ONNX_MODEL_PATH = next((p for p in _onnx_candidates if p.exists()), _onnx_candidates[0])
print(f"ONNX model path: {ONNX_MODEL_PATH} (exists={ONNX_MODEL_PATH.exists()})")

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
DEVICE = torch.device("cpu")  # Competition: CPU only
LOGS = {}

CFG = {
    "mode": MODE,
    "verbose": MODE == "train",
    "run_oof_baseline": False,
    "run_probe_check": False,
    "run_probe_grid": False,
    # v28: batch_files=32 in submit mode saves ~1 min vs v20's 16
    "batch_files": 16 if MODE == "train" else 4,  # v40: smaller batch for ONNX (less RAM than TF)
    "proxy_reduce": "max",
    "dryrun_n_files": 50 if MODE == "train" else 20,
    "require_full_cache_in_submit": False,
    "full_cache_input_dir": next(
        (p for p in [
            Path("/kaggle/input/perch-meta-0-943"),
            Path("/kaggle/input/datasets/atahalam/perch-meta-0-943"),
            Path("/kaggle/input/datasets/yuriygreben/perch-meta"),
            Path("/kaggle/input/perch-meta"),
            Path("/kaggle/input/datasets/jaejohn/perch-meta"),
        ] if p.exists()),
        Path("/kaggle/input/datasets/jaejohn/perch-meta"),
    ),
    "full_cache_work_dir": Path("/kaggle/working/perch_cache"),
    # v28: Per-taxon prior lambdas (from pantanal-distill)
    "best_fusion": {
        "lambda_event": 0.4,
        "lambda_texture": 1.0,
        "lambda_proxy_texture": 0.8,
        "smooth_texture": 0.35,  # weighted average for Amphibia/Insecta
        "smooth_event": 0.15,    # soft max-pool for Aves/Mammalia/Reptilia
    },
    # v28: SACRED v20 probe params (PCA=48, C=2.0) as LogReg fallback
    "frozen_best_probe": {
        "pca_dim": 128,
        "min_pos": 5,
        "C": 0.75,
        "alpha": 0.45,
    },
    # v28: MLP params (from pantanal-distill, replaces LogReg as primary)
    "probe_backend": "mlp",
    "mlp_params": {
        "hidden_layer_sizes": (384, 256, 128),
        "activation": "relu",
        "max_iter": 500,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 20,
        "random_state": 42,
        "learning_rate_init": 5e-4,
        "alpha": 0.005,
    },
    # v28: ProtoSSM v2 architecture (from pantanal-distill)
    "proto_ssm": {
        "d_model": 256,
        "d_state": 24,
        "n_ssm_layers": 3,
        "dropout": 0.12,
        "n_sites": 20,
        "meta_dim": 24,
    },
    # v28: ProtoSSM training config
    "proto_ssm_train": {
        "n_epochs": 70,
        "lr": 8e-4,
        "weight_decay": 1e-3,
        "patience": 15,
        "pos_weight_cap": 30.0,
        "distill_weight": 0.15,
        "label_smoothing": 0.03,
        "oof_n_splits": 3,
    },
    # v28: ResidualSSM config (from pantanal-distill)
    "residual_ssm": {
        "d_model": 64,
        "d_state": 8,
        "dropout": 0.1,
        "correction_weight": 0.0,
        "n_epochs": 20,
        "lr": 8e-4,
        "patience": 8,
    },
    # v28: Temperature scaling (per-taxon)
    "temperature": {
        "aves": 1.10,
        "texture": 0.95,
    },
    # v28: File-level confidence scaling
    "file_level_top_k": 2,
    # v61: Global MLP weights (pre-trained on train_audio Perch embeddings)
    "global_mlp_weight": 0.06,  # Weight in final blend (0.06 = 6%)
    "global_mlp_path": Path("/kaggle/input/perch-mlp-weights/perch_mlp_weights.npz"),
}

CFG["full_cache_work_dir"].mkdir(parents=True, exist_ok=True)

print("ONNX Runtime:", ort.__version__)
print("PyTorch:", torch.__version__)
print("Competition dir exists:", BASE.exists())
print("Model dir exists:", MODEL_DIR.exists())


# %% cell 4
# Cell 3 -- Data loading and preprocessing (identical to v20)
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
        return {"file_id": None, "site": None, "date": pd.NaT, "time_utc": None, "hour_utc": -1, "month": -1}
    file_id, site, ymd, hms = m.groups()
    dt = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
    return {"file_id": file_id, "site": site, "date": dt, "time_utc": hms, "hour_utc": int(hms[:2]), "month": int(dt.month) if pd.notna(dt) else -1}

def union_labels(series):
    return sorted(set(lbl for x in series for lbl in parse_soundscape_labels(x)))

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

windows_per_file = sc_clean.groupby("filename").size()
full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
sc_clean["file_fully_labeled"] = sc_clean["filename"].isin(full_files)

label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}
Y_SC = np.zeros((len(sc_clean), N_CLASSES), dtype=np.uint8)
for i, labels in enumerate(sc_clean["label_list"]):
    idxs = [label_to_idx[lbl] for lbl in labels if lbl in label_to_idx]
    if idxs:
        Y_SC[i, idxs] = 1

full_truth = sc_clean[sc_clean["file_fully_labeled"]].sort_values(["filename", "end_sec"]).reset_index(drop=False)
Y_FULL_TRUTH = Y_SC[full_truth["index"].to_numpy()]

print("sc_clean:", sc_clean.shape)
print("Y_SC:", Y_SC.shape)
print("Full files:", len(full_files))
print("Trusted full windows:", len(full_truth))


# %% cell 5
# Cell 4 -- v48: Load Perch v2 ONNX session + mapping + genus proxies
BEST = CFG["best_fusion"]

# v40: ONNX session replaces TF SavedModel (9x faster)
_onnx_t0 = time.time()
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 1  # v48: single thread for deterministic results (v45 had 0.004 variance with threads=2)
sess_options.inter_op_num_threads = 1
sess_options.enable_mem_pattern = False  # v40: reduce memory allocation
sess_options.enable_cpu_mem_arena = False  # v40: don't pre-allocate memory pool
ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH), sess_options)
print(f"v40: ONNX session loaded in {time.time()-_onnx_t0:.1f}s")

bc_labels = pd.read_csv(MODEL_DIR / "assets" / "labels.csv").reset_index().rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
NO_LABEL_INDEX = len(bc_labels)
MANUAL_SCIENTIFIC_NAME_MAP = {}
taxonomy = taxonomy.copy()
taxonomy["scientific_name_lookup"] = taxonomy["scientific_name"].replace(MANUAL_SCIENTIFIC_NAME_MAP)
bc_lookup = bc_labels.rename(columns={"scientific_name": "scientific_name_lookup"})
mapping = taxonomy.merge(bc_lookup[["scientific_name_lookup", "bc_index"]], on="scientific_name_lookup", how="left")
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
idx_active_texture = np.array([label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) in TEXTURE_TAXA], dtype=np.int32)
idx_active_event = np.array([label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) not in TEXTURE_TAXA], dtype=np.int32)

idx_mapped_active_texture = idx_active_texture[MAPPED_MASK[idx_active_texture]]
idx_mapped_active_event = idx_active_event[MAPPED_MASK[idx_active_event]]
idx_unmapped_active_texture = idx_active_texture[~MAPPED_MASK[idx_active_texture]]
idx_unmapped_active_event = idx_active_event[~MAPPED_MASK[idx_active_event]]
idx_unmapped_inactive = np.array([i for i in UNMAPPED_POS if PRIMARY_LABELS[i] not in ACTIVE_CLASSES], dtype=np.int32)

# Genus proxies for unmapped species
unmapped_df = mapping[mapping["bc_index"] == NO_LABEL_INDEX].copy()
unmapped_non_sonotype = unmapped_df[~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)].copy()

def get_genus_hits(scientific_name):
    genus = str(scientific_name).split()[0]
    hits = bc_labels[bc_labels["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)].copy()
    return genus, hits

proxy_map = {}
for _, row in unmapped_non_sonotype.iterrows():
    target = row["primary_label"]
    sci = row["scientific_name"]
    genus, hits = get_genus_hits(sci)
    if len(hits) > 0:
        proxy_map[target] = {"bc_indices": hits["bc_index"].astype(int).tolist()}

PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
SELECTED_PROXY_TARGETS = sorted([t for t in proxy_map.keys() if CLASS_NAME_MAP.get(t) in PROXY_TAXA])
selected_proxy_pos = np.array([label_to_idx[c] for c in SELECTED_PROXY_TARGETS], dtype=np.int32)
selected_proxy_pos_to_bc = {label_to_idx[target]: np.array(proxy_map[target]["bc_indices"], dtype=np.int32) for target in SELECTED_PROXY_TARGETS}
idx_selected_proxy_active_texture = np.intersect1d(selected_proxy_pos, idx_active_texture)
idx_selected_prioronly_active_texture = np.setdiff1d(idx_unmapped_active_texture, selected_proxy_pos)
idx_selected_prioronly_active_event = np.setdiff1d(idx_unmapped_active_event, selected_proxy_pos)

print(f"Mapped: {MAPPED_MASK.sum()} / {N_CLASSES}")
print(f"Proxy targets: {len(SELECTED_PROXY_TARGETS)}")


# %% cell 6
# Cell 5 -- Metrics and helper utilities
# v28: Per-taxon smoothing (from pantanal-distill)

def macro_auc_skip_empty(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# Texture taxa (Amphibia, Insecta): weighted average with neighbors
def smooth_cols_fixed12(scores, cols, alpha=0.35):
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

# v28 NEW: Event taxa (Aves, Mammalia, Reptilia): soft max-pool with neighbors
def smooth_events_fixed12(scores, cols, alpha=0.15):
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

# v28: File-level confidence scaling (from timeoptimized notebook)
def file_level_confidence_scale(preds, n_windows=12, top_k=2):
    N, C = preds.shape
    assert N % n_windows == 0
    view = preds.reshape(-1, n_windows, C)
    sorted_view = np.sort(view, axis=1)
    top_k_mean = sorted_view[:, -top_k:, :].mean(axis=1, keepdims=True)
    scaled = view * top_k_mean
    return scaled.reshape(N, C)

print("Utilities defined (per-taxon smoothing, file-level confidence scaling)")


# %% cell 7
# Cell 6 -- v48: Perch ONNX inference with embeddings + spatial + parallel loading
def read_soundscape_60s(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        raise ValueError(f"Unexpected sample rate {sr}")
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    elif len(y) > FILE_SAMPLES:
        y = y[:FILE_SAMPLES]
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
        iterator = tqdm(iterator, total=(n_files + batch_files - 1) // batch_files, desc="Perch batches")
    for start in iterator:
        batch_paths = paths[start:start + batch_files]
        batch_n = len(batch_paths)
        x = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        batch_row_start = write_row
        x_pos = 0
        # v48: parallel file loading with ThreadPoolExecutor
        def _load_one(path):
            return read_soundscape_60s(path)
        with ThreadPoolExecutor(max_workers=min(4, batch_n)) as pool:
            batch_audio = list(pool.map(_load_one, batch_paths))
        for j, path in enumerate(batch_paths):
            y = batch_audio[j]
            x[x_pos:x_pos + N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
            meta_info = parse_soundscape_filename(path.name)
            stem = path.stem
            row_ids[write_row:write_row + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
            filenames[write_row:write_row + N_WINDOWS] = path.name
            sites[write_row:write_row + N_WINDOWS] = meta_info["site"]
            hours[write_row:write_row + N_WINDOWS] = int(meta_info["hour_utc"])
            x_pos += N_WINDOWS
            write_row += N_WINDOWS
        # v48: ONNX inference -- request embedding + label + spatial_embedding
        emb, logits, spatial_emb = ort_session.run(
            ["embedding", "label", "spatial_embedding"], {"inputs": x.astype(np.float32)})
        emb = emb.astype(np.float32, copy=False)
        logits = logits.astype(np.float32, copy=False)
        # v48: spatial_embedding shape (B, 16, 4, 1536) -- save mean over spatial dim
        # spatial_emb_temporal = spatial_emb.mean(axis=2)  # (B, 16, 1536) -- for future LSE experiments
        del spatial_emb  # free memory (not used in probe pipeline yet)
        scores[batch_row_start:write_row, MAPPED_POS] = logits[:, MAPPED_BC_INDICES]
        embeddings[batch_row_start:write_row] = emb
        for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
            sub = logits[:, bc_idx_arr]
            proxy_score = sub.max(axis=1) if proxy_reduce == "max" else sub.mean(axis=1)
            scores[batch_row_start:write_row, pos] = proxy_score.astype(np.float32)
        del x, logits, emb
        gc.collect()
        if hasattr(ort_session, "_sess"): pass  # keep session alive
    meta_df = pd.DataFrame({"row_id": row_ids, "filename": filenames, "site": sites, "hour_utc": hours})
    return meta_df, scores, embeddings


# %% cell 8
# Cell 7 -- Load or compute full-file Perch cache
def resolve_full_cache_paths():
    candidates = [
        (CFG["full_cache_work_dir"] / "full_perch_meta.parquet", CFG["full_cache_work_dir"] / "full_perch_arrays.npz"),
        (Path("/kaggle/working/full_perch_meta.parquet"), Path("/kaggle/working/full_perch_arrays.npz")),
    ]
    if CFG["full_cache_input_dir"].exists():
        candidates.append((CFG["full_cache_input_dir"] / "full_perch_meta.parquet", CFG["full_cache_input_dir"] / "full_perch_arrays.npz"))
    for meta_path, npz_path in candidates:
        if meta_path.exists() and npz_path.exists():
            return meta_path, npz_path
    return None, None

cache_meta, cache_npz = resolve_full_cache_paths()
if cache_meta is not None:
    print("Loading cached Perch outputs:", cache_meta)
    meta_full = pd.read_parquet(cache_meta)
    arr = np.load(cache_npz)
    scores_full_raw = arr["scores_full_raw"].astype(np.float32)
    emb_full = arr["emb_full"].astype(np.float32)
else:
    print("No cache found. Running Perch on trusted full files...")
    full_paths = [BASE / "train_soundscapes" / fn for fn in full_files]
    meta_full, scores_full_raw, emb_full = infer_perch_with_embeddings(full_paths, batch_files=CFG["batch_files"], verbose=True, proxy_reduce=CFG["proxy_reduce"])
    meta_full.to_parquet(CFG["full_cache_work_dir"] / "full_perch_meta.parquet", index=False)
    np.savez_compressed(CFG["full_cache_work_dir"] / "full_perch_arrays.npz", scores_full_raw=scores_full_raw, emb_full=emb_full)

full_truth_aligned = full_truth.set_index("row_id").loc[meta_full["row_id"]].reset_index()
Y_FULL = Y_SC[full_truth_aligned["index"].to_numpy()]
print("meta_full:", meta_full.shape, "emb_full:", emb_full.shape)


# %% cell 9
# Cell 8 -- Fold-safe metadata prior tables (v28: per-taxon smoothing)
def fit_prior_tables(prior_df, Y_prior):
    prior_df = prior_df.reset_index(drop=True)
    global_p = Y_prior.mean(axis=0).astype(np.float32)
    site_keys = sorted(prior_df["site"].dropna().astype(str).unique().tolist())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_n = np.zeros(len(site_keys), dtype=np.float32)
    site_p = np.zeros((len(site_keys), Y_prior.shape[1]), dtype=np.float32)
    for s in site_keys:
        i = site_to_i[s]
        mask = prior_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        site_p[i] = Y_prior[mask].mean(axis=0)
    hour_keys = sorted(prior_df["hour_utc"].dropna().astype(int).unique().tolist())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_n = np.zeros(len(hour_keys), dtype=np.float32)
    hour_p = np.zeros((len(hour_keys), Y_prior.shape[1]), dtype=np.float32)
    for h in hour_keys:
        i = hour_to_i[h]
        mask = prior_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        hour_p[i] = Y_prior[mask].mean(axis=0)
    sh_to_i, sh_n_list, sh_p_list = {}, [], []
    for (s, h), idx in prior_df.groupby(["site", "hour_utc"]).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n_list)
        idx = np.array(list(idx))
        sh_n_list.append(len(idx))
        sh_p_list.append(Y_prior[idx].mean(axis=0))
    sh_n = np.array(sh_n_list, dtype=np.float32)
    sh_p = np.stack(sh_p_list).astype(np.float32) if sh_p_list else np.zeros((0, Y_prior.shape[1]), dtype=np.float32)
    return {"global_p": global_p, "site_to_i": site_to_i, "site_n": site_n, "site_p": site_p, "hour_to_i": hour_to_i, "hour_n": hour_n, "hour_p": hour_p, "sh_to_i": sh_to_i, "sh_n": sh_n, "sh_p": sh_p}

def prior_logits_from_tables(sites, hours, tables, eps=1e-4):
    n = len(sites)
    p = np.repeat(tables["global_p"][None, :], n, axis=0).astype(np.float32, copy=True)
    site_idx = np.fromiter((tables["site_to_i"].get(str(s), -1) for s in sites), dtype=np.int32, count=n)
    hour_idx = np.fromiter((tables["hour_to_i"].get(int(h), -1) if int(h) >= 0 else -1 for h in hours), dtype=np.int32, count=n)
    sh_idx = np.fromiter((tables["sh_to_i"].get((str(s), int(h)), -1) if int(h) >= 0 else -1 for s, h in zip(sites, hours)), dtype=np.int32, count=n)
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

# v28: fuse with BOTH texture and event smoothing (pantanal-distill style)
def fuse_scores_with_tables(base_scores, sites, hours, tables,
                            lambda_event=BEST["lambda_event"],
                            lambda_texture=BEST["lambda_texture"],
                            lambda_proxy_texture=BEST["lambda_proxy_texture"],
                            smooth_texture=BEST["smooth_texture"],
                            smooth_event=BEST["smooth_event"]):
    scores = base_scores.copy()
    prior = prior_logits_from_tables(sites, hours, tables)
    if len(idx_mapped_active_event):
        scores[:, idx_mapped_active_event] += lambda_event * prior[:, idx_mapped_active_event]
    if len(idx_mapped_active_texture):
        scores[:, idx_mapped_active_texture] += lambda_texture * prior[:, idx_mapped_active_texture]
    if len(idx_selected_proxy_active_texture):
        scores[:, idx_selected_proxy_active_texture] += lambda_proxy_texture * prior[:, idx_selected_proxy_active_texture]
    if len(idx_selected_prioronly_active_event):
        scores[:, idx_selected_prioronly_active_event] = lambda_event * prior[:, idx_selected_prioronly_active_event]
    if len(idx_selected_prioronly_active_texture):
        scores[:, idx_selected_prioronly_active_texture] = lambda_texture * prior[:, idx_selected_prioronly_active_texture]
    if len(idx_unmapped_inactive):
        scores[:, idx_unmapped_inactive] = -8.0
    # v28: per-taxon smoothing
    scores = smooth_cols_fixed12(scores, idx_active_texture, alpha=smooth_texture)
    scores = smooth_events_fixed12(scores, idx_active_event, alpha=smooth_event)
    return scores.astype(np.float32, copy=False), prior

print("Prior tables and per-taxon smoothing defined.")


# %% cell 10
# Cell 9 -- OOF base/prior meta-features (from v20)
def build_oof_base_prior(scores_full_raw, meta_full, sc_clean, Y_SC, n_splits=5, verbose=True):
    groups_full = meta_full["filename"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    oof_base = np.zeros_like(scores_full_raw, dtype=np.float32)
    oof_prior = np.zeros_like(scores_full_raw, dtype=np.float32)
    fold_id = np.full(len(meta_full), -1, dtype=np.int16)
    splits = list(gkf.split(scores_full_raw, groups=groups_full))
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        tr_idx, va_idx = np.sort(tr_idx), np.sort(va_idx)
        val_files = set(meta_full.iloc[va_idx]["filename"].tolist())
        prior_mask = ~sc_clean["filename"].isin(val_files).values
        tables = fit_prior_tables(sc_clean.loc[prior_mask].reset_index(drop=True), Y_SC[prior_mask])
        va_base, va_prior = fuse_scores_with_tables(scores_full_raw[va_idx], sites=meta_full.iloc[va_idx]["site"].to_numpy(), hours=meta_full.iloc[va_idx]["hour_utc"].to_numpy(), tables=tables)
        oof_base[va_idx] = va_base
        oof_prior[va_idx] = va_prior
        fold_id[va_idx] = fold
    return oof_base, oof_prior, fold_id

print("Building honest 5-fold OOF meta-features...")
oof_base, oof_prior, oof_fold_id = build_oof_base_prior(scores_full_raw, meta_full, sc_clean, Y_SC, n_splits=5, verbose=False)
_oof_candidates = [
    Path("/kaggle/input/perch-meta-0-943/full_oof_meta_features.npz"),
    Path("/kaggle/input/datasets/atahalam/perch-meta-0-943/full_oof_meta_features.npz"),
]
for _oof_file in _oof_candidates:
    if _oof_file.exists():
        _oof = np.load(_oof_file)
        if _oof["oof_base"].shape == oof_base.shape and _oof["oof_prior"].shape == oof_prior.shape:
            oof_base = _oof["oof_base"].astype(np.float32)
            oof_prior = _oof["oof_prior"].astype(np.float32)
            oof_fold_id = _oof["fold_id"].astype(np.int16)
            print(f"Loaded cached OOF meta-features from {_oof_file}")
        else:
            print(f"Cached OOF shape mismatch, keeping rebuilt features: {_oof_file}")
        break
baseline_oof_auc = macro_auc_skip_empty(Y_FULL, oof_base)
print(f"OOF baseline AUC: {baseline_oof_auc:.6f}")


# %% cell 11
# Cell 10 -- Classwise embedding-probe helpers (v28: pantanal-distill features)
def build_class_features(emb_proj, raw_col, prior_col, base_col):
    prev_base, next_base, mean_base, max_base, std_base = seq_features_1d(base_col)
    diff_mean = base_col - mean_base
    diff_prev = base_col - prev_base
    diff_next = base_col - next_base
    feats = np.concatenate([
        emb_proj,
        raw_col[:, None],
        prior_col[:, None],
        base_col[:, None],
        prev_base[:, None],
        next_base[:, None],
        mean_base[:, None],
        max_base[:, None],
        std_base[:, None],
        diff_mean[:, None],
        diff_prev[:, None],
        diff_next[:, None],
        (raw_col * prior_col)[:, None],
        (raw_col * base_col)[:, None],
        (prior_col * base_col)[:, None],
    ], axis=1)
    return feats.astype(np.float32, copy=False)

print("build_class_features defined")


# %% cell 12
# Cell 11 -- Fit final prior tables, PCA, and MLP probes
BEST_PROBE = CFG["frozen_best_probe"]
final_prior_tables = fit_prior_tables(sc_clean.reset_index(drop=True), Y_SC)

# PCA on embeddings
emb_scaler = StandardScaler()
emb_full_scaled = emb_scaler.fit_transform(emb_full)
n_comp = min(int(BEST_PROBE["pca_dim"]), emb_full_scaled.shape[0] - 1, emb_full_scaled.shape[1])
emb_pca = PCA(n_components=n_comp)
Z_FULL = emb_pca.fit_transform(emb_full_scaled).astype(np.float32)
print(f"PCA: {emb_full.shape} -> {Z_FULL.shape}, explained var: {emb_pca.explained_variance_ratio_.sum():.4f}")

# v28: Train MLP(128) probes per class (from pantanal-distill, replaces LogReg)
full_pos_counts = Y_FULL.sum(axis=0)
PROBE_CLASS_IDX = np.where(full_pos_counts >= int(BEST_PROBE["min_pos"]))[0].astype(np.int32)

probe_models = {}
for cls_idx in PROBE_CLASS_IDX:
    y = Y_FULL[:, cls_idx]
    if y.sum() == 0 or y.sum() == len(y):
        continue
    X_cls = build_class_features(Z_FULL, raw_col=scores_full_raw[:, cls_idx], prior_col=oof_prior[:, cls_idx], base_col=oof_base[:, cls_idx])
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    # Oversampling for class balance (MLP does not support sample_weight)
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
print(f"Wall time: {(time.time() - _WALL_START) / 60:.1f} min")


# %% cell 13
# Cell 12 -- ProtoSSM v2: Selective State Space Model (from pantanal-distill)

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
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
        B_t = self.B_proj(x_conv)
        C_t = self.C_proj(x_conv)
        A = -torch.exp(self.A_log)
        y = self._selective_scan(x_conv, dt, A, B_t, C_t)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(self, x, dt, A, B, C):
        batch, T, D = x.shape
        N = self.d_state
        h = torch.zeros(batch, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            dt_t = dt[:, t, :, None]
            dA = torch.exp(A[None] * dt_t)
            dB = dt_t * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            y_t = (h * C[:, t, None, :]).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


class ProtoSSMv2(nn.Module):
    def __init__(self, d_input=1536, d_model=128, d_state=16, n_ssm_layers=2,
                 n_classes=234, n_windows=12, dropout=0.15, n_sites=20, meta_dim=16):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.input_proj = nn.Sequential(nn.Linear(d_input, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
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
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))
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
        self.register_buffer("class_to_family", torch.tensor(class_to_family, dtype=torch.long))

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape
        h = self.input_proj(emb) + self.pos_enc[:, :T, :]
        if site_ids is not None and hours is not None:
            s_emb = self.site_emb(site_ids)
            h_emb = self.hour_emb(hours)
            meta = self.meta_proj(torch.cat([s_emb, h_emb], dim=-1))
            h = h + meta[:, None, :]
        for fwd, bwd, merge, norm in zip(self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm):
            residual = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h = merge(torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = norm(h + residual)
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        temp = F.softplus(self.proto_temp)
        sim = torch.matmul(h_norm, p_norm.T) * temp + self.class_bias[None, None, :]
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            species_logits = alpha * sim + (1 - alpha) * perch_logits
        else:
            species_logits = sim
        family_logits = None
        if self.family_head is not None:
            family_logits = self.family_head(h.mean(dim=1))
        return species_logits, family_logits, h

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

print("ProtoSSM v2 architecture defined.")


# %% cell 14
# Cell 13 -- ProtoSSM v2 Training + ResidualSSM helpers

def build_taxonomy_groups(taxonomy_df, primary_labels):
    for col in ["family", "order", "class_name"]:
        if col in taxonomy_df.columns:
            group_map = taxonomy_df.set_index("primary_label")[col].to_dict()
            break
    else:
        group_map = {label: "Unknown" for label in primary_labels}
    groups = sorted(set(group_map.values()))
    grp_to_idx = {g: i for i, g in enumerate(groups)}
    class_to_group = [grp_to_idx.get(group_map.get(label, "Unknown"), 0) for label in primary_labels]
    return len(groups), class_to_group, grp_to_idx

def build_site_mapping(meta_df):
    sites = meta_df["site"].unique().tolist()
    site_to_idx = {s: i + 1 for i, s in enumerate(sites)}
    return site_to_idx, len(sites) + 1

def reshape_to_files(flat_array, meta_df, n_windows=N_WINDOWS):
    filenames = meta_df["filename"].to_numpy()
    unique_files = []
    seen = set()
    for f in filenames:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)
    n_files = len(unique_files)
    new_shape = (n_files, n_windows) + flat_array.shape[1:]
    return flat_array.reshape(new_shape), unique_files

def get_file_metadata(meta_df, file_list, site_to_idx, n_sites_max):
    file_to_row = {}
    filenames = meta_df["filename"].to_numpy()
    sites_arr = meta_df["site"].to_numpy()
    hours_arr = meta_df["hour_utc"].to_numpy()
    for i, f in enumerate(filenames):
        if f not in file_to_row:
            file_to_row[f] = i
    site_ids = np.zeros(len(file_list), dtype=np.int64)
    hour_ids = np.zeros(len(file_list), dtype=np.int64)
    for fi, fname in enumerate(file_list):
        row = file_to_row.get(fname)
        if row is not None:
            site_ids[fi] = min(site_to_idx.get(sites_arr[row], 0), n_sites_max - 1)
            hour_ids[fi] = int(hours_arr[row]) % 24
    return site_ids, hour_ids

def train_proto_ssm_single(model, emb_train, logits_train, labels_train,
                           site_ids_train=None, hours_train=None,
                           emb_val=None, logits_val=None, labels_val=None,
                           site_ids_val=None, hours_val=None,
                           file_families_train=None, file_families_val=None,
                           cfg=None, verbose=True):
    if cfg is None:
        cfg = CFG["proto_ssm_train"]
    label_smoothing = cfg.get("label_smoothing", 0.0)
    emb_tr = torch.tensor(emb_train, dtype=torch.float32)
    logits_tr = torch.tensor(logits_train, dtype=torch.float32)
    labels_tr = torch.tensor(labels_train, dtype=torch.float32)
    if label_smoothing > 0:
        labels_tr = labels_tr * (1.0 - label_smoothing) + label_smoothing / 2.0
    site_tr = torch.tensor(site_ids_train, dtype=torch.long) if site_ids_train is not None else None
    hour_tr = torch.tensor(hours_train, dtype=torch.long) if hours_train is not None else None
    has_val = emb_val is not None
    if has_val:
        emb_v = torch.tensor(emb_val, dtype=torch.float32)
        logits_v = torch.tensor(logits_val, dtype=torch.float32)
        labels_v = torch.tensor(labels_val, dtype=torch.float32)
        site_v = torch.tensor(site_ids_val, dtype=torch.long) if site_ids_val is not None else None
        hour_v = torch.tensor(hours_val, dtype=torch.long) if hours_val is not None else None
    fam_tr = torch.tensor(file_families_train, dtype=torch.float32) if file_families_train is not None else None
    pos_counts = labels_tr.sum(dim=(0, 1))
    total = labels_tr.shape[0] * labels_tr.shape[1]
    pos_weight = ((total - pos_counts) / (pos_counts + 1)).clamp(max=cfg["pos_weight_cap"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["lr"], epochs=cfg["n_epochs"], steps_per_epoch=1, pct_start=0.1, anneal_strategy="cos")
    best_val_loss = float("inf")
    best_state = None
    wait = 0
    for epoch in range(cfg["n_epochs"]):
        model.train()
        species_out, family_out, _ = model(emb_tr, logits_tr, site_ids=site_tr, hours=hour_tr)
        loss_bce = F.binary_cross_entropy_with_logits(species_out, labels_tr, pos_weight=pos_weight[None, None, :])
        loss_distill = F.mse_loss(species_out, logits_tr)
        loss = loss_bce + cfg["distill_weight"] * loss_distill
        if family_out is not None and fam_tr is not None:
            loss = loss + 0.1 * F.binary_cross_entropy_with_logits(family_out, fam_tr)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            if has_val:
                val_out, _, _ = model(emb_v, logits_v, site_ids=site_v, hours=hour_v)
                val_loss = F.binary_cross_entropy_with_logits(val_out, labels_v, pos_weight=pos_weight[None, None, :])
            else:
                val_loss = loss
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: train={loss.item():.4f} val={val_loss.item():.4f} wait={wait}")
        if wait >= cfg["patience"]:
            if verbose:
                print(f"  Early stop at epoch {epoch+1}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": best_val_loss}


def optimize_ensemble_weight(oof_proto_flat, oof_mlp_flat, y_true_flat):
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

print("ProtoSSM training and ensemble weight optimizer defined.")


# %% cell 15
# Cell 14 -- v72: Load pretrained ProtoSSM (10K files) + fine-tune on labeled data
_wall_before_proto = (time.time() - _WALL_START) / 60.0
print(f"Wall time before ProtoSSM: {_wall_before_proto:.1f} min")

# Reshape to file-level
emb_files, file_list = reshape_to_files(emb_full, meta_full)
logits_files, _ = reshape_to_files(scores_full_raw, meta_full)
labels_files, _ = reshape_to_files(Y_FULL, meta_full)
print(f"File-level shapes: emb={emb_files.shape}, logits={logits_files.shape}")

# Taxonomy groups and site mapping
n_families, class_to_family, fam_to_idx = build_taxonomy_groups(taxonomy, PRIMARY_LABELS)
site_to_idx_map, n_sites_mapped = build_site_mapping(meta_full)
n_sites_cfg = CFG["proto_ssm"]["n_sites"]
site_ids_all, hours_all = get_file_metadata(meta_full, file_list, site_to_idx_map, n_sites_cfg)

# Per-file family labels
file_families = np.zeros((len(file_list), n_families), dtype=np.float32)
for fi in range(len(file_list)):
    active = np.where(labels_files[fi].sum(axis=0) > 0)[0]
    for ci in active:
        file_families[fi, class_to_family[ci]] = 1.0

# Create model
ssm_cfg = CFG["proto_ssm"]
model = ProtoSSMv2(
    d_input=emb_full.shape[1], d_model=ssm_cfg["d_model"], d_state=ssm_cfg["d_state"],
    n_ssm_layers=ssm_cfg["n_ssm_layers"], n_classes=N_CLASSES, n_windows=N_WINDOWS,
    dropout=ssm_cfg["dropout"], n_sites=ssm_cfg["n_sites"], meta_dim=ssm_cfg["meta_dim"],
).to(DEVICE)

model.init_prototypes_from_data(torch.tensor(emb_full, dtype=torch.float32), torch.tensor(Y_FULL, dtype=torch.float32))
model.init_family_head(n_families, class_to_family)

# v72: Load pretrained weights from 10K soundscape files
_pretrain_paths = [
    Path("/kaggle/input/birdclef2026-protossm-500ep/protossm_500ep_best.pt"),
    Path("/kaggle/input/protossm-10k/protossm_500ep_best.pt"),
]
_pretrain_loaded = False
for _pp in _pretrain_paths:
    if _pp.exists():
        print(f"Loading pretrained ProtoSSM from {_pp}")
        _ckpt = torch.load(str(_pp), map_location="cpu", weights_only=False)
        _model_dict = model.state_dict()
        _loaded, _skipped = 0, 0
        for k, v in _ckpt.items():
            if k in _model_dict and v.shape == _model_dict[k].shape:
                _model_dict[k] = v
                _loaded += 1
            else:
                _skipped += 1
        model.load_state_dict(_model_dict)
        model = model.to(DEVICE)
        print(f"  Loaded {_loaded} params, skipped {_skipped} (shape mismatch)")
        _pretrain_loaded = True
        break

if not _pretrain_loaded:
    print("No pretrained weights found -- training from scratch")

# Fine-tune on labeled data (always run, even with pretrained weights)
print(f"ProtoSSM v2 parameters: {model.count_parameters():,}")

t0 = time.time()
model, _ = train_proto_ssm_single(
    model, emb_files, logits_files, labels_files.astype(np.float32),
    site_ids_train=site_ids_all, hours_train=hours_all,
    file_families_train=file_families,
    cfg=CFG["proto_ssm_train"], verbose=True,
)
print(f"ProtoSSM training time: {time.time() - t0:.1f}s")

# OOF ensemble weight
ENSEMBLE_WEIGHT_PROTO = 0.55
_weight_candidates = [
    Path("/kaggle/input/perch-meta-0-943/ensemble_weight_v20.json"),
    Path("/kaggle/input/datasets/atahalam/perch-meta-0-943/ensemble_weight_v20.json"),
]
for _weight_file in _weight_candidates:
    if _weight_file.exists():
        ENSEMBLE_WEIGHT_PROTO = float(json.loads(_weight_file.read_text())["ensemble_weight_proto"])
        print(f"Loaded ProtoSSM ensemble weight from {_weight_file}: {ENSEMBLE_WEIGHT_PROTO:.2f}")
        break
print(f"Ensemble weight (ProtoSSM): {ENSEMBLE_WEIGHT_PROTO}")
print(f"Wall time after ProtoSSM: {(time.time() - _WALL_START) / 60:.1f} min")


# %% cell 16
# Cell 15 -- v29/v40: ResidualSSM REMOVED (ablation experiment)
# v28 had ResidualSSM second-pass correction, v29 removes it to:
#   1. Save ~30s timeout margin
#   2. Reduce overfitting from second pass
#   3. Give ProtoSSM more training budget (80 vs 60 epochs)
res_model = None
CORRECTION_WEIGHT = 0.0
print("v29: ResidualSSM SKIPPED (ablation). ProtoSSM weight=0.55, epochs=80")
print(f"Wall time: {(time.time() - _WALL_START) / 60:.1f} min")


# %% cell 17
# Cell 16 -- v48: Infer Perch on hidden test (batch_files=4, threads=1 for determinism)
test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
if len(test_paths) == 0:
    print(f"Hidden test not mounted. Dry-run on first {CFG['dryrun_n_files']} train soundscapes.")
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:CFG["dryrun_n_files"]]
else:
    print(f"Hidden test files: {len(test_paths)}")

# Wall-time guard
_elapsed = (time.time() - _WALL_START) / 60.0
print(f"Wall time before test inference: {_elapsed:.1f} min")

meta_test, scores_test_raw, emb_test = infer_perch_with_embeddings(
    test_paths, batch_files=CFG["batch_files"], verbose=True, proxy_reduce=CFG["proxy_reduce"])
print(f"Wall time after test inference: {(time.time()-_WALL_START)/60:.1f} min")
print(f"scores_test_raw: {scores_test_raw.shape}, emb_test: {emb_test.shape}")


# %% cell 18
# Cell 16b -- EffNet-B0 5-fold inference on test soundscapes (v53: Perch + EffNet blend)
import timm
import librosa

_wall_effnet_start = time.time()

# -- EffNet audio processing (different from Perch) --
EFFNET_N_FFT = 1024
EFFNET_HOP = 512
EFFNET_MELS = 128
EFFNET_FMIN = 20
EFFNET_FMAX = 16000
EFFNET_SR = 32000
EFFNET_CLIP = 5.0
EFFNET_N_SAMPLES = int(EFFNET_SR * EFFNET_CLIP)

def compute_logmel_effnet(audio):
    """Log-mel spectrogram matching EffNet training exactly."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=EFFNET_SR, n_fft=EFFNET_N_FFT, hop_length=EFFNET_HOP,
        n_mels=EFFNET_MELS, fmin=EFFNET_FMIN, fmax=EFFNET_FMAX, power=2.0)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    vmin, vmax = log_mel.min(), log_mel.max()
    log_mel = (log_mel - vmin) / (vmax - vmin + 1e-8)
    return log_mel[np.newaxis, :, :].astype(np.float32)

# -- EffNet model (must match training architecture) --
class BirdCLEFEffNetLogits(nn.Module):
    def __init__(self, backbone_name='tf_efficientnet_b0_ns', pretrained=False,
                 num_species=234, dropout=0.3, in_chans=1):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                          num_classes=0, in_chans=in_chans)
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, 256, 256)
            emb_dim = self.backbone(dummy).shape[-1]
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Dropout(dropout),
                                  nn.Linear(emb_dim, num_species))
    def forward(self, spec):
        if spec.shape[-2] != 256 or spec.shape[-1] != 256:
            spec = F.interpolate(spec, size=(256, 256), mode='bilinear', align_corners=False)
        return self.head(self.backbone(spec))

# -- Load EffNet checkpoints --
_effnet_dirs = [
    Path("/kaggle/input/datasets/mlnjsh/birdclef2026-effnet-5fold"),
    Path("/kaggle/input/birdclef2026-effnet-5fold"),
    Path("/kaggle/input/mlnjsh/birdclef2026-effnet-5fold"),
]
EFFNET_DIR = None
for d in _effnet_dirs:
    if d.exists():
        EFFNET_DIR = d
        break
if EFFNET_DIR is None:
    EFFNET_DIR = _effnet_dirs[0]  # fallback

effnet_models = []
if EFFNET_DIR.exists():
    ckpt_files = sorted(EFFNET_DIR.rglob('tf_efficientnet_b0_ns_fold*.pt'))
    if not ckpt_files:
        ckpt_files = sorted(EFFNET_DIR.rglob('*.pt'))
    for ckpt in ckpt_files[:5]:
        m = BirdCLEFEffNetLogits(pretrained=False)
        m.load_state_dict(torch.load(str(ckpt), map_location='cpu'))
        m.eval()
        effnet_models.append(m)
    print(f"Loaded {len(effnet_models)} EffNet checkpoints from {EFFNET_DIR}")
else:
    print(f"WARNING: EffNet dir not found: {EFFNET_DIR}")

# -- EffNet inference on test (no TTA to save time) --
effnet_logits = np.zeros((len(meta_test), N_CLASSES), dtype=np.float32)

if len(effnet_models) > 0:
    row_idx = 0
    for fi, fpath in enumerate(test_paths):
        try:
            audio, orig_sr = sf.read(str(fpath), dtype='float32', always_2d=False)
            if orig_sr != EFFNET_SR:
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=EFFNET_SR)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        except Exception as e:
            print(f"EffNet load error {fpath.name}: {e}")
            row_idx += N_WINDOWS
            continue

        for w in range(N_WINDOWS):
            s = w * EFFNET_N_SAMPLES
            e_idx = s + EFFNET_N_SAMPLES
            if e_idx <= len(audio):
                chunk = audio[s:e_idx]
            else:
                chunk = audio[s:]
                if len(chunk) < EFFNET_N_SAMPLES:
                    chunk = np.pad(chunk, (0, EFFNET_N_SAMPLES - len(chunk)))

            spec = compute_logmel_effnet(chunk)
            spec_t = torch.from_numpy(spec).unsqueeze(0)
            with torch.no_grad():
                preds = []
                for m in effnet_models:
                    logits = m(spec_t)
                    preds.append(torch.sigmoid(logits).numpy()[0])
                avg_prob = np.mean(preds, axis=0).astype(np.float32)
                # Convert to logits for blending
                avg_prob = np.clip(avg_prob, 1e-7, 1 - 1e-7)
                effnet_logits[row_idx] = np.log(avg_prob / (1 - avg_prob))
            row_idx += 1

        if (fi + 1) % 100 == 0:
            print(f"  EffNet [{fi+1}/{len(test_paths)}] {(time.time()-_wall_effnet_start)/60:.1f} min")

    print(f"EffNet inference done: {effnet_logits.shape}, {(time.time()-_wall_effnet_start)/60:.1f} min")
else:
    print("No EffNet models loaded -- skipping EffNet inference")
print(f"Wall time after EffNet: {(time.time()-_WALL_START)/60:.1f} min")


# %% cell 19
# Cell 17 -- Score fusion: ProtoSSM v2 + MLP + ResidualSSM

# Step 1: ProtoSSM v2 inference on test
emb_test_files, test_file_list = reshape_to_files(emb_test, meta_test)
logits_test_files, _ = reshape_to_files(scores_test_raw, meta_test)
test_site_ids, test_hours = get_file_metadata(meta_test, test_file_list, site_to_idx_map, n_sites_cfg)

model.eval()
with torch.no_grad():
    proto_out, _, _ = model(
        torch.tensor(emb_test_files, dtype=torch.float32),
        torch.tensor(logits_test_files, dtype=torch.float32),
        site_ids=torch.tensor(test_site_ids, dtype=torch.long),
        hours=torch.tensor(test_hours, dtype=torch.long))
    proto_scores_flat = proto_out.numpy().reshape(-1, N_CLASSES).astype(np.float32)
print(f"ProtoSSM test scores: {proto_scores_flat.shape}")

# Step 2: Prior-fused base scores
test_base_scores, test_prior_scores = fuse_scores_with_tables(
    scores_test_raw, meta_test["site"].to_numpy(), meta_test["hour_utc"].to_numpy(), final_prior_tables)

# v27: Pseudo site priors (embedded as base64)
import base64, io
_PSEUDO_PRIORS_B64 = "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"
PSEUDO_PRIOR_ALPHA = 0.02
_buf = io.BytesIO(base64.b64decode(_PSEUDO_PRIORS_B64))
_d = np.load(_buf, allow_pickle=True)
pseudo_site_priors = _d["site_prior_matrix"]
pseudo_global_prior = _d["global_prior"]
pseudo_sites_list = list(_d["sites"])
if PSEUDO_PRIOR_ALPHA > 0:
    test_sites = meta_test["site"].to_numpy()
    site_to_pidx = {s: i for i, s in enumerate(pseudo_sites_list)}
    prior_matrix = np.tile(pseudo_global_prior, (len(test_sites), 1))
    for site, idx in site_to_pidx.items():
        mask = (test_sites == site)
        if mask.any():
            prior_matrix[mask] = pseudo_site_priors[idx]
    test_base_scores = (1 - PSEUDO_PRIOR_ALPHA) * test_base_scores + PSEUDO_PRIOR_ALPHA * prior_matrix
    print(f"Applied pseudo site priors (alpha={PSEUDO_PRIOR_ALPHA})")

# Step 3: MLP probe scores on test
emb_test_scaled = emb_scaler.transform(emb_test)
Z_TEST = emb_pca.transform(emb_test_scaled).astype(np.float32)
mlp_scores = test_base_scores.copy()

for cls_idx, clf in probe_models.items():
    X_cls_test = build_class_features(Z_TEST, raw_col=scores_test_raw[:, cls_idx],
                                       prior_col=test_prior_scores[:, cls_idx],
                                       base_col=test_base_scores[:, cls_idx])
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X_cls_test)[:, 1].astype(np.float32)
        pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
    else:
        pred = clf.decision_function(X_cls_test).astype(np.float32)
    # v20: Per-class adaptive alpha
    n_pos = int(full_pos_counts[cls_idx])
    min_pos = int(BEST_PROBE["min_pos"])
    alpha_cls = 0.25 + 0.25 * min(1.0, (n_pos - min_pos) / 40.0)
    mlp_scores[:, cls_idx] = (1.0 - alpha_cls) * test_base_scores[:, cls_idx] + alpha_cls * pred


# Step 3b: Global MLP inference (v61: pre-trained on train_audio Perch embeddings)
GLOBAL_MLP_WEIGHT = CFG.get("global_mlp_weight", 0.0)
_gmlp_path = CFG.get("global_mlp_path")
if GLOBAL_MLP_WEIGHT > 0 and _gmlp_path is not None and _gmlp_path.exists():
    _gw = np.load(str(_gmlp_path))
    # 3-layer MLP: 1536 -> 768 -> 384 -> N_CLASSES (numpy forward pass, no PyTorch needed)
    _h = emb_test @ _gw['fc1_weight'].T + _gw['fc1_bias']
    _h = np.maximum(_h, 0)  # ReLU
    _h = _h @ _gw['fc2_weight'].T + _gw['fc2_bias']
    _h = np.maximum(_h, 0)  # ReLU
    global_mlp_logits = (_h @ _gw['fc3_weight'].T + _gw['fc3_bias']).astype(np.float32)
    del _gw, _h
    print(f"Global MLP logits: {global_mlp_logits.shape}, range [{global_mlp_logits.min():.3f}, {global_mlp_logits.max():.3f}]")
else:
    global_mlp_logits = None
    if GLOBAL_MLP_WEIGHT > 0:
        print(f"WARNING: Global MLP weights not found at {_gmlp_path}, skipping")
    else:
        print("Global MLP: DISABLED (weight=0)")

# Step 4: Ensemble (v61: Perch + EffNet + Global MLP)
perch_scores = (ENSEMBLE_WEIGHT_PROTO * proto_scores_flat + (1.0 - ENSEMBLE_WEIGHT_PROTO) * mlp_scores).astype(np.float32)
# v61: 3-way blend: Perch + EffNet + Global MLP
BLEND_WEIGHT_EFFNET = 0.08  # EffNet weight (proven in v53)
_blend_gmlp = GLOBAL_MLP_WEIGHT if global_mlp_logits is not None else 0.0
BLEND_WEIGHT_PERCH = 1.0 - BLEND_WEIGHT_EFFNET - _blend_gmlp
final_test_scores = BLEND_WEIGHT_PERCH * perch_scores
if effnet_logits is not None and np.abs(effnet_logits).sum() > 0:
    final_test_scores = final_test_scores + BLEND_WEIGHT_EFFNET * effnet_logits
    print(f"  + EffNet ({BLEND_WEIGHT_EFFNET:.0%})")
else:
    BLEND_WEIGHT_PERCH += BLEND_WEIGHT_EFFNET
    final_test_scores = BLEND_WEIGHT_PERCH * perch_scores
    print("  EffNet: SKIPPED (no scores)")
if global_mlp_logits is not None and _blend_gmlp > 0:
    final_test_scores = final_test_scores + _blend_gmlp * global_mlp_logits
    print(f"  + Global MLP ({_blend_gmlp:.0%})")
else:
    print("  Global MLP: SKIPPED")
final_test_scores = final_test_scores.astype(np.float32)
print(f"Blend weights: Perch={BLEND_WEIGHT_PERCH:.0%}, EffNet={BLEND_WEIGHT_EFFNET:.0%}, GlobalMLP={_blend_gmlp:.0%}")
print(f"Ensemble scores: {final_test_scores.shape}")

# Step 5: Residual correction
if res_model is not None and CORRECTION_WEIGHT > 0:
    first_pass_test_files, _ = reshape_to_files(final_test_scores, meta_test)
    res_model.eval()
    with torch.no_grad():
        test_corr = res_model(
            torch.tensor(emb_test_files, dtype=torch.float32),
            torch.tensor(first_pass_test_files, dtype=torch.float32),
            site_ids=torch.tensor(test_site_ids, dtype=torch.long),
            hours=torch.tensor(test_hours, dtype=torch.long)
        ).numpy().reshape(-1, N_CLASSES).astype(np.float32)
    final_test_scores = final_test_scores + CORRECTION_WEIGHT * test_corr
    print(f"Applied residual correction (weight={CORRECTION_WEIGHT})")
else:
    print("Residual correction: SKIPPED")

print(f"Score range: [{final_test_scores.min():.3f}, {final_test_scores.max():.3f}]")


# %% cell 20
# Cell 18 -- Submission: per-taxon temperature + file-level confidence scaling
print(f"Wall time at submission: {(time.time()-_WALL_START)/60:.1f} min")

# v28: Per-taxon temperature scaling (from pantanal-distill)
temp_cfg = CFG["temperature"]
T_AVES = temp_cfg["aves"]
T_TEXTURE = temp_cfg["texture"]
# V51: Per-class optimized temperatures from 0.943 dataset
_temp_candidates = [
    Path("/kaggle/input/perch-meta-0-943/oof_temps_v19.npy"),
    Path("/kaggle/input/datasets/atahalam/perch-meta-0-943/oof_temps_v19.npy"),
]
_temp_file = next((p for p in _temp_candidates if p.exists()), None)
if _temp_file is not None:
    class_temperatures = np.load(_temp_file).astype(np.float32)
    print(f"Loaded per-class temperatures from 0.943 dataset: mean={class_temperatures.mean():.3f}")
else:
    print("Per-class temps not found, using per-taxa defaults")
    class_temperatures = np.ones(N_CLASSES, dtype=np.float32) * T_AVES
    for ci, label in enumerate(PRIMARY_LABELS):
        cn = CLASS_NAME_MAP.get(label, "Aves")
        if cn in TEXTURE_TAXA:
            class_temperatures[ci] = T_TEXTURE
print(f"Temperature: Aves={T_AVES}, Texture={T_TEXTURE}")

scaled_scores = final_test_scores / class_temperatures[None, :]
probs = sigmoid(scaled_scores)

# v28: File-level confidence scaling (from timeoptimized notebook)
top_k = CFG.get("file_level_top_k", 0)
if top_k > 0:
    print(f"Applying file-level confidence scaling (top_k={top_k})")
    probs = file_level_confidence_scale(probs, n_windows=N_WINDOWS, top_k=top_k)
    probs = np.clip(probs, 0.0, 1.0)

# Build submission
submission = pd.DataFrame(probs, columns=PRIMARY_LABELS)
submission.insert(0, "row_id", meta_test["row_id"].values)
submission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].astype(np.float32)

expected_rows = len(test_paths) * N_WINDOWS
assert len(submission) == expected_rows, f"Expected {expected_rows}, got {len(submission)}"
assert submission.columns.tolist() == ["row_id"] + PRIMARY_LABELS
assert not submission.isna().any().any()
assert (submission[PRIMARY_LABELS] >= 0).all().all() and (submission[PRIMARY_LABELS] <= 1).all().all()

submission.to_csv("submission.csv", index=False)

print(f"Saved submission.csv -- shape: {submission.shape}")
print(f"Prob range: [{probs.min():.6f}, {probs.max():.6f}]")
print(f"Mean prob: {probs.mean():.4f}")
print(f"Total wall time: {(time.time()-_WALL_START)/60:.1f} min")
print(submission.iloc[:3, :8])


# %% cell 21
# Cell 19 -- Final diagnostics
if MODE == "train":
    print("Modeled class count:", len(probe_models))
else:
    print("v28 submit mode completed.")
    print(f"Total wall time: {(time.time()-_WALL_START)/60:.1f} min")

