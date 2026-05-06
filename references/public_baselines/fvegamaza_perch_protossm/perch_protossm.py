# %% cell 1
import subprocess, sys
from pathlib import Path

_WHL = Path('/kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel')
if _WHL.exists():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
        str(_WHL / 'tensorboard-2.20.0-py3-none-any.whl')], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
        str(_WHL / 'tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl')], check=True)
    print('Installed TF 2.20.0 from Kaggle dataset wheels.')
else:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
        'tensorflow==2.20.0', 'tensorboard==2.20.0'], check=True)
    print('Installed TF 2.20.0 from PyPI.')

# %% cell 2
# Cell 1 â€” Mode switch
MODE = "submit" 

assert MODE in {"train", "submit"}

print("MODE =", MODE)

# %% cell 3
# Cell 2 â€” Imports and run config
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gc
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

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

BASE = Path("/kaggle/input/competitions/birdclef-2026")
MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12

DEVICE = torch.device("cpu")  # Competition constraint

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
    "require_full_cache_in_submit": True,
    # Candidates checked in order; first match wins
    "cache_input_dirs": [
        Path("/kaggle/input/datasets/jaejohn/perch-meta"),
        Path("/kaggle/input/perch-meta"),
    ],
    "full_cache_work_dir": Path("/kaggle/working/perch_cache"),

    # frozen baseline fusion params
    "best_fusion": {
        "lambda_event": 0.4,
        "lambda_texture": 1.0,
        "lambda_proxy_texture": 0.8,
        "smooth_texture": 0.35,
        "smooth_event": 0.15,
    },

    # ProtoSSM architecture
    "proto_ssm": {
        "d_model": 128,
        "d_state": 16,
        "n_ssm_layers": 2,
        "dropout": 0.15,
        "n_prototypes": 1,  # per class
    },

    # ProtoSSM training
    "proto_ssm_train": {
        "n_epochs": 120,
        "lr": 2e-3,
        "weight_decay": 1e-3,
        "val_ratio": 0.15,
        "patience": 20,
        "pos_weight_cap": 30.0,
        "distill_weight": 0.3,     # weight for Perch logit distillation loss
        "proto_margin": 0.1,       # margin for prototypical contrastive loss
    },

    # frozen probe params (fallback / comparison)
    "frozen_best_probe": {
        "pca_dim": 64,
        "min_pos": 8,
        "C": 0.50,
        "alpha": 0.40,
    },

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

CFG["full_cache_work_dir"].mkdir(parents=True, exist_ok=True)

print("TensorFlow:", tf.__version__)
print("PyTorch:", torch.__version__)
print("Competition dir exists:", BASE.exists())
print("Model dir exists:", MODEL_DIR.exists())

# %% cell 4
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

# %% cell 5
# Cell 3 â€” Load Perch, mapping, and selective frog proxies
BEST = CFG["best_fusion"]
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

# %% cell 6
# Cell 4 â€” Metrics and helper utilities
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

# %% cell 7
# Cell 5 â€” Perch inference with embeddings + selective proxies
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

        for path in batch_paths:
            y = read_soundscape_60s(path)
            x[x_pos:x_pos + N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)

            meta = parse_soundscape_filename(path.name)
            stem = path.stem

            row_ids[write_row:write_row + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
            filenames[write_row:write_row + N_WINDOWS] = path.name
            sites[write_row:write_row + N_WINDOWS] = meta["site"]
            hours[write_row:write_row + N_WINDOWS] = int(meta["hour_utc"])

            x_pos += N_WINDOWS
            write_row += N_WINDOWS

        outputs = infer_fn(inputs=tf.convert_to_tensor(x))
        logits = outputs["label"].numpy().astype(np.float32, copy=False)
        emb = outputs["embedding"].numpy().astype(np.float32, copy=False)

        scores[batch_row_start:write_row, MAPPED_POS] = logits[:, MAPPED_BC_INDICES]
        embeddings[batch_row_start:write_row] = emb

        # Selected frog proxies
        for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
            sub = logits[:, bc_idx_arr]
            if proxy_reduce == "max":
                proxy_score = sub.max(axis=1)
            elif proxy_reduce == "mean":
                proxy_score = sub.mean(axis=1)
            else:
                raise ValueError("proxy_reduce must be 'max' or 'mean'")
            scores[batch_row_start:write_row, pos] = proxy_score.astype(np.float32)

        del x, outputs, logits, emb
        gc.collect()

    meta_df = pd.DataFrame({
        "row_id": row_ids,
        "filename": filenames,
        "site": sites,
        "hour_utc": hours,
    })

    return meta_df, scores, embeddings

# %% cell 8
# Cell 6 â€” Load or compute full-file Perch cache
def resolve_full_cache_paths():
    candidates = []

    # Pre-loaded input dataset (checked first: most reliable)
    for d in CFG["cache_input_dirs"]:
        if d.exists():
            candidates.append((
                d / "full_perch_meta.parquet",
                d / "full_perch_arrays.npz",
            ))

    # Working dir (computed earlier in same session)
    candidates.append((
        CFG["full_cache_work_dir"] / "full_perch_meta.parquet",
        CFG["full_cache_work_dir"] / "full_perch_arrays.npz",
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
        print(f"  proxy_reduce={pr!r:6s} â†’ OOF baseline AUC = {auc:.6f}")

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

# %% cell 9
# Cell 7 â€” Fold-safe metadata prior tables
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

# %% cell 10
# Cell 8 â€” Honest OOF base/prior meta-features (required for final stacker fit)
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

# %% cell 11
# Cell 9 â€” Classwise embedding-probe helpers
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

# %% cell 12
# ProtoSSM â€” Prototypical State Space Model

class SelectiveSSM(nn.Module):
    """Simplified Mamba-style selective state space model.
    
    Input-dependent (selective) discretization of continuous-time SSM:
        dx/dt = Ax + Bu,  y = Cx + Du
    where A, B, C are functions of the input (selectivity).
    
    For T=12 bioacoustic windows, the sequential scan is efficient on CPU.
    """
    
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Input projection: x -> (x_ssm, z_gate)
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        
        # Causal conv1d for local context before SSM
        self.conv1d = nn.Conv1d(
            d_model, d_model, d_conv,
            padding=d_conv - 1, groups=d_model
        )
        
        # SSM parameters
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        
        # A initialized as structured matrix (HiPPO-inspired)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D is the skip connection
        self.D = nn.Parameter(torch.ones(d_model))
        
        # B and C projections â€” input-dependent = selective
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        """x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)"""
        B_size, T, D = x.shape
        
        # Split into SSM path and gate
        xz = self.in_proj(x)  # (B, T, 2D)
        x_ssm, z = xz.chunk(2, dim=-1)
        
        # Causal conv1d
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # Compute input-dependent SSM parameters
        dt = F.softplus(self.dt_proj(x_conv))  # (B, T, D)
        B_t = self.B_proj(x_conv)               # (B, T, N)
        C_t = self.C_proj(x_conv)               # (B, T, N)
        A = -torch.exp(self.A_log)               # (D, N), negative for stability
        
        # Sequential scan (efficient for T=12)
        y = self._selective_scan(x_conv, dt, A, B_t, C_t)
        
        # Gated output
        y = y * F.silu(z)
        return self.out_proj(y)
    
    def _selective_scan(self, x, dt, A, B, C):
        """Selective scan with input-dependent discretization.
        
        x:  (batch, T, D)
        dt: (batch, T, D) â€” step sizes
        A:  (D, N) â€” state matrix (log-space, already negated)
        B:  (batch, T, N) â€” input matrix
        C:  (batch, T, N) â€” output matrix
        """
        batch, T, D = x.shape
        N = self.d_state
        
        h = torch.zeros(batch, D, N, device=x.device, dtype=x.dtype)
        ys = []
        
        for t in range(T):
            dt_t = dt[:, t, :, None]          # (batch, D, 1)
            dA = torch.exp(A[None] * dt_t)     # (batch, D, N)
            dB = dt_t * B[:, t, None, :]       # (batch, D, N)
            
            h = h * dA + x[:, t, :, None] * dB  # state update
            y_t = (h * C[:, t, None, :]).sum(-1) # output projection
            ys.append(y_t)
        
        y = torch.stack(ys, dim=1)  # (batch, T, D)
        return y + x * self.D[None, None, :]  # skip connection


class ProtoSSM(nn.Module):
    """Prototypical State Space Model for temporal bioacoustic event detection.
    
    Architecture:
    1. Linear projection of Perch embeddings (1536 -> d_model)
    2. Bidirectional Selective SSM for temporal context
    3. Prototypical cosine similarity classification
    4. Gated fusion with Perch foundation model logits
    
    Args:
        d_input: Perch embedding dimension (1536)
        d_model: Internal model dimension
        d_state: SSM state dimension
        n_ssm_layers: Number of bidirectional SSM layers
        n_classes: Number of species classes (234)
        n_windows: Windows per file (12)
        dropout: Dropout rate
    """
    
    def __init__(self, d_input=1536, d_model=128, d_state=16,
                 n_ssm_layers=2, n_classes=234, n_windows=12, dropout=0.15):
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
        
        # 2. Learnable positional encoding for temporal position within file
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        
        # 3. Bidirectional SSM layers with residual connections
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
        
        # 4. Learnable class prototypes (initialized from data)
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        
        # 5. Per-class gated fusion with Perch logits
        #    sigmoid(alpha) blends: alpha*proto + (1-alpha)*perch
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))
        
        # 6. Taxonomic auxiliary head (set after loading taxonomy)
        self.n_families = 0
        self.family_head = None
    
    def init_prototypes_from_data(self, embeddings, labels):
        """Initialize prototypes as normalized class-mean embeddings.
        
        embeddings: (N, d_input) raw Perch embeddings  
        labels: (N, n_classes) binary label matrix
        """
        with torch.no_grad():
            h = self.input_proj(embeddings)  # (N, d_model)
            for c in range(self.n_classes):
                mask = labels[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)
    
    def init_family_head(self, n_families, class_to_family):
        """Initialize taxonomic auxiliary head.
        
        n_families: number of unique families
        class_to_family: (n_classes,) mapping class index to family index
        """
        self.n_families = n_families
        self.family_head = nn.Linear(self.d_model, n_families)
        self.register_buffer('class_to_family', torch.tensor(class_to_family, dtype=torch.long))
    
    def forward(self, emb, perch_logits=None):
        """
        emb: (B, T, d_input) â€” Perch embeddings per file
        perch_logits: (B, T, n_classes) â€” Perch mapped logits (optional)
        
        Returns: 
            species_logits: (B, T, n_classes)
            family_logits: (B, T, n_families) or None
            h_temporal: (B, T, d_model) â€” for analysis
        """
        B, T, _ = emb.shape
        
        # Project embeddings
        h = self.input_proj(emb)  # (B, T, d_model)
        h = h + self.pos_enc[:, :T, :]
        
        # Bidirectional SSM
        for fwd, bwd, merge, norm in zip(
            self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm
        ):
            residual = h
            h_f = fwd(h)                        # forward scan
            h_b = bwd(h.flip(1)).flip(1)         # backward scan
            h = merge(torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = norm(h + residual)               # residual + layernorm
        
        h_temporal = h  # save for analysis
        
        # Prototypical cosine similarity
        h_norm = F.normalize(h, dim=-1)          # (B, T, d_model)
        p_norm = F.normalize(self.prototypes, dim=-1)  # (C, d_model)
        temp = F.softplus(self.proto_temp)
        sim = torch.matmul(h_norm, p_norm.T) * temp    # (B, T, C)
        
        # Gated fusion with Perch logits
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]  # (1, 1, C)
            species_logits = alpha * sim + (1 - alpha) * perch_logits
        else:
            species_logits = sim
        
        # Taxonomic auxiliary prediction
        family_logits = None
        if self.family_head is not None:
            h_pool = h.mean(dim=1)  # (B, d_model) â€” file-level
            family_logits = self.family_head(h_pool)  # (B, n_families)
        
        return species_logits, family_logits, h_temporal
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


print("ProtoSSM architecture defined.")
print(f"Parameter count (d_model=128, 2 layers): {ProtoSSM(d_model=128, n_ssm_layers=2).count_parameters():,}")


# %% cell 13
# ProtoSSM Training Loop

def build_family_mapping(taxonomy_df, primary_labels):
    """Build class-to-family index mapping from taxonomy."""
    if "family" not in taxonomy_df.columns:
        # Derive family from class_name or use order as fallback
        if "order" in taxonomy_df.columns:
            family_map = taxonomy_df.set_index("primary_label")["order"].to_dict()
        elif "class_name" in taxonomy_df.columns:
            family_map = taxonomy_df.set_index("primary_label")["class_name"].to_dict()
        else:
            family_map = {label: "Unknown" for label in primary_labels}
    else:
        family_map = taxonomy_df.set_index("primary_label")["family"].to_dict()
    families = sorted(set(family_map.values()))
    fam_to_idx = {f: i for i, f in enumerate(families)}
    class_to_family = []
    for label in primary_labels:
        fam = family_map.get(label, "Unknown")
        class_to_family.append(fam_to_idx.get(fam, 0))
    return len(families), class_to_family, fam_to_idx

def reshape_to_files(flat_array, meta_df, n_windows=N_WINDOWS):
    """Reshape flat (n_windows*n_files, ...) to (n_files, n_windows, ...).
    
    Groups by filename from meta_df, preserving file order.
    Returns reshaped array and list of unique filenames.
    """
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

def train_proto_ssm(model, emb_files, logits_files, labels_files, 
                    file_families=None, cfg=None, verbose=True):
    """Train ProtoSSM with multi-task loss and early stopping.
    
    Args:
        model: ProtoSSM instance
        emb_files: (n_files, n_windows, d_input) Perch embeddings
        logits_files: (n_files, n_windows, n_classes) Perch mapped logits
        labels_files: (n_files, n_windows, n_classes) binary labels
        file_families: (n_files, n_families) multi-hot family labels (optional)
        cfg: training config dict
    
    Returns:
        model with best weights loaded
        training history dict
    """
    if cfg is None:
        cfg = CFG["proto_ssm_train"]
    
    n_files = len(emb_files)
    n_val = max(1, int(n_files * cfg["val_ratio"]))
    
    # Deterministic train/val split by file
    perm = torch.randperm(n_files, generator=torch.Generator().manual_seed(42))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    
    # Convert to tensors
    emb_train = torch.tensor(emb_files[train_idx], dtype=torch.float32)
    logits_train = torch.tensor(logits_files[train_idx], dtype=torch.float32)
    labels_train = torch.tensor(labels_files[train_idx], dtype=torch.float32)
    
    emb_val = torch.tensor(emb_files[val_idx], dtype=torch.float32)
    logits_val = torch.tensor(logits_files[val_idx], dtype=torch.float32)
    labels_val = torch.tensor(labels_files[val_idx], dtype=torch.float32)
    
    # Family labels for auxiliary loss
    fam_train = fam_val = None
    if file_families is not None and model.family_head is not None:
        fam_train = torch.tensor(file_families[train_idx], dtype=torch.float32)
        fam_val = torch.tensor(file_families[val_idx], dtype=torch.float32)
    
    # Class weights for imbalanced data
    pos_counts = labels_train.sum(dim=(0, 1))  # (C,)
    total = labels_train.shape[0] * labels_train.shape[1]
    pos_weight = ((total - pos_counts) / (pos_counts + 1)).clamp(max=cfg["pos_weight_cap"])
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"], 
        epochs=cfg["n_epochs"], steps_per_epoch=1,
        pct_start=0.1, anneal_strategy='cos'
    )
    
    best_val_loss = float('inf')
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    
    for epoch in range(cfg["n_epochs"]):
        # === Train ===
        model.train()
        species_out, family_out, _ = model(emb_train, logits_train)
        
        # Primary loss: weighted BCE
        loss_bce = F.binary_cross_entropy_with_logits(
            species_out, labels_train,
            pos_weight=pos_weight[None, None, :]
        )
        
        # Knowledge distillation loss: MSE between model output and Perch logits
        loss_distill = F.mse_loss(species_out, logits_train)
        
        # Total loss
        loss = loss_bce + cfg["distill_weight"] * loss_distill
        
        # Taxonomic auxiliary loss
        if family_out is not None and fam_train is not None:
            loss_family = F.binary_cross_entropy_with_logits(family_out, fam_train)
            loss = loss + 0.1 * loss_family
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # === Validate ===
        model.eval()
        with torch.no_grad():
            val_out, val_fam, _ = model(emb_val, logits_val)
            val_loss = F.binary_cross_entropy_with_logits(
                val_out, labels_val,
                pos_weight=pos_weight[None, None, :]
            )
            
            # Compute validation AUC
            val_pred = val_out.reshape(-1, val_out.shape[-1]).numpy()
            val_true = labels_val.reshape(-1, labels_val.shape[-1]).numpy()
            try:
                val_auc = macro_auc_skip_empty(val_true, val_pred)
            except Exception:
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
            print(f"  Epoch {epoch+1:3d}: train={loss.item():.4f} val={val_loss.item():.4f} "
                  f"auc={val_auc:.4f} lr={lr_now:.6f} wait={wait}")
        
        if wait >= cfg["patience"]:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    if verbose:
        print(f"  Training complete. Best val_loss={best_val_loss:.4f}")
        # Report fusion alpha distribution
        with torch.no_grad():
            alphas = torch.sigmoid(model.fusion_alpha).numpy()
            print(f"  Fusion alpha: mean={alphas.mean():.3f} min={alphas.min():.3f} max={alphas.max():.3f}")
            print(f"  Proto temperature: {F.softplus(model.proto_temp).item():.3f}")
    
    return model, history

print("ProtoSSM training functions defined.")


# %% cell 14
# Cell 10 â€” Probe tuning (train mode only)
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

# %% cell 15
# Cell 11 â€” Freeze final probe params
if BEST_PROBE is None:
    BEST_PROBE = CFG["frozen_best_probe"]

print("Final BEST_PROBE =", BEST_PROBE)

# Optional â€” rerun best OOF probe once for diagnostics / caching
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

# %% cell 16
# Cell 12 â€” Fit final prior tables on all labeled soundscapes
final_prior_tables = fit_prior_tables(sc_clean.reset_index(drop=True), Y_SC)

print("Built final prior tables for inference.")
print("OOF baseline AUC used for stacker training:", baseline_oof_auc)

# %% cell 17
# Cell 13 â€” Fit embedding scaler + PCA on all trusted full windows
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

# %% cell 18
# Instantiate and train ProtoSSM

# Reshape training data to file-level tensors
emb_files, file_list = reshape_to_files(emb_full, meta_full)
logits_files, _ = reshape_to_files(scores_full_raw, meta_full)
labels_files, _ = reshape_to_files(Y_FULL, meta_full)

print(f"Reshaped to file-level: emb={emb_files.shape}, logits={logits_files.shape}, labels={labels_files.shape}")
print(f"Files: {len(file_list)}")

# Build family mapping for taxonomic auxiliary loss
n_families, class_to_family, fam_to_idx = build_family_mapping(taxonomy, PRIMARY_LABELS)
print(f"Taxonomic families: {n_families}")

# Build per-file family labels (multi-hot)
file_families = np.zeros((len(file_list), n_families), dtype=np.float32)
for fi in range(len(file_list)):
    active_classes = np.where(labels_files[fi].sum(axis=0) > 0)[0]
    for ci in active_classes:
        file_families[fi, class_to_family[ci]] = 1.0

# Instantiate model
ssm_cfg = CFG["proto_ssm"]
model = ProtoSSM(
    d_input=emb_full.shape[1],  # 1536
    d_model=ssm_cfg["d_model"],
    d_state=ssm_cfg["d_state"],
    n_ssm_layers=ssm_cfg["n_ssm_layers"],
    n_classes=N_CLASSES,
    n_windows=N_WINDOWS,
    dropout=ssm_cfg["dropout"],
).to(DEVICE)

# Initialize prototypes from class-mean embeddings
emb_flat_tensor = torch.tensor(emb_full, dtype=torch.float32)
labels_flat_tensor = torch.tensor(Y_FULL, dtype=torch.float32)
model.init_prototypes_from_data(emb_flat_tensor, labels_flat_tensor)
print("Prototypes initialized from class means.")

# Initialize taxonomic head
model.init_family_head(n_families, class_to_family)
print(f"Family head initialized: {n_families} families")

print(f"ProtoSSM parameters: {model.count_parameters():,}")

# Train
t0 = time.time()
model, train_history = train_proto_ssm(
    model,
    emb_files, logits_files, labels_files.astype(np.float32),
    file_families=file_families,
    cfg=CFG["proto_ssm_train"],
    verbose=True,
)
train_time = time.time() - t0
print(f"ProtoSSM training time: {train_time:.1f}s")

# Also train MLP probes as ensemble component (same as V11)
# Keep the proven MLP probe pipeline for ensemble with ProtoSSM
PROBE_CLASS_IDX = np.where(Y_FULL.sum(axis=0) >= int(CFG["frozen_best_probe"]["min_pos"]))[0].astype(np.int32)

probe_models = {}
oof_base_for_probes = oof_base
oof_prior_for_probes = oof_prior

for cls_idx in tqdm(PROBE_CLASS_IDX, desc="Training MLP probes (ensemble)", disable=not CFG["verbose"]):
    y = Y_FULL[:, cls_idx]
    if y.sum() == 0 or y.sum() == len(y):
        continue
    
    X_cls = build_class_features(
        Z_FULL,
        raw_col=scores_full_raw[:, cls_idx],
        prior_col=oof_prior_for_probes[:, cls_idx],
        base_col=oof_base_for_probes[:, cls_idx],
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


# %% cell 19
# Cell 15 â€” Diagnostics
if MODE == "train":
    if grid_results is not None:
        best_row = grid_results.iloc[0]
        print(f"Best honest OOF probe AUC: {best_row['probe_oof_auc']:.6f}")
        print(f"Delta over honest OOF baseline: {best_row['delta']:.6f}")
else:
    print("Skipping train diagnostics in submit mode.")

# %% cell 20
# Cell 16 â€” Infer Perch on hidden test with embeddings
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


# %% cell 21
# Score Fusion: ProtoSSM + MLP Probes + Priors

# --- Step 1: ProtoSSM inference on test ---
emb_test_files, test_file_list = reshape_to_files(emb_test, meta_test)
logits_test_files, _ = reshape_to_files(scores_test_raw, meta_test)

emb_test_tensor = torch.tensor(emb_test_files, dtype=torch.float32)
logits_test_tensor = torch.tensor(logits_test_files, dtype=torch.float32)

model.eval()
with torch.no_grad():
    proto_out, _, _ = model(emb_test_tensor, logits_test_tensor)
    proto_scores = proto_out.numpy()  # (n_files, n_windows, n_classes)

# Flatten back to (n_rows, n_classes)
proto_scores_flat = proto_scores.reshape(-1, N_CLASSES).astype(np.float32)

print(f"ProtoSSM test scores: {proto_scores_flat.shape}")
print(f"Score range: {proto_scores_flat.min():.3f} to {proto_scores_flat.max():.3f}")

# --- Step 2: Prior-fused base scores (same as V11) ---
test_base_scores, test_prior_scores = fuse_scores_with_tables(
    scores_test_raw,
    sites=meta_test["site"].to_numpy(),
    hours=meta_test["hour_utc"].to_numpy(),
    tables=final_prior_tables,
)

# --- Step 3: MLP probe scores (same as V11) ---
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

# --- Step 4: Ensemble fusion ---
# ProtoSSM handles temporal context + Perch distillation
# MLP probes handle per-class feature engineering + prior fusion
# Blend: 0.5 * ProtoSSM + 0.5 * MLP
ENSEMBLE_WEIGHT_PROTO = 0.5

final_test_scores = (
    ENSEMBLE_WEIGHT_PROTO * proto_scores_flat +
    (1.0 - ENSEMBLE_WEIGHT_PROTO) * mlp_scores
).astype(np.float32)

print(f"Final ensemble scores: {final_test_scores.shape}")
print(f"Score range: {final_test_scores.min():.3f} to {final_test_scores.max():.3f}")


# %% cell 22
# Cell 18 â€” Build submission with temperature scaling
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

TEMPERATURE = 1.10
submission = pd.DataFrame(sigmoid(final_test_scores / TEMPERATURE), columns=PRIMARY_LABELS)
submission.insert(0, "row_id", meta_test["row_id"].values)
submission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].astype(np.float32)

expected_rows = len(test_paths) * N_WINDOWS
assert len(submission) == expected_rows, f"Expected {expected_rows}, got {len(submission)}"
assert submission.columns.tolist() == ["row_id"] + PRIMARY_LABELS
assert not submission.isna().any().any()

submission.to_csv("submission.csv", index=False)

print("Saved submission.csv")
print("Submission shape:", submission.shape)
print(submission.iloc[:3, :8])


# %% cell 23
# Cell 19 â€” Diagnostics
if MODE == "train":
    print("=== ProtoSSM Training Summary ===")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Training time: {train_time:.1f}s")
    print(f"Final train loss: {train_history['train_loss'][-1]:.4f}")
    print(f"Best val loss: {min(train_history['val_loss']):.4f}")
    print(f"Best val AUC: {max(train_history['val_auc']):.4f}")
    
    # Fusion alpha analysis
    with torch.no_grad():
        alphas = torch.sigmoid(model.fusion_alpha).numpy()
        high_proto = (alphas > 0.5).sum()
        high_perch = (alphas <= 0.5).sum()
        print(f"\nFusion alpha distribution:")
        print(f"  ProtoSSM-dominant (alpha>0.5): {high_proto} classes")
        print(f"  Perch-dominant (alpha<=0.5): {high_perch} classes")
    
    print(f"\nMLP probes: {len(probe_models)} classes")
    
    print("\nSubmission probability stats:")
    print(submission.iloc[:, 1:].stack().describe())
else:
    print("Submit mode completed.")
    print(f"ProtoSSM parameters: {model.count_parameters():,}")

