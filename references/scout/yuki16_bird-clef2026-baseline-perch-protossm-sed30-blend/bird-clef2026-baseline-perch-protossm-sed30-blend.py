# %% cell 1
# -- Cell 0: Install ONNX Runtime (offline wheel) ---------------------------
# TF SavedModel is NOT needed since we use ONNX exclusively.
# Only the ONNX Runtime wheel from perch-onnx-for-birdclef-2026 is required.
import subprocess, sys, os
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")

# Locate ONNX Runtime wheel (search recursively under /kaggle/input)
onnx_whls = sorted(INPUT_ROOT.rglob("onnxruntime-*.whl"))
if onnx_whls:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--no-deps", str(onnx_whls[0])], check=True)
    print(f"ONNX Runtime installed: {onnx_whls[0].name}")
else:
    print("WARNING: ONNX Runtime wheel not found — trying pip install")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"], check=True)

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
    print(f"ONNX Runtime available: {ort.__version__}")
except ImportError:
    _ONNX_AVAILABLE = False
    raise RuntimeError("ONNX Runtime not available. Check dataset attachment.")


# %% cell 2
# -- Cell 1: Imports & config --------------------------------------------------
import os, re, gc, time, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import soundfile as sf
# import tensorflow as tf  # TF not required when ONNX is used
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.isotonic import IsotonicRegression
import concurrent.futures
from tqdm.auto import tqdm

# tf.experimental.numpy.experimental_enable_numpy_behavior()  # TF not required when ONNX is used
#try:
#     tf.config.set_visible_devices([], "GPU")  # TF not required when ONNX is used
#except Exception:
#    pass

_WALL_START = time.time()

# -- Paths --------------------------------------------------------------------
BASE      = Path("/kaggle/input/competitions/birdclef-2026")
MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")
WORK_DIR  = Path("/kaggle/working/cache")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# -- Audio config -------------------------------------------------------------
SR             = 32_000
WINDOW_SEC     = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES   = 60 * SR
N_WINDOWS      = 12

# -- Training config ----------------------------------------------------------
CFG = {
    "batch_files": 16,
    "run_oof": False,          # set True for local CV (slower)
    "verbose": False,
    "dryrun_n_files": 0,
    "proto_ssm_train": {
        "n_epochs": 40, "lr": 8e-4, "weight_decay": 1e-3,
        "patience": 8, "pos_weight_cap": 25.0,
        "distill_weight": 0.15, "mixup_alpha": 0.4,
    },
    "residual_ssm": {
        "d_model": 64, "d_state": 8, "n_epochs": 20, "lr": 8e-4, "patience": 6,
        "correction_weight": 0.30,
    },
    "mlp_params": {
        "hidden_layer_sizes": (128, 64), "max_iter": 200,
        "early_stopping": True, "validation_fraction": 0.15,
        "n_iter_no_change": 10, "random_state": 42,
        "learning_rate_init": 5e-4, "alpha": 0.005,
    },
}
print("Config loaded")
print(f"  ONNX available: {_ONNX_AVAILABLE}")

# %% cell 3
# -- Cell 2: Data loading & label parsing -------------------------------------
taxonomy          = pd.read_csv(BASE / "taxonomy.csv")
sample_sub        = pd.read_csv(BASE / "sample_submission.csv")
soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES      = len(PRIMARY_LABELS)
label_to_idx   = {c: i for i, c in enumerate(PRIMARY_LABELS)}

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

def parse_fname(name):
    m = FNAME_RE.match(name)
    if not m:
        return {"site": "unknown", "hour_utc": -1}
    _, site, _, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}

def union_labels(series):
    out = set()
    for x in series:
        if pd.notna(x):
            for t in str(x).split(";"):
                t = t.strip()
                if t:
                    out.add(t)
    return sorted(out)

# Build per-window label table from labeled soundscapes
sc = (soundscape_labels
      .groupby(["filename", "start", "end"])["primary_label"]
      .apply(union_labels)
      .reset_index(name="label_list"))

sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
sc["row_id"]  = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)

_meta = sc["filename"].apply(parse_fname).apply(pd.Series)
sc = pd.concat([sc, _meta], axis=1)

Y_SC = np.zeros((len(sc), N_CLASSES), dtype=np.uint8)
for i, lbls in enumerate(sc["label_list"]):
    for lbl in lbls:
        if lbl in label_to_idx:
            Y_SC[i, label_to_idx[lbl]] = 1

# Keep only files with all 12 windows labeled
windows_per_file = sc.groupby("filename").size()
full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
sc["fully_labeled"] = sc["filename"].isin(full_files)

full_rows = (sc[sc["fully_labeled"]]
             .sort_values(["filename", "end_sec"])
             .reset_index(drop=False))
Y_FULL = Y_SC[full_rows["index"].to_numpy()]

print(f"Classes: {N_CLASSES} | Fully-labeled files: {len(full_files)}")
print(f"Full-file windows: {len(full_rows)} | Active classes: {int((Y_FULL.sum(0) > 0).sum())}")

# %% cell 4
# -- Cell 3: Load Perch model (ONNX preferred) ---------------------------------
try:
    import tensorflow as tf
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    try: tf.config.set_visible_devices([], "GPU")
    except: pass
    birdclassifier = tf.saved_model.load(str(MODEL_DIR))
    infer_fn       = birdclassifier.signatures["serving_default"]
    print("TF SavedModel loaded")
except Exception as e:
    birdclassifier = None
    infer_fn = None
    print(f"TF SavedModel not loaded (ONNX will be used): {e}")

# Use ONNX if available (150x faster than TF SavedModel on CPU)
ONNX_PERCH_PATH = Path("/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/perch_v2.onnx")
USE_ONNX = _ONNX_AVAILABLE and ONNX_PERCH_PATH.exists()

if USE_ONNX:
    _so = ort.SessionOptions()
    _so.intra_op_num_threads = 4
    ONNX_SESSION    = ort.InferenceSession(str(ONNX_PERCH_PATH), sess_options=_so,
                                            providers=["CPUExecutionProvider"])
    ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
    ONNX_OUT_MAP    = {o.name: i for i, o in enumerate(ONNX_SESSION.get_outputs())}
    print("Using ONNX Perch (fast)")
else:
    print("Using TF SavedModel Perch (slow fallback)")

# Map competition species labels to Perch label indices
bc_labels = (pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
             .reset_index()
             .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"}))
NO_LABEL  = len(bc_labels)

mapping = (taxonomy
           .merge(bc_labels.rename(columns={"scientific_name": "scientific_name"}),
                  on="scientific_name", how="left"))
mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL).astype(int)
lbl2bc = mapping.set_index("primary_label")["bc_index"]

BC_INDICES    = np.array([int(lbl2bc.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)
MAPPED_MASK   = BC_INDICES != NO_LABEL
MAPPED_POS    = np.where(MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_IDX = BC_INDICES[MAPPED_MASK].astype(np.int32)
print(f"Mapped: {MAPPED_MASK.sum()} / {N_CLASSES} species have a Perch logit")

# Genus-level proxy logits for unmapped species
UNMAPPED_POS   = np.where(~MAPPED_MASK)[0].astype(np.int32)
CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
PROXY_TAXA     = {"Amphibia", "Insecta", "Aves"}
proxy_map      = {}

unmapped_df = taxonomy[taxonomy["primary_label"].isin(
    [PRIMARY_LABELS[i] for i in UNMAPPED_POS])].copy()

for _, row in unmapped_df.iterrows():
    target = row["primary_label"]
    genus  = str(row["scientific_name"]).split()[0]
    hits   = bc_labels[bc_labels["scientific_name"].astype(str)
                       .str.match(rf"^{re.escape(genus)}\s", na=False)]
    if len(hits) > 0 and CLASS_NAME_MAP.get(target) in PROXY_TAXA:
        proxy_map[label_to_idx[target]] = hits["bc_index"].astype(int).tolist()

print(f"Unmapped with genus proxy: {len(proxy_map)} / {len(UNMAPPED_POS)}")

# %% cell 5
# -- Cell 4: Perch inference engine -------------------------------------------

def read_60s(path):
    """Load a 60-second audio file as float32 mono array."""
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    else:
        y = y[:FILE_SAMPLES]
    return y


def run_perch(paths, batch_files=16, verbose=True):
    """
    Run Perch on a list of 60-second soundscape files.
    Returns (meta_df, scores[N*12, 234], embeddings[N*12, 1536]).
    Uses multithreaded I/O prefetch to overlap disk reads with inference.
    """
    paths  = [Path(p) for p in paths]
    n_rows = len(paths) * N_WINDOWS

    row_ids   = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites     = np.empty(n_rows, dtype=object)
    hours     = np.zeros(n_rows, dtype=np.int16)
    scores    = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embs      = np.zeros((n_rows, 1536),      dtype=np.float32)

    wr  = 0
    itr = tqdm(range(0, len(paths), batch_files), desc="Perch") if verbose else range(0, len(paths), batch_files)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io_exec:
        next_paths   = paths[0:batch_files]
        future_audio = [io_exec.submit(read_60s, p) for p in next_paths]

        for start in itr:
            batch_paths = next_paths
            batch_n     = len(batch_paths)
            batch_audio = [f.result() for f in future_audio]

            next_start = start + batch_files
            if next_start < len(paths):
                next_paths   = paths[next_start:next_start + batch_files]
                future_audio = [io_exec.submit(read_60s, p) for p in next_paths]

            x  = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            br = wr

            for bi, path in enumerate(batch_paths):
                y    = batch_audio[bi]
                meta = parse_fname(path.name)
                stem = path.stem
                x[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
                row_ids  [wr:wr + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
                filenames[wr:wr + N_WINDOWS] = path.name
                sites    [wr:wr + N_WINDOWS] = meta["site"]
                hours    [wr:wr + N_WINDOWS] = meta["hour_utc"]
                wr += N_WINDOWS

            # ONNX or TF inference
            if USE_ONNX:
                outs   = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: x})
                logits = outs[ONNX_OUT_MAP["label"]].astype(np.float32)
                emb    = outs[ONNX_OUT_MAP["embedding"]].astype(np.float32)
            else:
                out    = infer_fn(inputs=tf.convert_to_tensor(x))
                logits = out["label"].numpy().astype(np.float32)
                emb    = out["embedding"].numpy().astype(np.float32)

            scores[br:wr, MAPPED_POS] = logits[:, MAPPED_BC_IDX]
            embs  [br:wr]             = emb

            for pos_idx, bc_idxs in proxy_map.items():
                bc_arr = np.array(bc_idxs, dtype=np.int32)
                scores[br:wr, pos_idx] = logits[:, bc_arr].max(axis=1)

            del x, logits, emb, batch_audio
            gc.collect()

    meta_df = pd.DataFrame({"row_id": row_ids, "filename": filenames,
                             "site": sites, "hour_utc": hours})
    return meta_df, scores, embs


print("Perch inference engine defined")

# %% cell 6
# -- Cell 5: Build or load Perch training cache --------------------------------
print(f"USE_ONNX = {USE_ONNX}")

EXTERNAL_CACHE_DIRS = [
    Path("/kaggle/input/notebooks/vyankteshdwivedi/notebook1b25083f0d"),
    Path("/kaggle/input/datasets/jaejohn/perch-meta"),
]

CACHE_META_LOCAL = WORK_DIR / "perch_meta.parquet"
CACHE_NPZ_LOCAL  = WORK_DIR / "perch_arrays.npz"

SCORE_KEYS = ["scores", "sc", "logits", "perch_scores", "preds", "arr_0"]
EMB_KEYS   = ["embs", "emb", "embeddings", "features", "perch_embs", "arr_1"]


def _find_external_cache():
    for d in EXTERNAL_CACHE_DIRS:
        meta = d / "perch_meta.parquet"
        npz  = d / "perch_arrays.npz"
        if meta.exists() and npz.exists():
            return meta, npz
    return None, None


def _pick_array(arr, candidates, shape_hint_cols):
    for k in candidates:
        if k in arr.files:
            return arr[k], k
    for k in arr.files:
        v = arr[k]
        if v.ndim == 2 and v.shape[1] == shape_hint_cols:
            return v, k
    raise KeyError(f"None of {candidates} found. Available: {arr.files}")


def _build_cache():
    train_paths = [BASE / "train_soundscapes" / fn for fn in full_files
                   if (BASE / "train_soundscapes" / fn).exists()]
    print(f"Building cache from {len(train_paths)} train files...")
    t0 = time.time()
    meta_b, sc_b, emb_b = run_perch(train_paths, CFG["batch_files"], verbose=True)
    print(f"  Done in {time.time()-t0:.1f}s")
    meta_b.to_parquet(CACHE_META_LOCAL)
    np.savez(CACHE_NPZ_LOCAL,
             scores=sc_b.astype(np.float32),
             embs=emb_b.astype(np.float32),
             primary_labels=np.array(PRIMARY_LABELS))
    return CACHE_META_LOCAL, CACHE_NPZ_LOCAL


ext_meta, ext_npz = _find_external_cache()
if ext_meta is not None:
    CACHE_META, CACHE_NPZ = ext_meta, ext_npz
    print(f"Using external cache: {CACHE_META.parent}")
elif CACHE_META_LOCAL.exists() and CACHE_NPZ_LOCAL.exists():
    CACHE_META, CACHE_NPZ = CACHE_META_LOCAL, CACHE_NPZ_LOCAL
    print(f"Using local cache")
else:
    print("No cache found — building from scratch")
    CACHE_META, CACHE_NPZ = _build_cache()

meta_tr = pd.read_parquet(CACHE_META)
_arr    = np.load(CACHE_NPZ)

sc_tr_raw,  sk = _pick_array(_arr, SCORE_KEYS, N_CLASSES)
emb_tr_raw, ek = _pick_array(_arr, EMB_KEYS,   1536)
sc_tr  = sc_tr_raw.astype(np.float32)
emb_tr = emb_tr_raw.astype(np.float32)

# Align Y labels to cache row order
if "row_id" not in meta_tr.columns:
    end_sec = np.tile(np.arange(5, 65, 5), len(meta_tr) // N_WINDOWS)
    meta_tr["row_id"] = (meta_tr["filename"].str.replace(".ogg", "", regex=False)
                          + "_" + pd.Series(end_sec).astype(str))

row_id_to_index = full_rows.set_index("row_id")["index"]
Y_FULL_aligned  = Y_SC[row_id_to_index.loc[meta_tr["row_id"]].to_numpy()]

print(f"sc_tr: {sc_tr.shape}  emb_tr: {emb_tr.shape}  Y_FULL_aligned: {Y_FULL_aligned.shape}")

# %% cell 7
# -- Cell 6: Helper functions (prior, MLP probes, smoothing, calibration) ------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def macro_auc(y_true, y_score):
    """Competition metric: macro-averaged ROC-AUC, skipping all-negative classes."""
    keep = y_true.sum(axis=0) > 0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")


# ── Prior tables (site + hour frequency) ─────────────────────────────────────
def build_prior_tables(sc_df, Y_labels):
    sc_df    = sc_df.reset_index(drop=True)
    global_p = Y_labels.mean(axis=0).astype(np.float32)
    site_keys = sorted(sc_df["site"].dropna().astype(str).unique())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_p    = np.zeros((len(site_keys), Y_labels.shape[1]), dtype=np.float32)
    site_n    = np.zeros(len(site_keys), dtype=np.float32)
    for s in site_keys:
        i = site_to_i[s]
        mask = sc_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        if mask.sum() > 0:
            site_p[i] = Y_labels[mask].mean(axis=0)
    hour_keys = sorted(sc_df["hour_utc"].dropna().astype(int).unique())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_p    = np.zeros((len(hour_keys), Y_labels.shape[1]), dtype=np.float32)
    hour_n    = np.zeros(len(hour_keys), dtype=np.float32)
    for h in hour_keys:
        i = hour_to_i[h]
        mask = sc_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        if mask.sum() > 0:
            hour_p[i] = Y_labels[mask].mean(axis=0)
    return {"global_p": global_p, "site_to_i": site_to_i, "site_p": site_p,
            "site_n": site_n, "hour_to_i": hour_to_i, "hour_p": hour_p, "hour_n": hour_n}


def apply_prior(scores, sites, hours, tables, lambda_prior=0.4):
    """Add site/hour frequency prior (as logits) to raw Perch scores."""
    eps = 1e-4
    n   = len(scores)
    out = scores.copy()
    p   = np.tile(tables["global_p"], (n, 1))
    for i, h in enumerate(hours):
        h = int(h)
        if h in tables["hour_to_i"]:
            j  = tables["hour_to_i"][h]
            w  = tables["hour_n"][j] / (tables["hour_n"][j] + 8.0)
            p[i] = w * tables["hour_p"][j] + (1 - w) * tables["global_p"]
    for i, s in enumerate(sites):
        s = str(s)
        if s in tables["site_to_i"]:
            j  = tables["site_to_i"][s]
            w  = tables["site_n"][j] / (tables["site_n"][j] + 8.0)
            p[i] = w * tables["site_p"][j] + (1 - w) * p[i]
    p = np.clip(p, eps, 1 - eps)
    out += lambda_prior * (np.log(p) - np.log1p(-p))
    return out.astype(np.float32)


# ── Per-taxon temperature scaling ────────────────────────────────────────────
TEXTURE_TAXA = {"Amphibia", "Insecta"}
temperatures = np.array([
    0.95 if CLASS_NAME_MAP.get(l) in TEXTURE_TAXA else 1.10
    for l in PRIMARY_LABELS], dtype=np.float32)


# ── MLP probes on Perch embeddings ───────────────────────────────────────────
def build_sequential_features(scores_col, n_windows=12):
    N = len(scores_col)
    x     = scores_col.reshape(-1, n_windows)
    prev  = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
    next_ = np.concatenate([x[:, 1:], x[:, -1:]], axis=1)
    mean  = np.repeat(x.mean(axis=1), n_windows)
    max_  = np.repeat(x.max(axis=1),  n_windows)
    std   = np.repeat(x.std(axis=1),  n_windows)
    return prev.reshape(-1), next_.reshape(-1), mean, max_, std


def train_mlp_probes(emb, scores_raw, Y, min_pos=5, pca_dim=64, alpha_blend=0.4):
    """Train per-class MLP on PCA-compressed Perch embeddings."""
    scaler = StandardScaler()
    emb_s  = scaler.fit_transform(emb)
    pca    = PCA(n_components=min(pca_dim, emb_s.shape[1] - 1))
    Z      = pca.fit_transform(emb_s).astype(np.float32)
    print(f"Embedding PCA: {emb.shape} -> {Z.shape}  "
          f"variance={pca.explained_variance_ratio_.sum():.2%}")

    probe_models = {}
    active = np.where(Y.sum(axis=0) >= min_pos)[0]
    print(f"Training MLP probes for {len(active)} species...")
    MAX_ROWS = 3000

    for ci in tqdm(active, desc="MLP probes"):
        y = Y[:, ci]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        prev, next_, mean, max_, std = build_sequential_features(scores_raw[:, ci])
        X = np.hstack([Z, scores_raw[:, ci:ci+1],
                       prev[:, None], next_[:, None],
                       mean[:, None], max_[:, None], std[:, None]])
        n_pos = int(y.sum())
        pos_idx = np.where(y == 1)[0]
        repeat = min(8, max(1, (MAX_ROWS - len(y)) // max(n_pos, 1)))
        X_bal  = np.vstack([X, np.tile(X[pos_idx], (repeat, 1))])
        y_bal  = np.concatenate([y, np.ones(n_pos * repeat, dtype=y.dtype)])
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", max_iter=200,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=10, random_state=42,
            learning_rate_init=5e-4, alpha=0.005)
        clf.fit(X_bal, y_bal)
        probe_models[ci] = clf

    print(f"Trained {len(probe_models)} MLP probes")
    return probe_models, scaler, pca, alpha_blend


def apply_mlp_probes(emb_test, scores_test, probe_models, scaler, pca, alpha_blend=0.4):
    """Vectorized MLP probe inference using PyTorch batched matmul."""
    if len(probe_models) == 0:
        return scores_test.copy()
    emb_s  = scaler.transform(emb_test)
    Z_test = pca.transform(emb_s).astype(np.float32)
    result = scores_test.copy()
    for ci, clf in probe_models.items():
        prev, next_, mean, max_, std = build_sequential_features(scores_test[:, ci])
        X_test = np.hstack([Z_test, scores_test[:, ci:ci+1],
                             prev[:, None], next_[:, None],
                             mean[:, None], max_[:, None], std[:, None]])
        prob  = clf.predict_proba(X_test)[:, 1].astype(np.float32)
        logit = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
        result[:, ci] = (1 - alpha_blend) * scores_test[:, ci] + alpha_blend * logit
    return result


# ── Isotonic calibration ─────────────────────────────────────────────────────
def calibrate_thresholds(oof_probs, Y_FULL, n_windows=12):
    """Per-class isotonic regression calibration."""
    n_cls     = oof_probs.shape[1]
    n_files   = len(oof_probs) // n_windows
    file_oof  = oof_probs.reshape(n_files, n_windows, n_cls).max(axis=1)
    file_y    = Y_FULL.reshape(n_files, n_windows, n_cls).max(axis=1)
    grid      = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    thresholds = np.full(n_cls, 0.5, dtype=np.float32)
    for c in range(n_cls):
        y_true = file_y[:, c]
        if y_true.sum() < 3:
            continue
        try:
            ir    = IsotonicRegression(out_of_bounds="clip")
            y_cal = ir.fit_transform(file_oof[:, c], y_true)
        except Exception:
            y_cal = file_oof[:, c]
        best_f1, best_t = 0.0, 0.5
        for t in grid:
            pred = (y_cal >= t).astype(int)
            tp = ((pred==1)&(y_true==1)).sum()
            fp = ((pred==1)&(y_true==0)).sum()
            fn = ((pred==0)&(y_true==1)).sum()
            f1 = 2*tp / (2*tp + fp + fn + 1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds


# ── Post-processing helpers ───────────────────────────────────────────────────
def file_confidence_scale(probs, n_windows=12, top_k=2, power=0.4):
    """Suppress uncertain files by scaling down by top-k mean raised to power."""
    view      = probs.reshape(-1, n_windows, probs.shape[1])
    top_k_mean = np.sort(view, axis=1)[:, -top_k:, :].mean(axis=1, keepdims=True)
    return (view * np.power(top_k_mean, power)).reshape(probs.shape)


def adaptive_delta_smooth(probs, n_windows=12, base_alpha=0.20):
    """Smooth uncertain windows toward neighbors, leave confident windows intact."""
    result = probs.copy()
    view   = probs.reshape(-1, n_windows, probs.shape[1])
    out    = result.reshape(-1, n_windows, probs.shape[1])
    for t in range(n_windows):
        conf  = view[:, t, :].max(axis=-1, keepdims=True)
        alpha = base_alpha * (1.0 - conf)
        if t == 0:
            nbr = (view[:, t, :] + view[:, t+1, :]) / 2.0
        elif t == n_windows - 1:
            nbr = (view[:, t-1, :] + view[:, t, :]) / 2.0
        else:
            nbr = (view[:, t-1, :] + view[:, t+1, :]) / 2.0
        out[:, t, :] = (1.0 - alpha) * view[:, t, :] + alpha * nbr
    return result


print("Helper functions defined")

# %% cell 8
# -- Cell 7: LightProtoSSM with cross-attention + SWA -------------------------

class SelectiveSSM(nn.Module):
    """Simplified selective state space model layer (Mamba-inspired)."""
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj  = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d   = nn.Conv1d(d_model, d_model, d_conv,
                                   padding=d_conv - 1, groups=d_model)
        self.dt_proj  = nn.Linear(d_model, d_model, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log  = nn.Parameter(torch.log(A))
        self.D      = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B_sz, T, D = x.shape
        xz       = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_conv   = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv   = F.silu(x_conv)
        dt = F.softplus(self.dt_proj(x_conv))
        A  = -torch.exp(self.A_log)
        B  = self.B_proj(x_conv)
        C  = self.C_proj(x_conv)
        h  = torch.zeros(B_sz, D, self.d_state)
        ys = []
        for t in range(T):
            dA = torch.exp(A[None] * dt[:, t, :, None])
            dB = dt[:, t, :, None] * B[:, t, None, :]
            h  = h * dA + x[:, t, :, None] * dB
            ys.append((h * C[:, t, None, :]).sum(-1))
        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


class LightProtoSSM(nn.Module):
    """
    Bidirectional SSM sequence model over Perch embeddings.
    Learns temporal context across all 12 windows of a soundscape.
    Cross-attention heads let each window attend to all others.
    """
    def __init__(self, d_input=1536, d_model=128, d_state=16,
                 n_classes=234, n_windows=12, dropout=0.15,
                 n_sites=20, meta_dim=16,
                 use_cross_attn=True, cross_attn_heads=2):
        super().__init__()
        self.n_classes    = n_classes
        self.n_windows    = n_windows
        self.use_cross_attn = use_cross_attn

        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.pos_enc   = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb  = nn.Embedding(n_sites, meta_dim)
        self.hour_emb  = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        self.ssm_fwd   = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_bwd   = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_merge = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(2)])
        self.ssm_norm  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.drop      = nn.Dropout(dropout)

        if use_cross_attn:
            self.cross_attn = nn.ModuleList([
                nn.MultiheadAttention(d_model, num_heads=cross_attn_heads,
                                      dropout=dropout, batch_first=True)
                for _ in range(2)])
            self.cross_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])

        self.prototypes   = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp   = nn.Parameter(torch.tensor(5.0))
        self.class_bias   = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

    def init_prototypes(self, emb_tensor, labels_tensor):
        with torch.no_grad():
            h = self.input_proj(emb_tensor)
            for c in range(self.n_classes):
                mask = labels_tensor[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape
        h = self.input_proj(emb) + self.pos_enc[:, :T, :]
        if site_ids is not None and hours is not None:
            meta = self.meta_proj(torch.cat(
                [self.site_emb(site_ids), self.hour_emb(hours)], dim=-1))
            h = h + meta[:, None, :]
        for i, (fwd, bwd, merge, norm) in enumerate(zip(
                self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm)):
            res = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h   = self.drop(merge(torch.cat([h_f, h_b], dim=-1)))
            h   = norm(h + res)
            if self.use_cross_attn:
                attn_out, _ = self.cross_attn[i](h, h, h)
                h = self.cross_norm[i](h + attn_out)
        h_n  = F.normalize(h, dim=-1)
        p_n  = F.normalize(self.prototypes, dim=-1)
        sim  = (torch.matmul(h_n, p_n.T) * F.softplus(self.proto_temp)
                + self.class_bias[None, None, :])
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            return alpha * sim + (1 - alpha) * perch_logits
        return sim

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_light_proto_ssm(emb_full, scores_full, Y_full, meta_full,
                           n_epochs=40, patience=8, lr=1e-3, n_sites=20):
    """Train LightProtoSSM with SWA (Stochastic Weight Averaging)."""
    n_files = len(emb_full) // N_WINDOWS
    emb_f   = emb_full.reshape(n_files, N_WINDOWS, -1)
    log_f   = scores_full.reshape(n_files, N_WINDOWS, -1)
    lab_f   = Y_full.reshape(n_files, N_WINDOWS, -1).astype(np.float32)

    fnames  = meta_full["filename"].unique()
    sites_u = sorted(meta_full["site"].unique())
    site2i  = {s: i + 1 for i, s in enumerate(sites_u)}

    site_ids = np.array([
        min(site2i.get(meta_full.loc[meta_full["filename"]==fn,"site"].iloc[0], 0), n_sites-1)
        for fn in fnames], dtype=np.int64)
    hour_ids = np.array([
        int(meta_full.loc[meta_full["filename"]==fn,"hour_utc"].iloc[0]) % 24
        for fn in fnames], dtype=np.int64)

    model = LightProtoSSM(n_classes=N_CLASSES, n_sites=n_sites,
                          use_cross_attn=True, cross_attn_heads=2)
    model.init_prototypes(
        torch.tensor(emb_full, dtype=torch.float32),
        torch.tensor(Y_full,   dtype=torch.float32))
    print(f"LightProtoSSM params: {model.count_parameters():,}")

    emb_t  = torch.tensor(emb_f,    dtype=torch.float32)
    log_t  = torch.tensor(log_f,    dtype=torch.float32)
    lab_t  = torch.tensor(lab_f,    dtype=torch.float32)
    site_t = torch.tensor(site_ids, dtype=torch.long)
    hour_t = torch.tensor(hour_ids, dtype=torch.long)

    pos_cnt    = lab_t.sum(dim=(0, 1))
    total      = lab_t.shape[0] * lab_t.shape[1]
    pos_weight = ((total - pos_cnt) / (pos_cnt + 1)).clamp(max=25.0)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=n_epochs, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy="cos")

    swa_model   = torch.optim.swa_utils.AveragedModel(model)
    swa_start   = int(n_epochs * 0.65)
    swa_sched   = torch.optim.swa_utils.SWALR(opt, swa_lr=4e-4)
    best_loss, best_state, wait = float("inf"), None, 0

    for ep in range(n_epochs):
        model.train()
        out  = model(emb_t, log_t, site_ids=site_t, hours=hour_t)
        loss = (F.binary_cross_entropy_with_logits(
                    out, lab_t, pos_weight=pos_weight[None, None, :])
                + 0.15 * F.mse_loss(out, log_t))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep >= swa_start:
            swa_model.update_parameters(model); swa_sched.step()
        else:
            sched.step()

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if ep >= swa_start:
        torch.optim.swa_utils.update_bn(emb_t.unsqueeze(0), swa_model)
        model = swa_model
    else:
        model.load_state_dict(best_state)

    model.eval()
    print(f"LightProtoSSM done — best loss={best_loss:.4f}")
    return model, site2i


def run_tta_proto(proto_model, emb_files, sc_files, site_t, hour_t,
                  shifts=(0, 1, -1, 2, -2)):
    """TTA: average ProtoSSM predictions across circular time shifts."""
    proto_model.eval()
    all_preds = []
    emb_t = torch.tensor(emb_files, dtype=torch.float32)
    sc_t  = torch.tensor(sc_files,  dtype=torch.float32)
    for shift in shifts:
        e = torch.roll(emb_t, shift, dims=1) if shift else emb_t
        s = torch.roll(sc_t,  shift, dims=1) if shift else sc_t
        with torch.no_grad():
            out = proto_model(e, s, site_ids=site_t, hours=hour_t).numpy()
        if shift:
            out = np.roll(out, -shift, axis=1)
        all_preds.append(out)
    return np.mean(all_preds, axis=0)


print("LightProtoSSM defined")

# %% cell 9
# -- Cell 8: ResidualSSM (second-pass error correction) -----------------------

class ResidualSSM(nn.Module):
    """
    Lightweight second-pass model: learns to correct errors from first-pass.
    Output initialized to zero so corrections start small.
    """
    def __init__(self, d_input=1536, d_scores=234, d_model=64, d_state=8,
                 n_classes=234, n_windows=12, dropout=0.1, n_sites=20, meta_dim=8):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_input + d_scores, d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.site_emb  = nn.Embedding(n_sites, meta_dim)
        self.hour_emb  = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.pos_enc   = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.ssm_fwd   = SelectiveSSM(d_model, d_state)
        self.ssm_bwd   = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm  = nn.LayerNorm(d_model)
        self.ssm_drop  = nn.Dropout(dropout)
        self.output_head = nn.Linear(d_model, n_classes)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, emb, first_pass, site_ids=None, hours=None):
        B, T, _ = emb.shape
        x = torch.cat([emb, first_pass], dim=-1)
        h = self.input_proj(x) + self.pos_enc[:, :T, :]
        if site_ids is not None and hours is not None:
            meta = self.meta_proj(torch.cat(
                [self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings-1)),
                 self.hour_emb(hours.clamp(0, 23))], dim=-1))
            h = h + meta.unsqueeze(1)
        res = h
        h_f = self.ssm_fwd(h)
        h_b = self.ssm_bwd(h.flip(1)).flip(1)
        h   = self.ssm_drop(self.ssm_merge(torch.cat([h_f, h_b], dim=-1)))
        h   = self.ssm_norm(h + res)
        return self.output_head(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_residual_ssm(emb_full, first_pass_flat, Y_full,
                       site_ids, hour_ids,
                       n_epochs=20, patience=6, lr=1e-3, correction_weight=0.30):
    """Train ResidualSSM to predict (Y - sigmoid(first_pass))."""
    n_files   = len(emb_full) // N_WINDOWS
    emb_f     = emb_full.reshape(n_files, N_WINDOWS, -1)
    fp_f      = first_pass_flat.reshape(n_files, N_WINDOWS, -1)
    lab_f     = Y_full.reshape(n_files, N_WINDOWS, -1).astype(np.float32)
    fp_prob   = sigmoid(fp_f)
    residuals = lab_f - fp_prob

    n_val   = max(1, int(n_files * 0.15))
    perm    = torch.randperm(n_files, generator=torch.Generator().manual_seed(42)).numpy()
    val_i   = perm[:n_val]; train_i = perm[n_val:]

    emb_t = torch.tensor(emb_f,    dtype=torch.float32)
    fp_t  = torch.tensor(fp_f,     dtype=torch.float32)
    res_t = torch.tensor(residuals, dtype=torch.float32)
    site_t = torch.tensor(site_ids, dtype=torch.long)
    hour_t = torch.tensor(hour_ids, dtype=torch.long)

    model = ResidualSSM(n_classes=N_CLASSES)
    print(f"ResidualSSM params: {model.count_parameters():,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=n_epochs, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy="cos")

    best_loss, best_state, wait = float("inf"), None, 0
    for ep in range(n_epochs):
        model.train()
        corr = model(emb_t[train_i], fp_t[train_i],
                     site_ids=site_t[train_i], hours=hour_t[train_i])
        loss = F.mse_loss(corr, res_t[train_i])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(
                model(emb_t[val_i], fp_t[val_i],
                      site_ids=site_t[val_i], hours=hour_t[val_i]),
                res_t[val_i]).item()
        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    model.load_state_dict(best_state)
    print(f"ResidualSSM done — best val MSE={best_loss:.6f}")
    return model, correction_weight


print("ResidualSSM defined")

# %% cell 10
# -- Cell 9: Test inference (Perch) -------------------------------------------
test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))

if not test_paths:
    n = 20
    print(f"No hidden test files — dry-run on {n} train files")
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:n]
else:
    print(f"Hidden test files: {len(test_paths)}")

meta_te, sc_te, emb_te = run_perch(test_paths, CFG["batch_files"], verbose=True)
print(f"Test scores: {sc_te.shape}")

# %% cell 11
# -- Cell 10: Full ProtoSSM + ResidualSSM pipeline ----------------------------
N_SITES_CAP = 20

def get_site_hour_ids(meta_df, fnames, site2i, n_sites_cap=N_SITES_CAP):
    site_ids = np.array([
        min(site2i.get(meta_df.loc[meta_df["filename"]==fn,"site"].iloc[0], 0), n_sites_cap-1)
        for fn in fnames], dtype=np.int64)
    hour_ids = np.array([
        int(meta_df.loc[meta_df["filename"]==fn,"hour_utc"].iloc[0]) % 24
        for fn in fnames], dtype=np.int64)
    return site_ids, hour_ids


# Step A: Train LightProtoSSM
t0 = time.time()
proto_model, site2i_tr = train_light_proto_ssm(
    emb_tr, sc_tr, Y_FULL_aligned, meta_tr,
    n_epochs=40, patience=8, lr=1e-3)
print(f"ProtoSSM training: {time.time()-t0:.1f}s")

# Step B: ProtoSSM inference on TEST with TTA
n_test_files = len(sc_te) // N_WINDOWS
test_fnames  = meta_te.drop_duplicates("filename")["filename"].tolist()
test_site_ids, test_hour_ids = get_site_hour_ids(meta_te, test_fnames, site2i_tr)

proto_te_out = run_tta_proto(
    proto_model,
    emb_te.reshape(n_test_files, N_WINDOWS, -1),
    sc_te.reshape(n_test_files,  N_WINDOWS, -1),
    site_t=torch.tensor(test_site_ids, dtype=torch.long),
    hour_t=torch.tensor(test_hour_ids, dtype=torch.long),
    shifts=(0, 1, -1, 2, -2),
)
proto_scores_flat = proto_te_out.reshape(-1, N_CLASSES).astype(np.float32)

# Step C: Prior tables + MLP probes
prior_tables   = build_prior_tables(sc, Y_SC)
sc_te_adjusted = apply_prior(
    sc_te, sites=meta_te["site"].to_numpy(),
    hours=meta_te["hour_utc"].to_numpy(),
    tables=prior_tables, lambda_prior=0.4)

probe_models, emb_scaler, emb_pca, alpha_blend = train_mlp_probes(
    emb_tr, sc_tr, Y_FULL_aligned, min_pos=5, pca_dim=64, alpha_blend=0.4)
sc_te_adjusted = apply_mlp_probes(
    emb_te, sc_te_adjusted, probe_models, emb_scaler, emb_pca, alpha_blend)

# Step D: First-pass ensemble (ProtoSSM + MLP)
first_pass_flat = 0.5 * proto_scores_flat + 0.5 * sc_te_adjusted

# Step E: Build training first-pass for ResidualSSM calibration
tr_fnames = meta_tr.drop_duplicates("filename")["filename"].tolist()
tr_site_ids, tr_hour_ids = get_site_hour_ids(meta_tr, tr_fnames, site2i_tr)
n_tr_files = len(sc_tr) // N_WINDOWS

proto_tr_out  = run_tta_proto(
    proto_model,
    emb_tr.reshape(n_tr_files, N_WINDOWS, -1),
    sc_tr.reshape(n_tr_files,  N_WINDOWS, -1),
    site_t=torch.tensor(tr_site_ids, dtype=torch.long),
    hour_t=torch.tensor(tr_hour_ids, dtype=torch.long),
    shifts=(0, 1, -1, 2, -2),
)
proto_tr_flat = proto_tr_out.reshape(-1, N_CLASSES).astype(np.float32)

sc_tr_prior = apply_prior(
    sc_tr, sites=meta_tr["site"].to_numpy(),
    hours=meta_tr["hour_utc"].to_numpy(),
    tables=prior_tables, lambda_prior=0.4)
sc_tr_mlp   = apply_mlp_probes(
    emb_tr, sc_tr_prior, probe_models, emb_scaler, emb_pca, alpha_blend)
first_pass_tr = 0.5 * proto_tr_flat + 0.5 * sc_tr_mlp

# Step F: ResidualSSM
t0 = time.time()
res_model, correction_weight = train_residual_ssm(
    emb_tr, first_pass_tr, Y_FULL_aligned,
    tr_site_ids, tr_hour_ids,
    n_epochs=20, patience=6, lr=1e-3, correction_weight=0.30)
print(f"ResidualSSM training: {time.time()-t0:.1f}s")

# Apply correction to test
# ResidualSSM expects: emb(1536-dim) as 1st arg, first_pass(234-dim) as 2nd arg
res_model.eval()
with torch.no_grad():
    test_correction = res_model(
        torch.tensor(emb_te.reshape(n_test_files, N_WINDOWS, -1), dtype=torch.float32),
        torch.tensor(first_pass_flat.reshape(n_test_files, N_WINDOWS, -1), dtype=torch.float32),
        site_ids=torch.tensor(test_site_ids, dtype=torch.long),
        hours   =torch.tensor(test_hour_ids, dtype=torch.long),
    ).numpy()

final_scores = first_pass_flat + correction_weight * test_correction.reshape(-1, N_CLASSES)

# Step G: Temperature scaling + sigmoid + post-processing
final_scores = final_scores / temperatures[None, :]
probs        = sigmoid(final_scores)
probs        = file_confidence_scale(probs, n_windows=N_WINDOWS, top_k=2, power=0.4)
probs        = adaptive_delta_smooth(probs, n_windows=N_WINDOWS, base_alpha=0.20)
probs        = np.clip(probs, 0.0, 1.0)

# Save ProtoSSM submission
sub_proto = pd.DataFrame(probs.astype(np.float32), columns=PRIMARY_LABELS)
sub_proto.insert(0, "row_id", meta_te["row_id"].values)
sub_proto.to_csv("/kaggle/working/submission_protossm.csv", index=False)
print(f"ProtoSSM submission saved: {sub_proto.shape}")
print(f"Wall time so far: {(time.time()-_WALL_START)/60:.1f} min")

del emb_tr, sc_tr, proto_model, res_model
gc.collect()
print("Memory freed. Starting SED inference.")

# %% cell 12
# -- Cell 11: Distilled SED ONNX inference (5-fold) ---------------------------
N_MELS_SED = 256
N_FFT_SED  = 2048
HOP_SED    = 512
FMIN_SED   = 20
FMAX_SED   = 16000
TOP_DB_SED = 80


def find_sed_dir():
    hits = sorted(Path("/kaggle/input").rglob("sed_fold0.onnx"))
    if not hits:
        raise FileNotFoundError(
            "sed_fold0.onnx not found. "
            "Attach tuckerarrants/bc2026-distilled-sed-public to this notebook.")
    return hits[0].parent


def make_sed_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so,
                                 providers=["CPUExecutionProvider"])


def audio_to_mel_sed(chunks):
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(
            y=x, sr=SR, n_fft=N_FFT_SED, hop_length=HOP_SED,
            n_mels=N_MELS_SED, fmin=FMIN_SED, fmax=FMAX_SED, power=2.0)
        s = librosa.power_to_db(s, top_db=TOP_DB_SED)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)


def file_to_sed_chunks(path):
    y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr0 != SR:
        y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
    n = 60 * SR
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    else:
        y = y[:n]
    chunks = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
    ends   = np.arange(1, N_WINDOWS + 1) * WINDOW_SEC
    return chunks, ends


# Load 5 SED fold models
sed_dir   = find_sed_dir()
sed_paths = sorted(sed_dir.glob("sed_fold*.onnx"),
                   key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1)))
sed_sessions = [make_sed_session(p) for p in sed_paths]
print(f"SED folds loaded: {[p.name for p in sed_paths]}")

# Run SED on the same test files
sed_rows, sed_preds = [], []
_t0_sed = time.time()

for i, path in enumerate(test_paths, 1):
    chunks, ends = file_to_sed_chunks(path)
    mel   = audio_to_mel_sed(chunks)
    p_sum = np.zeros((len(chunks), N_CLASSES), dtype=np.float32)
    for sess in sed_sessions:
        outs       = sess.run(None, {sess.get_inputs()[0].name: mel})
        clip_logits = outs[0]
        frame_max   = outs[1].max(axis=1)
        p_sum += 0.5 * (1/(1+np.exp(-np.clip(clip_logits, -50, 50)))) \
               + 0.5 * (1/(1+np.exp(-np.clip(frame_max,   -50, 50))))
    p_mean = p_sum / len(sed_sessions)
    if len(p_mean) > 1:
        p_mean = gaussian_filter1d(p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
    stem = path.stem
    sed_rows.extend([f"{stem}_{int(t)}" for t in ends])
    sed_preds.append(p_mean)
    if i == 1 or i % 50 == 0 or i == len(test_paths):
        print(f"SED: {i}/{len(test_paths)} | {time.time()-_t0_sed:.1f}s")

sed_preds_arr = np.concatenate(sed_preds, axis=0)
sed_sub = pd.DataFrame(np.clip(sed_preds_arr, 0.0, 1.0), columns=PRIMARY_LABELS)
sed_sub.insert(0, "row_id", sed_rows)
sed_sub.to_csv("/kaggle/working/submission_sed.csv", index=False)
print(f"SED submission saved: {sed_sub.shape}")

# %% cell 13
# -- Cell 12: Multi-strategy blend -------------------------------------------
#
# Results so far:
#   rank_proto60_sed40  -> 0.939  (best)
#   rank_proto45_sed55  -> 0.937  (SED hurts: ProtoSSM > SED in quality)
#
# Next experiments (ProtoSSM-heavy + logit avg):
#   'rank_proto70_sed30' : rank-avg ProtoSSM 70% + SED 30%
#   'rank_proto80_sed20' : rank-avg ProtoSSM 80% + SED 20%
#   'logit_proto70_sed30': logit-avg ProtoSSM 70% + SED 30%  (no rank flattening)
#   'logit_proto60_sed40': logit-avg ProtoSSM 60% + SED 40%
#   'sed_only'           : SED standalone score check
#
BLEND_MODE = 'logit_proto70_sed30'
EPS = 1e-5

proto_df = pd.read_csv("/kaggle/working/submission_protossm.csv")
sed_df   = pd.read_csv("/kaggle/working/submission_sed.csv")

cols   = [c for c in proto_df.columns if c != "row_id"]
sed_df = sed_df.set_index("row_id").loc[proto_df["row_id"]].reset_index()

pa = np.clip(proto_df[cols].to_numpy(np.float32), EPS, 1 - EPS)
pb = np.clip(sed_df  [cols].to_numpy(np.float32), EPS, 1 - EPS)

# Logit (log-odds) representations — preserve prediction variance better than rank avg
la = np.log(pa / (1 - pa))
lb = np.log(pb / (1 - pb))

# Rank representations
ra = pd.DataFrame(pa).rank(axis=0, pct=True).to_numpy(np.float32)
rb = pd.DataFrame(pb).rank(axis=0, pct=True).to_numpy(np.float32)

blend_map = {
    # Rank-average variants (ProtoSSM-heavy)
    'rank_proto60_sed40' : (0.60 * ra + 0.40 * rb,                        "Rank avg Proto60/SED40 (baseline 0.939)"),
    'rank_proto70_sed30' : (0.70 * ra + 0.30 * rb,                        "Rank avg Proto70/SED30"),
    'rank_proto80_sed20' : (0.80 * ra + 0.20 * rb,                        "Rank avg Proto80/SED20"),
    # Logit-average variants — preserves high-confidence peaks
    'logit_proto60_sed40': (1/(1+np.exp(-(0.60*la + 0.40*lb))),           "Logit avg Proto60/SED40"),
    'logit_proto70_sed30': (1/(1+np.exp(-(0.70*la + 0.30*lb))),           "Logit avg Proto70/SED30"),
    # SED standalone
    'sed_only'           : (pb,                                            "SED only"),
}

blended, desc = blend_map[BLEND_MODE]

final_sub       = proto_df.copy()
final_sub[cols] = np.clip(blended, 0.0, 1.0).astype(np.float32)
final_sub.to_csv("/kaggle/working/submission.csv", index=False)

# Always save SED standalone for quick resubmission
sed_standalone       = proto_df[["row_id"]].copy()
sed_standalone[cols] = np.clip(pb, 0.0, 1.0).astype(np.float32)
sed_standalone.to_csv("/kaggle/working/submission_sed_standalone.csv", index=False)

print(f"submission.csv   : {final_sub.shape}")
print(f"Strategy         : {desc}")
print(f"BLEND_MODE       : {BLEND_MODE}")
print(f"Score range      : [{final_sub[cols].values.min():.4f}, {final_sub[cols].values.max():.4f}]")
print(f"Mean / Std       : {final_sub[cols].values.mean():.4f} / {final_sub[cols].values.std():.4f}")
print(f"Total wall time  : {(time.time()-_WALL_START)/60:.1f} min")
print()
print("To try other strategies, change BLEND_MODE and re-run this cell only:")
for k, (_, d) in blend_map.items():
    marker = " <- current" if k == BLEND_MODE else ""
    print(f"  '{k}': {d}{marker}")

