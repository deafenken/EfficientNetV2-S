# %% cell 1
!pip install -q --no-index \
    -r /kaggle/input/notebooks/ttahara/birdclef-2026-download-wheels/requirements.txt \
    --find-links=/kaggle/input/notebooks/ttahara/birdclef-2026-download-wheels/wheels

# %% cell 2
RANDOM_SEED = 1086

import os
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# %% cell 3
import gc
import math
import copy
import random
import typing as tp
from pathlib import Path

from time import time
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm, trange

import soundfile

import timm
import torch
import torchaudio
from torchvision.transforms import v2 as tvt_v2

from torch import nn

from joblib import load as jb_load, Parallel, delayed
from concurrent.futures import ThreadPoolExecutor

import openvino as ov

# %% cell 4
ROOT = Path.cwd().parent

INPUT = ROOT / "input"
DATA = INPUT / "competitions" / "birdclef-2026"
TRAIN_AUDIO = DATA / "train_audio"
TRAIN_SS = DATA / "train_soundscapes"
TEST_SS = DATA / "test_soundscapes"

TRAINED_MODEL = INPUT / "notebooks" / "ttahara/birdclef-2026-hgnetv2-b0-baseline-training"

N_FOLDS = 4
N_SPECIES = 234
N_CLASSES = 5

DEBUG = False
RANK_AVG = False

USE_OPENVINO = True

# %% cell 5
taxonomy = pd.read_csv(DATA / "taxonomy.csv")

SPECIES = taxonomy.primary_label.values.tolist()

label2idx = {label: idx for idx, label in enumerate(taxonomy.primary_label.values)}
idx2label = {idx: label for label, idx in label2idx.items()}

label2cls = {label: cls for label, cls in taxonomy[["primary_label", "class_name"]].values}

CLASSES = sorted(taxonomy["class_name"].unique())
cls2idx = {cls: idx for idx, cls in enumerate(CLASSES)}

# %% cell 6
def set_random_seed(seed: int = 42, deterministic: bool = True):
    """Set seeds"""
    os.environ["PYTHONHASHSEED"] = str(seed)  # python
    random.seed(seed)  # python
    np.random.seed(seed)  # cpu
    torch.manual_seed(seed)  # cpu
    if torch.cuda.is_available():  # gpu
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic

set_random_seed(RANDOM_SEED)

# %% cell 7
sample_sub = pd.read_csv(DATA / "sample_submission.csv")
IS_TEST_ENV = len(sample_sub) > 10

if IS_TEST_ENV:
    # about 600 audio files in test environment
    test_ss_paths = []
    added = set()
    for row_id in sample_sub["row_id"].values:
        file_id = "_".join(row_id.split("_")[:-1])
        if file_id in added:
            continue
        added.add(file_id)
        test_ss_paths.append(TEST_SS / f"{file_id}.ogg")
else:
    if DEBUG:
        # debug by 600 audio files in train_soundscapes 
        test_ss_paths = sorted(TRAIN_SS.iterdir())[:600]
    else:
        # fast submit
        test_ss_paths = sorted(TRAIN_SS.iterdir())[:10]

# %% cell 8
test_ss_segs = []
for p in test_ss_paths:
    for i in range(0, 60, 5):
        test_ss_segs.append(f"{p.stem}_{i + 5}")

# %% cell 9
for row_id in test_ss_segs[:24]:
    print(row_id)

# %% cell 10
class LogMelSpectrogramTransform(nn.Module):
    """"""
    def __init__(self, mel_spectrogram_params: tp.Dict, top_db: float, lms_shape=tp.Tuple[int, int]):
        """"""
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**mel_spectrogram_params)
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db)
        self.resize = tvt_v2.Resize(size=lms_shape)

    @torch.no_grad()
    def forward(self, wave):
        """
        wave: (B, sampling_rate * segment_sec)
        """
        mel_spec = self.mel_transform(wave)
        lms = self.db(mel_spec)  # shape: (B, n_mels, time)
        lms = self.resize(lms)   # shape: (B, *(lms_shape))
        
        batch_size = lms.shape[0]
        lms_flat = lms.reshape(batch_size, -1)
        lms_min = lms_flat.min(dim=1)[0][:, None, None]
        lms_max = lms_flat.max(dim=1)[0][:, None, None]
        lms = (lms - lms_min) / (lms_max - lms_min + 1e-7)

        return lms[:, None, :, :]

# %% cell 11
mel_spectrogram_params = dict(
    sample_rate= 32_000,
    n_fft      = 2048,
    win_length = 626,
    hop_length = 313,
    f_min      = 20,
    n_mels     = 256,
    power      = 2.0,
    center     = True,
    pad_mode   = "reflect",
    norm       = "slaney",
    mel_scale  = 'htk',
)
top_db = 80
lms_shape = (256, 512)  # for 10-sec clip

lms_transform = LogMelSpectrogramTransform(
    mel_spectrogram_params, top_db=top_db, lms_shape=lms_shape).eval()

# %% cell 12
wave_list = []
shifted_wave_list = []
sample_rate = 32_000 
max_sec = 60
duration = sample_rate * max_sec
shift = int(sample_rate * 2.5)

for path in tqdm(test_ss_paths):

    with soundfile.SoundFile(path) as f:
        n_frames = f.frames

        wave = np.zeros(shift + duration + shift, dtype="float32")

        if n_frames < duration + shift:
            wave[shift:shift + n_frames] = f.read(dtype="float32")
        else:
            wave[shift:shift + duration + shift] = f.read(frames=duration + shift, dtype="float32")
    
    for seg_sec in range(0, max_sec, 5):
        wave_list.append(
            wave[seg_sec * sample_rate: (seg_sec + 5) * sample_rate + shift * 2])


num_test_ss_audios = len(test_ss_paths)  
num_test_ss_segs = len(wave_list)
num_shifted_test_ss_segs = len(shifted_wave_list)

print(num_test_ss_segs, num_test_ss_segs / 12)

# %% cell 13
wave_batches = []
batch_size = 12

for i in trange(0, len(wave_list), batch_size):
    wave_batches.append(np.stack(wave_list[i: i + batch_size], axis=0))

# %% cell 14
lms_batches = Parallel(n_jobs=4, verbose=1)(
    delayed(lms_transform)(torch.from_numpy(waves)) for waves in wave_batches
)

# %% cell 15
if USE_OPENVINO:
    lms_batches = [
        np.ascontiguousarray(lms, dtype=np.float32) for lms in lms_batches]

del wave_batches
gc.collect()

# %% cell 16
if USE_OPENVINO:
    def compile_ov_model(ov_model_path):
        compiled_model = ov.compile_model(
            str(ov_model_path), "CPU", {
                "PERFORMANCE_HINT": "THROUGHPUT",
                "INFERENCE_NUM_THREADS": 4,
                'NUM_STREAMS': 2}
        )
        return compiled_model
    
    model_list = []
    for fold_id in range(N_FOLDS):
        compiled_model = compile_ov_model(
            # str(TRAINED_MODEL / f"best_model_fold{fold_id}_256x256.xml"))
            str(TRAINED_MODEL / f"best_model_fold{fold_id}_256x512.xml"))  # for 10-sec clip
        model_list.append(compiled_model)
        del compiled_model

else:

    class LSEPooling(nn.Module):
        """"""
        def __init__(
            self,
            pool_axis  : int   = 1,
            temperature: float = 1.0,
            trainable  : bool  = False, 
        ):
            """"""
            super().__init__()
            self.pool_axis = pool_axis
            log_T = torch.tensor(math.log(1.0))
            if trainable:
                self.log_T = nn.Parameter(log_T)
            else:
                self.register_buffer("log_T", log_T)
            
    
        def forward(self, h: torch.FloatTensor):
            """
            h: (B, L, C) or (B, C, L)
            """
            deno = h.shape[self.pool_axis]
            T = torch.exp(self.log_T)
            h_pool = T * (   # (B, C)
                torch.logsumexp(h / T, axis=self.pool_axis)
                - math.log(deno))
            return h_pool


    class LSEHead(nn.Module):
        """"""
        
        def __init__(
            self,
            num_features    : int,
            num_classes     : int,
            dropout         : float = 0.2,
            is_lse_trainable: bool = False,
        ):
            """"""
            super().__init__()
            self.cls_fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, num_features), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes),
            )
            self.lse_pool = LSEPooling(pool_axis=1, trainable=is_lse_trainable)
    
        def forward(self, h):
            """
            h : (B, C, Freq, Time)
            """
            h = h.mean(axis=2)                 # (B, C, Time)
            h = h.transpose(1, 2)              # (B, Time, C)
            tw_logits = self.cls_fc(h)         # (B, Time, N_CLS)
            logits = self.lse_pool(tw_logits)  # (B, N_CLS)
    
            return logits


    class LSEModel(nn.Module):
        """"""
        def __init__(
            self,
            model_name      : str,
            pretrained      : bool,
            drop_path_rate  : float,
            num_classes     : int,
            head_dropout    : float = 0.0,
            is_lse_trainable: bool = False,
        ):
            """"""
            super().__init__()
            self.backbone = timm.create_model(
                model_name, pretrained=pretrained, in_chans=1,
                global_pool="", num_classes=0, drop_path_rate=drop_path_rate)
    
            # Some backbone's num_features don't match its output
            dummy_input = torch.randn(1, 1, 256, 256)
            self.backbone.eval()
            with torch.no_grad():
                dummy_output = self.backbone(dummy_input)
            self.backbone.train()
            num_features = dummy_output.shape[1]
            self.head = LSEHead(
                num_features, num_classes, head_dropout, is_lse_trainable)
    
        def forward(self, x):
            h = self.backbone(x)   # (B, C, Freq, Time)
            logits = self.head(h)  # (B, N_CLS)
            return logits


    model_list = []
    exsample_input = torch.randn(12, 1, 256, 256, dtype=torch.float32)
    for fold_id in range(N_FOLDS):
        model = LSEModel("hgnetv2_b0.ssld_stage2_ft_in1k", False, 0.0, 234, 0.5, 1.0)
        model.load_state_dict(
            torch.load(TRAINED_MODEL / f"best_model_fold{fold_id}.pt")
        )
        model = model.cpu().eval()
        model = torch.jit.trace(model, exsample_input)
        model = model.eval().share_memory()

        model_list.append(model)
        del model

# %% cell 17
def ov_async_infer_with_order(model, lms_batches, num_requests=4):
    """"""
    idx_batches = []
    tmp_idx = 0
    for b in lms_batches:
        b_size = len(b)
        idx_batches.append(np.arange(tmp_idx, tmp_idx + b_size))
        tmp_idx += b_size
    n_records = tmp_idx

    infer_queue = ov.AsyncInferQueue(model, num_requests)
    
    # array for predict result
    logit_arr = np.zeros((n_records, N_SPECIES), dtype=np.float32)
    
    start_time = time()
    
    def callback(request, userdata):
        input_idxs = userdata
        output = request.get_output_tensor().data
        logit_arr[input_idxs] = output

    infer_queue.set_callback(callback)
    input_name = model.inputs[0].get_any_name()
    
    for idxs, lms in zip(idx_batches, lms_batches):
        infer_queue.start_async({input_name: lms}, userdata=idxs)

    infer_queue.wait_all()

    print(f"... Done by {time() - start_time:.2f} sec")
    return logit_arr

# %% cell 18
def torch_multi_thread_infer_with_order(model, lms_tensor_batches, num_workers):
    """"""
    idxs_batch_pairs = []
    tmp_idx_start = 0
    for batch in lms_tensor_batches:
        tmp_idx_end = tmp_idx_start + batch.shape[0]
        tmp_idxs = np.arange(tmp_idx_start, tmp_idx_end)
        idxs_batch_pairs.append((tmp_idxs, batch))
        tmp_idx_start = tmp_idx_end

    n_records = tmp_idx_end

    start_time = time()
    
    @torch.no_grad()
    def infer_for_thread_pool(idxs_batch_pair):
        idxs, batch = idxs_batch_pair
        return idxs, model(batch).numpy()

    logit_arr = np.zeros((n_records, N_SPECIES))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for idxs, logits in executor.map(infer_for_thread_pool, idxs_batch_pairs):
            logit_arr[idxs] = logits

    print(f"... Done by {time() - start_time:.2f} sec")

    return logit_arr

# %% cell 19
def rank_normalize(x):
    r_x = np.zeros_like(x) 
    for i in range(x.shape[1]):
        r_x_i = pd.Series(x[:, i]).rank(method="max")
        r_x[:, i] = r_x_i / r_x_i.shape[0]
    return r_x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# %% cell 20
if USE_OPENVINO:
    inference_func = ov_async_infer_with_order
else:
    inference_func = torch_multi_thread_infer_with_order

# %% cell 21
test_preds_arr = np.zeros(
    (len(model_list), num_test_ss_segs, N_SPECIES), dtype=np.float32)

for model_id, model in enumerate(model_list):
    print(f"[model: {model_id}]")
    test_pred = inference_func(model, lms_batches, 8)
    test_preds_arr[model_id] = test_pred
    del model

# %% cell 22
if RANK_AVG:
    for model_id in range(len(model_list)):
        test_preds_arr[model_id] = rank_normalize(test_preds_arr[model_id])
else:
    for model_id in range(len(model_list)):
        test_preds_arr[model_id] = sigmoid(test_preds_arr[model_id])

# %% cell 23
sub_df = pd.DataFrame(
    test_preds_arr.mean(axis=0),
    columns=SPECIES, 
    index=pd.Series(test_ss_segs, name="row_id"),
).reset_index()

display(sub_df.head())

# %% cell 24
sub_df = pd.merge(
    sample_sub[["row_id"]], sub_df,
    on="row_id", how="left")

sub_df.to_csv("submission.csv", index=False)
print(sub_df.shape)
display(sub_df.head())
