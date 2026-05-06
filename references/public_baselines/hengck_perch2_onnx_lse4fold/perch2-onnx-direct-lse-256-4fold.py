# %% cell 1
#try:
#    from nnAudio.Spectrogram import STFT
#except:
#    #!pip install nnaudio
#    !pip install nnaudio --no-index --find-links=/kaggle/input/datasets/hengck23/hengck23-birdclef-2026-00/nnaudio-0.3.4-py3-none-any.whl
#from nnAudio.Spectrogram import STFT

try: 
    import onnxruntime
except:
    !pip install '/kaggle/input/datasets/lixin73/birdclef2026-v27-onnx-perch-meta-forum-v1-lb872/runtime_wheels/onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'

import onnxruntime as ort
import librosa

print('import ok!!!')

# %% cell 2
#!mkdir -p /kaggle/working/offline_pkgs
#!pip download nnaudio --no-deps -d /kaggle/working/offline_pkgs
#!ls -lh /kaggle/working/offline_pkgs


# !pip install --no-index --find-links=/kaggle/working/offline_pkgs nnaudio
# from nnAudio.Spectrogram import STFT
# print("OK")



# %% cell 3
import sys
sys.path.append('/kaggle/input/datasets/hengck23/hengck23-birdclef-2026-01')
#from model import *

import librosa
import numpy as np
import pandas as pd
import glob
import time
import torch

from concurrent.futures import ThreadPoolExecutor


def get_filename():
    valid_file = sorted(glob.glob(f"{valid_dir}/*.ogg"))
    filename = [f.split('/')[-1] for f in valid_file]
    return filename


'''
 test_soundscapes directory will be populated with approximately 600 recordings 
 to be used for scoring. They are 1 minute long 
 5sec = 12 clip
'''

KAGGLE_DIR =  '/kaggle/input/competitions/birdclef-2026'
DEVICE = 'cpu'
NUM_CLASS = 234

MODE ='submit' # 'submit' local 


 
if MODE=='submit':
    valid_dir = f'{KAGGLE_DIR}/test_soundscapes'
    valid_filename = get_filename()

if len(valid_filename)==0:
    print('valid_filename is zero ... use train for dry run')
    MODE='local'
    valid_dir = f'{KAGGLE_DIR}/train_soundscapes'
    valid_filename = [
       'BC2026_Train_0027_S22_20211129_014500.ogg',
       'BC2026_Train_0014_S13_20220228_214500.ogg',
       'BC2026_Train_0004_S08_20250607_070007.ogg',
       'BC2026_Train_0049_S22_20220203_221500.ogg',
       'BC2026_Train_0029_S22_20211211_023000.ogg',
       'BC2026_Train_0009_S09_20250828_000000.ogg',
       'BC2026_Train_0026_S22_20211128_004500.ogg',
       'BC2026_Train_0063_S19_20241214_190000.ogg',
       'BC2026_Train_0041_S22_20220105_223000.ogg',
       'BC2026_Train_0010_S09_20250828_000000.ogg',
       'BC2026_Train_0043_S22_20220112_040000.ogg',
       'BC2026_Train_0050_S22_20220205_214500.ogg',
       'BC2026_Train_0060_S15_20250617_062700.ogg',
       'BC2026_Train_0011_S03_20211114_200000.ogg',
       'BC2026_Train_0037_S22_20211226_021500.ogg',
       'BC2026_Train_0065_S23_20241124_040002.ogg',
    ]


print('MODE', MODE)
print('valid_filename:', len(valid_filename), valid_filename[:2])

#
##name to label  ----
SAMPLE_DF = pd.read_csv(f'{KAGGLE_DIR}/sample_submission.csv')
LABELNAME =list(SAMPLE_DF.columns)[1:]
NAME_TO_LABEL = {
    r: i for i, r in enumerate(LABELNAME)
}
LABEL_TO_NAME = {v:k for k, v in NAME_TO_LABEL.items()}

NUM_LABEL= len(NAME_TO_LABEL) #234
print('config ok!!!')





# %% cell 4
import typing as tp
import timm
import torch
import torchaudio
import torchvision
import math

from torch import nn
from torch.nn import functional as F



ONNX_FILE='/kaggle/input/datasets/hengck23/hengck23-onnx-perchv2/perch_v2_no_dft.onnx'

def to_bind(binding_fn, name, x, shape, device):
    if x is None:
        x = torch.empty(shape, device=device, dtype=torch.float32)
    binding_fn(
        name=name,
        device_type=device.type,
        device_id=device.index or 0,
        element_type=np.float32,
        shape=shape,
        buffer_ptr=x.data_ptr(),
    )
    return x


class OnnxFeatureExtractor(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.session = ort.InferenceSession(
            ONNX_FILE,
            providers=[#'TensorrtExecutionProvider', 'CUDAExecutionProvider',
                       'CPUExecutionProvider']
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.float().contiguous()
        B = x.shape[0]
        device = x.device

        binding = self.session.io_binding()

        x = to_bind(binding.bind_input, 'inputs', x, x.shape, device)
        spatial_embedding = to_bind(binding.bind_output, 'spatial_embedding', None, (B,16, 4, 1536), device)
        embedding = to_bind(binding.bind_output, 'embedding', None, (B,1536), device)
        spectrogram = to_bind(binding.bind_output, 'spectrogram', None, (B,500, 128), device)

        self.session.run_with_iobinding(binding)

        return spatial_embedding, embedding, spectrogram




def lse_pool(x, dim=1, r=10.0):
    T = x.size(dim)
    return torch.logsumexp(r * x, dim=dim) / r - math.log(T) / r


class DirectNet(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.D = nn.Parameter(torch.zeros(1)) #for device
        self.output_type=['loss', 'infer']

        self.perchv2 = OnnxFeatureExtractor()
        self.head = nn.Sequential(
            nn.BatchNorm1d(1536),
            nn.Linear(1536, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 234)
        )

    def forward(self, batch):
        device = self.D.device
        
        if batch['spatial_embed'] is None:
            audio =  batch['audio'].to(device)
            p_spatial_embed, p_embed, p_spec = self.perchv2(audio)
            # (B, 16, 4, 1536), (B,1536),  (B, 500, 128)
            spatial_embed = p_spatial_embed
        else:
            spatial_embed = batch['spatial_embed'].to(device)

        B = spatial_embed.shape[0]
        
        #----------------------------------------------------------
        last = spatial_embed
        last = last.mean(dim=2)  # (B, T, 1536)
        #time_logit = self.head(last.detach().reshape(-1,2048)).reshape(B,-1,234)  # (B, T, 234)
        time_logit = self.head(last.reshape(-1,1536)).reshape(B,-1,234)  # (B, T, 234)
        logit = lse_pool(time_logit, dim=1)  # (B, 234)

        output = {}
        if 'loss' in self.output_type:
            onehot = batch['onehot'].to(device)
            weight = batch['weight'].to(device) 
            output['bce_loss'] = bce_with_weight(logit, onehot, weight)


        if 'infer' in self.output_type:
            output['prob'] = F.sigmoid(logit)
            output['time_logit'] = time_logit
            output['time_prob']  = F.sigmoid(time_logit)

        return output



# %% cell 5

MEL_SPECT_PARAM = dict(
    sample_rate=32_000,
    n_fft=2048,
    win_length=626,
    hop_length=313,
    f_min=20,
    n_mels=256,
    power=2.0,
    center=True,
    pad_mode="reflect",
    norm="slaney",
    mel_scale='htk',
)
MEL_TOP_DB = 80
MEL_SHAPE = (256, 256)


class LogMelSpectrogramTransform(nn.Module):
    """"""

    def __init__(self, mel_spectrogram_params: tp.Dict, top_db: float, shape=tp.Tuple[int, int]):
        """"""
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**mel_spectrogram_params)
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db)
        self.resize = torchvision.transforms.Resize(size=shape)

    @torch.no_grad()
    def forward(self, wave):
        """
        wave: (B, sampling_rate * segment_sec)
        """
        mel = self.mel_transform(wave)
        mel = self.db(mel)  # shape: (B, n_mels, time)  torch.Size([32, 256, 512])
        mel = self.resize(mel)  # shape: (B, *(lms_shape)) #torch.Size([32, 256, 512])

        batch_size = mel.shape[0]
        flat = mel.reshape(batch_size, -1)
        fmin = flat.min(dim=1)[0][:, None, None]
        fmax = flat.max(dim=1)[0][:, None, None]
        mel = (mel - fmin) / (fmax - fmin + 1e-7)

        return mel[:, None, :, :]


##########################################################################3


class DistillNet(nn.Module):
    def __init__(
        self,
        pretrained=False
    ):
        super().__init__()
        self.D = nn.Parameter(torch.zeros(1))  # for device
        self.output_type = ['loss', 'infer']

        self.logmelspec = LogMelSpectrogramTransform(
            mel_spectrogram_params=MEL_SPECT_PARAM,
            top_db=MEL_TOP_DB,
            shape=MEL_SHAPE
        )
        self.backbone = timm.create_model(
            "hgnetv2_b0.ssld_stage2_ft_in1k",
            # "tf_efficientnet_b0.ns_jft_in1k",
            pretrained=pretrained,
            in_chans=1,
            drop_path_rate=0.0,
            features_only=False,
            global_pool="",
            num_classes=0,
        )
        # self.head = nn.Linear(2048, 234)

        self.distill = nn.Linear(2048, 1536)
        self.head = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 234)
        )

    def forward(self, batch):
        device = self.D.device
        B = batch['audio'].shape[0]
        audio = batch['audio'].to(device)
        #p_spatial_embed, p_embed, p_spec = self.perchv2(audio)
        # (B, 16, 4, 1536), (B,1536),  (B, 500, 128)

        # ----------------------------------------------------------
        if batch['spec'] is None:
            audio = batch['audio'].to(device)
            spec = self.logmelspec(audio)
        else:
            spec = batch['spec'].to(device)

        last = self.backbone(spec)
        last = last.mean(dim=2)  # (B, 2048, T)
        last = last.transpose(1, 2)  # (B, T, 2048)
        time_logit = self.head(last.detach().reshape(-1, 2048)).reshape(B, -1, 234)  # (B, T, 234)
        logit = lse_pool(time_logit, dim=1)  # (B, 234)

        #distill = self.distill(last.mean(1))

        output = {}
        if 'loss' in self.output_type:
            pass
        if 'infer' in self.output_type:
            output['prob'] = F.sigmoid(logit)
            output['time_logit'] = time_logit
            output['time_prob'] = F.sigmoid(time_logit)

        return output

# %% cell 6
# load model
direct_nets =[]
for i,checkpoint in enumerate([
    '/kaggle/input/datasets/hengck23/hengck23-birdclef-2026-01/onnx-direct-lse-256-fix-t05-4fold-06a/fold0-00012600.pth',
    '/kaggle/input/datasets/hengck23/hengck23-birdclef-2026-01/onnx-direct-lse-256-fix-t05-4fold-06a/fold1-00012600.pth',
    '/kaggle/input/datasets/hengck23/hengck23-birdclef-2026-01/onnx-direct-lse-256-fix-t05-4fold-06a/fold2-00012645.pth',
    '/kaggle/input/datasets/hengck23/hengck23-birdclef-2026-01/onnx-direct-lse-256-fix-t05-4fold-06a/fold3-00012645.pth',
]):
    net=DirectNet()
    f = torch.load(checkpoint, map_location=lambda storage, loc: storage, weights_only=False)
    print(net.load_state_dict(f['state_dict'], strict=False))
    
    net.output_type=['infer']
    net=net.eval()
    net.to(DEVICE)
    direct_nets.append(net)

distill_nets=[] 
for i,checkpoint in enumerate([
    '/kaggle/input/datasets/hengck23/hengck23-birdclef-2026-01/onnx-joint-distill-lse-256-fix-t05-4fold-06a/fold0-00012600.pth',
    '/kaggle/input/datasets/hengck23/hengck23-birdclef-2026-01/onnx-joint-distill-lse-256-fix-t05-4fold-06a/fold1-00012600.pth',
    '/kaggle/input/datasets/hengck23/hengck23-birdclef-2026-01/onnx-joint-distill-lse-256-fix-t05-4fold-06a/fold2-00012645.pth',
    '/kaggle/input/datasets/hengck23/hengck23-birdclef-2026-01/onnx-joint-distill-lse-256-fix-t05-4fold-06a/fold3-00012645.pth',

    '/kaggle/input/datasets/hengck23/hengck23-birdclef-2026-01/onnx-distill-lse-256-fix-t05-4fold-06a/fold0-00017220.pth',
    '/kaggle/input/datasets/hengck23/hengck23-birdclef-2026-01/onnx-distill-lse-256-fix-t05-4fold-06a/fold1-00012600.pth',
    '/kaggle/input/datasets/hengck23/hengck23-birdclef-2026-01/onnx-distill-lse-256-fix-t05-4fold-06a/fold2-00012645.pth',
    '/kaggle/input/datasets/hengck23/hengck23-birdclef-2026-01/onnx-distill-lse-256-fix-t05-4fold-06a/fold3-00012645.pth',

]):
    net=DistillNet(
        pretrained=False,
    )
    f = torch.load(checkpoint, map_location=lambda storage, loc: storage, weights_only=False)
    print(net.load_state_dict(f['state_dict'], strict=False))
    
    net.output_type=['infer']
    net=net.eval()
    net.to(DEVICE)
    distill_nets.append(net)
 
print('model ok!!!')

# %% cell 7

#load soundscape
def load_soundscape(filename):
    soundscape_file =f'{valid_dir}/{filename}'
    wave, _ = librosa.load(soundscape_file, sr=32_000, mono=True)
    wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return wave

print("loading soundscape files...")
start_time = time.time()
with ThreadPoolExecutor(max_workers=4) as executor:
    soundscape = list(executor.map(load_soundscape, valid_filename))
print(f"loaded {len(soundscape)} files in {time.time() - start_time:.1f}s")


print('data ok!!!')

# %% cell 8
submit_df =[]
num = len(soundscape)


start_time = time.time()
for i in range(num):
    ss = soundscape[i]
    audio = ss.reshape(-1, 160_000)
    prob=0
    
    batch = {
        'audio': torch.from_numpy(audio).float(),
    } 
    
    #----  
    batch['spatial_embed'], _, _ = direct_nets[0].perchv2(batch['audio'])
    # (B, 16, 4, 1536), (B,1536),  (B, 500, 128) 
    with torch.no_grad():
        for net in direct_nets:
            ouput = net(batch)
            prob += ouput['prob'].data.float().cpu().numpy()  
    
    #---- 
    batch['spec'] = distill_nets[0].logmelspec(batch['audio']) 
    
    with torch.no_grad(): 
        for net in direct_nets[:4]:
            ouput = net(batch)
            prob += ouput['prob'].data.float().cpu().numpy() 

    prob = prob/8   

    
    #-------------- start of trick --------------------------------
    if 1:
        prob1 = 0 
        for shift in [
            int(160_000 / 4), -int(160_000 / 4),
        ]:
            batch1 = {
                'audio': torch.from_numpy(audio).float(),
            } 
            batch1['audio'] = torch.roll(batch1['audio'], shift)  
            batch1['spec'] = distill_nets[0].logmelspec(batch1['audio']) #would be faster if spect module is shared
            for net in distill_nets[4:]:
                ouput = net(batch1)
                prob1 += ouput['prob'].data.float().cpu().numpy() 
    
        prob1 = prob1/8
        prob = (2/3)*prob1+(1/3)*prob
     
        #---- 
        # temporal smoothing
        SMOOTH_EVENT   = np.array([0.20, 0.60, 0.20])
        SMOOTH_TEXTURE = np.array([0.35, 0.30, 0.35])
        
        def do_smooth(p, w):
            padded = np.pad(p, ((1, 1), (0, 0)), mode='edge') #(14, 234)
            smoothed = w[0] * padded[:-2] + w[1] * padded[1:-1] + w[2] * padded[2:]
            return smoothed 
            
        prob = do_smooth(prob, SMOOTH_EVENT)
        
        
    #----------------- end of trick --------------------------------
 
        
    stem = valid_filename[i][:-4]
    row_id =[
        f'{stem}_{j*5+5}' for j in range(12)
    ] #BC2026_Test_0001_S05_20250227_010002_5
    col = list(LABEL_TO_NAME.values()) 
    df = pd.DataFrame(prob, columns=col)
    df.insert(0, 'row_id', row_id)
    submit_df.append(df)
    
    time_taken = (time.time() -start_time) / 60
    print(f'\r {i} {valid_filename[i]}: {time_taken:3.1f} min', end='', flush=True)
print()

submit_df = pd.concat(submit_df, axis=0).reset_index(drop=True)
#submit_df = submit_df[SAMPLE_DF.columns.tolist()]
print(submit_df.shape)
print(submit_df)
submit_df.to_csv('submission.csv', index=False)



time_taken = (time.time() -start_time) / 60
print('all test soundscaped (600) : ',time_taken/len(valid_filename)*600, 'min')

# %% cell 9
if MODE=='local':
    import sklearn.metrics
    def compute_lb(prob, truth):
        prob = prob.reshape(-1, 234)
        truth = truth.reshape(-1, 234)

        nonzero = truth.sum(axis=0) > 0
        roc = sklearn.metrics.roc_auc_score(
            truth[:, nonzero],
            prob[:, nonzero],
            average=None
        )
        roc_macro = roc.mean()
        return roc_macro



    def label_str_to_onehot(label_str, name_to_label=NAME_TO_LABEL, num_classes=NUM_LABEL):
        y = np.zeros(num_classes, dtype=np.float32)
        for x in str(label_str).split(";"):
            x = x.strip()
            if x in name_to_label:
                y[name_to_label[x]] = 1.0
        return y


    # helper: generate full 12 rows (0–60 sec, step=5 sec)
    def generate_full_timeline(filename):
        starts = pd.timedelta_range(start='0s', periods=12, freq='5s')
        ends = starts + pd.Timedelta(seconds=5)

        df = pd.DataFrame({
            'filename': filename,
            'start': starts,
            'end': ends,
        })
        return df


    def expand_to_12_rows(df):
        out = []

        for fname, g in df.groupby('filename'):
            g = g.copy()

            # convert to timedelta (important!)
            g['start'] = pd.to_timedelta(g['start'])
            g['end'] = pd.to_timedelta(g['end'])

            # full template
            full = generate_full_timeline(fname)

            # merge
            merged = full.merge(
                g[['start', 'primary_label']],
                on='start',
                how='left'
            )

            # fill missing labels
            merged['primary_label'] = merged['primary_label'].apply(
                lambda x: '' if pd.isna(x) else x
            )

            # re-create end (safe)
            merged['end'] = merged['start'] + pd.Timedelta(seconds=5)

            # optional: convert back to string format
            merged['start'] = merged['start'].astype(str)
            merged['end'] = merged['end'].astype(str)

            out.append(merged)

        return pd.concat(out, ignore_index=True)

    pass
    submit_df = submit_df.set_index('row_id')
    prob = submit_df.values
    prob = prob.reshape(-1, 12, 234)
    print('prob.shape', prob.shape)

    truth_df = pd.read_csv(f'{KAGGLE_DIR}/train_soundscapes_labels.csv')  #
    truth_df = truth_df.drop_duplicates()

    truth =[]
    for f in valid_filename:
        d = truth_df[truth_df['filename']==f]
        d = d.sort_values(by='start').reset_index(drop=True)
        d = expand_to_12_rows(d)
        d['onehot'] = d['primary_label'].apply(label_str_to_onehot)
        h = np.stack(d['onehot'].values)
        truth.append(h)
    truth = np.stack(truth)
    print('truth.shape', truth.shape)
    print('lb:', compute_lb(prob, truth))
