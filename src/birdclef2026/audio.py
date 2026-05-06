from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio


def load_audio(path, sample_rate):
    path = Path(path)
    try:
        waveform, sr = torchaudio.load(str(path))
    except Exception:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        waveform = torch.from_numpy(data.T)
    waveform = waveform.float()
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform.squeeze(0)


def crop_or_pad(waveform, length_samples, random_crop=False, start_sample=None):
    num_samples = waveform.numel()
    if num_samples >= length_samples:
        if start_sample is None:
            if random_crop and num_samples > length_samples:
                max_start = num_samples - length_samples
                start_sample = int(torch.randint(0, max_start + 1, (1,)).item())
            else:
                start_sample = 0
        start_sample = max(0, min(int(start_sample), num_samples - length_samples))
        return waveform[start_sample : start_sample + length_samples]

    pad = length_samples - num_samples
    return F.pad(waveform, (0, pad))


def extract_segment(waveform, sample_rate, clip_seconds, end_time=None):
    length_samples = int(round(sample_rate * clip_seconds))
    if end_time is None:
        return crop_or_pad(waveform, length_samples, random_crop=False)
    end_sample = int(round(float(end_time) * sample_rate))
    start_sample = end_sample - length_samples
    return crop_or_pad(waveform, length_samples, random_crop=False, start_sample=start_sample)


class LogMelExtractor(torch.nn.Module):
    def __init__(self, audio_config):
        super().__init__()
        sample_rate = int(audio_config.get("sample_rate", 32000))
        n_fft = int(audio_config.get("n_fft", 2048))
        hop_length = int(audio_config.get("hop_length", 512))
        n_mels = int(audio_config.get("n_mels", 128))
        f_min = float(audio_config.get("f_min", 20))
        f_max = float(audio_config.get("f_max", sample_rate // 2))
        top_db = float(audio_config.get("top_db", 80))
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            normalized=False,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db)

    def forward(self, waveform):
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.mel(waveform)
        mel = self.to_db(mel)
        mean = mel.mean()
        std = mel.std().clamp_min(1e-6)
        return (mel - mean) / std
