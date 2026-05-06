import math

import torch
from torch.utils.data import Dataset

from .audio import LogMelExtractor, crop_or_pad, extract_segment, load_audio


class BirdCLEFDataset(Dataset):
    def __init__(self, dataframe, target_columns, audio_config, random_crop):
        self.df = dataframe.reset_index(drop=True)
        self.target_columns = list(target_columns)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.target_columns)}
        self.audio_config = audio_config
        self.random_crop = random_crop
        self.sample_rate = int(audio_config.get("sample_rate", 32000))
        self.clip_seconds = float(audio_config.get("clip_seconds", 5.0))
        self.length_samples = int(round(self.sample_rate * self.clip_seconds))
        self.mel = LogMelExtractor(audio_config)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform = load_audio(row["path"], self.sample_rate)
        end_time = row.get("end_time")
        if end_time is not None and not _is_nan(end_time):
            waveform = extract_segment(waveform, self.sample_rate, self.clip_seconds, end_time=end_time)
        else:
            waveform = crop_or_pad(waveform, self.length_samples, random_crop=self.random_crop)
        features = self.mel(waveform)

        target = torch.zeros(len(self.target_columns), dtype=torch.float32)
        for label in row["labels"]:
            idx = self.label_to_idx.get(label)
            if idx is not None:
                target[idx] = 1.0
        return features, target


def _is_nan(value):
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False

