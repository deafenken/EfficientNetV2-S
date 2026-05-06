import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .audio import LogMelExtractor, extract_segment, load_audio
from .config import load_config, resolve_data_root
from .metadata import get_target_columns, parse_row_id
from .model import create_model
from .utils import ensure_dir, get_device


AUDIO_SUFFIXES = [".ogg", ".wav", ".flac", ".mp3", ".m4a"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--fallback", choices=["zeros", "uniform"], default="zeros")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    data_root = resolve_data_root(config)
    paths = config.get("paths", {})
    sample_path = data_root / paths.get("sample_submission", "sample_submission.csv")
    test_dir = data_root / paths.get("test_soundscapes", "test_soundscapes")

    checkpoint = load_checkpoint(args.checkpoint)
    output_columns = get_target_columns(data_root, config)
    if checkpoint is not None:
        model_columns = checkpoint["target_columns"]
    else:
        model_columns = output_columns

    rows = build_submission_rows(sample_path, test_dir, config)
    if checkpoint is None:
        print("No checkpoint supplied; writing fallback submission.")
        submission = make_fallback_submission(rows, output_columns, args.fallback)
    else:
        predictions = predict_rows(checkpoint, rows, config)
        submission = rows_to_submission(rows, model_columns, output_columns, predictions)

    output = Path(args.output)
    if output.parent != Path("."):
        ensure_dir(output.parent)
    submission.to_csv(output, index=False, float_format="%.8f")
    print(f"wrote {output} shape={submission.shape}")


def load_checkpoint(path):
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def build_submission_rows(sample_path, test_dir, config):
    rows = []
    test_dir = Path(test_dir)
    if Path(sample_path).exists():
        sample = pd.read_csv(sample_path)
        for row_id in sample["row_id"].astype(str):
            audio_id, end_time = parse_row_id(row_id)
            audio_path = find_audio_file(test_dir, audio_id)
            rows.append({"row_id": row_id, "audio_id": audio_id, "end_time": end_time, "path": audio_path})
        return rows

    files = []
    if test_dir.exists():
        for suffix in AUDIO_SUFFIXES:
            files.extend(test_dir.rglob(f"*{suffix}"))
    clip_seconds = float(config.get("audio", {}).get("clip_seconds", 5.0))
    segments_per_file = int(config.get("infer", {}).get("segments_per_file", 12))
    for path in sorted(files):
        audio_id = path.stem
        for idx in range(1, segments_per_file + 1):
            end_time = idx * clip_seconds
            row_id = f"{audio_id}_{int(end_time)}"
            rows.append({"row_id": row_id, "audio_id": audio_id, "end_time": end_time, "path": path})
    if not rows:
        raise RuntimeError("No sample_submission.csv and no test audio files found.")
    return rows


def find_audio_file(test_dir, audio_id):
    test_dir = Path(test_dir)
    for suffix in AUDIO_SUFFIXES:
        candidate = test_dir / f"{audio_id}{suffix}"
        if candidate.exists():
            return candidate
    if test_dir.exists():
        for suffix in AUDIO_SUFFIXES:
            matches = list(test_dir.rglob(f"{audio_id}{suffix}"))
            if matches:
                return matches[0]
    return None


def make_fallback_submission(rows, target_columns, mode):
    value = 1.0 / len(target_columns) if mode == "uniform" and target_columns else 0.0
    data = {"row_id": [row["row_id"] for row in rows]}
    for col in target_columns:
        data[col] = value
    return pd.DataFrame(data)


def predict_rows(checkpoint, rows, config):
    device = get_device()
    target_columns = checkpoint["target_columns"]
    model_cfg = checkpoint.get("config", config).get("model", {})
    model = create_model(
        num_classes=len(target_columns),
        name=model_cfg.get("name", "resnet18"),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    audio_cfg = checkpoint.get("config", config).get("audio", config.get("audio", {}))
    sample_rate = int(audio_cfg.get("sample_rate", 32000))
    clip_seconds = float(audio_cfg.get("clip_seconds", 5.0))
    batch_size = int(config.get("infer", {}).get("batch_size", 32))
    mel = LogMelExtractor(audio_cfg).to(device)

    grouped = defaultdict(list)
    missing = []
    for row in rows:
        if row["path"] is None:
            missing.append(row["row_id"])
        else:
            grouped[str(row["path"])].append(row)

    predictions = {}
    with torch.no_grad():
        for path, group_rows in tqdm(grouped.items(), desc="infer files"):
            waveform = load_audio(path, sample_rate)
            features = []
            row_ids = []
            for row in group_rows:
                segment = extract_segment(waveform, sample_rate, clip_seconds, end_time=row["end_time"])
                features.append(mel(segment.to(device)).cpu())
                row_ids.append(row["row_id"])
                if len(features) == batch_size:
                    infer_batch(model, features, row_ids, predictions, device)
                    features, row_ids = [], []
            if features:
                infer_batch(model, features, row_ids, predictions, device)

    if missing:
        print(f"warning: {len(missing)} rows did not resolve to audio files; filled with zeros.")
    return predictions


def infer_batch(model, features, row_ids, predictions, device):
    batch = torch.stack(features).to(device)
    logits = model(batch)
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    for row_id, prob in zip(row_ids, probs):
        predictions[row_id] = prob


def rows_to_submission(rows, model_columns, output_columns, predictions):
    model_to_idx = {label: idx for idx, label in enumerate(model_columns)}
    data = {"row_id": []}
    for col in output_columns:
        data[col] = []
    zeros = np.zeros(len(model_columns), dtype=np.float32)
    for row in rows:
        row_id = row["row_id"]
        probs = predictions.get(row_id, zeros)
        data["row_id"].append(row_id)
        for col in output_columns:
            idx = model_to_idx.get(col)
            data[col].append(float(probs[idx]) if idx is not None else 0.0)
    return pd.DataFrame(data)


if __name__ == "__main__":
    main()
