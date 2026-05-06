from pathlib import Path
import os

import yaml


def load_config(config_path):
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(config_path)
    return config


def resolve_data_root(config):
    paths = config.get("paths", {})
    env_root = os.environ.get("BIRDCLEF_DATA_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    if paths.get("data_root"):
        candidates.append(Path(paths["data_root"]))

    candidates.extend(
        [
            Path("/kaggle/input/birdclef-2026"),
            Path("/kaggle/input/competitions/birdclef-2026"),
            Path("/kaggle/input/birdclef-2026-repack"),
        ]
    )

    train_csv_name = paths.get("train_csv", "train.csv")
    sample_name = paths.get("sample_submission", "sample_submission.csv")
    for candidate in candidates:
        if (candidate / train_csv_name).exists() or (candidate / sample_name).exists():
            return candidate
    return candidates[0]


def path_from_config(config, key, default_name):
    root = resolve_data_root(config)
    rel = config.get("paths", {}).get(key, default_name)
    return root / rel

