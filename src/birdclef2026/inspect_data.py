import argparse
from pathlib import Path

import pandas as pd

from .config import load_config, resolve_data_root
from .metadata import build_train_metadata, get_target_columns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    data_root = resolve_data_root(config)
    print(f"data_root: {data_root}")
    print(f"exists: {data_root.exists()}")

    paths = config.get("paths", {})
    for key in ["train_csv", "taxonomy_csv", "sample_submission", "soundscape_labels_csv"]:
        rel = paths.get(key)
        if rel:
            path = data_root / rel
            print(f"{key}: {path} exists={path.exists()}")
            if path.exists():
                df = pd.read_csv(path, nrows=3)
                print(df.head(3).to_string(index=False))

    target_columns = get_target_columns(data_root, config)
    print(f"targets: {len(target_columns)}")
    print(f"first targets: {target_columns[:10]}")

    metadata = build_train_metadata(data_root, config, target_columns)
    print(f"training rows with existing audio: {len(metadata)}")
    if not metadata.empty:
        print(metadata[["filename", "primary_label", "source"]].head(10).to_string(index=False))
        print(metadata["source"].value_counts().to_string())
        print(metadata["primary_label"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()

