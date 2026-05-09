from pathlib import Path
import ast
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


NO_CALLS = {"", "nan", "none", "no_call", "nocall", "no call", "background"}


def _read_csv(path, nrows=None):
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path, nrows=nrows)


def _split_label_cell(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        text = str(value).strip()
        if text.lower() in NO_CALLS:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                raw = parsed if isinstance(parsed, (list, tuple, set)) else [parsed]
            except (SyntaxError, ValueError):
                raw = [text]
        else:
            raw = re.split(r"[;,]", text)
    labels = []
    for item in raw:
        item = str(item).strip().strip("'\"")
        if item.lower() not in NO_CALLS:
            labels.append(item)
    return labels


def get_target_columns(data_root, config):
    data_root = Path(data_root)
    paths = config.get("paths", {})
    sample_path = data_root / paths.get("sample_submission", "sample_submission.csv")
    sample_head = _read_csv(sample_path, nrows=0)
    if sample_head is not None and "row_id" in sample_head.columns:
        return [c for c in sample_head.columns if c != "row_id"]

    taxonomy_path = data_root / paths.get("taxonomy_csv", "taxonomy.csv")
    taxonomy = _read_csv(taxonomy_path)
    if taxonomy is not None:
        for col in ["scientific_name", "primary_label", "common_name"]:
            if col in taxonomy.columns:
                return sorted(taxonomy[col].dropna().astype(str).unique().tolist())
    raise FileNotFoundError("Could not infer target columns from sample_submission.csv or taxonomy.csv")


def build_taxonomy_mapper(data_root, config, target_columns):
    data_root = Path(data_root)
    paths = config.get("paths", {})
    taxonomy_path = data_root / paths.get("taxonomy_csv", "taxonomy.csv")
    taxonomy = _read_csv(taxonomy_path)
    target_set = set(str(c) for c in target_columns)

    mapper = {}
    if taxonomy is None:
        return mapper

    raw_col = "primary_label" if "primary_label" in taxonomy.columns else None
    if raw_col is None:
        return mapper

    candidate_cols = ["primary_label", "scientific_name", "common_name"]
    for _, row in taxonomy.iterrows():
        raw = str(row.get(raw_col, "")).strip()
        if not raw:
            continue
        mapped = None
        for col in candidate_cols:
            if col in taxonomy.columns:
                value = str(row.get(col, "")).strip()
                if value in target_set:
                    mapped = value
                    break
        if mapped is None and raw in target_set:
            mapped = raw
        if mapped is not None:
            mapper[raw] = mapped
    return mapper


def map_labels(values, mapper, target_columns):
    target_set = set(str(c) for c in target_columns)
    labels = []
    for value in values:
        value = str(value).strip()
        mapped = mapper.get(value, value)
        if mapped in target_set:
            labels.append(mapped)
    return sorted(set(labels))


def build_train_metadata(data_root, config, target_columns):
    data_root = Path(data_root)
    paths = config.get("paths", {})
    data_cfg = config.get("data", {})
    mapper = build_taxonomy_mapper(data_root, config, target_columns)
    rows = []

    if data_cfg.get("include_train_audio", True):
        train_csv = data_root / paths.get("train_csv", "train.csv")
        train_audio_dir = data_root / paths.get("train_audio", "train_audio")
        train_df = _read_csv(train_csv)
        if train_df is not None:
            for _, row in train_df.iterrows():
                filename = str(row.get("filename", "")).strip()
                primary_raw = str(row.get("primary_label", "")).strip()
                raw_labels = [primary_raw]
                if "secondary_labels" in train_df.columns:
                    raw_labels.extend(_split_label_cell(row.get("secondary_labels")))
                labels = map_labels(raw_labels, mapper, target_columns)
                if filename and labels:
                    rows.append(
                        {
                            "path": str(train_audio_dir / filename),
                            "filename": filename,
                            "labels": labels,
                            "primary_label": labels[0],
                            "source": "train_audio",
                            "end_time": np.nan,
                        }
                    )

    if data_cfg.get("include_train_soundscapes", False):
        labels_csv = data_root / paths.get("soundscape_labels_csv", "train_soundscapes_labels.csv")
        soundscape_dir = data_root / paths.get("train_soundscapes", "train_soundscapes")
        sound_df = _read_csv(labels_csv)
        if sound_df is not None and "primary_label" in sound_df.columns:
            for _, row in sound_df.iterrows():
                raw_labels = _split_label_cell(row.get("primary_label"))
                labels = map_labels(raw_labels, mapper, target_columns)
                if not labels:
                    continue
                filename = str(row.get("filename", "")).strip()
                row_id = str(row.get("row_id", "")).strip()
                end_time = row.get("end_time", row.get("seconds", np.nan))
                if not filename and row_id:
                    audio_id, parsed_end = parse_row_id(row_id)
                    filename = audio_id + ".ogg"
                    if pd.isna(end_time) and parsed_end is not None:
                        end_time = parsed_end
                if filename:
                    rows.append(
                        {
                            "path": str(soundscape_dir / filename),
                            "filename": filename,
                            "labels": labels,
                            "primary_label": labels[0],
                            "source": "train_soundscapes",
                            "end_time": end_time,
                        }
                    )

    # Pseudo-label R1: rows produced by predict_pseudo.py over the unlabeled
    # soundscape recordings. CSV columns: filename, end_time, primary_label,
    # secondary_labels (space-separated), prob_primary. Treat them like
    # train_soundscapes (window-level end_time, primary=1.0 + secondary at
    # secondary_weight) with source="train_soundscapes_pseudo" so the sampler
    # can optionally down-weight them. Rows whose primary is "nocall" or below
    # `pseudo_min_prob` (if set) are dropped.
    pseudo_csv_path = data_cfg.get("pseudo_label_csv")
    if pseudo_csv_path:
        pseudo_path = Path(pseudo_csv_path)
        if not pseudo_path.is_absolute():
            pseudo_path = data_root / pseudo_path
        soundscape_dir = data_root / paths.get("train_soundscapes", "train_soundscapes")
        pseudo_df = _read_csv(pseudo_path)
        if pseudo_df is not None and "primary_label" in pseudo_df.columns:
            min_prob = float(data_cfg.get("pseudo_min_prob", 0.0))
            kept = 0
            for _, row in pseudo_df.iterrows():
                primary_raw = str(row.get("primary_label", "")).strip()
                if primary_raw.lower() in NO_CALLS:
                    continue
                prob = float(row.get("prob_primary", 1.0)) if "prob_primary" in pseudo_df.columns else 1.0
                if prob < min_prob:
                    continue
                raw_labels = [primary_raw]
                if "secondary_labels" in pseudo_df.columns:
                    raw_labels.extend(_split_label_cell(row.get("secondary_labels")))
                labels = map_labels(raw_labels, mapper, target_columns)
                if not labels:
                    continue
                primary = primary_raw if primary_raw in labels else labels[0]
                filename = str(row.get("filename", "")).strip()
                end_time = row.get("end_time", np.nan)
                if filename:
                    rows.append(
                        {
                            "path": str(soundscape_dir / filename),
                            "filename": filename,
                            "labels": labels,
                            "primary_label": primary,
                            "source": "train_soundscapes_pseudo",
                            "end_time": end_time,
                        }
                    )
                    kept += 1

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    return df


def parse_row_id(row_id):
    text = str(row_id)
    if "_" not in text:
        return text, None
    audio_id, maybe_end = text.rsplit("_", 1)
    try:
        return audio_id, float(maybe_end)
    except ValueError:
        return text, None


def split_train_val(df, val_fraction, seed):
    if df.empty:
        return df.copy(), df.copy()
    labels = df["primary_label"].astype(str)
    counts = labels.value_counts()
    can_stratify = counts.min() >= 2 and len(counts) > 1
    stratify = labels if can_stratify else None
    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        random_state=seed,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

