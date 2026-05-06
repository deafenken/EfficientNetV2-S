import numpy as np
import pandas as pd


def sigmoid_clip(scores, clip=30.0):
    scores = np.asarray(scores, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -clip, clip)))


def file_level_confidence_scale(probs, n_windows=12, top_k=2):
    probs = np.asarray(probs, dtype=np.float32)
    if top_k <= 0:
        return probs.copy()
    if probs.shape[0] % n_windows != 0:
        raise ValueError(f"Expected row count divisible by {n_windows}, got {probs.shape[0]}")
    view = probs.reshape(-1, n_windows, probs.shape[1])
    sorted_view = np.sort(view, axis=1)
    top_k_mean = sorted_view[:, -top_k:, :].mean(axis=1, keepdims=True)
    return (view * top_k_mean).reshape(probs.shape).astype(np.float32, copy=False)


def rank_aware_scaling(probs, n_windows=12, power=0.4):
    probs = np.asarray(probs, dtype=np.float32)
    if probs.shape[0] % n_windows != 0:
        raise ValueError(f"Expected row count divisible by {n_windows}, got {probs.shape[0]}")
    view = probs.reshape(-1, n_windows, probs.shape[1])
    file_max = view.max(axis=1, keepdims=True)
    scaled = view * np.power(file_max, power)
    return scaled.reshape(probs.shape).astype(np.float32, copy=False)


def adaptive_delta_smooth(probs, n_windows=12, base_alpha=0.20):
    probs = np.asarray(probs, dtype=np.float32)
    if base_alpha <= 0:
        return probs.copy()
    if probs.shape[0] % n_windows != 0:
        raise ValueError(f"Expected row count divisible by {n_windows}, got {probs.shape[0]}")
    result = probs.copy()
    view = result.reshape(-1, n_windows, probs.shape[1])
    source = probs.reshape(-1, n_windows, probs.shape[1])
    for idx in range(1, n_windows - 1):
        confidence = source[:, idx, :].max(axis=-1, keepdims=True)
        alpha = base_alpha * (1.0 - confidence)
        neighbor_avg = (source[:, idx - 1, :] + source[:, idx + 1, :]) / 2.0
        view[:, idx, :] = (1.0 - alpha) * source[:, idx, :] + alpha * neighbor_avg
    return result.reshape(probs.shape).astype(np.float32, copy=False)


def apply_per_class_thresholds(probs, thresholds):
    probs = np.asarray(probs, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if probs.shape[1] != len(thresholds):
        raise ValueError(f"Expected {probs.shape[1]} thresholds, got {len(thresholds)}")

    scaled = np.empty_like(probs)
    for idx, threshold in enumerate(thresholds):
        above = probs[:, idx] > threshold
        scaled[above, idx] = 0.5 + 0.5 * (probs[above, idx] - threshold) / (1 - threshold + 1e-8)
        scaled[~above, idx] = 0.5 * probs[~above, idx] / (threshold + 1e-8)
    return np.clip(scaled, 0.0, 1.0)


def align_submission_to_sample(submission, sample_submission, fill_value=0.0):
    columns = sample_submission.columns.tolist()
    if "row_id" not in columns:
        raise ValueError("sample_submission must contain row_id")
    aligned = sample_submission[["row_id"]].merge(submission, on="row_id", how="left")
    target_columns = [c for c in columns if c != "row_id"]
    aligned[target_columns] = aligned[target_columns].fillna(fill_value).astype(np.float32)
    return aligned[columns]


def validate_submission(submission, sample_submission):
    if submission.columns.tolist() != sample_submission.columns.tolist():
        raise ValueError("Submission columns do not match sample_submission")
    if len(submission) != len(sample_submission):
        raise ValueError(f"Submission row count {len(submission)} != sample {len(sample_submission)}")
    if submission["row_id"].tolist() != sample_submission["row_id"].tolist():
        raise ValueError("Submission row_id order does not match sample_submission")
    target = submission.drop(columns=["row_id"])
    if target.isna().any().any():
        raise ValueError("Submission contains NaN values")
    values = target.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Submission contains non-finite values")
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError("Submission probabilities must be in [0, 1]")
    return True


def read_and_validate_submission(path, sample_path):
    submission = pd.read_csv(path)
    sample = pd.read_csv(sample_path)
    validate_submission(submission, sample)
    return submission

