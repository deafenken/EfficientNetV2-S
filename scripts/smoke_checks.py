#!/usr/bin/env python3
import numpy as np
import pandas as pd

from birdclef2026.lgbm_features import build_probe_features
from birdclef2026.postprocess import (
    adaptive_delta_smooth,
    align_submission_to_sample,
    file_level_confidence_scale,
    rank_aware_scaling,
    sigmoid_clip,
    validate_submission,
)


def main():
    n_rows = 24
    n_classes = 4
    emb = np.zeros((n_rows, 8), dtype=np.float32)
    raw = np.linspace(-2, 2, n_rows).astype(np.float32)
    features = build_probe_features(
        emb,
        raw_scores=raw,
        prior_scores=raw * 0.1,
        base_scores=raw * 0.2,
        hour_utc=np.arange(n_rows) % 24,
        site_id=np.ones(n_rows),
        window_idx=np.arange(n_rows) % 12,
    )
    assert features.shape[0] == n_rows
    assert np.isfinite(features).all()

    scores = np.linspace(-3, 3, n_rows * n_classes, dtype=np.float32).reshape(n_rows, n_classes)
    probs = sigmoid_clip(scores)
    probs = file_level_confidence_scale(probs, n_windows=12, top_k=2)
    probs = rank_aware_scaling(probs, n_windows=12, power=0.4)
    probs = adaptive_delta_smooth(probs, n_windows=12, base_alpha=0.2)
    assert probs.shape == (n_rows, n_classes)
    assert np.isfinite(probs).all()
    assert probs.min() >= 0.0 and probs.max() <= 1.0

    sample = pd.DataFrame({"row_id": ["b", "a"], "c0": [0.0, 0.0], "c1": [0.0, 0.0]})
    sub = pd.DataFrame({"row_id": ["a", "b"], "c0": [0.2, 0.1], "c1": [0.4, 0.3]})
    aligned = align_submission_to_sample(sub, sample)
    assert aligned["row_id"].tolist() == ["b", "a"]
    validate_submission(aligned, sample)
    print("smoke checks ok")


if __name__ == "__main__":
    main()

