import numpy as np


def seq_features_1d(values, n_windows=12):
    values = np.asarray(values, dtype=np.float32)
    if len(values) % n_windows != 0:
        raise ValueError(f"Expected length divisible by {n_windows}, got {len(values)}")
    x = values.reshape(-1, n_windows)

    prev_v = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
    next_v = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
    mean_v = np.repeat(x.mean(axis=1), n_windows)
    max_v = np.repeat(x.max(axis=1), n_windows)
    std_v = np.repeat(x.std(axis=1), n_windows)
    return prev_v, next_v, mean_v, max_v, std_v


def build_probe_features(
    embedding_projection,
    raw_scores,
    prior_scores,
    base_scores,
    hour_utc=None,
    site_id=None,
    window_idx=None,
    n_windows=12,
):
    embedding_projection = np.asarray(embedding_projection, dtype=np.float32)
    raw_scores = np.asarray(raw_scores, dtype=np.float32)
    prior_scores = np.asarray(prior_scores, dtype=np.float32)
    base_scores = np.asarray(base_scores, dtype=np.float32)

    prev_base, next_base, mean_base, max_base, std_base = seq_features_1d(
        base_scores, n_windows=n_windows
    )
    diff_mean = base_scores - mean_base
    diff_prev = base_scores - prev_base
    diff_next = base_scores - next_base

    parts = [
        embedding_projection,
        raw_scores[:, None],
        prior_scores[:, None],
        base_scores[:, None],
        prev_base[:, None],
        next_base[:, None],
        mean_base[:, None],
        max_base[:, None],
        std_base[:, None],
        diff_mean[:, None],
        diff_prev[:, None],
        diff_next[:, None],
        (raw_scores * prior_scores)[:, None],
        (raw_scores * base_scores)[:, None],
        (prior_scores * base_scores)[:, None],
    ]

    if hour_utc is not None:
        hour_utc = np.asarray(hour_utc, dtype=np.float32)
        valid_hour = (hour_utc >= 0) & (hour_utc < 24)
        safe_hour = np.where(valid_hour, hour_utc, 0.0)
        hour_sin = np.where(valid_hour, np.sin(2 * np.pi * safe_hour / 24), 0.0)
        hour_cos = np.where(valid_hour, np.cos(2 * np.pi * safe_hour / 24), 0.0)
        is_dawn = valid_hour & (safe_hour >= 4) & (safe_hour <= 7)
        is_dusk = valid_hour & (safe_hour >= 17) & (safe_hour <= 20)
        is_night = valid_hour & ((safe_hour >= 21) | (safe_hour <= 3))
        parts.extend(
            [
                hour_sin.astype(np.float32)[:, None],
                hour_cos.astype(np.float32)[:, None],
                is_dawn.astype(np.float32)[:, None],
                is_dusk.astype(np.float32)[:, None],
                is_night.astype(np.float32)[:, None],
                (raw_scores * is_night.astype(np.float32))[:, None],
                (raw_scores * is_dawn.astype(np.float32))[:, None],
            ]
        )

    if site_id is not None:
        site_id = np.asarray(site_id, dtype=np.float32)
        site_known = (site_id > 0).astype(np.float32)
        site_norm = np.clip(site_id, 0, 64) / 64.0
        parts.extend(
            [
                site_known[:, None],
                site_norm[:, None],
                (raw_scores * site_known)[:, None],
            ]
        )

    if window_idx is not None:
        window_idx = np.asarray(window_idx, dtype=np.float32)
        win_norm = window_idx / max(float(n_windows - 1), 1.0)
        win_early = window_idx <= 2
        win_late = window_idx >= n_windows - 3
        is_peak = base_scores == max_base
        parts.extend(
            [
                win_norm.astype(np.float32)[:, None],
                win_early.astype(np.float32)[:, None],
                win_late.astype(np.float32)[:, None],
                is_peak.astype(np.float32)[:, None],
            ]
        )

    score_range = max_base - base_scores
    score_accel = diff_prev - diff_next
    parts.extend(
        [
            score_range[:, None],
            score_accel[:, None],
            np.clip(raw_scores, 0, None)[:, None],
            (raw_scores**2)[:, None],
        ]
    )

    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)

