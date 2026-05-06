#!/usr/bin/env python3
import json
import sys
from pathlib import Path


OLD_APPLY_THRESHOLDS = """def apply_per_class_thresholds(scores, thresholds, n_windows=12):
    \"\"\"V17: Apply per-class thresholds to convert scores to binary predictions.\"\"\"
    N, C = scores.shape
    assert C == len(thresholds)
    
    # For competition, we submit probabilities but threshold for metrics
    # Apply threshold as a scaling factor that sharpens confident predictions
    scaled = np.copy(scores)
    
    for c in range(C):
        t = thresholds[c]
        # Sharpen: push above-threshold scores higher, below-threshold lower
        mask_above = scores[:, c] > t
        scaled[mask_above, c] = 0.5 + 0.5 * (scores[mask_above, c] - t) / (1 - t + 1e-8)
        scaled[~mask_above, c] = 0.5 * scores[~mask_above, c] / (t + 1e-8)
    
    return np.clip(scaled, 0, 1)
"""

NEW_CALIBRATION_BLOCK = """def compute_frequency_based_thresholds(Y, n_windows=12, base=0.50, scale=0.15):
    \"\"\"Rare species get lower thresholds, common species get higher thresholds.\"\"\"
    n_files = Y.shape[0] // n_windows
    file_y = Y.reshape(n_files, n_windows, -1).max(axis=1)
    pos_freq = file_y.mean(axis=0)
    log_freq = np.log(pos_freq + 1e-6)
    log_max = np.log(pos_freq.max() + 1e-6)
    thresholds = base - scale * log_freq / (log_max + 1e-6)
    return np.clip(thresholds, 0.20, 0.70).astype(np.float32)


def compute_frequency_based_temperatures(Y, n_windows=12, t_rare=0.85, t_common=1.15):
    \"\"\"Rare species stay sharper, common species get softer temperatures.\"\"\"
    n_files = Y.shape[0] // n_windows
    file_y = Y.reshape(n_files, n_windows, -1).max(axis=1)
    pos_freq = file_y.mean(axis=0)
    log_freq = np.log(pos_freq + 1e-6)
    log_max = np.log(pos_freq.max() + 1e-6)
    temps = t_rare + (t_common - t_rare) * log_freq / (log_max + 1e-6)
    return np.clip(temps, 0.80, 1.30).astype(np.float32)


def compute_frequency_based_ensemble_weights(Y, n_windows=12, w_rare=0.30, w_common=0.70):
    \"\"\"Rare species lean more on probe/prior, common species lean more on ProtoSSM.\"\"\"
    n_files = Y.shape[0] // n_windows
    file_y = Y.reshape(n_files, n_windows, -1).max(axis=1)
    pos_freq = file_y.mean(axis=0)
    log_freq = np.log(pos_freq + 1e-6)
    log_max = np.log(pos_freq.max() + 1e-6)
    weights = w_rare + (w_common - w_rare) * log_freq / (log_max + 1e-6)
    return np.clip(weights, 0.10, 0.90).astype(np.float32)


def apply_per_class_thresholds(scores, thresholds, n_windows=12):
    \"\"\"Sharpen probabilities with class-wise thresholds.\"\"\"
    N, C = scores.shape
    assert C == len(thresholds)
    scaled = np.copy(scores)
    for c in range(C):
        t = thresholds[c]
        mask_above = scores[:, c] > t
        scaled[mask_above, c] = 0.5 + 0.5 * (scores[mask_above, c] - t) / (1 - t + 1e-8)
        scaled[~mask_above, c] = 0.5 * scores[~mask_above, c] / (t + 1e-8)
    return np.clip(scaled, 0, 1)
"""

NEW_POSTPROCESS_CELL = """# Cell 18 — A22 optimized post-processing pipeline
print(\"\\n\" + \"=\" * 60)
print(\"A22 OPTIMIZED PANTANAL POST-PROCESSING\")
print(\"=\" * 60)

PER_CLASS_THRESHOLDS = compute_frequency_based_thresholds(Y_FULL, n_windows=N_WINDOWS)
PER_CLASS_TEMPS = compute_frequency_based_temperatures(Y_FULL, n_windows=N_WINDOWS)
PER_CLASS_WEIGHTS = compute_frequency_based_ensemble_weights(Y_FULL, n_windows=N_WINDOWS)

print(
    f\"Frequency defaults: thresholds mean={PER_CLASS_THRESHOLDS.mean():.3f}, \"
    f\"temps mean={PER_CLASS_TEMPS.mean():.3f}, weights mean={PER_CLASS_WEIGHTS.mean():.3f}\"
)

if CFG.get(\"use_per_class_ensemble_weight\", True):
    print(\"[1] Using per-class ensemble weights\")
    final_test_scores = (
        PER_CLASS_WEIGHTS[None, :] * proto_scores_flat
        + (1.0 - PER_CLASS_WEIGHTS[None, :]) * mlp_scores
    ).astype(np.float32)
else:
    print(f\"[1] Using global ensemble weight: ProtoSSM={ENSEMBLE_WEIGHT_PROTO:.2f}\")
    final_test_scores = (
        ENSEMBLE_WEIGHT_PROTO * proto_scores_flat
        + (1.0 - ENSEMBLE_WEIGHT_PROTO) * mlp_scores
    ).astype(np.float32)

if res_model is not None and CORRECTION_WEIGHT > 0:
    print(f\"[2] Applying residual correction (weight={CORRECTION_WEIGHT:.2f})\")
    first_pass_test_files, _ = reshape_to_files(final_test_scores, meta_test)
    first_pass_test_t = torch.tensor(first_pass_test_files, dtype=torch.float32).to(DEVICE)

    res_model.eval()
    with torch.no_grad():
        test_correction = res_model(
            emb_test_tensor, first_pass_test_t,
            site_ids=test_site_tensor, hours=test_hour_tensor
        ).cpu().numpy()

    final_test_scores += CORRECTION_WEIGHT * test_correction.reshape(-1, N_CLASSES).astype(np.float32)
else:
    print(\"[2] Residual correction skipped\")

print(
    f\"[3] Per-class temperatures: mean={PER_CLASS_TEMPS.mean():.3f}, \"
    f\"range=[{PER_CLASS_TEMPS.min():.2f}, {PER_CLASS_TEMPS.max():.2f}]\"
)
scaled_scores = final_test_scores / PER_CLASS_TEMPS[None, :]
probs = 1.0 / (1.0 + np.exp(-scaled_scores))

top_k = CFG.get(\"file_level_top_k\", 2)
if top_k > 0:
    print(f\"[4] File-level confidence scaling (top_k={top_k})\")
    probs = file_level_confidence_scale(probs, n_windows=N_WINDOWS, top_k=top_k)
    probs = np.clip(probs, 0.0, 1.0)

if CFG.get(\"rank_aware_scale\", False):
    power = CFG.get(\"rank_aware_power\", 0.35)
    print(f\"[5] Rank-aware scaling (power={power})\")
    probs = rank_aware_scaling(probs, n_windows=N_WINDOWS, power=power)
    probs = np.clip(probs, 0.0, 1.0)


def adaptive_delta_smooth_conf(scores, n_windows, base_alpha=0.12, min_conf=0.70):
    n_files = scores.shape[0] // n_windows
    out = scores.copy()
    view = out.reshape(n_files, n_windows, -1)
    src = scores.reshape(n_files, n_windows, -1)
    for i in range(1, n_windows - 1):
        conf = src[:, i, :].max(axis=-1, keepdims=True)
        alpha = np.where(conf < min_conf, base_alpha * (1.0 - conf), 0.0)
        neighbor_avg = 0.5 * (src[:, i - 1, :] + src[:, i + 1, :])
        view[:, i, :] = (1.0 - alpha) * src[:, i, :] + alpha * neighbor_avg
    return out


alpha = CFG.get(\"delta_shift_alpha\", 0.12)
if alpha > 0:
    min_conf = CFG.get(\"smooth_min_conf\", 0.70)
    print(f\"[6] Adaptive delta smoothing (alpha={alpha}, min_conf={min_conf})\")
    probs = adaptive_delta_smooth_conf(probs, N_WINDOWS, base_alpha=alpha, min_conf=min_conf)
    probs = np.clip(probs, 0.0, 1.0)

print(
    f\"[7] Per-class threshold sharpening: mean={PER_CLASS_THRESHOLDS.mean():.3f}, \"
    f\"range=[{PER_CLASS_THRESHOLDS.min():.2f}, {PER_CLASS_THRESHOLDS.max():.2f}]\"
)
probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS, n_windows=N_WINDOWS)

submission = pd.DataFrame(probs, columns=PRIMARY_LABELS)
submission.insert(0, \"row_id\", meta_test[\"row_id\"].values)
submission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].astype(np.float32)

expected_rows = len(test_paths) * N_WINDOWS
assert len(submission) == expected_rows, f\"Expected {expected_rows}, got {len(submission)}\"
assert submission.columns.tolist() == [\"row_id\"] + PRIMARY_LABELS
assert not submission.isna().any().any()

submission.to_csv(\"submission.csv\", index=False)

print(\"\\nSaved submission.csv\")
print(\"Submission shape:\", submission.shape)
print(f\"Final score range: {probs.min():.6f} to {probs.max():.6f}\")
print(f\"Final mean: {probs.mean():.4f}\")
print(submission.iloc[:3, :8])
"""

SKIP_PSEUDO_CELL = """# A22: skip pseudo-label generation and extra export cells in Kaggle submit runs
print(\"Skipping pseudo-label generation for A22 optimized submit variant.\")
"""


def patch_cell(src: str) -> str:
    if 'DEVICE = torch.device("cuda")  # Competition constraint' in src:
        src = src.replace(
            'DEVICE = torch.device("cuda")  # Competition constraint',
            'DEVICE = torch.device("cpu")  # A22: force CPU-safe PyTorch path for Kaggle runtime compatibility',
        )

    if 'CFG["frozen_best_probe"] = {\n    "pca_dim": 128, "min_pos": 5, "C": 0.75, "alpha": 0.45\n}\nprint("✅ V18 CFG loaded")' in src:
        src = src.replace(
            'CFG["frozen_best_probe"] = {\n    "pca_dim": 128, "min_pos": 5, "C": 0.75, "alpha": 0.45\n}\nprint("✅ V18 CFG loaded")',
            'CFG["frozen_best_probe"] = {\n    "pca_dim": 128, "min_pos": 5, "C": 0.75, "alpha": 0.55\n}\nCFG["probe_backend"] = "lgbm"\nCFG["lgbm_params"] = {\n    "objective": "binary",\n    "metric": "auc",\n    "boosting_type": "gbdt",\n    "num_leaves": 63,\n    "learning_rate": 0.05,\n    "feature_fraction": 0.8,\n    "bagging_fraction": 0.8,\n    "bagging_freq": 5,\n    "min_child_samples": 10,\n    "lambda_l1": 0.1,\n    "lambda_l2": 0.1,\n    "n_estimators": 250,\n    "verbose": -1,\n    "random_state": 42,\n    "n_jobs": -1,\n}\nCFG["use_per_class_temperature"] = True\nCFG["use_per_class_ensemble_weight"] = True\nCFG["adaptive_smooth"] = True\nCFG["smooth_min_conf"] = 0.70\nCFG["rank_aware_power"] = 0.35\nCFG["delta_shift_alpha"] = 0.12\nprint("✅ A22 CFG loaded (CPU-safe + LGBM probes + frequency-based post-process)")',
        )

    if OLD_APPLY_THRESHOLDS in src:
        src = src.replace(OLD_APPLY_THRESHOLDS, NEW_CALIBRATION_BLOCK)

    if '# Cell 18 — V17: Full post-processing pipeline' in src:
        src = NEW_POSTPROCESS_CELL

    if '# --- 极小 Batch 修复版 Step 4 ---' in src:
        src = SKIP_PSEUDO_CELL

    if '# ── BIRD_CLEF 2026: 超级集成伪标签生成引擎' in src:
        src = SKIP_PSEUDO_CELL

    if 'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")' in src:
        src = src.replace(
            'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
            'DEVICE = torch.device("cpu")',
        )

    return src


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: create_pantanal_optimized_notebook.py input_notebook.ipynb output_notebook.ipynb"
        )

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nb = json.loads(input_path.read_text(encoding="utf-8"))
    changed = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        new_src = patch_cell(src)
        if new_src != src:
            cell["source"] = new_src.splitlines(keepends=True)
            changed += 1

    if changed == 0:
        raise RuntimeError("No notebook cells were modified")

    if nb.get("cells") and nb["cells"][0].get("cell_type") == "markdown":
        source = nb["cells"][0].get("source", [])
        if isinstance(source, str):
            source = [source]
        source.extend(
            [
                "\n",
                "\n",
                "A22: optimized Pantanal ONNX with CPU-safe ProtoSSM, LGBM probes, and stronger frequency-aware post-processing.\n",
            ]
        )
        nb["cells"][0]["source"] = source

    output_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {output_path} ({changed} cells changed)")


if __name__ == "__main__":
    main()
