#!/usr/bin/env python3
import json
import sys
from pathlib import Path


A23_DIAGNOSTIC_CELL = r'''# A23 diagnostic gate: stage-wise metrics for future variants
print("\n" + "=" * 72)
print("A23 DIAGNOSTIC GATE")
print("=" * 72)

import json
from pathlib import Path


def _sigmoid_diag(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _safe_auc_diag(y_true, scores):
    try:
        return float(macro_auc_skip_empty(y_true.astype(np.float32), scores.astype(np.float32)))
    except Exception as e:
        print(f"  AUC skipped: {e}")
        return None


def _stage_summary(name, scores, y_true=Y_FULL):
    arr = np.asarray(scores, dtype=np.float32)
    return {
        "name": name,
        "macro_auc": _safe_auc_diag(y_true, arr),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _family_auc_diag(scores, y_true=Y_FULL):
    out = {}
    arr = np.asarray(scores, dtype=np.float32)
    labels = np.array(PRIMARY_LABELS)
    families = sorted({CLASS_NAME_MAP.get(label, "Unknown") for label in PRIMARY_LABELS})
    for fam in families:
        idx = np.array([i for i, label in enumerate(labels) if CLASS_NAME_MAP.get(label, "Unknown") == fam], dtype=np.int32)
        if len(idx) == 0:
            continue
        auc = _safe_auc_diag(y_true[:, idx], arr[:, idx])
        if auc is not None:
            out[fam] = auc
    return out


def _postprocess_train_scores(train_scores):
    scaled = train_scores / class_temperatures[None, :]
    pp = _sigmoid_diag(scaled)
    top_k = CFG.get("file_level_top_k", 0)
    if top_k > 0:
        pp = file_level_confidence_scale(pp, n_windows=N_WINDOWS, top_k=top_k)
        pp = np.clip(pp, 0.0, 1.0)
    if CFG.get("rank_aware_scale", False):
        pp = rank_aware_scaling(pp, n_windows=N_WINDOWS, power=CFG.get("rank_aware_power", 0.5))
        pp = np.clip(pp, 0.0, 1.0)
    alpha = CFG.get("delta_shift_alpha", 0.0)
    if alpha > 0:
        pp = adaptive_delta_smooth(pp, n_windows=N_WINDOWS, base_alpha=alpha)
        pp = np.clip(pp, 0.0, 1.0)
    thresholds = np.full(N_CLASSES, 0.5, dtype=np.float32)
    return apply_per_class_thresholds(pp, thresholds, n_windows=N_WINDOWS)


diagnostics = {
    "variant": "A23 diagnostic gate",
    "score_anchor_public": 0.928,
    "mode": globals().get("MODE", "hidden_test" if RUN_INFERENCE else "dry_run"),
    "run_inference": bool(RUN_INFERENCE),
    "n_rows": int(Y_FULL.shape[0]),
    "n_classes": int(Y_FULL.shape[1]),
    "notes": [
        "oof_base uses GroupKFold prior tables and is the cleanest diagnostic stage.",
        "proto/probe/residual train diagnostics are proxy checks because the public A0 notebook trains final models once.",
        "Future A24-A30 submissions should use this file to decide which isolated changes deserve a leaderboard slot.",
    ],
    "stages": [],
    "family_auc": {},
}

stage_inputs = []
stage_inputs.append(("raw_perch_logits", scores_full_raw))
stage_inputs.append(("oof_prior_fused_logits", oof_base))
stage_inputs.append(("final_prior_fused_train_logits", train_base_scores))
stage_inputs.append(("proto_train_logits_proxy", proto_train_scores.reshape(-1, N_CLASSES)))
stage_inputs.append(("probe_train_logits_proxy", mlp_train_scores_flat))
stage_inputs.append(("first_pass_train_logits_proxy", first_pass_files.reshape(-1, N_CLASSES)))

try:
    emb_all_r = torch.tensor(emb_files, dtype=torch.float32)
    fp_all_r = torch.tensor(first_pass_files, dtype=torch.float32)
    site_all_r = torch.tensor(site_ids_all, dtype=torch.long)
    hour_all_r = torch.tensor(hours_all, dtype=torch.long)
    res_model.eval()
    with torch.no_grad():
        corr_all = res_model(emb_all_r, fp_all_r, site_ids=site_all_r, hours=hour_all_r).numpy()
    residual_corrected_train = (
        first_pass_files + CORRECTION_WEIGHT * corr_all
    ).reshape(-1, N_CLASSES).astype(np.float32)
    stage_inputs.append(("residual_corrected_train_logits_proxy", residual_corrected_train))
    diagnostics["family_auc"]["residual_corrected_train_logits_proxy"] = _family_auc_diag(residual_corrected_train)

    postprocessed_train = _postprocess_train_scores(residual_corrected_train)
    stage_inputs.append(("postprocessed_train_probs_proxy", postprocessed_train))
    diagnostics["family_auc"]["postprocessed_train_probs_proxy"] = _family_auc_diag(postprocessed_train)
except Exception as e:
    diagnostics["residual_train_diagnostic_error"] = repr(e)
    print(f"Residual/postprocess train diagnostic skipped: {e}")

for name, scores in stage_inputs:
    summary = _stage_summary(name, scores)
    diagnostics["stages"].append(summary)
    print(
        f"{name:42s} auc={summary['macro_auc']} "
        f"mean={summary['mean']:.5f} std={summary['std']:.5f}"
    )

diagnostics["family_auc"]["raw_perch_logits"] = _family_auc_diag(scores_full_raw)
diagnostics["family_auc"]["oof_prior_fused_logits"] = _family_auc_diag(oof_base)
diagnostics["family_auc"]["first_pass_train_logits_proxy"] = _family_auc_diag(
    first_pass_files.reshape(-1, N_CLASSES)
)

if not RUN_INFERENCE:
    try:
        meta_full_indexed = meta_full.set_index("row_id")
        test_row_ids = meta_test["row_id"].values
        matched = np.array([rid in meta_full_indexed.index for rid in test_row_ids])
        diagnostics["dry_run_matched_rows"] = int(matched.sum())
        if matched.sum() > 0:
            full_positions = [meta_full_indexed.index.get_loc(rid) for rid in test_row_ids[matched]]
            diagnostics["dry_run_final_auc"] = _safe_auc_diag(Y_FULL[full_positions], probs[matched])
            diagnostics["dry_run_final_mean"] = float(probs[matched].mean())
            print(
                f"dry_run_final_auc={diagnostics['dry_run_final_auc']} "
                f"matched_rows={diagnostics['dry_run_matched_rows']}"
            )
    except Exception as e:
        diagnostics["dry_run_error"] = repr(e)

out_path = Path("a23_diagnostics.json")
out_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
print(f"Saved {out_path}")
print("=" * 72)
'''


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: create_a23_diagnostic_notebook.py input.ipynb output.ipynb")

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nb = json.loads(input_path.read_text(encoding="utf-8"))

    if nb.get("cells") and nb["cells"][0].get("cell_type") == "markdown":
        source = nb["cells"][0].get("source", [])
        if isinstance(source, str):
            source = [source]
        source.extend(
            [
                "\n",
                "\n",
                "A23: diagnostic gate. Predictions are intentionally unchanged; the notebook writes `a23_diagnostics.json` for OOF/proxy stage analysis.\n",
            ]
        )
        nb["cells"][0]["source"] = source

    nb.setdefault("cells", []).append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": A23_DIAGNOSTIC_CELL.splitlines(keepends=True),
        }
    )

    output_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {output_path} with A23 diagnostic cell")


if __name__ == "__main__":
    main()
