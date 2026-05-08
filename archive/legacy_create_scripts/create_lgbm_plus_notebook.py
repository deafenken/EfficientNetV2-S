#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def replace_once(text, old, new):
    if old not in text:
        raise ValueError(f"snippet not found:\n{old[:300]}")
    return text.replace(old, new, 1)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: create_lgbm_plus_notebook.py input.ipynb output.ipynb")

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    nb = json.loads(input_path.read_text(encoding="utf-8"))

    if nb.get("cells") and nb["cells"][0].get("cell_type") == "markdown":
        nb["cells"][0]["source"].extend(
            [
                "\n",
                "\n",
                "Local plus edits: enable time/window/site probe features and align hidden-test submission order.\n",
            ]
        )

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))

        if "Install TF 2.20" in src and "tensorflow-2.20.0" in src:
            src = """# Install TF 2.20 (required for Perch v2 StableHLO compatibility)
import subprocess
import sys
from pathlib import Path

tf_wheel_roots = [
    Path("/kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels"),
    Path("/kaggle/input/bc26-tensorflow-2-20-0/wheel"),
    Path("/kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel"),
]

installed = False
for root in tf_wheel_roots:
    tb = root / "tensorboard-2.20.0-py3-none-any.whl"
    tf = root / "tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    if tb.exists() and tf.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(tb)], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(tf)], check=True)
        print(f"Installed TensorFlow 2.20.0 wheels from {root}")
        installed = True
        break

if not installed:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tensorflow==2.20.0", "tensorboard==2.20.0"], check=True)
    print("Installed TensorFlow 2.20.0 from PyPI")
"""

        if "def build_class_features" in src:
            src = replace_once(
                src,
                """    if hour_utc is not None:\n        hour_sin = np.sin(2 * np.pi * hour_utc / 24).astype(np.float32)\n        hour_cos = np.cos(2 * np.pi * hour_utc / 24).astype(np.float32)\n        is_dawn  = ((hour_utc >= 4) & (hour_utc <= 7)).astype(np.float32)\n        is_dusk  = ((hour_utc >= 17) & (hour_utc <= 20)).astype(np.float32)\n        is_night = ((hour_utc >= 21) | (hour_utc <= 3)).astype(np.float32)\n""",
                """    if hour_utc is not None:\n        hour_utc = np.asarray(hour_utc, dtype=np.float32)\n        valid_hour = (hour_utc >= 0) & (hour_utc < 24)\n        safe_hour = np.where(valid_hour, hour_utc, 0.0)\n        hour_sin = np.where(valid_hour, np.sin(2 * np.pi * safe_hour / 24), 0.0).astype(np.float32)\n        hour_cos = np.where(valid_hour, np.cos(2 * np.pi * safe_hour / 24), 0.0).astype(np.float32)\n        is_dawn  = (valid_hour & (safe_hour >= 4) & (safe_hour <= 7)).astype(np.float32)\n        is_dusk  = (valid_hour & (safe_hour >= 17) & (safe_hour <= 20)).astype(np.float32)\n        is_night = (valid_hour & ((safe_hour >= 21) | (safe_hour <= 3))).astype(np.float32)\n""",
            )
            src = replace_once(
                src,
                """        ]\n\n    # ── NEW 2: Within-file window position""",
                """        ]\n\n    if site_id is not None:\n        site_id = np.asarray(site_id, dtype=np.float32)\n        site_known = (site_id > 0).astype(np.float32)\n        site_norm = np.clip(site_id, 0, 64) / 64.0\n        parts += [\n            site_known[:, None],\n            site_norm[:, None],\n            (raw_col * site_known)[:, None],\n        ]\n\n    # ── NEW 2: Within-file window position""",
            )
            src = replace_once(
                src,
                """    if window_idx is not None:\n        n = len(base_col)\n""",
                """    if window_idx is not None:\n        window_idx = np.asarray(window_idx, dtype=np.float32)\n""",
            )

        if "meta_full = pd.read_parquet(cache_meta_path)" in src:
            src = replace_once(
                src,
                "meta_full = pd.read_parquet(cache_meta_path)\n",
                "meta_full = pd.read_parquet(cache_meta_path)\nmeta_full[\"hour_utc\"] = pd.to_numeric(meta_full[\"hour_utc\"], errors=\"coerce\").fillna(-1).astype(np.int16)\nmeta_full[\"window_idx\"] = meta_full.groupby(\"filename\").cumcount().astype(np.int16)\n",
            )

        if "meta_test, scores_test_raw, emb_test = infer_perch_with_embeddings" in src:
            src = replace_once(
                src,
                """meta_test, scores_test_raw, emb_test = infer_perch_with_embeddings(\n    test_paths,\n    batch_files=CFG["batch_files"],\n    verbose=True,\n    proxy_reduce=CFG["proxy_reduce"],\n)\n""",
                """meta_test, scores_test_raw, emb_test = infer_perch_with_embeddings(\n    test_paths,\n    batch_files=CFG["batch_files"],\n    verbose=True,\n    proxy_reduce=CFG["proxy_reduce"],\n)\nmeta_test["hour_utc"] = pd.to_numeric(meta_test["hour_utc"], errors="coerce").fillna(-1).astype(np.int16)\nmeta_test["window_idx"] = meta_test.groupby("filename").cumcount().astype(np.int16)\n""",
            )

        if "site_ids_all, hours_all = get_file_metadata" in src:
            src = replace_once(
                src,
                "site_ids_all, hours_all = get_file_metadata(meta_full, file_list, site_to_idx, n_sites_cfg)\n",
                "site_ids_all, hours_all = get_file_metadata(meta_full, file_list, site_to_idx, n_sites_cfg)\nsite_ids_full_rows = np.array(\n    [min(site_to_idx.get(str(s), 0), n_sites_cfg - 1) for s in meta_full[\"site\"].to_numpy()],\n    dtype=np.int16,\n)\n",
            )

        if "test_site_ids, test_hours = get_file_metadata" in src:
            src = replace_once(
                src,
                """test_site_ids, test_hours = get_file_metadata(\n    meta_test, test_file_list, site_to_idx, CFG["proto_ssm"]["n_sites"]\n)\n""",
                """test_site_ids, test_hours = get_file_metadata(\n    meta_test, test_file_list, site_to_idx, CFG["proto_ssm"]["n_sites"]\n)\ntest_site_ids_rows = np.array(\n    [min(site_to_idx.get(str(s), 0), CFG["proto_ssm"]["n_sites"] - 1) for s in meta_test["site"].to_numpy()],\n    dtype=np.int16,\n)\n""",
            )

        src = src.replace(
            """        base_col=oof_base[:, cls_idx],\n    )""",
            """        base_col=oof_base[:, cls_idx],\n        hour_utc=meta_full["hour_utc"].to_numpy(),\n        site_id=site_ids_full_rows,\n        window_idx=meta_full["window_idx"].to_numpy(),\n    )""",
        )
        src = src.replace(
            """        base_col=test_base_scores[:, cls_idx],\n    )""",
            """        base_col=test_base_scores[:, cls_idx],\n        hour_utc=meta_test["hour_utc"].to_numpy(),\n        site_id=test_site_ids_rows,\n        window_idx=meta_test["window_idx"].to_numpy(),\n    )""",
        )
        src = src.replace(
            """        base_col=train_base_scores[:, cls_idx],\n    )""",
            """        base_col=train_base_scores[:, cls_idx],\n        hour_utc=meta_full["hour_utc"].to_numpy(),\n        site_id=site_ids_full_rows,\n        window_idx=meta_full["window_idx"].to_numpy(),\n    )""",
        )

        if "# --- Build submission ---" in src:
            src = replace_once(
                src,
                """submission = pd.DataFrame(probs, columns=PRIMARY_LABELS)\nsubmission.insert(0, "row_id", meta_test["row_id"].values)\nsubmission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].astype(np.float32)\n\nexpected_rows = len(test_paths) * N_WINDOWS\n""",
                """submission = pd.DataFrame(probs, columns=PRIMARY_LABELS)\nsubmission.insert(0, "row_id", meta_test["row_id"].values)\nsubmission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].astype(np.float32)\n\nif RUN_INFERENCE:\n    submission = sample_sub[["row_id"]].merge(submission, on="row_id", how="left")\n    submission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].fillna(0.0).astype(np.float32)\n    submission = submission[["row_id"] + PRIMARY_LABELS]\n\nexpected_rows = len(sample_sub) if RUN_INFERENCE else len(test_paths) * N_WINDOWS\n""",
            )

        cell["source"] = src.splitlines(keepends=True)

    output_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
