#!/usr/bin/env python3
import json
from pathlib import Path


SRC = Path("references/public_baselines/pacerq_v51_onnx_cache/v51-onnx-cache-0-943-params.ipynb")
OUT_DIR = Path("references/public_baselines/pacerq_v51_onnx_cache/ablations")


def replace_once(source: str, old: str, new: str) -> str:
    if old not in source:
        raise ValueError(f"pattern not found:\n{old[:240]}")
    return source.replace(old, new, 1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb = json.loads(SRC.read_text(encoding="utf-8"))

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))

        if '"full_cache_input_dir": Path("/kaggle/input/datasets/yuriygreben/perch-meta")' in source:
            source = replace_once(
                source,
                '    "full_cache_input_dir": Path("/kaggle/input/datasets/yuriygreben/perch-meta") if Path("/kaggle/input/datasets/yuriygreben/perch-meta").exists() else Path("/kaggle/input/perch-meta") if Path("/kaggle/input/perch-meta").exists() else Path("/kaggle/input/datasets/jaejohn/perch-meta"),\n',
                '    "full_cache_input_dir": next(\n'
                '        (p for p in [\n'
                '            Path("/kaggle/input/perch-meta-0-943"),\n'
                '            Path("/kaggle/input/datasets/atahalam/perch-meta-0-943"),\n'
                '            Path("/kaggle/input/datasets/yuriygreben/perch-meta"),\n'
                '            Path("/kaggle/input/perch-meta"),\n'
                '            Path("/kaggle/input/datasets/jaejohn/perch-meta"),\n'
                '        ] if p.exists()),\n'
                '        Path("/kaggle/input/datasets/jaejohn/perch-meta"),\n'
                '    ),\n',
            )

        if "Building honest 5-fold OOF meta-features..." in source:
            source = replace_once(
                source,
                'print("Building honest 5-fold OOF meta-features...")\n'
                'oof_base, oof_prior, oof_fold_id = build_oof_base_prior(scores_full_raw, meta_full, sc_clean, Y_SC, n_splits=5, verbose=False)\n'
                'baseline_oof_auc = macro_auc_skip_empty(Y_FULL, oof_base)\n',
                'print("Building honest 5-fold OOF meta-features...")\n'
                'oof_base, oof_prior, oof_fold_id = build_oof_base_prior(scores_full_raw, meta_full, sc_clean, Y_SC, n_splits=5, verbose=False)\n'
                '_oof_candidates = [\n'
                '    Path("/kaggle/input/perch-meta-0-943/full_oof_meta_features.npz"),\n'
                '    Path("/kaggle/input/datasets/atahalam/perch-meta-0-943/full_oof_meta_features.npz"),\n'
                ']\n'
                'for _oof_file in _oof_candidates:\n'
                '    if _oof_file.exists():\n'
                '        _oof = np.load(_oof_file)\n'
                '        if _oof["oof_base"].shape == oof_base.shape and _oof["oof_prior"].shape == oof_prior.shape:\n'
                '            oof_base = _oof["oof_base"].astype(np.float32)\n'
                '            oof_prior = _oof["oof_prior"].astype(np.float32)\n'
                '            oof_fold_id = _oof["fold_id"].astype(np.int16)\n'
                '            print(f"Loaded cached OOF meta-features from {_oof_file}")\n'
                '        else:\n'
                '            print(f"Cached OOF shape mismatch, keeping rebuilt features: {_oof_file}")\n'
                '        break\n'
                'baseline_oof_auc = macro_auc_skip_empty(Y_FULL, oof_base)\n',
            )

        if "ENSEMBLE_WEIGHT_PROTO = 0.55" in source:
            source = replace_once(
                source,
                'ENSEMBLE_WEIGHT_PROTO = 0.55\n'
                'print(f"Ensemble weight (ProtoSSM): {ENSEMBLE_WEIGHT_PROTO}")\n',
                'ENSEMBLE_WEIGHT_PROTO = 0.55\n'
                '_weight_candidates = [\n'
                '    Path("/kaggle/input/perch-meta-0-943/ensemble_weight_v20.json"),\n'
                '    Path("/kaggle/input/datasets/atahalam/perch-meta-0-943/ensemble_weight_v20.json"),\n'
                ']\n'
                'for _weight_file in _weight_candidates:\n'
                '    if _weight_file.exists():\n'
                '        ENSEMBLE_WEIGHT_PROTO = float(json.loads(_weight_file.read_text())["ensemble_weight_proto"])\n'
                '        print(f"Loaded ProtoSSM ensemble weight from {_weight_file}: {ENSEMBLE_WEIGHT_PROTO:.2f}")\n'
                '        break\n'
                'print(f"Ensemble weight (ProtoSSM): {ENSEMBLE_WEIGHT_PROTO}")\n',
            )

        cell["source"] = source.splitlines(keepends=True)

    out = OUT_DIR / "a6_v51_cached_oof_weight060.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
