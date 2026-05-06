#!/usr/bin/env python3
import json
import sys
from pathlib import Path


VARIANTS = {
    "a7_v3_proto_045": {
        "title": "A7: original V3 with ProtoSSM blend weight 0.45",
        "replacements": [
            ("ENSEMBLE_WEIGHT_PROTO = 0.5", "ENSEMBLE_WEIGHT_PROTO = 0.45"),
        ],
    },
    "a8_v3_proto_055": {
        "title": "A8: original V3 with ProtoSSM blend weight 0.55",
        "replacements": [
            ("ENSEMBLE_WEIGHT_PROTO = 0.5", "ENSEMBLE_WEIGHT_PROTO = 0.55"),
        ],
    },
    "a9_v3_residual_030": {
        "title": "A9: original V3 with ResidualSSM correction weight 0.30",
        "replacements": [
            ('"correction_weight": 0.35', '"correction_weight": 0.30'),
        ],
    },
    "a10_v3_probe_050_050": {
        "title": "A10: original V3 with probe ensemble 50 MLP / 50 LGBM",
        "replacements": [
            (
                '"probe_ensemble_weights": {"mlp": 0.60, "lgbm": 0.40}',
                '"probe_ensemble_weights": {"mlp": 0.50, "lgbm": 0.50}',
            ),
            ("# 0.60", "# 0.50"),
            ("# 0.40", "# 0.50"),
        ],
    },
    "a11_v3_no_threshold": {
        "title": "A11: original V3 without per-class threshold sharpening",
        "replacements": [
            (
                "probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS, n_windows=N_WINDOWS)",
                "print(\"Skipping per-class threshold sharpening for ablation A11\")",
            ),
        ],
    },
    "a12_v3_proto_060": {
        "title": "A12: original V3 with ProtoSSM blend weight 0.60",
        "replacements": [
            ("ENSEMBLE_WEIGHT_PROTO = 0.5", "ENSEMBLE_WEIGHT_PROTO = 0.60"),
        ],
    },
    "a13_v3_proto_055_residual_030": {
        "title": "A13: original V3 with ProtoSSM 0.55 and ResidualSSM 0.30",
        "replacements": [
            ("ENSEMBLE_WEIGHT_PROTO = 0.5", "ENSEMBLE_WEIGHT_PROTO = 0.55"),
            ('"correction_weight": 0.35', '"correction_weight": 0.30'),
        ],
    },
    "a14_v3_proto_055_no_threshold": {
        "title": "A14: original V3 with ProtoSSM 0.55 and no threshold sharpening",
        "replacements": [
            ("ENSEMBLE_WEIGHT_PROTO = 0.5", "ENSEMBLE_WEIGHT_PROTO = 0.55"),
            (
                "probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS, n_windows=N_WINDOWS)",
                "print(\"Skipping per-class threshold sharpening for ablation A14\")",
            ),
        ],
    },
    "a15_v3_residual_030_no_threshold": {
        "title": "A15: original V3 with ResidualSSM 0.30 and no threshold sharpening",
        "replacements": [
            ('"correction_weight": 0.35', '"correction_weight": 0.30'),
            (
                "probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS, n_windows=N_WINDOWS)",
                "print(\"Skipping per-class threshold sharpening for ablation A15\")",
            ),
        ],
    },
}


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: create_v3_score_ablation_notebooks.py original_v3.ipynb output_dir"
        )

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    base_nb = json.loads(input_path.read_text(encoding="utf-8"))
    for name, spec in VARIANTS.items():
        nb = json.loads(json.dumps(base_nb))
        if nb.get("cells") and nb["cells"][0].get("cell_type") == "markdown":
            nb["cells"][0]["source"].extend(["\n", "\n", spec["title"] + "\n"])

        replacement_hits = {old: 0 for old, _ in spec["replacements"]}
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            for old, new in spec["replacements"]:
                hits = src.count(old)
                if hits:
                    src = src.replace(old, new)
                    replacement_hits[old] += hits
            cell["source"] = src.splitlines(keepends=True)

        missing = [old for old, hits in replacement_hits.items() if hits == 0]
        if missing:
            raise RuntimeError(f"{name}: replacement text not found: {missing}")

        out_path = output_dir / f"{name}.ipynb"
        out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
