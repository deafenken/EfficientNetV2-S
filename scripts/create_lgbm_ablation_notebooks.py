#!/usr/bin/env python3
import json
import sys
from pathlib import Path


VARIANTS = {
    "a2_no_residual": {
        "title": "A2: skip ResidualSSM correction",
        "replacements": [],
        "drop_residual_cell": True,
    },
    "a3_proto_035": {
        "title": "A3: lower ProtoSSM blend weight to 0.35",
        "replacements": [
            ("ENSEMBLE_WEIGHT_PROTO = 0.5", "ENSEMBLE_WEIGHT_PROTO = 0.35"),
        ],
        "drop_residual_cell": False,
    },
    "a4_no_threshold_sharpen": {
        "title": "A4: skip per-class threshold sharpening",
        "replacements": [
            (
                "probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS, n_windows=N_WINDOWS)",
                "print(\"Skipping per-class threshold sharpening for ablation A4\")",
            ),
        ],
        "drop_residual_cell": False,
    },
}


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: create_lgbm_ablation_notebooks.py plus.ipynb output_dir")

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    base_nb = json.loads(input_path.read_text(encoding="utf-8"))
    for name, spec in VARIANTS.items():
        nb = json.loads(json.dumps(base_nb))
        if nb.get("cells") and nb["cells"][0].get("cell_type") == "markdown":
            nb["cells"][0]["source"].extend(["\n", "\n", spec["title"] + "\n"])

        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if spec.get("drop_residual_cell") and "# ResidualSSM: second-pass boosting" in src:
                src = (
                    "# ResidualSSM: second-pass boosting\n"
                    "print(\"Skipping ResidualSSM correction for ablation A2\")\n"
                )
            for old, new in spec.get("replacements", []):
                if old not in src:
                    continue
                src = src.replace(old, new)
            cell["source"] = src.splitlines(keepends=True)

        out_path = output_dir / f"{name}.ipynb"
        out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

