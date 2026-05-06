#!/usr/bin/env python3
import json
import sys
from pathlib import Path


A24_MIRROR_BLOCK = r'''# A24 sonotype mirroring: conservative transfer from the needless090 reference.
if CFG.get("a24_sonotype_mirroring", False):
    A24_MIRROR_GROUPS = (
        ("47158son15", "47158son16"),
        ("47158son09", "47158son12"),
        ("47158son02", "47158son14"),
        ("47158son13", "47158son21", "47158son22", "47158son23"),
    )
    label_to_idx_a24 = {label: i for i, label in enumerate(PRIMARY_LABELS)}
    applied_mirror_groups = 0
    for group in A24_MIRROR_GROUPS:
        idx = [label_to_idx_a24[label] for label in group if label in label_to_idx_a24]
        if len(idx) < 2:
            continue
        group_max = probs[:, idx].max(axis=1, keepdims=True)
        probs[:, idx] = group_max
        applied_mirror_groups += 1
    print(f"A24: sonotype mirroring applied to {applied_mirror_groups} groups")

Path("a24_config.json").write_text(
    json.dumps(
        {
            "variant": "A24 V3 probe-heavy softpost mirror",
            "public_anchor": 0.928,
            "proto_weight": float(ENSEMBLE_WEIGHT_PROTO),
            "residual_correction_weight": float(CFG["residual_ssm"]["correction_weight"]),
            "rank_aware_power": float(CFG["rank_aware_power"]),
            "delta_shift_alpha": float(CFG["delta_shift_alpha"]),
            "threshold_sharpening": bool(CFG.get("use_threshold_sharpening", True)),
            "sonotype_mirroring": bool(CFG.get("a24_sonotype_mirroring", False)),
            "notes": [
                "A23 diagnostics favored the probe path over proto/residual proxies.",
                "Residual correction and hard threshold sharpening are disabled.",
                "Sonotype mirroring is the small CPU-legal transfer from needless090's 0.934 reference.",
            ],
        },
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
'''


def _cell_source(cell):
    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    return "".join(source)


def _set_cell_source(cell, source):
    cell["source"] = source.splitlines(keepends=True)


def _replace_once(source, old, new, label):
    count = source.count(old)
    if count != 1:
        raise ValueError(f"expected exactly one {label}, found {count}")
    return source.replace(old, new, 1)


def main():
    if len(sys.argv) not in (1, 3):
        raise SystemExit(
            "usage: create_a24_probe_heavy_softpost_notebook.py [input.ipynb output.ipynb]"
        )

    if len(sys.argv) == 3:
        input_path = Path(sys.argv[1])
        output_path = Path(sys.argv[2])
    else:
        input_path = Path(
            "references/public_baselines/youssefmo942009_version_3_lgbm/version_3_lgbm.ipynb"
        )
        output_path = Path(
            "references/public_baselines/youssefmo942009_version_3_lgbm/"
            "score_ablations/a24_v3_probe_heavy_softpost_mirror.ipynb"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nb = json.loads(input_path.read_text(encoding="utf-8"))

    if nb.get("cells") and nb["cells"][0].get("cell_type") == "markdown":
        source = _cell_source(nb["cells"][0])
        source += (
            "\n\n"
            "A24: probe-heavy soft post-processing variant. "
            "Uses A23 diagnostics to lower ProtoSSM weight, disable ResidualSSM final correction, "
            "skip hard threshold sharpening, and apply conservative sonotype mirroring.\n"
        )
        _set_cell_source(nb["cells"][0], source)

    replacements = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        original = source

        source = source.replace('"rank_aware_power": 0.4,', '"rank_aware_power": 0.35,')
        source = source.replace('"delta_shift_alpha": 0.20,', '"delta_shift_alpha": 0.10,')
        source = source.replace('"correction_weight": 0.35,', '"correction_weight": 0.0,')

        if "ENSEMBLE_WEIGHT_PROTO = 0.5" in source:
            source = _replace_once(
                source,
                "ENSEMBLE_WEIGHT_PROTO = 0.5\n",
                (
                    "ENSEMBLE_WEIGHT_PROTO = 0.40\n"
                    "CFG[\"use_threshold_sharpening\"] = False\n"
                    "CFG[\"a24_sonotype_mirroring\"] = True\n"
                    "print(\"A24 config: proto=0.40 residual=0.00 rank_power=0.35 "
                    "delta=0.10 threshold=off mirror=on\")\n"
                ),
                "ENSEMBLE_WEIGHT_PROTO assignment",
            )

        if "probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS, n_windows=N_WINDOWS)" in source:
            source = _replace_once(
                source,
                "probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS, n_windows=N_WINDOWS)\n",
                (
                    "if CFG.get(\"use_threshold_sharpening\", True):\n"
                    "    probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS, n_windows=N_WINDOWS)\n"
                    "else:\n"
                    "    print(\"A24: threshold sharpening skipped\")\n"
                ),
                "threshold sharpening call",
            )

        if "# --- Build submission ---" in source:
            source = _replace_once(
                source,
                "# --- Build submission ---",
                A24_MIRROR_BLOCK + "\n\n# --- Build submission ---",
                "Build submission marker",
            )

        if source != original:
            replacements += 1
            _set_cell_source(cell, source)

    if replacements == 0:
        raise RuntimeError("no A24 replacements were applied")

    output_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {output_path} with A24 probe-heavy softpost mirror edits")


if __name__ == "__main__":
    main()
