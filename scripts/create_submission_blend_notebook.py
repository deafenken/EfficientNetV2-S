#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def notebook_source(variant_name, sources, weights, blend_mode):
    source_json = json.dumps(sources, indent=4)
    weights_json = json.dumps(weights, indent=4)
    return f'''# {variant_name}: output-level ensemble for BirdCLEF 2026
from pathlib import Path
import json

import numpy as np
import pandas as pd


VARIANT_NAME = {variant_name!r}
BLEND_MODE = {blend_mode!r}
SOURCES = {source_json}
WEIGHTS = {weights_json}

INPUT_ROOT = Path("/kaggle/input")
COMP_DIR = INPUT_ROOT / "competitions" / "birdclef-2026"
sample = pd.read_csv(COMP_DIR / "sample_submission.csv")
labels = sample.columns[1:].tolist()


def find_submission(source):
    expected = INPUT_ROOT / source / "submission.csv"
    if expected.exists():
        return expected
    matches = [p for p in INPUT_ROOT.rglob("submission.csv") if source in str(p)]
    if matches:
        return matches[0]
    all_matches = list(INPUT_ROOT.rglob("submission.csv"))
    raise FileNotFoundError(
        f"Could not find submission.csv for {{source}}. Available: {{all_matches[:20]}}"
    )


def dense_rank_unit(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def rank_frame(df):
    out = np.empty((len(df), len(labels)), dtype=np.float32)
    arr = df[labels].to_numpy(np.float32)
    for j in range(arr.shape[1]):
        out[:, j] = dense_rank_unit(arr[:, j])
    return out


loaded = {{}}
for source in SOURCES:
    path = find_submission(source)
    df = pd.read_csv(path)
    assert list(df.columns) == ["row_id"] + labels, source
    missing = set(sample["row_id"]) - set(df["row_id"])
    extra = set(df["row_id"]) - set(sample["row_id"])
    if missing or extra:
        raise ValueError(
            f"row_id mismatch for {{source}}: missing={{list(missing)[:5]}} extra={{list(extra)[:5]}}"
        )
    df = df.set_index("row_id").loc[sample["row_id"]].reset_index()
    loaded[source] = df
    print(f"Loaded {{source}} from {{path}} shape={{df.shape}}")

weight_sum = float(sum(WEIGHTS.values()))
weights = {{k: float(v) / weight_sum for k, v in WEIGHTS.items()}}
print("Normalized weights:", weights)

if BLEND_MODE == "rank":
    blended = np.zeros((len(sample), len(labels)), dtype=np.float32)
    for source, weight in weights.items():
        blended += weight * rank_frame(loaded[source])
elif BLEND_MODE == "prob":
    blended = np.zeros((len(sample), len(labels)), dtype=np.float32)
    for source, weight in weights.items():
        blended += weight * loaded[source][labels].to_numpy(np.float32)
else:
    raise ValueError(BLEND_MODE)

blended = np.clip(blended, 0.0, 1.0)
submission = pd.DataFrame(blended, columns=labels)
submission.insert(0, "row_id", sample["row_id"].values)
submission.to_csv("submission.csv", index=False)

Path("blend_config.json").write_text(
    json.dumps(
        {{
            "variant": VARIANT_NAME,
            "blend_mode": BLEND_MODE,
            "sources": SOURCES,
            "weights": weights,
            "anchor_scores": {{
                "A0 original public V3 LGBM": 0.928,
                "A24 V3 probe-heavy softpost mirror": 0.928,
                "A25 mtoshidesu 0.932 V4": 0.922,
            }},
        }},
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

print(f"Saved submission.csv {{submission.shape}}")
print(f"mean={{blended.mean():.6f}} std={{blended.std():.6f}} min={{blended.min():.6g}} max={{blended.max():.6f}}")
'''


def write_notebook(path, source):
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    if len(sys.argv) not in (1, 2):
        raise SystemExit("usage: create_submission_blend_notebook.py [output_dir]")

    output_dir = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(
        "references/ensembles"
    )

    specs = [
        {
            "filename": "a26_a0_a24_rank_ensemble.ipynb",
            "variant_name": "A26 A0/A24 rank ensemble",
            "sources": [
                "birdclef-2026-v3-lgbm-original",
                "birdclef-2026-a24-v3-probe-heavy-softpost-mirror",
            ],
            "weights": {
                "birdclef-2026-v3-lgbm-original": 0.60,
                "birdclef-2026-a24-v3-probe-heavy-softpost-mirror": 0.40,
            },
            "blend_mode": "rank",
        }
    ]

    for spec in specs:
        out_path = output_dir / spec["filename"]
        write_notebook(
            out_path,
            notebook_source(
                spec["variant_name"],
                spec["sources"],
                spec["weights"],
                spec["blend_mode"],
            ),
        )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
