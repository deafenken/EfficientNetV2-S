#!/usr/bin/env python3
"""Build A33: wliilamsam 0.943 base + konbu17 train_audio head + mattiaangeli rescue rules.

Inputs:
  - references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/birdclef-2026-0-943-onnx-perch-sequence-sed.ipynb

Output:
  - references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/a33_konbu_head_mattia_rescue.ipynb

The base notebook's last cell is the 2-way rank blend (Cell 12). A33 replaces it with:
  1. konbu17 INJECT cell that consumes the still-alive emb_te to write submission_head.csv.
  2. A 3-way blend cell: mattiaangeli rescue gates (Proto vs SED) + HEAD branch rank-blended in.
"""
from __future__ import annotations

import json
from pathlib import Path


BASE_NB = Path(
    "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/"
    "birdclef-2026-0-943-onnx-perch-sequence-sed.ipynb"
)
OUT_NB = Path(
    "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/"
    "a33_konbu_head_mattia_rescue.ipynb"
)


HEAD_INJECT_SOURCE = """\
# Cell INJECT (A33) — apply konbu17 train_audio head -> submission_head.csv
# Consumes emb_te which is still alive in this kernel scope from the Perch inference cell.
import glob

import numpy as np
import pandas as pd

print("[A33 head inject] applying train_audio head")
candidates = sorted(glob.glob("/kaggle/input/**/head_weights_train_audio.npz", recursive=True))
print(f"  candidates: {candidates}")
HEAD_PATH = candidates[0]
hw = np.load(HEAD_PATH, allow_pickle=True)
W = hw["W"].astype(np.float32)
b = hw["b"].astype(np.float32)
trained_mask = hw["trained_mask"].astype(bool)
print(f"  W={W.shape}, b={b.shape}, trained={int(trained_mask.sum())}/234")

assert emb_te.shape[1] == W.shape[1], (
    f"feat dim mismatch: emb_te={emb_te.shape[1]} W={W.shape[1]}"
)

head_logits = emb_te.astype(np.float32) @ W.T + b
head_logits = head_logits * trained_mask.reshape(1, -1).astype(np.float32)
head_probs = 1.0 / (1.0 + np.exp(-np.clip(head_logits, -30, 30)))

head_sub = pd.DataFrame(head_probs.astype(np.float32), columns=PRIMARY_LABELS)
head_sub.insert(0, "row_id", meta_te["row_id"].values)
head_sub.to_csv("submission_head.csv", index=False)
print(f"  saved submission_head.csv: {head_sub.shape}")
print(f"  head_logits range: [{head_logits.min():.2f}, {head_logits.max():.2f}]")
"""


THREE_WAY_BLEND_SOURCE = """\
# Cell 12 (A33) — 3-way rank blend (Proto + SED + Head) with mattiaangeli rescue gates
# Base mix: PROTO + SED + HEAD via per-class rank percentile.
# Rescues are gated on Proto vs SED only (Head is not yet calibrated against site/hour priors).
import numpy as np
import pandas as pd

EPS = 1e-5

# ---- mix weights (head added at konbu17's recommended 0.15) ----
HEAD_W  = 0.15
SED_W   = 0.35
PROTO_W = 1.0 - SED_W - HEAD_W  # 0.50

# ---- mattiaangeli rescue knobs ----
FAKE_ONLY_THR   = 0.50
SED_LOW_THR     = 0.05
FAKE_ONLY_BLEND = 0.12

PROTO_CONT_RADIUS    = 3
PROTO_CONT_DF        = 2.0
PROTO_CONT_SCALE     = 1.20
PROTO_CONT_RANK_THR  = 0.88
PROTO_LOCAL_RANK_THR = 0.75
SED_CONT_LOW_THR     = 0.12
PROTO_CONT_BLEND     = 0.15

SED_ONLY_RANK_THR = 0.95
FAKE_RANK_LOW_THR = 0.80
SED_ONLY_BLEND    = 0.12

# ---- load three branches ----
proto_df = pd.read_csv("submission_protossm.csv")
sed_df   = pd.read_csv("submission_sed.csv")
head_df  = pd.read_csv("submission_head.csv")

cols = [c for c in proto_df.columns if c != "row_id"]

sed_df  = sed_df.set_index("row_id").loc[proto_df["row_id"]].reset_index()
head_df = head_df.set_index("row_id").loc[proto_df["row_id"]].reset_index()

pa = np.clip(proto_df[cols].to_numpy(np.float32), EPS, 1.0 - EPS)
pb = np.clip(sed_df[cols].to_numpy(np.float32),   EPS, 1.0 - EPS)
pc = np.clip(head_df[cols].to_numpy(np.float32),  EPS, 1.0 - EPS)

xa = pd.DataFrame(pa).rank(axis=0, pct=True).to_numpy(np.float32)
xb = pd.DataFrame(pb).rank(axis=0, pct=True).to_numpy(np.float32)
xc = pd.DataFrame(pc).rank(axis=0, pct=True).to_numpy(np.float32)

# ---- 3-way base blend ----
pred = PROTO_W * xa + SED_W * xb + HEAD_W * xc

# ---- 1) Proto fake-only rescue (Proto confident, SED missed) ----
fake_only = (pa > FAKE_ONLY_THR) & (pb < SED_LOW_THR)
pred = np.where(fake_only, (1.0 - FAKE_ONLY_BLEND) * pred + FAKE_ONLY_BLEND * xa, pred)

# ---- 2) Proto temporal-continuity rescue with fat-tail (Student-t df=2) kernel ----
row_ids  = proto_df["row_id"].astype(str).to_numpy()
file_ids = np.array(["_".join(r.split("_")[:-1]) for r in row_ids])

offs = np.arange(-PROTO_CONT_RADIUS, PROTO_CONT_RADIUS + 1, dtype=np.float32)
proto_kernel = (1.0 + (offs / PROTO_CONT_SCALE) ** 2 / PROTO_CONT_DF) ** (
    -(PROTO_CONT_DF + 1.0) / 2.0
)
proto_kernel = (proto_kernel / proto_kernel.sum()).astype(np.float32)

pa_ctx = pa.copy()
R = PROTO_CONT_RADIUS
for fid in pd.unique(file_ids):
    m = file_ids == fid
    x = pa[m]
    if len(x) > 1:
        xp = np.pad(x, ((R, R), (0, 0)), mode="edge")
        pa_ctx[m] = sum(proto_kernel[i] * xp[i:i + len(x)] for i in range(2 * R + 1))

xctx = pd.DataFrame(pa_ctx).rank(axis=0, pct=True).to_numpy(np.float32)

proto_cont = (
    (xctx > PROTO_CONT_RANK_THR) &
    (xa   > PROTO_LOCAL_RANK_THR) &
    (pb   < SED_CONT_LOW_THR) &
    (~fake_only)
)
pred = np.where(
    proto_cont,
    (1.0 - PROTO_CONT_BLEND) * pred + PROTO_CONT_BLEND * np.maximum(xa, xctx),
    pred,
)

# ---- 3) Rare local SED-spike rescue ----
sed_only = (
    (xb > SED_ONLY_RANK_THR) &
    (xa < FAKE_RANK_LOW_THR) &
    (~fake_only) &
    (~proto_cont)
)
pred = np.where(sed_only, (1.0 - SED_ONLY_BLEND) * pred + SED_ONLY_BLEND * xb, pred)

# ---- write final submission ----
final_sub = proto_df.copy()
final_sub[cols] = pred.astype(np.float32)
final_sub.to_csv("submission.csv", index=False)

print(
    f"A33 saved submission.csv: {final_sub.shape}"
    f" | mix={PROTO_W:.0%} proto + {SED_W:.0%} sed + {HEAD_W:.0%} head"
    f" | rescues fake_only={int(fake_only.sum())}"
    f" proto_cont={int(proto_cont.sum())} sed_only={int(sed_only.sum())}"
)
"""


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def main() -> None:
    if not BASE_NB.exists():
        raise FileNotFoundError(BASE_NB)

    nb = json.loads(BASE_NB.read_text())
    cells = nb["cells"]

    # Sanity-check: the last code cell of the base is the 2-way Cell 12 blend.
    last_src = "".join(cells[-1].get("source", []))
    assert "PROTO_W = 0.6" in last_src and "SED_W   = 0.4" in last_src, (
        "wliilamsam base last cell does not look like the expected Cell 12 blend; "
        "refusing to silently overwrite. First 200 chars:\n" + last_src[:200]
    )

    # Drop the original Cell 12.
    cells = cells[:-1]

    cells.append(markdown_cell(
        "## A33 — konbu17 train-audio head + mattiaangeli rescue gates\n\n"
        "Replaces the original 2-way ProtoSSM/SED rank blend with a 3-way rank blend "
        "(Proto / SED / Head, mix 0.50 / 0.35 / 0.15) and adds three rescue gates from "
        "mattiaangeli's better-blend (fake-only, Proto temporal continuity with fat-tail "
        "Student-t kernel, rare SED-spike). Head branch must be attached as a dataset "
        "containing `head_weights_train_audio.npz` (e.g. `konbu17/bird26-train-audio-head-v1`).\n"
    ))
    cells.append(code_cell(HEAD_INJECT_SOURCE))
    cells.append(code_cell(THREE_WAY_BLEND_SOURCE))

    nb["cells"] = cells

    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT_NB} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
