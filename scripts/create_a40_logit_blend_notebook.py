#!/usr/bin/env python3
"""Build A40: A33 with the final blend changed from rank-percentile to logit-space
weighted averaging. The mattiaangeli rescue gates are kept and applied in probability
space after sigmoid (so they still gate the same way), but the *base* mix uses logits.

Why logit blend can be different from rank blend:
  - Rank blend collapses extreme confidence — the top-ranked sample gets pct=1 regardless
    of whether the model said 0.95 or 0.99999. Logit blend preserves that magnitude.
  - For the rare/high-confidence calls (which dominate the ranking metric in tail classes),
    logit blend can give them more weight.

The 3-way mix uses the same weights as A33 (PROTO=0.50, SED=0.35, HEAD=0.15) but in
logit space:
  combined_logit = PROTO_W * logit(p_proto) + SED_W * logit(p_sed) + HEAD_W * logit(p_head)
  combined_prob  = sigmoid(combined_logit)
Then mattiaangeli's rescues run on the combined probabilities.
"""
from __future__ import annotations
import json
from pathlib import Path

BASE_NB = Path(
    "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/"
    "a33_konbu_head_mattia_rescue.ipynb"
)
OUT_NB = Path(
    "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/"
    "a40_a33_logit_blend.ipynb"
)


NEW_BLEND_CELL = """\
# Cell 12 (A40) — 3-way LOGIT-SPACE blend (Proto + SED + Head) + mattiaangeli rescue
# A40 differs from A33 only in the base blend math: instead of per-column rank percentile
# we use logit-space weighted averaging. Rescue gates run on the resulting probabilities.

import numpy as np
import pandas as pd

EPS = 1e-5

# ---- mix weights (same as A33) ----
HEAD_W  = 0.15
SED_W   = 0.35
PROTO_W = 1.0 - SED_W - HEAD_W  # 0.50

# ---- mattiaangeli rescue knobs (unchanged from A33) ----
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

# A40 KEY: logit-space weighted average of the three branches
def logit(p):
    return np.log(p / (1.0 - p))

logit_combined = PROTO_W * logit(pa) + SED_W * logit(pb) + HEAD_W * logit(pc)
pred = 1.0 / (1.0 + np.exp(-logit_combined))   # back to probability space
pred = pred.astype(np.float32)

# ---- For rescue gates we still need per-column ranks of pa and pb separately ----
xa = pd.DataFrame(pa).rank(axis=0, pct=True).to_numpy(np.float32)
xb = pd.DataFrame(pb).rank(axis=0, pct=True).to_numpy(np.float32)

# ---- 1) Proto fake-only rescue ----
fake_only = (pa > FAKE_ONLY_THR) & (pb < SED_LOW_THR)
pred = np.where(fake_only, (1.0 - FAKE_ONLY_BLEND) * pred + FAKE_ONLY_BLEND * xa, pred)

# ---- 2) Proto temporal-continuity rescue with fat-tail kernel ----
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
    f"A40 saved submission.csv: {final_sub.shape}"
    f" | LOGIT mix={PROTO_W:.0%} proto + {SED_W:.0%} sed + {HEAD_W:.0%} head"
    f" | rescues fake_only={int(fake_only.sum())}"
    f" proto_cont={int(proto_cont.sum())} sed_only={int(sed_only.sum())}"
)
"""


def main() -> None:
    nb = json.loads(BASE_NB.read_text())
    target_idx = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c.get("source", []))
        if "Cell 12 (A33)" in s and "3-way rank blend" in s:
            target_idx = i
            break
    if target_idx is None:
        raise RuntimeError("A40: could not find A33 final blend cell")
    nb["cells"][target_idx]["source"] = NEW_BLEND_CELL.splitlines(keepends=True)
    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT_NB} ({len(nb['cells'])} cells, replaced cell #{target_idx})")


if __name__ == "__main__":
    main()
