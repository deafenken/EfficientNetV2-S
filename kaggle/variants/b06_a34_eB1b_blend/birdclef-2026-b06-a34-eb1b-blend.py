# %% [markdown]
# # BirdCLEF 2026 — b06: A34 + eB1b rank-blend
#
# Hybrid ensemble: take A34's strong public LB submission (Konbu Head SED +
# ProtoSSM + Mattia rescue, LB 0.943) and add diversity from our eB1b
# (NFNet-L0 + BG noise + cross-species mixup, LB 0.798).
#
# Both submissions are taken as **already-existing kernel outputs** via
# `kernel_sources` in kernel-metadata.json — no re-inference is done here.
# This kernel only reads two CSVs, rank-normalizes them, weighted-averages,
# and writes /kaggle/working/submission.csv. Total runtime: < 1 minute.
#
# Weight: 0.85 × A34 + 0.15 × eB1b. The strong model carries the score; the
# weak one provides architectural diversity (NFNet-L0 normalizer-free vs
# A34's BN-heavy ProtoSSM+SED ensemble). If eB1b's errors are uncorrelated
# with A34's, the blend lifts past 0.943; if they're correlated, blend drops
# toward weighted average (~0.92). Worst case we still produce a valid
# submission and learn something about diversity.
#
# Fail-soft: if A34's submission.csv is missing (kernel_source not resolved),
# fall back to writing A34-only or zero-stub.

# %% cell 1 — fail-soft stub first (always emit a valid submission.csv)
from pathlib import Path
import pandas as _pd

_KAGGLE_INPUT = Path("/kaggle/input/competitions/birdclef-2026")
if not _KAGGLE_INPUT.exists():
    _KAGGLE_INPUT = Path("/kaggle/input/birdclef-2026")
_SAMPLE_SUB = _KAGGLE_INPUT / "sample_submission.csv"

_OUT = Path("/kaggle/working/submission.csv")
_stub = _pd.read_csv(_SAMPLE_SUB)
for _c in _stub.columns[1:]:
    _stub[_c] = 0.0
_stub.to_csv(_OUT, index=False)
print(f"[stub] wrote zero-fallback submission.csv ({len(_stub)} rows × {len(_stub.columns)} cols)")

# %% cell 2 — locate the two source submissions
# kernel_sources mount kernel outputs at /kaggle/input/<kernel-slug>/. The
# directory contains whatever the source kernel left in /kaggle/working/,
# typically submission.csv.
import glob

A34_GLOBS = [
    "/kaggle/input/birdclef-2026-a34-a33-epoch-upscale/submission.csv",
    "/kaggle/input/notebooks/longkunshicandyman/birdclef-2026-a34-a33-epoch-upscale/submission.csv",
    "/kaggle/input/**/longkunshicandyman/birdclef-2026-a34-a33-epoch-upscale/submission.csv",
]
B03_GLOBS = [
    "/kaggle/input/birdclef-2026-b03-eb1b-nfnet-bgnoise-fold0/submission.csv",
    "/kaggle/input/notebooks/winbeaux/birdclef-2026-b03-eb1b-nfnet-bgnoise-fold0/submission.csv",
    "/kaggle/input/**/winbeaux/birdclef-2026-b03-eb1b-nfnet-bgnoise-fold0/submission.csv",
]


def _first_hit(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


A34_CSV = _first_hit(A34_GLOBS)
B03_CSV = _first_hit(B03_GLOBS)
print(f"[locate] A34 → {A34_CSV}")
print(f"[locate] b03 → {B03_CSV}")

# %% cell 3 — blend or fall back
import sys
import traceback

try:
    if not A34_CSV:
        # No A34 attached. Best we can do is the zero stub OR b03 alone if it
        # is attached. b03 alone is LB 0.798 — still a valid run.
        print("[FATAL] A34 submission.csv not found under /kaggle/input/.")
        print("        Check kernel-metadata.json kernel_sources includes the A34 slug.")
        if B03_CSV:
            print("[fallback] copying b03 (eB1b) submission.csv as the kernel output (LB 0.798)")
            df = _pd.read_csv(B03_CSV)
            df.to_csv(_OUT, index=False)
        sys.exit(0)
    if not B03_CSV:
        print("[FATAL] b03 (eB1b) submission.csv not found under /kaggle/input/.")
        print("        Falling back to A34-only (LB 0.943).")
        df = _pd.read_csv(A34_CSV)
        df.to_csv(_OUT, index=False)
        sys.exit(0)

    import numpy as np

    df_a34 = _pd.read_csv(A34_CSV)
    df_b03 = _pd.read_csv(B03_CSV)
    print(f"[load] A34 shape={df_a34.shape}, b03 shape={df_b03.shape}")

    # Align by row_id (both should follow sample_submission order; be defensive).
    if list(df_a34.columns) != list(df_b03.columns):
        common = [c for c in df_a34.columns if c in df_b03.columns]
        df_a34 = df_a34[common]
        df_b03 = df_b03[common]
        print(f"[align] column intersection: {len(common)} cols")
    df_b03 = df_b03.set_index("row_id").loc[df_a34["row_id"]].reset_index()

    target_cols = [c for c in df_a34.columns if c != "row_id"]

    # Per-column rank-percentile normalization on each. A34 is ALREADY rank
    # percentiles after its 3-way rescue blend, so rank-of-rank is a no-op
    # (monotone). b03 is raw sigmoid; rank-normalize so the two branches mix
    # on the same scale (BirdCLEF metric is rank-based, not abs prob).
    a34 = df_a34[target_cols].rank(axis=0, pct=True).to_numpy(np.float32)
    b03 = df_b03[target_cols].rank(axis=0, pct=True).to_numpy(np.float32)

    W_A34 = 0.85
    W_B03 = 0.15
    blend = (W_A34 * a34 + W_B03 * b03).astype(np.float32)

    out = df_a34.copy()
    out[target_cols] = blend
    out.to_csv(_OUT, index=False)
    print(f"[blend] {W_A34:.2f} × A34 + {W_B03:.2f} × b03 → submission.csv shape={out.shape}")

    # QC: report rough probability spread so we can sanity-check the kernel
    # output even before clicking Submit.
    prob_mat = out.iloc[:, 1:].to_numpy()
    print(f"[qc] blend min/mean/max: {prob_mat.min():.4f} / {prob_mat.mean():.4f} / {prob_mat.max():.4f}")
    print(f"[qc] rows where max prob < 0.05 (cold): {(prob_mat.max(axis=1) < 0.05).sum()}")

except BaseException as _e:
    print(f"[FATAL] blend failed: {type(_e).__name__}: {_e}", flush=True)
    traceback.print_exc()
    print("[FATAL] keeping zero-fallback submission.csv; exiting cleanly", flush=True)
