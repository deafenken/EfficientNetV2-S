# Needless090 Iter Pseudo Perch SED Notes

Source notebook:

- `https://www.kaggle.com/code/needless090/birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s`
- Archived notebook: `references/public_baselines/needless090_iter_pseudo_perch_sed/birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s.ipynb`
- Readable extraction: `references/public_baselines/needless090_iter_pseudo_perch_sed/birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s.py`

## What It Actually Changes

The notebook is not a brand-new pipeline. Most of the body is still the familiar V18-style Perch + ProtoSSM + probe + residual stack. The main extra pieces are:

1. CPU Perch inference fallbacks:
   - tries ONNX Perch first,
   - falls back to TFLite Perch,
   - falls back again to the original TensorFlow SavedModel path.
2. A separate SED ensemble head:
   - EfficientNet B0/B3 RGB mel models,
   - five extra mono-mel SED models (`v5_focal`, `ce_s123`, `ce_s456`, `v5_pseudo`, `v5_pseudo2`).
3. Rank averaging:
   - final Perch/ProtoSSM probabilities are rank-averaged with SED probabilities using `70/30`.
4. Sonotype mirroring:
   - a few sonotype labels are grouped and replaced by the group max probability.

## Dependency Reality

The advertised `0.934` path depends on extra SED weight datasets that do not appear to be publicly mountable right now:

- `needless090/birdclef2026-sed-ensemble`
- `needless090/birdclef2026-sed-v5-trio`

The only clearly public extra dataset we could confirm from the code path is:

- `needless090/birdclef2026-perch-tflite`

Public inputs already known in this repo still cover the ONNX Perch side:

- `rishikeshjani/perch-onnx-for-birdclef-2026`
- `i2nfinit3y/onnxruntime`

## Transferable Ideas

These parts are worth carrying into our own route:

1. Fast CPU Perch inference via ONNX/TFLite.
2. Sonotype mirroring groups:
   - `47158son15 <-> 47158son16`
   - `47158son09 <-> 47158son12`
   - `47158son02 <-> 47158son14`
   - `47158son13 / 47158son21 / 47158son22 / 47158son23`
3. Rank blending between heterogeneous heads, but only if we have our own second modality.

## Not Transferable As-Is

Do not package this notebook unchanged as the next leaderboard attempt unless the SED weights become publicly accessible. Without those SED datasets, the main score-carrying difference from A0 is missing.

## Recommendation

- Use this notebook as a reference for:
  - ONNX/TFLite CPU inference,
  - sonotype mirroring,
  - future rank blending with our own SED branch.
- Do not treat it as a direct reproduction candidate right now.
