# mtoshidesu 0.932 Test Mod Version 4

Source: https://www.kaggle.com/code/mtoshidesu/0-932-test-mod-version-4/notebook

## Status

Pulled on 2026-04-24 as `A25` candidate.

Kaggle metadata:

- CPU only, no internet, no TPU.
- Competition source: `birdclef-2026`.
- Dataset sources: `jaejohn/perch-meta`, `rishikeshjani/perch-onnx-for-birdclef-2026`.
- Kernel source: `ashok205/tf-wheels`.
- Model source: `google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1`.

The source says it avoids third-party feature caches and builds Perch features from scratch with ONNX. The metadata still includes `jaejohn/perch-meta`, but `EXTERNAL_CACHE_DIRS = []`, so the cache is not used by the code path.

## Transferable Ideas

- ONNX Perch inference path with multithreaded audio loading.
- Genus proxy logits for unmapped species.
- Lighter MLP probe with PCA 64 and vectorized inference.
- LightProtoSSM cross-attention variant.
- Isotonic/F1 threshold optimization, although the submit-mode threshold fitting is not true OOF and may overfit.
- Lighter file confidence scaling and adaptive delta smoothing.

## Risks

- The published 0.932 claim needs direct verification under our account.
- Threshold calibration is fitted on train predictions from full-train models in submit mode, not grouped OOF.
- It rebuilds Perch train cache inside the run, so runtime is higher than A0/A24.
