# Public LGBM Baseline Notes

Source notebook:

- https://www.kaggle.com/code/youssefmo942009/version-3-lgbm
- Captured `scriptVersionId`: `312843604`
- Local archive: `references/public_baselines/youssefmo942009_version_3_lgbm/version_3_lgbm.ipynb`
- Readable extraction: `references/public_baselines/youssefmo942009_version_3_lgbm/version_3_lgbm.py`
- Local plus copy: `references/public_baselines/youssefmo942009_version_3_lgbm/version_3_lgbm_plus.py`
- Kaggle-ready plus notebook: `references/public_baselines/youssefmo942009_version_3_lgbm/version_3_lgbm_plus.ipynb`

## What The Baseline Actually Does

The title says LGBM, but the pipeline is a Perch-based ensemble:

1. Install TensorFlow 2.20 wheels from an attached Kaggle input.
2. Load Google Perch v2 CPU SavedModel:
   `/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1`
3. Load an external Perch cache with:
   - `full_perch_meta.parquet`
   - `full_perch_arrays.npz`
4. Map Perch labels to BirdCLEF target columns using `taxonomy.csv`.
5. Infer Perch logits and 1536-d embeddings for 12 five-second windows per soundscape.
6. Build site/hour prior tables from `train_soundscapes_labels.csv`.
7. Train a ProtoSSM model over full 12-window sequences.
8. Train per-class MLP + LGBM probes on PCA-reduced embeddings plus score/prior features.
9. Train a small ResidualSSM to correct first-pass predictions.
10. Apply temperature scaling, file-level confidence scaling, rank-aware scaling, delta smoothing, and threshold sharpening.

## Required Kaggle Inputs

To reproduce the public notebook on Kaggle, attach at least:

- BirdCLEF+ 2026 competition data.
- Google Perch v2 model input.
- TensorFlow 2.20 wheel input used by the notebook, or internet enabled so the notebook can fall back to PyPI.
- The `perch-meta` dataset containing the cached train soundscape Perch arrays.

Without `perch-meta`, the public notebook cannot train its ProtoSSM/probe stack as written.

## Local Plus Changes

The first local improvement is intentionally conservative:

- The public `build_class_features()` already defines time-of-day and 12-window position features, but calls never pass `hour_utc` or `window_idx`.
- `version_3_lgbm_plus.py` passes `hour_utc`, `site_id`, and `window_idx` into the MLP/LGBM probe feature builder for training, test, and residual-training feature paths.
- It also aligns final hidden-test output to `sample_submission.csv` row order before saving.

The reusable implementation lives in `src/birdclef2026/lgbm_features.py` so we can apply the same feature family to our own embeddings later.

Regenerate the plus notebook from the archived public notebook with:

```bash
python scripts/create_lgbm_plus_notebook.py \
  references/public_baselines/youssefmo942009_version_3_lgbm/version_3_lgbm.ipynb \
  references/public_baselines/youssefmo942009_version_3_lgbm/version_3_lgbm_plus.ipynb
```

Prepare a Kaggle kernel directory with:

```bash
KAGGLE_USERNAME=your_name make prepare-kaggle-kernel
```

Then push it after configuring `.kaggle/kaggle.json`:

```bash
make push-kaggle-kernel
make kernel-status
```

## Next Improvements

1. Make a fresh Perch cache from train soundscapes instead of depending on a public cache.
2. Add OOF diagnostics for each component: Perch-only, prior-fused, probe, ProtoSSM, residual.
3. Tune probe blending per class instead of fixed `alpha=0.50`.
4. Try class-family temperature scaling and per-taxon post-processing using OOF AUC/logloss.
5. Ensemble this Perch stack with our log-mel CNN stack after both have sound local CV.
