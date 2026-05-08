# legacy_blend_pipeline/

Support code for the blend-era submission flow.

| File | What it did |
|---|---|
| `lgbm_features.py` | Probe-feature builder for the LGBM postprocessing route (originally `src/birdclef2026/lgbm_features.py`). Designed to consume Perch embeddings + raw scores + site/hour priors and emit features for a stacked LGBM head. |
| `prepare_kaggle_variants.py` | Bulk-rendered the variant notebooks produced by `legacy_create_scripts/` into individual Kaggle kernel directories. |
| `submit_completed_variants.py` | Polled the variant kernels and `kaggle competitions submit` whichever finished `COMPLETE`. |

These are kept for archaeology only. The new flow trains our own SED models
locally and submits via a single inference notebook (see Plan A/B/C in
`docs/plans/`).
