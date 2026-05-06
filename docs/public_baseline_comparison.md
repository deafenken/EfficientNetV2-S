# Public Baseline Comparison

## Archived Notebooks

| Notebook | Local path | Page metadata date | Role |
| --- | --- | --- | --- |
| `youssefmo942009/version-3-lgbm` | `references/public_baselines/youssefmo942009_version_3_lgbm/` | 2026-04-19 | Later, heavier ensemble with ProtoSSM, MLP+LGBM probes, ResidualSSM, and extra post-processing. |
| `fvegamaza/birdclef-2026-perch-protossm` | `references/public_baselines/fvegamaza_perch_protossm/` | 2026-03-24 | Earlier, cleaner Perch + ProtoSSM + MLP probe pipeline; useful as an ablation anchor. |

## Main Differences

| Component | Perch + ProtoSSM | Version 3 LGBM |
| --- | --- | --- |
| Perch v2 inference | Yes | Yes |
| Train soundscape cache | Optional compute/cache flow | Required `perch-meta` cache |
| Site/hour priors | Yes | Yes |
| Probe model | MLP / LogReg fallback | MLP + LGBM ensemble |
| Temporal model | ProtoSSM | Larger ProtoSSM v2 with cross-attention |
| Second-pass correction | No | ResidualSSM |
| Post-processing | Temperature scaling | Temperature, file confidence, rank scaling, adaptive smoothing, threshold sharpening |

## Working Plan

1. Use `version_3_lgbm_plus.ipynb` for the first Kaggle reproduction because it contains the strongest public stack.
2. Keep `fvegamaza_perch_protossm` as a simpler ablation baseline when a new component looks suspicious.
3. Record public/private score, notebook version, attached inputs, and config deltas after every Kaggle run.
4. Move reusable pieces into `src/birdclef2026/` only after they prove useful in at least one run.

## First Ablations

| Run | Change | Why |
| --- | --- | --- |
| A0 | Original `version_3_lgbm` | Anchor public baseline. |
| A1 | `version_3_lgbm_plus` | Tests time/window/site features and sample-order alignment. |
| A2 | A1 without ResidualSSM | Residual target is noisy; verify it really helps. |
| A3 | A1 with lower `ENSEMBLE_WEIGHT_PROTO` | Probe stack may be more calibrated for some non-bird taxa. |
| A4 | A1 with no threshold sharpening | Ranking metrics can dislike aggressive probability reshaping. |

Generate the A2-A4 notebooks with:

```bash
make ablations
```

The notebooks are written under `references/public_baselines/youssefmo942009_version_3_lgbm/ablations/`.

Package A0-A4 as Kaggle kernels with:

```bash
make prepare-kaggle-variants
```
