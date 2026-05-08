# legacy_create_scripts/

Notebook-variant generators from the blend-era. Each script reads a public
baseline notebook from `references/public_baselines/` and emits a tweaked
copy under `kaggle/` that adjusts blend weights, thresholds, rescue gates,
or postprocessing knobs.

| Script | Variant target |
|---|---|
| `create_a23_diagnostic_notebook.py` | A23 V3 diagnostic gate |
| `create_a24_probe_heavy_softpost_notebook.py` | A24 V3 probe-heavy softpost |
| `create_a33_merged_blend_notebook.py` | A33 konbu head + mattia rescue |
| `create_a34_epoch_upscale_notebook.py` | A34 epoch upscale |
| `create_a36_ensemble_notebook.py` | A36 nina ensemble fork |
| `create_a37_weight_tilt_notebook.py` | A37 blend weight tilt |
| `create_a38_multihead_ssm_notebook.py` | A38 multi-head SSM |
| `create_a39_threshold_sharpen_notebook.py` | A39 threshold sharpen |
| `create_a40_logit_blend_notebook.py` | A40 logit blend |
| `create_a41_amplified_rescue_notebook.py` | A41 amplified rescue |
| `create_lgbm_ablation_notebooks.py` | LGBM ablation grid |
| `create_lgbm_plus_notebook.py` | V3 LGBM plus |
| `create_pacerq_v51_ablation_notebooks.py` | pacerq V51 ablations |
| `create_pantanal_optimized_notebook.py` | pantanal ONNX |
| `create_submission_blend_notebook.py` | submission blend |
| `create_two_pass_ssm_variants.py` | two-pass SSM variants |
| `create_v3_score_ablation_notebooks.py` | V3 score ablations |

These were the engine behind the A0–A41 submission stream that ended at
LB 0.929. See `docs/optimization_roadmap.md` and `docs/plans/README.md` for
the post-mortem.
