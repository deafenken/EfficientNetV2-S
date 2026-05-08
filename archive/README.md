# archive/

Artifacts from the **blend-era optimization route (A0–A41)** that the project
pivoted away from on 2026-05-08. Kept (not deleted) for historical reference;
none of these are wired into the active training or submission pipeline.

The current direction is documented in `docs/plans/`:

- `plan_a_stable_silver_to_gold.md`
- `plan_b_aggressive_gold_push.md`
- `plan_c_diversification_perch_transformer.md`
- `review_feasibility.md`

## Why the blend route was abandoned

`docs/optimization_roadmap.md` shows 41 submissions (A0 → A41) on top of public
0.92x–0.94x notebooks produced a net change of **+0.001** on the public LB. The
ceiling was set by whichever public notebook we forked, so further weight-twiddling
or threshold-sharpening could not break out. The decision to commit to training
our own SED models on 4×L40 made the variant-generation scripts obsolete.

## Layout

```
archive/
├── legacy_create_scripts/    # 17 create_aXX_*.py notebook generators
├── legacy_blend_pipeline/    # supporting helpers (prepare/submit/lgbm_features)
└── optimize_outdated.md      # stale code-review (predates the SED training plan)
```

If anything in here needs to be revived, copy back to `scripts/` or
`src/birdclef2026/` and re-add the corresponding Makefile target.
