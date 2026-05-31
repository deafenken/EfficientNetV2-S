# docs/ index

Start here. Grouped by what you're trying to do.

### Submitting / versioning
- **[SUBMISSION_WORKFLOW.md](SUBMISSION_WORKFLOW.md)** — end-to-end runbook: train → register → dry-run → submit → record LB.
- **[NAMING.md](NAMING.md)** — decodes every prefix / number / proper noun (a-line, b-line, `e*` configs, datasets, kernels) + the naming convention for new work.
- **[kaggle_submission_pipeline.md](kaggle_submission_pipeline.md)** — the submit pipeline's hard-won bug post-mortem (IPv4, 409 idempotency, legacy CLI, CPU-only, dataset-ready race).
- Source of truth for *what scored what*: **[../submissions/manifest.yaml](../submissions/manifest.yaml)**.

### Competition facts & strategy
- **[competition_notes.md](competition_notes.md)** — task, data, and the offline **CPU-only** submission/runtime constraints.
- **[optimization_roadmap.md](optimization_roadmap.md)** — LB state, medal cuts, and the full a-line (A0–A41) ablation history.
- **[public_baseline_comparison.md](public_baseline_comparison.md)** · **[public_lgbm_baseline.md](public_lgbm_baseline.md)** — public Perch/LGBM baseline analysis.
- **[external_2025_2nd_place_takeaways.md](external_2025_2nd_place_takeaways.md)** · **[external_2025_2nd_place_diff.md](external_2025_2nd_place_diff.md)** — carry-over techniques from BC2025 2nd place.
- **plans/** — [plan_a (stable silver)](plans/plan_a_stable_silver_to_gold.md) · [plan_b (aggressive gold)](plans/plan_b_aggressive_gold_push.md) · [plan_c (diversification)](plans/plan_c_diversification_perch_transformer.md) + reviews.

### Infra / layout
- **[PROJECT_LAYOUT.md](PROJECT_LAYOUT.md)** — repository structure.
- **[remote_access.md](remote_access.md)** — VPS → WSL → target-server → docker chain.
