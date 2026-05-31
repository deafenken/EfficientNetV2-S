# Submission Workflow — make a new BirdCLEF-2026 submission

End-to-end runbook for the **b-line** (our own trained models). Everything that
touches a checkpoint or the network runs **on the L40 box** (where `outputs/`
lives); the VPS only writes code + `git push`. Naming/term reference:
[`NAMING.md`](NAMING.md). Source of truth for versions/scores:
[`../submissions/manifest.yaml`](../submissions/manifest.yaml). Historical
bug post-mortem: [`kaggle_submission_pipeline.md`](kaggle_submission_pipeline.md).

---

## 0. One-time: Kaggle auth (split-stack)

Two non-interchangeable credentials, both under `~/.kaggle/` (never in the repo):

| File | Used by | Why |
|---|---|---|
| `~/.kaggle/access_token` (PAT) | `kagglehub.dataset_upload` | dataset upload |
| `~/.kaggle/kaggle.json` (`{"username","key"}`) | `kaggle kernels push` | PAT lacks the kernels scope (→ 403/409) |

`chmod 600 ~/.kaggle/*`. Or set `$KAGGLE_API_TOKEN` / `$KAGGLE_USERNAME` +
`$KAGGLE_KEY` (see [`.env.example`](../.env.example)). Resolver:
[`scripts/kaggle/auth.py`](../scripts/kaggle/auth.py).

---

## 1. Train (L40) — produces the checkpoint

```bash
make train-ddp EXP=configs/exp/<exp>.yaml FOLD=0
make assemble-oof EXP_DIR=outputs/exp/<exp> KIND=swa   # writes oof_auc + swa_oof.npz
# → outputs/exp/<exp>/fold_0/swa.pt
```
> Heads-up: `oof_auc.txt` is computed on **focal train clips** and badly
> over-states soundscape LB (we've seen 0.98 OOF → 0.798 LB). Treat it as a
> rough relative signal, not a LB predictor, until a soundscape-domain
> validation exists.

## 2. Register the variant in the manifest

Add a row under `b_variants` in [`submissions/manifest.yaml`](../submissions/manifest.yaml):

```yaml
  - id: b10_<config>_<technique>
    exp_config: configs/exp/<exp>.yaml
    fold: 0
    checkpoint: outputs/exp/<exp>/fold_0/swa.pt
    dataset_handle: winbeaux/birdclef2026-<exp>-pkg-ckpt
    dataset_version: null
    kernel_id: winbeaux/birdclef-2026-b10-<...>
    kernel_version: null
    notes_label: "<exp> fold-0 SWA + package"
    lb_public: null
    notes: "<what changed>"
```

Create the kernel dir `kaggle/variants/b10_.../` with `kernel-metadata.json`
(`enable_gpu:false`, `enable_internet:false`, `competition_sources:["birdclef-2026"]`,
`dataset_sources:["winbeaux/birdclef2026-<exp>-pkg-ckpt"]`) and one jupytext
`b10_....py`, plus `kaggle/datasets/birdclef2026-<exp>-pkg-ckpt/dataset-metadata.json`.
(Easiest: copy the closest existing b-variant and edit.) See the naming rules in
[`NAMING.md §7`](NAMING.md#7-convention-for-naming-new-things).

## 3. Verify staging WITHOUT spending a submission

```bash
make audit-submissions                                   # manifest ↔ disk consistency
uv run python scripts/kaggle/submit.py b10_... --dry-run  # stage + convert, no network
```

## 4. Submit (uploads dataset + pushes kernel)

```bash
uv run python scripts/kaggle/submit.py b10_...
# blend-only / re-push kernel without re-uploading the dataset:
uv run python scripts/kaggle/submit.py b10_... --kernel-only
```
Steps: stage payload → `kagglehub` dataset upload (IPv4-only) → poll until the
dataset is "complete" → convert `.py`→`.ipynb` → `kaggle kernels push`.

## 5. Final manual step (Kaggle requires it)

Open the kernel URL, wait for the run to finish, click **"Submit to
Competition"**. After the LB shows, record it in the manifest row:
`kernel_version`, `dataset_version`, `lb_public` (and `lb_private` post-deadline).
The manifest is human-diffable on purpose — just edit + commit:

```bash
git add submissions/manifest.yaml && git commit -m "chore(submit): record b10 LB=0.xxx" && git push origin winbeau
```

---

## Legacy / compatibility

- `scripts/22_kaggle_submit.py --variant <id>` still works (thin shim → the package).
- `scripts/17_…` / `scripts/18_…` are deprecated shims (forward to `b01` / `b02`).
- Canonical entry point going forward: **`scripts/kaggle/submit.py`**.

## Before retiring the old scripts (verification owed)

The consolidation is behavior-preserving but has not yet been confirmed against a
live push. Once you next submit, do **one** `--dry-run` then a real
`--kernel-only` re-push of a known variant (e.g. `b03_eB1b_nfnet_bgnoise`) and
confirm the pushed kernel is identical to before. After that, the old 17/18/22
shims can be deleted.
