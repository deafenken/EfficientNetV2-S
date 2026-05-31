# Naming Conventions — BirdCLEF-2026

> One page that decodes **every prefix, letter, number, and proper noun** used
> across this repo's experiments, submission variants, Kaggle datasets, and
> kernels. If you see a name you don't recognize, it is defined here.
>
> Source of truth for *what scored what*: [`submissions/manifest.yaml`](../submissions/manifest.yaml).
> Full a-line ablation history: [`docs/optimization_roadmap.md`](optimization_roadmap.md).

---

## 0. Quick decoder

| You see… | It is a… | Lives in | Example |
|---|---|---|---|
| `e01`, `eB1b`, `eC2` | **experiment / training config** | `configs/exp/<id>.yaml` | `eB1b_nfnet_bgnoise` |
| `a0`..`a41` | **a-line submission variant** (public-notebook clone/blend) | `kaggle/variants/a*/` | `a29_nina_ensemble_solutions` |
| `b01`..`b09` | **b-line submission variant** (our own trained model) | `kaggle/variants/b*/` | `b03_eB1b_nfnet_bgnoise` |
| `birdclef2026-<exp>-pkg-ckpt` | **Kaggle dataset** (our packaged code + checkpoint) | `kaggle/datasets/<slug>/` | `birdclef2026-eb1b-pkg-ckpt` |
| `<owner>/birdclef-2026-…` | **Kaggle kernel** (the runnable notebook) | Kaggle | `winbeaux/birdclef-2026-b03-…` |

**Two letter-namespaces, do not confuse them:**
- **`a` / `b`** prefix a **submission variant** (something pushed to Kaggle).
- **`e`** prefixes a **training experiment config** (something trained on the L40).
  A `b`-variant usually wraps one `e`-config (e.g. `b03` ships the `eB1b` checkpoint).

---

## 1. Experiment config ids (`configs/exp/<id>.yaml`)

Format: **`e<FAMILY><n><rev>_<short_name>`**

- **`e`** — "experiment" (a training run config; deep-merged over `configs/data.yaml`).
- **`<FAMILY>`** — the axis being varied:
  - *(none)* → the **EfficientNetV2-S baseline line** (`e01`–`e04`).
  - **`B`** → **Backbone** swap (try a different architecture).
  - **`C`** → **Config/head** ablation (loss / head / hyper-param tweak on a fixed backbone).
- **`<n>`** — sequential number within the family.
- **`<rev>`** — optional lowercase revision letter (`b`, `c`, …): same base, incremental change.
- **`_<short_name>`** — human tag describing the key change.

| Config id | Backbone | What it is |
|---|---|---|
| `e01_v2s_5fold` | EfficientNetV2-S | baseline, 5-fold scaffold |
| `e02_v2s_5fold_v2` | EfficientNetV2-S | + ESC-50 background-noise aug |
| `e03_v2s_fixed` | EfficientNetV2-S | + BC2025 warm-start + validation-protocol fix |
| `e04_v2s_pseudo` | EfficientNetV2-S | + pseudo-labels |
| `eB1_nfnet_l0` | eca_nfnet_l0 (NFNet) | NFNet, ImageNet warm-start, **all domain adapters OFF** → LB 0.75 |
| `eB1b_nfnet_bgnoise` | eca_nfnet_l0 | `eB1` **rev b**: + ESC-50 BG noise + forced cross-species MixUp → **LB 0.798** (best b-line) |
| `eB1c_nfnet_pseudo` | eca_nfnet_l0 | `eB1` **rev c**: + R1 pseudo-labels → LB 0.793 (regression, discarded) |
| `eB1b_R2_clean` | eca_nfnet_l0 | clean **R2** retrain over `eB1b`: target-domain (soundscape) BG + RandomAmplitude + LS 0.02 |
| `eB2_convnextv2_tiny` | ConvNeXtV2-tiny | backbone diversity candidate |
| `eB3_efficientnetv2_b2` | EfficientNetV2-b2 | larger EffNetV2 backbone |
| `eC1_head_lr_mul1` | (e-line) | head LR multiplier = 1× (vs 3× backbone) |
| `eC2_pos_weight2` | (e-line) | BCE `pos_weight = 2` |
| `eC3_head_dropout02` | (e-line) | head dropout = 0.2 |

> `R1`/`R2` = **pseudo-label round** 1 / 2 (Noisy-Student iteration), not a config family.

---

## 2. Submission variant ids (`kaggle/variants/<id>/`)

There are **two lines** with different purposes, owners, and file formats.

### 2a. `a`-line — public-notebook clones & blends

Format: **`a<N>_<slug>`** — `N` is sequential; `<slug>` names the source author
and/or technique. These are reproductions/derivations of public Kaggle
notebooks, pushed as raw **`.ipynb`** under the **`longkunshicandyman/`** Kaggle
account via the legacy `kaggle` CLI. They are **frozen history** — not pushed by
the new submit tool.

Purpose: probe the public field. **Conclusion: plateaued at 0.928–0.929; public
clones do not transfer upward for us.** Highlights (full table in
`optimization_roadmap.md`):

| Variant | Source / technique | LB |
|---|---|---|
| `a0_v3_lgbm_original` | youssefmo942009 **V3 LGBM** (Perch emb + ProtoSSM + LGBM probe) | **0.928** (anchor) |
| `a1`–`a4` | V3 LGBM-plus tweaks | 0.924–0.925 |
| `a5`/`a6` | pacerq **V51 ONNX cache** | 0.921 |
| `a7`–`a15` | V3 proto / residual / threshold ablations | ~0.928 (ties) |
| `a16`/`a17` | marynaborovska **two-pass SSM** advanced PP / test-TTA | 0.927 / 0.926 |
| `a18` | koushikrudra **winner-position** 0928 | 0.928 |
| `a19` | yuriygreben **improved ensemble** | 0.923 |
| `a20` | lingyu07 **Perch + YAMNet** fast blend | 0.928 |
| `a21`/`a22` | dingjiarun **Pantanal ONNX** (+optimized) | — |
| `a23`/`a24` | V3 diagnostic-gate / probe-heavy softpost mirror | 0.928 |
| `a25` | mtoshidesu 0.932 V4 (claimed 0.932) | 0.922 |
| `a26` | a0+a24 **rank ensemble** | — |
| `a27` | Tucker **distilled SED** | 0.912 |
| `a28` | ttahara **HGNetV2-b0** inference | — |
| `a29` | nina **ensemble of solutions** | **0.929** (a-line best) |
| `a30` | wliilamsam **0.943 base** (Perch+ProtoSSM+SED) — verbatim repro | — |
| `a31` | konbu17 **train-audio head** | — |
| `a32` | mattiaangeli **better blend** 0.943 | — |
| `a33` | konbu head + mattia **rescue** | — |
| `a34` | a33 **epoch upscale** | — |
| `a35` | udaysonawane **V181** | — |
| `a36`–`a41` | a33/a35 blends, tilts, multihead-SSM, threshold/logit/amplified tweaks | — |

### 2b. `b`-line — our own trained models

Format: **`b<NN>_<config>_<technique>`** — `NN` is zero-padded sequential;
`<config>` is the `e`-config that produced the checkpoint; `<technique>` is the
headline idea. Source kept as **jupytext `.py`** in git, converted to `.ipynb`
at push time. Pushed under the **`winbeaux/`** Kaggle account by
`scripts/kaggle/submit.py`. **This is the line the submit tooling manages.**

| Variant | Wraps config | What | LB |
|---|---|---|---|
| `b01_eB1_nfnet_l0` | `eB1_nfnet_l0` | NFNet baseline (adapters off) | 0.75 |
| `b02_e01_v2s_baseline` | `e01_v2s_5fold` | V2-S baseline (LB control) | — |
| `b03_eB1b_nfnet_bgnoise` | `eB1b_nfnet_bgnoise` | NFNet + BG noise + cross-species MixUp | **0.798** (b-line best) |
| `b04_eB1c_nfnet_pseudo` | `eB1c_nfnet_pseudo` | + R1 pseudo-labels (failed round) | 0.793 |
| `b05_e03_v2s_fixed` | `e03_v2s_fixed` | V2-S warm-start + val fix | — |
| `b06_a34_eB1b_blend` | *(none — blend)* | rank-blend `a34` + `b03` (uses `kernel_sources`) | — |
| `b07` | — | **renamed to `b08`** (intentional gap, not missing) | — |
| `b08_deep_ensemble` | `eB1b_nfnet_bgnoise` | 4-way **RRF** deep ensemble (A34 + NFNet) | — |
| `b09_eB1b_R2_clean` | `eB1b_R2_clean` | clean retrain: target-domain BG + RandomAmp + LS 0.02 + ±2.5s TTA + chunk smooth | *(not trained yet)* |

---

## 3. Kaggle dataset & kernel naming

- **Dataset slug:** `birdclef2026-<exp>-pkg-ckpt` under owner `winbeaux`.
  - `pkg` = the **packaged** `src/birdclef2026/` + `configs/data.yaml`.
  - `ckpt` = the SWA **checkpoint** (`swa.pt`) for that experiment's fold.
  - e.g. `winbeaux/birdclef2026-eb1b-pkg-ckpt`. (`ebr2` = the `eB1b_R2_clean` slug.)
- **Kernel id:** `<owner>/birdclef-2026-<variant>-<desc>[-fold0]`
  - a-line owner = `longkunshicandyman`; b-line owner = `winbeaux`.
  - `kernel-metadata.json` always has `enable_internet: false`, `enable_gpu: false`
    (offline CPU code competition) and `competition_sources: ["birdclef-2026"]`.

---

## 4. Numbering rules & gaps

- Numbers are **sequential and never reused** within a line/family.
- A **gap is allowed only with a documented reason.** Current known gap:
  - **`b07` → `b08`**: b07 ("c0 deep ensemble") was renamed to b08 (commit `dc9d398`).
- `make audit-submissions` flags any *undocumented* gap (whitelist in `scripts/kaggle/audit.py`).

---

## 5. Kaggle accounts / owners

| Owner | Used for |
|---|---|
| `longkunshicandyman` | a-line kernels (public-notebook clones) |
| `winbeaux` | b-line datasets **and** kernels (our own models) |
| team display name | **Longkun Shi (Candyman)** (LB team) |

---

## 6. Glossary of proper nouns

### Backbones (CNNs we train)
- **EfficientNetV2-S / -b2** (`tf_efficientnetv2_s`, `efficientnetv2_b2`) — timm CNN backbones; "V2-S" = the small variant.
- **eca_nfnet_l0 (NFNet)** — Normalizer-Free Net with ECA attention; the b-line's main backbone.
- **ConvNeXtV2-tiny** — modern CNN backbone (diversity candidate).
- **ResNet18** — original throwaway baseline backbone.
- **HGNetV2-b0** — PaddlePaddle/timm backbone used by the ttahara public notebook (a28).

### Pretrained / embedding models (frozen, public)
- **Perch / Perch v2** — Google's bird-vocalization embedding model (1536-d), the core of the public 0.943 line. Used frozen via ONNX.
- **BirdNET / SurfPerch** — other public bird-audio embedding models.
- **YAMNet** — Google general audio-event embedding model (used in a20 blend).
- **perch-meta** — public Kaggle dataset of precomputed Perch embeddings/metadata (`jaejohn/perch-meta`).

### Public-solution names (where an a-variant came from)
- **V3 LGBM** — youssefmo942009's "version 3" notebook: Perch emb → ProtoSSM + MLP/LGBM probe. Our `a0` anchor (0.928).
- **ProtoSSM** — *Prototype State-Space Model*: a light sequence head over per-window Perch embeddings.
- **ResidualSSM** — a residual variant of the SSM head.
- **SED** — *Sound Event Detection*: an attention/segmentation head producing framewise then clip-level probs.
- **Pantanal** — the Brazilian wetland region; BirdCLEF-2026 test = Pantanal passive soundscapes. "Pantanal-*" notebooks = models trained/tuned for that domain.
- **winner-position / two-pass / better-blend / rescue / V51 / V181** — labels of specific public notebooks (see the a-line table for author).
- **Nina ensemble** — nina2025's "ensemble of solutions" notebook (our a29, 0.929).

### Training / loss terms
- **fold / 5-fold** — cross-validation split (StratifiedGroupKFold by recording). Most b-experiments are **single-fold (fold_0)** so far.
- **SWA** — Stochastic Weight Averaging (average of last-N epoch weights → `swa.pt`).
- **EMA** — Exponential Moving Average of weights during training.
- **FocalBCE** — focal loss on top of BCE (`alpha`, `gamma`); multi-label.
- **pos_weight** — positive-class weight in BCE (counters trivial all-zero collapse).
- **LS (label smoothing)** — softens one-hot targets; `ls_mode: positive` smooths only positives.
- **MixUp (cross-species)** — blend two clips + their labels; `cross_species_p=1.0` forces distinct species to teach overlap.
- **RandomAmplitude** — log-uniform gain scaling (models near/far SNR).
- **background noise / BG / ESC-50** — overlay ambient noise. **ESC-50** is a public environmental-sound dataset; "target-domain BG" = overlay the competition's own `train_soundscapes`.
- **warm-start** — initialize the backbone from public BC2025 2nd-place weights.

### Inference / post-processing terms
- **OOF (out-of-fold)** — validation predictions on held-out folds; `oof_auc.txt` is the macro ROC-AUC. ⚠ computed on **focal train clips**, so it over-states soundscape LB (see the 0.98 OOF → 0.798 LB gap).
- **TTA (test-time augmentation)** — average predictions over time-shifted crops (e.g. `±2.5 s`).
- **chunk smoothing** — 1-D conv over a file's 12 windows to smooth per-window probs.
- **RRF (Reciprocal Rank Fusion)** — ensembling by averaging `1/rank` instead of raw probs (rank 1 = top).
- **rank-blend / logit-blend / tilt** — ensemble mixes by rank, by logit, or by weighted ratio.
- **pseudo-label (PL) / Noisy Student** — a teacher labels unlabeled soundscapes; the student trains on them. `R1`/`R2` = round 1/2.
- **TopN / winner-position** — file-level post-processing: scale each window by the file's top class probs.

### Infra terms
- **pkg-ckpt** — see §3 (packaged code + checkpoint dataset).
- **fail-soft** — the submission notebook always emits a valid (zero) `submission.csv` even if inference crashes, so the run never errors out.
- **code competition / offline** — submissions run as Kaggle notebooks with **no internet** and (here) **CPU only**; weights must arrive as attached datasets.
- **jupytext `# %%`** — cell-delimited `.py` format; `scripts/_py_to_ipynb.py` converts it to `.ipynb` at push time.

---

## 7. Convention for naming NEW things

When you create the next experiment / submission, follow these so the names
stay decodable and `make audit-submissions` stays green:

1. **New training config** → `configs/exp/e<FAMILY><n><rev>_<short_name>.yaml`.
   - Pick the family: none = V2-S line, `B` = new backbone, `C` = head/loss tweak.
   - Reuse the base number + a new `rev` letter when it's an incremental change
     to an existing config (`eB1` → `eB1b` → `eB1c`).
2. **New submission** → `kaggle/variants/b<NN>_<config>_<technique>/` containing
   `kernel-metadata.json` + one jupytext `<id>.py`. Take the next free `NN`
   (never reuse; document any rename as a gap in `audit.py`'s whitelist).
3. **New checkpoint dataset** → `kaggle/datasets/birdclef2026-<exp>-pkg-ckpt/dataset-metadata.json`,
   owner `winbeaux`, slug derived from the experiment.
4. **Register it** in [`submissions/manifest.yaml`](../submissions/manifest.yaml)
   under `b_variants` (id, exp_config, checkpoint, dataset_handle, kernel_id,
   notes_label). The submit tool reads it from there.
5. **Submit** with `uv run python scripts/kaggle/submit.py <id>` (add `--dry-run`
   first to verify staging without spending a submission).
6. **Record the result** by editing the manifest row (`kernel_version`,
   `dataset_version`, `lb_public`). See [`docs/SUBMISSION_WORKFLOW.md`](SUBMISSION_WORKFLOW.md).
