# Project Layout

> Target structure after Commit 2/3. Some subdirectories under
> `src/birdclef2026/` are placeholders in Commit 1 (their `__init__.py`
> documents what will go inside). The legacy flat modules at the package
> root keep working until they are formally superseded.

## Top-level

```
birdclef-2026/
├── configs/
│   ├── baseline.yaml          # legacy single-process baseline config
│   └── exp/                   # one yaml per experiment (e01_*, e02_*, ...)
├── src/birdclef2026/          # the Python package (PYTHONPATH=src)
│   ├── data/                  # dataset / transforms / folds   [Commit 2]
│   ├── models/                # backbones / SED head / losses  [Commit 2]
│   ├── training/              # DDP train loop, EMA, SWA, NS   [Commit 2]
│   ├── inference/             # OOF / test / ensemble / submit [Commit 3]
│   ├── utils/                 # cv / mel_stats / io / metrics  [Commit 2]
│   └── *.py                   # legacy flat modules (deprecated, kept until replaced)
├── scripts/                   # numbered, executable in order
│   ├── 00_setup_env.sh        # uv sync + GPU/Kaggle check        (idempotent)
│   ├── 01_download_data.sh    # main + aux dataset download       (idempotent)
│   ├── 02_compute_mel_stats.py # global mean/std over train       [Commit 2]
│   ├── 03_make_folds.py       # StratifiedGroupKFold csvs         [Commit 2]
│   ├── 10_train.sh            # torchrun wrapper for an exp yaml  [Commit 2]
│   ├── 20_predict_oof.sh      # OOF prediction                    [Commit 3]
│   ├── 30_pseudo_label.py     # 1-round noisy student pseudo gen  [Commit 3]
│   ├── 40_export_onnx.py      # checkpoint → ONNX → OpenVINO      [Commit 3]
│   ├── 50_make_submission.py  # ensemble + submission CSV         [Commit 3]
│   ├── download_data.sh       # raw kaggle CLI wrapper (called by 01_*)
│   ├── download_aux_data.sh   # raw aux wrapper (called by 01_*)
│   ├── notebook_to_py.py      # notebook → .py converter
│   ├── prepare_kaggle_kernel.sh
│   ├── kaggle_cmd.sh          # generic kaggle CLI auth wrapper
│   └── smoke_checks.py        # postprocess sanity (run via `make smoke`)
├── outputs/                   # gitignored runtime artifacts
│   ├── folds/                 # fold_0.csv … fold_4.csv
│   ├── stats/                 # global mel mean/std
│   ├── exp/<exp_id>/
│   │   ├── fold_*/best.pt     # checkpoints
│   │   ├── fold_*/log.csv     # train/val loss
│   │   ├── oof_logits.npy
│   │   └── oof_auc.txt
│   └── submissions/sub_<exp>_<tag>.csv
├── notebooks/
│   └── kaggle_inference.ipynb # final submission notebook (Kaggle-side)
├── kaggle/                    # kaggle kernel metadata (versioned dirs)
├── references/                # historical public baselines (read-only)
├── archive/                   # blend-era artifacts (read-only)
├── docs/
│   ├── PROJECT_LAYOUT.md      # this file
│   ├── competition_notes.md
│   ├── optimization_roadmap.md
│   ├── public_*.md
│   └── plans/                 # gold-medal-track plans + review
└── Makefile                   # convenience targets (see `make help`)
```

## Workflow contract

The repo is laid out for a **two-machine workflow**:

- This VPS (Claude's environment): **code only** — read, edit, commit, push.
  No `uv sync`, no training, no large downloads.
- Training machine (4×L40 server): **execution only** — `git pull`, run the
  numbered scripts, send logs back.

Per experiment, Commit 2 will produce:

```
configs/exp/eNN_<name>.yaml      # complete config (model + data + train)
scripts/run_eNN.sh                # one-line torchrun command
```

The training-machine operator runs:

```bash
git pull
bash scripts/00_setup_env.sh         # only when deps change
bash scripts/01_download_data.sh     # only on a fresh box
bash scripts/run_eNN.sh              # the actual experiment
```

Outputs land under `outputs/exp/eNN/` (gitignored). Operator pastes
`oof_auc.txt` + any errors back; Claude decides the next experiment.

## Naming conventions

- `eNN_<short_name>.yaml` — experiment configs, NN is zero-padded
  (`e01_v2s_5fold`, `e02_nfnet_5fold`, …).
- `outputs/exp/eNN/fold_K/` — per-fold artifacts.
- `sub_<exp>_<tag>.csv` — submission CSVs (`sub_e01_r0.csv`,
  `sub_ensemble_v1.csv`).

## Where things go

| Need to … | File |
|---|---|
| Add a new training experiment | `configs/exp/eNN_*.yaml` + `scripts/run_eNN.sh` |
| Tweak augmentation | `src/birdclef2026/data/transforms.py` |
| Try a new backbone | `src/birdclef2026/models/backbones.py` |
| Change CV split | `src/birdclef2026/data/folds.py` + rerun `scripts/03_make_folds.py` |
| Build the final ensemble | `src/birdclef2026/inference/ensemble.py` |
| Generate `submission.csv` | `scripts/50_make_submission.py --exp e01,e02,…` |
