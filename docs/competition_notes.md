# Competition Notes

## Verified on 2026-04-20

- Competition slug: `birdclef-2026`.
- Task: identify calling species from audio recordings made in the Brazilian Pantanal.
- Species scope includes birds, amphibians, mammals, reptiles, and insects.
- Train audio is short single-recording audio, resampled to 32 kHz and converted to `.ogg`.
- Hidden test data is mounted only during Kaggle notebook scoring; `test_soundscapes/` contains about 600 one-minute `.ogg` soundscapes.
- The public sample submission should be treated as the source of truth for target columns and `row_id` format.

Sources:

- Kaggle dataset mirror/repack summary: https://www.kaggle.com/datasets/llkh0a/birdclef-2026-repack
- Kaggle launch post with entry deadline/prize context: https://www.linkedin.com/posts/kaggle_birdclef-2026-activity-7437551576271175682-tixg
- LifeCLEF/BirdCLEF 2026 page: https://www.imageclef.org/BirdCLEF2026

## Current assumptions

- `sample_submission.csv` target columns are authoritative.
- `train.csv.primary_label` may need taxonomy mapping before it matches target columns.
- `train_soundscapes_labels.csv` is useful but noisier; first baseline keeps it off.
- Because this is multi-label audio tagging, inference uses sigmoid probabilities, not softmax.

## First scoring baseline

- 5 second clips.
- 32 kHz audio.
- Log-mel spectrogram.
- ResNet18 with a one-channel first convolution.
- BCEWithLogitsLoss.
- Hidden test: split each one-minute soundscape into 5 second windows and predict each sample-submission `row_id`.

## Submission / runtime constraints (IMPORTANT)

This is an **offline code competition**. Our submission kernels run with
`enable_internet: false` and `enable_gpu: false` (see any
`kaggle/variants/*/kernel-metadata.json`), so:

- **CPU-only inference.** No GPU at scoring time. Model weights must be attached
  as Kaggle **datasets** (`*-pkg-ckpt`); nothing is downloaded at runtime.
- **Runtime budget is the binding constraint.** The hidden test is ~600 one-minute
  soundscapes (≈the size that the kernel must score within the competition's
  notebook time limit). BirdCLEF code competitions have historically capped the
  scoring run at **≤ 90 minutes CPU** — **confirm the exact limit on the
  competition's "Code Requirements" tab** before assuming. This is why the
  ensemble size / model count is bounded and why ONNX/OpenVINO fp16 CPU
  inference is the lever for fitting more models in budget.
- **Never hardcode the species count.** Read target columns (and their count)
  from `sample_submission.csv` at runtime — it is the source of truth.
- **Fail-soft.** The kernel always writes a valid (zero-filled) `submission.csv`
  first, then fills it in, so a mid-run crash still produces a scorable file.

