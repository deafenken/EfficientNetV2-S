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

