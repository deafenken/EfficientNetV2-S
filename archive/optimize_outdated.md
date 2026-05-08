# EfficientNetV2-S SED Training Code Review

## Current Status

I could not find the EfficientNetV2-S SED training code in the current workspace.

Searches were run from the project root:

- `/Volumes/ORICO/kaggle/BirdCLEF+ 2026`
- Keywords checked: `EfficientNetV2`, `efficientnetv2`, `efficientnetv2_s`, `efficientnet_v2_s`, `tf_efficientnetv2_s`, `EffNetV2`, `SED`
- Non-reference project code checked under `src/`, `scripts/`, `configs/`, `docs/`, and recent modified files

The only SED/EfficientNet-related hits are existing public references or variant-generation scripts, not a new EfficientNetV2-S training implementation.

## Blocking Issues

1. The target training file is missing from this repository, so I cannot verify implementation bugs yet.

2. The local BirdCLEF data is incomplete. `data/birdclef-2026/` currently only contains:
   - `sample_submission.csv`
   - `train_soundscapes_labels.csv`

   It does not contain `train.csv`, `taxonomy.csv`, `train_audio/`, or `train_soundscapes/`, so a full local training smoke run cannot work in the current checkout.

3. `requirements.txt` does not include the likely dependencies for an EfficientNetV2-S SED training stack, especially `timm`. If the new code uses `timm.create_model("tf_efficientnetv2_s...")`, the project environment will not be reproducible without updating dependencies or using a Kaggle wheel input.

4. If the intended run target is Kaggle with internet disabled, pretrained EfficientNetV2-S weights and any extra wheels must be mounted as Kaggle inputs. The training code should not assume online model download.

## Review Checklist Once Code Is Present

These are the points I would verify in the EfficientNetV2-S SED code:

1. Data and labels
   - Class order must come from `sample_submission.csv` or `taxonomy.csv` and be saved into every checkpoint.
   - `primary_label` and `secondary_labels` need correct multi-hot encoding.
   - Soundscape labels must be window-aligned by `row_id` / `end_time`, not file-level copied to every segment unless that is intentional.
   - Validation split should be grouped by recording/file, not random segment rows.

2. Audio and spectrogram
   - Training crop length must match inference crop length.
   - Sample rate, `n_fft`, `hop_length`, mel bins, normalization, and image resize must be identical between train and inference.
   - Padding/cropping should handle short clips safely.
   - Augmentations should not break label semantics for weak labels.

3. Model
   - EfficientNetV2-S input channels must match the spectrogram tensor shape.
   - SED head should expose both framewise and clipwise outputs if frame-level pooling is used.
   - Pooling choice should be explicit: max, mean, attention, or log-sum-exp.
   - Checkpoint should include backbone name, mel config, class list, fold id, and threshold/calibration metadata.

4. Loss and metrics
   - Use `BCEWithLogitsLoss`, focal BCE, or asymmetric loss on logits, not sigmoid probabilities.
   - Any `pos_weight` must be clipped; raw rare-class weights can destabilize training.
   - Validation should report labelwise macro AUC and skip classes with only one label value.
   - Do not tune thresholds on the same predictions used for final validation without an OOF split.

5. Runtime and Kaggle compatibility
   - Mixed precision should be CUDA-only guarded.
   - `num_workers`, cache paths, and output paths should work both locally and in Kaggle.
   - For competition submission, inference must generate rows in exact `sample_submission.csv` order.
   - The trained SED branch should export a compact inference artifact, not require full training code in the submission notebook.

## Next Action

Put the EfficientNetV2-S SED training code into this repo or point me to the exact file/notebook path. The most useful format for review is a `.py` training script plus its config, or the notebook converted with `scripts/notebook_to_py.py`.
