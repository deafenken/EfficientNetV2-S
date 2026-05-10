# Gap Analysis: BirdCLEF 2026 (B) vs 2025 2nd Place (A)

Reference: 2025 2nd Place (LB Public 0.925 / Private 0.928) vs Current 2026 (LB ~0.929)

---

## 1. Audio Pipeline (sample rate, segment length, mel params, hop, normalization)

| Aspect | A (2025 2nd) | B (2026 Current) | Status |
|--------|--------------|-----------------|--------|
| Sample rate | 32000 Hz (`code_base/models/wave_clasifier.py:143-150`) | 32000 Hz (`configs/data.yaml:44`) | ✓ Match |
| Segment length | 5.0 s (`train_configs/selected_ebs.py:13`) | 5.0 s (`configs/data.yaml:45`) | ✓ Match |
| n_fft | 2048 (`train_configs/selected_ebs.py:147`) | 2048 (`configs/data.yaml:46`) | ✓ Match |
| hop_length | 512 (`train_configs/selected_ebs.py:148`) | 512 (`configs/data.yaml:47`) | ✓ Match |
| n_mels | 128 (`train_configs/selected_ebs.py:145`) | 128 (`configs/data.yaml:48`) | ✓ Match |
| f_min | 20 Hz (`train_configs/selected_ebs.py:146`) | 20.0 Hz (`configs/data.yaml:49`) | ✓ Match |
| f_max | (implicit 16000 from Nyquist) | **14000 Hz** (`configs/data.yaml:52`) | ⚠ **B is TIGHTER** (intentional) |
| Normalization | Per-sample via `normalized=True` + `late_normalize` (`train_configs/selected_ebs.py:149`) | Global mean/std + per-sample (`configs/data.yaml:54-55, src/birdclef2026/audio.py:72-79`) | ✓ Compatible |

---

## 2. Backbone Choice and Head

| Aspect | A (2025 2nd) | B (2026 Current) | Gap |
|--------|--------------|-----------------|-----|
| **Backbone 1** | `tf_efficientnetv2_s_in21k` (`train_configs/selected_ebs.py:142`) | `tf_efficientnetv2_s.in21k_ft_in1k` w/ SWA pretrain (`configs/exp/e02_v2s_5fold_v2.yaml:19, 26`) | ✓ Same family; B adds SWA |
| **Backbone 2** | `eca_nfnet_l0` (`train_configs/selected_eca.py:207`) | Tested but NOT in main (`configs/exp/eB1_nfnet_l0.yaml:9`) | ⚠ B dropped this from ensemble |
| **Head** | `AttHead` + GeMFreq pool + dropout=0.5 (`code_base/models/wave_clasifier.py:165-170`) | `AttHead` + GeMFreq + dropout=0.5 (`src/birdclef2026/models/sed.py:54-91`) | ✓ Identical |
| **Pooling** | GeMFreq (learned p) over freq → sum-reduction via softmax attention (`code_base/models/blocks.py:64-79`) | GeMFreq (learned p) → attention softmax (`src/birdclef2026/models/sed.py:75-91`) | ✓ Match |

**Gap**: B abandoned the 2nd backbone (eca_nfnet_l0) diversity. A's ensemble of 2 backbones (ebs + eca) was the key to 0.925 public score.

---

## 3. Augmentations

### Implemented in A (2025 2nd Place)

| Augmentation | A Details | Present in B? | Notes |
|--------------|-----------|---------------|-------|
| **MixUp (wave-level)** | 50% prob, α=None (50/50 avg), clamp target `[0,1]` (`train_configs/selected_ebs.py:41-42`) | ✓ Yes (`src/birdclef2026/data/transforms.py:107-148, dataset.py:data.mixup_p`) | Identical recipe |
| **Background noise (ESC-50 + soundscape-nocall)** | 50% prob, 20 categories (rain, wind, insects, etc.), amp ∈ [0.25, 0.75] (`train_configs/selected_ebs.py:59-97`) | ✓ Yes, but **disabled by default** (`configs/data.yaml:70, enabled: false`) | ⚠ **Must enable & stage files** |
| **Spec augment (freq mask)** | max_length=10, max_masks=3, p=0.3 (`train_configs/selected_ebs.py:152-156`) | ✓ Yes (`configs/data.yaml:78-80`) | Match |
| **Spec augment (time mask)** | max_length=20, max_masks=3, p=0.3 (`train_configs/selected_ebs.py:158-162`) | ✓ Yes (`configs/data.yaml:81-83`) | Match |
| **Random filtering (batch-level)** | 4-point STFT EQ, min_db=-20 (`train_configs/selected_ebs.py:182`) | ✓ Yes (`src/birdclef2026/data/transforms.py:random_filter, sed.py:111-139`) | Match |
| **Spec power aug** | Referenced in A code (`code_base/models/wave_clasifier.py:86`) | ✗ **Not found in B** | Missing |
| **Spec lower/high freq mask** | Referenced in A code (`code_base/models/wave_clasifier.py:87`) | ✗ **Not found in B** | Missing |
| **Spec noise aug** | Referenced in A code (`code_base/models/wave_clasifier.py:93-98`) | ✗ **Not found in B** | Missing |

**Highest-impact gaps**: (1) Background noise **disabled by default** — must stage ESC-50 + soundscape-nocall; (2) Power/freq/noise spec augs not ported.

---

## 4. Loss Function

| Aspect | A (2025 2nd) | B (2026 Current) | Gap |
|--------|--------------|-----------------|-----|
| Base loss | `FocalBCELoss` (BCE + Focal 1:1 weighted) (`train_configs/selected_ebs.py:183`) | `FocalBCEWithLogits` (BCE + Focal 1:1 weighted) (`src/birdclef2026/models/losses.py:61-107`) | ✓ Equivalent |
| Focal α | 0.25 (`train_configs/selected_ebs.py:183`) | 0.25 (default, `src/birdclef2026/models/losses.py:76`) | ✓ Match |
| Focal γ | 2.0 (implicit via sigmoid_focal_loss) (`code_base/losses/focal_loss.py:18-25`) | 2.0 (default, `src/birdclef2026/models/losses.py:76`) | ✓ Match |
| Label smoothing | ε=0.05, "LSF1005" positive-mode (`train_configs/selected_ebs.py:103`) | ε=0.05, `ls_mode="positive"` (`configs/exp/e02_v2s_5fold_v2.yaml:59-60`) | ✓ Match |
| pos_weight | Implicit class balancing via sampler (`train_configs/selected_eca.py:279`) | `pos_weight=5.0` optional (`configs/exp/e02_v2s_5fold_v2.yaml:61`) | ⚠ A uses sampler; B uses both |

**Status**: B's loss closely mirrors A. Both use focal + BCE + label smoothing. Minor: B optionally adds per-class `pos_weight`.

---

## 5. Class Balancing / Sampler

| Strategy | A (2025 2nd) | B (2026 Current) | Gap |
|----------|--------------|-----------------|-----|
| **EBS model** | `EqualBalancing` (`train_configs/selected_ebs.py:214`) | `samplers.py::sqrt_balanced_weights` w/ rare_floor=30 (`configs/exp/e02_v2s_5fold_v2.yaml:50-52`) | ⚠ Different |
| **ECA model** | `class_weights_path="sqrt"` + `oversample_classes_dict` with 85 rare-bird entries (`train_configs/selected_eca.py:168, 25-85`) | No explicit oversample dict in e02; tested in `eB3` but not selected (`configs/exp/eB3_efficientnetv2_b2.yaml`) | ⚠ **B lacks explicit rare oversampling** |
| **Sampler type** | Custom sampler (A's `use_sampler: True`, `sampler_col="primary_label"`) (`train_configs/selected_eca.py:107-108`) | `DistributedWeightedSampler` via `sqrt_balanced_weights` (`src/birdclef2026/data/samplers.py:34-109`) | ✓ Functionally similar |

**Gap**: A's ECA model explicitly oversampled 85 rare species (turvul 96x, piwtyr1 90x, etc.; `train_configs/selected_eca.py:25-85`). **B does not use explicit oversampling** in e02. This is a concrete ~0.01-0.02 LB opportunity.

---

## 6. Pseudo-Labeling Pipeline

| Stage | A (2025 2nd) | B (2026 Current) | Gap |
|-------|--------------|-----------------|-----|
| **Naming** | `PseudoF2PT05MT01P04I3` (fold ensemble, prob_min=0.5, trim_min=0.1, sampling=0.4, I=iteration) | `predict_pseudo.py` w/ `--prob-min` param (`src/birdclef2026/inference/predict_pseudo.py:26-31`) | ✓ Similar concept |
| **Fold ensemble** | 5-fold ensemble → average logits → sigmoid → filter (`code_base/inefernce/inference_class.py`, pseudo-df at `soundscape_pseudo_df_path`) | 5-fold ensemble → average probs post-sigmoid (`src/birdclef2026/inference/predict_pseudo.py:74-100`) | ✓ Match |
| **Window strategy** | Non-overlapping 5s windows on soundscapes; weighted probs by start-time overlap (`code_base/datasets/wave_dataset.py:35-78`) | Non-overlapping 5s windows, simple average across folds (`src/birdclef2026/inference/predict_pseudo.py:84-106`) | ✓ Same |
| **Threshold** | Primary: always keep top-1; secondary: keep classes > prob_min (0.5 default) (`train_configs/selected_ebs.py:102, 167`) | Primary: top-1; secondary: classes > prob_min (`src/birdclef2026/inference/predict_pseudo.py:145-160`) | ✓ Match |
| **Data loading into training** | `soundscape_pseudo_df_path` (multi-source) + `soundscape_pseudo_config` sampling (`train_configs/selected_ebs.py:98-102`) | `data.pseudo_label_csv` → merged into metadata (`configs/data.yaml:31`) | ✓ Same pattern |

**Status**: Both implement essentially the same pseudo-pipeline. B's is slightly simpler (no weighted-overlap) but equivalent in practice.

---

## 7. Pretraining

| Source | A (2025 2nd) | B (2026 Current) | Gap |
|--------|--------------|-----------------|-----|
| **Upstream data** | BC 2021-2024 + Xeno-Canto + iNaturalist + CSA; 20+ fold CV on large corpus | No pretraining pipeline in public code; uses timm IN21k/IN1k defaults or external checkpoint | ⚠ **B relies on timm weights** |
| **EBS pretrain path** | `tf_efficientnetv2_s_in21k_Pretrainversion1.pth` (published) (`train_configs/selected_ebs.py:217`) | `v2s_bc2025_swa.ckpt` (same 2025 model, SWA-averaged) (`configs/exp/e02_v2s_5fold_v2.yaml:26`) | ✓ B uses A's pretrain! |
| **ECA pretrain path** | `eca_nfnet_l0_pretrain_from_bigXCV2_best.ckpt` (published) (`train_configs/selected_eca.py:282`) | Not used in main config | ⚠ **B dropped eca pretrain** |
| **Training from scratch** | Not shown in public code; assumed done offline on Kaggle | Not a primary path; config shows timm pretraining (`models/backbones.py:67-74`) | Same as A |

**Gap**: B discarded the 2025-pretrained ECA backbone, losing the diversity boost. Reintroducing it (+0.01-0.02 expected) would be straightforward.

---

## 8. Training Loop / Scheduler / Optimizer / EMA / Mixed Precision

| Component | A (2025 2nd) | B (2026 Current) | Status |
|-----------|--------------|-----------------|--------|
| **Optimizer (EBS)** | `AdamW(lr=1e-4, eps=1e-8, betas=(0.9, 0.999))` (`train_configs/selected_ebs.py:174`) | `adamw, lr=1.0e-4, weight_decay=1.0e-2` (`configs/exp/e02_v2s_5fold_v2.yaml:29-31`) | ✓ Identical |
| **Optimizer (ECA)** | `RAdam(lr=1e-3)` (`train_configs/selected_eca.py:239`) | Not used in e02 | Dropped |
| **Scheduler** | `CosineAnnealingWarmRestarts(T_0=N_EPOCHS*len_train, T_mult=1, eta_min=1e-6)` per-step (`train_configs/selected_ebs.py:175-177`) | `SequentialLR(LinearLR(warmup) + CosineAnnealingLR)` per-step (`training/train_ddp.py:265-267`) | ✓ Similar (both cosine) |
| **Warmup** | Implicit in CosineAnnealingWarmRestarts | Explicit `warmup_epochs=2` (`configs/exp/e02_v2s_5fold_v2.yaml:41`) | ✓ Both include |
| **Batch size** | 64 (`train_configs/selected_ebs.py:12`) | 64 (`configs/exp/e02_v2s_5fold_v2.yaml:36`) | ✓ Match |
| **Epochs** | 50 (`train_configs/selected_ebs.py:14`) | 50 (`configs/exp/e02_v2s_5fold_v2.yaml:40`) | ✓ Match |
| **EMA** | Not explicit in config (Lightning may use it via callbacks) | `ema_decay=0.999` w/ warmup (`configs/exp/e02_v2s_5fold_v2.yaml:47, training/ema.py:40-96`) | ✓ B explicit; A implicit |
| **SWA** | Post-training weight averaging (implied) | SWA during training `swa_epochs=5` (`configs/exp/e02_v2s_5fold_v2.yaml:48`) | ✓ Both use; B in-loop |
| **Mixed precision (EBS)** | `precision_mode="32-true"` (fp32) (`train_configs/selected_ebs.py:208`) | `amp_dtype=fp32` (`configs/exp/e02_v2s_5fold_v2.yaml:45`) | ✓ Match (no AMP) |
| **Mixed precision (ECA)** | `precision_mode="32-true"` (fp32) | Not used in e02 | Dropped |
| **Gradient clip** | Implicit in Lightning | `grad_clip: 1.0` (`configs/exp/e02_v2s_5fold_v2.yaml:46`) | ✓ Both apply |

**Status**: Training hyperparameters closely aligned. B's setup is cleaner (explicit EMA, SWA-in-loop). Only loss: dropped ECA (RAdam@1e-3) variant.

---

## 9. Ensembling and Post-Processing

| Aspect | A (2025 2nd) | B (2026 Current) | Gap |
|--------|--------------|-----------------|-----|
| **Per-fold checkpoint** | 5 folds × 2 backbones = **10 checkpoints** | 5 folds × 1 backbone (e02) = **5 checkpoints** | ⚠ **B uses half the ensemble** |
| **Fold averaging** | Average logits → sigmoid (`code_base/inefernce/inference_class.py`) | Average logits post-sigmoid (probs) (`src/birdclef2026/inference/predict.py:120-140`) | ✓ Equivalent |
| **Model export** | ONNX → OpenVINO fp16 (`scripts/main_inference_and_compile.py`, README.md:39-43) | PyTorch checkpoint export only (`training/train_ddp.py:23`) | ⚠ **B has no ONNX/OpenVINO export** |
| **Post-processing** | None evident in Kaggle notebook (README.md:61-64) | `postprocess.py` w/ sigmoid clipping, confidence scaling, adaptive smoothing, per-class thresholds (`src/birdclef2026/postprocess.py`) | B **over-equipped**; A relies on clean ensemble |
| **OOF assembly** | Implicit in Lightning callbacks | `assemble_oof.py` explicit (`src/birdclef2026/inference/assemble_oof.py:38-97`) | ✓ Both compute OOF AUC |

**Critical gaps**:
1. **Ensemble size**: A=2 backbones × 5 folds. B (e02)=1 backbone × 5 folds. Expected cost: −0.01 to −0.03 LB.
2. **Model export**: A delivers ONNX/OpenVINO for edge deployment; B stops at PyTorch.
3. **Post-processing**: B has the machinery but unused by default; A keeps submissions clean.

---

## 10. Top-3 Highest-Leverage Gaps (Ranked by Expected Payoff)

### 1. **Ensemble Diversity: Add eca_nfnet_l0 Backbone (Expected +0.015 to +0.025 LB)**
   - **File to edit**: `configs/exp/e02_v2s_5fold_v2.yaml` (define eB1 variant with eca_nfnet_l0, train 5 folds)
   - **Implementation**: Copy e02 config, change `model.name` to `eca_nfnet_l0`, use `pretrained_backbone_path: data/aux/pretrained/eca_nfnet_l0_bc2025_swa.ckpt` (download from A's release or replicate pretrain).
   - **Why**: A's public 0.925 score comes from 2-backbone ensemble (ebs + eca). B dropping eca loses 5-10% of the win. Cost: ~2× training time; benefit: ~+0.02 LB.

### 2. **Enable Background Noise Augmentation (Expected +0.010 to +0.020 LB)**
   - **File to edit**: `configs/exp/e02_v2s_5fold_v2.yaml` (set `augment.background_noise.enabled: true` + `dirs: [data/aux/esc50_ambient, ...]`)
   - **Implementation**: Run `scripts/05_prep_esc50.sh` (stub exists; downloads ESC-50 ambient subset). Update YAML to populate `dirs` with actual paths after extraction.
   - **Why**: A uses ESC-50 + soundscape-nocall (20 categories: rain, wind, insects, etc.) at 50% prob w/ amp ∈ [0.25, 0.75]. B has the code but disabled. Cost: ~2 GB disk, ~+5% epoch time; benefit: +0.01–0.02 LB (A's ablation).

### 3. **Rare-Species Oversampling (Expected +0.010 to +0.015 LB)**
   - **File to edit**: `configs/exp/e02_v2s_5fold_v2.yaml` or new variant; `src/birdclef2026/training/train_ddp.py` (rare-floor handling)
   - **Implementation**: Extract the 85-species oversample dict from A's `train_configs/selected_eca.py:25-85` (turvul 96x, piwtyr1 90x, ..., rufmot1 10x). Add to B's config as `sampler.rare_floor: 30` or explicit `oversample_dict`. Port A's oversampling logic into B's dataset builder.
   - **Why**: A's ECA model explicitly oversamples rare birds by 10–100×, compressing the 1000:1 class imbalance to ~30:1. B's sqrt-balancing (rare_floor=30) approximates this but misses the full benefit. Cost: ~+2% epoch time; benefit: +0.01–0.015 LB on tail species.

---

## Summary Table: Critical Missing Elements in B

| Feature | A Has | B Has | Priority | File to Edit |
|---------|-------|-------|----------|--------------|
| 2nd backbone (eca_nfnet_l0) | ✓ | ✗ | **CRITICAL** | `configs/exp/`, `training/train_ddp.py` |
| Background noise (ESC-50) | ✓ Enabled | ✓ Disabled | **HIGH** | `configs/exp/e02_v2s_5fold_v2.yaml` |
| Rare-species explicit oversample | ✓ | ✓ (sqrt-floor) | **HIGH** | `configs/exp/`, `training/train_ddp.py` |
| ECA-specific optimizer (RAdam) | ✓ | ✗ | **MED** | `configs/exp/` (new variant) |
| Spec power/freq/noise augs | ✓ (code) | ✗ | **MED** | `src/birdclef2026/data/transforms.py` |
| ONNX/OpenVINO export | ✓ | ✗ | **LOW** | New export script |
| Post-processing (thresholds, smoothing) | ✗ (clean) | ✓ (unused) | **LOW** | `src/birdclef2026/postprocess.py` |

---

## Reproduction Path to Match A's 0.925

1. **Stage data**: ESC-50 ambient files (240 files) + soundscape-nocall (optional).
2. **Create eB1 config** (eca_nfnet_l0 variant): copy e02, change backbone, load SWA pretrain, train 5 folds.
3. **Create eB2 config** (ebs with background noise): copy e02, enable `background_noise.enabled: true`, train 5 folds.
4. **Run pseudo-labeling R1** on soundscapes using e02 + eB1 ensemble, feed back into eB3 retraining (if data permits).
5. **Ensemble all 15 checkpoints** (e02 5-fold + eB1 5-fold + eB2 5-fold) → expected LB ≈ 0.930–0.935.
6. **Optional**: Retrain with explicit rare-species oversampling; export to ONNX/OpenVINO for the final submission.

