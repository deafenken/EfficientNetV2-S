#!/usr/bin/env python3
import json
import shutil
from pathlib import Path


USERNAME = "longkunshicandyman"

VARIANTS = [
    {
        "name": "a0_v3_lgbm_original",
        "slug": "birdclef-2026-v3-lgbm-original",
        "title": "BirdCLEF 2026 V3 LGBM Original",
        "notebook": "references/public_baselines/youssefmo942009_version_3_lgbm/version_3_lgbm.ipynb",
        "code_file": "version_3_lgbm.ipynb",
    },
    {
        "name": "a18_winner_position_0928",
        "slug": "birdclef-2026-a18-winner-position-0928",
        "title": "BirdCLEF 2026 A18 Winner Position 0928",
        "notebook": "references/public_baselines/koushikrudra_0928_winner_position/0-928-winner-position.ipynb",
        "code_file": "0-928-winner-position.ipynb",
        "dataset_sources": [
            "jaejohn/perch-meta",
            "dingjiarun/birds-train2",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
    },
    {
        "name": "a19_improved_ensemble",
        "slug": "birdclef-2026-a19-improved-ensemble",
        "title": "BirdCLEF 2026 A19 Improved Ensemble",
        "notebook": "references/public_baselines/yuriygreben_improved_ensemble/birdclef-2026-improved-ensemble.ipynb",
        "code_file": "birdclef-2026-improved-ensemble.ipynb",
        "dataset_sources": [
            "yuriygreben/perch-meta",
            "dingjiarun/birds-train2",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
    },
    {
        "name": "a20_perch_yamnet_fast_blend",
        "slug": "birdclef-2026-a20-perch-yamnet-fast-blend",
        "title": "BirdCLEF 2026 A20 Perch YAMNet Fast Blend",
        "notebook": "references/public_baselines/lingyu07_0928_perch_yamnet_blend/0-928-perch-yamnet-fast-blend.ipynb",
        "code_file": "0-928-perch-yamnet-fast-blend.ipynb",
        "dataset_sources": [
            "lingyu07/perch-protossm-ext33-artifacts",
            "lingyu07/external-audio-unified-allwav-20260408",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
        "model_sources": [
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
            "google/yamnet/tensorflow2/yamnet/1",
        ],
    },
    {
        "name": "a21_pantanal_onnx",
        "slug": "birdclef-2026-a21-pantanal-onnx",
        "title": "BirdCLEF 2026 A21 Pantanal ONNX",
        "notebook": "references/public_baselines/dingjiarun_pantanal_onnx/pantanal-distill-birdclef2026-onnx.ipynb",
        "code_file": "pantanal-distill-birdclef2026-onnx.ipynb",
        "dataset_sources": [
            "lixin73/birdclef2026-v27-onnx-perch-meta-forum-v1-lb872",
            "jaejohn/perch-meta",
        ],
        "model_sources": [
            "google/bird-vocalization-classifier/tensorflow2/perch_v2/2",
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
        ],
        "enable_gpu": True,
    },
    {
        "name": "a22_pantanal_onnx_optimized",
        "slug": "birdclef-2026-a22-pantanal-onnx-optimized",
        "title": "BirdCLEF 2026 A22 Pantanal ONNX Optimized",
        "notebook": "references/public_baselines/dingjiarun_pantanal_onnx/ablations/a22_pantanal_onnx_optimized.ipynb",
        "code_file": "a22_pantanal_onnx_optimized.ipynb",
        "dataset_sources": [
            "lixin73/birdclef2026-v27-onnx-perch-meta-forum-v1-lb872",
            "jaejohn/perch-meta",
        ],
        "model_sources": [
            "google/bird-vocalization-classifier/tensorflow2/perch_v2/2",
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
        ],
        "enable_gpu": False,
    },
    {
        "name": "a23_v3_diagnostic_gate",
        "slug": "birdclef-2026-a23-v3-diagnostic-gate",
        "title": "BirdCLEF 2026 A23 V3 Diagnostic Gate",
        "notebook": "references/public_baselines/youssefmo942009_version_3_lgbm/score_ablations/a23_v3_diagnostic_gate.ipynb",
        "code_file": "a23_v3_diagnostic_gate.ipynb",
    },
    {
        "name": "a24_v3_probe_heavy_softpost_mirror",
        "slug": "birdclef-2026-a24-v3-probe-heavy-softpost-mirror",
        "title": "BirdCLEF 2026 A24 V3 Probe Heavy Softpost Mirror",
        "notebook": "references/public_baselines/youssefmo942009_version_3_lgbm/score_ablations/a24_v3_probe_heavy_softpost_mirror.ipynb",
        "code_file": "a24_v3_probe_heavy_softpost_mirror.ipynb",
    },
    {
        "name": "a25_mtoshidesu_0932_v4",
        "slug": "birdclef-2026-a25-mtoshidesu-0932-v4",
        "title": "BirdCLEF 2026 A25 Mtoshidesu 0932 V4",
        "notebook": "references/public_baselines/mtoshidesu_0932_test_mod_v4/0-932-test-mod-version-4.ipynb",
        "code_file": "0-932-test-mod-version-4.ipynb",
        "dataset_sources": [
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
        "kernel_sources": ["ashok205/tf-wheels"],
        "model_sources": [
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
        ],
    },
    {
        "name": "a26_a0_a24_rank_ensemble",
        "slug": "birdclef-2026-a26-a0-a24-rank-ensemble",
        "title": "BirdCLEF 2026 A26 A0 A24 Rank Ensemble",
        "notebook": "references/ensembles/a26_a0_a24_rank_ensemble.ipynb",
        "code_file": "a26_a0_a24_rank_ensemble.ipynb",
        "dataset_sources": [],
        "kernel_sources": [
            "longkunshicandyman/birdclef-2026-v3-lgbm-original",
            "longkunshicandyman/birdclef-2026-a24-v3-probe-heavy-softpost-mirror",
        ],
        "model_sources": [],
    },
    {
        "name": "a27_tucker_distilled_sed",
        "slug": "birdclef-2026-a27-tucker-distilled-sed",
        "title": "BirdCLEF 2026 A27 Tucker Distilled SED",
        "notebook": "references/public_baselines/tucker_bc2026_distilled_sed/bc2026-distilled-sed.ipynb",
        "code_file": "bc2026-distilled-sed.ipynb",
        "dataset_sources": [
            "tuckerarrants/bc2026-distilled-sed-public",
            "tuckerarrants/birdclef-2026-waveform-cache",
            "tuckerarrants/perch-v2-no-dft-onnx",
        ],
        "kernel_sources": [],
        "model_sources": [],
    },
    {
        "name": "a28_ttahara_hgnetv2_b0_inference",
        "slug": "birdclef-2026-a28-ttahara-hgnetv2-b0-inference",
        "title": "BirdCLEF 2026 A28 Ttahara HGNetV2 B0 Inference",
        "notebook": "references/public_baselines/ttahara_hgnetv2_b0_inference/birdclef-2026-hgnetv2-b0-baseline-inference.ipynb",
        "code_file": "birdclef-2026-hgnetv2-b0-baseline-inference.ipynb",
        "dataset_sources": [],
        "kernel_sources": [
            "ttahara/birdclef-2026-download-wheels",
            "ttahara/birdclef-2026-hgnetv2-b0-baseline-training",
        ],
        "model_sources": [],
    },
    {
        "name": "a29_nina_ensemble_solutions",
        "slug": "birdclef-2026-a29-nina-ensemble-solutions",
        "title": "BirdCLEF 2026 A29 Nina Ensemble Solutions",
        "notebook": "references/public_baselines/nina2025_ensemble_solutions/birdclef-2026-ensemble-of-solutions.ipynb",
        "code_file": "birdclef-2026-ensemble-of-solutions.ipynb",
        "dataset_sources": [
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
            "hideyukizushi/sgkfk-202604041716",
        ],
        "kernel_sources": [
            "stpeteishii/gpu-birdclef-2026-mobilenetv3-train",
            "ashok205/tf-wheels",
            "hideyukizushi/bird26-reprod-perch-proto-residualssm-train-s7177",
        ],
        "model_sources": [
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
        ],
    },
    # ─────────────────────────────────────────────────────────────────
    # Phase 2: 0.943 public family (post 2026-04-28). Public SED weights
    # at tuckerarrants/bc2026-distilled-sed-public unblocked the 0.93+ tier.
    # Shared base = wliilamsam 0.943 ONNX Perch + Sequence + 5-fold SED.
    # ─────────────────────────────────────────────────────────────────
    {
        "name": "a30_wliilamsam_0943_base",
        "slug": "birdclef-2026-a30-wliilamsam-0-943-base",
        "title": "BirdCLEF 2026 A30 Wliilamsam 0.943 Base",
        "notebook": "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/birdclef-2026-0-943-onnx-perch-sequence-sed.ipynb",
        "code_file": "birdclef-2026-0-943-onnx-perch-sequence-sed.ipynb",
        "dataset_sources": [
            "tuckerarrants/bc2026-distilled-sed-public",
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
        "kernel_sources": ["ashok205/tf-wheels"],
        "model_sources": [
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
        ],
    },
    {
        "name": "a31_konbu17_train_audio_head",
        "slug": "birdclef-2026-a31-konbu17-train-audio-head",
        "title": "BirdCLEF 2026 A31 Konbu17 Train Audio Head",
        "notebook": "references/public_baselines/konbu17_train_audio_head_0943/bird26-wliilamsam-0943-with-train-audio-head.ipynb",
        "code_file": "bird26-wliilamsam-0943-with-train-audio-head.ipynb",
        "dataset_sources": [
            "tuckerarrants/bc2026-distilled-sed-public",
            "konbu17/bird26-train-audio-head-v1",
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
        "kernel_sources": ["ashok205/tf-wheels"],
        "model_sources": [
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
        ],
    },
    {
        "name": "a32_mattiaangeli_better_blend",
        "slug": "birdclef-2026-a32-mattiaangeli-better-blend",
        "title": "BirdCLEF 2026 A32 Mattiaangeli Better Blend",
        "notebook": "references/public_baselines/mattiaangeli_better_blend_0943/birdclef-2026-0-943-better-blend.ipynb",
        "code_file": "birdclef-2026-0-943-better-blend.ipynb",
        "dataset_sources": [
            "tuckerarrants/bc2026-distilled-sed-public",
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
        "kernel_sources": ["ashok205/tf-wheels"],
        "model_sources": [
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
        ],
    },
    {
        "name": "a33_konbu_head_mattia_rescue",
        "slug": "birdclef-2026-a33-konbu-head-mattia-rescue",
        "title": "BirdCLEF 2026 A33 Konbu Head + Mattia Rescue",
        "notebook": "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/a33_konbu_head_mattia_rescue.ipynb",
        "code_file": "a33_konbu_head_mattia_rescue.ipynb",
        "dataset_sources": [
            "tuckerarrants/bc2026-distilled-sed-public",
            "konbu17/bird26-train-audio-head-v1",
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
        "kernel_sources": ["ashok205/tf-wheels"],
        "model_sources": [
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
        ],
    },
    {
        # A39 = A33 + activate the per-class threshold sharpening that wliilamsam left
        # commented out (cell 7f-3 + cell 10 line ~165). One-line uncomment.
        "name": "a39_a33_threshold_sharpen",
        "slug": "birdclef-2026-a39-a33-threshold-sharpen",
        "title": "BirdCLEF 2026 A39 A33 Threshold Sharpen",
        "notebook": "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/a39_a33_threshold_sharpen.ipynb",
        "code_file": "a39_a33_threshold_sharpen.ipynb",
        "dataset_sources": [
            "tuckerarrants/bc2026-distilled-sed-public",
            "konbu17/bird26-train-audio-head-v1",
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
        "kernel_sources": ["ashok205/tf-wheels"],
        "model_sources": [
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
        ],
    },
    {
        # A40 = A33 with final blend changed from rank-percentile to logit-space weighted
        # averaging. Rescue gates unchanged. Different blend math = different ranking signal.
        "name": "a40_a33_logit_blend",
        "slug": "birdclef-2026-a40-a33-logit-blend",
        "title": "BirdCLEF 2026 A40 A33 Logit Blend",
        "notebook": "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/a40_a33_logit_blend.ipynb",
        "code_file": "a40_a33_logit_blend.ipynb",
        "dataset_sources": [
            "tuckerarrants/bc2026-distilled-sed-public",
            "konbu17/bird26-train-audio-head-v1",
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
        "kernel_sources": ["ashok205/tf-wheels"],
        "model_sources": [
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
        ],
    },
    {
        # A41 = A33 with mattiaangeli rescue boost coefficients amplified
        # (FAKE_ONLY 0.12->0.18, PROTO_CONT 0.15->0.22, SED_ONLY 0.12->0.18). Thresholds
        # unchanged — only the magnitude of the rescue is amplified.
        "name": "a41_a33_amplified_rescue",
        "slug": "birdclef-2026-a41-a33-amplified-rescue",
        "title": "BirdCLEF 2026 A41 A33 Amplified Rescue",
        "notebook": "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/a41_a33_amplified_rescue.ipynb",
        "code_file": "a41_a33_amplified_rescue.ipynb",
        "dataset_sources": [
            "tuckerarrants/bc2026-distilled-sed-public",
            "konbu17/bird26-train-audio-head-v1",
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
        "kernel_sources": ["ashok205/tf-wheels"],
        "model_sources": [
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
        ],
    },
    {
        # A38 = A33 base with the SelectiveSSM upgraded to v181's multi-head version
        # (params=4 by default, ~2x compute over single-head). Forward signature unchanged
        # so LightProtoSSM caller is untouched. CPU-safe: A33 ran ~3.85h, +10-20% from
        # multi-head SSM keeps us well under the 9h limit.
        "name": "a38_a33_multihead_ssm",
        "slug": "birdclef-2026-a38-a33-multihead-ssm",
        "title": "BirdCLEF 2026 A38 A33 Multihead SSM",
        "notebook": "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/a38_a33_multihead_ssm.ipynb",
        "code_file": "a38_a33_multihead_ssm.ipynb",
        "dataset_sources": [
            "tuckerarrants/bc2026-distilled-sed-public",
            "konbu17/bird26-train-audio-head-v1",
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
        "kernel_sources": ["ashok205/tf-wheels"],
        "model_sources": [
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
        ],
    },
    {
        # A37 = A36 with final blend tilted toward A33 (W_A33=0.55, W_v181=0.45).
        # Same in-kernel ensemble pipeline; only the final blend cell's two weights change.
        "name": "a37_a33_a35_tilt_55_45",
        "slug": "birdclef-2026-a37-a33-a35-tilt-55-45",
        "title": "BirdCLEF 2026 A37 A33 A35 Tilt 55 45",
        "notebook": "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/a37_a33_a35_tilt_55_45.ipynb",
        "code_file": "a37_a33_a35_tilt_55_45.ipynb",
        "dataset_sources": [
            "tuckerarrants/bc2026-distilled-sed-public",
            "needless090/birdclef2026-perch-tflite",
            "konbu17/bird26-train-audio-head-v1",
            "jaejohn/perch-meta",
            "tuckerarrants/perch-v2-no-dft-onnx",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
        "kernel_sources": ["ashok205/tf-wheels"],
        "model_sources": [
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
        ],
    },
    {
        # A36 = in-kernel ensemble of A33 (wliilamsam+head+rescue, LB 0.942) and
        # A35 (udaysonawane v181, LB 0.941). Spearman A33<>A35 = 0.94 means ~6% of
        # samples have very different rankings — the ensemble pre-condition.
        # CPU time: A35 ~4.6h + A33 ~3.85h ≈ 8.5h vs 9h limit. ~30% timeout risk.
        "name": "a36_a33_a35_blend",
        "slug": "birdclef-2026-a36-a33-a35-blend",
        "title": "BirdCLEF 2026 A36 A33 A35 Blend",
        "notebook": "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/a36_a33_a35_blend.ipynb",
        "code_file": "a36_a33_a35_blend.ipynb",
        "dataset_sources": [
            "tuckerarrants/bc2026-distilled-sed-public",
            "needless090/birdclef2026-perch-tflite",
            "konbu17/bird26-train-audio-head-v1",
            "jaejohn/perch-meta",
            "tuckerarrants/perch-v2-no-dft-onnx",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
        "kernel_sources": ["ashok205/tf-wheels"],
        "model_sources": [
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
        ],
    },
    {
        # A35 = direct reproduction of udaysonawane/birdclef2026-v181.
        # Independent stack from A33: ProtoSSM v4 with cross-attention, FocalCNN backbone,
        # Mixup+CutMix, species-frequency aware focal loss, isotonic calibration, 5-fold SED.
        # Risk: heavier than wliilamsam 0.943; CPU time may approach 9hr limit.
        "name": "a35_udaysonawane_v181",
        "slug": "birdclef-2026-a35-udaysonawane-v181",
        "title": "BirdCLEF 2026 A35 Udaysonawane V181",
        "notebook": "references/public_baselines/udaysonawane_v181/birdclef2026-v181.ipynb",
        "code_file": "birdclef2026-v181.ipynb",
        "dataset_sources": [
            "tuckerarrants/bc2026-distilled-sed-public",
            "needless090/birdclef2026-perch-tflite",
            "jaejohn/perch-meta",
            "tuckerarrants/perch-v2-no-dft-onnx",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
        "kernel_sources": ["ashok205/tf-wheels"],
        "model_sources": [
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
        ],
    },
    {
        # A34 = A33 + limprog v0.9999 epoch upscale (ProtoSSM/ResidualSSM/MLP probes)
        # + mattiaangeli's calibrated SED_W = 0.40 (PROTO 0.45, SED 0.40, HEAD 0.15).
        # Two patches over A33; everything else identical.
        "name": "a34_a33_epoch_upscale",
        "slug": "birdclef-2026-a34-a33-epoch-upscale",
        "title": "BirdCLEF 2026 A34 A33 Epoch Upscale",
        "notebook": "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/a34_a33_epoch_upscale.ipynb",
        "code_file": "a34_a33_epoch_upscale.ipynb",
        "dataset_sources": [
            "tuckerarrants/bc2026-distilled-sed-public",
            "konbu17/bird26-train-audio-head-v1",
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
        "kernel_sources": ["ashok205/tf-wheels"],
        "model_sources": [
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
        ],
    },
    {
        "name": "a7_v3_proto_045",
        "slug": "birdclef-2026-v3-a7-proto-045",
        "title": "BirdCLEF 2026 V3 A7 Proto 045",
        "notebook": "references/public_baselines/youssefmo942009_version_3_lgbm/score_ablations/a7_v3_proto_045.ipynb",
        "code_file": "a7_v3_proto_045.ipynb",
    },
    {
        "name": "a8_v3_proto_055",
        "slug": "birdclef-2026-v3-a8-proto-055",
        "title": "BirdCLEF 2026 V3 A8 Proto 055",
        "notebook": "references/public_baselines/youssefmo942009_version_3_lgbm/score_ablations/a8_v3_proto_055.ipynb",
        "code_file": "a8_v3_proto_055.ipynb",
    },
    {
        "name": "a9_v3_residual_030",
        "slug": "birdclef-2026-v3-a9-residual-030",
        "title": "BirdCLEF 2026 V3 A9 Residual 030",
        "notebook": "references/public_baselines/youssefmo942009_version_3_lgbm/score_ablations/a9_v3_residual_030.ipynb",
        "code_file": "a9_v3_residual_030.ipynb",
    },
    {
        "name": "a10_v3_probe_050_050",
        "slug": "birdclef-2026-v3-a10-probe-050-050",
        "title": "BirdCLEF 2026 V3 A10 Probe 050 050",
        "notebook": "references/public_baselines/youssefmo942009_version_3_lgbm/score_ablations/a10_v3_probe_050_050.ipynb",
        "code_file": "a10_v3_probe_050_050.ipynb",
    },
    {
        "name": "a11_v3_no_threshold",
        "slug": "birdclef-2026-v3-a11-no-threshold",
        "title": "BirdCLEF 2026 V3 A11 No Threshold",
        "notebook": "references/public_baselines/youssefmo942009_version_3_lgbm/score_ablations/a11_v3_no_threshold.ipynb",
        "code_file": "a11_v3_no_threshold.ipynb",
    },
    {
        "name": "a12_v3_proto_060",
        "slug": "birdclef-2026-v3-a12-proto-060",
        "title": "BirdCLEF 2026 V3 A12 Proto 060",
        "notebook": "references/public_baselines/youssefmo942009_version_3_lgbm/score_ablations/a12_v3_proto_060.ipynb",
        "code_file": "a12_v3_proto_060.ipynb",
    },
    {
        "name": "a13_v3_proto_055_residual_030",
        "slug": "birdclef-2026-v3-a13-proto-055-residual-030",
        "title": "BirdCLEF 2026 V3 A13 Proto 055 Residual 030",
        "notebook": "references/public_baselines/youssefmo942009_version_3_lgbm/score_ablations/a13_v3_proto_055_residual_030.ipynb",
        "code_file": "a13_v3_proto_055_residual_030.ipynb",
    },
    {
        "name": "a14_v3_proto_055_no_threshold",
        "slug": "birdclef-2026-v3-a14-proto-055-no-threshold",
        "title": "BirdCLEF 2026 V3 A14 Proto 055 No Threshold",
        "notebook": "references/public_baselines/youssefmo942009_version_3_lgbm/score_ablations/a14_v3_proto_055_no_threshold.ipynb",
        "code_file": "a14_v3_proto_055_no_threshold.ipynb",
    },
    {
        "name": "a15_v3_residual_030_no_threshold",
        "slug": "birdclef-2026-v3-a15-residual-030-no-threshold",
        "title": "BirdCLEF 2026 V3 A15 Residual 030 No Threshold",
        "notebook": "references/public_baselines/youssefmo942009_version_3_lgbm/score_ablations/a15_v3_residual_030_no_threshold.ipynb",
        "code_file": "a15_v3_residual_030_no_threshold.ipynb",
    },
    {
        "name": "a16_two_pass_advanced_pp",
        "slug": "birdclef-2026-a16-two-pass-advanced-pp",
        "title": "BirdCLEF 2026 A16 Two Pass Advanced PP",
        "notebook": "references/public_baselines/marynaborovska_two_pass_ssm_advanced_pp/birdclef-26-two-pass-ssm-advanced-pp.ipynb",
        "code_file": "birdclef-26-two-pass-ssm-advanced-pp.ipynb",
        "dataset_sources": [
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
    },
    {
        "name": "a17_two_pass_test_tta",
        "slug": "birdclef-2026-a17-two-pass-test-tta",
        "title": "BirdCLEF 2026 A17 Two Pass Test TTA",
        "notebook": "references/public_baselines/marynaborovska_two_pass_ssm_advanced_pp/ablations/a17_two_pass_test_tta.ipynb",
        "code_file": "a17_two_pass_test_tta.ipynb",
        "dataset_sources": [
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
        ],
    },
    {
        "name": "a1_v3_lgbm_plus",
        "slug": "birdclef-2026-perch-lgbm-plus",
        "title": "BirdCLEF 2026 Perch LGBM Plus",
        "notebook": "references/public_baselines/youssefmo942009_version_3_lgbm/version_3_lgbm_plus.ipynb",
        "code_file": "version_3_lgbm_plus.ipynb",
    },
    {
        "name": "a2_no_residual",
        "slug": "birdclef-2026-lgbm-plus-a2-no-residual",
        "title": "BirdCLEF 2026 LGBM Plus A2 No Residual",
        "notebook": "references/public_baselines/youssefmo942009_version_3_lgbm/ablations/a2_no_residual.ipynb",
        "code_file": "a2_no_residual.ipynb",
    },
    {
        "name": "a3_proto_035",
        "slug": "birdclef-2026-lgbm-plus-a3-proto-035",
        "title": "BirdCLEF 2026 LGBM Plus A3 Proto 035",
        "notebook": "references/public_baselines/youssefmo942009_version_3_lgbm/ablations/a3_proto_035.ipynb",
        "code_file": "a3_proto_035.ipynb",
    },
    {
        "name": "a4_no_threshold_sharpen",
        "slug": "birdclef-2026-lgbm-plus-a4-no-threshold",
        "title": "BirdCLEF 2026 LGBM Plus A4 No Threshold",
        "notebook": "references/public_baselines/youssefmo942009_version_3_lgbm/ablations/a4_no_threshold_sharpen.ipynb",
        "code_file": "a4_no_threshold_sharpen.ipynb",
    },
    {
        "name": "a5_v51_onnx_cache_0943",
        "slug": "birdclef-2026-v51-onnx-cache-0943",
        "title": "BirdCLEF 2026 V51 ONNX Cache 0943",
        "notebook": "references/public_baselines/pacerq_v51_onnx_cache/v51-onnx-cache-0-943-params.ipynb",
        "code_file": "v51-onnx-cache-0-943-params.ipynb",
        "dataset_sources": [
            "yuriygreben/perch-meta",
            "atahalam/perch-meta-0-943",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
            "i2nfinit3y/onnxruntime",
        ],
        "kernel_sources": [],
    },
    {
        "name": "a6_v51_cached_oof_weight060",
        "slug": "birdclef-2026-v51-cached-oof-weight-060",
        "title": "BirdCLEF 2026 V51 Cached OOF Weight 060",
        "notebook": "references/public_baselines/pacerq_v51_onnx_cache/ablations/a6_v51_cached_oof_weight060.ipynb",
        "code_file": "a6_v51_cached_oof_weight060.ipynb",
        "dataset_sources": [
            "yuriygreben/perch-meta",
            "atahalam/perch-meta-0-943",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
            "i2nfinit3y/onnxruntime",
        ],
        "kernel_sources": [],
    },
]


def base_metadata(variant):
    return {
        "id": f"{USERNAME}/{variant['slug']}",
        "title": variant["title"],
        "code_file": variant["code_file"],
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": variant.get("enable_gpu", False),
        "enable_internet": variant.get("enable_internet", False),
        "enable_tpu": variant.get("enable_tpu", False),
        "dataset_sources": variant.get("dataset_sources", ["jaejohn/perch-meta"]),
        "competition_sources": ["birdclef-2026"],
        "kernel_sources": variant.get("kernel_sources", ["ashok205/tf-wheels"]),
        "model_sources": variant.get(
            "model_sources",
            ["google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"],
        ),
    }


def main():
    out_root = Path("kaggle/variants")
    out_root.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        out_dir = out_root / variant["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        notebook_path = Path(variant["notebook"])
        if not notebook_path.exists():
            raise FileNotFoundError(notebook_path)
        shutil.copy2(notebook_path, out_dir / variant["code_file"])
        (out_dir / "kernel-metadata.json").write_text(
            json.dumps(base_metadata(variant), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{variant['name']}: {USERNAME}/{variant['slug']} -> {out_dir}")


if __name__ == "__main__":
    main()
