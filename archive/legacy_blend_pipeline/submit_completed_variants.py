#!/usr/bin/env python3
import subprocess
from pathlib import Path


COMPETITION = "birdclef-2026"
OUT_ROOT = Path("outputs/kaggle_variants")

VARIANTS = [
    ("a0_original", "longkunshicandyman/birdclef-2026-v3-lgbm-original", "A0 original public V3 LGBM", 1),
    ("a7_v3_proto_045", "longkunshicandyman/birdclef-2026-v3-a7-proto-045", "A7 original V3 proto blend 0.45", 1),
    ("a8_v3_proto_055", "longkunshicandyman/birdclef-2026-v3-a8-proto-055", "A8 original V3 proto blend 0.55", 1),
    ("a9_v3_residual_030", "longkunshicandyman/birdclef-2026-v3-a9-residual-030", "A9 original V3 residual correction 0.30", 1),
    ("a10_v3_probe_050_050", "longkunshicandyman/birdclef-2026-v3-a10-probe-050-050", "A10 original V3 probe blend 0.50/0.50", 1),
    ("a11_v3_no_threshold", "longkunshicandyman/birdclef-2026-v3-a11-no-threshold", "A11 original V3 without threshold sharpening", 1),
    ("a1_plus", "longkunshicandyman/birdclef-2026-perch-lgbm-plus", "A1 plus time site window features", 1),
    ("a2_no_residual", "longkunshicandyman/birdclef-2026-lgbm-plus-a2-no-residual", "A2 plus without ResidualSSM", 1),
    ("a3_proto_035", "longkunshicandyman/birdclef-2026-lgbm-plus-a3-proto-035", "A3 plus proto weight 0.35", 1),
    ("a4_no_threshold", "longkunshicandyman/birdclef-2026-lgbm-plus-a4-no-threshold", "A4 plus no threshold sharpening", 1),
    # Phase 2 (0.943 public family)
    ("a30_wliilamsam_0943_base", "longkunshicandyman/birdclef-2026-a30-wliilamsam-0-943-base", "A30 wliilamsam 0.943 base reproduction", 1),
    ("a31_konbu17_train_audio_head", "longkunshicandyman/birdclef-2026-a31-konbu17-train-audio-head", "A31 konbu17 train_audio head on 0.943 base", 1),
    ("a32_mattiaangeli_better_blend", "longkunshicandyman/birdclef-2026-a32-mattiaangeli-better-blend", "A32 mattiaangeli better-blend rescue gates", 1),
    ("a33_konbu_head_mattia_rescue", "longkunshicandyman/birdclef-2026-a33-konbu-head-mattia-rescue", "A33 konbu head + mattiaangeli rescue gates", 1),
    ("a34_a33_epoch_upscale", "longkunshicandyman/birdclef-2026-a34-a33-epoch-upscale", "A34 A33 + limprog epoch upscale + SED_W 0.40", 1),
    ("a35_udaysonawane_v181", "longkunshicandyman/birdclef-2026-a35-udaysonawane-v181", "A35 udaysonawane v181 ProtoSSM v4 + CNN + 5-fold SED", 1),
    ("a36_a33_a35_blend", "longkunshicandyman/birdclef-2026-a36-a33-a35-blend", "A36 in-kernel rank blend of A33 + A35 (50/50)", 1),
    ("a37_a33_a35_tilt_55_45", "longkunshicandyman/birdclef-2026-a37-a33-a35-tilt-55-45", "A37 A33+A35 in-kernel blend tilted 55/45 toward A33", 1),
    ("a38_a33_multihead_ssm", "longkunshicandyman/birdclef-2026-a38-a33-multihead-ssm", "A38 A33 + v181 multi-head SelectiveSSM (params=4)", 1),
    ("a39_a33_threshold_sharpen", "longkunshicandyman/birdclef-2026-a39-a33-threshold-sharpen", "A39 A33 + activate per-class threshold sharpening", 1),
    ("a40_a33_logit_blend", "longkunshicandyman/birdclef-2026-a40-a33-logit-blend", "A40 A33 + logit-space blend (replace rank blend)", 1),
    ("a41_a33_amplified_rescue", "longkunshicandyman/birdclef-2026-a41-a33-amplified-rescue", "A41 A33 + amplified mattiaangeli rescue gates", 1),
]


def run_kaggle(args, check=True):
    cmd = ["bash", "scripts/kaggle_cmd.sh", *args]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc


def status_for(ref):
    proc = run_kaggle(["kernels", "status", ref], check=False)
    text = (proc.stdout + proc.stderr).strip()
    if "KernelWorkerStatus." in text:
        return text.rsplit("KernelWorkerStatus.", 1)[-1].split('"', 1)[0].split()[0], text
    return "UNKNOWN", text


def find_submission(out_dir):
    direct = out_dir / "submission.csv"
    if direct.exists():
        return direct
    matches = sorted(out_dir.rglob("submission.csv"))
    return matches[0] if matches else None


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_done = True
    for name, ref, message, version in VARIANTS:
        out_dir = OUT_ROOT / name
        marker = out_dir / ".submitted"
        status, raw = status_for(ref)
        print(f"{name}: {status}")

        if marker.exists():
            print(f"  already submitted: {marker}")
            continue

        if status not in {"COMPLETE", "SUCCEEDED"}:
            all_done = False
            print(f"  not ready")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  submitting kernel version {version} as code competition submission")
        proc = run_kaggle(
            [
                "competitions",
                "submit",
                "-c",
                COMPETITION,
                "-k",
                ref,
                "-f",
                "submission.csv",
                "-m",
                message,
                "-v",
                str(version),
            ],
            check=False,
        )
        (out_dir / "submit_stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (out_dir / "submit_stderr.txt").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode == 0:
            marker.write_text(message + "\n", encoding="utf-8")
            print("  submitted")
        else:
            all_done = False
            print("  submit failed")
            print((proc.stdout + proc.stderr).strip()[-1000:])

    return 0 if all_done else 1


if __name__ == "__main__":
    raise SystemExit(main())
