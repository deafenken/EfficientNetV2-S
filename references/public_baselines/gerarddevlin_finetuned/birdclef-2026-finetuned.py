# %% cell 1
import os, sys, subprocess, pandas as pd

bundle_dir = "/kaggle/input/datasets/gerarddevlin/bundle13/kaggle_bundle_v52_bag5_s44_rw010"
ckpt_path = f"{bundle_dir}/checkpoint.pth"
adapter_path = f"{bundle_dir}/perch_adapter.pth"

competition_dir = "/kaggle/input/competitions/birdclef-2026"
if not os.path.exists(competition_dir):
    competition_dir = "/kaggle/input/birdclef-2026"

model_dir = "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"

onnx_path = "/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/perch_v2_ir9.onnx"
if not os.path.exists(onnx_path):
    onnx_path = "/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/perch_v2.onnx"

extra_cache_dirs = "/kaggle/input/datasets/jaejohn/perch-meta"

print("bundle:", os.path.exists(bundle_dir))
print("ckpt:", os.path.exists(ckpt_path))
print("adapter:", os.path.exists(adapter_path))
print("competition:", os.path.exists(competition_dir))
print("model:", os.path.exists(model_dir))
print("onnx:", os.path.exists(onnx_path))
print("cache:", os.path.exists(extra_cache_dirs))

cmd = [
    sys.executable, f"{bundle_dir}/scripts/infer_pt.py",
    "--checkpoint", ckpt_path,
    "--perch-adapter", adapter_path,
    "--perch-adapter-weight", "0.2",
    "--base", competition_dir,
    "--model-dir", model_dir,
    "--onnx-path", onnx_path,
    "--extra-cache-dirs", extra_cache_dirs,
    "--output", "/kaggle/working/submission.csv",
]

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

sub = pd.read_csv("/kaggle/working/submission.csv")
print("submission shape:", sub.shape)
display(sub.head(3))

