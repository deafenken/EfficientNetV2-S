"""The four Kaggle submit pipeline steps.

Faithfully extracted from the old ``scripts/22_kaggle_submit.py`` (every
hard-won fix preserved: IPv4-only DNS for the L40 VPN egress, dataset-ready
polling before kernel push, legacy-CLI kernel push with the PAT stripped from
the subprocess env). Operates on ``manifest.Variant`` objects.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import auth
from manifest import Variant


def _kernel_files(variant: Variant) -> tuple[Path, Path]:
    """Locate the variant's single ``*.py`` jupytext source + its .ipynb path."""
    py_files = sorted(
        p for p in variant.kernel_dir.glob("*.py") if not p.name.startswith("_")
    )
    if len(py_files) != 1:
        sys.exit(
            f"[FATAL] {variant.kernel_dir} must contain exactly one .py jupytext "
            f"source; found {len(py_files)}: {[p.name for p in py_files]}"
        )
    py = py_files[0]
    return py, py.with_suffix(".ipynb")


# --------------------------------------------------------------------------- #
# 1. Stage payload
# --------------------------------------------------------------------------- #
def stage_payload(variant: Variant, root: Path) -> None:
    payload = variant.payload_dir
    print(f"[stage] payload at {payload}")
    if payload.exists():
        shutil.rmtree(payload)
    payload.mkdir(parents=True)

    shutil.copytree(
        root / "src/birdclef2026",
        payload / "src/birdclef2026",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy(root / "configs/data.yaml", payload / "data.yaml")

    ckpt = root / variant.checkpoint
    if not ckpt.is_file():
        sys.exit(f"[FATAL] checkpoint missing: {ckpt}")
    shutil.copy(ckpt, payload / "swa.pt")

    for src_rel, dst_name in variant.extra_payload:
        src = root / src_rel
        if not src.is_file():
            sys.exit(f"[FATAL] extra_payload missing: {src}")
        shutil.copy(src, payload / dst_name)

    # dataset-metadata.json is resolved from the handle slug (this is the line
    # the old scripts/18 got WRONG — it hardcoded the eb1 path for the e01
    # dataset, corrupting e01's metadata on re-run).
    ds_slug = variant.dataset_handle.split("/", 1)[1]
    ds_meta = root / "kaggle/datasets" / ds_slug / "dataset-metadata.json"
    if not ds_meta.is_file():
        sys.exit(f"[FATAL] dataset-metadata.json missing: {ds_meta}")
    shutil.copy(ds_meta, payload / "dataset-metadata.json")

    total = sum(p.stat().st_size for p in payload.rglob("*") if p.is_file())
    print(f"[stage] {total / 1024 / 1024:.1f} MB total")
    for p in sorted(payload.rglob("*"))[:30]:
        if p.is_file():
            print(f"  {p.relative_to(payload)}")


# --------------------------------------------------------------------------- #
# 2. Push dataset (kagglehub, IPv4-only)
# --------------------------------------------------------------------------- #
def push_dataset(variant: Variant) -> None:
    # The L40 box egresses via a university VPN with broken IPv6 routing to the
    # GCS upload endpoints — requests.put hangs in sock.connect for ~5 min. Force
    # IPv4 globally before importing kagglehub so its session uses our resolver.
    import socket

    _orig_getaddrinfo = socket.getaddrinfo

    def _v4_only(host, port, family=0, *args, **kwargs):
        return _orig_getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)

    socket.getaddrinfo = _v4_only

    import kagglehub

    print(f"[ds] uploading → {variant.dataset_handle} (IPv4-only)")
    stamp = subprocess.run(
        ["date", "+%Y-%m-%d_%H%M"], capture_output=True, text=True
    ).stdout.strip()
    notes = f"{variant.notes_label} {stamp}"
    kagglehub.dataset_upload(
        handle=variant.dataset_handle,
        local_dataset_dir=str(variant.payload_dir),
        version_notes=notes,
        ignore_patterns=["__pycache__/", "*.pyc", ".git/", "dataset-metadata.json"],
    )
    print(f"[ds] OK → https://www.kaggle.com/datasets/{variant.dataset_handle}")
    _wait_for_dataset_ready(variant.dataset_handle)


def _wait_for_dataset_ready(handle: str, timeout_s: int = 600, poll_s: int = 10) -> None:
    """Poll ``kaggle datasets status`` until queryable.

    Without this, ``kaggle kernels push`` races the dataset processor and
    silently strips the dataset_source → kernel runs without the ckpt →
    fail-soft to an all-zero submission.
    """
    print(f"[ds] waiting for {handle} to finish processing…")
    deadline = time.monotonic() + timeout_s
    env = auth.push_env()
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["kaggle", "datasets", "status", handle],
            env=env, capture_output=True, text=True,
        )
        out = (result.stdout + result.stderr).lower()
        if "complete" in out or "ready" in out:
            print(f"[ds] ready ({result.stdout.strip()})")
            return
        time.sleep(poll_s)
    sys.exit(
        f"[FATAL] dataset {handle} did not become ready within {timeout_s}s. "
        f"Re-run with --kernel-only after manual confirmation at "
        f"https://www.kaggle.com/datasets/{handle}."
    )


# --------------------------------------------------------------------------- #
# 3. Convert .py → .ipynb
# --------------------------------------------------------------------------- #
def convert_notebook(variant: Variant, root: Path) -> None:
    py, ipynb = _kernel_files(variant)
    print(f"[nb] {py.name} → {ipynb.name}")
    subprocess.check_call(
        [sys.executable, str(root / "scripts/_py_to_ipynb.py"), str(py), str(ipynb)]
    )


# --------------------------------------------------------------------------- #
# 4. Push kernel (legacy kaggle CLI — PAT lacks the kernels scope)
# --------------------------------------------------------------------------- #
def push_kernel(variant: Variant, root: Path) -> None:
    auth.require_legacy_json()
    env = auth.push_env()
    cmd = ["kaggle", "kernels", "push", "-p", str(variant.kernel_dir)]
    print(f"[kernel] $ {' '.join(cmd)}")
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        sys.exit(f"[FATAL] kaggle kernels push exited rc={rc}")
    kernel_id = json.loads((variant.kernel_dir / "kernel-metadata.json").read_text())["id"]
    print(f"[kernel] OK → https://www.kaggle.com/code/{kernel_id}")
    print()
    print("Next: open the kernel URL, wait for the run to finish, then click")
    print("      'Submit to Competition' to enter the BirdCLEF 2026 LB.")
    print(f"Then record the result:  edit submissions/manifest.yaml → {variant.id}")
    print("  (set kernel_version, dataset_version, lb_public). See docs/SUBMISSION_WORKFLOW.md")
