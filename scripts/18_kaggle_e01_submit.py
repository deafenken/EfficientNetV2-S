#!/usr/bin/env python3
"""End-to-end Kaggle submission for e01 V2-S baseline fold-0 (LB control).

What it does (~3-5 min once the access token is in place):
  1. Stage src/birdclef2026/ + configs/data.yaml + outputs/exp/eB1_nfnet_l0/fold_0/swa.pt
     into a Kaggle dataset payload dir
  2. ``kagglehub.dataset_upload`` to ``winbeaux/birdclef2026-eb1-pkg-ckpt``
     (creates on first run, versions on subsequent runs)
  3. Convert the jupytext .py → .ipynb
  4. ``kagglesdk.KaggleClient.kernels.save_kernel`` to push the kernel
     (kernel runs autonomously on Kaggle once pushed)

The user clicks "Submit to Competition" on the run output afterwards
(BirdCLEF 2026 is a code competition; final submission is per-run).

Auth (split-stack: PAT for kagglehub, legacy json for kaggle CLI):
  - Dataset upload uses ``kagglehub.dataset_upload`` which honors
    ``KAGGLE_API_TOKEN`` (auto-loaded from ``~/.kaggle/access_token``).
  - Kernel push uses the legacy ``kaggle kernels push`` CLI, which needs
    ``~/.kaggle/kaggle.json`` (``{"username": ..., "key": ...}``). PAT
    tokens do not carry the kernels.write / kernels.get scopes, so
    kagglesdk's ``save_kernel`` returns 409 on first push and ``get_kernel``
    returns 403. The fix is the legacy json — get one at
    https://www.kaggle.com/settings → 'API' → 'Create New Token'.

Usage:
  cd birdclef-2026 && uv run python scripts/18_kaggle_e01_submit.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD = Path("/tmp/birdclef2026-e01-pkg-ckpt")
CKPT_REL = "outputs/exp/e01_v2s_5fold/fold_0/swa.pt"
KERNEL_DIR = ROOT / "kaggle/variants/b02_e01_v2s_baseline"
KERNEL_PY = KERNEL_DIR / "birdclef-2026-b02-e01-v2s-baseline-fold0.py"
KERNEL_IPYNB = KERNEL_DIR / "birdclef-2026-b02-e01-v2s-baseline-fold0.ipynb"
DATASET_HANDLE = "winbeaux/birdclef2026-e01-pkg-ckpt"


# --------------------------------------------------------------------------- #
# 0. Auth
# --------------------------------------------------------------------------- #


def ensure_token() -> None:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    token_path = Path.home() / ".kaggle/access_token"
    if not token_path.is_file():
        sys.exit(
            f"[FATAL] no token: set $KAGGLE_API_TOKEN or place a PAT at {token_path}"
        )
    os.environ["KAGGLE_API_TOKEN"] = token_path.read_text().strip()


# --------------------------------------------------------------------------- #
# 1. Stage payload
# --------------------------------------------------------------------------- #


def stage_payload() -> None:
    print(f"[stage] payload at {PAYLOAD}")
    if PAYLOAD.exists():
        shutil.rmtree(PAYLOAD)
    PAYLOAD.mkdir(parents=True)

    pkg_src = ROOT / "src/birdclef2026"
    pkg_dst = PAYLOAD / "src/birdclef2026"
    shutil.copytree(pkg_src, pkg_dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    shutil.copy(ROOT / "configs/data.yaml", PAYLOAD / "data.yaml")

    ckpt = ROOT / CKPT_REL
    if not ckpt.is_file():
        sys.exit(f"[FATAL] checkpoint missing: {ckpt}")
    shutil.copy(ckpt, PAYLOAD / "swa.pt")

    shutil.copy(
        ROOT / "kaggle/datasets/birdclef2026-eb1-pkg-ckpt/dataset-metadata.json",
        PAYLOAD / "dataset-metadata.json",
    )

    total = sum(p.stat().st_size for p in PAYLOAD.rglob("*") if p.is_file())
    print(f"[stage] {total / 1024 / 1024:.1f} MB total")
    for p in sorted(PAYLOAD.rglob("*"))[:30]:
        if p.is_file():
            print(f"  {p.relative_to(PAYLOAD)}")


# --------------------------------------------------------------------------- #
# 2. Push dataset (kagglehub)
# --------------------------------------------------------------------------- #


def push_dataset() -> None:
    # Some networks (notably university VPN egress like the L40 box) have
    # broken IPv6 routing to GCS upload endpoints — `requests.put` hangs in
    # `sock.connect(sa)` for ~5 min before timing out. Force IPv4 globally
    # before importing kagglehub so its session uses our patched resolver.
    import socket
    _orig_getaddrinfo = socket.getaddrinfo
    def _v4_only(host, port, family=0, *args, **kwargs):
        return _orig_getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)
    socket.getaddrinfo = _v4_only

    import kagglehub

    print(f"[ds] uploading → {DATASET_HANDLE} (IPv4-only)")
    notes = f"e01 fold-0 SWA + package {os.popen('date +%Y-%m-%d_%H%M').read().strip()}"
    kagglehub.dataset_upload(
        handle=DATASET_HANDLE,
        local_dataset_dir=str(PAYLOAD),
        version_notes=notes,
        ignore_patterns=["__pycache__/", "*.pyc", ".git/", "dataset-metadata.json"],
    )
    print(f"[ds] OK → https://www.kaggle.com/datasets/{DATASET_HANDLE}")


# --------------------------------------------------------------------------- #
# 3. Convert .py → .ipynb (delegates to scripts/_py_to_ipynb.py)
# --------------------------------------------------------------------------- #


def convert_notebook() -> None:
    print(f"[nb] {KERNEL_PY.name} → {KERNEL_IPYNB.name}")
    subprocess.check_call([
        sys.executable,
        str(ROOT / "scripts/_py_to_ipynb.py"),
        str(KERNEL_PY),
        str(KERNEL_IPYNB),
    ])


# --------------------------------------------------------------------------- #
# 4. Push kernel (kagglesdk, no legacy kaggle CLI auth bootstrap)
# --------------------------------------------------------------------------- #


def push_kernel() -> None:
    """Push the notebook via the legacy ``kaggle`` CLI.

    The kagglesdk + PAT path returns 403 on get_kernel / 409 on save_kernel
    because the new Personal Access Token format does not carry kernel
    read/write scopes. Kernel push (and ``kaggle competitions submit``) still
    require a legacy ``kaggle.json`` with ``{"username": ..., "key": ...}``.
    """
    kaggle_json = Path.home() / ".kaggle/kaggle.json"
    if not kaggle_json.is_file():
        sys.exit(
            "[FATAL] need legacy ~/.kaggle/kaggle.json for kernel push.\n"
            "        Get one at https://www.kaggle.com/settings → 'API' →\n"
            "        'Create New Token' (downloads kaggle.json with\n"
            "        username + key), drop it in ~/.kaggle/, chmod 600,\n"
            "        and re-run. The dataset above is already uploaded."
        )

    # kaggle CLI authenticates from kaggle.json; remove KAGGLE_API_TOKEN from
    # this subprocess's env so it doesn't try the bearer-auth path that the
    # PAT would mis-trigger.
    env = {k: v for k, v in os.environ.items() if k != "KAGGLE_API_TOKEN"}

    cmd = ["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)]
    print(f"[kernel] $ {' '.join(cmd)}")
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        sys.exit(f"[FATAL] kaggle kernels push exited rc={rc}")
    print(f"[kernel] OK → https://www.kaggle.com/code/{json.loads((KERNEL_DIR / 'kernel-metadata.json').read_text())['id']}")
    print()
    print("Next: open the kernel URL, wait for the run to finish, then click")
    print("      'Submit to Competition' to enter the BirdCLEF 2026 LB.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    kernel_only = "--kernel-only" in sys.argv
    if kernel_only:
        print("[main] --kernel-only: skipping stage + dataset upload")
        convert_notebook()
        push_kernel()
        return
    ensure_token()
    stage_payload()
    push_dataset()
    convert_notebook()
    push_kernel()


if __name__ == "__main__":
    main()
