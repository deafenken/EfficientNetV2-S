#!/usr/bin/env python3
"""End-to-end Kaggle submission for eB1 NFNet-L0 fold-0.

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

Auth:
  - Reads ``~/.kaggle/access_token`` (the user's PAT) and exports
    ``KAGGLE_API_TOKEN``. Both ``kagglehub`` and ``kagglesdk`` honor it.
  - The legacy ``kaggle`` CLI is not used (it can't auth with this PAT).

Usage:
  cd birdclef-2026 && uv run python scripts/17_kaggle_eB1_submit.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD = Path("/tmp/birdclef2026-eb1-pkg-ckpt")
CKPT_REL = "outputs/exp/eB1_nfnet_l0/fold_0/swa.pt"
KERNEL_DIR = ROOT / "kaggle/variants/b01_eB1_nfnet_l0"
KERNEL_PY = KERNEL_DIR / "birdclef-2026-b01-eb1-nfnet-l0.py"
KERNEL_IPYNB = KERNEL_DIR / "birdclef-2026-b01-eb1-nfnet-l0.ipynb"
DATASET_HANDLE = "winbeaux/birdclef2026-eb1-pkg-ckpt"


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
    notes = f"eB1 fold-0 SWA + package {os.popen('date +%Y-%m-%d_%H%M').read().strip()}"
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
    from kagglesdk import KaggleClient
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest

    meta = json.loads((KERNEL_DIR / "kernel-metadata.json").read_text())
    nb_text = KERNEL_IPYNB.read_text()
    # Server expects a single source string per cell, not a list.
    nb_obj = json.loads(nb_text)
    for cell in nb_obj.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])
    nb_text = json.dumps(nb_obj)

    req = ApiSaveKernelRequest()
    req.id = meta.get("id_no")
    req.slug = meta["id"]
    req.new_title = meta["title"]
    req.text = nb_text
    req.language = meta["language"]
    req.kernel_type = meta["kernel_type"]
    req.is_private = bool(meta.get("is_private", True))
    req.enable_gpu = bool(meta.get("enable_gpu", False))
    req.enable_tpu = bool(meta.get("enable_tpu", False))
    req.enable_internet = bool(meta.get("enable_internet", False))
    req.dataset_data_sources = meta.get("dataset_sources", [])
    req.competition_data_sources = meta.get("competition_sources", [])
    req.kernel_data_sources = meta.get("kernel_sources", [])
    req.model_data_sources = meta.get("model_sources", [])
    req.category_ids = meta.get("keywords", [])

    print(f"[kernel] pushing {meta['id']}")
    with KaggleClient(api_token=os.environ["KAGGLE_API_TOKEN"]) as client:
        resp = client.kernels.kernels_api_client.save_kernel(req)
    if getattr(resp, "_error", None):
        sys.exit(f"[FATAL] kernel push failed: {resp._error}")
    # KaggleObject exposes its fields as descriptor-backed attributes (no ()).
    url = getattr(resp, "url", None) or f"https://www.kaggle.com/code/{meta['id']}"
    version = getattr(resp, "version_number", None)
    print(f"[kernel] OK → {url}")
    print(f"[kernel] version = {version}")
    print()
    print("Next: open the kernel URL, wait for the run to finish, then click")
    print("      'Submit to Competition' to enter the BirdCLEF 2026 LB.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ensure_token()
    stage_payload()
    push_dataset()
    convert_notebook()
    push_kernel()


if __name__ == "__main__":
    main()
