#!/usr/bin/env python3
"""Generic Kaggle submission driver for BirdCLEF 2026 LB-verification runs.

This is a parametrized replacement for ``17_kaggle_eB1_submit.py`` /
``18_kaggle_e01_submit.py``. Each variant (``b01``..``b05``) is registered
in :data:`VARIANTS` below; the rest of the pipeline (stage payload →
upload dataset → convert .py → push kernel) is identical.

Usage:
  cd birdclef-2026
  uv run python scripts/22_kaggle_submit.py --variant b03_eB1b_nfnet_bgnoise
  uv run python scripts/22_kaggle_submit.py --variant b04_eB1c_nfnet_pseudo --kernel-only

The four pipeline steps (~3-5 min once the access token is in place):
  1. Stage src/birdclef2026/ + configs/data.yaml + outputs/exp/<exp>/fold_0/swa.pt
     into the variant's payload dir under /tmp/.
  2. ``kagglehub.dataset_upload`` to ``winbeaux/<dataset-slug>`` (creates on
     first run, versions on subsequent runs).
  3. Convert the jupytext .py → .ipynb via ``scripts/_py_to_ipynb.py``.
  4. ``kaggle kernels push`` the kernel (kernel runs autonomously on Kaggle
     once pushed; user clicks "Submit to Competition" on the run output).

Auth (split-stack: PAT for kagglehub, legacy json for kaggle CLI):
  - Dataset upload uses ``kagglehub.dataset_upload`` which honors
    ``KAGGLE_API_TOKEN`` (auto-loaded from ``~/.kaggle/access_token``).
  - Kernel push uses the legacy ``kaggle kernels push`` CLI, which needs
    ``~/.kaggle/kaggle.json`` (``{"username": ..., "key": ...}``). PAT
    tokens do not carry the kernels.write / kernels.get scopes, so
    kagglesdk's ``save_kernel`` returns 409 on first push and ``get_kernel``
    returns 403. The fix is the legacy json — get one at
    https://www.kaggle.com/settings → 'API' → 'Create New Token'.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Variant registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Variant:
    """A single LB-verification submission variant.

    Attributes
    ----------
    name :
        Stable variant slug used on the CLI (matches the kernel dir name).
    exp_ckpt_path :
        Path to the SWA checkpoint relative to the repo root. Must exist
        on the machine running this script (i.e. wherever training output
        was synced to).
    payload_dir :
        Local staging dir (under /tmp). Will be wiped + recreated each run.
    dataset_handle :
        ``<owner>/<slug>`` for the Kaggle dataset (created on first push,
        versioned thereafter). The matching ``dataset-metadata.json`` lives
        at ``kaggle/datasets/<slug>/dataset-metadata.json``.
    kernel_dir :
        Path to the kernel dir under ``kaggle/variants/`` (contains the
        ``kernel-metadata.json`` and the jupytext .py source).
    notes_label :
        Short label embedded in the dataset version notes (e.g. "eB1b
        fold-0 SWA + package").
    """

    name: str
    exp_ckpt_path: str
    payload_dir: Path
    dataset_handle: str
    kernel_dir: Path
    notes_label: str
    # When True, skip stage_payload + push_dataset entirely (the variant has
    # no ckpt of its own — it only pushes a kernel that consumes other
    # kernels' outputs via kernel_sources, e.g. a blend kernel).
    kernel_only: bool = False
    # Extra files (path relative to repo root → destination basename in
    # the payload) that should be copied alongside swa.pt / data.yaml.
    # b07 needs `nfnet_oof_auc.npy` shipped into the eB1b pkg so the
    # kernel can per-class-weight NFNet's RRF contribution.
    extra_payload: tuple[tuple[str, str], ...] = ()


VARIANTS: dict[str, Variant] = {
    "b01_eB1_nfnet_l0": Variant(
        name="b01_eB1_nfnet_l0",
        exp_ckpt_path="outputs/exp/eB1_nfnet_l0/fold_0/swa.pt",
        payload_dir=Path("/tmp/birdclef2026-eb1-pkg-ckpt"),
        dataset_handle="winbeaux/birdclef2026-eb1-pkg-ckpt",
        kernel_dir=ROOT / "kaggle/variants/b01_eB1_nfnet_l0",
        notes_label="eB1 fold-0 SWA + package",
    ),
    "b02_e01_v2s_baseline": Variant(
        name="b02_e01_v2s_baseline",
        exp_ckpt_path="outputs/exp/e01_v2s_5fold/fold_0/swa.pt",
        payload_dir=Path("/tmp/birdclef2026-e01-pkg-ckpt"),
        dataset_handle="winbeaux/birdclef2026-e01-pkg-ckpt",
        kernel_dir=ROOT / "kaggle/variants/b02_e01_v2s_baseline",
        notes_label="e01 fold-0 SWA + package",
    ),
    "b03_eB1b_nfnet_bgnoise": Variant(
        name="b03_eB1b_nfnet_bgnoise",
        exp_ckpt_path="outputs/exp/eB1b_nfnet_bgnoise/fold_0/swa.pt",
        payload_dir=Path("/tmp/birdclef2026-eb1b-pkg-ckpt"),
        dataset_handle="winbeaux/birdclef2026-eb1b-pkg-ckpt",
        kernel_dir=ROOT / "kaggle/variants/b03_eB1b_nfnet_bgnoise",
        notes_label="eB1b fold-0 SWA + package",
    ),
    "b04_eB1c_nfnet_pseudo": Variant(
        name="b04_eB1c_nfnet_pseudo",
        exp_ckpt_path="outputs/exp/eB1c_nfnet_pseudo/fold_0/swa.pt",
        payload_dir=Path("/tmp/birdclef2026-eb1c-pkg-ckpt"),
        dataset_handle="winbeaux/birdclef2026-eb1c-pkg-ckpt",
        kernel_dir=ROOT / "kaggle/variants/b04_eB1c_nfnet_pseudo",
        notes_label="eB1c fold-0 SWA + package",
    ),
    "b05_e03_v2s_fixed": Variant(
        name="b05_e03_v2s_fixed",
        exp_ckpt_path="outputs/exp/e03_v2s_fixed/fold_0/swa.pt",
        payload_dir=Path("/tmp/birdclef2026-e03-pkg-ckpt"),
        dataset_handle="winbeaux/birdclef2026-e03-pkg-ckpt",
        kernel_dir=ROOT / "kaggle/variants/b05_e03_v2s_fixed",
        notes_label="e03 fold-0 SWA + package",
    ),
    # b06 has no ckpt — it's a blend-only kernel that reads two existing
    # kernel outputs (A34 + b03 eB1b) via kernel_sources and rank-averages
    # them. `kernel_only=True` short-circuits stage_payload + push_dataset.
    "b06_a34_eB1b_blend": Variant(
        name="b06_a34_eB1b_blend",
        exp_ckpt_path="",
        payload_dir=Path("/tmp/_b06_unused"),
        dataset_handle="",
        kernel_dir=ROOT / "kaggle/variants/b06_a34_eB1b_blend",
        notes_label="b06 A34+eB1b rank-blend",
        kernel_only=True,
    ),
    # b07 = c0 deep ensemble. Reuses the eB1b pkg dataset; we re-version it
    # to add `nfnet_oof_auc.npy` alongside the existing swa.pt + src/ +
    # data.yaml. The kernel itself forks A34 (cells 0-39) and inlines our
    # NFNet inference + 4-way RRF + R1-R5 blend in cell F.
    "b07_c0_deep_ensemble": Variant(
        name="b07_c0_deep_ensemble",
        exp_ckpt_path="outputs/exp/eB1b_nfnet_bgnoise/fold_0/swa.pt",
        payload_dir=Path("/tmp/birdclef2026-eb1b-pkg-ckpt"),
        dataset_handle="winbeaux/birdclef2026-eb1b-pkg-ckpt",
        kernel_dir=ROOT / "kaggle/variants/b07_c0_deep_ensemble",
        notes_label="eB1b SWA + nfnet_oof_auc.npy (c0 deep ensemble)",
        extra_payload=(
            (
                "outputs/exp/eB1b_nfnet_bgnoise/fold_0/nfnet_oof_auc.npy",
                "nfnet_oof_auc.npy",
            ),
        ),
    ),
}


def _kernel_files(variant: Variant) -> tuple[Path, Path]:
    """Locate the variant's .py source and resolve the matching .ipynb path.

    The kernel dir is expected to contain exactly one ``*.py`` jupytext
    source whose name matches the kernel's ``code_file`` once converted to
    .ipynb. We auto-discover it instead of hard-coding so adding a variant
    is a one-line change in :data:`VARIANTS`.
    """
    py_files = sorted(p for p in variant.kernel_dir.glob("*.py") if not p.name.startswith("_"))
    if len(py_files) != 1:
        sys.exit(
            f"[FATAL] {variant.kernel_dir} must contain exactly one .py jupytext "
            f"source; found {len(py_files)}: {[p.name for p in py_files]}"
        )
    py = py_files[0]
    ipynb = py.with_suffix(".ipynb")
    return py, ipynb


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


def stage_payload(variant: Variant) -> None:
    payload = variant.payload_dir
    print(f"[stage] payload at {payload}")
    if payload.exists():
        shutil.rmtree(payload)
    payload.mkdir(parents=True)

    pkg_src = ROOT / "src/birdclef2026"
    pkg_dst = payload / "src/birdclef2026"
    shutil.copytree(pkg_src, pkg_dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    shutil.copy(ROOT / "configs/data.yaml", payload / "data.yaml")

    ckpt = ROOT / variant.exp_ckpt_path
    if not ckpt.is_file():
        sys.exit(f"[FATAL] checkpoint missing: {ckpt}")
    shutil.copy(ckpt, payload / "swa.pt")

    # Extra files declared on the Variant (e.g. b07 ships nfnet_oof_auc.npy
    # alongside swa.pt). Each entry is (src_rel_to_root, dst_basename).
    for src_rel, dst_name in variant.extra_payload:
        src = ROOT / src_rel
        if not src.is_file():
            sys.exit(f"[FATAL] extra_payload missing: {src}")
        shutil.copy(src, payload / dst_name)

    # Each variant's dataset-metadata.json lives next to this script's
    # repo at kaggle/datasets/<slug>/. We resolve it from dataset_handle so
    # adding a variant doesn't require a path constant here.
    ds_slug = variant.dataset_handle.split("/", 1)[1]
    ds_meta = ROOT / "kaggle/datasets" / ds_slug / "dataset-metadata.json"
    if not ds_meta.is_file():
        sys.exit(f"[FATAL] dataset-metadata.json missing: {ds_meta}")
    shutil.copy(ds_meta, payload / "dataset-metadata.json")

    total = sum(p.stat().st_size for p in payload.rglob("*") if p.is_file())
    print(f"[stage] {total / 1024 / 1024:.1f} MB total")
    for p in sorted(payload.rglob("*"))[:30]:
        if p.is_file():
            print(f"  {p.relative_to(payload)}")


# --------------------------------------------------------------------------- #
# 2. Push dataset (kagglehub)
# --------------------------------------------------------------------------- #


def push_dataset(variant: Variant) -> None:
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

    print(f"[ds] uploading → {variant.dataset_handle} (IPv4-only)")
    notes = f"{variant.notes_label} {os.popen('date +%Y-%m-%d_%H%M').read().strip()}"
    kagglehub.dataset_upload(
        handle=variant.dataset_handle,
        local_dataset_dir=str(variant.payload_dir),
        version_notes=notes,
        ignore_patterns=["__pycache__/", "*.pyc", ".git/", "dataset-metadata.json"],
    )
    print(f"[ds] OK → https://www.kaggle.com/datasets/{variant.dataset_handle}")
    # Kaggle marks a freshly-uploaded dataset as "processing" for ~30-90s
    # before files are queryable. If we push the kernel during that window,
    # `kaggle kernels push` silently drops the dataset_source with a
    # warning ("not valid dataset sources") and the kernel runs without
    # the ckpt → fail-soft to all-zero submission. Poll until ready.
    _wait_for_dataset_ready(variant.dataset_handle)


def _wait_for_dataset_ready(handle: str, timeout_s: int = 600, poll_s: int = 10) -> None:
    """Poll ``kaggle datasets status`` until the dataset is queryable.

    The status command exits 0 with text like "complete" once Kaggle has
    finished file processing. Errors exit non-zero (we treat as transient
    and keep polling up to ``timeout_s``). Without this, kernel push races
    the dataset processor and silently strips the dataset_source.
    """
    import time
    print(f"[ds] waiting for {handle} to finish processing…")
    deadline = time.monotonic() + timeout_s
    env = {k: v for k, v in os.environ.items() if k != "KAGGLE_API_TOKEN"}
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
# 3. Convert .py → .ipynb (delegates to scripts/_py_to_ipynb.py)
# --------------------------------------------------------------------------- #


def convert_notebook(variant: Variant) -> None:
    kernel_py, kernel_ipynb = _kernel_files(variant)
    print(f"[nb] {kernel_py.name} → {kernel_ipynb.name}")
    subprocess.check_call([
        sys.executable,
        str(ROOT / "scripts/_py_to_ipynb.py"),
        str(kernel_py),
        str(kernel_ipynb),
    ])


# --------------------------------------------------------------------------- #
# 4. Push kernel (legacy kaggle CLI; PAT path returns 403/409 on kernels)
# --------------------------------------------------------------------------- #


def push_kernel(variant: Variant) -> None:
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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--variant", required=True, choices=sorted(VARIANTS.keys()),
        help="Which submission variant to push (see VARIANTS in this file).",
    )
    parser.add_argument(
        "--kernel-only", action="store_true",
        help="Skip stage + dataset upload; only convert .py → .ipynb and push the kernel.",
    )
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    print(f"[main] variant: {variant.name}")
    print(f"[main] dataset: {variant.dataset_handle or '(none — blend-only kernel)'}")
    print(f"[main] kernel : {variant.kernel_dir.relative_to(ROOT)}")

    if args.kernel_only or variant.kernel_only:
        if variant.kernel_only:
            print("[main] kernel-only variant: skipping stage + dataset upload")
        else:
            print("[main] --kernel-only: skipping stage + dataset upload")
        convert_notebook(variant)
        push_kernel(variant)
        return

    ensure_token()
    stage_payload(variant)
    push_dataset(variant)
    convert_notebook(variant)
    push_kernel(variant)


if __name__ == "__main__":
    main()
