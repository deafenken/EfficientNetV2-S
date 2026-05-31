"""Single Kaggle credential resolver (split-stack auth).

BirdCLEF-2026 needs TWO Kaggle auth paths and they are not interchangeable:

  * Dataset upload (``kagglehub.dataset_upload``) honours the new Personal
    Access Token (PAT): ``$KAGGLE_API_TOKEN`` or ``~/.kaggle/access_token``.
  * Kernel push (``kaggle kernels push``) and ``kaggle competitions submit``
    need the LEGACY ``~/.kaggle/kaggle.json`` (``{"username", "key"}``). The
    PAT format does NOT carry the kernels.write / kernels.get scopes, so the
    kagglesdk path returns 409 on save_kernel and 403 on get_kernel. The fix is
    the legacy json (kaggle.com/settings → API → Create New Token).

Credentials live ONLY under ``~/.kaggle/`` (``chmod 600``), never in the repo
tree. See docs/SUBMISSION_WORKFLOW.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PAT_ENV = "KAGGLE_API_TOKEN"
PAT_FILE = Path.home() / ".kaggle/access_token"
LEGACY_JSON = Path.home() / ".kaggle/kaggle.json"


def ensure_pat_token() -> str:
    """Ensure ``$KAGGLE_API_TOKEN`` is set (for ``kagglehub.dataset_upload``).

    Precedence: ``$KAGGLE_API_TOKEN`` env → ``~/.kaggle/access_token`` file.
    """
    tok = os.environ.get(PAT_ENV)
    if tok:
        return tok
    if not PAT_FILE.is_file():
        sys.exit(
            f"[FATAL] no Kaggle PAT: set ${PAT_ENV} or place a token at {PAT_FILE}\n"
            f"        (kaggle.com/settings → API → 'Create New Token')."
        )
    tok = PAT_FILE.read_text().strip()
    os.environ[PAT_ENV] = tok
    return tok


def require_legacy_json() -> Path:
    """Return ``~/.kaggle/kaggle.json`` for kernel push; exit if missing."""
    if not LEGACY_JSON.is_file():
        sys.exit(
            "[FATAL] need legacy ~/.kaggle/kaggle.json for kernel push.\n"
            "        kaggle.com/settings → API → 'Create New Token' downloads a\n"
            "        kaggle.json with username+key; drop it in ~/.kaggle/, chmod 600.\n"
            "        The PAT cannot push kernels (missing kernels scope)."
        )
    return LEGACY_JSON


def push_env() -> dict[str, str]:
    """Env for the ``kaggle kernels push`` subprocess.

    Strip ``$KAGGLE_API_TOKEN`` so the CLI authenticates from kaggle.json
    instead of the bearer-auth path the PAT mis-triggers.
    """
    return {k: v for k, v in os.environ.items() if k != PAT_ENV}
