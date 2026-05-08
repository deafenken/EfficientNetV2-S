"""Cross-cutting helpers (audio, config, metrics, IO).

In Commit 1 this is a placeholder. The legacy flat modules

- ``birdclef2026.audio``     (mel feature extraction)
- ``birdclef2026.config``    (YAML config loader)
- ``birdclef2026.metrics``   (macro AUC)
- ``birdclef2026.utils``     (seed / device / ensure_dir)

still live at the package root and continue to function. Commit 2 will move
them under this subpackage and add:

- ``cv.py``         — StratifiedGroupKFold helpers + fold-csv I/O
- ``mel_stats.py``  — global mean/std computation over the training set
- ``io.py``         — checkpoint, OOF, log writers
"""
