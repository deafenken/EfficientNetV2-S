"""Inference, ensemble blending, and submission generation (Commit 3).

Filled:
- ``predict.py``       — sliding-window test inference + N-checkpoint ensemble.
- ``assemble_oof.py``  — concat per-fold ``oof.npz`` (or ``swa_oof.npz``)
                         into a single cross-validated OOF + macro-AUC.

Planned (Commit 3 follow-ups):
- ``ensemble.py``      — multi-arch / weighted blend across exp dirs.
- ``pseudo_label.py``  — generate soft labels on train_soundscapes from OOF.

The legacy flat module ``birdclef2026.infer`` is retained at package root
for the old ``make infer-fallback`` path; new code should import from
``birdclef2026.inference``.
"""
