"""Training entrypoints and loop utilities.

Filled in Commit 2:
- ``train_ddp.py`` — torchrun-compatible main entry (bf16 + DDP + EMA + cosine + warmup + FocalBCE)
- ``ema.py``       — exponential moving average shadow weights
- ``swa.py``       — last-N-epoch SWA averaging
- ``pseudo.py``    — single-round noisy-student loop

The legacy flat module ``birdclef2026.train`` still exists at the package
root and is invoked by ``make train`` / ``make debug-train``; it will be
replaced by ``training.train_ddp`` in Commit 2.
"""
