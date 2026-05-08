"""Data loading, augmentation, and CV-fold utilities.

Filled in Commit 2:
- ``dataset.py``    — SED-friendly multi-label Dataset (secondary 0.3, group sampling)
- ``transforms.py`` — mel + augmentation stack (MixUp(label-max), SpecAug, BgMix)
- ``folds.py``      — StratifiedGroupKFold by recording id

The legacy flat module ``birdclef2026.dataset`` still exists at the package
root and stays functional until Commit 2 deprecates it.
"""
