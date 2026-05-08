"""Model definitions: backbones, SED head, losses.

Filled in Commit 2:
- ``backbones.py`` — timm wrappers (EfficientNetV2-S, NFNet-L0, ConvNeXt-T)
- ``sed.py``       — Adavanne-style framewise + clipwise + attention pooling
- ``losses.py``    — Focal BCE + label smoothing + secondary-label reweighting

The legacy flat module ``birdclef2026.model`` (ResNet18-only) stays at the
package root until Commit 2 supersedes it.
"""
