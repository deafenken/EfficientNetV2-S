"""Inference, ensemble blending, and submission generation.

Filled in Commit 2:
- ``predict_oof.py``     — per-fold OOF prediction with TTA hooks
- ``predict_test.py``    — test-soundscape prediction (Kaggle CPU path)
- ``ensemble.py``        — Quantile-Mix(α=0.5) + power calibration + TopN postproc
- ``make_submission.py`` — sample-submission-aligned CSV writer
- ``export_onnx.py``     — checkpoint → ONNX → OpenVINO (INT8/FP16)

The legacy flat module ``birdclef2026.infer`` stays at the package root
and is invoked by ``make infer`` / ``make infer-fallback`` until superseded.
"""
