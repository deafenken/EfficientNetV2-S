# Notebooks

Use this folder for local exploration notebooks. For Kaggle submission, keep the inference path equivalent to:

```bash
PYTHONPATH=src python -m birdclef2026.infer \
  --config configs/baseline.yaml \
  --checkpoint /kaggle/input/YOUR_MODEL_DATASET/best.pt \
  --output /kaggle/working/submission.csv
```

