# BirdCLEF+ 2026

BirdCLEF+ 2026 开工仓库。第一版目标很朴素：先有一个能本地训练、能在 Kaggle hidden test 上生成合规 `submission.csv` 的 PyTorch baseline，然后再逐步把分数打上去。

## 当前骨架

- `configs/baseline.yaml`: baseline 配置。
- `src/birdclef2026/`: 数据读取、音频特征、模型、训练、推理代码。
- `scripts/download_data.sh`: Kaggle 数据下载脚本。
- `docs/competition_notes.md`: 赛题要点和当前假设。
- `docs/public_lgbm_baseline.md`: 公开 Perch + MLP/LGBM baseline 复现笔记和本地 plus 改动。
- `docs/public_baseline_comparison.md`: 两个公开 Perch baseline 的对比和第一批 ablation。
- `docs/optimization_roadmap.md`: 当前 public score 复盘、A23-A30 的 CPU 合规优化路线和提交门槛。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

下载数据前需要先在 Kaggle 页面加入比赛并接受规则，然后配置 Kaggle API token。

```bash
mkdir -p .kaggle
# 把 kaggle.json 放到 .kaggle/kaggle.json，或设置 KAGGLE_USERNAME / KAGGLE_KEY
chmod 600 .kaggle/kaggle.json
make download
make download-aux
```

检查数据结构：

```bash
PYTHONPATH=src python -m birdclef2026.inspect_data --config configs/baseline.yaml
```

跑一个小样本冒烟训练：

```bash
PYTHONPATH=src python -m birdclef2026.train --config configs/baseline.yaml --debug
```

正式 baseline 训练：

```bash
PYTHONPATH=src python -m birdclef2026.train --config configs/baseline.yaml
```

生成提交文件：

```bash
PYTHONPATH=src python -m birdclef2026.infer \
  --config configs/baseline.yaml \
  --checkpoint outputs/baseline/best.pt \
  --output submission.csv
```

准备公开 LGBM plus Kaggle notebook：

```bash
KAGGLE_USERNAME=your_kaggle_name make prepare-kaggle-kernel
make push-kaggle-kernel
make kernel-status
```

## 路线

1. 先验证提交闭环：数据、训练、Kaggle notebook 推理、`submission.csv` 格式。
2. 复现公开 Perch + MLP/LGBM + ProtoSSM baseline，并以 `version_3_lgbm_plus.ipynb` 做第一版增强。
3. 用 `train_audio` 训练 5 秒 log-mel + ResNet18 的稳健 baseline，作为异构 ensemble 成员。
4. 加入 `train_soundscapes_labels.csv`、分层 CV、class-balanced sampling 和弱标签处理。
5. 替换 backbone：EfficientNet / ConvNeXt / BEATs / AST，配合 mixup、specaugment、long audio pooling。
6. 做 public notebooks/discussions 追踪，提炼社区已确认的坑和高分技巧。

当前更具体的优化执行顺序见：

```bash
docs/optimization_roadmap.md
```
