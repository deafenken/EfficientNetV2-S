# Plan A — 稳健路线：2025 2nd Place 复刻 + 2026 域内监督加成

> 写于 2026-05-08。当前位置 LB 0.929 / rank 704，金牌区 ~0.949+（top ~15）。
> 假设：4×L40 (48 GB×4)，比赛 6/3 截止，剩 ~4 周训练 + 1 周提交磨。

## 目标
- **私榜锚点**：0.943–0.948（稳进银牌，金牌区下沿）。
- **核心策略**：复刻 BirdCLEF 2025 2nd Place（Sydorskyi & Gonçalves，Pri 0.928）的成熟管线，把 2026 新增的 `train_soundscapes_labels.csv` 当一等公民监督。
- **可行性高、踩坑少、能稳定通关**——最低风险路径。

## 为什么走这条路
1. 2025 2nd 的 GitHub 代码完整可复现，有 51 stars、包含 OpenVINO 推理脚本（<https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place>）。
2. 当前仓库训练侧几乎为零（ResNet18 单通道、随机 split、零 augmentation），fork 现成框架比从零自建省 2 周。
3. 2026 比 2025 多一份 `train_soundscapes_labels.csv`（专家在 long soundscape 上的人工标注），把它当主训集而不是 valid，这是公开金牌方案没用过的杠杆。

## 模型族
- 主力 backbone（5-fold each）：
  - `tf_efficientnetv2_s.in21k_ft_in1k`（timm）
  - `eca_nfnet_l0`（timm）
  - **可选第 3 路（diversity）**：`convnext_tiny.in12k_ft_in1k`
- 共用 SED head：Adavanne-style framewise + clipwise + GeM pooling + 512-d FC + dropout 0.25/0.5。
- 推理输出公式：`p = 0.75 * clipwise + 0.25 * time_max(framewise)`（沿用 2023 1st / 2025 2nd）。

## 数据 pipeline
- 重写 `src/birdclef2026/dataset.py` 与 `metadata.py`：
  - **CV**：StratifiedGroupKFold n_splits=5，stratify=primary_label，groups=filename（主作者/录音机若可得用 author）。当前 `metadata.py:188-198` 的 `train_test_split` 必须废掉。
  - **训练 crop**：5 s 随机；推理 5 s 非重叠（60 s → 12 windows）。
  - **Mel**：n_fft=2048, hop=512, n_mels=128, fmin=50, fmax=14000（2026 多昆虫/蛙鸣不能卡 8 kHz），`top_db=80`，全局 mean/std 而不是逐样本归一化（`audio.py:77-79` 的 bug）。
  - **secondary_labels**：权重 0.3 软目标。
  - **类平衡**：sqrt-balancing；少于 100 样本的物种做整数倍复制到 100。
  - **罕见类**整体不进 valid，留训练集里。
- 额外数据：
  - 历届 BirdCLEF 2021–2025 的 `train_audio` 做 backbone 预训练（共 ~819k 录音）。
  - ESC-50 + FSD50K-non-bird + soundscape 静音段作为 background-mix 池。

## 增广（顺序固定，p 标在右）
1. Time shift ±1 s（p=0.5）
2. Background-Mix（ESC-50 / 历届 soundscape 静音段，SNR 0–10 dB，p=0.5）
3. Random gain ±6 dB（p=0.5）
4. **MixUp α=0.5**，label = element-wise max（p=0.5）
5. SpecAugment：time mask 2× width≤30，freq mask 2× width≤16
6. RandomFiltering（模拟麦克风传函）（p=0.3）
7. **Pitch shift / time stretch 不用**（实测害大于益）

## 损失
- 主：`FocalBCE(α=1.0, γ=2.0)` + label smoothing 0.005–0.01。
- 辅：原 BCE 线性组合 0.5 / 0.5 当备选 ablation。
- 类权重：`w_i ∝ (count_i / Σ count)^(-0.5)`。

## 训练配置（4×L40 DDP）
- batch size：每卡 64（NFNet-L0 / ConvNeXt 取 48），全局 256。
- AMP：**bf16**（Ada 上等同 fp16 速度，无需 GradScaler）。
- channels_last + cudnn.benchmark=True；`torch.compile(mode="default")`（CNN 都兼容）。
- 优化器：
  - V2-S / ConvNeXt：AdamW lr=1e-4, wd=1e-2；
  - NFNet-L0：RAdam lr=1e-3。
- LR schedule：CosineAnnealing → 1e-6，warmup 2 epoch。
- Epochs：50；EMA decay=0.999；最后 5 epoch SWA。
- Grad clip：1.0。
- 预计耗时（基于公开 throughput）：
  - V2-S 单 fold ≈ 2 h；3 backbone × 5 fold ≈ 30 h GPU 总（4 卡 DDP 实际墙钟 ~8 h/round）。

## 伪标签（3 轮迭代，关键涨分）
- **Round 0 (R0)**：仅 `train_audio` + `train_soundscapes_labels.csv`（hard label）跑 5-fold base。
- **R1**：用 R0 的 5-fold OOF 平均给 `train_soundscapes` 全量打软标签；过滤 `max(p) < 0.1` 的 chunk（视为 nocall），保留 `max(p) > 0.5` 的；其他做 confidence-weighted soft。
- **R2**：训练集 = labeled hard + R1 soft（1:1 by sample），MixUp 把硬软标签融合。
- **R3**：用 R2 全 ensemble 重打一遍，迭代收敛即停。
- 每轮对 pseudo prob 做 power 变换 `p ← p^1.82`（2025 1st 验证过）。

## 推理（CPU 90 min，internet-off）
- 导出每个模型为 ONNX → OpenVINO fp16。
- 一次 mel 提取共享所有模型。
- Ensemble 公式：
  ```
  p_cnn = 0.4*p_v2s + 0.4*p_nfnet + 0.2*p_convnext
  p_final = 0.5 * p_cnn + 0.5 * rank_mean(p_cnn)   # Quantile-Mix(α=0.5)
  p_final = p_final ** 1.5                          # power calibration
  # TopN(N=1) 后处理：每段把每类概率乘以「该 file 里同类的最大值」
  ```
- 时间预算：5 模型 × 5 fold = 25 个 OpenVINO 推理 ≈ 60 min；mel + I/O ≈ 10 min；后处理 ≈ 5 min。

## 4 周日程
| 周 | 工作 | 产出 |
|---|---|---|
| W1 | 重写 dataset/metadata（GroupKFold, secondary 0.3, fmax=14k）；fork 2025 2nd 框架；EffNetV2-S 单 backbone 单 fold 跑通；建立 OOF 评估闭环（macro AUC 本地复现 LB）。 | OOF baseline ~0.92 |
| W2 | 5-fold V2-S + NFNet-L0；R0 ensemble；OpenVINO 推理脚本；提交 R0。 | LB 0.93+ |
| W3 | R1 + R2 伪标签迭代；加入 ConvNeXt 第三路；每轮提交一次。 | LB 0.94+ |
| W4 | R3 + SWA + power calibration + per-class threshold 搜索；最终 ensemble 调参。 | LB 0.943–0.948 |
| 余 | 留 3 天给 sub day 提交磨/防 OOM/防 90 min 超时。 | 锁定双提交 |

## 风险
- **低**：路径成熟，2025 公开代码仓库可直接复用。
- 主要风险：(1) 2026 类别（234 类）与 2025 (206) 不同，pretrained checkpoint 需重头训 head。(2) Pantanal 区域物种与 xeno-canto 数据分布差异大，靠 `train_soundscapes_labels.csv` 弥补。(3) OpenVINO 跑 5 backbone × 5 fold 可能贴近 90 min 上限——必要时砍到 3 backbone × 3 fold。

## 期望产出
- 私榜 0.943–0.948（top 30–80），银牌锁定，金牌看 shake-up。
- 完整可复用的 4×L40 DDP 训练框架，方便 Plan B/C 叠加。
