# Plan B — 猛冲金牌：Multi-Iterative Noisy Student × Perch v2 KD 双引擎

> 写于 2026-05-08。基于 Plan A 完成稳健基线后**继续向上爆杆**的高风险高回报路线。
> 假设：4×L40 (48 GB×4)；4 周训练 + 1 周提交磨；可接受 ≥80 GPU·h/round 的伪标迭代。

## 目标
- **私榜锚点**：0.948–0.952（金牌区，top 5–15 真实尝试）。
- **核心策略**：复刻 BirdCLEF 2025 1st (Babych) 的 **Multi-Iterative Noisy Student**（4 轮自蒸馏、6+ backbone 池），叠加 Perch v2 1536-d embedding 的 KD 教师 + 长上下文 head 当独立分支。
- 高复杂度，但是公开数据集允许的天花板。

## 为什么走这条路
1. 公开 LB 当前最高 0.925（Perch+ProtoSSM），离金牌还差 0.024。Plan A 复刻 25 年 2nd 期望 0.943–0.948——刚好压在金/银边缘。要稳进金，必须叠加 25 年 1st 的多轮 NS + 跨家族 ensemble。
2. Perch 2.0（EfficientNet-B3 backbone，1536-d, 32 kHz, 5 s）已经是 BirdSet AUC 0.908，把它当**软标签教师**对 unlabeled soundscape 一次性灌输，比单纯自蒸馏快 2–3 倍收敛。
3. `train_soundscapes_labels.csv` 在 R0 阶段就能给"in-domain hard label"，破坏 xeno-canto → soundscape 的域 gap。

## 模型族（最终 ensemble）
| 路线 | Backbone | 数量 | 输入 | 角色 |
|---|---|---|---|---|
| CNN-SED | `tf_efficientnetv2_s.in21k_ft_in1k` | 4 (4 seeds × 5 fold) | 5 s mel 128×312 | 主力 |
| CNN-SED | `tf_efficientnet_b3_ns` | 3 | 5 s mel | 主力 |
| CNN-SED | `tf_efficientnetv2_b3` | 3 | 5 s mel | 主力 |
| CNN-SED | `tf_efficientnet_b0_ns` | 2 | 5 s mel | **伪标教师**（小模型不过拟合） |
| Long-ctx CNN | `eca_nfnet_l0` | 2 | **30 s mel** crop | 长上下文 diversity |
| Freq-axis SED | `convnext_tiny` 改 freq-attention | 2 | 5 s mel | trick 加成 |
| Perch-KD | Perch 2.0 frozen → 2-layer Transformer head (d=256) | 1 | 12×5 s = 60 s | KD diversity 支线 |

最终推理：14–18 模型 ensemble，CPU 90 min 内塞下需做 **OpenVINO INT8 + ONNX shared mel**。

## SED head 细节
- 主 head：Adavanne framewise + clipwise + GeM + 512-d FC。
- **Frequency-axis SED 变体**（2025 25th 暗示有效）：把 attention pooling 从 time 维换到 freq 维，单独一路 backbone。
- 输出：`p = 0.75 * clipwise + 0.25 * time_max(framewise)`，加 segment-level Gaussian smoothing (σ=1.0, window=3) 跨 60 s。

## 数据 pipeline（沿用 Plan A 但多两层）
- StratifiedGroupKFold n=5，groups=author 优先 / filename 兜底。
- secondary_labels 权重 0.3。
- 多 crop：5 s 主、30 s（5th place）、60 s（长 context 头）。
- 历届 BirdCLEF 21–25 数据 + Xeno-Canto Pantanal 邻近物种全量预训练 ~819k 录音（用 EffNetV2-S 单 backbone 跑 1 个 epoch 当 init）。
- **Silero-VAD** 删 train 中人声段（2025 top-2% 关键发现）。
- Background pool：ESC-50 + FSD50K-non-bird + 历届 soundscape nocall。

## 增广（每路线略调）
- 主路线同 Plan A。
- 伪标教师（B0-NS）增广更弱：仅 SpecAug + gain，不用 MixUp（避免标签噪声放大）。
- Long-ctx 头：30 s crop + 5 s 子窗 attention + RandFiltering 强（p=0.5）。

## 损失
- FocalLoss(α=1.0, γ=2.0) 全员；label smoothing 0.005。
- KD 分支：student loss = 0.5 * BCE(hard) + 0.5 * KLDiv(student, Perch_soft@T=2)。

## Multi-Iterative Noisy Student 主循环（4 轮自蒸馏）
- **R0**：labeled (`train_audio` + `train_soundscapes_labels.csv` hard) → train 全 backbone 池。
- **R1**：R0 的 mean ensemble 给所有 `train_soundscapes` 打软标 → 过滤 `max(p) < 0.1` 清零；剩余做 power 变换 `p ← p^1.82`。
- **R2**：训练集 = 50% labeled + 50% pure pseudo（**Babych 配方**），重训整个池子。Input noise: SpecAug+MixUp+BgMix；Model noise: dropout 0.5 + stochastic depth 0.2。
- **R3**：R2 ensemble 再打软标，重训。
- **R4**：第 4 轮以 small-model（B0-NS/V2-B3）当 student，蒸馏到 ensemble teacher（V2-S+NFNet-L0），输出最终 ensemble。
- 关键：**OOF-based teacher 选择**（不用 LB 调），每轮看 5-fold 平均 macro AUC。

## Perch v2 KD 支线（独立训练，diversity 100%）
1. 用 `google-research/perch` TF/JAX checkpoint 冻结取 1536-d embedding（5 s 平均池化输出）。
2. 12 个 5 s 窗 → (12, 1536) 序列 → 2 层 Transformer (d=256, h=4) → 234-class logits。
3. 训练数据：labeled hard + R3 ensemble pseudo。
4. 损失：BCE + Focal。
5. AdamW lr=1e-3 (head only)，bs=256（embedding 已 cache 到 disk），50 epoch < 1 h on L40。
6. 推理：Perch TFLite ~16 min/600 文件 + Transformer head ~3 min。

## 训练配置（4×L40 DDP）
- 主 CNN：每卡 BS=64，全局 256；bf16；AdamW lr=1e-4 cosine→1e-6，warmup 5%。
- B0-NS：每卡 BS=128，全局 512。
- Long-ctx (30 s)：每卡 BS=16，全局 64；grad accumulation 4。
- Perch-KD head：每卡 BS=256，全局 1024（embedding cache）。
- `find_unused_parameters=False`，channels_last，`torch.compile`（CNN）。
- EMA 0.999；SWA 最后 5 epoch；Grad clip 1.0；50 epoch（每轮 NS 30–40 epoch）。
- **预计 GPU·h**（基于 ~700 img/s/L40 V2-S）：
  - R0 7 backbone × 5 fold ≈ 35 fold × 2 h ≈ 70 GPU·h ≈ 18 h 4-GPU 墙钟。
  - R1–R4 每轮同量级 × 30/50 epoch 折扣 ≈ 12 h × 4 = 48 h 墙钟。
  - Perch KD ≈ 3 h。
  - **总计 ~70 h 墙钟**，4 周内可塞 1 套主流程 + 1 次重训。

## 推理（CPU 90 min，关键工程挑战）
- 14–18 模型 ensemble 不能直接跑，必须：
  1. ONNX → **OpenVINO INT8**（V2-S/B0/B3-NS 都验证过 INT8 稳）；NFNet-L0/ConvNeXt 留 fp16。
  2. **Mel 一次提取**给所有 CNN 共享。
  3. Perch TFLite 单独跑（已有 ~16 min benchmark）。
  4. 长 context 头单独跑（30 s 输入，~5 min）。
- Ensemble 公式（多层）：
  ```python
  # CNN 池子内部
  p_v2s   = mean(4 seeds × 5 fold V2-S)
  p_b3ns  = mean(3 × B3-NS)
  p_v2b3  = mean(3 × V2-B3)
  p_b0    = mean(2 × B0-NS)
  p_nfnet = mean(2 × NFNet-L0 long-ctx)
  p_freq  = mean(2 × ConvNeXt freq-axis)

  p_cnn = (0.30*p_v2s + 0.20*p_b3ns + 0.15*p_v2b3
          + 0.10*p_b0 + 0.15*p_nfnet + 0.10*p_freq)
  p_perch = perch_ssm_head_output

  # Quantile-Mix
  p_main = 0.5 * mean(p_cnn, p_perch) + 0.5 * rank_mean(p_cnn, p_perch)
  p_final = p_main ** 1.5

  # TopN(N=1) per-file post + Gaussian time smoothing(σ=1)
  ```
- 时间预算（极限）：
  - INT8 V2-S/B0/B3 12 模型 × 600 file × 12 win ≈ 35 min
  - fp16 NFNet/ConvNeXt 4 模型 ≈ 18 min
  - Perch TFLite ≈ 16 min
  - mel + IO + 后处理 ≈ 12 min
  - **合计 ~81 min**，剩 9 min buffer。如果超，砍 freq-axis 或 30s long-ctx 一路。

## 5 周日程
| 周 | 工作 | 产出 |
|---|---|---|
| W1 | 完成 Plan A 的 dataset/loader/SED head 框架；EffNetV2-S 单 backbone 跑通 R0 单 fold；OOF 评估闭环；预训练历届数据 1 epoch。 | infra ready |
| W2 | R0 完成（7 backbone × 5 fold），开始 R1 伪标。Perch embedding 全量 cache，KD head 训练。OpenVINO INT8 量化 pipeline。 | LB 0.935 |
| W3 | R1 + R2 完成，加 long-ctx 30 s 分支和 freq-axis ConvNeXt。每轮提交。 | LB 0.943 |
| W4 | R3 + R4 蒸馏；ensemble 权重 Optuna；per-class threshold；TopN/power calibration 调参。 | LB 0.948+ |
| W5 | 推理时长压榨；二备份提交；shake-up 防御（保 Plan A 当 anchor 提交）。 | 双提交锁定 |

## 风险
- **高**：14+ 模型 ensemble 在 90 min CPU 上必须工程极致，任何一路挂了就要砍枝；INT8 量化某些 backbone 会掉 ~0.001-0.002。
- 复杂度：4 轮 NS + KD + Perch + 长 ctx + freq-axis = 6 个独立子流程，单人 4 周很紧。
- Confirmation bias：≥3 轮自蒸馏会强化 R0 错误，必须每轮看 OOF 决定是否继续。
- 单点：如果 Perch v2 在 Pantanal 物种上迁移效果差（Perch 2.0 训练数据偏北美），KD 支线收益会被打折，预算 -0.005。

## 缓解
- 把 Plan A 的最后 ensemble 当**始终保留的备份提交**（Kaggle 两个 final selection 槽位，A 一枚 + B 一枚）。
- 每轮 NS 训练前先在 100 物种子集上 30 epoch 验证 ensemble 走向。
- 如果 R3/R4 的 OOF 不再上升，提前停止，节省时间做推理工程。

## 期望产出
- 私榜 0.948–0.952（top 5–15），真正的金牌尝试。
- 复用了 Plan A 的全部 infra，5 周内可达。
