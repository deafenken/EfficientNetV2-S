# BirdCLEF+ 2026 三套金牌方案索引

> 写于 2026-05-08，基于当前 LB 0.929 / rank 704、目标金牌区 ≥0.949。
> 调研依据：4 个并行 agent 调研产物（2026 公开 LB / 历届金牌 / 模型架构与 L40 配方 / 仓库锐评）。

## 现状一句话总结
仓库目前是「公开 notebook 拼装的二流工作台」——`src/birdclef2026/` 只有 ResNet18 单通道 + 随机 split + 零增广，A0–A41 全部在调别人的 blend 权重。**4×L40 完全没用上**。要冲金，必须先承认 blend 路死了，开始自训强模型。详见仓库锐评：模型 head 缺、CV 泄漏、loss 太朴素、零伪标签。

## 三套方案对比

| 维度 | Plan A 稳健 | Plan B 猛冲 | Plan C 差异化 |
|---|---|---|---|
| **目标分数** | 0.943–0.948 | 0.948–0.952 | 0.945–0.950 |
| **金牌概率** | ~25% | ~45% | ~35% |
| **银牌保底** | ✅ 高 | ⚠️ 看伪标收敛 | ⚠️ 看跨家族迁移 |
| **核心思路** | 复刻 2025 2nd | Multi-Iter Noisy Student × Perch KD | 跨家族 ensemble (CNN+Transformer+Perch) |
| **Backbone 数** | 2–3（同家族） | 14–18（深度迭代） | 14（4 路跨家族） |
| **伪标轮数** | 3 轮 | 4 轮（含 Perch KD） | 1–2 轮 |
| **训练 GPU·h** | ~30–40 | ~70 | ~50–60 |
| **推理时长** | ~75 min | ~81 min（紧） | ~84 min（最紧） |
| **复杂度** | 低 | 高 | 中 |
| **Shake-up 抗性** | 中 | 中（伪标可能过拟合） | **高**（跨家族 diversity） |
| **代码复用** | fork 2025 2nd 直接改 | A 完成后增量叠加 | A 完成后并行加路 |
| **周期** | 4 周 + 1 周提交磨 | 5 周 | 4–5 周 |

## 推荐执行顺序

1. **第 1 周**：无论选哪套，都先做 Plan A 的 **infra 建设**（重写 dataset/metadata/SED head/DDP 训练循环），这是三套共用底盘。
2. **第 2 周末决策点**：跑出 Plan A R0 单 fold 后看 OOF，若 ≥ 0.91 → 信心足，可走 Plan B；若 = 0.89-0.91 → 选 Plan C 靠 diversity 弥补；若 < 0.89 → 老实留在 Plan A 把 5-fold 完成。
3. **第 5 周**：保留 **Plan A 最终 ensemble + Plan B 或 C 的最终 ensemble** 作为 Kaggle 两个 final selection 槽位，shake-up 防御。

## 不要做的事（来自仓库锐评）
- ❌ 继续生成 `create_aXX_*_notebook.py` 改 blend weights / threshold。这条路 41 次提交净增 0.001，已死。
- ❌ 继续 fork 公开 0.943 notebook 当 base——你被锁在别人的 ceiling 上。
- ❌ 用 `train_test_split(stratify=primary_label)` 当 CV——同录音跨 train/val 完全泄漏。
- ❌ 逐样本 mel mean/std 归一化——抹平能量信息。
- ❌ 4 卡训练只用 1 卡 batch 32——L40 单卡能装 64-128。

## 必做的事（三套共用）
- ✅ **GroupKFold by author/filename**，5-fold；罕见类全进训练。
- ✅ **SED head**（Adavanne framewise+clipwise+GeM），输出公式 `0.75·clipwise + 0.25·time_max`。
- ✅ **secondary_labels 权重 0.3** 软目标。
- ✅ **FocalBCE(α=1, γ=2)** + label smoothing 0.005。
- ✅ **MixUp(α=0.5, label-max) + SpecAug + BgMix(SNR 0–10dB) + RandomFiltering**。
- ✅ **bf16 + DDP 4 卡 + channels_last + torch.compile（CNN）+ EMA + SWA last 5**。
- ✅ **`train_soundscapes_labels.csv` 当一等公民**（2026 新增的 in-domain hard label，公开方案没用过）。
- ✅ **OpenVINO INT8/fp16 推理 + Quantile-Mix(α=0.5) ensemble + power calibration p^1.5**。
- ✅ **本地 OOF 评估闭环**——不要再摸黑提交调参。

## 文件目录

- `plan_a_stable_silver_to_gold.md` — Plan A 稳健路线
- `plan_b_aggressive_gold_push.md` — Plan B 猛冲金牌
- `plan_c_diversification_perch_transformer.md` — Plan C 差异化跨家族
- `review_feasibility.md` — Review agent 可行性评判（待生成）

## 关键参考链接
- BirdCLEF 2025 2nd Place（最完整可复现金牌代码）：<https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place>
- BirdCLEF 2025 1st 写作（Multi-Iter Noisy Student）：<https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n>
- BirdCLEF 2025 5th（30/60s 长 crop）：<https://github.com/myso1987/BirdCLEF-2025-5th-place-solution>
- BirdCLEF 2024 1st Kefir：<https://www.kaggle.com/competitions/birdclef-2024/discussion/512197>
- BirdCLEF 2024 3rd CPMP：<https://github.com/jfpuget/birdclef-2024>
- Perch 2.0 论文：<https://arxiv.org/html/2508.04665v1>
- Perch 2.0 代码：<https://github.com/google-research/perch>
- BEATs：<https://github.com/microsoft/unilm/tree/master/beats>
- PaSST：<https://github.com/kkoutini/PaSST>
- 2026 公开 0.925 (Perch+ProtoSSM)：<https://www.kaggle.com/code/imaadmahmood/birdclef-2026-perch-v2-protossm-0-925>
