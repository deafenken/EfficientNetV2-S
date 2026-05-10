# 2025 2nd Place 对当前仓库的可借鉴点

参考：`VSydorskyy/BirdCLEF_2025_2nd_place`（Public 0.925 / Private 0.928，Pantanal 数据集）
完整逐项 diff：`docs/external_2025_2nd_place_diff.md`
当前 LB 锚点：A29 = 0.929（详见 `docs/optimization_roadmap.md`）

---

## TL;DR

我们的训练管线在**音频前端、损失、调度器、伪标签流程**上和 2nd-place 已经对齐甚至更精细（EMA/SWA 显式、warmup 显式、postprocess 模块完整）。

**真实差距集中在 3 件事上**：
1. 单 backbone（缺第二条 eca_nfnet_l0 分支）
2. 背景噪声增强代码已写但默认关掉
3. 稀有种过采样靠 sqrt-floor 近似，没用 A 公开的 85 类显式 dict

这 3 项保守估计可叠加 **+0.03 ~ +0.05 LB**，对照 silver cut（≈0.944）和我们的 0.929，是最直接的补强方向。

---

## 高优先级（建议立刻做，工程量小、收益明确）

### 1. 启用 ESC-50 背景噪声增强 (+0.010 ~ +0.020 LB)

- 现状：`src/birdclef2026/data/transforms.py` 已有 `BackgroundNoiseMixer` 实现；`configs/data.yaml:70` 默认 `enabled: false`，`dirs` 也没填。
- A 的配方：50% 概率混入，amp ∈ [0.25, 0.75]，源是 ESC-50 + soundscape-nocall（雨/风/虫等 20 类）。参考 `external_refs/BirdCLEF_2025_2nd_place/train_configs/selected_ebs.py:59-97`。
- 落地：
  1. 跑 `scripts/05_prep_esc50_ambient.sh` 拉 ESC-50 ambient 子集（~2 GB）
  2. 在 `configs/exp/e02_v2s_5fold_v2.yaml` 里 override `augment.background_noise.enabled: true` 并填 `dirs`
  3. 重训 5 fold，对比 e02 baseline OOF
- 备注：CLAUDE.md 说 e02 的 BG noise 已经在 commit `22fd8ef` 后启用，但只对 e02 生效；e01 / 探索 trial 还要分别打开。

### 2. 加上第二个 backbone（eca_nfnet_l0）做集成 (+0.015 ~ +0.025 LB)

- 现状：`configs/exp/eB1_nfnet_l0.yaml` 只是单 trial，没在主线 ensemble 里出现；`src/birdclef2026/inference/predict.py` 的多 ckpt 平均逻辑已经能直接接第二条分支。
- A 的做法：`tf_efficientnetv2_s + eca_nfnet_l0` 两个 backbone 各跑 5 fold，logits 平均。这是 0.925 的核心结构，不是 marginal trick。
- 落地：
  1. 把 `eB1_nfnet_l0.yaml` 升格为正式实验：5 fold、相同 epoch、相同 ESC-50 设置
  2. 复用 A 公开的 `eca_nfnet_l0_pretrain_from_bigXCV2_best.ckpt`（Kaggle dataset `vladimirsydor/bird-clef-2025-all-pretrained-models`），避免自己跑超大预训练
  3. 推理时和 e02 5-fold 做 10-ckpt logits 平均
- 风险点：训练时间 ×2；显存占用相近，L40×4 单机能扛。

### 3. 稀有种显式 oversample dict (+0.010 ~ +0.015 LB)

- 现状：`src/birdclef2026/data/samplers.py` 用 `sqrt_balanced_weights` + `rare_floor=30` 做近似平滑，**没有 per-class 显式倍数**。
- A 的做法：`external_refs/BirdCLEF_2025_2nd_place/train_configs/selected_eca.py:25-85` 列了 85 个稀有鸟种的显式倍数（turvul ×96、piwtyr1 ×90 …），把 1000:1 的类不平衡压到 ~30:1。
- 落地：
  1. 把那 85 项 dict 抠出来，存成 `configs/oversample/eca_2025_rare85.yaml`
  2. **过滤一遍**：保留 dict 里在 2026 物种集合内的 key（2025/2026 物种集不完全一样）
  3. 在 `samplers.py` 里加一条 `oversample_dict` 路径，dict 里出现的物种采样权重 = 显式倍数，未出现的走原 sqrt 公式
  4. 推荐先在 eB1（nfnet 那条）启用，因为 A 也是只在 ECA 模型上用 dict，EBS 模型用 EqualBalancing

---

## 中优先级（值得排期，但不紧急）

### 4. 加上 spec power / lower-high freq mask / spec noise 增强

- A 的实现：`code_base/augmentations/{spec_power_augment.py, spec_lowerhigh_augment.py, spec_noise_augment.py}`
- 我们目前只有 freq mask + time mask + random_filter。这三个在 A 里是默认开启的。
- 落地：直接把那三个文件移植到 `src/birdclef2026/data/transforms.py` 里，挂到 `augment` 配置树下。
- 预期：单项 +0.002 ~ +0.005 LB，作为锦上添花。

### 5. eB1 用 RAdam@1e-3 而不是 AdamW@1e-4

- A 的细节：EBS 用 AdamW@1e-4，ECA 用 RAdam@1e-3。我们的 e02 是 AdamW@1e-4 没问题；如果做 eB1，建议照 A 切换到 RAdam@1e-3。
- 落地：`training/train_ddp.py` 已有 optimizer 工厂逻辑，加个 `radam` 分支即可。

---

## 低优先级 / 可选

### 6. ONNX → OpenVINO fp16 导出

- A 在 `scripts/main_inference_and_compile.py` 里有完整 export 流程，最终 Kaggle 提交用的是 OpenVINO fp16 模型，CPU 推理快 3-5 倍。
- 我们的 Kaggle notebook 现在用的是 ONNX cache（A5 路线），如果想自己出 ckpt 则需要这一步。
- 不紧急的原因：当前提交策略是直接调 public 数据集里的 cached embedding，不卡推理速度。

### 7. Pseudo-label 加上 weighted-overlap 窗口

- A 的伪标签是非重叠 5 s 窗口 + 按起止时间重叠加权（`code_base/datasets/wave_dataset.py:35-78`），我们的 `predict_pseudo.py` 是简单平均。差异很小，估计 +0.001 ~ +0.003。
- 优先级低于 R2 自身（伪标签轮次比窗口加权更重要）。

---

## 已对齐 / 不需要动

| 组件 | 状态 |
|---|---|
| 采样率 / 段长 / mel 参数 | 完全一致（32k / 5s / n_mels=128 / hop=512 / fmin=20） |
| AttHead + GeMFreq | 实现一致 |
| Focal-BCE + LS=0.05 | 公式一致，参数一致 |
| MixUp（wave-level，p=0.5） | 一致 |
| Spec time/freq mask | 一致 |
| Random filter (4-point STFT EQ) | 一致 |
| 调度器（cosine + warmup） | 等价（A 用 CosineAnnealingWarmRestarts，我们用 SequentialLR） |
| EMA / SWA | 我们更显式更可控 |
| Postprocess | 我们多一套 threshold/smoothing，A 只有干净集成（不构成差距） |
| 伪标签 fold ensemble | 一致 |

---

## 推荐执行顺序

1. **本周**：动作 1（开 BG noise）+ 动作 3（稀有种 dict） — 都只改配置/数据，不动模型
2. **下周**：动作 2（加 eca_nfnet_l0） — 是收益最大的单项，但要训 5 fold
3. **再之后**：动作 4 / 5 锦上添花
4. **如果出了带 e02 + eB1 的 10-ckpt 集成**，对比 A29（0.929）应该能直接进 silver 区间（≈0.944）

---

## 关于「公开 0.946」的说明

用户提到的 0.946 是 BirdCLEF+ 2026 Kaggle 的某条公开 notebook 分数，**对应的是 Kaggle notebook 而不是 GitHub 仓库**。我们查不到它的 GitHub 镜像。

但 0.946 这个分段（A29=0.929，silver≈0.944，gold≈0.949）的核心架构，公开讨论里都指向 **Perch v2 embedding + ProtoSSM + MLP/LGBM probes + SED head + ensemble**——这套思路在 `references/public_baselines/` 里已经完整收录（wliilamsam_0943、mattiaangeli_0943、konbu17_0943、needless090_0935 等）。

**也就是说**：

- 想推 Perch + ProtoSSM 路线 → 看 `references/public_baselines/` 里的 0943 三件套，不用再 clone GitHub
- 想推自训练 CNN 路线（我们 e02 在做的事）→ VSydorskyy 这个 2nd place 是目前能拿到的**最强可比开源参照**，本文档基于它做对照
