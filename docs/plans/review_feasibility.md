# 三方案可行性评判（Grandmaster/系统工程师视角）

> 写于 2026-05-08。距比赛结束（6/3）约 4 周。当前 LB 0.929 / rank 704。
> 评判范围：`plan_a_stable_silver_to_gold.md`、`plan_b_aggressive_gold_push.md`、
> `plan_c_diversification_perch_transformer.md`、`README.md`。
> 参考：`src/birdclef2026/`（实现现状）、`docs/optimization_roadmap.md`（A0–A33 教训）、
> `configs/baseline.yaml`、`pyproject.toml`。

---

## 0. 总体冷启动事实核对

**仓库现状极薄，远低于三套方案默认假设的"infra ready"**：
- `dataset.py:24-39` 单 channel mel + `BCEWithLogitsLoss`，无任何增广；`audio.py:77-79`
  确实是逐样本 mean/std 归一化（README 锐评点中）。
- `metadata.py:185-198` `train_test_split(stratify=primary_label)` 同录音泄漏未修。
- `model.py` 只支持 `resnet18`（`raise ValueError` 写死），timm 没装。
- `train.py:91` `torch.cuda.amp.GradScaler` + `autocast`，**fp16 路径**，三套方案都说 "bf16
  + 无 GradScaler"，需重写。**无 DDP 入口**（`launch=python -m`，单卡），
  4×L40 完全没用。
- `pyproject.toml` 没有 `timm`、`onnx`、`openvino`、`tensorflow`/`tflite-runtime`、`fairseq`/`beats`、
  `silero-vad`、`mamba-ssm`。三套方案要的依赖全部要新装。
- `optimization_roadmap.md` Phase 4 已经写明「公榜 ≤0.93 必须自训」——和 README
  「blend 路死了」一致，方向无争议。

**事实结论**：W1 不存在"修小改"，是从空地建房。所有方案的 W1 时间预算严重低估。

---

## 1. Plan A — 稳健路线：技术正确但日程乐观

| 维度 | 分（10） | 备注 |
|---|---:|---|
| 技术正确性 | 7.5 | 配方主流、无致命错误 |
| 时间预算 | 6.0 | W1 一周搭 infra + 跑 R0 不现实 |
| 推理可行性 | 8.0 | 75 min 是三套里最稳的 |
| 期望分预测 | 7.0 | 0.943–0.948 略乐观，0.940–0.945 更现实 |
| 硬件匹配 | 8.0 | 对 4×L40 利用充分 |

### 正确的部分
- SED head 公式 `0.75·clipwise + 0.25·time_max`（plan_a 行 22）与 2023 1st、2025 2nd
  一致，没毛病。
- bf16+DDP+channels_last+`torch.compile` 配方是 Ada/L40 标准答案，比当前 `train.py:91`
  fp16+GradScaler 更适合 bf16 GPU。
- `secondary_labels` 权重 0.3、Focal(α=1, γ=2)、MixUp(α=0.5, label-max) 都是
  BirdCLEF 老金牌配方。
- GroupKFold by author/filename 解决当前 `metadata.py:192` 的同录音泄漏，必做。

### 硬伤
1. **W1 单周完成"重写 dataset/metadata + fork 2025 2nd + 单 fold 跑通 + OOF 闭环"**
   （行 85）严重乐观。当前仓库连 timm/SED head/DDP launch script 都没有；2025 2nd
   `BirdCLEF_2025_2nd_place` 仓库本身的 audio I/O 假设和我们的 `audio.py` 不兼容，迁
   移成本 ≥3 天；GroupKFold 还要补 author 抽取（taxonomy.csv 没 author 列，得
   parse Xeno-Canto 文件名前缀），1 天。**实际 W1 末能拿到第一份 OOF 已经是工程奇迹**。
2. **`fmax=14000`（行 28）和 `f_max=16000`（baseline.yaml 行 22）冲突**，说明作者没核对配
   置文件。fmax=14k 是为了避开 antialiasing artefact，但 32 kHz Nyquist 16 kHz 之内
   两者差异极小，**fmax=15000 更安全**——14k 会切掉部分高频蝙蝠/虫鸣（2026 真有 Insecta
   类，roadmap 提到）。
3. **NFNet-L0 + RAdam lr=1e-3**（行 56）数字偏高。NFNet 官方建议 1e-4 量级 + AGC
   (Adaptive Grad Clipping)，没 AGC 直接 RAdam 1e-3 容易 NaN。建议改 lr=5e-4 + AGC
   或退到 AdamW 1e-4。
4. **Epochs=50 + EMA + SWA last 5**（行 58）对 V2-S 偏多，30 epoch + cosine 已收敛；
   多出来的 20 epoch 是 ~12 GPU·h 净浪费。
5. **训练耗时估算 "V2-S 单 fold ≈2h, 3 backbone × 5 fold ≈ 30h GPU 总"**（行 60-61）
   严重低估：
   - V2-S 全量 train_audio (~310k 录音 → ~1.5M 5s clip) bs=256 全局，700 img/s/L40 ×
     4 卡 ≈ 2800 img/s，1 epoch ≈ 9 min；50 epoch ≈ 7.5 h 单 fold 4-GPU 墙钟。
   - **作者在「单 fold ≈ 2h」和「单 fold 7.5h 4-GPU 墙钟」之间偷换概念**——「2h」
     是单卡折算 GPU·h ÷ 4？那 3 backbone × 5 fold 应是 ~110 GPU·h，不是 30。
   - 实际 R0 一遍最少 30–35 h 4-GPU 墙钟，W2 末勉强出 R0。
6. **R3 描述（行 67-68）一句话糊弄过去**："用 R2 全 ensemble 重打一遍，迭代收敛即停"。
   没说收敛判据；2025 1st Babych 的实测是 R2 → R3 增益 +0.002 量级，再下去
   diminishing return 严重。**3 轮迭代实际只值 +0.005–0.008，不是行 88 写的 +0.014**。
7. **"7 backbone × 5 fold = 25 个 OpenVINO 推理 ≈ 60 min"**（行 80）算错：3 backbone × 5 fold
   = 15，不是 25。即便 15，OpenVINO INT8 V2-S/NFNet-L0/ConvNeXt 各自 600 文件 × 12 win
   实测 ~3–5 min/模型 × 15 = 45–75 min，加 mel+IO 已 ≥75 min。**贴近上限，无 buffer**。

### 修订建议
- baseline.yaml: `f_max=15000`，`n_mels=128` 保留，`top_db=80` 保留。
- 砍 ConvNeXt 第三路在 W3 做（不是 W4），保 V2-S × 5 + NFNet-L0 × 5 当核心。
- NFNet 用 AdamW 1e-4 + AGC，别用 RAdam 1e-3。
- W1 改成"infra + V2-S 单 fold smoke"，OOF 闭环挪到 W2 头。
- R 轮砍到 R0 + R2（双轮即可），节省 ~15h 给推理工程。
- 期望分校到 **0.940–0.945**，金牌概率 ~15–20%。

---

## 2. Plan B — Multi-Iter NS × Perch KD：最有金牌相但工程必然爆炸

| 维度 | 分（10） | 备注 |
|---|---:|---|
| 技术正确性 | 7.0 | 主路对，KD 公式略糙 |
| 时间预算 | 3.5 | 70 GPU·h 墙钟严重低估 |
| 推理可行性 | 4.0 | 81 min 几乎 100% 超时 |
| 期望分预测 | 6.0 | 0.948–0.952 是上限不是中位 |
| 硬件匹配 | 6.0 | 4×48GB 装得下，但 14–18 模型推理塞不进 |

### 正确的部分
- Multi-Iter Noisy Student（4 轮自蒸馏）确为 2025 1st Babych 配方核心；R2 50% labeled +
  50% pseudo + Input/Model noise（行 54）公式照抄无误。
- Perch v2 1536-d embedding + 2-layer Transformer head（行 25, 60-66）是合法的 KD/
  diversity 路线，单 GPU 1.5h 训一次的耗时估算合理（embedding cache 后是纯 head
  训练）。
- INT8 量化主 CNN、长 ctx 留 fp16、共享 mel 提取——工程方向对。
- `find_unused_parameters=False`（行 72）+ channels_last + `torch.compile`（仅 CNN，行 72）
  正确——Perch head 是 TF/JAX 不能 compile，BEATs/PaSST dynamic shape 不友好（plan_c 行
  49 自己也写了 "不开 torch.compile"）。

### 硬伤（这里最多）
1. **总 GPU·h 估算严重错算**（行 74-78）：
   - "R0 7 backbone × 5 fold ≈ 35 fold × 2 h ≈ 70 GPU·h ≈ 18 h 4-GPU 墙钟" —— 把 GPU·h
     当 4-GPU 墙钟除 4，可数学上对，但每 fold "2h" 单位不明且乐观（参见 Plan A 同一笔账，
     单 fold 4-GPU 墙钟 7.5h）。**正确量级：R0 ~250 GPU·h ≈ 65 h 4-GPU 墙钟**。
   - R1–R4 "每轮同量级 × 30/50 epoch 折扣 ≈ 12h × 4 = 48 h" —— 4 轮自蒸馏每轮训练量
     不会比 R0 少多少（要重训整个池子），实际 R1+R2+R3+R4 ≈ 4 × 50h = 200 h 4-GPU 墙钟。
   - **总墙钟 ≥250 h ≈ 10.5 天纯训练**，4 周内单流水线塞不下"主流程 + 1 次重训"。
   - 这个是全方案最致命的数字错——**Plan B 4 周做不完，6–8 周才现实**。
2. **CPU 90 min 推理 81 min 估算几乎一定爆掉**（行 106-111）：
   - "INT8 V2-S/B0/B3 12 模型 × 600 file × 12 win ≈ 35 min" —— OpenVINO INT8 V2-S
     在 Kaggle CPU（4 vCPU Xeon 2.0GHz）实测 ~0.3–0.5 s / 5s clip，600 × 12 = 7200 win
     × 0.4s = 2880 s = 48 min **单模型**，12 模型 × 48 min 显然不可能（除非作者意思是
     12 路全 batch 并行 + mel 共享，但这要 ~32 GB RAM Kaggle 给 16 GB，OOM）。
   - 14–18 模型 ensemble 在 Kaggle CPU 90 min 历史上**无人成功**，2025 1st 是 6–8 模型；
     2024 1st Kefir 是 6 模型。**14+ 是把比赛看成无限算力**。
3. **Perch v2 的 license/可用性未核对**：Perch 2.0 论文 2025 年发，代码在
   `google-research/perch` 但 BirdCLEF 2026 公榜 Perch 路线全是 TFLite 格式，TFLite 在
   Kaggle CPU 推理 600 文件实测 16 min（plan 行 65）—— 这个数对，但叠加 14 个 CNN 后
   总预算就破 90。
4. **R4 蒸馏方向反了**（行 56）："第 4 轮以 small-model（B0-NS/V2-B3）当 student，蒸馏
   到 ensemble teacher"——通常 KD 是 teacher → student，teacher 是 ensemble，student
   是单/小模型。这里"蒸馏到 teacher"是反向写法，可能作者意思是「以 small model 当
   student，从 teacher ensemble 学」，但措辞会让工程师写反。
5. **freq-axis ConvNeXt + Long-ctx 30s NFNet-L0** 都列在 R0 池子（行 24, 24）——这两路是
   trick / 实验性，Babych 1st 没用 freq-axis，5th place 才用 30 s。把实验性路线塞进
   R0 强制 5 fold 是浪费预算。
6. **bs=64/卡 V2-S → 全局 256** vs **bs=128/卡 B0-NS → 全局 512**（行 69-70）——B0
   bs=512 全局，对 234 类多标签 BCE 来说梯度方差太大，收敛不稳定，建议保持 256 全局。
7. **OOF-based teacher 选择**（行 57）—— 没说怎么选。Babych 的实操是「按 OOF macro
   AUC ≥ 当前 best - 0.001 才进 teacher 池」，原方案没写阈值，4 轮跑下来 confirmation
   bias 风险高。
8. **缺 Silero-VAD 集成步骤**（行 39）—— 一句话提到删人声，但 Silero-VAD 在 32kHz 下
   resample 到 16kHz 处理 500k+ 文件单卡 GPU 1 天才能跑完，未列入预算。

### 修订建议
- **强烈建议砍轮数**：4 轮 NS → 2 轮（R0 + R2），收益 ~80% 但时间砍半。
- **Backbone 池砍到 5–6**：V2-S × 5 fold + B3-NS × 3 + NFNet-L0 × 3 + Perch-KD × 1，
  砍掉 freq-axis / V2-B3 / 长 ctx 30s。
- **推理预算重做**：OpenVINO INT8 实测打 benchmark，按 600 文件 × 12 win × 0.3s /
  模型 ≈ 36 min/单模型，6 模型并行 batch + 共享 mel ≈ 70 min CPU。**没空间留 freq-axis**。
- 数字校正：R0 真实 4-GPU 墙钟 ~65h；2 轮 + KD = 130h ≈ 5.5 天，4 周可塞但留 buffer 不
  足。
- 期望分校到 **0.945–0.949**，金牌概率 ~25–30%。

---

## 3. Plan C — 跨家族 ensemble：方向最对但 BEATs/PaSST 是大坑

| 维度 | 分（10） | 备注 |
|---|---:|---|
| 技术正确性 | 7.0 | Mamba/PaSST 选型乐观 |
| 时间预算 | 5.5 | PaSST 30h 墙钟现实，但单点风险高 |
| 推理可行性 | 5.0 | PaSST OpenVINO 导出是真坑 |
| 期望分预测 | 7.5 | 0.945–0.950 中位现实 |
| 硬件匹配 | 8.5 | L40 48GB 跑 PaSST/BEATs 没问题 |

### 正确的部分
- "跨家族 diversity > 同家族迭代深度" 的策略是 Kaggle 公认 shake-up 防御方法，私
  榜抗性强（行 8-9, 14）。
- BEATs/PaSST 在 BirdCLEF 系列上确实没人系统试过，理论上有 alpha，但**有理由没人
  试**——见硬伤。
- Perch 2.0 + 12-token Mamba/Transformer head 的方向（行 53-65）是公榜 0.925
  Perch+ProtoSSM 的自然演化，能用 cached embedding，训推都便宜。
- 推理预算 84 min（行 110）比 Plan B 81 min 略宽松，但前提是 PaSST OpenVINO 导出
  成功。

### 硬伤
1. **BEATs/PaSST 的 OpenVINO 导出是个真问题**（行 144 自己也承认）：
   - PaSST 的 patchout 操作是动态 mask，ONNX 导出会展开成静态 graph 但 patchout
     mask 固定后掉 0.005–0.01 精度。
   - BEATs 用 fairseq 算子，ONNX-OpenVINO 通路实测会卡住几个 op（gumbel softmax、
     iter3 codebook lookup），需手写算子或 fp32 fallback。
   - **保守估计 1 周纯工程**——plan_c 没单独列工时。
2. **PaSST 单 fold 10h 4-GPU 墙钟**（行 50）数字偏低：
   - PaSST-S patch 16 stride 10 + 10s 输入 = ~9800 patch token，bs=16/卡 全局 64，
     L40 bf16 ≈ 30–40 it/s，1 epoch 训练样本 1.5M / 64 ≈ 23k step / 30 it/s = 13 min；
     30 epoch ≈ 6.5 h 单 fold——这个倒是基本对，但**前提是没 grad checkpoint，开了
     checkpoint 慢 30%**。10s 上下文用了 ~16k token 还说 1k token（行 14）—— 数字
     不一致。
3. **Mamba head 在 12-token 短序列上明确不如 Transformer**（plan_c 自己也说"先做
   ablation"，行 143）。SSM 在 ≥256 token 才比 Transformer 有效。**12 token 直接用
   小 Transformer 别试 Mamba**——`mamba-ssm` 还要装 Triton + CUDA 11.8 build，凭空多
   半天工程。
4. **"路 3 单条录音可拼成 5-cap × 5 s 即认为是单 file embedding 序列"**（行 60-61）逻辑
   有 bug——`train_audio` 单条录音长度从 3s 到 600s+ 不等，固定拼 5×5=25s 会丢失大量
   长录音信息或 pad 短录音；**正确做法是对每条录音随机采 12×5s（含重复）**。
5. **路 2 增广 "MixUp α 降到 0.3, BgMix p 降到 0.3"**（行 80）—— 没给依据。BEATs/PaSST
   官方 paper 是 SpecAug-only，连 MixUp 都没用。建议彻底关 MixUp，只 SpecAug+BgMix。
6. **总墙钟 93h**（行 98）和 Plan B 70h 的差距没解释——Plan C 模型更少（14 vs 14–18），
   反而更慢？因为 PaSST 30h ≈ V2-S 5 fold 全部时间。**这暴露 Plan B 70h 是错算**，
   两套同一表低估同一笔账。
7. **不做伪标 R2/R3** 是放弃了 +0.005–0.010 的"标准涨分点"——shake-up 抗性换分数上限
   是有道理，但 0.945–0.950 的预测假定 PaSST 单路 OOF 0.93+（行 51），这个数字**没历史
   依据**，BEATs/PaSST 在 BirdCLEF 公开实验最佳是 0.91 量级。
8. **Perch + Mamba head 推理 20 min**（行 107）—— Perch TFLite 16 min 对，但**Mamba
   ONNX 导出要 cuda kernel 替换为 selective_scan 的 ONNX op，Kaggle CPU 上要额外编**——
   实操 1 天工程。

### 修订建议
- **PaSST 改成 BEATs-iter3**（BEATs 已有公开 PyTorch 权重 + 简单 ONNX 路径），
  砍 PaSST。
- **路 3 head 改小 Transformer (d=256, 4 layer, 8 head)**，砍 Mamba。
- **加一轮 R1 伪标**（plan_c 自己留了 1–2 轮的口子，行 84），用路 1+路 4 ensemble 给
  路 1+路 4 重训，预算 +12 h 4-GPU 墙钟。
- 期望分校到 **0.943–0.948**，金牌概率 ~20–25%。

---

## 4. 跨方案对比与最终建议

### 数字"拍脑袋"清单
- Plan A 行 60-61：3 backbone × 5 fold ≈ 30 GPU·h（实际 ~110）。
- Plan B 行 74-78：总 70 h 墙钟（实际 ≥250 h）。
- Plan B 行 106-111：推理 81 min（实际 ≥110 min）。
- Plan C 行 98：93h 墙钟（数量级对，但前提 PaSST 不爆）。
- 三方案都给推理 ≤90 min 留 ≤10 min buffer——**Kaggle CPU 抖动 ±15% 是常态**，应
  留 ≥20 min。

### 应该砍的路线
- **Plan B 的 R3/R4 + freq-axis + 30s long-ctx**：净增益 < 0.005，时间成本 80h。
- **Plan C 的 PaSST + Mamba**：单点风险高，工程成本 1.5 周。
- **Plan A 的第三路 ConvNeXt**：放到 W4 当涨分备选，不强求。

### 优先级
1. **Plan A 的 infra 部分必做**（W1–W2）——这是三套共用底盘，没它什么都做不了。
2. **Plan B 的 R0 + 1 轮 NS（R2 配方）+ Perch-KD 单路**——把 Plan B 砍一半，作为
   主攻金牌路径。
3. **Plan C 的路 3（Perch + 小 Transformer head）作为独立 diversity 支线**——成本
   极低（embedding cache 后训 1.5h），合并到主 ensemble 当第 4 路。

### 不该做的事
- 不要 4 轮 NS。R2 之后立刻投入推理工程。
- 不要 PaSST/BEATs。BirdCLEF 历届无金牌路径，4 周内验证不完。
- 不要 Mamba。12-token 序列用小 Transformer 即可。
- 不要 14+ 模型 ensemble。**目标 6–8 模型推理在 90 min 内可控**。

### 风险对冲
- Kaggle 双 final selection 槽位 → **A 子集（V2-S × 5 + NFNet-L0 × 5 R0 ensemble）+
  混合 ensemble（A 的 R0 + B 的 R2 + C 的 Perch-head）**。前者 shake-up 友好，后者冲
  金。

---

## 5. 最终建议（合成路线 = A 底盘 + B 一轮 NS + C 第 3 路）

**冲金最优执行路径**：W1–W2 用 Plan A 配方（去 ConvNeXt、fmax=15k、NFNet 改 AdamW1e-4+AGC）建 DDP+SED+OOF 底盘并跑出 R0 (V2-S×5 + NFNet-L0×5)；W3 叠加 1 轮 Babych 配方 R2 伪标（50% labeled + 50% pseudo, p^1.82, Input/Model noise）+ Plan C 路 3（Perch v2 cached embedding + 小 Transformer head）；W4 推理工程压榨（OpenVINO INT8 + 共享 mel + Quantile-Mix(α=0.5) + power p^1.5 + per-class threshold）+ ensemble Optuna；W5 双提交锁定（A 子集做 anchor、A+B+C 混合做冲金），目标私榜 0.945–0.949、金牌概率约 25–30%。
