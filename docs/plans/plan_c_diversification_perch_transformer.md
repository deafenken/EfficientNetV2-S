# Plan C — 差异化路线：Perch v2 + 长上下文 Transformer + BEATs/PaSST 跨家族

> 写于 2026-05-08。与 Plan A/B 不同，这条路线赌"模型家族多样性 > 单家族迭代深度"。
> 假设：4×L40 (48 GB×4)；4–5 周；适合**少数高质量模型 + 强 ensemble diversity**的玩家。

## 目标
- **私榜锚点**：0.945–0.950（金牌区下沿，shake-up 友好）。
- **核心策略**：少数路（4 路），但跨 3 个完全不同的模型家族（CNN-SED / Transformer audio / Perch embedding head），靠 diversity 而非自蒸馏深度涨分。
- 适合不愿意搞 ≥3 轮 NS 的玩家，对 shake-up 抗性更强。

## 为什么走这条路
1. 公开 0.925 (Perch+ProtoSSM) 直接给了一个起点，告诉我们 **Perch v2 单路就能到 0.92+**，剩下的 0.025 靠跨家族 diversity 比靠 self-distill 收敛更稳。
2. BirdCLEF 历届公开方案 ensemble 都是同家族 CNN-SED，BEATs/PaSST 类 Transformer 几乎没人系统试过——这是一块**没被吃过的 alpha**。
3. 4×L40 单卡 48 GB 能装得下 PaSST/BEATs（patch sequence ~1k token, fp16 + grad checkpointing）。私榜常见 shake-up 是过度拟合公榜的多 fold 同 backbone 集合，跨家族抗 shake。
4. 比 Plan B 复杂度低、风险中等，留更多工程预算给推理时间压榨。

## 模型族（最终 ensemble 4 路）
| 路 | Backbone | 数量 | 输入 | 训练 epoch | 角色 |
|---|---|---|---|---|---|
| 1. CNN-SED | `tf_efficientnetv2_s.in21k_ft_in1k` | 5-fold | 5 s mel 128×312 | 50 | 主力锚点 |
| 2. Transformer-SED | `BEATs-iter3` 或 `PaSST-S` | 3-fold | **10 s** mel 128×998 | 30 | 跨家族 diversity |
| 3. Perch-Long-Ctx | Perch v2 frozen → Mamba/Transformer head | 1（重复 3 seed） | 60 s = 12×5 s embedding | 50 | 长上下文 + Perch 先验 |
| 4. Freq-axis CNN | `eca_nfnet_l0` 改 freq-attention | 3-fold | 5 s mel | 50 | 频率轴 SED trick |

合计 5+3+3+3 = 14 模型，但因 Path 3 用 cached embedding 推理便宜，Path 2 跑 OpenVINO/ONNX，整体推理预算可控。

## 各路细节

### 路 1：EfficientNetV2-S CNN-SED（与 Plan A 相同 baseline）
- 5-fold StratifiedGroupKFold by author。
- Adavanne SED head + GeM + 512-d FC。
- FocalBCE(α=1, γ=2) + LS 0.005。
- AdamW lr=1e-4, cosine→1e-6, warmup 2 ep。
- bf16 + torch.compile + EMA + SWA last 5。
- 这一路提供"不出错的下限"。

### 路 2：BEATs / PaSST Transformer-SED（**diversity 主力**）
- 候选：
  - **BEATs-iter3**（微软 unilm，AudioSet mAP 0.486，PyTorch 即用）
    <https://github.com/microsoft/unilm/tree/master/beats>
  - **PaSST-S** (`passt_s_swa_p16_128_ap476`)
    <https://github.com/kkoutini/PaSST>（patchout 显著降显存）
- 输入：10 s @ 32 kHz, log-mel 128 × 998。
- L40 48 GB / bf16 / grad checkpoint：每卡 BS 16；DDP 全局 64。
- 优化器：AdamW 5e-5 + layerwise LR decay 0.75（Transformer 必备）。
- Schedule：cosine + 5% warmup，30 epoch。
- 增广：SpecAug 强 + MixUp α=0.3 + BgMix p=0.3（Transformer 对 MixUp 更敏感，调低）。
- 不开 `torch.compile`（dynamic shape 不友好），开 channels_last。
- 推理：导出 ONNX → OpenVINO fp16，10 s 输入 1 win/5 s 重叠 1 次（每段 11 win）。
- 时间预算：单 fold ~10 h on 4×L40。3 fold ~30 h。
- 期望：单路 OOF 0.93+，与 CNN ensemble 加权后涨 +0.005-0.010。

### 路 3：Perch v2 + 长上下文 head（**60 s context，公开 0.925 起点**）
- Embedding：`google-research/perch` Perch 2.0 → 5 s 输入 → 1536-d mean embedding。
- 全量预提取，cache 到 `data/embeddings/perch_v2/`（labeled + unlabeled soundscape，磁盘 ~50–80 GB）。
- Head 候选：
  - **Mamba-mini (d_state=16, layers=4, d_model=512)**——SSM 在 12-token 序列上效率高。
  - **小 Transformer (d=512, 4 layers, 8 heads)** + cls token。
- 输入：单条 60 s soundscape 的 12 个 5 s embedding 序列 → 234 logits（多标签 BCE）。
- 训练：`train_audio` 单条录音可拼成 5-cap × 5 s 即认为是单 file embedding 序列；`train_soundscapes` 天然 60 s 切。
- 损失：BCE + Focal；secondary 0.3。
- AdamW lr=1e-3 (head only)，bs=256/卡，全局 1024，50 epoch < 1.5 h on 4×L40。
- 推理：Perch TFLite ~16 min/600 file + head ~3 min < 20 min total。
- 这条路是 Plan C 的"差异化"核心——不通过 mel CNN 学习，靠 Perch 跨物种先验 + 长上下文 SSM 解。

### 路 4：Freq-axis ConvNeXt-Tiny / NFNet-L0
- 把 SED attention pooling 从 time 轴改到 freq 轴（参考 BirdCLEF 2025 25th 复盘）。
- 直觉：鸟鸣物种特征带集中在某些频段，freq-attention 能更直接学这种 prior。
- 5-fold；其余配置同路 1；ConvNeXt 因 dropout/stoch-depth 强，不需 EMA。
- 期望：单路 OOF 0.92，但与路 1 完全互补（attention 维度正交）。

## 数据 pipeline
- 与 Plan A 相同的 GroupKFold + secondary 0.3 + fmax=14k + 全局 mean/std。
- Silero-VAD 删人声（仅训练侧）。
- 共享 mel cache（h5py）路 1/4 复用；路 2 单独存（10 s 输入）；路 3 用 embedding cache。
- `train_soundscapes_labels.csv` 当 in-domain 主训集（路 1/2/4 都用）。

## 增广
- 路 1/4：同 Plan A 标准 SED 增广。
- 路 2 (Transformer)：MixUp α 降到 0.3，BgMix p 降到 0.3，SpecAug 加强 (time mask 2× width 60, freq mask 2× width 24)。
- 路 3 (Perch head)：embedding 维度做 dropout 0.1 + label smoothing 0.005，无音频侧增广。

## 伪标签策略（轻量化版本）
- **只迭代 1–2 轮**（Plan B 是 4 轮）。
- R0：4 路独立训练。
- R1：4 路 ensemble（含 Perch 路）给 `train_soundscapes` 打软标 → 路 1/2/4 用伪标重训 1 次（不重训 Perch head）。
- 不再迭代——把节省的时间投入到 Path 2 (Transformer) 和推理工程。

## 训练配置（4×L40 DDP）总览
| 路 | 单 fold 墙钟 | fold 数 | 总墙钟 |
|---|---|---|---|
| 1. V2-S | ~2 h | 5 | 10 h |
| 2. PaSST | ~10 h | 3 | 30 h |
| 3. Perch head | ~1.5 h | 3 seed | 4.5 h |
| 4. Freq-axis NFNet | ~2.5 h | 5 | 12.5 h |
| Pre-train (历届数据) | 一次性 | — | ~6 h |
| Pseudo R1 重训 | 同上量级 ×60% | — | ~30 h |
| **合计** | — | — | **~93 h 墙钟** |

可在 4 周内塞下，留 1 周给推理工程。

## 推理（CPU 90 min）
| 路 | 工具 | 时间预算 |
|---|---|---|
| 1. V2-S × 5 fold | OpenVINO INT8 | ~12 min |
| 2. PaSST × 3 fold | OpenVINO fp16 | ~30 min |
| 3. Perch v2 + Mamba head | TFLite + ONNX | ~20 min |
| 4. NFNet-L0 freq-axis × 3 fold | OpenVINO fp16 | ~10 min |
| Mel + IO + 后处理 | — | ~12 min |
| **合计** | — | **~84 min**（剩 6 min buffer） |

Ensemble 公式：
```python
# 各路 fold 内 mean
p_cnn  = mean(5 × V2-S)
p_tx   = mean(3 × PaSST)
p_perch= mean(3 × Perch-head)
p_freq = mean(3 × NFNet-freq)

# 跨家族加权（凭 OOF 调，先验权重）
p_mean = 0.30*p_cnn + 0.30*p_tx + 0.25*p_perch + 0.15*p_freq
p_rank = rank_mean(p_cnn, p_tx, p_perch, p_freq)
p_final = 0.5 * p_mean + 0.5 * p_rank        # Quantile-Mix
p_final = p_final ** 1.5                      # power calibration

# TopN(N=1) per file + Gaussian time smoothing
```

权重调参：用 5-fold OOF 跑 Optuna 50 trials 选最优组合。

## 4–5 周日程
| 周 | 工作 | 产出 |
|---|---|---|
| W1 | dataset/SED head infra；V2-S 单 fold + Perch embedding cache；OOF 闭环。 | LB 0.92 (路 1 单路) |
| W2 | 路 1 五 fold；路 3 Mamba head；路 4 freq-axis NFNet；BEATs/PaSST 选型 + 单 fold smoke。 | LB 0.93+ (3 路 ensemble) |
| W3 | 路 2 三 fold 完成；R1 伪标签；4 路 ensemble；OpenVINO INT8 量化。 | LB 0.94+ |
| W4 | 路 1/2/4 R1 重训；ensemble Optuna；推理时长压榨；per-class threshold；power calibration。 | LB 0.945+ |
| W5 (5 天) | 二备份提交；shake-up 防御；预提交检查。 | 双提交锁定 |

## 风险
- **中等**：BEATs/PaSST 在 BirdCLEF 类 fine-grained 上没被验证过（只在 AudioSet 通用），可能单路 OOF 不到 0.93——届时砍掉路 2，回退到 Plan A 双 backbone。
- Perch v2 训练数据可能未充分覆盖 Pantanal 物种 → 若 Perch head OOF < 0.91 则降权或弃用。
- Mamba SSM head 在 12-token 短序列上未必比 Transformer 好，先做 ablation。
- 推理工程：OpenVINO 对 PaSST/BEATs 的 patch_embedding 需要手动 trace，可能掉精度——保留 fp32 fallback。

## 与 Plan A/B 的关系
- 与 Plan A **共享 infra**（dataset, SED head, V2-S backbone），代码复用率 ~70%。
- 与 Plan B 的差异：Plan B 走深度（4 轮 NS、6+ backbone 同家族）；Plan C 走宽度（4 路跨家族、1 轮伪标）。
- **理想组合**：Plan A 跑出基线后，根据时间剩余决定走 Plan B（深）还是 Plan C（宽），或两者各占 2 提交槽位。

## 期望产出
- 私榜 0.945–0.950（top 10–30），shake-up 抗性强。
- 一套**跨家族 ensemble 模板**，复用到 BirdCLEF 2027+ 也是金牌路径。
