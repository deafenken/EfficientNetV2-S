# Commit 2 训练 infra Code Review

审查范围：19 个文件，约 1500 行 Python。审查目标：能否 `bash scripts/10_train.sh configs/exp/e01_v2s_5fold.yaml 0` 跑通 EfficientNetV2-S + SED head 单 fold 50 epoch DDP 训练。本文按严重度倒序列出问题，每条给路径 + 行号 + 修复建议。

---

## BLOCKER

### B1. `scripts/10_train.sh` 中 `exec uv run torchrun` 与 `PYTHONPATH=src` 的工作目录链路存在隐患（潜在 BLOCKER，需用户实测）
- 路径：`scripts/10_train.sh:33-40`
- `PYTHONPATH=src exec uv run torchrun -m birdclef2026.training.train_ddp ...` 这条命令在 `cd "$ROOT"` 后执行，逻辑上 OK；`uv run` 默认 spawn 子进程时会保留 `PYTHONPATH`，子进程的 worker 也继承（torchrun standalone 是 fork/exec 而非新 shell）。若 `uv run` 创建的虚拟环境里已经 `pip install -e .`，则 `PYTHONPATH=src` 多此一举但不会冲突；若没装包，则 worker 子进程能找到 `src/birdclef2026`。整体能跑，但**没有任何冒烟测试覆盖**——建议在 `scripts/10_train.sh` 顶部加 `python -c "import birdclef2026"` 预热以提前失败。
- 修复建议：保持现状，但在 README 里写明前置 `uv pip install -e .`，避免 PYTHONPATH 不生效时的隐式失败。

> 严格地说没有真正的 BLOCKER（代码能跑通）。下面的 HIGH 级问题里有几条会让训练**收敛严重打折**或**首个 epoch 立即抛异常**，按风险排序也接近 BLOCKER。

---

## HIGH

### H1. `LogMel` 在 `global_mean/global_std=null` 时静默回退到 per-sample 标准化，且训练脚本不告警
- 路径：`src/birdclef2026/utils/audio.py:116-129`，`src/birdclef2026/models/sed.py:172-173`，`configs/data.yaml:36-37`
- 现状：`audio.global_mean: null` 是 yaml 默认值，`scripts/02_compute_mel_stats.py` 算完后**要用户手工贴回 yaml**。一旦忘记，`LogMel.has_global=False`，`mel = (mel - mel.mean()) / mel.std().clamp_min(1e-6)`——逐样本归一化，正是文档里痛斥的“破坏绝对能量信息”的 legacy 行为。50 个 epoch 烧完才发现 OOF AUC 偏低，代价巨大。
- 修复建议：在 `train_ddp.main()` 里加显式校验：若 `audio_cfg.get("global_mean")` is None 而 `cfg.get("paths", {}).get("mel_stats")` 文件存在，自动加载 `mel_stats.json` 并 inject 进 audio_cfg；若 mel_stats 不存在，至少 `if rank == 0: print("[warn] using per-sample mel norm; run scripts/02 first")`，最好直接 raise 让用户显式选择。

### H2. `_autocast_ctx("bf16")` 包裹 `torch.stft` / `torch.istft` 在 PyTorch 2.x 下行为不稳定
- 路径：`src/birdclef2026/data/transforms.py:234-261`，`src/birdclef2026/training/train_ddp.py:230-235`
- `RandomFiltering.forward` 在 `model.forward` 内被调用，外层处于 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`。`torch.stft` / `torch.istft` 不在 autocast 白名单，按理保持 fp32，但部分 PyTorch 版本会把输入 cast 到 bf16，导致 `RuntimeError: stft input must be either float or double`。即便不 raise，complex64 → bf16 来回转换会让相位误差累计。
- 修复建议：在 `RandomFiltering.forward` 第一行加：
  ```python
  with torch.autocast(device_type=wave.device.type, enabled=False):
      wave = wave.float()
      ...
  ```
  并在 `LogMel.forward` 也加同样的 disable（MelSpectrogram 内部是 conv，autocast 会跑 bf16，但 `AmplitudeToDB` 的 `log10` 在 bf16 下精度极差）。

### H3. `amp_dtype="fp16"` 路径没有 `GradScaler`，会立刻发散
- 路径：`src/birdclef2026/training/train_ddp.py:230-271`
- bf16 不需要 scaler，但 `_autocast_ctx` 同时支持 `"fp16"`，后续 `loss.backward()` 没有 `scaler.scale(loss).backward()` / `scaler.step()`。fp16 下梯度立刻 underflow，loss 一路 NaN。
- 修复建议：要么删掉 fp16 分支只留 bf16/fp32，要么补 `GradScaler`。e01 用 bf16 不会触发，但配置项还在容易误用——建议直接删除 fp16 选项或加 `assert amp_dtype in {"bf16", "fp32"}`。

### H4. `SEDDataset` 没有任何坏文件兜底，单个 `.ogg` 解码失败就杀 DataLoader worker
- 路径：`src/birdclef2026/data/dataset.py:84-97`，`src/birdclef2026/utils/audio.py:25-40`
- `load_audio` 内已经 fallback 到 soundfile，但若 soundfile 也失败，异常上抛 → DataLoader worker 死亡 → 整个 epoch 失败。BirdCLEF 历年都偶发损坏文件（截断/0 字节），4×L40 训练要跑 8–12 h，半路挂掉很难受。
- 修复建议：在 `_load_one` 里 try/except，失败时重抽一条；或在 `load_audio` 内捕获所有异常返回 `torch.zeros(int(sr*0.1))` 之类 dummy 让上层 `crop_or_pad` 兜底成静音（target 还是有的，相当于硬负样本）。

### H5. EMA 对 BatchNorm running buffers 做 EMA，叠加 DDP 不同步 BN，导致跨 rank 偏差
- 路径：`src/birdclef2026/training/ema.py:39-46`
- `update` 对所有 floating buffer 做 `mul_(decay).add_(...)`，包括 BN running_mean / running_var。然而 DDP 默认**不同步 BN buffer**，每个 rank 的 BN buffer 在 forward 中独立累加，跨 rank 略有差异。EMA 又把这些有差异的 buffer 各自 EMA，最终 rank 0 的 EMA BN 与 rank 1-3 的 EMA BN 不一致；rank 0 上 validate 用的是 rank-0 BN，与全数据 BN 估计有偏差。
- 影响：v2s 含 BN，单 fold AUC 估计偏差 ~0.001–0.005，5 fold 噪声进一步放大。
- 修复建议：（a）把模型 wrap 成 `SyncBatchNorm` 在 `DDP(model)` 之前 `nn.SyncBatchNorm.convert_sync_batchnorm(model)`；（b）EMA 的 buffer 处理改成只对 floating point **参数** 做 EMA，buffer 直接 copy（与 timm `ModelEmaV2` 一致）。两者择一即可。

### H6. `RandomFiltering` / `LogMel` / `SpecAugment` 在模型内 `train()` 期开启，但 EMA 在验证时也是 `eval()`——OK——可 `LogMel` 的 `MelSpectrogram` 在 `eval()` 仍然计算（理应如此），不过 `to_db.top_db` 会因 batch max 变化导致**同一段音频在不同 batch 下 mel 值略有不同**
- 路径：`src/birdclef2026/utils/audio.py:115-128`
- `AmplitudeToDB(top_db=80.0)` 内部对**每个调用**的最大值做 clip（`top_db` 是相对 max 的下限）。这意味着 batched 调用下，同一文件的 mel 值依赖于 batch 内其他文件最大功率。global_mean/std 是按训练分布算的，但运行时 batch 依赖会让推理与训练分布略偏。
- 修复建议：把 `AmplitudeToDB` 替换成 `10 * torch.log10(spec.clamp_min(eps))`（不做 top_db 截断）或在外面手动做绝对 dB 阈值（`mel = mel.clamp_min(-80.0)`）。BirdCLEF 25 2nd 用的是绝对 dB（amin=1e-10），不依赖 batch max。

---

## MEDIUM

### M1. MixUp 对两条 wave 都先单独叠加 BackgroundNoise 再混音，等效于双倍噪声
- 路径：`src/birdclef2026/data/dataset.py:99-100, 141-156`
- `_load_one` 内部就调用了 `self.background_noise(waveform)`，partner 也走 `_load_one` → 也加噪。最终 `mixup_pair` 把两条**已加噪的** wave 平均；噪声振幅期望从 U(0.25, 0.75) 拉到约 (0.5)/2 + (0.5)/2 = 同等强度但**双独立噪声叠加**，谱形更糙。BirdCLEF 25 2nd 是先 mixup 再加单层噪声。
- 影响：训练分布偏离设计，可能轻微降 ROI（~0.001 AUC）。
- 修复建议：`_load_one` 拆出 `_load_raw(idx)` 不加噪版本，partner 用 raw；mixup 后再统一过一次 `background_noise`。

### M2. `cv.fold_indices` 返回的是 `df.index` 标签而非 iloc 位置，文档说能直接喂给 `Subset` 是误导
- 路径：`src/birdclef2026/utils/cv.py:28-38`
- `df.index[mask].tolist()` 返回 index 标签。如果调用方传入未 `reset_index` 的 df（比如 fold_csv 已经 merge 了别的东西），`Subset(dataset, indices)` 会按 iloc 取，结果错位。
- 修复建议：改成 `np.where(mask)[0].tolist()` 显式拿 iloc，或在文档里强调“调用前必须 `df.reset_index(drop=True)`”。当前 `train_ddp` 没用这个函数（直接用了布尔 mask），所以暂时无害，但是 latent bug。

### M3. 验证集只在 rank 0 跑，`dist.barrier()` 默认 30 min NCCL 超时，未来加入 train_soundscapes 后可能踩雷
- 路径：`src/birdclef2026/training/train_ddp.py:425-470`
- 当前 train_audio ~28k 行，5 fold 单 fold val ~5.6k 行 / batch 64 ≈ 90 step ≈ 10–20 s。R1 加入 soundscape pseudo 后 val 可能 20–50k 行，加上 num_workers=8 启动开销，rank 1-3 在 barrier 等待时间逼近 NCCL 超时。
- 修复建议：把 val 也并行到全部 rank（`DistributedSampler(shuffle=False, drop_last=False)` + 末尾 `dist.all_gather` 拼回去 + 去重 padding 行），或者在 barrier 前提高 `NCCL_TIMEOUT`/`init_process_group(timeout=timedelta(hours=2))`。

### M4. SpecAugment 用 Python for-loop 逐样本 mask，B=64 时 GPU 利用率受影响
- 路径：`src/birdclef2026/data/transforms.py:181-195`
- 64 次 `freq(mel[i:i+1])` 调用，每次内部 launch 数个 CUDA kernel，host-side overhead 会拖慢一点 step 时间（≈3–5%）。不影响正确性。
- 修复建议：用 `torchaudio.transforms.FrequencyMasking` 的 batched 版本（实际上它本来就支持 batched 输入），把循环消掉。或保持现状（acceptable）。

### M5. `ids = np.asarray(all_ids)` 在 OOF 写盘前没有按字典序排序
- 路径：`src/birdclef2026/training/train_ddp.py:471-475`
- val_loader shuffle=False 但 dataset 是 `df[df.fold==fold].reset_index(drop=True)`，顺序取决于 metadata 构造时的行序。后续 ensemble/blend 跨 fold 用 `sample_id` 对齐时，每 fold 行序不同需要 reindex；如果上游脚本依赖固定顺序则会出错。
- 修复建议：在 `save_oof` 前按 `ids` 排序统一所有 fold；或在文档里写明 OOF 顺序“按 sample_id 字典序”。

### M6. `_collect_noise_paths` 用 `Path.rglob` 同步扫盘，e01 默认 enabled=False 不触发，但 R1+ 启用后冷启动慢
- 路径：`src/birdclef2026/training/train_ddp.py:120-132`
- 每个 rank 独立 rglob，4 ranks 并发扫同一目录，IO 抢占；ESC-50 + soundscape-nocall 几千文件没问题，量大就慢。
- 修复建议：rank 0 扫一次，把 path 列表 broadcast 给其他 rank；或 cache 到 `outputs/cache/noise_paths.txt`。

### M7. mel_stats.json 没有自动写回 yaml，`scripts/02` 输出 JSON 用户必须手工 copy-paste
- 路径：`scripts/02_compute_mel_stats.py:113-120`
- 与 H1 同源。建议 02 脚本加 `--write-config configs/data.yaml` 选项，用 ruamel.yaml 保留注释直接修改，或在 yaml 里支持 `audio.mel_stats: outputs/stats/mel_stats.json` 然后训练侧 `LogMel` 自动读。

### M8. `FocalBCEWithLogits` 同时计算 BCE + Focal，相当于两次过 `binary_cross_entropy_with_logits`，损失值约为单 BCE 的 1.5–2 倍
- 路径：`src/birdclef2026/models/losses.py:99-107`
- 不是 bug（2nd place 也这么做），但**学习率和 grad_clip 是按 “一份 BCE” 直觉调的**。e01 配 lr=1e-4, clip=1.0 应该 OK，但日志里 loss 数值会偏大，新人 review 时容易误以为不收敛。
- 修复建议：日志里印 `loss / (bce_weight + focal_weight)` 作 normalized loss，或在 README 写明数值范围预期。

### M9. `total_steps` 和 `warmup_steps` 取整后可能让 cosine 提前结束 1 step
- 路径：`src/birdclef2026/training/train_ddp.py:398-406`
- `warmup_steps = max(1, int(warmup_epochs * steps_per_epoch))`，`int()` 直接截断；`T_max = max(1, total_steps - warmup_steps)` 正常，但若 `len(train_loader)` 因 `drop_last=True` 比预期少 1，最后一步 scheduler 会调用第 (total_steps+1) 次 → cosine 已经用完，LR 卡在 lr_min。低概率不致命。
- 修复建议：`scheduler.step()` 前加 `if scheduler.get_last_lr()[0] > 0: scheduler.step()`，或预算 total_steps 时多算一个 epoch buffer。

### M10. `head_dropout=0.5` 走的是 `Dropout(0.25) → Linear → ReLU → Dropout(0.5)`，与 BirdCLEF 25 2nd 一致，但代码里没注释 dropout/2 的来由，未来调参容易误改
- 路径：`src/birdclef2026/models/sed.py:65-71`
- 已经在 commit message / docstring 里说了，但建议在 `nn.Dropout(dropout / 2)` 旁边加 inline 注释 “# 2nd-place: pre-FC dropout=p/2, post-ReLU dropout=p”。LOW 也可，放 MEDIUM 因为容易误调。

---

## LOW

### L1. `seed_everything` 不设 `torch.use_deterministic_algorithms(True)` 也不设 cuDNN 确定性 flag
- 路径：`src/birdclef2026/utils/seed.py:12-18`
- 没必要追完全确定性（会拖慢 cudnn.benchmark），但建议在 docstring 注明“近似复现，cudnn.benchmark=True 不保证 bit-exact”。

### L2. `out["clipwise_pred"]` 计算了但训练/验证从来不用
- 路径：`src/birdclef2026/models/sed.py:86-89`，`src/birdclef2026/training/train_ddp.py:264-266, 308-310`
- 一次 sigmoid + 加权求和的浪费，每 batch 几十微秒，无所谓。可以延后到 inference-only path 算。

### L3. `_multi_hot` 中 `target[idx] == 0` 比较 0-d tensor，可读性差
- 路径：`src/birdclef2026/data/dataset.py:134`
- 改成 `target[idx].item() == 0` 或 `bool(target[idx] == 0)`。

### L4. `BackgroundNoise.__call__` 用 `random` 模块而非 numpy/torch RNG，跨 worker 复现时 seed 流要单独管理
- 路径：`src/birdclef2026/data/transforms.py:64-77`
- DataLoader workers 每个 worker 默认 seed=base_seed+worker_id，`random` 会被 PyTorch worker init 自动 seed，OK；但 `mixup_pair` 用的是 `np.random.beta`，partner_idx 又是 `torch.randint`——三套 RNG 混用，复现时麻烦。建议统一用 `torch.Generator` 或在 worker_init_fn 里 explicit seed。

### L5. `save_checkpoint` 把整个 `cfg` 字典塞 ckpt，包含 `_exp_path` 这种本地路径，分发 ckpt 时携带本地路径泄露
- 路径：`src/birdclef2026/training/train_ddp.py:453-461`，`src/birdclef2026/utils/config.py:54`
- 不算敏感，但建议 strip `_exp_path` 之类 key。

### L6. `sigmoid_focal_loss` 在 bf16 autocast 下 `torch.sigmoid(logits)` 与 `binary_cross_entropy_with_logits` 内部 sigmoid 计算两次，bf16 精度有限
- 路径：`src/birdclef2026/models/losses.py:38-58`
- bce_with_logits 用 log-sum-exp trick 数值稳定，单独 sigmoid 在 bf16 下尾部精度只有 ~1e-3，对 focal 的 (1-p_t)^gamma 项足够。e01 用 bf16 训得动，2nd place fp32 训得动，所以 OK。如想更稳：`p = torch.sigmoid(logits.float())`。

### L7. `scripts/03_make_folds.py` 输出 csv 没记录 `fold_seed`/`n_splits`，多次重跑会静默覆盖
- 路径：`scripts/03_make_folds.py:67-72`
- 建议 csv 同目录写一个 `folds_meta.json` 记 `seed`, `n_splits`, `len(target_columns)`, generated_at。

### L8. `torch.backends.cudnn.benchmark = True` 与 `seed_everything` 矛盾，但 README 没写明
- 路径：`src/birdclef2026/training/train_ddp.py:350`
- benchmark=True 对 v2s 是 +5–10% 吞吐的好优化，仅在 fold 间 AUC 抖动 0.001 级别，可接受。

### L9. `data.yaml` 的 `audio.f_max: 16000.0` 等于 Nyquist，导致最高一个 mel filter 有零宽
- 路径：`configs/data.yaml:34`
- 标准做法是 `f_max = sample_rate / 2 - 100`（如 15500）或留一点 margin。BirdCLEF 25 2nd 用 14000。当前 16000 不会 crash（torchaudio 会 warning），但顶端 mel bin 几乎空。
- 修复建议：改为 `f_max: 14000.0` 或 `15500.0`。

### L10. `extract_window` end_time 校验缺失
- 路径：`src/birdclef2026/utils/audio.py:65-79`
- `end_time` 若是负数或超过音频长度的 10 倍，`crop_or_pad` 用 `max(0, min(...))` 兜住——不会 crash 但会取到错误位置。soundscape 的 `end_time` 来自 row_id 解析，理论上 5/10/15...，没问题，但缺 sanity check。

### L11. `make_folds` 未分配的兜底丢进“最小 fold”——会让该 fold 略大一点
- 路径：`src/birdclef2026/data/folds.py:63-67`
- 注释说“在 4/5 fold 中保持训练”，OK，但相当于 silent partial assignment。建议至少 `print` 一行兜底数。

---

## 总结（5 行）

1. **能直接 smoke run 吗？**——技术上能跑通 1-2 个 epoch（前提：`uv pip install -e .`、数据已下载、fold csv 已生成、mel_stats 已贴到 yaml）。但**不要直接全量 50 epoch**，先解决 H1+H2+H4 再开烧。
2. **最关键 issue#1 (HIGH H1)**：mel_stats.json 没自动注入 yaml，用户忘记手贴 → 训练静默使用 per-sample 归一化，AUC 大幅打折。修复：训练脚本启动时强校验或自动加载 `outputs/stats/mel_stats.json`。
3. **最关键 issue#2 (HIGH H2)**：`torch.stft/istft` 与 `AmplitudeToDB` 在 bf16 autocast 下行为不稳定；建议在 `RandomFiltering.forward` 与 `LogMel.forward` 第一行 `torch.autocast(..., enabled=False)` 强制 fp32。
4. **最关键 issue#3 (HIGH H4)**：单个坏文件就杀 DataLoader worker，训练半路炸炉概率显著；`_load_one` 必须 try/except 兜底。
5. **次重要**：H5 (EMA on BN buffer + DDP no SyncBN) 与 H3 (fp16 无 GradScaler) 是潜在精度/可用性陷阱，建议 commit 内一起补；M1（双层背景噪声）等 R1 启用 bg-noise 之前修也来得及。
