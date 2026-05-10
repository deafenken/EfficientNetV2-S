# Kaggle 提交全链路 — 踩坑笔记 + 操作手册

针对 BirdCLEF 2026（code competition）从 L40 训练机推送 dataset/kernel 并触发提交的完整流程，连同我们撞上的每一个 bug 和解决方法。后人按这个手册走可以省掉至少一天的来回试错。

代码入口：`scripts/17_kaggle_eB1_submit.py`（端到端 Python 编排器）+ `kaggle/variants/<variant>/`（每次提交的 notebook+metadata）+ `kaggle/datasets/<slug>/dataset-metadata.json`（数据集 metadata）。

---

## 1. 拓扑与角色

```
┌────────────┐  git push    ┌────────────┐  rmt run    ┌────────────────┐
│ VPS (write │ ───────────▶ │ GitHub      │ ──────────▶ │ L40 container  │
│  code only)│              │ winbeau    │             │ (run, push)    │
└────────────┘              └────────────┘             └────────────────┘
                                                              │ kagglehub.dataset_upload
                                                              │ kaggle kernels push
                                                              ▼
                                                       ┌────────────────┐
                                                       │ Kaggle         │
                                                       │ - dataset      │
                                                       │ - kernel (run) │
                                                       │ - submission   │
                                                       └────────────────┘
```

VPS 只写代码 + commit + push。所有需要 90 MB checkpoint 或者跑 Python 的步骤都在 L40 容器里执行。L40 的网络出口是大学网络，对 IPv6 路由有问题（见 §3.1）。

---

## 2. 认证 split-stack

**关键事实**：Kaggle 现在有两种 token，**互不通用**：

| Token | 文件 / 环境变量 | 适用 |
|-------|----------------|------|
| **PAT (Personal Access Token)** | `~/.kaggle/access_token` (38 字节裸 key) → `KAGGLE_API_TOKEN` env | `kagglehub` (datasets, models) ✓<br>`kagglesdk` Bearer auth ✓<br>**`kaggle` CLI ✗**<br>**kernels API ✗（403 / 409）** |
| **Legacy API token** | `~/.kaggle/kaggle.json` (`{"username":..., "key":...}`) → `KAGGLE_USERNAME`+`KAGGLE_KEY` env | `kaggle` CLI（含 `kernels push`）✓<br>kagglehub ✓ |

**PAT 缺 kernels.read / kernels.write scope**——直接调 `GetKernel` / `SaveKernel` 会 403 / 409。我们试过 kagglesdk 的 `save_kernel` 走 PAT，被服务端拒。

**唯一可靠方案**：

```python
# Dataset：kagglehub + PAT
import kagglehub
kagglehub.dataset_upload(handle, local_dir, version_notes)

# Kernel：legacy CLI + kaggle.json
subprocess.call(["kaggle", "kernels", "push", "-p", kernel_dir])
```

获取 legacy token：https://www.kaggle.com/settings → "API" → **"Create New Token"** → 浏览器自动下载 `kaggle.json` → `~/.kaggle/kaggle.json`，`chmod 600`。**这个跟 PAT 不是同一个东西，不要混用。**

---

## 3. 我们撞过的 bug 与修复

### 3.1 kagglehub 上传卡 0% 不动（IPv6 路由坏）

**症状**：`Uploading: 0%|...| 0.00/3.19k [00:00<?, ?B/s]` 一直不动，最终 `urllib3` `sock.connect()` 超时。`google.com` 用 curl 正常，但 `requests.put` 到 GCS signed URL 卡住。

**原因**：L40 大学网络出口对 IPv6 路由不通，但 Python `urllib3` 默认 happy-eyeballs 会先试 IPv6。GCS upload endpoint 解析到 IPv6 地址，TCP connect 卡住直到超时。

**修复**（commit `d835ebe`）：在 `import kagglehub` 之前 monkey-patch socket：

```python
import socket
_orig = socket.getaddrinfo
def _v4_only(host, port, family=0, *args, **kwargs):
    return _orig(host, port, socket.AF_INET, *args, **kwargs)
socket.getaddrinfo = _v4_only
import kagglehub  # 现在它的 session 拿到的全是 IPv4
```

### 3.2 `resp.url()` TypeError（kagglesdk 字段不是方法）

**症状**：`TypeError: 'str' object is not callable`，发生在 push **成功之后** 打 log 那一行。

**原因**：`kagglesdk.KaggleObject` 用 descriptor 把字段暴露为属性而非方法。`resp.url` 直接是 string，不是 `resp.url()`。

**修复**（commit `d4e881b`）：

```python
url = getattr(resp, "url", None) or f"https://www.kaggle.com/code/{slug}"
version = getattr(resp, "version_number", None)
```

### 3.3 Kernel push 409 Conflict（slug 自动改名）

**症状**：第一次 push 成功，第二次 push 报 `409 Client Error: Conflict for url: .../SaveKernel`。

**原因**：Kaggle server 把 **title** slugify 之后用作 canonical slug，**忽略了我们 metadata 里写的 id**。比如 title=`BirdCLEF 2026 b01 eB1 NFNet-L0 fold0` 被 server 转成 `birdclef-2026-b01-eb1-nfnet-l0-fold0`，我们的 id 写的是 `birdclef-2026-b01-eb1-nfnet-l0`（没有 `-fold0`）。第二次 push 想再 create 同 title 的 kernel → 跟现有 slug 冲突 → 409。

**排查命令**：`kaggle kernels list --user winbeaux` —— 看真实的 slug 是什么。

**修复**（commit `c649e2d`）：把 metadata 里的 `id` 改成跟 title slugify 后一致的值：

```json
"id": "winbeaux/birdclef-2026-b01-eb1-nfnet-l0-fold0"
```

或者更稳：保证 title 不含会被额外加 hyphen 的字符。

### 3.4 PAT 下 kernels API 全报 403/409

**症状**：用 `kagglesdk` 的 `client.kernels.get_kernel(...)` 拿 403，`save_kernel(...)` 拿 409，明明 PAT 在 kagglehub.whoami() 是好的。

**原因**：见 §2，PAT 不带 kernels scope。

**修复**：放弃 kagglesdk 路径，全部 fallback 到 `kaggle kernels push -p <dir>`（commit `62abc9c`）。

### 3.5 `kaggle kernels push` 默认会 trigger 一次 run

不算 bug，但要知道：push 完 kernel 在 Kaggle 会**自动跑一次 "Save & Run All"**，不需要再手动点。后续 `Submit to Competition` 用的是这次 run 的版本。

### 3.6 Kaggle P100 GPU 与 torch wheel 版本不兼容

**症状**：interactive run 在 placeholder 数据上看似过了（3 行没真实音频，全零填充 bypass GPU），但 `Submit to Competition` 触发的 hidden test rerun 报 "Notebook Threw Exception"。

**根因**（看 kernel log）：

```
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible
The current PyTorch install supports CUDA capabilities sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
```

ttahara 的 wheels kernel 装的是 sm_70+ 编译的 torch，P100 是 sm_60。Interactive run 没真实音频不调用 GPU forward，rerun 真的有 600 个 audio file → forward → CUDA error → exception。

**修复**（commit `0f4d5c4`）：notebook 强制 CPU + fp32：

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # 必须在 set env 之后才 import
```

kernel-metadata.json 同步 `enable_gpu: false`。NFNet-L0 在 CPU 上跑 ~7000 个 5s window 大约 5-15 min，远低于 9h cap。

### 3.7 Kernel rerun 错误信息**官方故意不暴露**

**关键事实**：BirdCLEF 这种 code competition，rerun 在 hidden test 上的 stderr / traceback **不通过任何 API 返回**——Kaggle 官方明说这是反 probing 措施（防止有人通过错误信息探测 hidden 数据集）。

文档：https://www.kaggle.com/code-competition-debugging（Errors & Debugging Tips 页）

你能拿到的只有错误 class 名（"Notebook Threw Exception" / "Submission CSV Not Found" / "Notebook Timeout" / ...），没有具体 traceback。

**对策——按官方建议 fail-soft**（commit `323201c`）：

1. **Notebook 第一个 cell 立刻写一份 zero-fallback `submission.csv`**（用 sample_submission 的真实 shape，全部预测列填 0.0）。这个步骤只依赖 pandas + 读 sample_submission，两个都是 Kaggle base image 必带，理论上不会失败。
2. **后续每个 cell 用 `try/except` 包住，崩了就 print + traceback + `sys.exit(0)`**（不 raise）。
3. **`predict.main()` 也一样 fail-soft**，崩了保留 stub。

效果：

| 提交结果 | 说明 |
|---------|------|
| 公榜分 ≈ 模型真实水平 | 全栈跑通 |
| 公榜分 ≈ 0.5 (baseline AUC) | predict 路径某一步崩了，但 stub 保住了 valid submission，可以继续 iterate |
| **不再出现** "Notebook Threw Exception" | 除非 cell 1 也崩（pandas + 读 csv 这两件极少失败） |

**核心教训**：在 code competition 里追"为什么 rerun 崩了"是死路。改用"score 即调试通道"——保证每次都拿到 valid submission，从分数推断哪一层有问题。

### 3.8 单个坏 .ogg 让整个 inference loop 死

`predict.py` 的 file loop 之前没 try/except `load_audio`——hidden test 里只要有一个 codec 损坏的 .ogg，整个 loop 就 raise 出来，全栈失败。

**修复**（同 commit `323201c`，`predict.py:240-247`）：

```python
for path, group_rows in tqdm(grouped.items()):
    try:
        waveform = load_audio(path, sample_rate)
    except Exception as e:
        n_load_fail += 1
        print(f"[infer] skip (load failed) {path}: {type(e).__name__}: {e}")
        continue
    ...
```

### 3.9 timm `pretrained=True` 在离线环境炸

`SEDModel.__init__` 默认 `pretrained=True`，timm 会去网络拉权重。Kaggle kernel `enable_internet: false`，离线 → 立刻报错。

**修复**（commit `5535473`，`predict.py:73-83`）：load checkpoint 时把 cfg 里的 `model.pretrained` 强制改成 False，因为 state_dict 反正会覆盖：

```python
cfg = dict(state.get("config") or fallback_cfg)
cfg["model"] = dict(cfg.get("model") or {})
cfg["model"]["pretrained"] = False
cfg["model"]["pretrained_backbone_path"] = None
model = build_model_from_config(cfg, num_classes=...)
```

### 3.10 `| tail -50` 把流式日志全 buffer 住

不算 Kaggle bug 但是常见操作坑：

```bash
uv run python scripts/17_kaggle_eB1_submit.py 2>&1 | tail -50  # ❌ 全程没输出，结束才打
```

`tail -50` 会等 stdin EOF 才输出最后 50 行。脚本跑 1h 全程黑屏。

**改成**：

```bash
nohup bash -c '... && uv run python scripts/17_kaggle_eB1_submit.py' > /tmp/submit.log 2>&1 < /dev/null &
disown
# 然后 tail -f /tmp/submit.log 实时跟
```

---

## 4. 操作快速参考

### 4.1 一次性环境准备（每台 L40 一次）

```bash
# kaggle CLI（legacy）
uv tool install kaggle

# kagglehub（PAT 认证）
cd /workspace/birdclef-2026 && uv pip install kagglehub

# 双 token 都要放
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
# PAT
echo "<paste 38-char access token>" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
# Legacy json (从 kaggle.com/settings 下载)
mv /tmp/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 4.2 一次新模型提交（端到端）

1. 模型训练完，checkpoint 落在 `outputs/exp/<exp_id>/fold_<k>/swa.pt`。
2. 复制 `kaggle/variants/b01_eB1_nfnet_l0/` 到 `kaggle/variants/<新名字>/`，改两个文件：
   - `kernel-metadata.json`：`id`、`title`、`dataset_sources`
   - `*.py`：`PKG_INPUT` 路径、`CKPT` 文件名、必要时 `--batch-size` / `--amp-dtype`
3. 改 `scripts/17_kaggle_eB1_submit.py` 顶部的 `CKPT_REL`、`KERNEL_DIR`、`DATASET_HANDLE`（或者写一个新的脚本，复用同样的 5 个函数）。
4. 跑：

```bash
cd /workspace/birdclef-2026 && git pull && uv run python scripts/17_kaggle_eB1_submit.py
```

5. 脚本结束后输出 kernel URL。Kaggle 自动跑一次 Save & Run（5-10 min）。
6. 跑完去 kernel 页面右上角 **"Submit to Competition"**。等 hidden test rerun（10-30 min for our setup）。
7. 在 https://www.kaggle.com/competitions/birdclef-2026/submissions 看分数。

### 4.3 只 push kernel，不重传 dataset

Dataset 90 MB 重传慢。如果只是改 notebook 不改 checkpoint：

```bash
uv run python scripts/17_kaggle_eB1_submit.py --kernel-only
```

### 4.4 检查 kernel 状态 / 拿日志

```bash
# 状态
kaggle kernels status winbeaux/<slug>

# 拉最新 run 的 log + outputs（注意：只能拿 interactive run，rerun 拿不到）
mkdir -p /tmp/kernel_out
kaggle kernels output winbeaux/<slug> -p /tmp/kernel_out
cat /tmp/kernel_out/<slug>.log
```

### 4.5 看提交列表 / 错误 class

```bash
# 表格视图
kaggle competitions submissions birdclef-2026 | head -10

# 通过 kagglesdk 拿到结构化字段（包括 error_description，但都是通用 class）
KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token) uv run python -c "
from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
with KaggleClient() as c:
    req = ApiListSubmissionsRequest(); req.competition_name = 'birdclef-2026'
    for s in list(c.competitions.competition_api_client.list_submissions(req).submissions)[:3]:
        print(s.ref, s.status, s.public_score, s.error_description[:80] if s.error_description else '')
"
```

---

## 5. Notebook 模板（fail-soft）

每个新提交的 .py（jupytext 格式）应该长这样：

```python
# %% cell 1 — 立刻写 zero-fallback submission.csv（不依赖任何 wheels）
from pathlib import Path
import pandas as _pd
_KAGGLE_INPUT = Path("/kaggle/input/competitions/birdclef-2026")
if not _KAGGLE_INPUT.exists():
    _KAGGLE_INPUT = Path("/kaggle/input/birdclef-2026")
_OUT = Path("/kaggle/working/submission.csv")
_stub = _pd.read_csv(_KAGGLE_INPUT / "sample_submission.csv")
for _c in _stub.columns[1:]: _stub[_c] = 0.0
_stub.to_csv(_OUT, index=False)
print(f"[stub] {len(_stub)} rows × {len(_stub.columns)} cols")

# %% cell 2 — pip install offline wheels (try/except + sys.exit(0))
import glob, subprocess, sys, traceback
try:
    _req = glob.glob("/kaggle/input/**/birdclef-2026-download-wheels/requirements.txt", recursive=True)[0]
    subprocess.check_call(["pip", "install", "-q", "--no-index", "-r", _req,
                          "--find-links", _req.replace("/requirements.txt", "/wheels")])
    print("[pip] OK")
except BaseException as e:
    print(f"[FATAL pip] {type(e).__name__}: {e}"); traceback.print_exc()
    sys.exit(0)

# %% cell 3 — 包导入 + 路径 (try/except + sys.exit(0))
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 必须在 import torch 之前
try:
    sys.path.insert(0, "/kaggle/input/<your-pkg-dataset>/src")
    import torch
    from yourpkg.inference import predict as _pm
    # ... 其他 setup
    print("[setup] OK")
except BaseException as e:
    print(f"[FATAL setup] {type(e).__name__}: {e}"); traceback.print_exc()
    sys.exit(0)

# %% cell 4 — predict (try/except, **不** sys.exit, 让 stub 保留)
try:
    sys.argv = ["predict", "--checkpoints", "...", "--output", str(_OUT),
                "--batch-size", "4", "--amp-dtype", "fp32"]
    _pm.main()
    print("[predict] OK")
except BaseException as e:
    print(f"[FATAL predict] {type(e).__name__}: {e}"); traceback.print_exc()
    print("[fallback] keeping zero stub")

# %% cell 5 — QC（不影响 submission，可选）
sub = _pd.read_csv(_OUT)
print(f"final shape: {sub.shape}")
print(f"prob max: {sub.iloc[:,1:].to_numpy().max():.4f}  (0.0 = stub used)")
```

---

## 6. 提交结果解读速查

提交后按公榜分判断哪一层出问题：

| 公榜分 | 含义 | 下一步 |
|--------|------|--------|
| ≈ 模型真实 LB（>0.5 显著） | 全栈跑通 | 看分数决策（够不够好） |
| **≈ 0.5** | predict 在 hidden test 崩了，stub 兜底 | 看 cell 4 之前的 stage 是不是都打了 OK，缩小范围；最常见是 audio loading / forward 的 corner case |
| **0.5 < x < 模型 val** | 部分 audio 加载失败被零填，还有部分推理成功 | 检查 `[infer] WARN: N audio files failed to load` 信息（在 interactive log），N 大就是音频解码问题 |
| **"Notebook Threw Exception"** | cell 1 之前都崩了，极罕见 | 看 base image 是不是变了；pandas / 路径问题 |
| **"Submission CSV Not Found"** | submission.csv 路径不对 | 检查 cell 1 是否真的 chdir 到 /kaggle/working/，或者 path 写错 |
| **"Notebook Timeout"** | 9h cap | batch_size 降一半 / 单折跑就完事 |
| **"Notebook Exceeded Allowed Compute"** | RAM 爆 | batch_size 再降 / 用 fp16 |

---

## 7. 已知不能用 / 不要尝试的路径

- ❌ **PAT + kaggle CLI**：CLI 启动 `authenticate()` 在 hello 阶段就要找 kaggle.json，没有就直接报 `OSError`，KAGGLE_API_TOKEN env 没用。
- ❌ **PAT + kagglesdk 调 kernels API**：scope 不够，403/409。
- ❌ **kagglehub 推 kernel**：1.0.1 没这个 API，`dir(kagglehub)` 里压根没有 kernel/notebook upload。
- ❌ **`| tail -N` 跟踪长任务进度**：tail 等 EOF 才输出，全程黑屏。改用 `nohup ... > log 2>&1 &; tail -f log`。
- ❌ **enable_gpu: true 用 P100**：torch wheel sm_70+ 不带 sm_60。要么 CPU，要么显式选 T4（如果该 comp 允许选 GPU type）。
- ❌ **指望 rerun 的 traceback**：官方 anti-probing，永远拿不到。

---

## 8. 改进 backlog（不是本次提交范围，备查）

- 把 dataset_upload 改成增量（只上传 mtime 变了的文件）：现在每次重跑都重传 90 MB。
- 把 kernel push 包成 `kaggle/variants/<slug>/Makefile`，每个 variant 独立可跑。
- 加一个 `scripts/18_check_submission.py`，自动列出最近 N 条提交、status、score、error class。
- Kernel notebook 里加 timing instrumentation，把每个 stage 耗时打印——以后 timeout 时知道是哪一步慢。

---

附：commit 关键节点

| commit | 修复 |
|--------|------|
| `14c761b` | 初版 submit infra（直接调用 kaggle CLI，PAT 路径，全部失败） |
| `37f4004` | 切到 split-stack: kagglehub for dataset + kagglesdk for kernel (后来发现 kernels 不能 PAT) |
| `d835ebe` | IPv4-only socket monkey-patch 解决 GCS upload hang |
| `d4e881b` | `resp.url()` → `resp.url` 修 KaggleObject TypeError |
| `c649e2d` | id alignment to title-slugified value，解 409 |
| `62abc9c` | 放弃 kagglesdk for kernel，回 legacy `kaggle kernels push` |
| `0f4d5c4` | CPU-only inference，避开 P100 sm_60 |
| `2c99839` | `--kernel-only` flag 跳过 dataset 重传 |
| `2ca93d7` | predict.main() try/except + traceback 保留 |
| `323201c` | 完整 fail-soft：cell 1 zero-stub + 每个 cell try/except + per-file load_audio 容错 |
