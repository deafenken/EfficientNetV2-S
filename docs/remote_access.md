# Remote access: tmux 链路 + `rmt` 命令手册

VPS 上的助手怎么去 GPU 容器里跑命令、读日志。日常入口是 `rmt`；底层是一条
持久化的 tmux session 把三跳 ssh 链常驻。本文一份说清楚两件事：

1. **链路怎么搭起来**（tmux session 'remote' 里的三跳）
2. **`rmt` 子命令怎么用**（日常 90% 的操作）

---

## TL;DR

```bash
rmt up                                  # 第一次或者链断了之后建链
rmt run "cd birdclef-2026 && suv && nvidia-smi"
rmt tail 80                             # 看当前 pane 最后 80 行
rmt attach                              # 接管，Ctrl+B D 退回给 Claude
rmt down                                # 杀掉 session（链彻底坏了用）
```

`rmt run` 默认 5 min 超时，跑长任务覆盖：`R_TIMEOUT=1800 rmt run "..."`。

---

## 拓扑

```
┌──────────────────┐  tailscale  ┌────────────────┐  Ivanti VPN  ┌──────────────────┐  docker exec  ┌────────────────┐
│ selabvps-pro     │ ──────────▶ │ win-wsl2       │ ───────────▶ │ target-server    │ ─────────────▶ │ container      │
│ 100.78.102.96    │  port 2222  │ 100.78.122.53  │   (走 Win    │ 222.29.98.132    │                │ t2v-eval       │
│ (Claude 在这)    │             │ (WSL 里 sshd)  │   主机 NAT)  │ port 30902       │                │ image v4       │
└──────────────────┘             └────────────────┘              └──────────────────┘                └────────────────┘
                                                                  user: intern                        zsh + uv (suv)
```

3 跳。每跳的鉴权方式 / 注意事项见底部 [各跳明细](#各跳明细)。

---

## `rmt` 工具

### 是什么

VPS 上的 bash 包装器：`~/.local/bin/rmt`（仅本机有，未提交进 repo —— 是 Claude
工作环境的一部分）。它在 tmux 里维持一个名为 `remote` 的 session，session 里
依次 `ssh -tt → ssh -tt target-server → docker exec t2v-eval zsh`，把三跳一次
握好。后续每次 `rmt run` 都是 `tmux send-keys` + marker 等待 + `capture-pane`
抓输出，省掉每次 ~3-4s 的 ssh/Ivanti/docker 握手。

> 名字避开了 zsh 内建别名 `r`（zsh 里 `r ≡ fc -e -`），所以叫 `rmt`。

### 子命令

| 子命令 | 作用 | 何时用 |
|---|---|---|
| `rmt up` | 幂等创建 session 并建三跳链 | 首次使用、`rmt run` 报 `no session`、Ivanti 被踢之后 |
| `rmt run "<cmd>"` | 在远端跑 `<cmd>`，等到完成后打印 stdout/stderr | 日常入口，单次命令尽量自包含 |
| `rmt tail [N]` | 看当前 pane 最后 N 行（默认 40），**不发命令** | 排查上一次 `rmt run` 出了什么、看后台任务的进度 |
| `rmt attach` | 你接管 tmux session，可看实时输出 / 手敲命令 | 调试、跑交互式工具；Ctrl+B D detach 还给 Claude |
| `rmt down` | kill 掉 session | 链彻底崩了重建之前用 |

### 状态约定

- Shell 状态（cwd / env / 半成品输入）**跨 `rmt run` 保留**。所以：
  - 单次命令尽量自包含（`cd ... && suv && <cmd>`），别依赖上次留下的 cwd。
  - 如果你看见命令输出里多了一堆环境变量或 prompt 残骸，就是上一条命令没收尾干净。
- `rmt run` 用唯一的 `__R_S_<rid>__` / `__R_E_<rid>__` marker 划输出边界，
  跟 prompt echo / 历史记录隔离。
- 链失效的典型信号：`rmt run` 卡住到 timeout、`rmt tail` 看见 `Connection
  closed` 或 docker `Exited (1)`。处理：`rmt down && rmt up`。

### 常用调用模板

```bash
# 看一眼远端是不是活的
rmt run "hostname && date"

# 跑训练前的环境检查
rmt run "cd birdclef-2026 && suv && uv --version && nvidia-smi"

# 启动后台训练（注意 nohup + 重定向，否则 tmux pane 会卡住等 fg）
rmt run "cd birdclef-2026 && suv && nohup python train.py > /tmp/train.log 2>&1 & echo PID=\$!"

# 看训练日志
rmt run "tail -50 /tmp/train.log"

# 找训练进程
rmt run "pgrep -af torchrun"

# 长任务（>5min）需要覆盖超时
R_TIMEOUT=1800 rmt run "cd birdclef-2026 && suv && python preprocess.py"
```

### 什么时候不要用 rmt

- **流式监控**：`rmt run` 是阻塞抓输出的，不适合长时间 `tail -f`。要持续看
  日志的，用 `Monitor` 工具配合 `ssh -p 2222 ... "ssh target-server 'docker
  exec t2v-eval ...'"` 直连一次拿流。
- **真正的交互**（vim、pdb、`ipython`）：`rmt attach` 进 tmux 自己敲。
- **会改 `remote` session shell 状态的命令**（比如 `set -e`、`exec ...`）：
  会污染后续 `rmt run`。需要的话先 `rmt down && rmt up` 重置。

---

## 链路怎么从零搭起来

正常情况下 `rmt up` 一条命令搞定。下面这部分是**底层细节**，链断了排查时看。

### 各跳明细

#### Hop 1: VPS → win-wsl2（tailscale 内网，sshd on 2222）

- WSL 的 `sshd` 监听 **2222**（不是 22）。
- 鉴权：VPS 上 `~/.ssh/id_ed25519`，公钥已在 WSL 的 `~/.ssh/authorized_keys` 里：
  ```
  ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPT8y/c5rE7lf+msRhhxwnWRb6MwvDjPfqvGXS9zHUSn winbeau@selabvps-pro
  ```
- 单跳验证：
  ```bash
  ssh -p 2222 winbeau@100.78.122.53 'hostname && whoami'
  # 期望: localhost / winbeau
  ```
- 走 tailscale，不依赖校园网 / VPN，随时可跑。

#### Hop 2: win-wsl2 → target-server

- WSL 的 `~/.ssh/config` 别名：
  ```
  Host target-server
      HostName 222.29.98.132
      Port 30902
      User intern
      IdentityFile ~/.ssh/id_ed25519
      # ProxyCommand ssh -W %h:%p jump-box   # 只在没连 VPN 时打开
  ```
- 走 Windows 主机的 Ivanti 校园 VPN 出口。**Ivanti 是单会话**：同一账号
  同时只能一个会话，被别处登录会顶掉（曾经被 Beijing 联通 IP
  114.248.18.238 顶过）。
- **不会被我的 ssh hop 影响**，可以放心嵌套。

#### Hop 3: target-server → docker 容器

- 容器名 `t2v-eval`，镜像 `winbeau-dev:v4`。
- `rmt up` 进容器用 `docker exec -it t2v-eval zsh`（tmux 内有 PTY，可以 `-it`）。
- 手工嵌套 ssh 时**不能用 `-it`**（没 PTY），改用 `docker exec -i t2v-eval zsh -ic '<cmd>'`：
  - `-i`：保留 stdin。
  - `zsh -ic`：interactive 让 `~/.zshrc` 加载，`suv` 等 alias 才有用。
- 容器里关键 alias：`suv` —— 加载 uv 的 PATH/env，`uv` 才能找到。

### 完整一行命令（不走 rmt 时验证用）

```bash
ssh -p 2222 winbeau@100.78.122.53 \
  "ssh target-server 'docker exec -i t2v-eval zsh -ic \"cd birdclef-2026 && suv && which uv && uv --version && pwd\"'"
```

quoting 三层（外 `""`、中 `''`、内 `\"\"`），层级和引号类型别乱。

---

## 故障排查

### `rmt run` 报 `no session`
链没建起来。`rmt up` 即可。

### `rmt run` 一直卡到 timeout
链中间某跳挂了。`rmt tail 80` 看最后输出找原因（常见：Ivanti 被顶、容器
`Exited`、ssh `Connection closed`）。然后 `rmt down && rmt up`。

### "已经有一个连接" / Ivanti 被踢
Ivanti 单会话被人顶掉，**不是我的 ssh 引起的**。处理顺序：
1. 看下是不是另一台机器（朋友 / 自己别处）正在用同一账号。
2. Windows 上重新连 Ivanti（断开 → 重连，必要时重启 Pulse Secure 服务）。
3. 等 1–2 分钟服务器侧 session 缓存过期。
4. `rmt down && rmt up` 重建链。

### 容器 `Exited (1)`
- 一般是上一个 `docker exec -it ... zsh` 的终端被关闭，PID 1 的 zsh 收到
  SIGHUP 退出。
- 处理：`ssh target-server 'docker start t2v-eval'`，再 `rmt down && rmt up`。
- 排查：`ssh target-server 'docker logs --tail 30 t2v-eval'`。

### setlocale warning
每跳都会冒一行 `bash: warning: setlocale: LC_ALL: cannot change locale
(en_US.UTF-8)`。无害，忽略。

---

## 不要做的事

- 不要在嵌套 ssh（非 tmux）里要 PTY (`-t` / `docker exec -it`)，没法分配。
- 不要 force / disconnect 强抢 Ivanti 会话，先确认没人在用。
- 不要在 VPS 上跑训练 / 装大依赖（资源限制，详见 `~/.claude/CLAUDE.md`）。
  重活都在 target-server 容器里跑。
- 不要在 `rmt run` 里跑前台长任务（>5min 又没 nohup），pane 会卡住。改用
  `nohup ... > /tmp/x.log 2>&1 &` + 后续 `rmt run "tail /tmp/x.log"` 轮询。

---

## Claude 权限

`.claude/settings.local.json` 里把 `Bash(rmt *)` 和 `Bash(tmux *)` 加进
`permissions.allow`，自动模式下不再每次提示。

---

## 长期方案（可选）

完全摆脱 Ivanti 单会话折腾的办法：在 target-server 上也装 tailscale 客户端，
加进同一个 tailnet，VPS 就能直连 target-server，整条链缩成两跳，且不依赖
Ivanti。
