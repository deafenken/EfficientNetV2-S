# Remote access chain (VPS → WSL → target-server → docker)

How the assistant on the shared VPS reaches the GPU box and runs commands inside the dev container.

## 拓扑

```
┌──────────────────┐  tailscale  ┌────────────────┐  Ivanti VPN  ┌──────────────────┐  docker exec  ┌────────────────┐
│ selabvps-pro     │ ──────────▶ │ win-wsl2       │ ───────────▶ │ target-server    │ ─────────────▶ │ container      │
│ 100.78.102.96    │  port 2222  │ 100.78.122.53  │   (走 Win    │ 222.29.98.132    │                │ t2v-eval       │
│ (assistant 在这) │             │ (WSL 里 sshd)  │   主机 NAT)  │ port 30902       │                │ image v4       │
└──────────────────┘             └────────────────┘              └──────────────────┘                └────────────────┘
                                                                  user: intern                        zsh + uv (suv)
```

Hop 数：3。每一跳的鉴权方式 / 注意事项见下。

## 各跳明细

### Hop 1: VPS → win-wsl2（tailscale 内网，sshd on 2222）

- WSL 的 `sshd` 监听 **2222**（不是 22），原因不明，记下来就行。
- 鉴权：VPS 上 `~/.ssh/id_ed25519`，公钥已加进 WSL 的 `~/.ssh/authorized_keys`：
  ```
  ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPT8y/c5rE7lf+msRhhxwnWRb6MwvDjPfqvGXS9zHUSn winbeau@selabvps-pro
  ```
- 验证命令：
  ```bash
  ssh -p 2222 winbeau@100.78.122.53 'hostname && whoami'
  # 期望: localhost / winbeau
  ```
- 这一跳完全走 tailscale，不依赖校园网 / VPN，可以随时跑。

### Hop 2: win-wsl2 → target-server

- WSL 的 `~/.ssh/config` 里有别名：
  ```
  Host target-server
      HostName 222.29.98.132
      Port 30902
      User intern
      IdentityFile ~/.ssh/id_ed25519
      # ProxyCommand ssh -W %h:%p jump-box   # 只在没连 VPN 时取消注释
  ```
- 走 Windows 主机的 Ivanti 校园 VPN 出口。**Ivanti 是单会话**：同一账号同时只能一个会话，被别处登录会顶掉（曾经被 Beijing 联通 IP 114.248.18.238 顶过）。
- 不会被我的 ssh hop 影响，可以放心嵌套。

### Hop 3: target-server → docker 容器

- 容器名 `t2v-eval`，镜像 `winbeau-dev:v4`。
- 进入方式：`docker exec -i t2v-eval zsh -ic '<cmd>'`
  - `-i`：保留 stdin。
  - `zsh -ic`：interactive 模式让 `~/.zshrc` 加载，`suv` 等 alias 才能用。
  - **不要用 `-it`** 在嵌套 ssh 里：会因为没分配 PTY 而报错。
- 容器里的关键 alias：`suv` —— 加载 uv 环境变量（PATH 之类）。先 `suv` 才能找到 `uv`。

### 完整一行命令（验证用）

```bash
ssh -p 2222 winbeau@100.78.122.53 \
  "ssh target-server 'docker exec -i t2v-eval zsh -ic \"cd birdclef-2026 && suv && which uv && uv --version && pwd\"'"
```

quoting 用了三层（外 `""`、中 `''`、内 `\"\"`），层级和引号类型不要乱。

## 常见问题

### "已经有一个连接" 提示

Ivanti 单会话被人顶掉，不是我的 ssh 引起的。处理顺序：
1. 看下是不是另一台机器（朋友 / 自己别处登录）正在用同一账号。
2. Windows 上重新连 Ivanti（断开 → 重连，必要时重启 Pulse Secure 服务）。
3. 等 1–2 分钟服务器侧 session 缓存过期。

### 容器 `Exited (1)`

- 一般是上一个 `docker exec -it ... zsh` 的终端被关闭，PID 1 的 zsh 收到 SIGHUP 退出。
- 处理：`ssh target-server 'docker start t2v-eval'`。
- 排查：`ssh target-server 'docker logs --tail 30 t2v-eval'`。

### setlocale warning

每跳都会冒一行 `bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)`。无害，忽略。

## 不要做的事

- 不要在嵌套 ssh 里要 PTY (`-t` 或 `docker exec -it`)，没法分配。
- 不要用 force / disconnect 强抢 Ivanti 会话，先确认没人在用。
- 不要在 VPS 上跑训练 / 装大依赖（资源限制，详见 `~/.claude/CLAUDE.md`）。重活都在 target-server 容器里跑。

## 长期方案（可选）

如果想完全摆脱 Ivanti 单会话的折腾：在 target-server 上也装 tailscale 客户端，加进同一个 tailnet，VPS 就能直连 target-server，整条链缩成两跳，且不依赖 Ivanti。

## VPS 端 tmux 缓存：`rmt` 包装器

每次 ssh 链都重握 3 次手太慢（~3-4s/次），所以在 VPS 上常驻一个 tmux session
名为 `remote`，里面挂着已经打通的三跳 shell。后续命令通过 `tmux send-keys`
注入 + marker 等待 + `tmux capture-pane` 抓输出，单次 < 0.5s。

### 工具
- 脚本：`~/.local/bin/rmt`（仅 VPS 上有，未提交到 repo —— Claude 工作环境的一部分）
- 名字避开 zsh 内建别名 `r`（zsh 里 `r` ≡ `fc -e -`）

### 子命令
| 子命令 | 用途 |
|---|---|
| `rmt up` | 幂等创建 session 并建链（VPS → WSL → target-server → docker） |
| `rmt run "<cmd>"` | 发送命令、等待完成、打印输出（默认上限 5 min，可 `R_TIMEOUT=...` 覆盖） |
| `rmt tail [N]` | 看当前 pane 最后 N 行（默认 40），不发送命令 |
| `rmt attach` | 你接管 session 看实时输出，Ctrl+B D detach 还给 Claude |
| `rmt down` | kill session（链断了重来时用） |

### 状态约定
- Shell 状态（cwd / env / 半成品输入）跨命令保留 —— 单次命令尽量自包含。
- Ivanti 被踢 / 容器重启 / WSL 重启 → 链失效，`rmt down && rmt up` 重建。
- `rmt run` 内部用唯一的 START/END marker 做输出边界，与 prompt echo / 历史
  记录隔离。

### Claude 权限
`.claude/settings.local.json` 里把 `Bash(rmt *)` 和 `Bash(tmux *)` 加进
`permissions.allow`，自动模式不再每次提示。
