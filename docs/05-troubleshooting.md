# 05 · 排障手册（症状 → 根因 → 修法）

> **English abstract** — Symptom-indexed troubleshooting for this port. Nearly every entry here looked like
> something else at first: "the engine started but never answers" was a hot-path `memory_reserved()` call;
> "ninja rule name is wrong" was an unescaped drive letter; "the model is broken" was a Windows Firewall
> Public-profile Block rule; "scp succeeded" was a truncated 2.2 GB file.

按"你看到的现象"查，不要按"你以为的原因"查。每条的根因都经过 A/B 验证。

---

## A. 根本起不来 / 编译类

### A1. `cudafe++ died with 0xC0000005` 或 `unsupported Microsoft Visual Studio version`

**根因**：CUDA 13 只支持 VS 2019–2022；机器上装了更新的 VS（2026），`vswhere -latest` 会挑到它。
**修法**：

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
set DISTUTILS_USE_SDK=1
set HIBANA_MSVC_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
```

**验证**：`cl` 输出 `19.4x`（MSVC 14.4x），不是 `19.5x`。

### A2. ninja 报 `expected build command name`（build 行看起来完全正常）

**根因**：**盘符没转义**。`build C:\t\out.o: cc C:\t\in.c` 会被 ninja 解析成"规则名 = `C`"。
错误信息**完全不指向盘符**，极易误判成规则写错。
**修法**：`C$:` —— `build C$:/t/out.o: cc C$:/t/in.c`。
**注意**：官方 ninja 1.13.1 与 PyPI 包内那份行为一致，**换 ninja 不解决**。

### A3. `ptxas error: Instruction 'cvt with .e2m1x2' not supported on .target 'sm_120'`

**根因**：FP4 指令只在 **family 架构** `120a` / `120f` 上可用；写普通 `sm_120` 就没有。
**修法**：`HIBANA_CUDA_ARCH=12.0a`，并确认 tvm_ffi 构建器传的是 `compute_120f/sm_120f`（它默认只按设备推 `120`）。
**为什么这个坑很隐蔽**：编译能过一半，到 `ptxas` 才炸，且只影响 FP4 相关的内核。

### A4. `-Xcompiler` 传多个 flag 时静默失效

**根因**：MSVC 下 `-Xcompiler -O2 -Xcompiler -MD` 的语义与 gcc 不同。
**修法**：合并成 `-Xcompiler=/O2,/MD` 单参数。

### A5. device 代码里 `std::optional::value_or` 编译失败

**根因**：MSVC 不允许在 device 代码里调 `value_or`。
**修法**：`--expt-relaxed-constexpr`，并把该头文件里的 `value_or` 改成 `has_value() ? *x : 0`。

### A6. `__attribute__((always_inline))` / `ushort` / `uint` / `M_PI` 未定义

**根因**：GNU 扩展与 POSIX 头在 MSVC 下不存在。
**修法**：`port/win_compat.h`（`ushort/uint/ulong`、`M_PI` 系列、`__restrict__`）通过 `--pre-include` / `/FI` 强制包含；
`__attribute__((always_inline))` → `__forceinline__`。

### A7. flashinfer 的 prefill 在某些形状下 OOM

**根因**：workspace 默认 128/256 MB，而 fp8 KV + head_dim 256 的 prefill 临时区实测要 **282 MiB**。
**修法**：workspace 提到 **512 MB / 1 GiB**。

---

## B. 服务起来了但不回话（最像"玄学"的一类）

### B1. `/health` 200、日志正常、但请求永远超时或返回空

**根因**：`can_admit()` 是**每步都调**的准入函数，而它调了 `torch.cuda.memory_reserved()` ——
这个函数会遍历分配器所有 segment，本机要**几秒**。结果引擎线程 100% CPU 空转，请求永远进不来。
**修法**：准入改用驱动 `mem_get_info()` + **250 ms TTL 缓存**。
**怎么确认**：`Get-Process python | Select CPU` 会看到引擎进程 CPU 打满但 GPU 空闲。

### B2. 显存明明够，却一路分到 200 GiB 才 OOM；`free` 恒为 0

**根因**：**WDDM 会超额提交**，且驱动把 `free` 报成 0。用 `free` 做容量判断必错。
**修法**：池预算 = `总显存 − torch 已 reserved − HIBANA_VRAM_RESERVE_GIB`，
且用 `memory_reserved()`（要缓存，见 B1），**不要用驱动的 free**。

### B3. `verify_state=snapshots` 一开，请求全堵在准入（`rounds=0`）

**根因**：snapshots 要给每个请求留一份状态副本，96 GiB 上要多占 ~15 GiB，和 3-bit 专家抢空间。
**修法**：回退到默认 `reconstruct`（它慢 2.8 ms/步，但装得下）。要开就得接受 `max_batch` 降到 8。

### B4. 局域网（另一台机器）连不上，本机 curl 正常

**根因**（本项目实测过的一例）：**Windows 防火墙**。192.168.1.x 那张网卡没有网关 ⇒ 被判成
「未识别的网络」= **Public** profile；而该服务的入站规则里带两条 **Public Block**，Block 优先于 Allow
⇒ 对方 SYN 被静默丢弃（表现为"请求卡住"，不是拒绝连接）。
**修法**（管理员）：

```powershell
# 1) 看该程序有没有 Block 规则
Get-NetFirewallRule -DisplayName "*<程序名>*" | Format-Table DisplayName,Profile,Action,Direction
# 2) 停用 Public Block，按端口放行指定网段
New-NetFirewallRule -DisplayName "LAN TCP 30004 (QRK)" -Direction Inbound -Action Allow `
  -Protocol TCP -LocalPort 30004 -RemoteAddress 192.168.1.0/24 -Profile Any
```

**A/B 已验因果**：重新启用 Public Block → 立刻超时；停用 → 立刻 200。
**改前先备份**：`netsh advfirewall export C:\fw-backup.wfw`（`netsh advfirewall import` 可回滚）。

### B5. 换模型/换服务后同一条 lane "没反应"

**根因**：一个端口同一时刻只服务**一个**进程启动的那个模型。切了 bat 但没重启 = lane 是死的；
DSH/客户端看起来就是"没反应"（子代理直接返回 null）。
**修法**：先 `/v1/models` 看当前是谁，再决定。别在客户端 debug。

---

## C. 质量 / 输出异常

### C1. prompt ≥1024 token 时引擎崩（`hyper_prefill.mix` 的 `_scaled_mm` 在 SM120 不支持）

**根因**：引擎既有 bug，`--context-length 262144` 实际只用得上单 chunk。
**修法**：本项目修到第二层（`qsa_score_exact` 的 JIT 链接失败，CUDA 13 + MSVC）后**回退**，未闭环。
**现状**：≤1 K prompt 正常；更长的对话能用，但 prefill 会走慢路径。

### C2. 长会话（~50 K token）每轮全量重算，`cached=0`，TTFT 30 s+

**根因**（未完全闭环，两个怀疑）：① prompt cache 只有 4.2 GiB，装不下单条 50 K 的状态；
② 系统提示里有易变块，前缀在 21 776 token 处断掉。
**现状**：这是**已知未解决**的长上下文问题。短会话不受影响。

### C3. 带图首轮很慢

**根因**：视觉塔 + 图片解码冷启动。
**现状**：首轮 ~6.1 s，之后 **0.08–0.16 s/轮**（全分辨率 2723 image token），
靠 96 MB 图片解码缓存 + 320 MB 视觉塔输出缓存（按图片指纹）。
**依赖**：JPEG/PNG 都要 `libturbojpeg`（本项目放在 `/jpeg/libturbojpeg.so.0`）。

### C4. 中文比英文慢 20%+

**根因**：中文的 `tok/round` 更低（2.40–2.59 vs 3.31–4.26），即中文草稿接受率更低（0.483–0.509）。
**这是模型/草稿头特性，不是配置问题**。且中文 token 信息密度更高，按"字/秒"算并不吃亏。

---

## D. 传输 / 存储 / 环境

### D1. `scp` 显示成功，但文件是半截的

**根因**：**目标盘写满时 `scp` 会写出截断文件且不明显报错**。
**修法**：传完必须比对**字节数 + md5**（`5 214 305 424 B` 这种精确值）。
我们差点用 2 244 MB 的截断 safetensors 启动引擎。

### D2. 模型放在 USB 盘上慢到不可用

**实测**：M:/K:/L: 这类 USB 盘 4–20 MB/s；C/B 的 NVMe 是 200 MB/s 级。
**规则**：权重、PLE 表、页缓存都必须在**本地 NVMe**。

### D3. `torch/lib` 里 DLL 加载失败 / 路径过长

**根因**：中文用户名 + `%USERPROFILE%` 深层路径 + Windows 260 字符限制。
**修法**：venv 放短路径（`B:\hibana-win\.venv`），不要放用户目录。

### D4. `HIBANA_HF_CACHE` 指向的目录结构不对，引擎找不到权重

**根因**：引擎认 **HF 缓存布局**（`hub/models--<org>--<name>/snapshots/<rev>/`），
`hf download --local-dir` 摊平的结构不认。
**修法**：按缓存布局落盘（或让 hf 用默认 cache 目录后整体搬过去）。

### D5. 重启脚本"跑成功了但服务没起来"

**根因**：`schtasks` 在权限/账户不匹配时会**静默 no-op 并返回成功**。
**修法**：`launch/restart.ps1` 的做法是 ——
`schtasks /end` + `/delete` 清掉旧任务，重新注册并 `/run`，然后**轮询 `/health` 到 200 才返回**。
**永远不要用固定 `sleep` 猜启动时间**。

### D6. 远端 `cmd //c ...` 什么都不做还返回 exit 0

**根因**：`cmd //c` 是本机 MSYS 的写法；Windows 上会**静默起一个交互式 cmd**，什么都不执行。
**修法**：Windows 上直接 `cmd /c`；远端 shell 的命令分隔符是 `&` 不是 `;`。

---

## E. 性能突然变差（先查这三件事，再查代码）

### E1. 有没有 co-tenant 在抢卡

本机实测过：另一个常驻服务（Strata）占 **~77 GB**；ComfyUI 占 **48 GB**。
它们**不换页**，抢的是**带宽**：同时跑生成任务时，推理服务被压慢到 **1/9**（并发 4 仍掉到 47–48 tok/s）。

```powershell
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

**修法**：ComfyUI 用 `POST /free`（释放缓存、不杀进程、不动文件）；Strata 要退出。

### E2. PLE 页缓存是不是冷的

刚开机 / 刚重启 / 刚清理过缓存时，c8 只有 129 tok/s（热的是 171+）。
**修法**：`launch/run-flashnext.bat` 已经在启动前跑 `preload_ple.ps1`；手工确认见 `preload.log`。

### E3. 是不是冷启动 + 图还没捕获

图捕获按桶**惰性**发生。五点测量必须先暖机（每点 2 轮丢弃），否则 c1 会低 10–20%。

---

## F. 小卡（<96 GB）怎么办 {#smaller-cards}

这套 3-bit profile 常驻 53 GiB + KV/状态池，**32 GB 卡装不下**。可选路径（按可行性排序）：

1. **降到更小上下文 + `max_batch 1` + 缩小 slots**：能跑起来，但 PLE 表仍要 47.7 GB 主机内存。
2. **换更小的模型**（本移植的补丁与工具链对同架构的 27B/30B 档同样适用，改 `HIBANA_W3_CHECKPOINT` 即可）。
3. **不要指望把 3-bit 专家再压**：46.1 GiB 已经是这层的下限，再压要重写内核。
4. **显存不够时先砍 KV**（`HIBANA_KV_POOL_TOKENS` 262144 → 131072），它比专家权重更容易让。

> 结论：**"同一张显卡"是硬约束**。这份仓库不承诺在其它卡上开箱即用，只承诺移植方法可迁移。
