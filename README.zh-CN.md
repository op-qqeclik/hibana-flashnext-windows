# Hibana on Windows（中文版）

**在一块 RTX PRO 6000 上原生跑 125B-A5B MoE（`Qwen3.8-Flash-Next`，NVFP4 trunk + 3-bit 专家），
不用 WSL、不用 Docker，聚合吞吐 2035 tok/s —— 以及把它调到这一步的全部过程与实测数据。**

![platform](https://img.shields.io/badge/platform-Windows%2010%2F11%20x64-0078D4)
![gpu](https://img.shields.io/badge/GPU-RTX%20PRO%206000%20Blackwell%2096GB-76B900)
![engine](https://img.shields.io/badge/engine-Hibana%200.1.0-black)
![license](https://img.shields.io/badge/license-MIT-blue)

> **这是一个"移植工具包 + 工程复盘"，不是引擎的分支。** 仓库里没有引擎源码，也没有模型权重
> （见 [NOTICE](NOTICE)）。里面有：幂等的 Windows 补丁器、一键工具链安装、生产启动脚本、
> 完整测量工具、MTP 草稿头蒸馏流程，以及 45 轮实测优化历史 —— **包括失败的那些**。

---

## 成绩

单机：**RTX PRO 6000 Blackwell 96 GB · Windows 10 19045 · 驱动 616.92 · 191 GB RAM**。
贪心解码，五点并发，`--max-batch 16`，深度表 `1:8,2:6,4:3,8:3,16:4`。

| 并发 | 贪心 tok/s | 采样(0.3) | 带宽估算上限 | 达成 |
|---|---|---|---|---|
| c1  | **273.5** | 186.0 | 455  | 60% |
| c2  | **527.7** | 324.0 | 700  | 75% |
| c4  | **724–847** | 524.8 | 930  | 78–91% |
| c8  | **1321.8** | 831.2 | 1570 | 84% |
| c16 | **2035.5** | 1298.8 | 1690 | **120%（估算偏低）** |

### 独立复测（SparkDash，2026-11-20 · temp 0.7 · 400 tok · thinking off）

| 负载 | ×1 | ×4 | ×16 |
|---|---|---|---|
| 散文 | **327.3** | 724.2 | 1883.4 |
| 代码 | **526.0** | 905.5 | 1998.1 |
| 结构化 | **314.8** | 961.2 | 2289.1 |

prefill 阶梯（每档唯一前缀）：**4 K 时 21 376 tok/s（TTFT 377 ms）→ 128 K 时 9 018 tok/s（TTFT 14.72 s）**。
两套 harness 在单流散文上相差 **0.5%**（327.3 vs 325.6），互相验证；但**上面的表和下面的表不能换算**
（构建、长度、采样都不同）。细节与唯一那条反常：[docs/07-sparkdash.md](docs/07-sparkdash.md)。

单流散文（日常体感口径）：**英 325.6 / 中 332.3 tok/s**。
相对 DGX Spark(GB10) 生产配方：单流 **1.8–2.4×**，16 并发 **~3.1×**。
冷启动 **~4 分钟**（47.7 GB PLE 表 + 权重 + 图捕获），热重启 ~35 s。

> 每个数字都带日期、prompt 集与测量条件。引用前请读 [docs/04-benchmarks.md](docs/04-benchmarks.md)；
> 看到任何"草稿头提速"的说法，先读 [docs/06-postmortem.md](docs/06-postmortem.md)。

---

## 为什么这件事难

引擎是为 GB10 Spark 写的：aarch64 Linux、**统一内存**、`273 GB/s`。
那边每个 token 要读 6.5 GB 权重 ⇒ ~24 ms，**把所有其它开销都埋掉了**。
RTX PRO 6000 有 `1792 GB/s` 且是**独立显存 + WDDM**，同样 6.5 GB 只要 3.6 ms，于是：

1. **WDDM 下 host↔device 握手 1.85 ms**（Linux 10–30 µs），而投机解码每个草稿列一次
   ⇒ 对策不是"更快地握手"，而是**把每轮的握手次数砍掉**（深度表）。
2. **PLE 表 47.7 GB 在主机侧**，每 token 要 16 行随机读，且要等草稿 token 才能算
   ⇒ `mmap` + 批量 fancy-indexing 取代逐列线程池（2048 行 **36 ms → 0.67 ms**）。
3. **每步 500–600 个小内核**原本被 24 ms 带宽掩盖，现在成主成本。
4. **Linux-only API** + **flashinfer 的 JIT 假设 Linux**（ninja / `-fPIC` / `.so`）。

解法分别在 [docs/01-porting.md](docs/01-porting.md) 和 [docs/02-tuning.md](docs/02-tuning.md)。

---

## 硬件/软件契约（"同一张显卡"的含义）

| | 要求 |
|---|---|
| GPU | **RTX PRO 6000 Blackwell 96 GB**（SM 120 / `compute_120a`），3-bit 档需 ≥90 GB 显存 |
| OS | Windows 10 21H2+ / 11 x64，**不用 WSL、不用 Docker** |
| 内存 | ≥128 GB（实测 191 GB；47.7 GB PLE 表常驻） |
| 磁盘 | ≥250 GB 本地 **NVMe**（135 GB trunk + 48 GB W3 + venv）。USB 盘实测 4–20 MB/s，不可用 |
| CUDA | 13.0（会建 `C:\cuda13` 目录联接，避开路径空格） |
| MSVC | **VS 2019/2022 BuildTools**（CUDA 13 拒绝 2026 工具集：`cudafe++` → `0xC0000005`） |
| Python | 3.12.11（`uv` 托管） |

其它卡（5090 32 GB / L40S / A100 / GB10 本身）**装不下这套 3-bit profile**；
小卡的取舍见 [docs/05-troubleshooting.md](docs/05-troubleshooting.md#smaller-cards)。

---

## 快速开始

```powershell
# 前置：你必须已经合法拥有 a) 一份 Hibana 0.1.0 代码   b) 模型权重的下载权限
git clone https://github.com/<you>/hibana-flashnext-windows.git
cd hibana-flashnext-windows

copy setup\config.example.ps1 setup\config.ps1        # 只改一次路径
powershell -ExecutionPolicy Bypass -File setup\preflight.ps1   # 12 项检查，告诉你还缺什么
powershell -ExecutionPolicy Bypass -File setup\bootstrap.ps1   # 工具链 + 补丁 + 权重 + 冒烟
launch\run-flashnext.bat                                       # -> http://127.0.0.1:30004
```

`bootstrap.ps1` 是幂等的：完成的阶段会跳过，补丁只在锚点仍未命中时才打，绝不碰 `config.ps1` 之外的路径。

验收：

```powershell
python port\verify.py                      # /health、/v1/models、/get_server_info、中文、代码
python bench\refcheck.py cmp base          # 位一致闸门，必须打印 IDENTICAL
python bench\sweep.py 1,2,4,8,16 700       # 上面那张五点表
python bench\eval_wide120.py myhead        # 草稿头裁决（唯一可信口径）
```

---

## 仓库地图

```
setup/     preflight.ps1（12 检查）· bootstrap.ps1（五阶段）· config.example.ps1
port/      winpatch.py（26 处幂等引擎补丁）· _plat.py（POSIX 垫片）· win_compat.h
           setup_windows.py（工具链）· verify.py
launch/    run-flashnext.bat（生产）· restart.ps1（轮询 /health，绝不 sleep）· status.ps1
           preload_ple.ps1 · run-flashnext-*.bat（A/B 变体）
perf/      六个热文件的参考 diff + hc-bf16 / wide-9rows 补丁
bench/     refcheck · sweep · accept_sampled · eval_wide120（120 条宽集）· soak · c1_prose ...
train/     语料生成 · 特征抽取 · KL/CE 蒸馏 · accept_len 评测（含负结果）
docs/      01 移植 · 02 提速 · 03 MTP 蒸馏 · 04 实测数据 · 05 排障 · 06 复盘
reference/ HANDOFF.md（3000 行逐轮原始记录）· LESSONS.md · kernel-fusion-spec.md
```

---

## 蒸馏的结论（一句话版，因为这是最有用的负结果）

我们完整跑了 SAA-Brain 的 4 步自喂 KL 蒸馏：自造语料 35 万 → 122 万 → 205 万 token、
8 步自喂 rollout、CE + KL_top32、512 专家上 rank-16/32 LoRA，产出 23 个草稿头（v2…v23）
（随仓库的训练脚本里这个常量是 4，复现 v 系列要改成 8）。
trainer 的 `accept_len` 在**跨语料留出集**上从 2.668 涨到 3.017（**+13.1%**），
引擎侧看起来是 **+11% 单流** —— 然后我们建了 120 条句式与训练语料完全不同的新 prompt，
五个头全部落在 **2.3%** 以内（0.5912 vs 0.5912 accept，3.986 vs 4.030 tok/round）。

**结论：MTP 草稿头的窄域蒸馏，泛化增益 ≈ 0。** 原头已接近该架构的泛化上限；
真正提速的是把 hyper-connection 从 fp8 换成 **bf16**（+4.8%，而且**多读字节反而更快**）。
完整流程、数字与抓出污染的方法在 [docs/03](docs/03-mtp-distillation.md) 与 [docs/06](docs/06-postmortem.md)。

---

## 提速构成（移植到第三个平台时该抄哪些）

| # | 改动 | 机制 | 实测 | 保留 |
|---|---|---|---|---|
| 1 | PLE `mmap` fancy-indexing | 2048 行 36 ms → 0.67 ms | c16 ×2.1 | ✅ |
| 2 | 按并发档位调投机深度 | 握手次数 ∝ 深度 | c16 492→903 | ✅ |
| 3 | PLE 线程池修复 | 不再每轮新建池 + 句柄复用 | 单流 1.5–1.7× | ✅ |
| 4 | FA2 plan 缓存 | plan 0.29→0.07 ms/轮 | c8/c16 +4% | ✅ |
| 5 | split-K 除数放开 | K=2560：80→200 block | zh +14%、c4 +13% | ✅ |
| 6 | SM 12.1 硬门放宽到 12.0 | 66 内核图重放 | code +8.6% | ✅ |
| 7 | bf16 hyper-connection | fp8 `_scaled_mm` 瓦片效率低 | **+4.8%** 单流 | ✅ |
| 8 | PLE 常驻（页缓存/内存） | 冷页 → 常驻 | c8 +33% | ✅ |
| 9 | 一次性批量 PLE | 少一次 gather | c4 −9%、c16 −12% | ❌ |
| 10 | `verify_state=snapshots` | 去掉 reconstruct | 96 GB 装不下 | ❌ |
| 11 | NVFP4 KV | 少字节 | FP4-KV 模块编不过 | ❌ |
| 12 | 窄域 MTP 蒸馏 | 提高接受率 | 宽集 ≈0% | ❌ |

所有 `→` 数字都来自同一台机器、A/B 交错、2–3 次取中位，且前置 `refcheck` 位一致闸门。

---

## 归属

蒸馏配方与"无损提速"方法论来自 `CharAzinabulu0079/SAA-Brain-3.8-flash-next-Hibana`；
3-bit W3 内核布局、PLE 设计与 MTP proposer 来自 Hibana 引擎。
本仓库只记录它们在**独立显存 + Windows 主机**上的行为，以及让它跑起来的胶水。

引用本文数字时请**同时**带上条件（GPU、驱动、prompt 集、贪心还是 temp 0.3）——
它们不能跨 prompt 集比较，这正是 [docs/06](docs/06-postmortem.md) 的全部教训。

原创部分 MIT，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
