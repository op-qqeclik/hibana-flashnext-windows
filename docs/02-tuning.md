# 02 · 速度优化全流程（45 轮实测复盘，按收益排序）

> **English abstract** — The optimisation playbook for a *discrete-GPU Windows* host where the engine's
> original cost model (built for 273 GB/s unified memory) no longer hides anything. Ordered by measured
> ROI, with the mechanism, the exact number, and — for the 15 things that did not work — why not.
> All numbers: greedy decoding, bit-exactness gate before acceptance, A/B interleaved, median of 2–3 runs.

阅读顺序建议：先读 §1（一句话诊断），再看 §2（清单），最后读 §5（已被否决的路，别重走）。
测量方法本身是这套流程的一半，见 §3；如果你只想抄一件事，抄 §3。

---

## 1. 一句话诊断：为什么同一份代码在这张卡上是另一个瓶颈

源机 GB10：每解码 token 读 **5.38 GB** 权重，273 GB/s ⇒ **~24 ms**，这 24 ms 把所有其它开销都**埋掉了**。
目标机 107：同样 5.38 GB，1.63 TB/s ⇒ **~3.6 ms**，于是被埋的东西全部浮上水面：

| 浮出来的开销 | Linux/GB10 上 | Windows/107 上 | 后果 |
|---|---|---|---|
| host↔device 事件同步 | 10–30 µs | **1.85 ms**（WDDM） | 每个 draft 列一次 ⇒ 每轮 8–9 次 |
| `cudaGraphLaunch`（~500 节点 verify 图） | — | **0.4–0.85 ms/轮**（Linux 的 10–20 倍，且随节点数线性涨） | — |
| 每步 ~500–600 个小 kernel | 被 24 ms 掩盖 | 232 个 trunk GEMM + 205 sm89 + 48 MoE + 1116 norm + 3129 quant 的串行派发 | 主机侧一度成为主成本 |
| PLE 表随机读（47.7 GB，主机侧） | 统一内存里是 page cache 命中 | NTFS + 独立显存，走 `preadv2` 的 Linux 池没了 | 单流 1.5× 损失 |
| 引擎按平台名静默降级 | `SM == (12,1)` 才开图融合 | 107 是 `(12,0)`，**默认关掉了 66 kernel 融合** | code 162→176 白丢 |

**所以优化的顺序不是"调参 → 调内核 → 调算法"，而是：先修静默降级 → 再削握手次数 → 再修主机侧 →
最后才是内核。** 前三步都在本文 §2 的前 6 项里，占了全部收益的 90%。

---

## 2. 优化清单（按实测收益排序）

所有条目都在同一台机器上、用 §3 的纪律测过。`→` 左边是改动前、右边是改动后，单位 tok/s。
"保留"一列是最终是否固化在 `launch/run-flashnext.bat`。

### 2.1 ✅ PLE 取行：`memmap` fancy-indexing 取代主机线程池（c16 ×2.1）

**症状**：16 并发只有 791 tok/s，而带宽账算得出 1690。
**机制**：PLE 表 320M 行 / 47.7 GB 在主机，每个 token 要 16 行（8 个 n-gram head × 2）。
原路径在 Windows 上走 `_gather_threads`：**每次调用新建一个 32 线程 `ThreadPoolExecutor`**，
且句柄缓存是函数内的 thread-local —— 一个投机轮里 gather 被调 9 次，每轮白掉一个线程池。
**改法**（`model/flashnext/ple.py`）：
1. 线程池挂到表对象上，只建一次；
2. 句柄缓存改成**表级** thread-local，跨轮复用分片句柄；
3. 行数 ≤ 8 直接当前线程读（页已在缓存里，派发比读还贵）；
4. `pool.map` 加 `chunksize`，2048 行的 verify gather 不再付 2048 次 future 跳转；
5. 更彻底的一步：verify 路径改 `np.memmap` + **fancy indexing 批量取行**（`_gather_mmap`）。

**实测**：2048 行 gather **36 ms → 0.67 ms**；c16 **791 → 1656**（×2.1）；单流 143.9 → 215.9（en）。
**为什么它能赢**：随机读的页错误是**串行**的，而线程池把"一次 fault"变成"一次调度"。
批量 fancy-indexing 让内核在同一批页上连续 fault，且 VMM 会做合并预读。
**保留** ✅（默认路径；`HIBANA_PLE_BATCH_ROWS=0` 可关）

### 2.2 ✅ 按并发档位给投机深度（c16 492 → 903）

**机制**：每轮的 draft 步数 = 每轮的 host 握手次数。WDDM 下每次握手 1.85 ms，而每多一个深度只多
**0.35 个 token**（第 2–8 个草稿的接受率只有 ~0.30）。所以高并发段深度越深越亏。

| 深度表 | c1 | c2 | c4 | c8 | c16 |
|---|---|---|---|---|---|
| 引擎默认 `16:6` | — | 321 | 341 | 378 | 623（采样 492） |
| `1:8,2:7,4:6,8:5,16:4` | 221 | 321 | 426 | 595 | **903**（采样 655–707） |
| `1:8,2:5,4:3,8:2,16:4` | 201 | 321 | **507** | **712** | 839 |
| **`1:8,2:6,4:3,8:3,16:4`（固化）** | **273** | **528** | 725–847 | 1031–1322 | 1608–2036 |

**为什么 c8 后来又用 3 而不是 2**：采样档（temp 0.3）下接受率更低，多一个草稿更值（+3.8%，交错实测）。
贪心档 3 与 2 打平（1060.8 vs 1055.8），共用一张表。
**保留** ✅ `HIBANA_SPEC_DEPTHS=1:8,2:6,4:3,8:3,16:4`

### 2.3 ✅ 修平台静默降级：SM 12.1 硬门 → 同时接受 12.0

一批图融合（`flashnext_fused` / `mlp_fusion` / `mamba2` / `qsa`）写的是 `if cap == (12, 1)`
—— 那是 GB10 的 SM 号。107 是 `(12, 0)`，于是"66 个 kernel 融成一个 graph replay"整个被跳过。
**实测**：code 162 → 176、zh_prose 86 → 93、接受率 0.517 → 0.563。
**教训**：移植后第一件事是 `grep -rn "12, *1\|sm_121\|GB10"` 把平台判定全找出来。
**保留** ✅

### 2.4 ✅ FA2 plan 的 schedule 缓存（draft 0.21→0.12 ms/步，verify 0.29→0.07 ms/轮）

**机制**：flashinfer 的 FA2 每次调用都要重算 split-K 的 plan。而 plan 是
**(split 大小, 每请求 chunk 数) 的纯函数** —— 引擎侧先证明了这一点（31 个长度到 32K 全部一致），
于是按 (split, chunks) 做键缓存，并校验命中时布局一致。
**实测**：draft 图 plan 0.21 → 0.12 ms/步、verify plan 0.29 → 0.07 ms/轮；c8/c16 **+4%**，逐位一致（md5 同前）。
同时把页表的每轮 `torch.cat` 改成**原地增长**。
**保留** ✅ `HIBANA_PLAN_CACHE=1`（`=0` 可回退）

### 2.5 ✅ split-K 放开除数限制（zh +14%，c4 +13%）

**机制**：trunk 的 231 个 GEMV 每个只有 24.8 MB，被 20 个列块 × **4** 个 K-split（80 block）摊开，
188 SM 填不满 ⇒ 每次 13.6 µs 几乎全是延迟。原逻辑"split 必须是 2 的幂且满足
`chunks % (2·splits) == 0`"，K=2560 是 20 个 chunk，走到 4 就停了 —— 其实 **5 和 10 也整除**。
**改法**（`kernels/w3.py::splits_for`）：候选从"2 的幂"改成"所有满足整除的除数
（1,2,3,4,5,6,8,10,12,16）"，选第一个达到 `SPLIT_TARGET` 的；K=2560 ⇒ splits=10（200 block）。
**实测**：en +4.7%、zh **+14.0%**、code +1.9%、c4 378 → 426、c8 440 → 595。
**规律**：轮越短（固定开销占比越高）受益越大；高并发基本不变。
**保留** ✅
> ⚠️ 但"按 SM 数放大 `SPLIT_TARGET`"是**中性**的（±2%），因为 split 数被整除条件锁死；
> 而自建微基准说"加 split-K 有 1.9–2.5 倍"是**假的**（见 §5.6）。

### 2.6 ✅ PLE 表启动预热（c8 +33%）

47.7 GB 顺序全读进 Windows 页缓存，约 15 s（3.3 GB/s）。**不预热**：c8 129.5、c16 198.7。
**预热后**：c8 171.8（+33%）、c16 218.8。已写进 `run-flashnext.bat`（日志 `results\preload.log`）。
**保留** ✅

### 2.7 ✅ hyper-connection 用 bf16 而不是 fp8（单流 +4.8%，且**多读字节反而更快**）

**改法**：`model/flashnext/model.py` 三处 `HyperConnection.load(...)` 显式 `precision="bf16"`。

| 配置 | accept | tok/round | round_ms | tok/s |
|---|---|---|---|---|
| fp8（原） | 0.5969 | 4.107 | 13.13–13.27 | 310.6 |
| **bf16（固化）** | 0.5945 | 4.067 | **12.49–12.52** | **325.6** |

每轮解码字节 **5.38 → 6.01 GB，反而更快**。原因：fp8 那条路走 `torch._scaled_mm` →
Ada 的 `sm89_xmma...32x64x64` 瓦片，瓶颈是**内核效率**不是带宽。这也解释了为什么"窄 GEMV 接管 sm89"更慢、
split-K 微基准更慢：**在这张卡上，少读字节 ≠ 更快**。
质量闸门：bf16 精度高于 fp8，接受率变化 −0.2%（噪声内），不是"放宽验收"。
**保留** ✅（补丁 `perf/hc-bf16.patch`）

### 2.8 ✅ 显存模型修正 + 准入函数去抖（不然服务根本不可用）

两个坑都会表现成"引擎起来了但不回话"：

1. 用驱动 `mem_get_info().free` 做准入 → WDDM 会超额提交，`free` 恒 0，20 GiB 里一直分到 200 GiB 才 OOM。
   ⇒ 改用 `torch.cuda.memory_reserved()` 做池预算。
2. 但 `memory_reserved()` 是 O(segments) 遍历，本机几秒 —— 而 `can_admit()` **每步都调**。
   ⇒ 驱动 `mem_get_info()` + **250 ms 缓存**。

**保留** ✅（`port/winpatch.py` 内）

### 2.9 ✅ 顺带修掉的其它真实开销

| 项 | 数值 | 说明 |
|---|---|---|
| PLE 预取路径的 `fadvise(WILLNEED)` | 走垫片后 0 成本 | Windows 上 `PrefetchVirtualMemory`，不做就退化成 no-op |
| 视觉塔输出缓存 + 图片解码缓存 | 96 MB / 320 MB | 按图片指纹缓存，带图首轮 6.1 s、之后 0.08–0.16 s/轮 |
| trunk `SPLIT_TARGET` 静态值 | — | 按 188 SM 标定；192/96/48 三档都测过，中性 |

---

## 3. 测量纪律（这套流程的一半，别跳过）

**没有闸门的"变快了"一律不信。** 我们三次被测量方法坑过（§5.9），代价是白干一天半。

### 3.1 每次改动必须过三道闸

```powershell
python bench\refcheck.py cmp base    # ① 位一致：输出 md5 必须 IDENTICAL（贪心、固定种子）
python bench\sweep.py 1,2,4,8,16 700       # ② 五点：c1/c2/c4/c8/c16 逐点，best-of-3
python bench\soak.py     # ③ 浸泡：4 分钟持续负载不崩、不漂移
```

`refcheck` 不同 ⇒ 直接回退，不进第二轮。**质量硬约束：不允许放宽 top-k / margin / 拒绝采样**
（`--throughput` profile 会把接受率从 0.597 干到 0.4445 —— 那是拿质量换速度，禁止）。

### 3.2 A/B 必须交错

两臂**背靠背交错**各 2–3 次取中位，不能"上午测 A、下午测 B"。理由：这张卡有 co-tenant
（ComfyUI / Strata 会占几十 GB 显存），且 GPU 频率随桌面负载漂移。
一次典型的交错跑法：

```
A B A B A B      → 取各自中位
```

### 3.3 三个已知的测量陷阱

1. **冷启动 / 热启动不可比**：图捕获按桶**惰性**发生，五点必须先暖机（各点先跑 2 轮丢弃）。
2. **GPU 必须安静**：桌面合成、浏览器、ComfyUI 都会让 c1 抖 10%。`nvidia-smi` 确认 util 0% 再开测。
3. **内核级微基准会骗人**：单次 `w3_linear` 的 Python 发射开销 20–40 µs，而 GPU 实际只有 12 µs
   ⇒ 你测的主要是主机侧。**必须用引擎内 CUPTI 窗口（`HIBANA_PROFILE_STEPS`）复核**。

### 3.4 引擎内探针（这是唯一可信的分阶段计时）

| 环境变量 | 作用 |
|---|---|
| `HIBANA_STEP_TRACE=1` | 每 200 步打印 `wall / gpu / host`；`host ≈ −0.1 ms` 表示主机侧已完全被遮住 |
| `HIBANA_PROFILE_STEPS=N` | 打开 torch.profiler 窗口（N 步） |
| `HIBANA_PROFILE_SENTINEL=<file>` | 哨兵文件存在时才开窗（否则永远抓到启动那几步） |
| `HIBANA_PHASE_STATS=1` | 引擎内分阶段计时 |

---

## 4. 天花板推导：为什么 c1 停在 250–270

这是全篇最值钱的算术，因为它告诉你"再优化也没用"。

1. **每轮有一次约 12.6 ms 的固定成本**，与批大小/深度基本无关：
   d=1 → 12.6 ms；d=4 → 14.3；d=8 → 15.6。每多"一个草稿步 + 一行验证"只加 **0.4–0.5 ms**。
   ⇒ c1 的物理上限 ≈ `tok/round ÷ 12.6 ms`；tok/round 3.94 ⇒ **313 tok/s**。
2. **主机侧已经不在关键路径上**：c1/c8/c16 的 `host` 都是 **−0.1 ms**（wall ≈ gpu）。
   所有主机侧优化到此为止，再改一行都没有收益。
3. **目标值隐含的接受率是现装草稿头的两倍**：链式接受率是位置相关的 —— 第 1 个草稿 ~0.85，
   **第 2–8 个各约 0.30**。要 455 tok/s 需要第 2–8 步 ≈ **0.76**；c16 的目标按接受率算**已经达成**。
   ⇒ 这是**分子问题**（每轮吐多少 token），不是分母问题（每轮多久）。引擎侧只能压分母 10–20%。
   > 该"需要多少接受率"的估算后来被上修：**0.76 → 0.87（平线）**（见 `docs/04` §9.5）。
   > 结论不变，只是更绝望：要 455 几乎需要草稿链的完美接受。

**结论**：c1 的 60% 达成率不是工程没做完，是架构给的。要动它必须改草稿头或改权重布局（§6）。

---

## 5. 已否决的路（15 条，附原因 —— 这一节比 §2 更省时间）

| # | 试过的东西 | 结果 | 为什么不行 |
|---|---|---|---|
| 1 | **一次性批量 PLE 取行**（`HIBANA_PLE_BATCHED=1`） | c4 −9%、c16 −12% | 逐列流式把"取行"和"draft 链"**重叠**了；一次性批量把重叠变成串行等待 |
| 2 | `HIBANA_VERIFY_STATE=snapshots` | 请求全堵在准入（`rounds=0`） | 能省 `_reconstruct` 2.8 ms/步，但要多占 ~15 GiB，96 GiB 上装不下 |
| 3 | 轮询式事件等待（自旋 `query()`） | 214.0 vs 214.3（中性） | 那 1.85 ms 是**真在等 GPU**，不是 WDDM 系统调用开销 |
| 4 | NVFP4 KV 缓存 | 编不过 | flashinfer 的 FP4-KV 模块在本移植里编不过；fp8 已够用 |
| 5 | `--throughput` profile | accept 0.4445 | 它会**放宽验收**（有损），违反"无损提速"约束 |
| 6 | 按 SM 数放大 `SPLIT_TARGET`（192/96/48 → 282/141/70） | ±2% | `chunks % (2·splits) == 0` 把 splits 锁在 4，放大目标值无效 |
| 7 | 自建微基准驱动的 split-K 补丁 | 引擎内 12.6 → **26.5 µs**（慢 2 倍） | 微基准测的是 Python 发射开销（20–40 µs）而非 GPU（12 µs） |
| 8 | 窄 GEMV 接管 sm89 / hyper wide 路径 / `DRAFT_VOCAB`（两种排序）/ `DRAFT_HEAD` w2/w3 | 全部为负 | `sm89_xmma` 瓦片的瓶颈是内核效率；换算法不解决 |
| 9 | `quant_fp8_small` 并行化（单 CTA → 多 CTA） | 净亏 | 见 §5.1：全局 max 要先归约，拆开就得多一次物化 |
| 10 | `silu` 融合、page-table 原地追加（部分）、rescore 宽度、`BOLT_EXPERTS` 变体 | 中性或负 | — |
| 11 | `HIBANA_SSM_STATE=int8` | accept 0.5896（−1.2%） | 精度换不来速度 |
| 12 | 深度 `1:6` | accept 0.6181 但 tok/round 3.943；折合 303 < 313 | 接受率升、但每轮 token 少 |
| 13 | DFlash2 块并行草稿头（现成权重，1.15 GB） | 接受率崩（zh **0.066**） | 权重是在官方 FP8 目标上训的，我们跑 3-bit 量化目标 ⇒ 特征偏移；必须在本机隐藏态上**重训** |
| 14 | 窄域 MTP 蒸馏（v5…v23） | 引擎侧 +11%，宽集上 **≈0%** | **评测污染**，见 `docs/06`；原头已接近架构泛化上限 |
| 15 | `--exact` 档（NVFP4 专家 63.3 GiB） | `only 7.3 GiB available` | 96 GiB 卡上装不下 KV 池 |

### 5.1 为什么 `quant_fp8_small` 不能并行化（值得单独说，因为太容易被"优化"）

`quant_fp8_small()` 的 scale 是**整个张量的全局 max**：

```python
peak  = max(|x| over all elements)
scale = max(peak / 448, 1e-12)
q     = clamp(div_rn(x, scale), -448, 448)
```

全局 max 意味着"先算完所有元素 → 一次跨 program 归约 → 才写 fp8"。单 kernel 做不到
（除非 cooperative groups / persistent kernel 做 grid 内同步）。所以现状的单 CTA
**已经是"没有 grid sync 前提下最省的形式"** —— 并行化它只会多一次物化。

同理，`inject_norm + quant_small` **不能融合**：`inject_combine_norm()` 返回的 `parts` 是每 block 的
`|y|` 最大值（给 `quant_fp8_parts` 用），与全局 max 是两套语义。

---

## 6. 还剩什么（作者级，需要改架构）

| 方向 | 预期 | 工作量 | 依据 |
|---|---|---|---|
| **trunk 231 次 3-bit GEMM 合并到 ~96 次** | c1 +3–6%（−0.54 ms/轮） | 大 | 同层共享输入的 q/k/v、gate/up 沿输出维拼接，数学等价；要离线重排量化权重布局（3-bit 打包 + per-row/32-group scale 对齐，是主要风险） |
| **块并行草稿头 DFlash2 重训** | 唯一能绕开"单 token 准确率 0.73 平线"的路 | 大 | 现成权重与本引擎特征 tap 不对齐（zh 0.066） |
| **`_layer_norm` + `_inject_norm` 合并** | +3–4% | 中 | 两者都是 norm 类，`_grouped_norm_kernel` 已存在，可复用 |
| **减少 verify 图节点数**（~500 节点 → 更少） | c1 +1.5–3% | 中 | WDDM 下 graph launch 随节点数线性涨（414 µs/次） |
| **PLE 高频行常驻显存** | c4/c8 +2–3% | 大 | 要从专家权重里挪 8–16 GB；作者结论是"别再开战役" |

---

## 7. 环境变量总表（107 上的实际用法）

| 变量 | 107 取值 | 作用 | 回退 |
|---|---|---|---|
| `HIBANA_CUDA_ARCH` | `12.0a` | nvcc 的 family arch（FP4 指令必须） | — |
| `HIBANA_MSVC_PATH` | `…\VS2022\BuildTools` | 钉住 CUDA 13 能用的 MSVC | — |
| `HIBANA_SPEC_DEPTHS` | `1:8,2:6,4:3,8:3,16:4` | 每并发档的 draft 深度（=每轮握手次数） | 引擎默认 |
| `HIBANA_SPEC` | `8` | 投机链最大深度 | — |
| `HIBANA_KV_FORMAT` | `fp8` | KV 精度（nvfp4 编不过） | — |
| `HIBANA_KV_POOL_TOKENS` | `524288` | KV 池（1M 触发 headroom 保护） | — |
| `HIBANA_STATE_SLOTS` | `36` | 状态槽（48 与 `max_batch 32` 装不下） | — |
| `HIBANA_VRAM_RESERVE_GIB` | `2` | 给桌面留的显存 | `4` 更稳 |
| `HIBANA_HOST_FLOOR_GIB` | `2` | 主机内存下限 | — |
| `HIBANA_GDN_PREFILL` | `triton` | GDN prefill 后端 | — |
| `HIBANA_SSD` | `0` | SSD 前缀层（本移植未启用收益） | — |
| `HIBANA_VISION` | `1` | 视觉塔 | `0` 省 0.84 GiB |
| `HIBANA_MTP_WEIGHTS` | `…\mtp\mtp_weights.v16reason.safetensors` | 挂载蒸馏草稿头（加载时转 FP8，零额外显存） | 删掉即用原头 |
| `HIBANA_PLAN_CACHE` | 默认开 | FA2 plan 缓存 | `=0` |
| `HIBANA_PLE_BATCH_ROWS` | 默认开 | 批量 PLE 取行（b≥4 时 +6%） | `=0` |
| `HIBANA_PLE_BATCHED` | **不设** | 一次性批量取行（已证更差） | 保持不设 |
| `HIBANA_VERIFY_STATE` | 默认 `reconstruct` | `snapshots` 在 96 GiB 上装不下 | — |
| `HIBANA_SSM_STATE` | 默认 fp32 | `int8` −1.2% | — |
| `HIBANA_PHASE_STATS` / `HIBANA_STEP_TRACE` / `HIBANA_PROFILE_STEPS` | 调试时开 | 引擎内计时/剖析 | 生产关闭 |

---

## 8. 收益总账（从"移植完成"到"最终"）

| 题面 | 移植完成 | 最终 | 倍率 |
|---|---|---|---|
| en_prose 单流 | 107.5 | ~326 | **3.0×** |
| zh_prose 单流 | 86.4 | ~332 | **3.8×** |
| code 单流 | 162.7 | ~435 | **2.7×** |
| c4 聚合 | — | 725–847 | — |
| c16 聚合 | 198.7 | **1907–2036** | **9.6–10.2×** |

相对 DGX Spark 的生产配方：单流 **1.8–2.4×**、c16 **~3.1×**。
相对本文档 §4 推出的带宽天花板：c1 60%、c2 75%、c4 78–91%、c8 84%、**c16 120%（天花板估算偏低）**。
