# 04 · 实测数据（全部来自同一台机器，带日期和口径）

> **English abstract** — Every measurement in this repository, with its date, prompt set, decoding mode
> (greedy vs temp 0.3), warm-up state and measurement instrument. Numbers measured before 2026-11-19 are
> flagged where they were later retracted. **Do not compare across prompt sets** — that is the single
> biggest methodological failure of this project (see `docs/06-postmortem.md`).

> ⚠️ 读之前请先接受三条前提：
> 1. **这套数字只在"同一台机器 + 同一套 prompt"内可比**。跨 prompt 集比较是本项目踩过最大的坑。
> 2. `贪心` 与 `temp 0.3` 是**两种不同的解码**，差距 15–27%，不是稳定性问题。
> 3. 凡标注 **【已作废】** 的数字都保留在文档里 —— 它们是方法论的证据，不是可引用的成绩。

---

## 0. 测试环境（复现的前提）

| 项 | 值 |
|---|---|
| 机器 | 107：**RTX PRO 6000 Blackwell Workstation Edition 96 GB**，SM 12.0 / 188 SM，驱动 **616.92** |
| OS | Windows 10.0.19045（x64），**原生**，无 WSL / 无 Docker |
| CPU/RAM | 191 GB DDR5（PLE 表 47.7 GB 靠页缓存） |
| 实测带宽 | bulk copy **1282 GB/s**、read **1173 GB/s**、MoE 内核 **1536–1608 GB/s**（标称 1792 的 86–90%） |
| 引擎 | hibana 0.1.0，`HIBANA_CUDA_ARCH=12.0a`，MSVC 2022 BuildTools，CUDA 13.0 |
| 服务 | `hibana serve flashnext --max-batch 16 --context-length 262144` → `:30004` |
| 模型 | Qwen3.8-Flash-Next：48 层 / hidden 2560 / 512 专家 / top-10 / head_dim 256 / 24 attn heads / 2 KV heads；`layer_types = [linear,linear,linear,full_attention] × 12`；`mtp_num_hidden_layers=1`；PLE 320M 行 FP8 表（47.7 GB，主机侧） |
| 权重量化 | trunk FP8 + routed experts **3-bit（46.1 GiB）** + lm_head FP8；每解码 token 读 **5.38 GB**（bf16 hyper-connection 后 6.01 GB） |
| 投机解码 | MTP 头（1 层），深度表 `1:8,2:6,4:3,8:3,16:4`，**精确拒绝采样**（不放宽 top-k / margin） |
| 质量闸门 | `refcheck.py cmp base` 必须 `IDENTICAL` |

**纪律**：五点全部先暖机（每点预热 2 轮丢弃）；A/B 交错、各 2–3 次取中位；测量窗口内 `nvidia-smi`
util = 0%、无其它常驻显存占用者（ComfyUI 需先 `POST /free`，Strata 需退出）。

---

## 1. 五点并发曲线（主成绩表）

**贪心解码**（`sweep.py 1,2,4,8,16 700`，英文技术散文题面，700 tok）

| 点 | 移植完成 11-16 | 最终 11-17 15:1x | 宽集修正 11-19 10:15 | **bf16 hc 之后 11-19** | 带宽估算天花板 | 达成率（最终） |
|---|---|---|---|---|---|---|
| c1 | 107.5 | 270.8 | 256.9 | **273.5** | 455 | **60%** |
| c2 | — | 405.9 | 492.6 | **527.7** | 700 | **75%** |
| c4 | — | 639.0 | 690.8 | **724–847** | 930 | **78–91%** |
| c8 | — | 1031.0 | 1135.5 | **1321.8** | 1570 | **84%** |
| c16 | 198.7 | 1608.4 | 1907.2 | **2035.5** | 1690 | **120%（超额）** |

> **另有独立工具的复测**：SparkDash（temp 0.7 / 400 tok / thinking off）给出 c1 散文 **327.3**，
> 与本表 bf16 后的 325.6 相差 **0.5%** —— 两套 harness 互相验证。但**两张表不可换算**（构建、长度、采样都不同），
> 只能各自纵向比。完整数据、128 K prefill 曲线与一条被作废的旧 bug：`docs/07-sparkdash.md`。

**采样解码**（temperature 0.3，`accept_sampled.py`）

| 点 | 11-17 | 说明 |
|---|---|---|
| c1 | 186.0 | 比贪心 −31% |
| c2 | 324.0 | −20% |
| c4 | 524.8 | −18% |
| c8 | 831.2 | −19% |
| c16 | 1298.8 | −19% |

> **采样为什么慢**：精确拒绝采样下，温度 0.3 的目标分布更尖，而 32B 级别的草稿头只能在 16 个候选上
> 近似 27B 级目标 ⇒ accept 0.79 → 0.62。这是**数学结果，不是 bug**，也不是可调参数。
> 采样档唯一无损的提升路径是更准的草稿头（权重/训练侧），见 `docs/03`。

> **关于 c16 的 120%**：天花板 1690 是按"每轮读 49 GB 专家权重 ÷ 1.6 TB/s"算的。
> 实测越过去了，说明该模型里的**跨轮权重复用/缓存命中**比估算假设的更好。**天花板是估算，实测才是事实**，
> 这条留在这里是为了提醒后来者不要把估算当上限。

---

## 2. 单流散文（日常体感口径）

2026-11-18 用户口径纠正：日常体感最强的是**英文散文与中文散文**，代码速度次要。
单流验收改用 `bench/c1_prose.py`（英/中各 3 篇、900 token、贪心、best-of-2）。

| 指标 | 值 | 备注 |
|---|---|---|
| 散文单流（bf16 hc 之后） | **en 325.6 / zh 332.3** | 安静 GPU |
| 散文单流（修正后基线，bf16 之前） | en ~305–310 / zh ~289 | 【权威】对 455 达成 67–69% |
| 散文 `round_ms` | **12.2–13.1 ms** | 比 BENCH 题面的 15.6 ms 更低 |
| 散文 `tok/round` | en 3.31–3.48 → **4.26**（bf16 前 → 后） | zh 2.40–2.59 → 2.49 |
| 散文接受率 | en 0.570–0.578 / zh 0.483–0.509 | 中文是短板 |
| 达成率（对 455） | en **67%** / zh **64%** | |

⇒ **散文完全由 `tok/round` 决定，不是 device 时间**。`round_ms` 已经比 BENCH 题面低 20%，
而速度还差 33%，缺口全在"每轮吐几个 token"。

---

## 3. 单流分域吞吐（`c1_domains.py`：贪心、600 tok、best-of-2、同窗口交错）

| 代际 | 代码 | 中文口语 | 英文散文 | 中文散文 |
|---|---|---|---|---|
| 原头 | 404.9 | 182.6 | ~296 | ~296 |
| v5（11-18 线上） | 449–482 | 189–198 | ~306 | ~290 |
| v9（11-18 收工） | — | — | 315 / 300（**§13 声称 +11.2%**） | 【已作废】 |
| **宽集 120 条重测（11-19）** | 五个头全部落在 **2.3% 内** | | | |

【已作废】详见 §6：`c1_prose.py` 的三条英文 prompt 与训练语料 `TECH_DOMAINS` 的第 2/4/3 条
**逐字重叠**，它测的是记忆。

---

## 4. device 时间构成（CUPTI，c1，30 步窗口，30 步合计 377.8 ms ≈ 12.6 ms/步）

| 类别 | 时间 | 占比 | 每步 | 主要 kernel（窗口内调用次数） |
|---|---|---|---|---|
| **MoE (w3)** | **176.04 ms** | **46.6%** | 5.9 ms | `w3_gemm`(6931) 82.3、`moe11gemm` bf16(1440) 28.1 + 3bit(1440) 16.5、`moe11prep`(1488) 6.4、`_moe_combine_pairs`(1488) 4.9、`quant_groups`(4704) 4.7 |
| GEMM（稠密） | 81.00 ms | 21.4% | 2.7 ms | `sm89_xmma_gemm_e4m3…32x64x64`(6165) **38.7**、cutlass wmma bf16(1080) 15.9、wmma s161616(1488) 5.5、`_expert_gemv_fused`(294) 4.8、`_fp8_gemv_quant`(1031) 8.2 |
| other | 73.53 ms | 19.5% | 2.45 ms | 大量小 kernel（`_quant_small` 101/步、`direct_copy` 128+/步、`_budget_select` 48/步） |
| norm | 26.83 ms | 7.1% | 0.89 ms | `_inject_norm`(2914) 12.6、`_layer_norm_fwd_1pass`(1116) 12.1 |
| gdn/mamba | 12.53 ms | 3.3% | 0.42 ms | `fused_sigmoid_gating_delta_rule_update`(1080) 12.0 |
| attention | 5.58 ms | 1.5% | 0.19 ms | flashinfer FA2 |
| ple | 2.30 ms | 0.6% | 0.08 ms | 主机侧 gather 落到 device 的部分 |

另一口径（`kernel-fusion-spec.md` §0，引 `HANDOFF` §13）：**「小内核堆」= 2.65 ms/步 = c1 的 19%**。

**注意 MoE 占比的两个口径**：46.6%（上表，按 kernel 名归类，有逐项明细）vs 52%（`LESSONS` §六.6 的早期粗算）。
本仓库统一采用**有明细的 46.6%**。

---

## 5. 主机侧账（CUPTI host，同一 30 步窗口，292.1 ms ≈ 9.7 ms/步，但已被 device 遮住）

| 项 | 时间 | 次数 | 单次 |
|---|---|---|---|
| `cudaGraphLaunch` | 98.05 ms | 237 | **414 µs（WDDM）** |
| `cudaEventQuery` | 35.81 ms | 213 023 | 0.17 µs |
| `cudaMemcpyAsync` | 21.01 ms | 1 881 | 11 µs |
| `cudaLaunchKernel` | 17.77 ms | 1 433 | 12 µs |
| `ChunkGatedDeltaRuleFunction` | 15.29 ms | 36 | 425 µs（疑似同步点） |
| `aten::copy_` | 11.04 ms | 2 759 | 4 µs |

- verify 图 ~**500 节点** ⇒ 单次 `cudaGraphLaunch` **0.4–0.85 ms/轮**；draft 图 ~50 节点 ⇒ 0.02–0.03 ms。
  **比值 20–30×**，且 WDDM 下 graph 启动成本是 Linux 的 **10–20 倍**、**随节点数线性涨**。
- draft 链：8 步 × 0.23 ms = 1.84 ms/轮，其中权重读 0.104 ms/步（1.25 TB/s），**另一半是 15–20 个小 kernel 的派发**。
- 最终态 `HIBANA_STEP_TRACE`：c1/c8/c16 的 `host` 都是 **−0.1 ms**（wall ≈ gpu）⇒ 主机侧已不在关键路径。

---

## 6. ★ 草稿头横向对比：120 条全新 prompt（唯一未被污染的宽集）

`eval_wide120.py` + `eval_prompts_120.json`：**技术说理 40 + 推理 40 + 叙事 40**，
三域句式模板与任何训练语料都不同；贪心、每条 240 token；接受率取引擎累计计数器差分。

| 草稿头 | accept | tok/round | round_ms | 折合 tok/s |
|---|---|---|---|---|
| v16reason（§13 声称 +11.2% 的那个） | 0.5969 | 4.107 | 13.22 | **310.7** |
| v5 | 0.5894 | 3.975 | 12.85 | 309.3 |
| v9 | 0.5922 | 4.059 | 13.19 | 307.7 |
| v22（骨架） | 0.5912 | 4.030 | 13.12 | 307.2 |
| **orig（原始未训练头）** | 0.5912 | 3.986 | 13.12 | **303.8** |

**全部落在 2.3% 以内。** 另用 40 条技术/推理 prompt × 700 token 交错 A/B：
orig 0.5980 / 3.934 vs v16reason 0.5931 / 3.863 —— **原头还略好**。

对照：同一窗口用被污染的窄集测，orig 292.4/295.9 vs v16reason 348.8/349.5（**+19%**）。
⇒ 那 +19% 是**对三道题的记忆**。这就是 `docs/06` 的全部内容。

---

## 7. 深度表扫描（`HIBANA_SPEC_DEPTHS`）

**三点消融**（`HIBANA_STEP_TRACE=1`，wall/gpu ms，c1）

| 深度 | wall | gpu | tok/round | 折算 tok/s |
|---|---|---|---|---|
| d=1 | **12.6** | 12.6 | 1.85 | ~147 |
| d=4 | 14.3 | 14.3 | 2.75 | ~192 |
| d=8 | 15.6 | 15.6 | 3.94 | ~253 |

⇒ 每多一个草稿步只 +0.4–0.5 ms，但只多 0.35 个 token（第 2–8 步接受率 ~0.30）。

**三点消融**（`HIBANA_SPEC_DEPTHS` 深度扫描，贪心，c1，**11-19 宽集口径复测**）

| 开关 | 宽集 accept | 结论 |
|---|---|---|
| 基线（v16reason、深度 8:3、fp32 SSM） | **0.5969** | 最优 |
| `HIBANA_SSM_STATE=int8` | 0.5896 | −1.2%，否决 |
| 深度 `1:6` | 0.6181 但 tok/round 3.943 | 折合 303 < 313，否决 |
| 深度 `4:4` | 0.5894 | 否决 |
| `--throughput` profile | **0.4445** | 启用放宽验收，**违反约束**，否决 |

---

## 8. 收益累积（每一步都过了位一致闸门）

| 阶段 | en_prose 单流 | c16 聚合 | 关键改动 |
|---|---|---|---|
| 移植完成（11-16 14:10） | 107.5 | 198.7 | 能跑 |
| + SAA 配方 + SM12.0 gate（14:30） | 143.9 | 218.8 | 生产配方、图融合解锁 |
| + 修 PLE 线程池（18:05） | 215.9 | 673.3 | 每轮不再新建线程池 |
| + 按并发深度表（21:20） | 214.5 | **903** | 每轮握手 11 → 4 |
| + split-K 除数放开（22:10） | 224.5 | 844.9 | 80 → 200 block |
| + 逐档深度调优（23:30） | 221.3 | 826–888 | c2 6 / c4 3 / c8 2 / c16 4 |
| + PLE memmap fancy-index（11-17 01:30） | — | **1656** | 2048 行 36 ms → 0.67 ms |
| + plan 缓存 + 页表原地增长（05:30） | — | 1699 | plan 0.29→0.07 ms/轮 |
| 最终基线（11-17 15:1x） | 270.8 | 1608.4 | 稳定态 |
| 宽集修正（11-19 10:15） | ~305–310 | 1907.2 | 口径纠正 |
| + bf16 hyper-connection（11-19） | **325.6** | **2035.5** | 少读字节 ≠ 更快 |

**从移植完成到最终：单流 ×3.0、c16 ×10.2。** 其中纯引擎侧（不含移植本身）贡献约 ×2.8 / ×8.2。

---

## 9. 这张表不能证明什么（写清楚，免得被误引）

1. **不能跨 prompt 集比较。** 同一份代码、同一个头，窄集与宽集差 19%（§6）。
2. **不能跨显存状态比较。** 同卡上有 co-tenant（ComfyUI 48 GB / Strata 77 GB）时，c1 会掉 10–15%。
3. **不能把 `temp 0.3` 与贪心混在一张表里比。** 差 15–27%，是解码差异。
4. **c4 的 "724–847" 是区间不是均值**：该点对显存余量最敏感，区间内波动来自 co-tenant 退出时机。
5. **天花板（455/700/930/1570/1690）是带宽估算，不是实测极限**；c16 已越过 120%，说明估算保守。

---

## 9.5 口径演变：同一个指标在不同轮次被改过（务必按日期引用）

这份工作跨了 4 天、45 轮，**同一指标被反复修订**。下表把修订全列出来，避免你引用了被推翻的那一版。

| 指标 | 早期值 | 修订值 | 谁改的 / 为什么 |
|---|---|---|---|
| MoE 占 device 时间 | 52 %（粗算） | **46.6 %**（CUPTI 逐 kernel，11-18） | 按 kernel 名归类后可核对；本文 §4 用这一版 |
| MoE 占 device 时间（再修订） | 46.6 % | **~7.6 % / ≈1.0 ms 每步**（HANDOFF 后期） | 归属重算（把 `moe11gemm`/`quant_groups` 等归到 trunk 侧）；两版差 6 倍，引用时必须带出处 |
| 每轮专家流量 | 11.6 GB | **6.1 GB** | 除数由 256 改为 512（实际专家数） |
| 批量 PLE 取行 | +6 %（c4/c8） | **+2~3 %** | 跨窗口漂移，原测偏高 |
| `c1_prose` 窄集增益 | +11 % ~ +25 % | **≈0 %**（120 条宽集） | 评测污染，见 §6 |
| 达到 455 tok/s 所需接受率 | 0.76 | **0.87（平线）** | 逐位置接受率按 0.73 平线重算 |

**引用规则**：本文 §1/§4/§5 的表都标了日期与仪器（CUPTI / STEP_TRACE / 外部 sweep），
它们是**可复现的原始记录**；HANDOFF 里的后期修订值请在 `reference/HANDOFF.md` 里按节回查。
如果两个数字冲突，**以"带仪器 + 带日期"的那个为准**。

---

## 10. 复现命令

```powershell
# 五点（贪心）
python bench\sweep.py 1,2,4,8,16 700

# 五点（采样 0.3）
python bench\accept_sampled.py sampled-0.3

# 单流散文
python bench\c1_prose.py prose

# 宽集 120 条（草稿头横向对比，唯一可信口径）
# swap the head first: set HIBANA_MTP_WEIGHTS=<orig export> and run launch\restart.ps1
python bench\eval_wide120.py orig
# then swap to the distilled export and run again
python bench\eval_wide120.py v16reason

# 位一致闸门
python bench\refcheck.py cmp base
```
