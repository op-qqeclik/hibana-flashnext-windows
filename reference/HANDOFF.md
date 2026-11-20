# 107 · hibana Flash-Next · 交接说明（2026-11-17 04:10）

## 现状（一句话）
Windows 原生跑通、稳定、单流 2.4–2.8× Spark 优化后、**16 并发 1709 tok/s（超估算上限 1690）**；
c1/c2/c4/c8 分别 48/55/67/67%。

## 服务
```
B:\hibana-win\run-flashnext.bat          # 启动（含 47.7 GB PLE 预载，约 4 分钟到 serving）
http://127.0.0.1:30004                    # OpenAI 兼容
B:\hibana-win\results\flashnext.log      # 日志
```
关键环境（脚本里已固化）：`HIBANA_SPEC_DEPTHS=1:8,2:6,4:3,8:2,16:4`、`HIBANA_MTP_WEIGHTS=B:\hibana-models\mtp\mtp_weights.safetensors`、
`HIBANA_KV_FORMAT=fp8`（nvfp4 在本移植编不过）、`HIBANA_VRAM_RESERVE_GIB=2`、`HIBANA_KV_POOL_TOKENS=524288`、
`HIBANA_STATE_SLOTS=36`、`HIBANA_GDN_PREFILL=triton`、`HIBANA_CUDA_ARCH=12.0a`、VS2022 + `C:\cuda13` + `win_compat.h`。

## 本目录文件（都是 107 上 B:\hibana-win 的镜像）
| 文件 | 改了什么 | 效果 |
|---|---|---|
| `hibana_model_flashnext_ple.py` | Windows 取行从「32 线程池逐行 seek+read」改成 **memmap fancy-indexing** | 2048 行 36 ms → 0.67 ms（53×）；16 并发 791 → 1709 |
| `hibana_kernels_w3.py` | `splits_for` 允许所有合去除数（≤16），K=2560 由 4 → 10 | 单流 +5~14% |
| `hibana_engine_loop.py` | 新增 `HIBANA_STEP_TRACE`（wall/gpu/host）与 `HIBANA_PROFILE_STEPS`+`SENTINEL`（设备/主机分解） | 诊断用，默认关 |
| `hibana_engine_spec_graphs.py` | `HIBANA_PLE_BATCHED` 开关（默认关；实测批量更差） | 已排除 |
| `hibana_engine_executor.py` | flashinfer workspace 128/256 MB → 512 MB/1 GiB | fp8 KV + 256 head_dim 的 prefill 临时区要 282 MiB |
| `hibana_engine_graphs.py` | 同上（decode ws 512 MB） | |
| `hibana_proposers_mtp_graphs.py` | `STATS`（每步 draft 的 copies/plan/launch 计时） | 诊断用 |
| `hibana__plat.py` | Windows 垫片：flock/pread/madvise/page_size/kill_tree + `wait_event`（轮询版，实测中性） | |
| `run-flashnext.bat` | 最终启动脚本 | |

## 下一步（唯一剩下的量级）
c1/c2/c4/c8 的差距 = 每步约 500 个小内核：
- 205 × `sm89_xmma_gemm_e4m3…`（flashinfer 注意力 GEMM，6.8 µs/次）
- 97 × `_inject_norm_kernel`（4.4 µs）、104 × `_quant_small`、101 × `_pack`
- 48 × MoE fc1/fc2（这部分是带宽，≈1.3 TB/s，动不了）、36 × gdn

单步 16.4 ms 构成：MoE 6.1 / gemm 2.7 / other 2.5 / norm 0.9 / gdn 0.4 / draft 串行间隙 3.7。
**要动的是 `kernels/csrc` 的融合层与图结构**（把 inject_norm+quant+pack 合成一层一个内核；attention GEMM 合并）。
这不是配置、不是端口问题，是引擎级改造。

## 已被否定的方向（不要再试）
- PLE 批量取行（更差，逐列流式是对的）
- 轮询式事件等待（中性，1.85 ms 是真在等 GPU）
- `HIBANA_VERIFY_STATE=snapshots`（96 GiB 装不下，请求堵在准入）
- `HIBANA_KV_FORMAT=nvfp4`（flashinfer FP4-KV 模块编不过）
- 加深草稿（每步是完整 MTP 头 forward，加深净亏）
- 按 SM 数放大 `SPLIT_TARGET`（`chunks % 2·splits` 卡死，已用除数方案替代）

---

# round 20 分析结论：单流 gap 的准确归属与实施方案

## 已确认的账（测量 + 代码）

c1 一轮 16.4 ms，device 12.6 ms，**gap 3.8 ms**。gap 由 draft 链的**主机串行工作**构成：
`mtp_graphs.run()` 每步 `_plan()`（0.28 ms，主机 CPU，flashinfer paged-prefill 的 Python 开销）
+ copies 0.04 + replay 调用 0.04 + `_column` 记录，共 **0.37 ms/步 × 8 步 ≈ 3.0 ms**。
（0.37 ms 的分解是在 c16 的 `STATS` 上测的：`copies 0.04 / plan 0.28 / replay-call 0.04`；
c1 走同一代码路径、b=1，数值未单独测，见下面的验证门槛。）

device 侧 12.6 ms 里 MoE+trunk 6.1 ms 已是 1.6 TB/s（不可动），故 **c1 真实地板 ≈ 316 tok/s**（不是 455）。

## 关键发现：整轮的 plan 在循环前就已知

`proposers/mtp.py::propose` 第 198–202 行：
```python
full.append(torch.cat([req.pages[1:pos], pages[i, :depth]]))   # 一次构建整轮每步要 attend 的 pages
step_pages = pages[:, :depth].t().contiguous()
```
且 `mtp_graphs.py::__init__` 里已有 `self._pinned`（**每步一个 pinned buffer**，注释明说
"the host plans every step of a round before the GPU reaches those copies"）。
也就是说：**设计上本来就允许把 8 个 plan 提前发出去**，只是现在仍在 `run()` 里同步执行。

## 实施方案（下一步直接照做）

1. 在 `DraftGraphs` 加 `plan_round(b, kvs, pad)`：用一个 `concurrent.futures.ThreadPoolExecutor(max_workers=1)`
   依次对 step 0..depth-1 调 `self._plan(st, list(kv)+pad, j)`，返回 futures。
2. `proposers/mtp.py::propose` 循环前调用 `plan_round(...)`（它已有全部 `kv` 切片所需数据）。
3. `DraftGraphs.run(step=j)` 里把 `self._plan(...)` 换成「等待第 j 个 future」。
4. **必须处理的竞态**：`_plan` 会改 `wrapper._pin_memory_int_workspace_buffer` 并让 `paged_prefill` 把 schedule
   拷到**同一组 graph-bound device buffer**。因此 plan(j) 必须满足：在 replay(j-1) 消费完之后、replay(j) 之前完成。
   - 做法：worker 线程内**顺序**执行 plan（单 worker），并在 `run(step=j)` 里 `fut[j].result()` 之后再 replay；
     `torch.cuda.current_stream()` 是设备级共享，拷贝顺序仍由调用时刻决定 → 若实测出现数值偏差，退化为
     「只预发 j+1 一个，且在 run(j) 的 replay 之后立即提交」。
5. **验证门槛（不可跳过）**
   - 数值：同一 prompt、temperature 0、seed 固定，比较改动前后 200 token 的**完全一致**（`token ids` 逐位相同）；
   - 吞吐：`bench_matrix.py` 的 c1/c2/c4/c8/c16 五点，对比基线 218/384/624/1055/1709；
   - 回归：soak 60 轮无失败。
6. **回退**：`proposers/mtp.py` 与 `mtp_graphs.py` 两个文件整体回退即可（已有原文件副本在 107 的 `.pre-*` 与本地存档）。

## 预期与不确定性

- 若 plan 真的能从关键路径移走：c1 3.8 ms → ~0.8 ms，`244 → ~290`（+19%），c2/c4 类似幅度的 10–15%。
- 不确定点：c1 的 0.37 ms/步是**在 c16 上测的**，c1 未单独测。实施前先花 5 分钟在 c1 上确认这个分解
  （带 `HIBANA_STEP_TRACE=1` 跑一次 400 token 生成，读 `draft step: N replays, copies/plan/replay-call` 那行）。
- 生成结论前必须重新取 c1 的 STATS：如果 plan 在 c1 只有 0.05 ms，那 gap 的来源另找（优先怀疑 `_column` 的 D2H 链）。

---

# round 21：上一节的方案被推翻（重要，勿照做）

## 1. c1 的分解已在 c1 上实测（不再是从 c16 推的）

带着 `HIBANA_STEP_TRACE=1` 在 c1 跑两次 400 token 生成，服务日志：

```
draft step: 200 replays, copies 0.04 ms, plan 0.29 ms, replay-call 0.04 ms
draft step: 400 replays, copies 0.04 ms, plan 0.28 ms, replay-call 0.04 ms
```

即 c1 每步 0.37 ms，8 步 ≈ **2.96 ms**，加上 draft→verify fence，对上了 3.8 ms 的 gap。
**plan 占其中 2.3 ms（78%）**，诊断成立。

## 2. 但「把 8 个 plan 提前预发」这个方案是错的，别做

读代码后确认：flashinfer 的 plan 把 schedule 写进 wrapper 的 pinned buffer，再拷到**同一个 graph-bound
device buffer**。每个 plan 覆盖前一步的 schedule，拷贝顺序靠「排在上一 replay 之后」的流序保证。
所以 plan(j) 必须在 replay(j-1) 之后、replay(j) 之前——**没有任何「提前算完 8 步」的空间**，
`_pinned` 那组每步一个的 host buffer 只解决「主机不被拷贝阻塞」，不代表 schedule 能并行存在。

若照 round 20 的方案做，结果会是：前 7 步读到第 8 步的 schedule → 注意力**静默算错**（不报错、只出垃圾）。
本节取代 round 20 的第 1–4 步。

## 3. 剩下的三条可走路径（按可行性排序）

1. **砍掉 plan 的绝对开销**（最有希望）：0.28 ms 对一个「1 行 query、几十个 page」的 prefill 来说是纯 Python 开销。
   方向：draft 的注意力不走 `BatchPrefillWithPagedKVCacheWrapper.plan()`，换成
   - `BatchPrefillWithRaggedKVCacheWrapper`（静态 indptr，一次 plan 服务整轮；ragged 不依赖 page 表可增量 plan），或
   - 自己写一个 1 行 paged prefill 的 Triton 内核（MTP 头只有 1 行，head_dim 固定，难度中等）。
   验收：c1 的 `plan` 降到 <0.05 ms，且 200 token 逐位一致。
2. **减少每轮的 plan 次数**：深度表 1:8 → 1:6，少 2 次 plan（省 0.6 ms）但 tok/round 从 3.98 掉到约 3.2，
   之前实测是净亏；除非路径 1 失败才回头试。
3. **接受现状**：c1 244/316 = 77%，c16 已在地板（101% 上限）。

## 4. 结论

单流剩下的 3.8 ms 里，2.3 ms 是 flashinfer plan 的固定 Python 开销——它是「引擎按 GB10 设计的每秒 10 步」
假设下的产物，在 1.6 TB/s 的卡上每步 0.28 ms 显得很贵。**要动它，改的是 draft 注意力后端，不是调度。**

---

# round 23：plan 的 0.28 ms 拆开了（有 0.15 ms 是 hibana 自己的 Python，可安全优化）

## 探针结果（107 上实测，同 shape：b=1 / 512 pages / 8 q-heads / 2 kv-heads / hd=128）

```
paged  plan ms: 0.122      # flashinfer BatchPrefillWithPagedKVCacheWrapper.plan()
ragged plan ms: 0.056      # flashinfer BatchPrefillWithRaggedKVCacheWrapper.plan()
```

而引擎里每步 `_plan` 是 **0.28 ms**。结论：

- flashinfer 的 plan 本身只占 **0.12 ms**；
- 剩下 **0.15 ms 是 `mtp_graphs._plan` 自己的每步 Python 开销**：
  `check_context_capacity`、`torch.tensor([0] + torch.tensor([p.shape[0] ...]).cumsum(0).tolist())`（两次建张量 + cumsum + tolist）、
  `torch.arange(0, b+1)`、`torch.ones(b)`、`_pinned` 的 while 循环。

## 因此下一步最划算的不是换注意力后端，而是这 0.15 ms

改法（**低风险、不改变流序、无竞态**，因为这些张量都是常量/主机侧）：

1. `_static()` 里预建 `st["plan_qo_indptr"] = torch.arange(0, b+1, dtype=torch.int32)` 与
   `st["plan_last"] = torch.ones(b, dtype=torch.int32)`（**注意原代码这两个是 CPU 张量**，别换成 device 版，flashinfer 的
   plan 需要主机侧的；这就是为什么不能复用 `st["qo_indptr"]`/`st["last_page_len"]`）；
2. `_plan()` 里 b==1 时 `indptr = torch.tensor([0, counts[0]], dtype=torch.int32)`（省掉 cumsum+tolist 链）；
   b>1 时用纯 Python 累加后一次建张量；
3. `check_context_capacity` 只在容量的 `max_context` 变化时算一次（每轮一次即可）。

预期：0.28 → **0.13–0.15 ms/步**，c1 一轮省 **1.0–1.2 ms** → `244 → 262–268`（+7~10%）。
更小的风险、更确定的数字，比换后端优先。

## 仍然可行的更大一步（若上面做完还想继续）

`ragged` 比 `paged` 快 2.2×（0.056 vs 0.122），但需要把 draft 的注意力从 paged 换成 ragged + 静态 indptr，
并保证 200 token 逐位一致。收益上限 ~0.07 ms/步（约 0.5 ms/轮），**性价比低于第 1 项**。

---

# round 24：微优化实测无效，已回退（负面结论）

按 round 23 的方案改了 `mtp_graphs._plan`：预建并缓存两个主机侧张量（`plan_qo` / `plan_last`）、
b==1 时 `indptr` 直接 `[0, count]`（去掉两次建张量 + `cumsum(0).tolist()` 链）。

**实测（c1，两次 400 token）：**

```
改前: draft step: 200 replays, copies 0.04 ms, plan 0.29 ms, replay-call 0.04 ms
改后: draft step: 200 replays, copies 0.04 ms, plan 0.29 ms, replay-call 0.04 ms
```

**完全没动。** 说明那 0.28 ms 不是 hibana 侧的重复建张量，而是 **flashinfer 的 `plan()` 在真实几何下就是 0.28 ms**
（我的隔离探针用 8/2 heads + hd=128 测到 0.122 ms，差在真实几何——很可能 head_dim 更大或带 `window_left`，
走了另一条 plan 路径）。已回退（`.pre-micro`），服务恢复标准启动脚本。

## 结论：这条路要动就得进 flashinfer 内部

下一步只有三个选择，都超出「改几行 Python」的范围：

1. **量清楚真实几何下 plan 的 0.28 ms 花在哪**：用 `cProfile` 包住一次 `paged_prefill(...)`，看是 Python 校验、
   kernel 选择表，还是 `plan_info` 构造。（约 15 分钟，值得先做）
2. 按几何对齐后重测 **ragged** 是否仍快 2.2×；若仍快，换后端（要逐位回归）。
3. 接受：c1 244/316 = 77%，c16 已 101% 上限。

## 方法论提醒（写给下一个接手的人）

隔离探针的结论**不能直接套到引擎里**：同样的 plan 在探针 shape 下 0.122 ms，在引擎真实几何下 0.28 ms。
**任何"改这里能省 X ms"的推断，必须在引擎里用 `STATS`/`STEP_TRACE` 复测才算数。**这一轮就是靠这个及时回退了。

---

# round 25–27：插桩失败三次，教训清单（先读这一节再动手）

三次尝试给 `_plan`/`run()` 挂 `cProfile`，全部没拿到数据。原因分别不同，但都属同一类错误：

| 轮次 | 做法 | 失败原因 |
|---|---|---|
| 25 | 探针挂在 `_plan()` 里，捕获得第一个调用即触发 | 捕获阶段对象状态与运行期不同（`attention`/`geometry` 可能未绑定），异常被捕获路径吞掉，服务照常但探针静默不跑 |
| 26 | 只做评估，未动手 | — |
| 27 | 移到 `run()`，锚点按真实文本匹配 | `ast.parse` 通过，但**没确认服务到 serving 就发请求**（这台机器加载约 4 分钟，`sleep 245` 卡在边界），请求可能发给了未就绪的进程 |

## 插桩的正确做法（下次照这个来）

1. **先确认就绪，再做实验**：轮询 `curl http://127.0.0.1:30004/health` 直到 200，再发第一个请求。**不要用固定 sleep。**
2. **探针必须有落盘的正反馈**：触发时先 `open(r"B:\hibana-win\results\planprof.done","w").write(repr(...))`，
   再打日志。这样"没执行"和"执行了但输出丢了"能区分开。
3. **一次只改一处**：探针位置与测量方法不要同时换，否则失败无法归因。
4. **改完必须重启进程**：否则"已回退"只发生在磁盘上，跑的还是旧代码。
5. **隔离探针的结论不能套进引擎**：同一个 `plan()` 在探针 shape 下 0.122 ms、引擎真实几何下 0.28 ms。
   任何"能省 X ms"的推断，都要用 `STATS`/`STEP_TRACE` 在引擎里复测。

## 现状（round 27 结束，已回退到干净版本）

- 引擎文件无探针残留（107 上保留 `.pre-prof` / `.pre2` / `.pre-micro` 三份备份）
- 服务：`run-flashnext.bat` 标准启动，`/health` 正常
- 性能：c1 218–245、c2 384、c4 624、c8 1055、c16 1709（101% 上限）
- 未完成：那 0.28 ms/步的 flashinfer plan 诊断（下一步见 round 23 的三个选项）

---

# round 28–32（2026-11-17 05:00）：把 plan 从关键路径上拿掉 —— 以及更重要的：先把账量准

> 上一轮（25–27）三次插桩失败。这一轮换了做法：**探针挂在 `DraftGraphs.run()`（必定执行），
> 先写 `results/planprof.done` 再做事**，并且只用哨兵文件点触发；工具是 `hibana/proposers/_planprof.py`。

## 1. 量准了：plan 不是 0.28 ms，而且 0.28 里有一半是「排在 GPU 后面的拷贝」

引擎内探针（c1，真实几何 24 q-heads / 2 kv-heads / hd 256 / page 1 / bf16，36 页）：

```
flashinfer plan 内部（pmark 分段，ms）:
  plan: workspace + dtype checks                 0.002
  plan: attrs + mask checks                      0.002
  plan: qo_indptr.to(cpu) + max_q_len            0.015
  plan: kv_lens host + pageable H2D + max()      0.040   <- 最大一块
  plan: copy qo_indptr  (pageable H2D, 2 int)    0.011
  plan: copy kv_indptr  (pageable H2D, 2 int)    0.008
  plan: copy last_page_len (pageable H2D, 1 int) 0.007
  plan: copy kv_indices (D2D)                    0.015
  plan: backend + module select                  0.001
  plan: args build                               0.001
  plan: module.plan C++ (schedule + memcpy)      0.016
  ------------------------------------------------------
  flashinfer plan 合计                           0.117
  hibana 自己（indptr/pages/qo/last）            0.023
  _plan() 合计                                   0.142   (36 页；300+ 页时 ~0.19)
```

`STATS` 里那个 0.21–0.24 ms 比隔离值高，是因为**从 pageable 内存发起的 3 个 H2D 拷贝会等流上已排队的
工作**（`copy_(non_blocking=True)` 对 pageable 源仍是阻塞的），而在隔离探针里队列是空的。
**结论：`plan` 的绝对开销里，可砍的是「每步 3 个 pageable H2D 拷贝 + 整次 plan 调用」，不是 kernel 选择。**

## 2. c1 的真实账（py-spy 采样引擎线程 + 引擎内计时器，静默期）

一轮 12.7 ms 的构成（引擎线程 25.2 s / 1357 轮，py-spy 250 Hz）：

| 阶段 | 占比 | 每轮 |
|---|---|---|
| `WAIT under AcceptGraphs.round`（verify+accept 在跑，主机在等） | 46.5% | 8.65 ms |
| `Scheduler._prefill_step`（我的负载是 300 token 请求，含 prompt 前填） | 11.0% | — |
| `VerifyGraphs.run`（含 verify 的 plan + 拷贝 + **图启动 0.7 ms**） | 9.6% | 1.78 ms |
| `FlashNextExtras.fill`（PLE 行，**其中大部分是在等 draft 链**） | 9.5% | 1.78 ms |
| `DraftGraphs.run`（8 步 × plan/拷贝/启动） | 9.3% | 1.73 ms |
| accept 主机段 / propose / prepare | 7.4% | 1.7 ms |

**关键更正**：`fill`（PLE）的 1.78 ms 主要是**等 draft 链的 GPU 时间**（9 列里每列 `events[j].synchronize()`
等的正是第 j 步），不是主机 gather 开销。按 c1 实测，PLE 已经被 draft 链重叠掉了 —— 所以
「每轮 8–9 次 PLE 主机握手」在**现在的代码上**不是单流瓶颈（`HIBANA_PLE_BATCHED=1` 当年更差也是这个原因），
真正的瓶颈是 **device 侧 verify+accept 8.65 ms（MoE 占其 52%）** 与 **verify 图启动 0.7 ms**。

另一个反直觉的数：`verify graph replay` 的 CPU 成本 0.6–0.85 ms/轮，而 draft 的 8 次 replay 各只要
0.02–0.03 ms —— 同一个 `cudaGraph.replay()`，差别在图规模：verify 图有 ~500 个节点，WDDM 下启动成本随节点数走。
**想省这 0.7 ms，只能让图里的内核变少（融合），不是调度问题。**

## 3. 拿到的优化：FA2 的 schedule 是 (split 大小, 每请求 chunk 数) 的纯函数

读 `flashinfer/data/include/flashinfer/attention/scheduler.cuh`：
`PrefillSplitQOKVIndptr` 里 cuda-graph 模式下 `split_kv` 恒为真，chunk 大小由
`PrefillBinarySearchKVChunkSize` 二分（下界 `max(128/page_size,1)`，容量 `2·SMs/num_kv_heads`），
其余 int workspace 内容全由这一对参数决定。**因此可以缓存：key 相同 ⇒ plan 会写出一模一样的 schedule。**

引擎内验证（`planprof.keycheck` 哨兵，31 个长度直到 32K 上下文，两种形状各 1 行/9 行）：

```
rows=1: 16 keys, consistent   key=(128,(1..188)) ... key=(129,(187,)) key=(149,(188,))
rows=9: 16 keys, consistent   key=(128,(1..32))  key=(175,(47,)) key=(349,(47,)) key=(596,(47,))
VERDICT: schedule cache exact
```

也就是 **plan 可以按 key 跳过，逐位一致**。实现（`proposers/mtp_graphs.py`、`engine/spec_graphs.py`）：

1. 每步的 index 指针改用**pinned 平铺缓冲、每步一格**（cm: 异步拷贝在 GPU 执行时才读源，共用一个槽会读到下一步的值 —— 第一次就是这样把 accept 率打到 0.558，逐位比对抓出来的）；
2. key 命中时只刷新 `kv_indptr` / `kv_indices`（qo_indptr、last_page_len 是常量，捕获时就已正确）；
3. `HIBANA_PLAN_CACHE=0` 可一键回到「每步都 plan」。

## 4. 本轮改动与效果（静默期实测）

| 文件 | 改了什么 | 效果 |
|---|---|---|
| `proposers/mtp_graphs.py` | draft 的 plan 缓存 + pinned 每步 indptr + 跳过常量 index 拷贝 | draft plan **0.21 → 0.12 ms/步**，8 步省 ~0.8 ms/轮 |
| `engine/spec_graphs.py` | verify 的 plan 同样缓存 + pinned indptr；新增 `VSTATS` 分项计时 | verify plan **0.29 → 0.07 ms/轮** |
| `proposers/_planprof.py`（新） | 哨兵触发的引擎内探针：plan 分段、cProfile、pageable/pinned 对照、backlog 阻塞测试、schedule key 校验 | 诊断用，默认不动 |
| `cache/request_state.py` + `engine/request.py` | 页表原地增长（`page_buf` 追加）替代每轮 `torch.cat` | 去掉每轮 O(context) 的设备拷贝（短上下文 ~0.3 ms/轮，长上下文更明显） |
| `engine/spec.py` | `spec_stats` always 暴露 `draft/verify/accept_ms_per_round` | 诊断用 |
| `flashinfer/prefill.py` | `_pmark()` 空钩子（12 处 phase 标记，不挂钩子时零成本）；`patch_fi.py` 可重放 | 诊断用 |

**五点并发（贪心，静默期，400 token/请求）**

| 点 | 会话开始时 | 现在 | 用户目标 |
|---|---|---|---|
| c1 | 218–245（best-of-3 256.7） | **248**（best-of-3 **273.2**） | 455 |
| c2 | 384 | **441** | 700 |
| c4 | 624 | **656** | 930 |
| c8 | 1055 | **1086** | 1570 |
| c16 | 1709 | **1699** | 1690（已在地板） |

`round_ms`（c1）：13.96 → **12.66**。验收：同一 prompt、temp 0、seed 1234、200 token，
`md5 ad4d58f48441deeace02967e2fe12c88` 改动前后**完全一致**（`refcheck.py cmp base`）。

## 5. 下一步（按性价比）

1. **减少 verify 图的节点数**：它同时压 device 时间和 0.7 ms 的启动成本。第一步是
   `_inject_norm`(94/步)+`_quant_small`(101/步)+`_budget_select`(48/步) 三层并一层。
   device 分解（c1，contended 采样）：moe 52.1% / gemm 19.6% / other 19.5% / norm 4.3% / ple 1.9% / gdn 1.6% / attn 1.0%。
2. c1 的地板：verify+accept 8.65（MoE 6.1 不可动）+ draft 链 2.4 ≈ **11 ms ⇒ ~350 tok/s**（主机降到 0 也只是这个数）。
   要到 455 必须动非 MoE 的 device 工作（上面第 1 条），或者提高 tok/round。
3. 深度：本轮在静默期复核 1:8 与 1:6，c1 分别 271.4 / 269.1（best-of-3），1:4 明显更差 —— **保持 1:8**。

## 6. 复现步骤（下次直接用）

```
B:\hibana-win\restart.ps1 -Bat B:\hibana-win\run-flashnext-tune.bat   # 停旧进程+plan任务启动+轮询 /health（不用 sleep）
set /p 深度 < results\depths.txt                                     # 深度扫描只改这个文件
.venv\Scripts\python.exe refcheck.py cmp base                       # 逐位一致 + c1 best-of-3
.venv\Scripts\python.exe sweep.py 1,2,4,8,16 400                    # 五点贪心曲线
.venv\Scripts\python.exe soak.py                                    # 4 分钟混合负载
# 诊断：往 results\ 放 planprof.on / planprof.keycheck，再发一个请求，读 results\planprof.done
# 主机侧火焰：.venv\Scripts\py-spy.exe record --pid <引擎 pid> -d 26 -r 250 -f speedscope -o results\host.speedscope
```

## 7. round 33：两个方法论补充（都是踩过的坑）

**（a）深度复核：1:8 仍然是对的，但理由要写清楚**

上一轮把 `HIBANA_SPEC_DEPTHS` 的 1:6/1:4 试验记为「净亏」。本轮用「把深度表写进 `results\depths.txt`、
`set /p` 读进 bat」的方式重跑，**第一次是空实验**：bat 里的 `set /p` 没生效，日志头那行
`verify depth by batch 1:8,...` 仍是默认值，而我没核对它 —— 于是「6 和 8 一样快」其实是在跑同一个配置。
修好之后（`[IO.File]::WriteAllText` 写无 BOM 的纯文本；cmd 的 `echo > file` 或 PS 的 `Set-Content` 都可能带 BOM/编码问题）：

| 深度（c1） | tok/round | accept | c1 最好 |
|---|---|---|---|
| 1:8 | 3.87 | 0.605 | 273.2 |
| 1:6 | 3.87 | 0.603 | 269.1 |
| 1:4 | 2.91 | 0.552 | 107.6（重压下，仅作参考） |

**1:6 的 tok/round 与 1:8 完全相同** —— 说明第 7、8 个草稿几乎从不被接受，加深只加成本。
1:4 则明显掉 token。所以 1:8 是正确的（与上一轮结论一致，但现在有「为什么」）。
**纪律**：改环境变量后必须核对日志头里的配置回显；不要相信「文件写了就等于生效」。

**（b）卡是共享的：`llama-server` 会间歇吃掉 40–85% 的 GPU**

本轮的测量窗口里有另一个 `llama-server.exe`（LM Studio）在跑，`nvidia-smi pmon` 显示它稳定占
40–85% SM、25–45% 显存带宽。同样的代码，c1 在它活跃时 120–160 tok/s、静默时 248–273 tok/s。
因此本轮所有性能数字都标了「静默期」，并且：
- 用 `nvidia-smi pmon -c N` 记录外部占用，只在外部 SM ≤ 5% 时取数；
- 用 best-of-3 / 取最小 round_ms，而不是单次；
- `final_accept.py` 就是这么写的（等静默 → 逐位比对 → 五点 → soak，全部落 `results\final.log`）。

**给下一个接手的人**：动手前先 `nvidia-smi pmon -c 3`。这张卡不是独占的，不看清这一点会把别人的负载
当成自己的回归。

## 8. 最终验收（2026-11-17 10:12–10:17，外部队列停止后，静默 GPU）

```
$ final_accept.py                     # 等 nvidia-smi pmon 外部占用 <=5% 再测，全部落 results\final.log
10:12:13 gpu quiet (foreign sm 5.0%)
10:12:16 greedy 200 tok: 200 tokens, md5 ad4d58f48441deeace02967e2fe12c88   <- 与基线一致
10:12:16 IDENTICAL
10:12:24 summary: c1 276.0 c2 409.7 c4 619.7 c8 1052.5 c16 1668.9
10:16:58 soak done: 99 rounds, first5 594 tok/s, last5 686 tok/s, min 383, max 928
```

- **逐位一致**：同一 prompt / temp 0 / seed 1234 / 200 token，md5 `ad4d58f4…` 与改动前完全相同（本轮共验证 3 次）。
- **soak**：4 分钟、99 轮混合负载（并发 8、200–500 token），**无失败、≤ 无衰减**（first5 594 → last5 686 tok/s）。
- **五点曲线**（三份静默期采样，取最好；单点噪声 ±5%，受长跑后的卡温/功耗墙影响）：

| 点 | 采样 05:04 | 10:12 | 10:20 | 报告值（best） | 基线（会话开始） |
|---|---|---|---|---|---|
| c1 | 248 | **276** | 256 | **276**（best-of-3 267.8–273.2） | 245–256.7 |
| c2 | **441** | 410 | 399 | **441** | 384 |
| c4 | **656** | 620 | 619 | **656** | 624 |
| c8 | **1086** | 1053 | 1044 | **1086** | 1055 |
| c16 | **1699** | 1669 | 1559 | **1699** | 1709 |

注：c2/c4/c8 的三次采样里 05:04（刚重启、卡温低）最高，10:12/10:20 在连续跑了几十分钟高并发后偏低，
差 3–7%，与功耗/温度墙和单样本口径一致；c16 在 1559–1699 之间摆动（已在带宽地板附近）。
诚实口径：**c1 +8%（best-of-3 257→273），c2 +8~15%，c4/c8 +0~5%，c16 不变（已在地板）**。

## 9. 收尾状态

- 服务：`run-flashnext.bat` 标准启动（无 trace/profiler 环境变量），`/health` 200。
- 代码：所有改动在 `B:\hibana-win`（venv 内 `flashinfer/prefill.py` 仅加了 `_pmark` 空钩子），
  备份 `mtp_graphs.py.pre-planprof` / `*.plan-cache`（5 个文件）/ `prefill.py.pre-hibana`。
- 回退：`HIBANA_PLAN_CACHE=0` 关掉 schedule 缓存（行为等价旧路径）；整文件回退用上面的备份。
- 本目录：新镜像 5 个文件 + `_planprof.py` + `patch_fi.py` + `refcheck.py` / `sweep.py` / `drive_c1.py` /
  `final_accept.py` / `restart.ps1` / `run-flashnext-tune.bat` / `flashinfer_prefill.py.patched`。
- 遗留（未做，按性价比排序）：① 融合 `_inject_norm`+`_quant_small`+`_budget_select`（同时压 device 与图节点数，
  verify 图 ~500 节点在 WDDM 下光启动就 0.6–0.85 ms/轮）；② 把 accept 图并进 verify 图（省一次大图启动）；
  ③ 若要碰 455，必须动 MoE 之外那 48% 的 device 工作。

## 10. 单流三题面（bench_final）—— 为什么它没涨，这很重要

10:21 实测（temp 0，best of 3）：

| 题面 | 本轮 | 会话开始 | 差 |
|---|---|---|---|
| en_prose（673 tok） | 250.0 | 249.9 | 0% |
| zh_prose（522 tok） | 192.8 | 192.6 | 0% |
| code（700 tok） | 434.5 | 451.6 | −4%（噪声） |

而同一台机器、同一份代码，`sweep.py` 的 c1 点从 245–257 涨到 248–276（+8%）。**两者不矛盾**：

- 本轮的收益是**每轮固定 ~1 ms 的主机侧开销**（plan 缓存 + 页表原地增长）。短上下文（200–400 token、
  页数 200–600）时一轮 12–14 ms，1 ms 占 7–8%；长生成（600–1000 token 上下文）时 split-KV 的 chunk 数、
  attention 与 MoE 的 device 时间都变大，一轮 16–20 ms，那 1 ms 只剩 5%，被摊薄。
- `bench_final.py` 的并发段用的是 **temp 0.3（采样档）**：c2 292 / c4 496 / c8 744 / c16 1142 tok/s，
  比贪心档（441/656/1086/1699）低 35–45%，因为采样档每个草稿步都要回主机做 sampling。
  **两个口径不能混着比**（上一轮报告里的 c16 1709 是贪心口径）。

结论（诚实版）：**这轮的主机侧优化在「短上下文 + 低并发」口径下 +8%，在长生成单流上基本持平；
c16 仍在地板。要把长生成单流也推上去，只能动 device 侧（第 9 节的三条）。**

## 11. round 34：把「PLE 批量取行」在**新代码**下重做一次 A/B（结论仍然是否定）

上一轮用 `HIBANA_PLE_BATCHED=1` 测出「批量更差」。本轮在 plan 缓存+页表原地增长之后重测一次
（同一份 bat、同一份深度表，只切 `results\ple_batched.txt`）：

| 手臂 | c1 best-of-3 | c8 | `fill` ms/轮 |
|---|---|---|---|
| 流式（默认） | 187.6 | 988.8 | 1.72 |
| 批量（`HIBANA_PLE_BATCHED=1`） | 194.6 | 1015.5 | 1.60 |

差 2–4%，**在测量噪声内**（那一小时外部占用回来了，c1 同一份代码在 187–268 之间摆动）。
`fill` 的差只有 0.12 ms/轮，与「9 列里大部分时间是在 `events[j].synchronize()` 等 draft 链」的结论一致。
**保持默认（流式），这条路径关闭。**

另外记一个坑：引擎自带的 `HIBANA_PROFILE_STEPS`（torch profiler）在本机会报
`ERROR: External init callback must run in same thread as registerClient (… != …)`（kineto 在引擎线程
启动，而 flashinfer 的 tvm_ffi 模块是在主线程注册的），窗口拿不到数据。要 device 分解就只用
**上一轮那份 c1 窗口**（moe 52.1 / gemm 19.6 / other 19.5 / norm 4.3 / ple 1.9 / gdn 1.6 / attn 1.0），
别指望在引擎里开 profiler。

## 12. round 35：并发档深度表复核（本轮唯一还没被复核过的旋钮）

上一轮的深度表 `1:8,2:6,4:3,8:2,16:4` 是在 PLE 取行修好之后调的。本轮把它在
「plan 缓存 + 页表原地增长」之后重测（每臂都重启、单样本，`sweep.py 1,8,16`，外部占用中等）：

| 深度表（c8, c16） | c1 | c8 | c16 |
|---|---|---|---|
| **8:2, 16:4（现行）** | 225.9 | **1045.8** | 1594.6 |
| 8:3, 16:5 | 230.9 | 962.6 | **1683.7** |
| 8:3, 16:6 | 239.6 | 1013.4 | 1681.0 |

- **c8：深度 2 最好**（1046 vs 1013/963）—— 行数一涨，verify 的 MoE 专家流量按行数走，多一行不划算；
- **c16：深度 5/6 的两次采样（1684/1681）比本窗口的 4（1595）高 5%，但静默窗口里 16:4 曾测到 1699**
  —— 结论不明，且 c16 已经贴着估算上限，**保持 16:4 不动**。
- 顺带确认：`tok/round` 会随深度单调涨（c8 20.7→23.2→22.6，c16 48.5→55.0→52.8），但 round_ms 涨得更快。

**结论：深度表维持 `1:8,2:6,4:3,8:2,16:4`。** 到这里，配置层面的可调项（kv 格式、verify 状态、spec 深度、
split-K、PLE 取行/批量、max-batch、MoE 预算、plan 缓存）已经全部被本轮或前几轮扫过一遍，
剩下的差距只在 `kernels/csrc` 与 `kernels/flashnext_fused.py` 的内核层（第 9 节三条）。

## 13. round 36：单流现在**已经被 device 吃满**——把 455 这个目标算清楚

拿上一轮那份 c1 device 窗口（30 个 decode step、821.9 ms，采样时卡被外部占用），按
「profile 窗口的单步 27.4 ms ÷ 实际单步 13.6 ms」= 2.01 归一化，得到 c1 每步的真实 device 预算：

| 类别 | 归一化 ms/步 | 占比 | 里面是什么 |
|---|---|---|---|
| MoE（w3 3-bit） | **7.10** | 52% | 9 行 × topk 8 ≈ 63 个不同专家 × 180 MB ≈ 11.4 GB @1.6 TB/s |
| gemm | 2.67 | 20% | flashinfer FA2 的 `sm89_xmma` 199 次/步 = 1.20；cutlass bf16 trunk = 0.92；其余投影 |
| other（小内核堆） | 2.65 | 19% | quant_small 0.50、direct_copy 0.40、moe prep 0.32、inject_norm 0.31、layer_norm 0.23、budget_select 0.20、silu 0.17、fp8_gemv 0.18、gemvx 0.16… |
| norm | 0.58 | 4% | |
| ple | 0.27 | 2% | |
| gdn | 0.22 | 2% | |
| attention 本体 | 0.13 | 1% | |
| **合计** | **13.62** | | **与实际单步 13.6 ms 吻合** |

关键结论（这条比任何优化都重要）：

1. **c1 的主机侧已经不在关键路径上了**：device 13.62 ≈ round 13.6。上一轮把 plan/页表/index 拷贝从
   关键路径上拿掉之后，主机工作已被完全遮住 —— **c1 再想快只能动 device**。
2. **MoE 7.10 ms 是带宽地板**（63 个专家矩阵、每个只服务约 1.14 行，没有行间复用；换 GEMM 也救不了）。
   把**全部非 MoE 都压到零**，单步也还有 7.1 ms ⇒ 3.87 tok/round ÷ 7.1 ms = **545 tok/s** 的上限，
   但那是「所有投影、量化、norm、拷贝都不耗时」的假想。
3. 现实一点：小内核堆 2.65 ms 里能通过融合/合并消掉的乐观估计是一半（≈1.3 ms），加上 attention GEMM 的
   合并（199 次/步 → 更少的 tile）省 ~0.4 ms，单步 ≈ 11.9 ms ⇒ **325 tok/s**；再做激进融合（每层 2 个内核）
   才可能到 **~350**。**455 需要的是「非 MoE 工作量减少 78%」，那不是调参，是换引擎结构。**
4. 所以 c1 的达成率 276/455 = 61% 里，那 455 本身是个偏乐观的估算；按「MoE 地板 + 现实可消的非 MoE」算，
   这张卡的这台引擎的**现实上限约 350–380 tok/s**。下一任接手的人请先看这张表，再决定要不要投内核工时。

（同一张表解释了为什么 c2–c8 也只有 63–71%：行数一多，MoE 的专家流量按行数线性涨，
而 tok/round 只按接受率涨 —— 这是投机解码在这张带宽型卡上的结构性代价。）

## 14. round 37：c8 的主机侧火焰（和 c1 完全不同的一幅图）

c1 上主机已经被 device 完全遮住（§13），但**并发一上来，主机又回到了关键路径上**。
py-spy 采样引擎线程（8 并发、28 s、约 1356 轮、953 tok/s）：

| 阶段 | 占引擎线程 | 每轮 |
|---|---|---|
| `WAIT under AcceptGraphs.round`（verify+accept 在跑） | 51.7% | **9.6 ms** |
| `Scheduler._prefill_step`（并发 8 时 8 条 300-token 题面在争前填） | 15.7% | — |
| `FlashNextExtras.fill`（PLE 行：3 列 × 8 请求，**这次不是在等，是真主机工作**） | 10.7% | **1.98 ms** |
| `AcceptGraphs.round` 主机段（`packet.tolist()` + 每请求 10 来项统计 + `push`） | 6.7% | 1.24 ms |
| `VerifyGraphs.run`（plan 0.07 + 拷贝 + 图启动） | 4.4% | 0.81 ms |
| `DraftGraphs.run`（深度 2 ⇒ 2 步） | 3.5% | 0.66 ms |
| `SpecDecoder.prepare`（页分配 + verify 输入拼装） | 3.5% | 0.64 ms |
| `MtpProposer.propose` | 2.6% | 0.49 ms |
| 图启动（`CUDAGraph.replay` 自身） | 4.3% | 0.80 ms |

**c8 一轮 19.9 ms = device 等待 9.6 + 主机 ≈ 6.6 + 前填摊入**。也就是说 c8 要往 1570 冲，
主机侧**还有 ~33% 的可压缩空间**（fill 2.0、accept 1.24、verify prep 0.81、prepare 0.64、启动 0.80）。
这是下一轮最值得做的地方：把 `fill` 改成一次批量（b≥4 时）、把 accept 的每请求 Python 循环压平、
把 verify 的 positions/tokens 拼装改成预建缓冲 —— 都是几十行、可逐位验收的改动，不需要动内核。

顺带修掉一个我自己引进来的开销：`_planprof.want()` 原来每 25 个草稿步 `os.path.exists` 两次，
在这台机器的过滤驱动下等于 **~0.13 ms/轮**（py-spy 里能看见 `want_keycheck` 0.17 s）。改成每 200 步一次。

## 15. round 38：**批量 PLE 行**在 b≥4 时真的赢了（+6%，c4/c8）

上一轮（round 34）A/B 出的「批量不划算」是在旧的批量实现上：那条路径用 **torch 在主机上做哈希**
（`rows_for` → `ngram_ids`，每请求十几次小算子）。这轮把它换成**numpy 哈希 + 一次 gather + 一次 pinned 上传**
（`FlashNextExtras._batch_verify_rows`，复用现成的 `PleLayer.column_ids`，与 `ngram_ids` 同一套 wrapping 算术），
并且只在**验证块够宽**时启用（`len(reqs) * (depth + 1) >= 16`），c1/c2 仍走原来的逐列流式。

**哈希等价性先做了单元验证**（`ple_hash_check.py`，200 组随机 token 序列、k=1..5、含 EOS 改写）：
`200/200 blocks identical` —— numpy 路径与 torch 路径给出的**表行 id 逐位相同**，
`np.stack(cols, 1).reshape(-1, heads)` 的排布也与 `rows_for` 的 (请求 × 列) 分块一致。

实测（三份采样）：

| 点 | 改前（三份中位） | 改后（三份中位） | 变化 |
|---|---|---|---|
| c1 | 265 | 264 | 0（仍走流式） |
| c2 | 415 | 415 | 0（14 行 < 16，仍走流式；阈值降到 12 反而掉到 405–412，已回退） |
| c4 | 639 | **676**（720/653/676） | **+6%** |
| c8 | 1053 | **1119**（1119/1079/1128） | **+6%** |
| c16 | 1670 | 1680 | +1% |

逐位一致：c1 `IDENTICAL`（流式路径未动）；c4+ 走批量路径的**行内容**由上面的单元验证保证，
并且 c8 的 `accept_rate 0.621 / tok/round 17.7` 与改前同量级（若表行取错，接受率会塌到 ~0）。

所以 §14 那条判断成立：**并发档的主机侧还有肉，而且比动内核便宜**。下一个可做的是
`AcceptGraphs.round` 的每请求 Python 循环（1.24 ms/轮 @c8）和 `VerifyGraphs` 的
`positions/tokens` 拼装（0.81 ms/轮 @c8）。

### round 38 收尾验收

- soak（新代码，c8 混合负载）：**104 轮 / 4 分钟无失败**，first5 690 → last5 784 tok/s（不衰减，min 390 / max 914）。
- 逐位一致：c1 `IDENTICAL`（流式路径未改）。
- 表行等价性：`ple_hash_check.py` 200/200（numpy 批量哈希 == torch `ngram_ids`，含 EOS 改写与排布）。
- 五点（本轮多份采样的最好/中位）：
  c1 **257–270**、c2 **407–424**、c4 **639–720（中位 676，+6%）**、c8 **1077–1128（中位 1119，+6%）**、c16 **1633–1712**。
- 状态：标准 `run-flashnext.bat` 启动，`/health` 200；本轮改动集中在
  `model/flashnext/extras.py`（`_batch_verify_rows` + `BATCH_ROWS=16` 阈值），已镜像并保留
  `.pre-*` 备份（107 上 `extras.py.plan-cache` 同目录另有旧版命名备份）。

- 回退开关：`HIBANA_PLE_BATCH_ROWS=0` → PLE 行永远走逐列流式（即本轮改动前的行为）。
  107 上另有 `extras.py.batch-ple`（本轮新版快照）。

### round 38 的修正：+6% 是漂移，**交错 A/B 的诚实结论是 +2~3%**

同一份代码在这台机器上 15–30 分钟内会漂 3–4%（长跑后的温度/功耗墙 + 外部 co-tenant），
所以我用**交错 A/B**（两次重启、各 3 份 c8 采样、背靠背）重测了批量 PLE：

| 手臂 | c8 三份采样 | 中位 |
|---|---|---|
| `HIBANA_PLE_BATCH_ROWS=0`（逐列流式） | 1017.1 / 1051.7 / 1079.8 | 1052 |
| `HIBANA_PLE_BATCH_ROWS=16`（批量） | 1028.4 / 1074.6 / 1114.8 | **1075（+2.2%）** |

**正确记录：c4/c8 +2~3%**（不是先前写的 +6%，那是拿不同时间窗的采样对比出来的）。
机制没问题（4 请求×4 行、8 请求×3 行时，少了 3–4 次 event wait + tolist + 多次 gather），
但收益的绝对值比第一版测量小。`HIBANA_PLE_BATCH_ROWS` 保留为开关（0 = 关）。

**方法论补充（写进 LESSONS 级别的教训）**：这台机器上的 A/B **必须交错**——
「改前 3 份、改后 3 份、中间隔十几分钟」会把 3–4% 的漂移算成收益。

另外记一笔负结果：把 `step_pages` 的 `torch.cat` 也改成原地追加（把最后两处 O(context) 拷贝都去掉）
**没有可测收益**（c8 交错前后都在 1075 附近），已回退成「只有 `_retained_pages` 用 page_buf」的简单版本。

## 16. round 40：本会话所有改动的**累计收益**（交错 A/B，两臂轮流各两次）

「全开」= schedule 缓存 + 原地页表 + 批量 PLE 行 + 去掉 plan/indptr 的主机拷贝；「全关」= 用环境变量回到
会话开始时的行为（`HIBANA_PLAN_CACHE=0 HIBANA_PLE_BATCH_ROWS=0`，另有 `.pre-*` 备份可完全等同上游）。
两臂背靠背交替、各两次：

| 点 | 全开 | 全关 | **累计收益** |
|---|---|---|---|
| c1 | 239.2 / 240.4 | 226.6 / 227.4 | **+5.6%** |
| c2 | 407.2 / 405.3 | 383.4 / 367.3 | **+8.3%** |
| c4 | 624.5 / 632.6 | 594.3 / 595.5 | **+5.6%** |
| c8 | 1061.9 / 1038.1 | 977.3 / 1005.1 | **+6.0%** |
| c16 | 1583.0（另一份 1156 受外部占用污染） | 1523.0 / 1550.7 | **~+3%（噪声大）** |

注意绝对值比 11:15 那批低 5–8%：这张卡长跑后会持续变慢（温度/功耗墙），
所以这个 +6% 是**同窗口内交错量出来的真值**，而「改前 245、改后 276」那种跨窗口对比会高估。
会话开始时（03:40）的基线 `245/384/624/1055/1709` 与「全关」两臂（227/375/595/991/1537）
之间的差也有同样的漂移成分 —— 换窗口比较时请以「全关」臂为准。

**一句话总结这一整天**：主机侧能拿的都拿了（plan schedule 缓存、页表原地增长、批量 PLE、
去掉每步的 index 拷贝 —— 合计 c1–c8 +5~8%，c16 已在地板），c1 现在 100% device-bound，
再往上只能动内核；而 device 预算里 MoE 占 52% 是带宽地板（§13），455 需要把非 MoE 压掉 78%，
现实上限约 350–380 tok/s。

- 遗留的测量工具（默认静默）：`HIBANA_PHASE_STATS=1` 时引擎每 100 轮打印
  `accept cost: wait/packet/page/conf/loop/publish`；`extras.py` 同开关打印 `fill cost (qsa/ple)`。
  这两组数就是 §14 那一轮火焰图的引擎内版本，改内核前先用它定位，不要用 py-spy 猜。

### round 41 收尾：稳定性复测

- soak（12:03–12:07，c8 混合负载、temp 0.3）：**103 轮 / 4 分钟无失败**，first5 744 → last5 700 tok/s，
  min 464 / max 858（末端下滑与同期的卡漂移吻合，不是泄漏）。
- 状态：`/health` 200、GPU 5% 空闲、标准 `run-flashnext.bat`、c1 逐位一致。
- 交付：HANDOFF §13–§16 + LESSONS §六（交错 A/B、引擎内计时 vs py-spy、profiler 归一化、
  主机饱和判据、WDDM 图节点成本、PLE 方向的终审结论）。

## 17. round 42（关机前最后一批测量）：采样档 vs 贪心档，以及「质量不走」这条硬约束

用户明确要求**不能靠放宽验收换吞吐**（Hibana 仓库自己的实测：放宽到全收能到 257 tok/s，但 GPOA-D 长推理
准确率明显掉分；4:2.5 / 5:3.0 这种中等放宽也只多 ~4 tok/s、中文散文可读性已经受损）。
所以本会话的**所有改动都限定在「逐位一致」这一类**（c1 md5 `ad4d58f4…` 未变、PLE 表行 200/200 相同），
启动脚本里没有任何 `accept/relax` 环境变量 —— 收益全部来自「同一份数学换更少的握手/更少的拷贝」。

关机前最后一批长窗口测量（`drive_ab.py`，各 20 s，`max_tokens 3000`）：

| 点 | 贪心 tok/s | 采样 0.3 tok/s | 采样/贪心 | 验收率（贪/采） |
|---|---|---|---|---|
| c1 | 149.8（round 30.6 ms） | 149.8（round 17.0 ms） | — | 0.638 / 0.508 |
| c4 | 599.4 | 509.6 | 85% | 0.705 / 0.509 |
| c8 | 1112.9 | 812.8 | 73% | 0.793 / 0.619 |
| c16 | 1758.7 | 1303.7 | 74% | 0.621 / 0.426 |

两点必须记住：

1. **c1 的两个数字不可直接比**：贪心每轮多吐 1.8 个 token，20 s 里上下文跑到 ~1500，而采样只到 ~900；
   单流的 round 时间随上下文从 15.9 ms（~300 tok）涨到 30.6 ms（~1500 tok）。**长生成的单流速度（~150 tok/s）
   比短生成（~250）低得多**，谈「单流天花板」必须钉住上下文长度口径。
2. **采样档的下降不是质量下降**：accept 0.79 → 0.62 是拒绝采样的数学结果，输出分布是对的。
   想用它换吞吐就得放宽验收（用户已否掉）。所以在「兼顾质量」下，天花板就是贪心档的天花板；
   采样档要提上来，唯一无质量代价的路是**更准的草稿头**（权重/训练层面），不是引擎调参。

## 18. 停机状态（2026-11-17 12:25）

107 已关机（`ping` 无响应、`ssh` No route to host），关机前最后一次检查（12:09）服务 `/health` 200、
GPU 6%、标准 `run-flashnext.bat`、c1 逐位一致。所有改动、备份、镜像与文档都在本地
`~/文档/deepseek-harness/default-workspace/`（`hibana-win-port/final-20261117/` 与项目报告），
107 上的备份为 `*.plan-cache`×5、`extras.py.batch-ple`、`prefill.py.pre-hibana`、
`mtp_graphs.py.pre-planprof`；运行期开关 `HIBANA_PLAN_CACHE=0` / `HIBANA_PLE_BATCH_ROWS=0` /
`HIBANA_PHASE_STATS=1`。

## 19. 口径变更：性能验收改用采样档（2026-11-17 起）

用户明确指示「不走贪心解码来测」，所以：

- **性能**：`accept_sampled.py`（temperature 0.3、512 token/流、固定 seed、每点 2 次取中位、计数器差分）
  是新的吞吐口径；`results/ref_sampled_<tag>.json` 存对照点。
  关键：**固定 token 预算**而不是固定时长 —— 快的手臂会在同样 20 s 内跑进更长的上下文（c1 round 从 15.9 → 30.6 ms），
  自我减速，两臂就不可比了。
- **质量**：仍是贪心逐位一致（`refcheck.py cmp base`，temperature 0 / seed 1234 / 200 token），
  吞吐只有在 token 一个没变时才上报。
- 开机清单见 `next-window.md`（0 口径 → 1 确认 health → 2 拷脚本 → 3 建采样基线 → 4 内核工作 → 5 已知坑）。

另外把这轮新增的两个工具也归档了：`drive_ab.py`（长窗口贪心/采样对比，带上下文增长的自减速提醒）、
`temp_cmp.py`（同一题面下贪心 vs 采样的验收率差异）；`engine_spec_graphs.py` 的 `verify cost:`
也一并挂到 `HIBANA_PHASE_STATS=1` 后面（默认静默）。

## 20. 修正 §9 第 ① 条：「融合 inject_norm+quant_small」不可行，改成「并行化 quant_fp8_small」

离线读透了 `kernels/flashnext_fused.py`（见 `kernel-fusion-spec.md`）：

- `inject_combine_norm()` **已经把 inject + combine + RMSNorm 融成一个 kernel**（profile 里的 94 次/步就是它）。
- `quant_fp8_small()` 的 FP8 scale 是**整张量全局 max**，所以「求 max」和「写 fp8」之间必须有一次物化 + 跨 program 归约，
  **不引入 grid 内同步就无法再融**。
- 真正的浪费在于 `_quant_small_kernel[(1,)]`：网格是**单个 CTA**，两遍扫 46K 个元素，单 SM 带宽 ⇒ ~10 µs/次、
  每步 48–100 次 = 0.5 ms/步。把它改成「多 CTA 求 block max → 两级归约 → 多 CTA 写 fp8」可到 0.2~0.25 ms/步（c1 +2%），
  且 `div_rn(x, scale)` 对同一个 scale 逐元素相同 ⇒ **输出 fp8 字节完全一致**。

另附离线回归测试 `ple_hash_test.py`：在**没有 GPU、没有权重**的机器上验证 PLE 的两条哈希路径
（`ngram_ids` vs `column_ids`）逐位一致（200 块 × k=1..5 × 含 EOS），本机（102, torch 2.15 CPU）已跑通 `200/200`。

## 21. round 43（机器重新开机后）：**采样档口径下的深度表复核 → c8 深度 2 → 3，+3.8%**

按新口径（采样、不走贪心）重建基线，并复核深度表——**贪心的最优深度在采样档未必最优**：

`accept_sampled.py base-sampled`（12:20，温度 0.3、512 token/流、每点 2 次取中位）：

| 点 | 采样 tok/s | tok/round | accept | round_ms |
|---|---|---|---|---|
| c1 | 188.7 | 2.69 | 0.505 | 14.3 |
| c2 | 328.1 | 5.38 | 0.488 | 16.4 |
| c4 | 519.0 | 9.64 | 0.518 | 18.6 |
| c8 | 838.0 | 17.77 | 0.637 | 21.3 |
| c16 | 1284.7 | 39.50 | 0.459 | 30.8 |

深度 A/B（**交错**，每臂两次，采样档）：

| 深度（c8,c16） | c8 | c16 |
|---|---|---|
| 8:2 / 16:4（原） | 826.2 · 774.6 → **800** | 1384.7 · 1312.5 → **1349** |
| 8:3 / 16:5 | 864.9 · 796.9 → **831（+3.8%）** | 1332.2 · 1266.8 → **1300（−3.6%）** |

结论：**c8 用深度 3**（采样下每个草稿的接受率低，多一个草稿的期望收益更高），**c16 保持 4**（多两行的
专家流量超过收益）。已把 `run-flashnext.bat` 的 `HIBANA_SPEC_DEPTHS` 改成 `1:8,2:6,4:3,8:3,16:4`
（备份 `run-flashnext.bat.depth8-2`），改后整套采样五点：185.2 / 347.6 / 558.9 / **908.1** / 1434.8，质量闸门 IDENTICAL。

**重要**：这一整批（908.1 / 1434.8）比 12:20 的基线高 5–12%，但其中 c16 的深度没变——说明**窗口本身快了 5–10%**
（刚开机、卡更凉）。所以只有交错出来的 +3.8%（c8）算收益，跨窗口的绝对值一律不可比。

**顺带一个负结论**（省掉下一轮的白工）：`quant_fp8_small` 的并行化**不划算**——它的 scale 是全局 max，
并行至少要 3 个 launch（求 parts / 归约 / 写 fp8），而单 CTA 版本只要 1 个 launch、~5 µs；
46K 元素上并行省下的 4 µs 抵不过多出来的 6 µs 启动成本。详见 `kernel-fusion-spec.md` §3 的补充。

## 22. round 44：**验收规则是精确的**（不是放宽），以及 c2/c4 深度复核收尾

用户那张图提醒的两条路里，第 2 条（放宽验收）在此前是「有没有被悄悄打开」的问题。核实结果：

- `hibana/api/validation.py`：`ACCEPT_DEFAULT = (1, 0.0)`（K=1 即**精确**）、`ACCEPT_SOFT_DEFAULT = 0.0`；
- 运行中的引擎命令行是 `python -m hibana serve flashnext --no-fetch --port 30004 --max-batch 16 --context-length 32768`
  —— **没有** `--accept-default` / `--accept-soft`；
- `accept_graphs.round` 里 `any_relaxed = not sampled and ...`：**采样档根本不走 relaxed 规则**；
- 之前 `spec_stats` 里看到的 `relaxed_accepts` 在 K=1、margin=0 时只会命中 argmax 本身，等价精确。

⇒ **本会话所有吞吐数字（贪心与采样）都是在精确验收下测得的，没有任何一处用质量换速度。**

采样档深度复核（交错，每臂两次）收尾：

| 深度 | c2 | c4 |
|---|---|---|
| 2:6 / 4:3（现） | 336.5 · 346.5 → **341** | 539.9 · 525.0 → **532** |
| 2:8 | 325.4 · 323.8 → **325（−5%）** | — |
| 4:5 | — | 522.2 · 541.6 → **532（0%）** |

⇒ c2 保持 6、c4 保持 3（4:5 无收益、还多 1 ms/轮）；只有 c8 从 2 调到 3（上轮 +3.8%）。
深度表最终为 `1:8,2:6,4:3,8:3,16:4`，与启动脚本一致。

**另一条探查结论**：采样档接受率低于贪心不是配置 bug。`draft_rows`/`draft_laws` 用的是**每个请求自己的
temperature/top_k/top_p**（`params` 里带 temperature，laws 就是实际抽样所用的 q），草稿与目标同温；
接受率下降来自草稿头的分布准确度（32B 草稿要在 16 个候选上近似 27B 目标），以及 DFlash2 的「候选支撑只有 16 个」
这一设计上限 —— 属于权重层面，不是引擎参数。改引擎救不了，只有更准的草稿头能救。

## 23. round 45：贪心开回来后的天花板口径，以及采样掉速的完整分解

用户问「去掉贪心掉速这么多吗、这是英文还是中文散文、要不把贪心开回来追天花板」。逐条回答：

**(1) 题面**：`accept_sampled.py` 用的是**英文技术散文**
（"Write a detailed technical explanation of paged attention, its page table, prefix caching and eviction…"）。
顺带对照（贪心、`bench_final.py`）：**en 250.0 / zh 192.8 / code 434.5 tok/s** —— 中文散文**按 token 算慢约 23%**
（信息密度更高、同样内容需要更少的 token，实际阅读速度相当），代码最快。

**(2) 掉速不是实现问题，全部在「每轮吐的 token 数」上**：

| 点 | 贪心 round_ms | 采样 round_ms | 贪心 tok/round | 采样 tok/round | 采样/贪心 |
|---|---|---|---|---|---|
| c1 | 15.3 | 14.3 | 3.98 | 2.69 | 0.68 |
| c2 | 15.9 | 16.4 | 6.41 | 5.38 | 0.84 |
| c4 | 19.2 | 18.3 | 12.21 | 9.64 | 0.79 |
| c8 | 22.0 | 22.7 | 23.18 | 17.77 | 0.77 |
| c16 | 30.6 | 30.8 | 48.46 | 39.50 | 0.82 |

**轮时间几乎不变**（说明没有隐藏开销），少的是接受率：温度 0.3 下目标分布更尖，而 32B 草稿头只能在
16 个候选上近似 27B 目标，精确拒绝采样必然少收。这是数学，不是可以调掉的东西。

**(3) 贪心开回来的天花板（当前配置 `1:8,2:6,4:3,8:3,16:4`）**：
**c1 260.6 / c2 406.0 / c4 638.0 / c8 1060.8 / c16 1593.2 tok/s**。
注意 **c8 的深度 3 在贪心下也没损失**（1060.8 对深度 2 时的 1055.8，轮 22.0 对 19.7 ms 但 tok/round 23.18 对 20.69）
—— 所以深度表两种口径共用，不必为贪心回退。

达成率（对贪心目标 455/700/930/1570/1690）：**57% / 58% / 69% / 68% / 94%**。

**(4) soak（采样档、深度 8:3）**：108 轮 / 4 分钟无失败，first5 717 → last5 806 tok/s（不衰减）。

## 24. round 46：全量交错的最终收益（贪心口径，两臂各两次）

「全开」= 本会话全部改动（plan schedule 缓存 + 原地页表 + 批量 PLE + 深度 8:3）；
「全关」= `HIBANA_PLAN_CACHE=0 HIBANA_PLE_BATCH_ROWS=0` + 深度表回退到 `8:2`（等价于会话开始时）。

| 点 | 全开 | 全关 | 收益 |
|---|---|---|---|
| c1 | 231.0 / 216.3 | 216.1 / 218.5 | **+2.9%**（c1 不吃深度改动） |
| c2 | 405.2 / 397.7 | 372.8 / 371.8 | **+7.8%** |
| c4 | 634.7 / 594.2 | 571.4 / 567.8 | **+7.9%** |
| c8 | 1007.6 / 1022.8 | 975.1 / 974.7 | **+4.1%** |
| c16 | 1569.2 / 1597.4 | 1520.9 / 1505.9 | **+4.6%** |

和 round 40 那批（+5.6/+8.3/+5.6/+6.0/+3.0）同量级；**这台机器的窗口漂移就是 ±5%**，
所以对外只能说「本会话改动 = c2/c4 约 +8%、c8/c16 约 +4%、c1 约 +3%，全部逐位一致、零质量代价」。

这轮也顺手排除了两个方向：
- **`direct_copy` 微优化不值得**：每轮真正要拷的只有 ~8 个小张量（verify 的 tokens/positions/pages/slots、
  accept 的 y/meta/relax、fill 的 qsa/ple），合计 20–40 µs；profile 里那 0.40 ms/步的大头是**选择型变体**
  （长上下文）里 48 层 features 的逐层拷贝，短上下文根本不跑。合并成 foreach 只省 ~10 µs。
- **`_budget_select_kernel` 48 次/步不是解码路径的浪费**：那 48 次来自 **prefill**（每层选择一次），
  解码的 dense 变体不调用 `select_pages`。所以「dense 变体跳过选择」这条没有收益，别再查。

## 25. round 47：把「天花板为何到不了」算成一张表（结论段）

三个独立来源一致指向同一结论：**目标值 455/700/930/1570/1690 假设了几乎完美的专家共享，而这张卡在低并发下做不到。**

**(1) c1 的逐字节核算（三个数互相印证）**
- 专家权重：46.1 GiB / 256 = **184 MB / 专家**；
- verify 9 行 × topk 8 = 72 个槽位，去重后 **63 个不同专家**（小批下同一专家的行间复用很少）；
- 于是每步 MoE 流量 ≈ 63 × 184 MB = **11.6 GB**；
- 实测 MoE 内核 7.10 ms/步（归一化后）⇒ **1.63 TB/s**，正好是这张卡的实测带宽。
⇒ MoE 已经在带宽地板上，**没有任何余量**。c1 的 15.3 ms/轮 = MoE 7.1 + 其余 8.2，主机侧已被完全遮住。

**(2) 并发上去后没有变好，反而更吃带宽**
c8 一轮 22.0 ms、tok/round 23.18：8 请求 × 3 行 = 24 行 × topk 8 = 192 个槽位，
去重后约 150 个专家 × 184 MB ≈ 27 GB ⇒ 理论 17 ms，占 22 ms 的 **77%**。
所以 c8 的 68% 达成率不是主机或调度问题，是「每轮要为少量行读大量专家矩阵」的结构代价。

**(3) 目标值隐含的假设**
要让 c1 到 455 tok/s（8.5 ms/轮），在 3.98 tok/round 下意味着**非 MoE 全部压到 1.4 ms**；
要让 c8 到 1570（35 ms 内 48 tok/round），意味着 MoE 要按 **~1.9 GB/ms** 走，即每轮只能读约 60 个专家
（现在是 ~150）。两者都需要「更少的专家被读到」——也就是 **topk↓ / 更多专家共享（有损路由）**，
或者**更准的草稿头**（同样的 MoE 流量下吐更多 token）。这两条都不在引擎里。

**可以诚实地写进交接的三句话**：
1. 本会话把主机侧彻底压到非关键路径，累计 c2/c4 +8%、c8/c16 +4%、c1 +3%，全部逐位一致、零质量代价；
2. device 侧 52% 是 MoE 带宽地板、19% 是 GEMM、19% 是每层十几个小内核 —— 剩下的都在 1–2% 量级，
   而且这台机器的窗口漂移 ±5%，**再往上做的每一项都无法被本地测量可信地区分出来**；
3. 真正的量级增量只有两条路：更准的草稿头（权重/训练），或每层 2 个内核级别的结构性重写（换引擎）。

## 26. round 48：三点消融把「天花板」彻底钉死 —— 固定的权重流量 + 草稿接受率

用引擎自己的**逐轮步迹**（`HIBANA_STEP_TRACE=1`，无 profiler、无归一化）在 c1 做深度消融：

| 深度 d | 轮步迹 wall/gpu (ms) | tok/round | tok/s |
|---|---|---|---|
| 1（verify 2 行） | **12.6** | 1.85 | 146.9 |
| 4（verify 5 行） | 14.3 | ~2.7 | 192.0 |
| 8（verify 9 行） | **15.6** | 3.94 | 252.8 |

**结论 1：每一轮有一次约 11–12 ms 的「固定权重流量」**。d=1（只有 2 行！）就要 12.6 ms，
而每多一个「草稿步 + verify 行」只加约 0.4–0.5 ms。换算：11.2 ms × 1.6 TB/s ≈ **18 GB/轮**的固定读取
（trunk 权重 + lm_head 1.3 GB + indexer + MTP 头），这与「c1 无论怎么调都停在 250–260 tok/s」完全吻合。

**结论 2：并发档靠摊销这个固定成本**：
c1 3.94/15.6 = 253；c8 23.2/19.0 = 1220；c16 48.5/29.7 = 1633。
如果 MoE 不花时间，上限分别是 3.94/12.6 = 313、23.2/12.6 = 1840、48.5/12.6 = 3850。

**结论 3：目标值隐含的草稿接受率约为现在的两倍。** 反推 tok/round 需求：

| 点 | 目标 tok/s | 现轮时 (ms) | 需要的 tok/round | 每请求 | 需要链式接受率 | 现在 |
|---|---|---|---|---|---|---|
| c1 | 455 | 15.6 | 7.1 | 7.1（d=8） | **0.76** | 0.37 |
| c2 | 700 | 16 | 11.2 | 5.6（d=6） | **0.77** | 0.37 |
| c4 | 930 | 19 | 17.7 | 4.4（d=3） | 1.13（不可能） | 0.37 |
| c8 | 1570 | 19 | 29.8 | 3.7（d=3） | **0.90** | 0.40 |
| c16 | 1690 | 29.7 | 50.2 | 3.1（d=4） | 0.53 | **0.61 ✓** |

也就是说：**c16 的目标其实已经达到（按接受率算超额），c1/c2/c8 的目标需要 0.76–0.90 的接受率，
而现装的 MTP 草稿头只能给 0.37–0.61。** 引擎侧再怎么优化 device 时间也只能把分母压 10–20%，
补不上这个 2× 的分子差 —— 分子是草稿头的分布准确度（权重/训练）。

**所以：这台引擎在这张卡上的实际上限，按现装草稿头算，就是 c1 ~300 / c2 ~450 / c4 ~700 / c8 ~1300 / c16 ~1700 tok/s 量级；
目标表是把接受率当成 0.8–0.9 算出来的。** 这条写进交接，避免下一任把「达不到」当成引擎 bug 再挖一遍。

## 27. round 49：深度 8 是**硬上限**（设计约束），以及「接受率 → 吞吐」决策表

**(1) 深度上限**：试 `HIBANA_SPEC_DEPTHS=1:12,...` 时引擎在启动阶段直接拒绝：

```
engine/depth.py:44  ValueError: depths per batch bucket must be in [0, 8]
```

`parse_depths` 的上界就是草稿头的 K=**8**（MTP 头训练出的草稿长度）。所以「再加深草稿换吞吐」这条路
**不是调参问题，而是权重/训练约束**——加深必须换一个训练过更长的草稿头。折旧结论：深度 8 已是上限，
每轮 token 数只能靠**接受率的质量**提升。

**(2) 接受率的边际价值（用 §26 的三点消融标定）**：c1 的实测点
（d=1 → 12.6 ms / 1.85 tok、d=4 → 14.3 / 2.75、d=8 → 15.6 / 3.94）说明草稿的链式接受是位置相关的：
**第 1 个草稿接受率 ~0.85，第 2–8 个每个约 0.30**（合计 2.94）。边际收益：每加深一步约 +0.30 token / +0.43 ms。

按此模型反推 c1：

| 第 1 个草稿的接受率 | 第 2–8 个每个 | c1 tok/s（d=8） |
|---|---|---|
| 0.85（现在） | 0.30 | **253**（实测 252.8 ✓） |
| 0.90 | 0.40 | ~330 |
| 0.95 | 0.55 | ~420 |
| 1.00（完美草稿） | 0.76 | **453 ≈ 目标 455** |

⇒ **455 这个目标等价于「草稿头几乎完美」（第 2–8 个草稿的接受率要到 0.76）**，
而现装 MTP 头给的是 0.30。引擎侧把 device 时间压 20% 最多把 253 抬到 ~300，补不上这个缺口。

**(3) 顺带记一次事故与恢复**：深度 12 的试验让引擎在启动时退出（上述 ValueError），
已用标准 `run-flashnext.bat` 恢复，`/health` 200、c1 259.1（best-of-3）、逐位一致。
`results/depths.txt` 只被 tune 脚本读取，标准脚本里的深度是硬编码的，所以不受影响。

## 28. round 50：把「silu 融进量化」也测了 —— **更慢**（第三个负结果）

机制上很有吸引力：GEMM 路径现在是 `F.silu(x)` + `quant_fp8_small(x)` 两个 launch，
融合后省一个 launch + 一次 bf16 中间张量的往返。先确认了**逐位可行**（`silu_check.py`：三种 fp32 写法
`x*(1/(1+exp(-x)))`、`x/(1+exp(-x))`、`x*sigmoid(x)` 在 bf16 输出上与原实现 **1.000000 一致**），
再写了融合 kernel 做微基准（`probe_quant.py`，N = 9×17408，300 次平均）：

| 版本 | 每调用 | 输出 |
|---|---|---|
| 现行（silu + 两遍量化，2 launch） | **0.039 ms** | — |
| 融合（1 launch，两遍内各自算一次 silu） | 0.059 ms | **fp8 字节与 scale 完全相同** ✓ |

**融合慢 51%**：省下的 launch（~2–3 µs）抵不过「第二遍要重算一次 silu」（156K 个 exp）。

⇒ 现行分解（把 silu 物化成 bf16，再两遍量化读它）**已经是这条路径上的局部最优**。
至此三个候选微优化全部被实测否决：
1. `quant_fp8_small` 并行化 —— 多 2 个 launch，净亏（§3b）；
2. `step_pages` 的 cat 改原地追加 —— 无可测收益，已回退（§15）；
3. `silu` 融进量化 —— 慢 51%（本节）。

**所以「剩下的 1–2% 小内核」不是一个待办清单，而是一个已被测穿的局部最优。**
要继续只能换结构（每层 2 个内核的重写）或换草稿头（§27）。

## 29. round 51（阶段一开工）：一次实测把 c1 的账**改对了** —— 显卡已经用掉约 80%，剩 15–20%

前几轮我用「profile 窗口归一化」得到 MoE 7.1 ms/步，据此说「MoE 占 52%」。**这一轮用两个独立探针把它推翻了。**

**(1) MoE 到底占多少：用 `HIBANA_MOE_BUDGET` 做消融（它限制一次前向读的 distinct 专家数）**

| 配置 | c1 步迹 gpu |
|---|---|
| 自然（9 行 → 约 63 个不同专家） | **15.6 ms** |
| `HIBANA_MOE_BUDGET=8` | 14.6 ms |
| `HIBANA_MOE_BUDGET=2` | 14.3 ms |

把专家从 63 砍到 2 只省 **1.3 ms** ⇒ **MoE 总共只有约 2–3 ms/步，不是 7.1 ms。**

**(2) trunk 的 fp8 GEMM 在 M=9 时效率如何（独立探针，48 个不同权重循环，保证走 HBM 而不是 L2）**

`[17408×5120]` fp8 权重 × 48 层 = 4.3 GB：
- M=9：**3.28 ms → 1.31 TB/s**
- M=16：3.16 ms → 1.35 TB/s
- M=64：3.30 ms → 1.30 TB/s

⇒ **trunk 的 GEMM 在 9 行时就跑到了峰值的 82%**，不是「小批次低效」。
（注意：单个权重矩阵 89 MB 能进 126 MB 的 L2，所以**重复同一个矩阵的微基准会测出 3 TB/s 的假数**——
这是我第一次写错的地方，必须用 48 个不同权重循环才是 HBM 真实值。）

**(3) 由此得到 c1 的真实预算（15.6 ms）**

| 组件 | ms | 效率 |
|---|---|---|
| trunk 稠密 GEMM（约 7.2 GB） | 5.5 | 82% 峰值 |
| 草稿链 8 步（每步 1 行，约 1.84 GB） | 2.4 | **48%**（M=1 的先天限制） |
| MoE 专家（约 2.6 GB） | 2.0 | 约峰值 |
| accept + attention + indexer + 杂项 | 3–4 | ? |
| 逐 kernel/图节点开销（残差） | 2–3 | ? |
| **合计** | **15.6** | |

**结论（修正此前判断）**：c1 的 15.6 ms 里约 **80% 已经在带宽附近**，剩下的 15–20% 才是可争取的：
草稿链的 M=1 低效（~1 ms，但它是投机性的、质量中性）+ 逐 kernel 开销（~2–3 ms）。
⇒ **阶段一的现实目标是 c1 258 → 285–300（+10~16%），不是 2–3 倍；455 仍然只能靠阶段二的草稿头。**

**方法论教训（写下来）**：① profile 窗口里的 kernel 时间必须按「窗口单步 ÷ 实测单步」归一化，
但即便这样，**不同 kernel 类别的并发占用不同，归一化会系统性高估带宽型 kernel**；
② 单矩阵微基准会被 L2 骗，用多个不同权重循环；
③ 判断「是否榨干」只能用「实测字节 ÷ 实测时间」对比峰值带宽。

## 30. round 52（阶段一实测）：卡在带宽上的部分已到 78–85%，剩下的 15% 是「8 步串行草稿」的派发开销

**(1) 引擎自己的 fp8 GEMV 在 M=1 时跑到 1.25 TB/s = 峰值的 78%**

`probe_gemv.py`（调 `hibana.kernels.fp8_gemv_cuda.fp8_gemv_quant_cuda`，MTP 头的 q_proj 形状 `[12288×2560]`，
**16 份不同权重 = 504 MB，远超 126 MB L2**，保证走 HBM）：

```
fp8 GEMV [12288x2560] M=1: 0.0253 ms per call, 0.503 GB per pass
achieved 1.25 TB/s (peak 1.6)
```

所以草稿链的投影本身不是「M=1 低效」，它已经是 78%。

**(2) 那 8 步的 0.23 ms/步花在哪**

一步草稿要读的权重约 130 MB（q 31.5 + o 15.7 + fc/fc_h 26 + 专家约 39 + hc mixers ~30）
→ 按 1.25 TB/s 只需 **0.104 ms**；实测 ~0.23 ms
⇒ **每步约 0.12 ms 是「15–20 个小 kernel 的派发 + attention + 小 GEMM」**（8 步合计约 1.0 ms）。

**(3) 阶段一总账（c1，15.6 ms）**

| 组件 | ms | 效率 | 还能拿多少（逐位一致前提下） |
|---|---|---|---|
| trunk 稠密 GEMM（~7.2 GB） | 5.5 | 82% | ≤1.0 |
| 草稿链 8 步 | 2.4 | 权重读 78%，但**一半是派发开销** | ~1.0（需融合小 kernel） |
| MoE 专家 | 2.0 | 约峰值 | ~0 |
| accept/attention/indexer/杂项 | 3–4 | ? | ? |
| 逐节点残差 | 2–3 | ? | ~0.5 |

⇒ **阶段一的天花板约 c1 285–300（+10~16%），前提是完成「草稿步内 6 个 hyper-connection 小 GEMM + elementwise 的融合」，
而且这些融合必须保持每个 GEMM 内部的归约顺序不变才逐位一致。**

**结论**：显卡本身没有「2–3 倍」的空间——带宽型阶段已经 78–85%。剩下的 15% 是**小 kernel 派发**，
要靠 Triton/CUDA 的分组 GEMM 与 elementwise 融合去拿，工作量与风险都高于阶段二（草稿头蒸馏/语义集，上界 +66%）。

## 31. round 53：引擎侧把草稿头相关的开关全试了一遍 —— 全是负结果

阶段一测完之后，我把阶段二**不需要训练就能动的旋钮**全试了（每个 1 次重启 + `accept_sampled`）：

| 开关 | c1 采样 tok/s | c8 | c16 | 贪心 md5 | 结论 |
|---|---|---|---|---|---|
| 基线（默认 fp8 头，rescore 8） | 188.7 | 838.0 | 1284.7 | IDENTICAL | — |
| `HIBANA_MTP_SAMPLE=1`（按 law 抽样草稿） | 170.5 | 776.1 | 1207.8 | IDENTICAL | **更慢**（多出 lattice walk，接受率没变 0.52） |
| `HIBANA_DRAFT_HEAD=w3`（3-bit 头） | 158.4 | 793.7 | 1200.6 | **DIFFERENT ✗** | 既慢又不逐位 → 直接否 |
| `HIBANA_DRAFT_HEAD=w2` + rescore 16 | 178.9–238.4（两次差 33%！） | 793.6 | 1318.4 | IDENTICAL | 两次采样落在噪声里，c8 稳定 -5% → 否 |

**顺带一个重要发现（测量方法）**：`accept_sampled.py` 的 c1 是「512 token × 2 次取中位」，
同一份配置两次跑出 **178.9 / 238.4**（差 33%）。单突发短程的方差远大于我以为的 ±5%。
⇒ 以后 c1 的采样口径必须**至少 4–6 次取中位**，或者改用固定 30 s 窗口（但那样又会有上下文自减速问题，见 §19）。
这一条要写进测量脚本的注释里。

**结论**：
- 阶段一（榨干显卡）：带宽型阶段实测 78–82%，MoE 到顶，hyper-connection 已被作者融合，图按深度分别捕获（无行填充浪费）。
  剩下的 10–16% 在「替换 cutlass/flashinbert kernel 选择」上，**会改数值** → 与逐位一致的验收口径冲突。
- 阶段二（草稿头）：引擎侧的三个旋钮全部负结果。真正的提升只能来自**训练侧**（蒸馏 / 语义集 / 更好的头），
  这台机器上没有训练管线与数据，本会话无法执行。

## 32. round 54：把最后一块没量过的（indexer）也消融了 —— 0 ms，阶段一到此终结

indexer（qsa：每层 qk 投影 + 写 key + 压缩 key）是「accept/attention/indexer/杂项 3–4 ms」里我唯一没量过的部分。
用 `HIBANA_ABLATE_INDEXER=1` 让 `replay()` 直接 `return None`（跳过全部逐层 key 维护，只测时间）：

| 配置 | c1 步迹 gpu |
|---|---|
| 基线 | 15.6 ms |
| **跳过 indexer 全部工作** | **15.8 ms** |

⇒ 短上下文（dense 变体）下 indexer 的逐层开销**在噪声内为 0**。（它只在上下文越过预算、注意力切到选择型时才真正干活。）

**至此 c1 的 15.6 ms 已被逐块量完：**

| 组件 | ms | 状态 |
|---|---|---|
| trunk 稠密 GEMM（~7.2 GB） | 5.5 | 82% 峰值 |
| 草稿链 8 步 | 2.4 | 权重 78%，一半是 15–20 个小 kernel 的派发 |
| MoE 专家 | 2.0–3.0 | 约峰值（消融证实：63→2 专家只省 1.3 ms） |
| hyper-connection（每层 2 个融合 kernel × 48 层） | ~1.5–2.0 | 作者已融合成 `inject_combine_norm` 单 kernel |
| lm_head + argmax | ~0.5 | 必要 |
| attention（9 行） | ~0.3–0.5 | flashinfer FA2 |
| accept | ~0.1 | 必要 |
| indexer | **0** | 本轮流式消融证实 |
| 残差（逐节点派发） | ~1–2 | 已无可合并对象 |

**结论：阶段一在「不改变数值」的前提下已经到头。** 每个组件要么在峰值带宽的 78–100%，要么已经是一个融合 kernel；
剩下的 1–2 ms 是「每层每分支至少一次 kernel 启动」的硬成本，只能靠跨层融合（换引擎）或换 kernel 实现（改数值）来消。

## 33. round 55：最后三个旋钮也测完 —— 引擎侧确实没有剩余可动的了

| 旋钮 | 结果 | 说明 |
|---|---|---|
| `HIBANA_SPEC_POLICY=budget`（自适应深度） | **引擎拒绝启动** | 明确报错：`budget (adaptive verification) serves the 27B's DFlash2 with HIBANA_SPEC above 0; Flash-Next, Nemotron and MTP keep their depth` —— 该策略不支持 Flash-Next |
| `HIBANA_BOLT_EXPERTS=fp8` | **IDENTICAL**，c1 255.7 / c8 1010 / c16 1501 | 比默认（b12x：258 / 1050 / 1570）略差 |
| `HIBANA_BOLT_EXPERTS=bf16` | **IDENTICAL**，c1 253.2 / c8 969 / c16 1528 | 同样略差 |
| indexer 逐层工作（`HIBANA_ABLATE_INDEXER=1` 消融） | **0 ms**（15.6 → 15.8） | 短上下文的 dense 变体里它本来就不干活 |

**两个附带结论**：① 三种 MoE 实现（b12x / fp8 / bf16）**逐位一致**（说明它们数学等价），默认的 b12x 最快；
② 自适应深度策略只对 DFlash2 开放，Flash-Next 的深度只能走静态表（已调到上限 8）。

**至此引擎侧的可调项清单全部测完**：深度（上限 8、表最优）、MoE 预算（有损）、MoE 实现（默认最优）、
草稿头格式/重打分/采样草稿（全负）、spec 策略（不支持）、plan 缓存与 PLE 批量（本会话已开）、
页表原地增长（已开）、quant/silu 合并（净亏）、indexer（0）、GEMV/GEMM 带宽（78–82%）、hyper-connection（已融合）。

## 34. round 56：安静窗口里的**权威基线**（GPU 0%、无 co-tenant）

| 点 | 贪心 tok/s（400 token） | 采样 tok/s（0.3、512 token/流、4 次取中位） | 目标 | 贪心达成 | 占带宽核算上限 |
|---|---|---|---|---|---|
| c1 | **270.8** | 186.0 | 455 | 59.5% | ~85%（320） |
| c2 | 405.9 | 324.0 | 700 | 58.0% | — |
| c4 | 639.0 | 524.8 | 930 | 68.7% | — |
| c8 | 1031.0 | 831.2 | 1570 | 65.7% | ~92%（1125） |
| c16 | **1608.4** | 1298.8 | 1690 | 95.2% | ~97% |

质量闸门：**IDENTICAL / PASS**。round_ms：14.8 / 17.2 / 17.8 / 22.1 / 29.7 ms。

**解读**：
- c1 的 270.8 已经是我在这台机器上见过的最好成绩（安静窗口），相当于带宽核算上限（~320）的 **85%**；
- c16 的 1608 相当于其带宽上限（~1125 GB÷?) 的 97%——**并发档基本榨干**；
- 目标表的 455/700/930/1570/1690 里，只有 c16 接近（95%），其余仍受「草稿接受率」限制（§26–§27）。

**这份数据应作为后续任何改动的对照基准**（同一个安静窗口内交错测量才有意义）。

## 35. round 57：层内是**严格串行链**，多流并行不成立；阶段一收口

我本来想用「双流捕获把 attention 分支与 MLP 分支重叠」来吃掉每层 kernel 的 ramp/drain。读了 `model.py` 的
`DecoderLayer.forward` 后确认**不成立**：

```python
h, res = self.attn_hc.mix(hyper, normed)
h      = all_reduce(self.mixer.forward(h, ctx))        # 注意力分支
hyper, normed = self.attn_hc.combine(h, res, then=self.mlp_hc)   # ← 注意力的输出喂给 MLP 分支
h, res = self.mlp_hc.mix(hyper, normed)
h      = all_reduce(self.moe(h, ctx))                  # MoE 分支
```

MLP 分支的输入是注意力分支经 hyper-connection combine 之后的结果 ⇒ **两条分支不独立**，
层内没有任何可重叠的并行度；每层就是一条 ~5 个内核的严格依赖链，48 层串起来 ~240 个内核。

**c1 的账（安静窗口，14.8 ms）**

| 项 | 字节 | 峰值时间 | 实测 |
|---|---|---|---|
| trunk 稠密 | 5.38 GB | 3.4 ms | 孤立 82% |
| 专家（9 行 × topk 8 ≈ 63 个不同专家 × 3.84 MB × 48 层） | ~11.6 GB | 7.3 ms | MoE 内核到顶 |
| 草稿链 8 步 | ~1.8 GB | 1.1 ms | 孤立 78% |
| **合计** | **~18.8 GB** | **11.8 ms** | **实测 14.8 ms** |

⇒ **带宽天花板 ≈ 3.98 tok ÷ 11.8 ms = 337 tok/s，实测 270.8 = 80%**。差的 20%（约 3 ms）
是 ~240 个严格串行内核的 ramp/drain（每个内核 60 µs 里只有 ~40 µs 在有效读写）。

**要吃掉这 20% 只有一条路**：把每层 5 个内核压成 2–3 个（跨算子融合：norm+quant 合并、router+topk+combine 合并、
attention 的 prep/compress 合并）。这些**不是 elementwise 合并**，需要自写 Triton/CUDA 内核，
而且会改变每个 kernel 内部的归约/分块 → **逐位一致大概率不保**。

**阶段一结论（可交付）**：
1. 没有「2–3 倍」空间——带宽型内核孤立测试都在 78–82%，MoE 到顶，层内无并行度；
2. 现实天花板 ~337（c1），当前 270.8 = **80%**；c16 已到 97%；
3. 剩下 20% 只能靠自写融合内核换数值，或换更快的 GEMM 实现（cutlass tile/GEMV 网格/flashinfer 后端）。

## 36. round 58：把 107 的 Flash-Next 接进 107 自己的 DSH（含视觉与思考等级）

用户要求「把模型加入 107DSH 先试试，图片视频、思考等级都要能用」，照 102/105 的既有做法（灵枢记忆
`mem_1792558189583` / `mem_1791310055179`：provider 写在 `~/.dsh/profiles/web/cordis.patch.yml` 的
`llm-pi-ai.config.providers` 下，密钥在 `~/.dsh/.credentials.yaml`）。

**(1) provider 已加**（备份 `cordis.patch.yml.bak-hibana107`）

```yaml
      hibana-107:
        displayName: Hibana @ 107 (Flash-Next 125B / RTX PRO 6000 / localhost)
        apiKeyEnv: DGX_TENSORFOLD_API_KEY      # 本机服务不需要真 key，复用已存在的这条
        api: openai-completions
        baseURL: http://127.0.0.1:30004/v1
        compat:
          thinkingFormat: chat-template
          chatTemplateKwargs:
            enable_thinking: {$var: thinking.enabled}
            reasoning_effort: {$var: thinking.effort, omitWhenOff: true}
        models:
          - id: RadixArk/Qwen3.8-Flash-Next-NVFP4
            name: Hibana 107 · Flash-Next 125B（本机 · 视觉 · MTP 投机 · 32K）
            contextWindow: 32768        # 引擎是 --context-length 32768
            input: [text, image]
            reasoningEfforts: {off: none, low: low, medium: medium, xhigh: xhigh}
```
`contextWindow` 与 `defaultMaxTokens` 按 107 的实际配置（32768 / 8192）填，**没有**照抄 105 的 262144。

**(2) 视觉原来是关的 —— 已打开并验证**：`engine/config.py:198` 有 `vision=_flag(e,"HIBANA_VISION")`。
`run-flashnext.bat` 加 `set HIBANA_VISION=1`（备份 `.pre-vision`），重启后：

```
text/think-off: 2 tok | thinking 0 chars
text/think-on : 43 tok | thinking 146 chars     ← enable_thinking 生效
image        : 56 tok | "…Left side: Red …"     ← 64×64 红蓝图，模型正确描述
```
（在加这个开关之前，图片请求返回 400 `this server runs a text-only model`。）

**(3) 视频**：引擎的媒体入口只认 `image_url`/`input_image`（`api/media.py`），**没有 video 类型**；
DSH 的模型 `input` 也只有 `text`/`image`。所以视频要按**帧**发（多条 image part），不能直接塞视频文件。

**(4) 一个已知坑**：`enable_thinking: true` 配 `reasoning_effort: "none"` 会让引擎返回 **HTTP 500**。
DSH 侧用 `omitWhenOff: true`（关思考时只发 `enable_thinking:false`、不发 effort），所以正常路径不会踩到。
思考等级实测：low/medium/xhigh 的推理长度 424/455/339 字符（不是单调），参数确实生效但不是硬上限。

**(5) 顺带修复**：上传 provider 那次我用的本地 `run-flashnext.bat` 里深度还是旧的 `8:2`，覆盖了线上的 `8:3`；
已改回 `1:8,2:6,4:3,8:3,16:4` 并重启验证（IDENTICAL）。

## 37. round 59：B 路线找到并在 107 上实测了「训练好的草稿头」；A 路线的第一条（attention 后端）不通

### B：蒸馏管线在 102 的 `~/mtp_train`，12 个头都在，但**都比 107 现用的差**

| 产物 | 说明 |
|---|---|
| `train.py` | 蒸馏 Flash-Next 的 MTP 头：**4 步自喂 rollout、CE + KL(top-32)、每专家 BF16 matmul + Rank-16 LoRA**；`--validate` 先用「它是服务端的头」对拍自己的前向；`--steps 784` 训练并导出 |
| `extract.py` / `features/` | 从模型抽特征 |
| `gen_{code,prose,zh,focus,par}.py` + `corpus*.jsonl` | 五套语料生成器（代码/散文/中文/焦点/并行） |
| `out_mtp2..13` | 12 个训练好的头（各 4.9 GB，`mtp_weights.safetensors`）——就是引擎能直接加载的格式 |

**107 上实测（把 mtp11 / mtp13 拷过去，171 MB/s，各 20 s）**：

| 头 | 接受率 | c1 贪心 | 贪心 md5 |
|---|---|---|---|
| 107 现用（11-16 14:27） | **0.605** | **260–270** | IDENTICAL（基准） |
| `out_mtp13`（11-17 02:28，最新） | 0.576 | 222 | **DIFFERENT** |
| `out_mtp11`（102 线上那版） | 0.582 | 214 | **DIFFERENT** |

⇒ **现用的头是这套里最好的**；两个新头更慢且改 md5，已全部回退并复验 IDENTICAL。

**⚠️ 一个必须记下的疑点**：换草稿头竟然改变了**贪心**的 200-token md5。按设计「精确校验 + `accept_top_k=1`」
输出应当与草稿无关。要么引擎存在一条不经精确校验就采用草稿 token 的路径，要么新头的「law」触发了
relaxed/soft 分支。这次两个候选本来就慢，所以没深挖，但**「换草稿头 = 质量中性」这个前提在这次实验里不成立**，
下次动 B 之前应该先查清楚（这是会静默换质量的那类问题）。

### A：第一条路（flashinfer 换后端）不通

`probe_backend.py`（同一几何：9 行 / 24 q 头 / 2 kv 头 / hd 256 / fp8 KV / page 1）：

| backend | 结果 |
|---|---|
| `fa2` / `auto` / `trtllm-gen` | `ninja exited with status 2`（这个形状的 kernel 变体在这台机器上编不过） |
| `fa3` | `ninja exited with status 1` |
| `cudnn` | `qo_indptr[-1]` 语义不同（cudnn paged prefill 要求另一种 indptr 约定） |

⇒ 换后腿不是「一行开关」，要继续 A 只能走**自写 kernel**（trunk fp8 GEMM 的 M=9 tile、M=1 GEMV 的 grid、
verify 的逐层融合）。A 的天花板仍是 c1 ~337（现 260–270），即 **+20~25%（c1）/ +5~8%（高并发）**。

### 顺带（用户报告的 413 已修）

`preflight_compress_failed: outbound body ~34,483 tokens vs model window 32,768`（其中 tools[] 占 17,526）。
已把引擎 `--context-length` 从 **32768 提到 262144**（KV 池本来就是 524288 pages，不用加显存；
重启后 91.2 GB 占用、`/health` 200），DSH 的 `hibana-107` provider 也同步改成 `contextWindow: 262144`、
`maxTokens: 32768`。刷新 GUI 后那条 413 应该消失。

## 38. round 60：B 的那个「换头改 md5」疑点查清了 —— **md5 只是同配置回归，不是跨配置的质量指标**

用户让我先查清「换草稿头为什么改变贪心 md5」再开训。做了三组对照（同一 prompt、temp 0、200 token）：

| 配置 | md5 | 首次分歧位置 | 与 depth0 的相似度 |
|---|---|---|---|
| **depth 0（完全不投机，纯 decode）** | `879b6bbc…` | — | 1.000 |
| depth 8 + 现用头 | `ad4d58f4…`（本会话的基准） | 第 67 字符 | 0.297 |
| depth 8 + mtp13 | `68eb5d25…` | 第 39 字符 | 0.313 |

**结论：投机解码的输出与纯 decode 的输出本来就不一样**，而且分歧发生在 8–15 个 token 之后（贪心从此走向不同分支）。
原因是**数值随批形状变**：纯 decode 走 1 行的 GEMV 路径、verify 走 9 行的 GEMM 路径（cuBLAS 的 tiling/累加顺序不同），
两边的最后几位不同 ⇒ 贪心在某个近并列的 token 上分叉。

**推论（很重要）**：
1. 本会话所有的 `IDENTICAL` 都是**同配置**回归（同深度、同批、同头），这个结论依然有效；
2. 但**不能**用 md5 比较不同草稿头 / 不同深度 —— 它们改的是「每轮走哪条路径」，本来就会换 token 流；
3. 所以之前 mtp13 的 `DIFFERENT` **不是质量 bug**，也不该成为否决理由；否决它的真正理由是
   **接受率 0.576 < 0.605、吞吐 222 < 260**（这两个才是跨配置可比的指标）；
4. 要比较头的「质量」，必须用真正的评测（perplexity / 任务准确率），不能靠 token 流比对。

## 39. round 61：**修好了引擎内 CUPTI**，第一次拿到解码轮的逐 kernel 归因 —— 46.6% 是 MoE 的 231 次小内核

**修法**（`engine/loop.py`，备份 `.pre-profwarm`）：CUPTI 的 client 注册是**线程局部**的，sentinel 触发的 profiler
在另一个线程启动就会报 `External init callback must run in same thread as registerClient`。
在 `_run_inference()` 开头做一次「空窗口」预热（`activities=[CPU, CUDA]`）把 CUPTI 注册到引擎线程，
之后 sentinel 窗口就能正常出表。（被 profile 的请求会慢到 11 tok/s，CUPTI 开销正常，所以只在短窗口用。）

**c1 解码轮逐 kernel（30 步，377.8 ms device = 12.6 ms/步）**

| 类别 | ms | 占比 |
|---|---|---|
| **moe (w3)** | **176.04** | **46.6%** |
| gemm | 81.00 | 21.4% |
| other | 73.53 | 19.5% |
| norm | 26.83 | 7.1% |
| gdn/mamba | 12.53 | 3.3% |
| attention | 5.58 | 1.5% |
| ple | 2.30 | 0.6% |

**Top 内核（调用次数 = 每轮总数）**

| 内核 | ms | 次数 | 每次 |
|---|---|---|---|
| `w3_gemm_kernel<1,1,0,3>`（MoE 专家） | 82.30 | **6931** | 11.9 µs |
| `sm89_xmma_gemm_e4m3…32x64x64` | 38.65 | 6165 | 6.3 µs |
| `moe11gemm_kernel<1,1>` | 28.12 | 1440 | 19.5 µs |
| `w3_gemm_kernel<1,1,0,2>` | 17.91 | 147 | — |
| `moe11gemm_kernel<0,1>` | 16.51 | 1440 | 11.5 µs |
| cutlass wmma bf16 `128x2` | 15.91 | 1080 | 14.7 µs |
| `_inject_norm_kernel` | 12.63 | 2914 | 4.3 µs |
| `_layer_norm_fwd_1pass_kernel` | 12.08 | 1116 | 10.8 µs |
| `fused_sigmoid_gating_delta_rule_update`（GDN） | 12.03 | 1080 | 11.1 µs |
| `_quant_small_kernel` | 10.01 | 3123 | 3.2 µs |
| `_fp8_gemv_quant_kernel` | 8.24 | 1031 | 8.0 µs |

**这就是原始任务书里「trunk 231 次启动」的答案**：`w3_gemm_kernel` 每**步**被调用 **231 次**（6931/30），
每次只有 11.9 µs —— 典型的**逐专家 launch-bound**：每轮要启动 ~700 个内核，每个内核的 ramp/drain 摊在
11.9 µs 里，带宽根本喂不满。

**因此 A（走 verify）的真正靶子是**：把 MoE 的逐专家内核**合成每层一次分组 GEMM**（`w3_gemm.cu`），
把 231 次/步降到 ~48 次/步。若成功，MoE 的 176 ms（46.6%）有望砍掉 30–50%，即整轮 **-15~23%**
（c1 260 → 300–335），正好覆盖阶段一的目标。这也是本项目剩余的唯一"量级"工程。

**次一级靶子**（都在 other/norm 的 100 ms = 26% 里）：`_layer_norm`(12.1)、`_inject_norm`(12.6)、`_quant_small`(10.0)
三者合计 34.7 ms = 9.2%，都是 4–11 µs 的小内核，可用 Triton 融合成每层 1–2 个。

## 40. round 62：B 路线按 SAA-Brain 官方配方开跑（102 上）

**仓库**：`CharAzinabulu0079/SAA-Brain-3.8-flash-next-Hibana`（102 那套蒸馏的上游官方版）。
三步：`generate_corpus.py`（从运行中的引擎采 on-policy 双语语料）→ `extract_mtp_features.py`（**停引擎**，
前向导出 `out["norm"]` [T,10240] + top-32 目标 logits）→ `train_mtp_kl.py`（**4 步自喂 + 0.5·CE + 0.5·KL_top32**，
`torch._grouped_mm` 训 512 专家 rank-16 LoRA + 核心 110M 全量 BF16，训完合回 31 个原生 `mtp.*` 张量）。

**本轮实测进度（全在 102，`/home/qqelick/mtp_work_saa`）**
1. 语料：`--count 700 --concurrency 14` → **538 条成功**（132 条 `en_think` 域报错，引擎侧限制），
   4 域（zh_chat 137 / code 143 / zh_visual 146 / en_chat 142），**235,767 tokens**，速率 171 tok/s；
2. 抽特征：先 `systemctl --user stop hibana-flashnext` 释放显存（121 GiB 里 82 GiB 空闲），
   模型加载 22 s / 53.1 GiB，**2 个 shard 共 568 序列、235,767 tokens、~1000 tok/s**，
   产出 `mtp_init.pt`(5.2G) / `target_embed.pt`(1.3G) / `target_lm_head.pt`(1.3G) / `features/shard_00x.pt`(4.7G)；
3. 训练：`--epochs 2 --lr 6e-5 --batch-size 2 --grad-accum 4 --num-anchors 64`，
   508 train / 60 val，128 step/epoch @1.9 s ≈ **16 分钟**。

**训练器自带的验证指标（正是我们需要的跨配置可比口径）**
```
[VAL init] accept_len=2.650 s0=0.679 | code:len=4.038/s0=0.910 en:len=2.192/s0=0.626 zh:len=2.301/s0=0.610
```
即：原生头在**我们这份语料**上平均接受 2.65 个草稿（3.65 tok/round），首个草稿接受率 0.679
（代码 0.910、英文 0.626、中文 0.610）。训练后同一指标若上升，即为头的改进（再用 107 的接受率复验）。

**顺带发现（102 的部署口径）**：102 的 systemd unit 里有 `HIBANA_ACCEPT_SOFT=0.5`（**放宽验收**）、
`HIBANA_DRAFT_HEAD=w2`、`HIBANA_THINKING_BUDGET=4096`、`HIBANA_SSM_STATE=int8`。
所以 102 的 170 tok/s 是**带 0.5 放宽**的数字，107 走的是精确验收（更慢但无损）——两边数字不可直接比。

## 41. round 63：B 路线两版蒸馏头实测 —— **引擎内都更差**（0.552 / 0.560 vs 现用 0.605），原因指向 trainer↔engine 保真度

**做法**：按 SAA-Brain 配方在 102 上跑完整管线（语料 538 条 / 235,767 tokens → 特征 568 序列 → 训练），
然后把导出的头拷到 107，用**引擎自己的接受率**验收（这是跨配置唯一可比的口径）。

| 版本 | 初始化 | 训练 | trainer 自测 `accept_len / s0` | **107 引擎 accept_rate** | c1 |
|---|---|---|---|---|---|
| 现用头（11-16） | — | 用户早前训练 | init 2.769 / 0.718 | **0.6054** | 256–260 |
| 原生头（未训） | — | — | 2.650 / 0.679 | — | — |
| `saa1` | **原生**头 | 2 ep, lr 6e-5 | 3.174 / 0.775（**+20%**） | **0.5519** ✗ | 204 |
| `saa2` | **107 现用**头 | 2 ep, lr 3e-5 | 3.113 / 0.766（**+12%**） | **0.5524** ✗ | 111* |
| `saa2` + 头保 BF16 | 同上 | `HIBANA_TRUNK_KEEP_BF16=^mtp\.` | — | **0.5602** ✗ | 212 |

（*c1 111 是当时 GPU 有外部占用，不代表配置。）

**三个结论**

1. **107 现用的头（0.605）本来就比原生（0.679 s0 → 引擎 0.605）和这两版都好** —— 用户早前的训练是有效的；
2. **两个新头在 trainer 自测里都明显变好（+12~20%），在引擎里却一致变差（0.552）** —— trainer 的指标**不能**外推到引擎；
3. **fp8 不是主因**：把 MTP 头保 BF16，只从 0.552 → 0.560（仍然远低于 0.605）。

**最可能的原因**（下一步要查）：trainer 是引擎 `mtp.py` 的「reference copy」，两侧的
hyper-connection/norm 约定或 draft 采样律（引擎是 **16 候选 shortlist + 目标头重打分后的 argmax**，
trainer 是整词表 top-k/KL）若有一处不一致，训出来的权重就会「在 trainer 里更好、在引擎里更差」。
要定位得做一次**同前缀对拍**：同一段 token，比较「引擎实际草稿」与「trainer 前向草稿」是否逐位相同。

**服务已恢复**：标准 `run-flashnext.bat` + 原草稿头，`/health` 200、c1 逐位一致、accept 0.6054。
102 的引擎也已恢复（`lane-fast.sh start`，health 200）——注意 102 的 unit 是 transient 的（`systemctl --user stop` 会销毁它，
恢复要用 `~/hibana/lane-fast.sh start`），而且 102 走的是 `HIBANA_ACCEPT_SOFT=0.5`（放宽验收）口径。

## 42. round 64：**推翻上一轮的 B 结论** —— 单 prompt 会骗人，六题套件显示两版头与现用头**等价**

上一轮我用 `refcheck`（单条英文技术 prompt）得出「saa1/saa2 更差（0.552 vs 0.605）」。这轮改用
**六题套件** `head_eval_suite.py`（en_tech / en_sys / zh_prose / zh_chat / code / reason，
每题的接受率取自引擎计数器差分）重测，结论完全不同：

**贪心（temp 0）平均接受率**

| 头 | en_tech | en_sys | zh_prose | zh_chat | code | reason | **均值** |
|---|---|---|---|---|---|---|---|
| 现用 | 0.5535 | 0.6219 | 0.5054 | 0.4950 | 0.6992 | 0.6182 | **0.5822** |
| saa2 | 0.5777 | 0.6150 | 0.4816 | 0.4536 | 0.6992 | 0.6156 | **0.5738** |

**采样（temp 0.3）平均接受率**

| 头 | en_tech | en_sys | zh_prose | zh_chat | code | reason | **均值** |
|---|---|---|---|---|---|---|---|
| 现用 | 0.5027 | 0.5194 | 0.4574 | 0.4162 | 0.6676 | 0.5833 | **0.5244** |
| saa2 | 0.5121 | 0.5505 | 0.4466 | 0.4353 | 0.6495 | 0.5490 | **0.5238** |

⇒ **两版头与现用头在统计上等价**（贪心 −1.4%、采样 −0.1%），差异是**按域重分配**：
saa2 在英文技术/系统题上更好（+2.4 / +3.1 个点），在中文散文和代码上更差。
**所以上一轮「更差」是单 prompt 的假象，这里更正。**

**这也解释了 SAA-Brain 的收益表**：他们的增益集中在**采样档**（roleplay +23%、prose +5-10%），
贪心代码几乎不变；而我们这次的语料只有 508 条/2 epoch（他们 3000 条），所以落在「等价」而不是「更好」。

**下一步（已启动）**：102 上重新采语料 **2500 条**（`corpus2.jsonl`，~2 小时，后台跑），
跑完后：抽特征 → 训练（更多 epoch）→ 用**六题套件 + 五点**双口径验收（不再依赖单条 prompt）。

## 43. round 65：A 侧两件事 —— ① 找到 sm89 GEMM 回退的机制；② 实测 int8 SSM 状态是「并发换单流」的交易

### ① verify 的 9 行 fp8 GEMM 为什么落到 SM89 内核

读 `TrunkLinear.__call__` 的分派规则：

| 行数 | 走哪条路 |
|---|---|
| `rows <= 4` | 引擎自写 **CUDA GEMV**（`fp8_gemv_quant_cuda`，单次 launch，自带量化） |
| `rows <= 4`（其他宽度） | `fk.fp8_gemv_quant_any` |
| 其他 | `prefill_gemm.fp8_gemm(...)`；**它只在 `FP8_ROWS = 2048` 附近有 cuBLASLt 枚举**，其余返回 `None` |
| `None` 时 | 回退 `torch._scaled_mm` → **`sm89_xmma_gemm_e4m3bf16...32x64x64`**（profile 里 6165 次 / 38.65 ms / 10.2%） |

⇒ **M=9 正好落在「GEMV 太窄、prefill 太宽」的空档**。要吃掉这 10.2%，就得给引擎自己的 fp8 GEMM 加一个
**M=8/16 的小 tile 实例**（CUDA 侧、可逐位校验），这是 A 路线剩下最具体的一块。

### ② `HIBANA_SSM_STATE=int8`（102 生产在用）在 107 上实测 = 并发换单流的交易

三轮五点（暖机后）：

| 点 | fp32（现用） | int8 SSM | 变化 |
|---|---|---|---|
| c1 | 260–278 | **240–242（三轮都很稳）** | **−8%** |
| c2 | 437 | 343–429 | ~−8% |
| c4 | 681 | 624–667 | ~−5% |
| c8 | 1125 | 814–1147 | ~0 |
| c16 | 1742 | **2153–2165** | **+24%** |

机理：int8 状态在多请求时省带宽（c16 大赚），但每一步要 dequantize（c1 单流时纯属额外开销）。
本目标最远的点是 c1（61%），所以**保持 fp32**，把 int8 记为「高并发可选项」。
（102 用它是因为它的口径是 16 并发。）

### ③ 一条测量纪律（本轮踩到）

**刚重启后的第一轮五点会低估 c2+**（图捕获按桶惰性发生）：同一份代码，首轮 c2 337 / c8 833，
暖机后 c2 437 / c8 1125。以后五点前先跑一轮暖机（或连跑两轮取后者）。

### ④ 本会话最好的一次五点（暖机后，静默窗口）
`c1 278.6 · c2 437.4 · c4 680.6 · c8 1125.1 · c16 1742.0`（对目标 455/700/930/1570/1690 → 61/62/73/72/103%）

## 44. round 66：A 的「M=9 fp8 tactic」这条路被工具链堵死 —— flashinfer 的 cuBLASLt fp8 runner 在 107 上编不过

想法：引擎 `prefill_gemm.fp8_gemm` 只在 `FP8_ROWS=2048` 用 flashinfer 的 cuBLASLt runner 选更快的算法，
M=9（verify）回退到 `torch._scaled_mm` 的 sm89 内核。若把 tactic 表扩到 M=9 的形状，就能白拿那 10.2%。

实测（`probe_fp8_tactics9.py`，直接调 `get_gemm_module().cublas_fp8_gemm_runner()`）：

```
cuda_0.o : error LNK2019: 无法解析的外部符号 cublasLtMatrixLayoutDestroy / cublasLtMatmulDescCreate /
           cublasLtMatmulPreferenceCreate / cublasLtMatmulAlgoGetHeuristic ...
ninja: build stopped: subcommand failed.
```

**flashinfer 的 `bmm_fp8` 在这个 Windows + CUDA13 组合上链接不出 cublasLt 符号**（和本项目早前
`batch_prefill` JIT 失败同一类问题）。⇒ 这条「Python 侧白拿」的路不存在，只能走
**引擎自己 `kernels/csrc` 里加一个 SM120 的 M=8/16 fp8 tile**（真 CUDA 工程，按天计）。

**A 路线到此收敛**：device 时间已到带宽的 78–82%（MoE 46.6% 是带宽地板、GEMV 78%、trunk GEMM 82%），
剩下 15–20% 全部需要自写内核，且改动会改变每个 kernel 内部的归约顺序（逐位一致必然不保，属用户已授权的
「放宽」范围，但仍要固定语料 + 六题套件 + 五点回归）。**这一步不适合在「机器随时可能被别的会话占用」的窗口里
开始**（本轮 102 上就有另一个 DSH 会话在做时钟扫描实验），先记进待办。

**因此接下来主力放在 B（分子）**：接受率每 +0.06 就是 c1 +10~15%，且风险低、可逆。

## 45. round 67：把「rollout 步数」对齐引擎的 8 步 → **B 第一次拿到正的均值收益**（代码域 +11.5%）

**动机**：SAA 的 trainer 默认 `rollout_steps=4`，而 107 的草稿深度是 **8**（`HIBANA_SPEC_DEPTHS` 的 1:8），
接受率恰恰死在第 2–8 步（每步 ~0.30）。把训练的自喂 rollout 也拉到 8 步，等于直接优化引擎真正用的尾部。

**改动**（`mtp_work_saa3/training/train_mtp_kl.py`，两处）：
`step_batch(..., rollout_steps=8)`（训练 + 验证各一处）、`step_weights = [0.8 ** s for s in range(rollout_steps)]`
（原来是硬编码 4 项 `[1.0, 0.8, 0.64, 0.5]`，8 步会 IndexError）。

**训练**（从 107 现用头继续，lr 3e-5，2 epoch，508 序列）：
`[VAL init 8 步] accept_len=3.371 s0=0.729 | code 6.127/0.934 en 2.386/0.672 zh 2.713/0.672`
`[VAL ep2  ] accept_len=4.073 s0=0.760 | code 6.940/0.938 en 3.276/0.715 zh 3.283/0.710`
⇒ **accept_len +20.8%**（4 步版只有 +12%），en +37%、zh +21%。

**107 引擎实测（六题套件，这是唯一可信口径）**

| 模式 | 头 | en_tech | en_sys | zh_prose | zh_chat | code | reason | **均值** |
|---|---|---|---|---|---|---|---|---|
| 贪心 | 现用 | 0.5535 | 0.6219 | 0.5054 | 0.4950 | 0.6992 | 0.6182 | **0.5822** |
| 贪心 | **saa3** | 0.5588 | 0.5968 | 0.5067 | 0.4892 | **0.7795** | 0.6042 | **0.5892（+1.2%）** |
| 采样 | 现用 | 0.5027 | 0.5194 | 0.4574 | 0.4162 | 0.6676 | 0.5833 | **0.5244** |
| 采样 | **saa3** | 0.5269 | 0.5469 | 0.4441 | 0.3939 | **0.6979** | 0.5899 | **0.5333（+1.7%）** |

⇒ **两种模式均值都为正**，**代码域 +11.5%（贪心）/ +4.5%（采样）**；代价是 en_sys/reason 略降。
结论：**rollout 对齐引擎深度是对的**，但语料太小（508 条）限制了幅度。

**下一步（语料齐了就做）**：`corpus2`（2500 条，当前 ~1000）跑完 → 抽特征 → **8 步 rollout + 4 epoch** 训一版，
用同一套件验收。预期把 +1~2% 推到 +4~8%。

## 46. round 68：**确立单流为主的验收口径**，v2（2000 条语料 + 8 步 rollout）单流 **+2.6%**

用户明确：**单流突破才是真突破**，日常最多用到 c1–c4，c8/c16 次要。所以 B 的验收口径改为
**单流分域吞吐**（`c1_domains.py`：en_prose / zh_prose / code / zh_chat，贪心、600 token、best-of-2），
不再是六题接受率或单条 refcheck。

**v2 = 2000 条语料（748k tokens，2000 序列）+ 8 步 rollout + 1 epoch（lr 3e-5，从现用头继续）**
trainer 自测：`accept_len 3.509 → 4.300（+22.5%）`、code 6.048→7.202、en 2.700→3.616、zh 2.711→3.295。

**107 单流实测（同窗口、best-of-2、600 token）**

| 域 | 现用头 | **v2** | 变化 |
|---|---|---|---|
| en_prose | 246.8 | 247.8 | +0.4% |
| zh_prose | 183.4 | **187.4** | +2.2% |
| code | 404.9 | **431.0** | **+6.4%** |
| zh_chat | 182.6 | 177.5 | −2.8% |
| **均值** | **254.4** | **260.9** | **+2.6%** |

⇒ **单流均值首次实现正收益**，主力来自代码域（+6.4%）与中文散文（+2.2%）；中文口语略降。
（此前 refcheck 报出的 c1 223 是测量窗口被 102 重启等动作污染的假读数——已用同窗口 best-of-2 复测澄清。）

**下一步**：v3 = 从 v2 继续训练 2 epoch（lr 2e-5，同一批 2000 条特征），用同一把尺子验收。
训练约 30 分钟，v3 出来后若单流均值继续上升就固化，否则停在 v2。

## 47. round 69：v3 已固化为默认草稿头 —— 单流均值 **+4.5%**（代码 +9.6%、中文散文 +7%），唯一代价是中文口语

**v3** = 2000 条语料 + 8 步 rollout + 1 epoch（从 v2 继续，lr 2e-5）。trainer 自测 `accept_len 4.297 → 4.453`（已饱和，
第 2 epoch 4.459 基本不动）。

**107 单流分域（`c1_domains.py`，贪心 600 token，best-of-2；v3 三轮取中位）**

| 域 | 原头 | v2 | **v3** | v3 vs 原头 |
|---|---|---|---|---|
| en_prose | 246.8 | 247.8 | ~246 | ≈0 |
| zh_prose | 183.4 | 187.4 | **~196** | **+7%** |
| code | 404.9 | 431.0 | **~444** | **+9.6%** |
| zh_chat | 182.6 | 177.5 | **~173** | **−5%** |
| **均值** | **254.4** | 260.9 | **~264.6** | **+4.0~4.5%** |

**已固化**：`run-flashnext.bat` 指向 `B:\hibana-models\mtp\mtp_weights.v3.safetensors`；
原 launcher 备份 `run-flashnext.bat.pre-v3`，原头备份 `mtp_weights.orig.safetensors`（一键回退）。

**唯一回退项是中文口语（−5%）** —— 用户明确日常以中文聊天为主，所以 v4 正在做**对话域加权**：
把 zh_chat / en_chat 各翻倍（3000 条：en_chat 1000 / zh_chat 1000 / code 500 / zh_visual 500），从 v3 继续 1 epoch（lr 2e-5），
目标是保住 code/zh_prose 的收益同时把 zh_chat 拉回来。

## 48. round 70：外部评审纠了两笔账，我逐一核实（**两笔都纠对了**）

外部前沿模型评审（用户转来）指出 §3.3 的分类错误，我核对源码与引擎自报数据后**确认它是对的**：

**(1) `w3_gemm` 6931 次/窗口 = 231 次/步 是「trunk 的 231 个 3-bit 线性层」，不是 MoE 专家。**
引擎启动日志原话：`w3: 231 calibrated trunk matrices from /home/qqelick/models/w3-flashnext` —— 231 与 6931/30 完全吻合。
真正的 MoE 是 `moe11gemm`（bf16 + 3bit 各 1440 次/窗口 = **48 次/步，即每层一次，已经是 grouped kernel**），
44.6 ms/30 步 = **1.49 ms/步（11.8%）**，不是 46.6%/5.9 ms。
⇒ **这直接解释了 Q2 的矛盾**：`HIBANA_MOE_BUDGET` 把 63 个专家砍到 2 个只省 1.3 ms，因为真实 MoE 计算本来就只有 1.49 ms。
⇒ **B3（重造 MoE 分组 GEMM）应立即降级**（它已经是 grouped，48 次/步）。

**(2) 专家流量算错 2 倍**：46.1 GiB ÷ **512**（不是 256）专家 ÷ 48 层 = **2.01 MB /（层·专家）**；
M=9 的 63 个不同专家 × 2.01 MB × 48 层 ≈ **6.1 GB/轮**（我原来写 11.6 GB，用了 256 个专家的除数）。
trunk 既然跑 `w3_gemm`，读量是 3-bit 口径（~2.4 GB），不是 FP8 的 5.38 GB。
⇒ 每轮真实流量 ≈ **8.5–10 GB** → 在 1.63 TB/s 上理论 **5.2–6.1 ms**，而实测 12.6 ms
⇒ **带宽利用率只有 ~50%，有 6 ms 左右不是带宽、而是"231+192+48 个串行小内核的启动/收敛 + WDDM 税"**。
单流的物理上限因此比原先估的 313 tok/s **高得多**（若能做到 80% 带宽，c1 可到 ~400）。

**同时证伪了评审的一条推断**（用实测）：

| 配置 | round_ms | 接受率 | tok/round | c1 / 单流均值 |
|---|---|---|---|---|
| 标准（v4 头，全词表） | ~15.6 | 0.60 | 3.87 | **~250 / ~272** |
| `HIBANA_DRAFT_VOCAB=32768` | 14.27 (−8.5%) | 0.537 | 2.80 | 176.6 / 207.7 |
| `HIBANA_DRAFT_VOCAB=65536` | 14.53 (−7%) | 0.543 | 3.15 | 197.5 / 222.0 |

⇒ 缩减草稿词表**只省 7–8.5% 的轮时间**（说明草稿链的 lm_head 读并不是 5 GB/轮的固定大头），
却让中文散文从 195 掉到 95–97 —— 因为 `draft_vocab()` 的排序是"special + **BPE id 顺序**"，
CJK token 的 id 很大、全被排除，中文草稿直接崩掉。
⇒ **该开关在本负载上是净亏**，已回退。（若要做，必须用真实语料频率重建短名单，而不是 tokenizer 顺序。）

**下一步（按评审建议调整优先级）**：
1. ~~MoE 分组 GEMM~~（降级：已是 grouped，只占 1.49 ms/步）
2. **M=9 的 192 次 `sm89_xmma_gemm`**：来自 HyperConnection 的 `down/up`（K/N=320，不是 128 倍数进不了 w3；rows=9 > GEMV_MAX_ROWS=4；又不在 FP8_ROWS=2048 附近 ⇒ 掉进 `torch._scaled_mm` 的 Ada 瓦片）。1.29 ms/步，做到 ~0.5 ms ⇒ **c1 +5%**。
3. **减少串行小内核**（norm 合并 0.89 ms/步、other 2.45 ms/步、兜底 231 个 trunk GEMM 的 ramp）。
4. **B1 v5 继续**（corpus3 已开跑：3000 条、修复了 en_think 域的 `reasoning_content` 丢失，现在 5 域齐全）。

### 48b. 同一配置的暖机复测（澄清"变慢"）

刚重启后的第一轮读数会被图捕获/缓存冷启动拉低（`refcheck` 首条 162→190），暖机后同一配置：
`c1_domains` → en 250.0 / zh_prose 196.7 / **code 498.3** / **zh_chat 201.1**，**均值 286.5**。
这是目前最好的单流均值：**vs 原头 254.4 = +12.6%**（v4 草稿头的功劳）。
⇒ 之前"变慢"的读数是冷启动假象，配置本身没退。
**已弃用的负优化**：`HIBANA_DRAFT_VOCAB=32768/65536`（−8.5% 轮时间但中文崩、整体 −20~26%）、
`HIBANA_DRAFT_HEAD=w2/w3`（无词表配合时接受率掉）、`HIBANA_SSM_STATE=int8`（c1 −8%）、`HIBANA_MTP_SAMPLE`、`HIBANA_BOLT_EXPERTS=fp8/bf16`。
**不做 WSL**（用户明确否决）——WDDM 税只能靠减少图节点/合并内核来摊薄。

### 48c. round 71：两条"吃掉 sm89 回退"的尝试**都是负优化**（已回退，附逐 kernel 证据）

外部评审建议"把 M=9 的 192 次 `sm89_xmma_gemm`（10.2%）换掉"。我试了两条路，都在引擎内 profile 验证：

**(1) 让 decode 走 `hyper_prefill` 的 wide 路径**（`mix` 里的 `_scaled_mm` → `combine_norm`+`up_gate_mean`）
→ **引擎启动即崩**：`RuntimeError: Rowwise scaling is not currently supported on your device`
（`hyper_prefill.mix` 用的是 rowwise `xs` scale，SM120 的 `_scaled_mm` 只吃 per-tensor）。已回退 `hyper.py.pre-wide`。

**(2) 把 K=320/640 的窄矩阵从 `_scaled_mm` 改到引擎自写 GEMV**（`NARROW_MAX_ROWS=16`）
`sm89` 确实**从 6165 次降到 0**，但代价是 `_fp8_gemv_quant_kernel` 从 1031 次 → **6693 次**：

| kernel | 改前 | 改后 |
|---|---|---|
| `sm89_xmma_gemm_e4m3...32x64x64` | 6165 次 / 46.66 ms（**7.6 µs/次**） | **0** |
| `_fp8_gemv_quant_kernel` | 1031 次 / 10.66 ms | **6693 次 / 76.04 ms（11.4 µs/次）** |
| device 总计（同一窗口尺度） | 453.4 ms | 394.2 ms*（*窗口步数不同，按 per-step 折算反而更高） |

⇒ **Ada 的 32x64x64 瓦片在 M=9 上比引擎自写 GEMV 快 1.5 倍**（7.6 µs vs 11.4 µs）。所谓"sm89 回退"其实不慢，
它的 10.2% 只是因为**调用次数多（205 次/步）**，不是因为单次慢。已回退 `linear.py.pre-narrow`。

**结论（修正外部评审的两条推断）**：
- "把 320 宽的矩阵塞进 GEMV/融合" 收益为**负**，因为自写 GEMV 在小 M 上不如 cuBLAS 的 Ada 瓦片；
- 真正的浪费是**调用次数**（231 个 trunk w3 GEMM + 205 个 sm89 + 48 MoE + 1116 norm + 3123 quant + 2914 inject），
  即"每层十几个小 kernel"，而不是某一条路径选错。**这正好回到 D3/D4：减少 kernel 数是唯一有量级的方向。**

### 48d. round 72：**频率版短名单也不行**（但暴露出一条值得给 B 路线的线索）

上一轮的 `DRAFT_VOCAB` 失败被我归因为"排序方式"（BPE id 顺序把 CJK 排除）。这轮用 **corpus3 的 101 万真实 token 按频次重建**：
17,268 个不同 id，短名单 17,297 行，**覆盖语料 100.0% 的 token**（含 33 个 special）。
引擎侧加了 `HIBANA_DRAFT_VOCAB_FILE`（torch 保存的 int64 短名单），绕开 tokenizer 顺序。

| 配置 | round_ms | accept | tok/round | 单流均值 |
|---|---|---|---|---|
| 标准（全词表） | ~15.6 | 0.622 | 3.62 | **~290** |
| 频率短名单 17K | **14.28（−8.5%）** | **0.547（−12%）** | 2.94 | 247.1 |

⇒ **短名单确实省下 1.3 ms/轮（外部评审这条判断对）**，但接受率掉 12%，净亏。已回退（`draft_vocab.py.pre-freq` 保留）。

**为什么覆盖 100% 还掉接受率**：`ShortlistHead` 的 argmax 只能落在短名单内，而评测 prompt 的词分布
与训练语料不同；同时 `rescore=8` 只把前 8 个候选拿回全量词表重排，剩下降级保留。
⇒ **让草稿头在短名单上训练过**（而非拿全词表头事后截断）才可能保住接受率。这是 B 路线的下一个课题，
已记入 TODO：`train_mtp_kl.py` 的 CE/KL 目标可以改成"短名单内的 softmax"，配合 `RESCORE` 兜底。

## 49. round 73：**v5 草稿头已固化** —— 单流均值 254.4 → **275~278（+8~9%）**，但收益按域分布

**v5 = corpus3（1634 条 / 103 万 token / 5 域含 `en_think`）+ 8 步 rollout + 2 epoch（lr 2e-5，从 v4 继续）**
trainer 自测：`accept_len 4.379(v4 起点) → 4.514(ep1) → 4.574(ep2)`，`s0 0.786 → 0.797`。

**107 单流分域（`c1_domains.py`，贪心 600 token best-of-2，交错 2 轮）**

| 域 | 原草稿头 | v4 | **v5（已固化）** | v5 vs 原头 |
|---|---|---|---|---|
| en_prose | 246.8 | 239–250 | 235–240 | **−4%** |
| zh_prose | 183.4 | 184–197 | 180–187 | ≈0 |
| code | 404.9 | 449–482 | **487–492** | **+20%** |
| zh_chat | 182.6 | 189–198 | 195–202 | **+7%** |
| **均值** | **254.4** | 272–279 | **275.5 / 276.2 / 278.1** | **+8~9%** |

**必须说清的取舍**：蒸馏头的收益集中在**代码（+20%）与中文口语（+7%）**，代价是**英文技术散文 −4%**。
`refcheck` 的 BENCH prompt（单条英文技术题）接受率从原头的 0.6054 掉到 0.552 —— 这条 prompt 恰好是蒸馏头最弱的域，
所以**以后单流验收以 `c1_domains` 的四域均值为准，不看单条 refcheck 的 c1**（它只作同配置回归用）。

已固化：`run-flashnext.bat` → `mtp_weights.v5.safetensors`；回退：`run-flashnext.bat.pre-v3`（原头）+ `mtp_weights.orig.safetensors`。

**B 路线累计**：原头 254.4 → v2 260.9 → v3 264.6 → v4 272.7 → **v5 276.6**（+8.7%）。

## 50. round 74：**单流验收口径改为「长散文」**，并用它纠正了 B 路线的方向

用户明确：日常体感最强的是**英文散文与中文散文**，代码速度次要。据此新增 `c1_prose.py`（英/中各 3 篇、900 token、贪心）
与 `prose_spec.py`（per-domain 的 `tok/round` / `round_ms` / 接受率，用计数器差分）。

**关键发现 1：散文的 `round_ms` 只有 12.1–13.0 ms**（比之前用 BENCH prompt 测的 15.6 ms 低），
所以散文的瓶颈**不是 device 时间，而是 `tok/round`**：

| 域 | 草稿头 | tok/round | 接受率 | round_ms | tok/s |
|---|---|---|---|---|---|
| en_prose | 原头 | 3.47 | 0.578 | 12.6–13.0 | 267.9 / 275.5 |
| en_prose | v5 | 3.48 | 0.578 | 12.9–13.2 | 263.4 / 270.1 |
| en_prose | v7（纯散文训） | **3.31** | **0.570** | 12.6–13.0 | 254.8 / 262.3 |
| zh_prose | 原头 | 2.59 | 0.509 | 12.3–12.7 | 204.5 / 210.1 |
| zh_prose | v5 | 2.55 | 0.508 | 12.1–12.3 | 207.7 / 209.8 |
| zh_prose | v7 | **2.40** | **0.483** | 12.4–12.5 | 192.1 / 193.4 |

**关键发现 2：散文的接受率在原头 / v4 / v5 / v7 之间几乎不变（0.578 / 0.508）**，
即"代码 +20%、口语 +7%"的收益**完全没有转移到散文**。
纯散文语料训出来的 v7 反而**更差**（3.47→3.31、2.59→2.40）——语料只有 404 条 / 35 万 token，
而且散文本身第 2–8 步的发散太大，单层串行 MTP 头学不动。

**结论（已固化）**：散文场景下 **v5 与原头等价**（en −1.8%、zh +0.7%，都在噪声内），
而 v5 在代码上 +20%、中文口语 +7%，所以**保留 v5**（`run-flashnext.bat` 已指向它），v7 不采用。

**散文要提速只能动 `tok/round`**：en 需 5.9（现 3.47）、zh 需 5.9（现 2.59），即链式接受率要从 0.51–0.58 抬到 ~0.75。
这条路在 404 条语料上已经证明走不动，下一步必须换量级（1200+ 条、更长训练）或换方案。

### 50b. round 75：试了 **DFlash2 块并行草稿头**（引擎原生支持）—— 加载成功但**接受率崩**，已回退

引擎里本来就有 `HIBANA_PROPOSER=dflash2` 与 `DRAFT_PINS`（正好指向 `incoai/Qwen3.8-Flash-Next-DFlash2`
revision `a22a1744e85c17364dcc54a6a420af7135d4c7f6`）。官方数据：接受长度 MTP 2.50–3.08 → DFlash2 **2.83–3.26（+10~15%）**，
且是块扩散（一次前向出整块 + selector 追踪一条连贯路径），理论上正好治我们的散文衰减。
权重 1.15 GB 已下载（HF 直连被墙，走 `hf-mirror.com` + `huggingface_hub` 成功）。

**实测（107，散文 900 token，MTP 对照见 §50）**

| 配置 | 域 | tok/round | 接受率 | round_ms |
|---|---|---|---|---|
| MTP v5（对照） | en_prose | **3.48** | **0.578** | 12.7–13.2 |
| MTP v5（对照） | zh_prose | **2.55** | **0.508** | 12.2–12.3 |
| DFlash2 k=8 | en_prose | 3.01 | 0.251 | 12.8–12.9 |
| DFlash2 k=8 | zh_prose | **1.53** | **0.066** | 12.5–12.9 |
| DFlash2 k=4 | en_prose | 2.62 | 0.406 | 12.3–12.5 |
| DFlash2 k=4 | zh_prose | 1.52 | 0.130 | 12.0–12.4 |

引擎日志确认它**正常加载并捕获了图**：`captured dflash2 backbone/projection graph for batch 1, depth 4`、
`captured exact-count draft feedback graphs: 9` —— 不是加载失败，而是**草稿与目标对齐失败**（中文几乎为 0）。

**最可能的原因（未验证）**：上游 DFlash2 依赖 `patches/qwen38-dflash2-capture.patch` 在 SGLang 里捕获
`target_layer_ids=[5,15,25,35,45]` 的特定隐藏状态；本引擎的对应 tap 与它未对齐（且本引擎的 Flash-Next 带
hyper-connection，第 5/15/25/35/45 层的语义可能不同）。另外官方评测用的是 temperature 1.0 + top-p 0.95 + xhigh，
我们是贪心，接受率不可直接对比。

**结论**：DFlash2 是**理论上最对的第四个方向**（块并行 → 不受第 2–8 步串行衰减），但**要用它必须先让特征 tap 对齐**，
这是引擎作者级别的改动（要读 `patches/qwen38-dflash2-capture.patch` 并与本引擎的 hidden-state 出口逐层比对）。
已回退到 MTP v5，权重与 launcher（`run-flashnext-dflash2{,-k4}.bat`）都留着，随时可继续。

### 50c. round 76：**trunk GEMM 在 c1 是延迟受限，不是带宽受限**（新证据，决定了 A 路线的天花板）

同一份代码在 c1（verify 9 行）与 c8（verify 24 行）的 CUPTI 逐 kernel 对照：

| kernel | c1（9 行） | c8（24 行） | 行数比 | 时间比 |
|---|---|---|---|---|
| `w3_gemm`（trunk，231 次/轮） | 6931 次 / 87.28 ms → **12.6 µs/次** | 7160 次 / 138.59 ms → **19.4 µs/次** | 2.67× | **1.54×** |

⇒ 行数涨 167%，时间只涨 54% ⇒ **c1 的 trunk GEMM 是 launch/latency-bound，不是带宽**：
每次只读约 10 MB（231 次共约 2.4 GB），在 1.63 TB/s 上本应 6.2 µs，实测 12.6 µs，即 **~51% 带宽**，
差的 6 µs 是每次 launch 的固定成本 + 小 M 的 tensor-core 利用率。
⇒ **c1 想要再快，只能减少 kernel 次数**（把每层 4.8 个共享输入的投影合成 1–2 个），
单靠"换更快的 GEMM 算法"没有意义（我在 §48c 已经用窄 GEMV 证伪过这条路）。
估算：231 → ~96 次，每次省 ~4 µs ⇒ **约 −0.54 ms/轮（散文 round 12.6 ms 的 4%）**。
这是 A 路线**最后一次**有量级的动作，且需要改权重布局（作者级）。

**散文口径下的现状（本轮定稿）**：en 253–275 tok/s、zh 195–210 tok/s，
`round_ms` 12.2–13.1、`tok/round` 3.31–3.48(en) / 2.40–2.59(zh)、接受率 0.57–0.58 / 0.48–0.51。

## 51. round 77：**安静 GPU 上散文的真实基线 = 297–303 tok/s**（外部队列退出），以及一个引擎既有 bug

### 51.1 外部 co-tenant 退出后，散文快了 12–15%

那个常驻 77 GB 的 Strata 进程（`D:\Strata\...\serve\server.py`）本轮不在了，GPU 空闲（0% util）后：

| 草稿头 | 安静窗口（本轮流） | 之前有 co-tenant 时 |
|---|---|---|
| 原头 | en 297.1–302.2 / zh 295.4–297.4，**均值 296.3 / 299.8** | en 269–286 / zh 266–287 |
| **v5** | en **307.9–310.0** / zh **287.8–297.1**，**均值 300.8 / 301.2** | en 272–281 / zh 264–280 |

⇒ **散文真实基线是 en ~306 / zh ~290（均值 ~299）**，比之前所有读数高 12–15%。
外部评审关于「WDDM 下 co-tenant 造成静默显存换页/带宽争抢」的方向是对的；
`Get-Counter '\GPU Adapter Memory(*)\Shared Usage'` 显示 shared 只有 0.47 GiB（**没有换页**），
所以是**带宽/L2 争抢**而不是换页 —— 这一点纠正了评审的机制猜测，但结论（先清 co-tenant 再测）相同。

**A/B 结论**：quiet 窗口下 v5 vs 原头 = en +1.9%、zh −1.4%，均值 +1.4%，
即 **v5 与原头在散文上仍然等价**（与 §50 一致，这次是可信的静默测量）。

### 51.2 发现引擎既有 bug：**任何 ≥1024 token 的 prompt 都会崩**

`POST /v1/chat/completions` 用一个 ~1450 token 的 prompt 会 503，日志：

```
File "hibana\kernels\hyper_prefill.py", line 181, in mix
    d = torch._scaled_mm(nq, down.weight.T, scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)
RuntimeError: Rowwise scaling is not currently supported on your device
```

`combine_norm` 产出的 `xs` 是 `[t, 1]` 行缩放，`mix` 又把它 `expand` 成 `[1, out_features]` 给 `scale_b`，
于是 `_scaled_mm` 选 RowWise 路径 —— **SM120 不实现**。
⇒ **`--context-length 262144` 实际上服务不了长 prompt**；只有 ≤1024 token（prefill 单 chunk）的请求能跑。
这解释了为什么之前所有测试都用短 prompt，也解释了 §48c 里"wide 路径启动即崩"——那不是我的补丁错，而是这条路径本身在 SM120 上不可用。

**尝试修复（未完成）**：把 `mix` 改成 tensorwise matmul + 事后乘行缩放（数学等价，只是因为
`_scaled_mm` 的 RowWise 限制而多一次 elementwise pass）。第一版报 `Invalid scaling configuration`
（`scale_a/scale_b` 必须都是 singleton），第二版把两者都改成 singleton + FP32 输出后，
**`_scaled_mm` 过了，但下游 `qsa_score_exact` 的 JIT 编译失败**：

```
cuda_0.o : error LNK2019: 无法解析的外部符号 __cudaRegisterFatBinary / __cudaLaunchKernel / __cudaGetKernel ...
qsa_score_exact.dll : fatal error LNK1120: 14 个无法解析的外部命令
```

这是 CUDA 13 + MSVC 的独立问题（该 kernel 只在长上下文的选择型路径里用到，所以短 prompt 从不触发）。
⇒ **长 prompt 支持需要两层修复**（rowwise scale + qsa kernel 编译），已回退 `hyper_prefill.py.pre-rowscale` 保持服务可用。

### 51.3 当前状态与下一步

- 线上 v5 头、标准配置、`/health` 200、**散文 en ~306 / zh ~290**。
- 散文语料：1400 条目标已完成 1098 条（56.7 万 token），还在跑。
- 下一步：① 用安静窗口重测五点（数字会比之前高，作为新基线）；② 散文语料跑完 → v8（纯散文 1400 条）。

## 52. round 78：**v8 纯散文草稿头训练完成（trainer 侧最佳），但 107 在实测前关机**

### 52.1 v8 = 纯散文语料 1400 条 / **121.7 万 token**（中英各半）+ 8 步 rollout + 3 epoch

语料由新写的 `training/generate_prose.py` 生成（20 个中英散文主题 × 900 字，`en_prose` 691 / `zh_prose` 709），
特征 1330 序列 / 121.7 万 token（24 GB），从 v5 继续训 3 epoch（lr 2e-5）。

**trainer 自测（8 步 rollout 口径，唯一可信的跨版本可比指标）**

| 版本 | 语料 | `accept_len` init → final | en | zh |
|---|---|---|---|---|
| v5 | 5 域 103 万 token | 4.379 → 4.574 | 3.82 | 3.51 |
| v7 | 纯散文 **35 万** token | 2.672 → **2.782** | 3.078 | 2.465 |
| **v8** | 纯散文 **121.7 万** token | 2.625 → **2.867（+9.2%）** | **3.186** | **2.566** |

⇒ **散文语料从 35 万扩到 122 万 token，`accept_len` 2.782 → 2.867（+3.1%），且 ep1→ep3 单调上升**
（2.811 → 2.855 → 2.867），说明还没饱和，继续加语料/epoch 仍有增量。
注意 v5 的 `accept_len` 4.574 是**混合 5 域**口径（含代码，代码域接受率本来就高），不能与纯散文的 2.867 直接比。

### 52.2 未完成的一步：107 在 A/B 之前关机

`scp mtp_weights.v8.safetensors` 时连接中断，`ping` 100% 丢包、`ssh` No route to host（约 09:30）。
所以 **v8 在引擎里的散文 A/B 没有做**。产物已在 102：
`/home/qqelick/mtp_work_saa10/out_mtp_v8/mtp_weights.safetensors`（md5 `0764b92d30d3…`，4.9 GiB）。

**开机后第一步**（脚本已备好）：
```
scp /home/qqelick/mtp_work_saa10/out_mtp_v8/mtp_weights.safetensors pro6000:'B:/hibana-models/mtp/mtp_weights.v8.safetensors'
scp run-flashnext-v8.bat pro6000:'B:/hibana-win/'
# 交错 A/B（散文，900 token）
for arm in v5 v8 v5 v8; do restart -Bat run-flashnext-$arm.bat; python c1_prose.py $arm; done
```
判据：散文均值（en+zh）/2 若 v8 > v5 + 3%（即 >310）则固化，否则保持 v5。

### 52.3 全部已训练草稿头清单（102 上，都可直接拷）

| 目录 | 版本 | 语料 | md5 前 12 |
|---|---|---|---|
| `mtp_work_saa/out_mtp_saa1` | saa1 | 5 域 538 条 | `32f577d348a5` |
| `mtp_work_saa2/out_mtp_saa2` | saa2 | 从现用头继续 | `159bd7ad90ce` |
| `mtp_work_saa3/out_mtp_saa3` | saa3 | 8 步 rollout 首版 | `89a35f887fcf` |
| `mtp_work_saa4/out_mtp_v2` | v2 | 5 域 75 万 token | `7b4e266e6951` |
| `mtp_work_saa5/out_mtp_v3` | v3 | 从 v2 续训 | `c9bdaa16059e` |
| `mtp_work_saa6/out_mtp_v4` | **v4** | chat×2 加权 | `2a04c368568e` |
| `mtp_work_saa7/out_mtp_v5` | **v5（线上）** | 5 域 103 万 token | `db3193327928` |
| `mtp_work_saa8/out_mtp_v6` | v6（**已弃**） | code×3 加权（训练被终止，与 v5 同 md5） | `db3193327928` |
| `mtp_work_saa9/out_mtp_v7` | v7 | 纯散文 35 万 token | `7bd53e363ddd` |
| `mtp_work_saa10/out_mtp_v8` | **v8（待测）** | 纯散文 **121.7 万** token | `0764b92d30d3` |

### 52.4 收尾时的两处状态修正（重要，避免下一个窗口踩坑）

1. **107 在 v8 的引擎 A/B 之前关机**（约 09:30，`ping` 100% 丢包 / `ssh` No route to host）。
   v8 权重在 102，md5 `0764b92d30d3…`；`run-flashnext-v8.bat` 已在本目录备好（本地 `hibana-107-work/`）。
2. **本地 launcher 模板漂移已修正**：`hibana-107-work/run-flashnext.bat` 一度指向**已弃用的 v4 头**、
   而 107 上实际跑的是 v5。已改回 `mtp_weights.v5.safetensors`，
   并确认四个 launcher 全部一致：`orig / v5 / v8`（+ 标准 `run-flashnext.bat` = v5），
   深度 `1:8,2:6,4:3,8:3,16:4`、`HIBANA_VISION=1`、`--context-length 262144`。

**注意**：`run-flashnext.remote.bat` 是本目录里一份**过期的远端快照**（8:2 深度 / 32K 上下文 / v4 头），
**不要**再用它当模板生成新 launcher —— 以 `hibana-107-work/run-flashnext.bat` 为准。

## 53. round 79：v8 的**可验证预测**与 v9 语料续产（107 仍关机，本轮无引擎数据）

107 仍不可达（`No route to host`），所以本轮只做三件不依赖 107 的事。

### 53.1 从 trainer 的成对验证反推 v8 的预期引擎收益

v8 的 `init`（= v5）与 `final` 用的是**同一套 70 条 held-out 散文序列**，所以是成对比较，可信：

| 指标 | v5（init） | v8（final） | 变化 |
|---|---|---|---|
| `accept_len` 全体 | 2.625 | **2.867** | **+9.2%** |
| `en` accept_len | 2.931 | **3.186** | **+8.7%** |
| `zh` accept_len | 2.336 | **2.566** | **+9.8%** |
| `s0`（第 1 个草稿的准确率） | 0.692 | 0.710 | +2.6% |

**关键点**：`s0` 只涨 2.6%，而 `accept_len` 涨 9.2% ⇒ **收益全部来自第 2 步之后**，
正是引擎链式衰减的位置 —— 这是 v7 之前没做到的（v7 的 `s0` 从 0.697 涨到 0.708 但总 `accept_len` 只 +4.1% 且引擎侧更差）。

**按 v5 的实测比例外推**（v5 的 trainer en 2.931 ↔ 引擎 tok/round 3.48；zh 2.336 ↔ 2.55）：

| 量 | v5 引擎实测 | v8 预期 |
|---|---|---|
| en `tok/round` | 3.48 | **~3.78（+8.7%）** |
| zh `tok/round` | 2.55 | **~2.80（+9.8%）** |
| en 散文 tok/s | ~306 | **~332** |
| zh 散文 tok/s | ~290 | **~318** |
| 散文均值 | ~299 | **~325（+8.6%）** |

这是**预测，不是实测** —— 107 开机后必须用 `c1_prose.py` 交错 A/B 验证；v7 的先例说明 trainer 指标并非必然转移。

### 53.2 v9 语料在产（107 不可用期间推进 B 路线）

`corpus_prose2.jsonl`：新 seed（19）、**1500 条**、`max-tokens 1100`，正在 102 上生成。
v9 管线已备好（`~/mtp_work_saa11/{run_extract.sh,run_train.sh}`，3 epoch / lr 2e-5，从 v8 继续）。

同时抽查了 v8 语料的质量：1400 条**答案前 80 token 全部互不相同**（1400/1400），8-gram 重复率 12.4%
（主要来自中文常见搭配）⇒ 语料多样性不是 v7 失败的原因，**规模**才是（35 万 → 122 万才见效）。

### 53.3 状态

- 107：关机中。开机后第一件事见 §52.2（v8 交错 A/B，判据 v8 > v5 + 3% 则固化）。
- 102：引擎在跑，v9 语料生成中。
- 文档：本节 + §52 已写入 HANDOFF（1840+ 行）；项目报告已含第二十轮。

### 53.4 v9 训练完成：散文蒸馏**已经饱和**

v9 = 第二份纯散文语料（新 seed 19、1500 条、**137 万 token**，en 568 / zh 582）+ 8 步 rollout + 3 epoch，
从 v8 继续（lr 2e-5）。

**trainer 成对验证（同一套 75 条 held-out 序列）**

| epoch | `accept_len` | en | zh |
|---|---|---|---|
| init（= v8） | 2.938 | 3.253 | 2.579 |
| ep1 | 3.024 | 3.352 | 2.649 |
| **ep2** | **3.076（+4.7%）** | **3.392** | 2.716 |
| ep3 | 3.036（回落） | 3.351 | 2.675 |

⇒ **ep2 见顶、ep3 回落** ⇒ 在 137 万 token / 3 epoch 这个量级上**散文蒸馏已经饱和**。
把 v8 与 v9 串起来看（注意两者的 val 集不同，只能比各自的相对增幅）：

| 阶段 | 语料 | 相对上一版 |
|---|---|---|
| v5 → v8 | 散文 0 → 122 万 token | **+9.2%** |
| v8 → v9 | +137 万 token（共 259 万） | **+4.7%** ⇒ 边际收益已减半 |

产物：`/home/qqelick/mtp_work_saa11/out_mtp_v9/mtp_weights.safetensors`（md5 `44c8f9606284…`）。

**同时排除一个组合实验**：把两份语料的特征合并（8 shard / **50 GB** / 259 万 token）训 v10 时，
训练进程在加载特征阶段被系统杀掉（`Train seqs=0`，121 GB 内存下 50 GB 特征 + 模型开销吃紧）。
⇒ **合并训练不可行**，只能单份语料串行续训（这也说明继续堆数据要先解决内存/流式加载）。

### 53.5 本轮结论

散文方向的三条路现在的状态：
1. **MTP 蒸馏**：v8（+9.2%）与 v9（+4.7%）都已训好，**待 107 开机做引擎 A/B**；预期散文均值 ~325（+8.6%），
   但 trainer→引擎的转移在 v7 上失败过，必须实测。
2. **DFlash2 块并行**：权重与 launcher 都在 107，实测接受率崩（zh 0.066），需重训或校准。
3. **A 路线减 kernel**：trunk GEMM 已证延迟受限，只剩「231 → 96 次」，作者级改动。

⚠️ 107 自 09:30 起关机，本轮**没有任何引擎侧新数据**；以上引擎相关结论均来自 09:30 之前。

## 54. round 80：**留出集验证成功** —— v9 的收益是真泛化，不是过拟合

107 仍关机，所以这一轮在 102 上做了一件**不需要引擎**也能下结论的事：
把 v5 / v8 / v9 三个头放到**同一套 75 条散文留出序列**上评（`corpus_prose2` 的 val split，
v5 与 v8 都没见过，v9 的训练也把它排除在外），用 trainer 自己的 8 步 rollout 指标（`--lr 0` 只读 init）。

| 头 | `accept_len` | 相对 v5 | en | zh | s0 |
|---|---|---|---|---|---|
| v5（线上） | 2.668 | — | 2.936 | 2.360 | 0.704 |
| v8 | 2.904 | **+8.8%** | 3.169 | 2.602 | 0.719 |
| **v9** | **3.017** | **+13.1%** | **3.327** | **2.662** | **0.729** |

**这是本项目第一个干净的跨语料泛化证据**：v5 与 v8 训练用的是**另一份**散文语料（`corpus_prose`，122 万 token），
评测用的是 `corpus_prose2` 的留出切片（137 万 token 语料里切出来的 75 条），两者主题/seed 都不同。
所以 +8.8% / +3.9% 的递增**不是对训练分布的过拟合**——这正好回答了 v7 留下的疑问
（v7 的 +4.1% 只在自己的 60 条 val 上成立，引擎侧反而更差）。

### 54.1 对引擎的预期（按实际口径算，不用 accept_len 直接等比）

引擎散文实测 v5：`tok/round` en 3.31–3.48、zh 2.40–2.59，其中含 1 个无法再省的基础 token：

| 域 | v5 已接受草稿数 | v9 按 +13% 折算 | v9 `tok/round` | v9 tok/s（round 不变） |
|---|---|---|---|---|
| en | 2.31–2.48 | 2.61–2.80 | 3.61–3.80 | 306 → **334–350** |
| zh | 1.40–1.59 | 1.58–1.80 | 2.58–2.80 | 290 → **300–318** |

⇒ **散文均值预期 ~325（+9%）**。这仍是**预测**，必须等 107 开机用 `c1_prose.py` 交错 A/B 实测。

### 54.2 待办（107 开机后，按顺序）

1. 拷 `mtp_weights.v9.safetensors`（md5 `44c8f9606284`）到 `B:\hibana-models\mtp\`，
   用 `run-flashnext-v9.bat`（= 标准配置 + v9 头）跑 `c1_prose.py` 交错 A/B：v5 / v8 / v9 各 2 轮。
2. 判据：均值 > v5 + 5% 则固化 v9；+2~5% 之间可考虑直接用（同一份数据也看五点是否退）。
3. 若引擎侧转移失败（像 v7 那样），说明 trainer 的 rollout 与引擎链仍有偏差，
   下一步要拿引擎的**实际接受计数**做校准，而不是继续堆语料。

### 54.3 散文蒸馏的完整曲线（全部为成对/留出验证）

| 版本 | 散文语料 | 留出 `accept_len` | 相对 v5 |
|---|---|---|---|
| v5 | 0（混合 5 域） | 2.668 | — |
| v7 | 35 万 token | 未测留出（自有 val 2.782，引擎更差） | 疑似过拟合 |
| v8 | 122 万 token | 2.904 | +8.8% |
| **v9** | +137 万 token（串行续训） | **3.017** | **+13.1%** |

⇒ **散文语料每翻一倍约再给 +4~9%**，且**串行续训（v8→v9）比合并训练更有效**
（合并 259 万 token 的 v10 因 50 GB 特征内存爆掉，已放弃）。

### 54.4 round 81：**rank-32 LoRA 再给 +1.4%**（同一留出集），语料+容量两条杠杆都还在

从 v9 继续，把 LoRA 秩从 16 提到 **32**、2 epoch、lr 1.5e-5：

| 头 | 留出集 `accept_len` | en | zh | s0 |
|---|---|---|---|---|
| v9（rank-16，线上候选） | 3.077 | 3.384 | 2.725 | 0.732 |
| **v10r32（rank-32）** | **3.120（+1.4%）** | **3.434** | **2.761** | 0.735 |

⇒ **容量也是有效杠杆，但在同一份语料上已接近饱和**（rank 16→32 只 +1.4%，而换语料 +4~9%）。
优先级因此明确：**继续扩语料 > 加秩**。v10r32 的产物在
`/home/qqelick/mtp_work_saa11/out_mtp_v10r32/mtp_weights.safetensors`，留作 v9 之后的第二候选。

### 54.5 散文蒸馏的杠杆清单（全部实测，非推测）

| 杠杆 | 实测增益 | 结论 |
|---|---|---|
| 语料 0 → 122 万 token（v5→v8，跨语料留出） | **+8.8%** | 有效 |
| 语料 +137 万（v8→v9，串行续训） | **+3.9%** | 有效，边际递减 |
| LoRA rank 16 → 32（v9→v10r32，同语料） | **+1.4%** | 有效但小 |
| 合并两份语料一起训（v10，259 万 token / 50 GB 特征） | — | **内存爆掉，不可行** |
| 纯散文 35 万 token（v7） | 自有 val +4.1% 但引擎更差 | **疑似过拟合，弃** |
| 短名单 `DRAFT_VOCAB`（两种排序） | −8.5% 轮时间但 accept −12% | 净亏，弃 |
| DFlash2 块并行（现成权重） | zh accept 0.066 | tap/分布不匹配，需重训 |
| c1 深度 4/6/8 | 噪声内 | 深度不是杠杆 |

**当前最佳候选**：v9（留出 +13.1% vs v5）与 v10r32（+14.7% vs v5），**都等 107 开机做引擎 A/B**。

## 55. round 82：**v9 固化 —— 散文 +11.2%、五点全线提升（107 回来后实测）**

107 于 13:21 恢复。v9 权重上传时撞上 **B: 盘写满**（只剩 36 KB，导致第一次上传只写了 2244 MB 的半截文件）。
清理掉我自己早前遗留的 `mtp11 / mtp13 / 半截 v9` 三个 4.9 GiB 文件（12.7 GB）后重传，
md5 `44c8f9606284` 与源一致（5,214,305,424 字节）。

### 55.1 散文（`c1_prose.py`，900 token，5 轮成对交错）

| 轮次 | v5 | v9 | 成对增益 |
|---|---|---|---|
| 1 | 292.8 | 326.1 | +11.4% |
| 2 | 290.6 | 323.4 | +11.3% |
| 3 | 275.4 | 302.8 | +9.9% |
| 4 | 280.0 | 319.5 | +14.1% |
| 5 | 276.8 | 302.6 | +9.3% |
| **均值** | **283.1** | **314.9** | **+11.2%（中位 +11.3%，最低 +9.3%）** |

⇒ **留出集的 +13.1% 转移到了引擎侧，实测 +11.2%，5/5 全正、无一轮为负。**

### 55.2 五点（`sweep.py`，同窗口成对，各 2 次）

| 点 | v5 | v9 | 提升 |
|---|---|---|---|
| c1 | 186.3 / 199.4 | 214.5 / 237.3 | **+17%** |
| c2 | 370.8 / 376.6 | 450.3 / 458.0 | **+21%** |
| c4 | 599.8 / 614.0 | 779.1 / 672.6 | **+20%** |
| c8 | 970.6 / 979.0 | 1010.8 / 1052.8 | +6% |
| c16 | 1700.0 / 1680.4 | 1713.6 / 1864.5 | +6% |

**并发档也涨了**（c2 +21%、c4 +20%）—— 因为 `tok/round` 上升对每一档都直接有效，
不只是单流。这一轮是本会话**除 host 侧优化外最大的一次性收益**。

### 55.3 引擎内分解（`prose_spec.py`，v9）

| 域 | tok/round | 接受率 | round_ms |
|---|---|---|---|
| en_prose | **4.26**（v5 时 3.31–3.48） | **0.608**（v5 时 0.570） | 14.6–14.9（v5 时 12.6–13.1） |
| zh_prose | 2.49（v5 时 2.40–2.59） | 0.493（v5 时 0.483） | 13.3–13.4 |

⇒ **英文散文的 `tok/round` 涨了 23–29%**（3.31 → 4.26），是本轮收益的主体；
中文散文只涨 4%，**仍是短板**（接受率 0.493，需要更多中文语料或不同的训练目标）。
注意 `round_ms` 同时上升（12.6 → 14.6）：接受更多草稿意味着每轮多提交一些 token，
状态更新/重建的随行工作也变多 —— 这是"分子变大、分母也略增"，净收益仍 +11%。

### 55.4 固化与回退

- 线上：`run-flashnext.bat` → **`mtp_weights.v9.safetensors`**
- 回退：`run-flashnext.bat.pre-v9`（= v5）、`mtp_weights.orig.safetensors`（原始头）
- 备用候选：`mtp_weights.v10r32`（留出集 +14.7%，比 v9 再 +1.4%）尚未上机测试

### 55.5 教训（写进 LESSONS）

**B: 盘写满会导致 `scp` 写出半截文件而且不报明显错误** —— 上传后必须比对**字节数与 md5**，
不能只看 `scp` 的返回码。这次差点用一个截断的 4.9 GiB safetensors 去重启引擎。

### 55.6 round 83：**逐位置接受率曲线拿到了，它推翻了"草稿链衰减"这个旧假设**

给 trainer 的验证加了逐位置条件接受率（`P(位置 s 命中 | 之前每一位都命中)`），在 v9 上测得：

```
位置:    1      2      3      4      5      6      7      8
接受率: 0.730  0.662  0.639  0.610  0.649  0.622  0.581  0.567
```

**曲线几乎是平的**（0.73 → 0.57，只衰减 22%）。

这推翻了从会话开始一直沿用的旧模型「第 1 个草稿 0.85、第 2–8 个各约 0.30」——
那个数字是用**原始头 + 混合语料**在**很短上下文**上反推的。用 v9 在长散文上重测，真实的失败模式是：

> **单 token 预测准确率只有 65–73%，而且基本不随位置衰减。**

⇒ 瓶颈不是"链式误差累积"，而是**MTP 头的单步准确率本身**。
这也解释了为什么继续堆语料只给 +4~9%/倍：它抬的是那条平线的高度，不是治衰减。

**中文更差的原因也定位到了**：`s0`（首 token 准确率）en 0.770 vs **zh 0.684**，
而后续位置的曲线形状两端几乎一致 ⇒ **中文的短板在"第一个 token 就猜不准"**，
不是链式问题。这指向两件事：中文语料的覆盖/权重，或中文 token 的熵本身更高。

### 55.7 由此推出的上限判断

| 目标 | 需要的逐位置接受率 | 现状 |
|---|---|---|
| accept_len 3.0（当前 v9） | ~0.65 平线 | 达到 |
| accept_len 4.0 | ~0.75 平线 | 需 +15% 单步准确率 |
| **accept_len 5.9（c1 455 tok/s）** | **~0.87 平线** | 需 +34% 单步准确率 |

**单层串行 MTP 头要再涨 34% 的单步准确率，没有先例支撑**。
所以 455 这个目标在 MTP 架构下基本不可达；能大幅改变的只有**块并行草稿**（DFlash2 这类
一次看整块上下文的结构），而现成 DFlash2 权重在本引擎的量化目标上不匹配（zh 接受率 0.066）。

### 55.8 本轮动作：中文加权版 v11zh

既然中文短板在首 token，用**中文过采样 ×2**（1500 zh / 750 en，复用现有特征、无需重新生成语料）
从 v10r32 继续训 2 epoch（rank 32），验证"中文加权"能否把 `s0_zh` 从 0.678 抬起来。

## 56. round 84-85：**三次"trainer 涨、引擎跌"的共同规律找到了**（这一条决定了后续所有训练）

### 56.1 三个失败的微调（全部在引擎里实测过）

| 实验 | trainer 指标 | 引擎英文散文 | 训练数据 |
|---|---|---|---|
| v11zh（中文过采样 ×2，rank 32） | +2.8%（zh +3.7%） | **−8%**（v9 303.3 → 286.4/278.9） | **复用** corpus2 特征（复制样本） |
| v12en（英文单域，rank 32，2 epoch） | **+4.5%**（3.257 → 3.403） | **−3%**（v9 en 318.2 → 308.4） | **复用** v8/v9 已训过的英文特征 |
| `HIBANA_DRAFT_HEAD_RESCORE=16 / 32` | accept 反而降（0.583→0.556/0.579） | −9% / −4% | — |

### 56.2 对照：唯一成功的那次用的是**全新语料**

| 实验 | trainer | 引擎 | 训练数据 |
|---|---|---|---|
| **v9** | +13.1%（留出集） | **+11.2%**（5/5 轮） | **全新** corpus_prose2（137 万 token，新 seed、新主题） |

⇒ **规律：在已经见过的数据上继续训（复制、过采样、同域重训）会让引擎性能下降，
哪怕 trainer 的验证指标在涨；只有加入全新语料才会转移成功。**
机理上说得通：trainer 的验证集如果是同分布的重采样，多训几轮就是记忆化，
bf16 下指标好看；而引擎侧是 fp8 量化 + 不同的链式路径，记忆化的权重反而偏离泛化最优。

### 56.3 因此本轮的动作：**新一份英文语料**

按要求把英文散文作为第一优先（用户的思考与推理主要走英文），
在 102 上启动 `corpus_prose3`：**2200 条、新 seed 777、新主题池**，目标 ~200 万 token，
生成速率约 150 tok/s（约 3.5 小时，跨轮进行）。
跑完后：抽特征 → 从 v9 继续训（rank 32、2 epoch）→ 107 上英文散文交错 A/B。

### 56.4 v9 当前英文散文的稳定读数（本窗口）

`c1_prose.py` 英文：**315.2 / 322.4 / 328.4 / 307.9 tok/s**（均值 ~318），
`accept 0.583`、`tok/round 4.26`、`round_ms 13.9–14.0`。
相对原始头（散文均值 ~296–300）与 v5（283.1）分别 +7% 与 +11.2%。

### 56.5 英文散文的设备预算（v9，CUPTI 窗口 383.5 ms，约 30 步）

| 组成 | ms/轮 | 占比 | 备注 |
|---|---|---|---|
| **trunk 3-bit GEMM**（`w3_gemm` 231 次） | **3.51** | 27.4% | 12.6 µs/次，单次带宽仅 ~51% |
| **MoE**（`moe11gemm`/prep/combine/quant ≈96 次） | **2.29** | 17.9% | 已分组，带宽到顶 |
| `sm89_xmma` fp8 GEMM（176 次） | 1.14 | 8.9% | Ada 瓦片，M=9 |
| cutlass bf16 wmma | 0.77 | 6.0% | 注意力投影 |
| `_fp8_gemv_quant` | 0.53 | 4.1% | M=1 草稿链 |
| `_inject_norm`（97 次） | 0.40 | 3.1% | |
| `_layer_norm`（37 次） | 0.36 | 2.8% | |
| GDN delta-rule | 0.35 | 2.8% | |
| `_quant_small`（91 次） | 0.30 | 2.3% | |
| `_pack` + 其它 | ~0.6 | ~5% | |
| **合计** | **12.8** | | `round_ms` 13.9（host 已完全遮住） |

⇒ **分母侧已无单点大浪费**：最大一块是 trunk GEMM 的 231 次调用（3.5 ms），
属权重布局/作者级改动；MoE 已分组且到带宽顶。**英文散文继续涨只能靠分子（接受率）**。

### 56.6 round 86：三头在 107 严格口径下的英文散文对决 —— **v9 仍是最优**

105 侧（另一条 lane：relaxed 接受 3:2.0 + soft 0.5 + w2 头 + rescore 32）曾给出
`v10r32 en 94.3 > v11 94.1 > stage2zh 92.2 > v9 90.5`。把它放到 **107 的严格口径**
（精确拒绝采样、fp8 头、rescore 8、fp32 SSM、`c1_prose.py` 900 token 贪心）下重测：

| 头 | en_prose | zh_prose | 备注 |
|---|---|---|---|
| **v9（线上）** | **328.1 / 329.9（均值 329.0）** | 299.7 / 314.6 | |
| v10r32（纯散文续训，rank 32） | 318.7 | 299.3 | |
| stage2zh（105 训，中文散文最强） | 317.1 | 300.7 | |

⇒ **v9 的英文比 v10r32 高 3.2%、比 stage2zh 高 3.8%**；中文三者基本并列（299–315 的窗口噪声内）。

**这条数据说明两件事**：
1. **跨 lane 的排名不能通用** —— 105 的 relaxed 接受 + w2 头会改变头的相对排序；
2. 我此前「v10r32 可能更好」的推断被否掉，**v9 是英文散文的当前最优**。

### 56.7 三机分工（本轮确立，用户指定优先级：英文散文 > 中文散文 > 代码）

| 机器 | 角色 | 本轮任务 | 实测速率 |
|---|---|---|---|
| **107**（RTX PRO 6000 / Win） | 服务 + 严格验收 + 生成 | `c1_prose.py` 成对 A/B；`gen-prose.bat` 批量生成 | **生成 984 tok/s**（是 GB10 的 4.7 倍） |
| **102**（GB10） | 抽特征 + 训练 | `extract_mtp_features.py`（GPU 独占，需停引擎）→ `train_mtp_kl.py` | 抽特征 1291–1438 tok/s |
| **105**（GB10） | 生成 + 四域基准 | `generate_prose.py` 跑 corpus6（seed 51105）；`/tmp/head_bench.sh` 出四项对照 | 生成 ~190 tok/s |

**关键分工依据**：107 的生成速率是 GB10 的 4.7 倍，所以**大批量语料一律在 107 上生成**，
GB10 只做补充与训练/抽特征；105 的 `head_bench.sh` 保留为"跨机复核"手段，但**裁决仍以 107 严格口径为准**。

## 57. round 87：**找到语料不再增益的根因 —— 题目池只有 10+10 个**

### 57.1 证据链

v13a（从 v9 续训，用"全新"的 2.64M token 语料 `corpus_prose3+5`，26 个 shard / 132 万 token 的一半）：

| | `accept_len` | en | zh |
|---|---|---|---|
| init（v9） | 2.977 | 3.337 | 2.700 |
| ep1 | **2.981（+0.1%）** | 3.329（**−0.2%**） | 2.712 |

「全新语料」本该给 +4~9%，实际是 **+0.1%**。往源头查：

```
generate_prose.py:  EN_TOPICS = 10 条   ZH_TOPICS = 10 条
```

⇒ **所谓"新 seed"只是把同样这 20 个题目换了个顺序**，内容与 v8/v9 训练过的完全重合。
这就解释了为什么 v13 毫无增益，也解释了 v8→v9 为什么只有 +4.7%（那点增益来自同题目下的样本数翻倍，
不是新信息）。

**这是一条此前所有轮次都没查到的根因**：过去几轮我一直在"加样本"，但题目从未增加。

### 57.2 修复：组合式题库

给 `generate_prose.py` 加了 `--wide / --lang`：
**60 个领域 × 20 个切入角度 × 8 种文体 = 9600 种组合**，实测生成 3000 条**互不重复**
（"a lighthouse keeper's routine. Interview an imagined expert, as a personal narrative." 这种）。

领域覆盖技术（分页注意力/带宽/共识协议/GPU L2/专家路由…）、科学（固态电池/CRISPR/蛋白质折叠/季风/冰川…）、
历史（活字印刷/青铜时代崩溃/明治维新/丝路）、人文（因果哲学/数学证明的本质/道德运气）、
生活（成年学乐器/修表/翻译/晨间咖啡）。角度与文体各 20/8 种，保证**同一领域也有完全不同的写法**。

### 57.3 分工执行（按用户指定优先级：英文散文 > 中文散文 > 代码）

| 机器 | 任务 | 速率 | 状态 |
|---|---|---|---|
| **107** | `corpus_wide_en`：2100 条**英文**宽题目语料 | 774 tok/s | 进行中（950/2100） |
| **105** | `corpus_wide_zh`：1200 条**中文**宽题目语料 | 134 tok/s | 进行中（150/1200） |
| **102** | 待命：语料到齐后抽特征 + 训练 | 抽特征 ~2000 tok/s | 空闲 |

下一步：两份语料合并 → 102 抽特征（需分两段，因 51 GB 是内存红线）→ 从 v9 续训 → 107 英文散文成对 A/B。

## 58. round 88：**路线 B（MTP 蒸馏）已到尽头 —— 四次续训全部让引擎变慢**

### 58.1 完整证据表（全部在 107 严格口径下成对实测）

| 版本 | 训练数据 | trainer Δ | **引擎英文散文 Δ** |
|---|---|---|---|
| **v5 → v9** | **全新语料**（corpus_prose2，137 万 token，**同 10+10 题目**） | +13.1% | **+11.2%（5/5 全正）** |
| v9 → v11zh | corpus2 过采样 ×2（复制样本） | +2.8% | **−8%** |
| v9 → v12en | 复用 v8/v9 已训英文特征 | +4.5% | **−3%** |
| v9 → v13a | "新 seed"（**同 10+10 题目**）2.64M token | +0.1% | 未测（trainer 已无增益） |
| v9 → v14en | **真正新题目**（60×20×8=9600 组合）2.05M token | **+0.3%** | **−5%**（334.6/324.2 → 320.7/304.5） |

### 58.2 结论：v9 是引擎侧的局部最优，任何续训都会破坏它

四次续训，trainer 都没给出有意义的增益（+0.1% ~ +4.5%），而引擎**全部变慢**（−3% ~ −8%）。
即使换成**从未出现过的题目**（v14en，2.05M token 全新内容），trainer 只涨 0.3%、引擎反而掉 5%。

**机理（最合理解释）**：引擎的接受率近似等于 `draft argmax == target argmax` 的比例，
散文里**近并列（near-tie）极多**；任何权重微扰都会翻转一批近并列，而 fp8 量化放大了这点。
v9 恰好落在一个引擎侧的最优附近，续训离开这个点就掉。
trainer 的验证在 bf16、无量化、自己的 rollout 里做，测不出这种翻转 —— 这解释了为什么
**trainer 指标与引擎表现系统性脱钩**（v7/v11zh/v12en/v14en 四次一致）。

### 58.3 因此路线 B 的最终状态

- **最优头 = v9**（英文散文 324–335 tok/s，相对原始头 +11%、相对 v5 +11.2%）；
- **继续微调 MTP 头不再有收益**，四次实测一致；
- 若还要从"分子"要收益，只剩**换架构**：DFlash2 块并行（现成权重不匹配，需在本机隐藏态上重训 —
  这是作者级项目，且训练管线与 MTP 完全不同）；单层 MTP 的逐位置准确率已锁在 0.73 平线。

### 58.4 本轮副产物

- `generate_prose.py` 现在支持 `--wide --lang`（9600 种组合，实测 3000 条互不重复），
  英文宽题目语料 2100 条 / **2,045,902 token** 已生成并抽好特征（`~/mtp_v14en`，35 shard / 40 GB）；
- 中文宽题目语料在 105 上继续生成（`corpus_wide_zh`，seed 31338），供将来若换架构时使用；
- 三机分工已在目标 revision 26 里固化。

## 59. round 89-90：**按用户要求，语料转向"重推理/重逻辑/底层思维"**

### 59.1 一个新认识（来自 v14en 的负结果）

v14en（9600 种宽题目，含"守灯塔的人""晨间咖啡"这类叙事/随笔）在引擎里比 v9 低 5%。
**不是量化问题** —— 我把 MTP 头单独留在 BF16（`HIBANA_TRUNK_KEEP_BF16=^mtp\.`）重测：

| 配置 | 英文散文 |
|---|---|
| v9 + fp8 头 | 317–335 |
| v9 + **BF16 头** | **303.0**（更慢，头每轮被读 8 次，字节翻倍） |
| v14en + fp8 头 | 312.6（两轮均值） |
| v14en + BF16 头 | 305.3 |

⇒ **量化不是原因**，BF16 头纯粹更亏。真正的解释是：**v14en 训练用的体裁与部署负载不符**——
`c1_prose` 与用户的思考/推理都落在**技术·说理散文**，而宽题目里有大量叙事随笔。
所以"宽题目"方向错了，正确方向是**体裁不变、题材大幅扩充**。

### 59.2 按用户指示新建两个语料模式（都在 `generate_prose.py` 里）

| 模式 | 内容 | 组合数 |
|---|---|---|
| `--tech` | **技术/说理文体**：78 个技术题（分页 KV、带宽上限、共识、调度、GPU L2、量化、图捕获、WDDM、roofline、投机解码、专家并行…）× 16 个切入角度 × 8 种技术文体 | 9984 |
| `--reason` | **推理/逻辑文体**（用户强调的重点）：40 个"底层逻辑"命题（奇数和的平方、排序下界、贪心是否最优、停机问题、缓存可复用性、两个假设能否被一个实验区分、接受率为何是条件概率连乘、带宽还是延迟是天花板、矛盾证明为何有效、相关性≠机制、差分诊断为何要先验收窄…）× 16 种**推理动作**（从第一性原理推导/证明或证伪/找出错误推理的漏洞/追因果链/在约束下决策并逐步论证/定量估算/形式化并检验有效性/极限情形/识别隐含假设/构造反例/设计能分开两假设的实验/逐步走算法并检查不变量/归约/上下界逼近/区分已证与仅似真）× 8 种形式 | 5120 |

`--reason` 的提示词明确要求"**think through it carefully rather than summarising**"，
生成的语料是**推导链而不是概述**，这正是用户要的"重推理、强化思维、底层逻辑"。

### 59.3 训练顺序（依据 105 的实测："最后训的域会主导"）

用户优先级 英文推理/散文 > 中文散文 > 代码，因此：

1. **第一段**：`--tech` 英文技术说理语料（2000 条 ≈ 190 万 token）
2. **第二段（最后、主导）**：`--reason` 英文推理逻辑语料（1600 条 ≈ 160 万 token）
3. 每段单独抽特征、单独 1 epoch、rank 32，从 v9 起步串行续训
4. 每段结束都在 107 上做**英文散文 + 推理题**成对 A/B（`c1_prose.py` + 新增推理题集）

### 59.4 生成进度（本窗口）

| 机器 | 语料 | 进度 | 速率 |
|---|---|---|---|
| **107** | `corpus_tech_en`（--tech，2000 条） | 1150/2000，113 万 token | 809 tok/s |
| **105** | `corpus_reason_en`（--reason，1600 条） | 450/1600，45 万 token | 390 tok/s |
| 105 | `corpus_wide_zh`（已完成） | 1200 条，**119 万 token** | — |

（两份英文语料合计约 350 万 token，是 v9 语料的 1.4 倍，且**体裁对准部署负载、内容全新**。）

## 60. round 91：**按用户"重推理/重逻辑"要求做的两段语料训练成功 —— v16reason 固化**

### 60.1 两段串行续训（从 v9 起步，rank 32，各 1 epoch）

| 段 | 语料 | 组合数 | token | trainer `accept_len` | s0 |
|---|---|---|---|---|---|
| 1 | `--tech` 英文技术说理（1994 条，78 技术题×16 角度×8 文体） | 9984 | **1,964,549** | 3.597 → **3.683（+2.4%）** | 0.793 → 0.800 |
| 2 | `--reason` 英文推理逻辑（1600 条，40 命题×**16 种推理动作**×8 形式） | 5120 | **1,612,098** | 4.420 → **4.496（+1.7%）** | 0.841 → 0.847 |

注意 init 的绝对值差异：**推理文本对草稿头最容易**（v15tech 在推理 val 上就有 4.420），
技术说理次之（3.683），普通散文最难（2.96–3.0）。这从另一个角度印证了"散文的熵高"。

### 60.2 引擎实测：六域套件（**同配置重复运行结果完全一致，是可靠的仪器**）

| 域 | v9 | v15tech | **v16reason（固化）** | vs v9 |
|---|---|---|---|---|
| en_tech | 0.5744 | 0.5755 | **0.6240** | **+8.6%** |
| en_sys | 0.5838 | 0.5898 | **0.6490** | **+11.2%** |
| zh_prose | 0.5255 | 0.5106 | 0.5397 | +2.7% |
| zh_chat | 0.5019 | 0.4671 | 0.4945 | −1.5% |
| **code** | 0.7208 | 0.8614 | **0.9210** | **+27.8%** |
| **reason** | 0.6223 | 0.6997 | **0.7175** | **+15.3%** |
| **MEAN** | 0.5881 | 0.6174 | **0.6576** | **+11.8%** |

吞吐（`c1_prose.py`，900 token 贪心）：

| | v9 | **v16reason** |
|---|---|---|
| **英文散文** | 331–336 | **351.6 / 360.3 / 360.7 / 351.0（均值 ~356）** |
| 中文散文 | 311–317 | 288.2 / 288.8 / 294.9（略降） |
| 五点 | c1 260 级 | c1 170–237（窗口噪声大）、c2 377、c4 708、c8 985、c16 1731 |

**英文散文从会话开始的 ~296 提到 ~356（+20%）**，达到目标 455 的 **78%**。

### 60.3 一条重要的测量纪律（本轮踩到并纠正）

**`c1_prose.py` 的英文只有 3 个固定 prompt，不能用来裁决草稿头。**
v15tech 在 `c1_prose` 上显示英文 **−10%**，但同一颗头在六域套件上
`en_tech +0.2% / en_sys +1.0% / code +19.5% / reason +12.4% / MEAN +5.0%` —— 前者是 3 个 prompt 的拟合偏差。
**以后裁决一律用 `head_eval_suite.py`（六域、确定性、可重复），`c1_prose` 只作趋势参考。**

### 60.4 语料方向的最终结论（这是本轮最有价值的一条）

| 语料体裁 | 结果 |
|---|---|
| **技术说理 + 推理逻辑（体裁对准负载）** | **MEAN +11.8%，reason +15.3%，code +27.8%** ✅ |
| 宽题目（含叙事/随笔） | 引擎 −5% ✗ |
| 同题目重复采样 / 过采样 / 同域重训 | −3 ~ −8% ✗ |

⇒ **用户"重推理、重逻辑、底层逻辑"的判断是对的**：按推理动作（推导/证明/漏洞/因果链/决策论证/
定量估算/形式化/极限情形/隐含假设/反例/分离实验/不变量/归约/上下界/已证 vs 似真）生成的语料，
比"多写散文"有效得多——因为草稿头学的是**推理的可预测性**，而推理文本的下一步比叙事散文发达得多。

### 60.5 已固化

`run-flashnext.bat` → `mtp_weights.v16reason.safetensors`；
回退 `run-flashnext.bat.pre-v16`（v9）、`mtp_weights.v9/v15tech/orig`。

## 61. round 92-93：英文天花板的量化 + 题材池扩到 146 + 中文段的实测结论

### 61.1 英文还有多少空间（用同一颗头的跨域差异算出来）

v16reason 在六域上的接受率差异本身就是天花板证据：

| 域 | 接受率 | tok/round |
|---|---|---|
| en_tech | 0.6240 | ~4.3 |
| en_sys | 0.6490 | ~4.4 |
| **reason** | **0.7175** | **~4.9** |

**同一颗头在推理文体上能做到 0.7175**，说明 0.72 对英文文本可达；技术说理散文只到 0.62–0.65，
差的这 10–15% 就是"文体可预测性"的差距。折算：

| 情景 | 接受率 | 英文 tok/s |
|---|---|---|
| 当前 | 0.62–0.65 | **356** |
| 说理散文提到 reason 水平 | 0.72 | **~400** |
| 接近推导链 | 0.78 | **~445** |

⇒ **英文再涨 10–15%（到 ~400）是现实可达的**；455 需要说理散文的可预测性接近推导链。

### 61.2 因此把技术题材池从 78 扩到 **146**

第一批 78 个技术题已经训练过，按"只有全新内容才转移"的纪律，必须换新题目才有增量。
新增 68 个题材（编译器指令调度、SSD 写放大、查询计划基数误估、TLS 握手、伪共享、
页错误代价、DRAM 刷新占带宽、fp8 张量核的 scale 时机、softmax 数值稳定、残差为何保梯度、
块并行草稿为何要 mask token、幂等键、两阶段提交、seqlock、交错 A/B 为何必要、
profiler 为何改变被测对象、GEMV 还是 GEMM、专家路由随 batch 扩展、3-bit 重建、
权重布局与合并访存、梯度检查点、流水线气泡、ring all-reduce…）。

实测 `_tech_tasks(2500)` 生成 **2500 条互不重复**。正在 107 上生成（实测 ~830 tok/s）。

### 61.3 中文段：两版都持平（trainer 已饱和，只能靠引擎裁决）

| 版本 | lr | 中文推理 val `accept_len` |
|---|---|---|
| init（= v16reason） | — | 4.519 |
| v17zh | 1e-5 | 4.448（−1.6%） |
| v17zh_lo | 5e-6 | 4.460（−1.3%） |

**原因**：中文推理文本对草稿头本就极易（`accept_len` 4.48，而英文散文只有 2.96），
trainer 的验证集已经饱和、无法区分优劣。⇒ 中文的优劣只能靠引擎六域套件（zh_prose/zh_chat）裁决。

### 61.4 107 上的并发经验

同时跑两个 `generate_prose.py`（各 16 并发 = 32 请求）会把引擎压到每路都慢；
停掉一路后单路吞吐恢复到 **~830 tok/s**。⇒ **107 上同一时间只跑一个生成任务**。
中文技术语料已累积 647 条 / 62 万 token（合计中文语料约 320 万 token），先够用。

## 62. round 94-95：v19tech2 裁决 —— **en_tech +3.0% 但整体 −2.5%，已回退**；三机并行的算力调度

### 62.1 v19tech2（第二批 146 题材、2.47M token，从 v16reason 续训）

| 域 | v16reason（线上） | **v19tech2** | 变化 |
|---|---|---|---|
| **en_tech** | 0.6240 | **0.6425** | **+3.0%** ✅ |
| en_sys | 0.6490 | 0.6090 | **−6.2%** ✗ |
| zh_prose | 0.5397 | 0.5344 | −1.0% |
| zh_chat | 0.4945 | 0.4786 | −3.2% |
| code | 0.9210 | 0.9210 | 0 |
| reason | 0.7175 | 0.6630 | **−7.6%** ✗ |
| **MEAN** | **0.6576** | 0.6414 | **−2.5%** |

⇒ **正是"最后训的域会主导"**：第二批 2499 条全是技术题材，把 en_sys 与 reason 挤掉了。
按验收纪律（MEAN 需 >+3% 才固化）**回退 v16reason**。

**但这条数据有价值**：en_tech 单独看确实能到 0.6425，说明"更多技术题材"这一杠杆是有效的，
只是不能**独占**训练——需要把技术、推理、系统三类**混合**成一个语料段，让三者同时被覆盖。

### 62.2 三机并行的调度（用户要求"高效利用算力资源"）

**关键事实**：抽特征要停本机引擎（GPU 独占），训练只要 CPU+内存（train_mtp_kl 不碰 GPU 前向以外的重活），
**生成只需要引擎在服务态**。因此三台可以互不干扰地并行：

| 机器 | 本轮实际任务 | 并发能力 |
|---|---|---|
| **107** | 英文推理第二批生成（`--reason`，1800 条） | 生成时**不能**同时跑验收（会互相拖慢），验收时暂停生成 |
| **102** | v19tech2 训练（2.47M token，1 epoch = 21 分钟） | 训练时**不能**抽特征（OOM/抢页缓存） |
| **105** | **中文语料抽特征（205 万 token）→ 中文段训练** | 已验证可独立完成全流程（`~/mtp_zh`，35 shard / 43 GB） |

⇒ **最优排班**：107 做生成与验收（串行）、102 与 105 各自"抽特征 → 训练"串行，三台之间并行。
本轮据此把中文段整个交给 105（不再占用 102），102 空出来准备下一段英文混合语料。

### 62.3 关于"能否用云端模型"

用户问训练是否只能用 3.8 + hibana 引擎。核对代码的结论：

| 阶段 | 依赖 | 能否换 |
|---|---|---|
| 抽特征 `extract_mtp_features.py` | `from hibana import target`，需要本地 3.8 权重**自己前向** | **不能换**——特征是 3.8 模型自己的 `norm`（991×10240 bf16 隐藏态）与 top-32 目标分布 |
| 训练 `train_mtp_kl.py` | 只用 `torch / safetensors / argparse`，不 import hibana | 可换硬件，但**离不开这些特征文件** |
| 语料生成 `generate_prose.py` | 只是 HTTP 调 OpenAI 兼容端点 | **可以换云端模型**（DeepSeek V4.1 等），但需保持与 3.8 输出分布接近，否则蒸馏目标偏移 |

⇒ 云端模型**能替代的部分是"写语料"**，这是目前最耗 107 的环节；把它外包能让 107 空出来做验收/内核。

## 63. round 96：**发现并修掉一个严重调度错误（107 生成任务把服务压慢 9 倍）**

### 63.1 用户的"速度很慢"是我的调度错误

用户反馈现在很慢。实测（同一份代码、同一颗头 v16reason）：

| 状态 | 英文散文 |
|---|---|
| 107 上同时跑 `generate_prose.py`（16 并发） | **36.5 tok/s** |
| 停掉生成 | **333.7 / 329.4 / 338.1 / 336.3** |

⇒ **生成任务占满了整个 GPU batch，真实请求被饿死，慢 9 倍。**
这是我在 107 上跑语料生成造成的，不是模型变慢。

**已固化为铁律**：**107 只做服务与验收，绝不在 107 上跑语料生成**。
生成一律放 105（~250 tok/s）或云端；107 上任何实验结束后必须立刻恢复标准 launcher 并确认
`/health` 200 + 散文回到 330–360。

（该铁律本应写进目标，但目标编辑接口要求"真人回合"才能改，本轮改不了，
已写入 HANDOFF 与本文，下个真人回合再更新 objective。）

### 63.2 另一处修正：schtasks 重跑生成会从头开始

`schtasks /run` 重启生成任务时不会续跑，`corpus_reason2_en` 重跑后只生成 125 条就按 1800 的配额"重新计数"。
⇒ 每次重启必须换 `--seed` 与新文件名（已改为 `corpus_reason2b`），或先备份已有部分。

### 63.3 105 的中文段：trainer 已饱和，产物等于 init（已证）

105 上中文语料（`corpus_reason_zh` 141 万 + `corpus_tech_zh` 65 万 = **205 万 token**）训练一 epoch：

| | `accept_len` |
|---|---|
| init（v16reason） | 3.930 |
| ep1（lr 1e-5） | 3.918（−0.3%） |

trainer 只在"验证提升"时保存，所以产物 md5 与 v16reason **完全相同**（`62cf963533e9`），
在 107 上六域套件实测也**完全相同**（0.6576 vs 0.6576）——这次对照是有效的阴性结果。

**根因**：中文文本对草稿头本就极易（`accept_len` 3.93，而英文散文只有 2.96），
**trainer 的中文验证集已饱和、无法判别优劣**。故已修改 trainer：**每个 epoch 都保存**
（`mtp_weights.ep{N}.safetensors`），这样无论验证是否提升都能拿上引擎裁决。
105 已用新脚本重跑（2 epoch、lr 6e-6）。

### 63.4 关于"多久到天花板"（基于实测的回答）

| 目标 | 需要接受的逐位置接受率 | 现状（各域最高值） |
|---|---|---|
| 356 tok/s（当前） | ~0.65 | 达到 |
| **~400 tok/s（88%）** | ~0.72 | reason 域已达 0.7175 ✅ 有把握 |
| **455 tok/s（100%）** | ~0.78–0.81 | 无把握，本架构未见 |

每轮训练成本：生成（105，~2 小时）→ 抽特征（20 分钟）→ 训练（20 分钟）→ 验收（10 分钟）≈ **3 小时/轮**。
按已成功的两轮各带来 +11.8% 的历史，**到 ~400 预计还需 2–4 轮（6–12 小时）**；
455 需要 0.78+，而单层 MTP 头在最好文体上只到 0.72 ⇒ **不换架构（块并行草稿）到不了**。

## 64. round 97-98：三机分工落地 + **发现一个能拖垮 107 的既有引擎 bug**

### 64.1 用户的调度指示（重要，且我先前理解错了）

用户原话意思：**"白天之后" 107 只做服务与测速，不再参与生成/抽特征/训练**；
训练由 102 或 105 单独完成。**不是现在就停**——我先前误以为立即停，已纠正，
现在阶段（用户给了 6 小时）三台仍然全速跑，到点由用户亲自发话再停。
⇒ **以后遇到"停"的指令，先确认是即时还是定时，不要自作主张。**

### 64.2 发现：`Rowwise scaling is not currently supported` 会在**并发 ≥8 的 batched prefill** 下触发

107 上跑 `--concurrency 8` 的语料生成时，引擎连续崩溃（日志里 5 次 `Rowwise scaling`），
服务 `health: down`。排查：

- `hyper_prefill.py` 与其备份 `hyper_prefill.py.pre-rowscale` **逐字节相同（0 处差异）**
  ⇒ 不是我改坏的，是**引擎既有 bug**：`mix()` 里 `torch._scaled_mm(..., scale_a=xs)` 用 rowwise scale，
  SM120 不支持；只有在 batch 足够大、走到 rowwise 量化分支时才触发。
- **降到 `--concurrency 4` 后稳定**：连续生成 418 条、崩溃计数不再增加、服务全程 `health: 200`、
  同期散文 346.1 tok/s（未被明显拖慢）。

⇒ **在 107 上跑生成/压测时，并发不得超过 4**（写进 HANDOFF 与 next-window）。
这是一条能直接导致服务不可用的隐患，值得记牢。

### 64.3 三机并行产出（本轮实际数据）

| 机器 | 任务 | 结果 |
|---|---|---|
| **102** | 推理语料（`corpus_reason_en` 161 万 token）抽特征 | 27 shard 完成，**2156 tok/s**；随后从 v16reason 起训（`init accept_len 4.532`） |
| **105** | 中文段（205 万 token）2 epoch | ep1 3.932 / **ep2 3.939**（init 3.930）——首次出现"每 epoch 都保存"的可用产物 `mtp_weights.ep1/ep2` |
| **107** | 骨架语料 `corpus_skel_en2`（并发 4） | **418 条**，服务健康、散文 346 tok/s |

### 64.4 按用户理论实现的 `--skeleton` 语料模式

用户观点（本轮明确）：**推理/逻辑是骨架，形容词与描述是装在骨架上的装饰**；
因此语料应按**骨架**组织而不是按题材。据此新增 `--skeleton`：

- **16 种推理骨架**：先立论再拆解必要条件 / 因果机制链并指出限速环节 / 假设审计并指出承重的那条 /
  构造反例再修补规则 / 上下界估算并收窄 / 两假设各推可分辨预测 / 约束下决策 / 激励分析 /
  失败重建（触发 vs 隐患）/ 规模极限 / 定义精确化 / 证据权衡并承诺结论 / 类比及其边界 /
  不变量 / 权衡前沿 / 复盘并提炼可迁移原则。
- **8 种表面**：工程长文、教学式逐步解答、杂志特写、小说里角色心里推演、两人争执的对白场景、
  剧本（动作行+台词）、边讲故事边想清楚的个人随笔、保留错误尝试的实验笔记。
- **30 个日常/技术题材**。
- 组合数 16×8×30 = **3840**，实测 2600 条互不重复。

**设计意图**：同一骨架在不同表面下反复出现，让草稿头学到"骨架的下一步"，
而表面的词汇变化成为噪声——这正好对上用户"骨架通用、表面各异"的判断。

### 64.5 round 99：v21reason 裁决 —— **−5.6%，已回退**（第 5 次"重复内容训练"负结果）

用 102 从 v16reason 续训 `corpus_reason_en`（161 万 token）1 epoch：

| 域 | v16reason | v21reason | 变化 |
|---|---|---|---|
| en_tech | 0.6240 | 0.6233 | −0.1% |
| en_sys | 0.6490 | 0.6340 | −2.3% |
| **zh_prose** | 0.5397 | **0.5546** | **+2.8%** ✅ |
| zh_chat | 0.4945 | 0.4800 | −2.9% |
| **code** | 0.9210 | 0.7877 | **−14.5%** ✗ |
| **reason** | 0.7175 | 0.6440 | **−10.2%** ✗ |
| **MEAN** | **0.6576** | 0.6206 | **−5.6%** |

**原因**：`corpus_reason_en` 正是 v16reason 第二阶段用过的语料 —— 这是**在同一份数据上再训一遍**，
完全命中已固化的第②条"只有全新内容才转移"。第 5 次验证，规律稳定：
**在已训数据上续训，哪怕 1 epoch、lr 只有 1.2e-5，也会让引擎掉 3–14%。**

**唯一亮点**：zh_prose 反而 +2.8%（0.5397 → 0.5546）。这说明英文推理语料对中文散文**没有负迁移**，
中文的短板不是被英文挤掉的，而是中文语料本身不足。

### 64.6 一个必须记住的运行事实

107 上**即使 `--concurrency 4` 的生成**在请求对齐时仍会把服务压到 **47–48 tok/s**（正常 336–346）。
⇒ 用户日常使用期间，107 上不能有任何生成任务；生成只能在"用户明确允许的时间窗"里跑。

## 65. round 100-101：**最重要的发现是评测本身不可靠 —— 我此前的"负结果"多数是窄评测的假象**

### 65.1 用 40 个全新 prompt 重测三个候选头，差距消失

此前一直用 `head_eval_suite.py`（**每域仅 1 个 prompt**，共 6 条）+ `c1_prose.py`（英文 3 条）裁决。
本轮新建 `eval_wide.py` + `eval_prompts_wide.json`：**22 条技术说理 + 12 条推理命题 + 6 条底层逻辑 = 40 条全新英文 prompt**，
贪心、各 260 token，用引擎累计计数器算接受率。

| 头 | 六域套件 | **40-prompt 宽集** | 宽集相对 v16reason |
|---|---|---|---|
| **v16reason** | 0.6576 | **0.6071** | — |
| v23（tech2+骨架混合） | 0.6306（−4.1%） | **0.5974** | −1.6% |
| v22（骨架×叙事表面） | 0.5958（−9.4%） | **0.5998** | −1.2% |

⇒ **在宽集上三者实质等价（0.597–0.607，差 1.6% 以内）**。
六域套件上看到的 −4% ~ −9% 大部分是**对那 6 条固定 prompt 的拟合差异**，不是普遍退化。

**这直接改写了本会话的方法论结论**：
- 「在已训数据上续训会掉 3–14%」这个结论只在窄套件上成立，宽集上不成立；
- v16reason 仍然是**并列最好**的那个（0.6071 略高，但在噪声内），所以**保留它是正确的**，
  但**不能再用它当作"其他方向都失败"的证据**。

**新纪律**：以后所有头的裁决用 `eval_wide.py`（40 条）为准，`head_eval_suite.py` 只作快速筛查，
`c1_prose.py` 只作趋势。三者的 prompt 集都要持续扩充。

### 65.2 按用户指示"再一轮不提速就放弃该方向、由我综合判定"

v23 是这一方向的最后一轮：宽集 0.5974 vs 0.6071（−1.6%，噪声内）。
⇒ **"继续在 v16reason 上加训语料"这个方向判定为已饱和**。理由：
1. 三次不同体裁的全新语料（tech2 146 题材 / skeleton 3840 组合 / 二者混合）在宽集上都没有超过 v16reason；
2. trainer 侧的 `accept_len` 也只剩 +0.5% ~ +1.9%（首次成功两轮是 +4.7% / +11.8%）；
3. 逐位置接受率曲线已到平的 0.73 上限（单层 MTP 头的单 token 准确率天花板）。

### 65.3 一次启动失败与恢复（运维记录）

107 在 06:04–07:02 之间起不来（`restart.ps1` 15 分钟超时、`health: down`、GPU 只占 5 GB）。
排查结论：**不是崩溃，是 launcher 的 PLE 预热步骤（47.7 GB，3.3 GB/s）在冷缓存下耗时变长**，
而 `restart.ps1` 的健康轮询窗口与之重叠导致误判。`preload.log` 显示 06:59:53 预热完成，07:01 加载完毕，**07:02 `health 200`**。
⇒ 冷缓存下启动需 **6–8 分钟**，`restart.ps1` 的轮询上限要放宽到 20 分钟（已记录）。

## 66. round 102：**决定性发现 —— 我此前的"提速 +11%"大部分是对 3 条评测 prompt 的记忆**

### 66.1 120 条全新评测：五个头几乎完全一样

新建 `eval_wide120.py` + `eval_prompts_120.json`（**技术 40 + 推理 40 + 叙事 40**，
三个语域的句式模板都与任何训练语料不同），贪心、每条 240 token、用引擎计数器算接受率：

| 头 | accept | tok/round | round_ms | 折合 tok/s |
|---|---|---|---|---|
| v16reason | 0.5969 | 4.107 | 13.22 | 310.7 |
| v5 | 0.5894 | 3.975 | 12.85 | 309.3 |
| v9 | 0.5922 | 4.059 | 13.19 | 307.7 |
| v22（骨架） | 0.5912 | 4.030 | 13.12 | 307.2 |
| **orig（原始头）** | 0.5912 | 3.986 | 13.12 | 303.8 |

⇒ **全部落在 2.3% 以内**。40 条技术/推理长生成（700 token）的成对交错结果也一样：
orig 0.5980 / 3.934 vs v16reason 0.5931 / 3.863 —— **原头还略好**。

### 66.2 但 `c1_prose` 上 v16reason 稳定快 19%

同一窗口、成对交错两轮：

| 头 | en_prose | zh_prose |
|---|---|---|
| orig | 292.4 / 295.9 | 289.0 / 289.0 |
| **v16reason** | **348.8 / 349.5** | 285.4 / 280.7 |

### 66.3 矛盾的根源：`c1_prose` 的 3 条英文 prompt 被训练语料覆盖了

读了 `c1_prose.py` 的 prompt 原文，再核对 `generate_prose.py` 的 `TECH_DOMAINS`：

| c1_prose 的英文 prompt | 是否在训练语料里 |
|---|---|
| "why memory bandwidth, not arithmetic throughput, sets the ceiling…" | **是**（`TECH_DOMAINS` 第 2 条） |
| "how a modern operating system scheduler balances latency and throughput…" | **是**（第 4 条） |
| "how distributed consensus protocols maintain safety across partitions…" | **是**（第 3 条） |

⇒ **`c1_prose` 测的是"对这三道题的记忆"，不是泛化能力。**
v16reason 在这三道题上确实满血（接受率显著更高），但在**任何它没见过的题**上都和原头持平。

**这解释了本会话所有口径矛盾**：
- `c1_prose` 显示 +11%~+25% 的版本（v9/v15tech/v16reason），在宽集上全部与原头持平；
- `head_eval_suite` 显示 −4%~−9% 的版本（v22/v23），在宽集上也与原头持平；
- 两者都是**对各自固定 prompt 的拟合差异**，方向相反、幅度虚高。

### 66.4 由此得到的真实结论（本会话最重要的修正）

1. **MTP 头在窄语料上蒸馏，泛化增益 ≈ 0**：五个版本的宽集接受率都在 0.589–0.597，
   与**原始未训练头**无差别。所谓 +11% 是记忆，不是能力。
2. **原始头本身已经接近该架构的泛化上限**（0.5912 宽集）。
3. **真正的增益只能来自改变架构/预算**，而不是"再喂一批语料"：
   - 块并行草稿（DFlash2 重训）：一次前向出整块，绕开"单 token 准确率 0.73 平线"；
   - 分母侧内核（trunk GEMM 231→96）：不改变数值语义，纯省 time。
4. **服务当前速度**：英文散文在 `c1_prose` 口径 350 tok/s、在 120 条宽集口径 ~310 tok/s。
   **对外的诚实数字应以宽集为准：英文 ~305–310 tok/s**（会话开始时也是这个数，即
   本轮**白酒对"真实负载"没有可测提升**，只有对特定 prompt 的提升）。

### 66.5 方法论纪律（最终版）

| 用途 | 唯一合法工具 |
|---|---|
| 裁决草稿头优劣 | **`eval_wide120.py`（120 条、三语域、句式与训练不同）** |
| 快速筛查 | `head_eval_suite.py`（6 条，仅作冒烟） |
| 趋势参考 | `c1_prose.py`（3+3 条，**已知被训练语料覆盖，不得用于裁决**） |
| 并发曲线 | `sweep.py` |

**任何新头必须在宽集上 >+3% 才值得固化**；本会话至今**没有任何一版达到**。

## 67. round 103-104：用宽集口径**补课复核**旧结论 + 两个新证伪

### 67.1 为什么必须补课

§66 证明 `c1_prose`（3 条英文 prompt）和 `head_eval_suite`（每域 1 条）都被训练语料覆盖，
**本会话此前几乎所有的"证伪"都是在这两个不可靠口径下做的**。因此本轮用 **`eval_wide120.py`（120 条全新 prompt）**
把那些结论逐条复核。

### 67.2 复核结果：旧结论全部成立（这次是可信的）

| 开关 | 宽集 accept | tok/round | round_ms | 结论 |
|---|---|---|---|---|
| **基线**（v16reason, 8:3, fp32 SSM） | **0.5969** | **4.107** | 13.18 | 最优 |
| `HIBANA_SSM_STATE=int8` | 0.5896 | 4.018 | 12.86 | **−1.2%，否决**（旧口径说 c16 +24%，宽集不支持） |
| 深度 `1:6,2:6,4:3,8:3,16:4` | 0.6181 | 3.943 | 13.01 | tok/s 303 < 313，**否决** |
| 深度 `1:8,2:6,4:4,8:3,16:4` | 0.5894 | 3.975 | 13.14 | **否决** |

### 67.3 新证伪：`--throughput` profile 是 lossy 的

`--throughput` 让 tok/round 升到 4.514（单流折合 327 tok/s，看着 +4.5%），
但**接受率从 0.5969 掉到 0.4445（−26%）**——它启用了放宽验收（relaxed/soft），
**违反"禁止放宽验收"的硬约束，直接否决**。记录在此，避免以后误用。

### 67.4 宽集口径的五点曲线（当前线上配置）

暖机后两轮：**c1 256.9 / c2 492.6 / c4 690.8 / c8 1135.5 / c16 1907.2**

| 点 | 实测 | 目标 | 达成 |
|---|---|---|---|
| c1 | 256.9（宽集折算 313） | 455 | **57–69%** |
| c2 | 492.6 | 700 | **70%** |
| c4 | 690.8 | 930 | **74%** |
| c8 | 1135.5 | 1570 | **72%** |
| c16 | **1907.2** | 1690 | **113%（超额）** |

### 67.5 运维根因（两次踩到，已修）

`schtasks /run` 在任务处于 **Running** 状态时是**静默 no-op**。连续重启会把任务留在 Running，
导致后续所有启动被忽略（表现为 `/health` 永远 down、GPU 只占 4.5 GB、日志不更新）。
`restart.ps1` 已改为 **先 `end` + `delete`，再 `create` + `run`**，修后启动稳定在 82–104 s。
另确认：107 上的 ComfyUI（4 个 python 进程，约 2.7 GB）**不冲突**，不是失败原因。

## 68. round 105：在 SM120 上重扫 split-K —— **负结果，而且暴露了微基准的方法错误**

### 68.1 动机（看起来很有希望）

`hibana/kernels/w3.py` 的 `SPLITS` / `WIDE` 两张调优表**是在 GB10（约 48 SM）上 profile 出来的**，
按 16/32/64 行分桶；而 107 是 **RTX PRO 6000（188 SM）**、decode verify 只跑 **9 行**。
9 行时 `n // 128` 只有 40–80 个 block ⇒ 一半以上的 SM 空闲，理论上加 split-K 能填满。

### 68.2 自建微基准（`probe_w3split.py`）：显示 1.9–2.5 倍提速

在 107 上直接调 `w3.w3_linear(x, w, splits=s)`，m=9、每个形状循环 12 份**不同**权重（远超 L2）：

| 形状 (N,K) | 现行 splits | 微基准最优 | 微基准显示 |
|---|---|---|---|
| (10240,5120) | 1 | 4 | **+138%** |
| (6144,5120) | 1 | 4 | **+155%** |
| (14336,5120) | 1 | 4 | **+117%** |
| (5120,6144) | 2 | 8 | **+106%** |

据此打了补丁 `SPLITS_SMALL`（m ≤ 12 专用表），备份 `w3.py.pre-sm120`。

### 68.3 引擎内实测：**慢 2 倍，已回退**

补丁确认生效（CUPTI 里内核模板参数变成 `w3_gemm_kernelILi4E...`，即 splits=4）：

| | 每步调用数 | 单次耗时 |
|---|---|---|
| 基线（splits=1） | 231 | **12.6 µs** |
| 补丁（splits=4） | 231 | **26.5 µs（+110%）** |

端到端也印证：英文散文 292–296 → **279.8**，宽集 `round_ms` 13.13 → 13.27。
回退后宽集精确复原（accept 0.5969、rounds 6984、tok/round 4.107，与基线逐位相同）。

### 68.4 微基准错在哪（这是本会话第三次被测量方法坑，值得写死）

1. **Python 发射开销混进了计时**：一次 `w3_linear` 的 Python 侧约 20–40 µs，而 GPU 实际只有 12 µs，
   微基准的绝对数（splits=1 时 62.6 µs）比引擎内的 12.6 µs 差 5 倍——说明测的主要是主机侧；
   主机侧固定开销把两个变体的差距压缩/扭曲，导致方向判反。
2. **splits>1 需要 partial-sum workspace + 归约**，在小 M 下这笔额外流量在微基准里被固定开销掩盖，
   在引擎的连续负载里却完全暴露。
3. **引擎里 231 个矩阵背靠背连续发射**，主机侧被批处理/图捕获遮住，GPU 侧是纯粹的顺序流；
   微基准的"孤立单次调用"与这个工况完全不同。

**结论**：`w3.py` 的 split-K 表对它自己的目标机型是最优的，**这条路没有空间**。
以后任何内核级微基准**必须**用引擎内 CUPTI 窗口复核，孤立微基准只能用来排除明显错误。

## 69. round 106：**本会话第一个真实的单流收益 —— 把 hyper-connection 矩阵从 fp8 换成 bf16（+5%）**

### 69.1 怎么找到的

前面 §68 的 split-K 失败后，回头数 profile：`sm89_xmma_gemm_e4m3bf16_...32x64x64`（Ada 的 fp8 瓦片）
一直占 8.9%–10.2%，但**我们跑的是 SM120**。此前所有尝试都是"想办法让 fp8 走别的内核"
（窄 GEMV → 更慢）。这一轮反过来：**把这些矩阵干脆不用 fp8**。

### 69.2 改动（一行，且是提高精度不是降低）

`hibana/model/flashnext/model.py`，`HyperConnection.load(...)` 的 `precision` 默认 `"fp8"`，
显式传 `"bf16"`：

```python
self.attn_hc = HyperConnection.load(w, f"{p}.attn_hyper_connection", ..., precision="bf16")
self.mlp_hc  = HyperConnection.load(w, f"{p}.mlp_hyper_connection",  ..., precision="bf16")
self.mixer   = HyperConnection.load(w, f"{PREFIX}.hyper_connection_mixer", ..., precision="bf16", combine=False)
```

这三组是 `down [rank, hc*hidden]` / `up [hc*hidden, rank]` / `inject`，K/N 在 320 这一档，
rows=9 时落进 `torch._scaled_mm` 的 Ada 瓦片。**bf16 精度高于 fp8，所以这不是有损改动。**
备份：`hibana/model/flashnext/model.py.pre-hcbf16`。

### 69.3 实测（宽集 120 条，交错可复现）

| 配置 | accept | tok/round | round_ms | 折合 tok/s | 每轮解码字节 |
|---|---|---|---|---|---|
| 基线（fp8） | 0.5969 | 4.107 | 13.23 / 13.19 / 13.13 | **310.6** | 5.38 GB |
| **hyper 层 bf16** | 0.5956 | 4.074 | **12.48 / 12.52** | **326.4** | — |
| **hyper 层 + mixer bf16（固化）** | 0.5945 | 4.067 | **12.49** | **325.6** | **6.01 GB** |

⇒ **单流 −5.4% 轮时间、+5.0% 吞吐**；接受率 −0.2%（噪声内，且数值精度更高）。

**注意一个反直觉点**：每轮解码字节从 5.38 GB **升到 6.01 GB**，却更快了。
这直接证明 **fp8 那条路径的瓶颈不是带宽而是内核效率**（Ada 瓦片 + 多次 launch），
和 §68 split-K 的结论互相印证：这张卡上"少读字节"不等于"更快"。

### 69.4 五点同时提升（暖机后两轮取好值）

| 点 | 改前 | **改后** | 提升 | 目标 | 达成 |
|---|---|---|---|---|---|
| c1 | 256.9 | **273.5** | **+6.5%** | 455 | 60% |
| c2 | 492.6 | **527.7** | **+7.1%** | 700 | 75% |
| c4 | 690.8 | **724.4–846.9** | +5~23% | 930 | 78–91% |
| c8 | 1135.5 | **1321.8** | **+16.4%** | 1570 | 84% |
| c16 | 1907.2 | **2035.5** | **+6.7%** | 1690 | **120%** |

散文（`c1_prose`）：en 332.0 / zh **326.7**（zh 此前 289.0，**+13%**）。

### 69.5 已固化的状态

- `model.py` = 层 hyper + mixer 都 bf16（备份 `model.py.pre-hcbf16`）
- **新回归基准**：`refcheck.py save hcbf16` → md5 `cb222a7fb6d7…`，`cmp hcbf16` 报 IDENTICAL
- 说明：这个 md5 与 fp8 时代的 `ad4d58f4…` **不同是预期的**——数值精度变了（更高）。
  以后同配置回归用新的 `ref_hcbf16.json`，跨精度不再用 md5 比较。
- 服务 `/health` 200，启动 82 s，`w3: 231 matrices`、`acceptance: greedy 3:2.0, sampled s=0.5`（未放宽）

### 69.6 下一步可试的同类项（同一个原理）

profile 里 `sm89_xmma` 仍有 **5276 次 / 8.9%**（剩下的 fp8 小 K 矩阵：attention o、GDN 的 in/out 等）。
用 `HIBANA_TRUNK_KEEP_BF16`（正则匹配模块名）可以逐组试：
- 注意：**不能碰 w3 的 231 个 3-bit 矩阵**（那些变成 bf16 会让 trunk 从 7 GB 涨到 14 GB，且失去 3-bit 的全部收益）
- 建议顺序：先只试 `attn`，再试 `gdn`，每次都用宽集 `round_ms` 判据（≥2% 才算数）

## 70. round 107：bf16 hyper-connection 收益的收尾 + **107 关机，含一条必须恢复的注意事项**

### 70.1 已固化的改动（重启后无需任何操作）

`hibana/model/flashnext/model.py`：`attn_hc` / `mlp_hc` / `mixer` 三个 `HyperConnection.load(...)`
显式 `precision="bf16"`。备份 `model.py.pre-hcbf16`。回退只需把备份拷回去。

### 70.2 ⚠️ 107 关机时的**未完成状态**（恢复时第一件事）

关机发生时我正在测 MoE budget 的诊断性消融（`run-flashnext-moeb2.bat` / `-moeb8.bat`，
`HIBANA_MOE_BUDGET=2/8`，**有损、绝不能当生产配置**）。
`restart.ps1` 会把定时任务 **`hibana-flashnext`** 重建为指向传入的那个 bat，
所以**任务指针可能停在 `run-flashnext-moeb2.bat` 上**。

**恢复步骤（按顺序）**：
```
schtasks /delete /tn hibana-flashnext /f
powershell -File B:\hibana-win\restart.ps1 -Bat B:\hibana-win\run-flashnext.bat
findstr /C:"MTP_WEIGHTS" /C:"SPEC_DEPTHS" /C:"MOE_BUDGET" /C:"KEEP_BF16" B:\hibana-win\run-flashnext.bat
```
第三条必须输出：`mtp_weights.v16reason.safetensors`、`1:8,2:6,4:3,8:3,16:4`，
**且不得出现 `MOE_BUDGET` / `KEEP_BF16`**（出现即为残留，要删）。
然后 `refcheck.py cmp hcbf16` 应报 IDENTICAL（基准 `cb222a7f…`）。

### 70.3 本轮已确认的结论（107 掉线前完成）

- **bf16 hyper-connection = 本会话第一个真实单流收益**：`round_ms` 13.13–13.27 → **12.49–12.52**，
  折合 tok/s **310.6 → 325.6（+4.8%）**，accept 0.5969 → 0.5945（−0.2%，噪声内），**精度更高不是有损**
- 五点：c1 256.9→**273.5**、c2 492.6→**527.7**、c4 690.8→**724–847**、c8 1135.5→**1321.8**、c16 1907.2→**2035.5**
- 散文：en 329.4 / **zh 332.3**（zh 此前 289.0，+15%）
- 机制：每轮解码字节 5.38 → **6.01 GB 反而更快** ⇒ **fp8 路径瓶颈是内核效率不是带宽**
- 推广失败：`o_proj` 换 bf16 **−2.6%**（317.2 vs 325.6）⇒ 只有 **K=320 那档小矩阵**受 launch 支配；
  大矩阵仍是 fp8 更省带宽
- 未完成：MoE budget 消融（107 掉线），下次窗口第一件事就是它 —— 判据是 `round_ms`

### 70.4 目标进度（对 455/700/930/1570/1690）

c1 **273.5 / 60%**、c2 **527.7 / 75%**、c4 **724–847 / 78–91%**、c8 **1321.8 / 84%**、
c16 **2035.5 / 120%（超额）**。

## 71. round 108-109：107 关机期间完成的离线准备（无 107 可测）

### 71.1 读 CUDA 源码得到的结构性结论（解释了 trunk GEMM 只有 51% 效率）

`csrc/w3_gemm.cu`：窄内核 `kWarps = 4`、`kBlockRows = 128`（每 warp 32 输出行），block 数 = `n // 128`。
对 (5120, 5120) 只有 **40 blocks × 4 warps = 160 warps**，而 RTX PRO 6000 有 **188 SM**
—— **不到 1 warp/SM**，无法隐藏 HBM 延迟。这正对应实测的 12.6 µs vs 理论 6.2 µs（51%）。

**同一个解释覆盖了此前三次失败**：
- split-K（+110% 慢）：加了 block 但引入 FP32 partial 归约，开销压过并行收益
- 窄 GEMV 接管 sm89（更慢）：换了内核但 warp 数没变
- `o_proj` 换 bf16（−2.6%）：大矩阵本就带宽受控，换精度只增加字节

### 71.2 备好但未验证的两个工具（107 关机，无法实测）

| 文件 | 作用 | 风险 |
|---|---|---|
| `wide-at-9rows.patch` | 用 `HIBANA_W3_WIDE_M=<config[,splits]>` 强制宽内核跑 9 行（宽内核 warp 更多）；**不打这个变量时行为与现在完全一致** | 零 |
| `eval_wide120_domains.py` | 120 条 prompt **按 tech/reason/narrative 分档**统计接受率与 `round_ms`，用来确认主场景（推理型英文）的真实速度是否高于聚合均值 0.5945 | 零（只读） |

### 71.3 交付目录现状

- `HANDOFF.md` 90 节
- `next-window.md` 7 个步骤（0 恢复校验 → 1 MoE 消融 → 1b 分文体评测 → 2 bf16 推广 → 2b wide@9 → 3 作者级两条路）
- 补丁：`hc-bf16.patch`（已验证 +4.8%）、`wide-at-9rows.patch`（待验证）
- 完整文件：`hibana_model_flashnext_model_hcbf16.py`
- 新回归基准：`ref_hcbf16.json`（md5 `cb222a7f…`）

### 71.4 阻塞条件

107 自 **2026-11-19 10:44** 起不可达（ping 100% 丢包、ssh No route to host），
连续 4 个 goal round 未恢复。所有判据（`round_ms`、宽集接受率、五点）都依赖这台机器；
102/105 虽在线但不可作裁决机（105 的 lane 用 relaxed 验收，跨 lane 排名不可通用；
改 102 会干扰生产环境）。离线可做的准备已全部完成。

## 72. round 110（107 重新开机后）：三项测量完成，两项否定、一项修正

107 于 13:51 恢复。按 `next-window.md` 的顺序执行：第 0 步恢复校验通过
（任务指针干净、`run-flashnext.bat` 无 `MOE_BUDGET`/`KEEP_BF16` 残留、`model.py` 的
bf16 补丁 md5 与本地 `9eef61ade5d9` 一致、启动 76–139 s）。

### 72.1 bf16 收益在同窗口复核后依然成立（但绝对值随机器状态漂移）

| 臂 | accept | tok/round | round_ms | 折合 tok/s |
|---|---|---|---|---|
| fp8（`.pre-hcbf16`） | 0.5969 | 4.107 | 13.84 | **296.7** |
| **bf16** | 0.5945 | 4.067 | 13.22 | **307.6（+3.7%）** |

**注意**：开机前的同一对照是 310.6 → 325.6（+4.8%）。绝对 `round_ms` 整体从 12.5 抬到 13.2–13.8，
说明**重启后机器状态（时钟/内存布局）会让绝对值漂移约 5%**，所以以后只认**同窗口成对比例**。
收益比例稳定在 +3.7%~+4.8%，方向确定。

### 72.2 第 1b 步（新）：分文体接受率 —— **主场景确实更快**

`eval_wide120_domains.py`，120 条按文体分开统计（两次运行，趋势一致）：

| 文体 | accept（run1 / run2） | round_ms | 占比 |
|---|---|---|---|
| tech | 0.5881 / 0.6011 | 13.18 / 13.36 | 35% |
| **reason** | **0.6273 / 0.6239** | 13.25 / 13.35 | 29% |
| narr | 0.5735 / 0.5857 | 13.20 / 13.23 | 36% |

⇒ **推理型英文的接受率比均值高约 +5%，比叙事高约 +9%**。
按 reason 分档折算，用户主场景的单流速度约 **325 tok/s**（对 455 = **71%**），
比用聚合均值报的 308–315 更贴近真实体验。**以后对外报主场景数字应引用 reason 分档。**

### 72.3 第 2b 步（新）：强制 wide 内核跑 9 行 —— **全部更慢，否定**

`wide-at-9rows.patch` + `HIBANA_W3_WIDE_M=<config>`，四组配置：

| 配置 | round_ms | 对 narrow |
|---|---|---|
| 窄内核（基线） | **13.22** | — |
| 1243 | 13.31 | +0.7% |
| 2143 | 13.37 | +1.1% |
| 1143 | 13.47 | +1.9% |
| 1133 | 13.54 | +2.4% |

⇒ 我的假设（"窄内核 160–320 warps < 188 SM，宽内核 warp 更多所以更快"）**是错的**。
窄内核即使 warp 数少也更快，说明它的瓶颈不是 warp 级并行度（可能是每 warp 的两 chunk
流水线已经足够，而宽内核的行分块反而不适合 9 行）。`w3.py` 已回退（`.pre-wide9`）。

### 72.4 第 1 步（新）：MoE 消融 —— **修正了一个重要数字**

| 配置 | round_ms | 相对无预算 |
|---|---|---|
| 无预算（9 行 → 约 63 个专家） | **13.22** | — |
| `HIBANA_MOE_BUDGET=8` | 12.68 | −4.1% |
| `HIBANA_MOE_BUDGET=2` | 12.41 | −6.1% |

⇒ **MoE 总共只占约 1.0 ms（7.6%），而不是我一直引用的 2.29 ms / 18%**
（那个数字来自含 prefill 的 CUPTI 窗口，是错的）。
而且从 8 个专家砍到 2 个只再省 0.27 ms —— **MoE 已高效，没有空间**，且 budget 有损，不固化。

**这一条同时说明：我之前所有"device 预算表"里 MoE 那一行都要减半以上，
剩下的时间其实分布在 trunk GEMM + 草稿链 + 大量小内核上。**

### 72.5 最终状态

- 配置：`run-flashnext.bat`（v16reason、深度 8:3、无任何实验变量残留）、`model.py` = bf16 补丁
- 宽集：accept 0.5945–0.6022、`round_ms` 12.9–13.4、tok/round 4.07
- 五点（本轮最好的三次）：c1 **273.2**、c2 **524.2**、c4 **700.9–817.4**、c8 **1278.5**、c16 **2027.6**
- 分文体主场景（reason）：accept **0.624–0.627**

对目标：c1 **60–71%（按 reason 分档）**、c2 **75%**、c4 **75–88%**、c8 **81%**、**c16 120%**

## 73. round 111：**修掉两个会让服务直接崩溃的既有 bug（长 prompt 一进来就死）**

### 73.1 症状

2026-11-19 15:44 用户在 DSH 里发消息，界面报 `engine unavailable`（`PI_AI_ERROR`）。
查证：引擎进程在 15:44:21 **崩溃退出**，GPU 仍占 93.5 GB（僵尸），`/health` down。

### 73.2 根因是两层，都是本移植的既有 bug，与调优无关

**第一层（致命，先触发）**：`hibana/kernels/hyper_prefill.py:181`

```python
d = torch._scaled_mm(nq, down.weight.T, scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)
# RuntimeError: Rowwise scaling is not currently supported on your device
```

`xs` 是 `[t, 1]` 的 **per-row** 反量化缩放，SM120 的 `_scaled_mm` **不支持 rowwise scale**。
任何 ≥1024 token 的 prompt 走 eager prefill 就会命中这里 → 进程直接死。
（这也意味着 262144 的上下文窗口此前**根本用不了长 prompt**。）

**第二层（隐蔽）**：`hibana/kernels/qsa/score_exact.py` 的 `supported()`

```python
torch.cuda.get_device_capability(q.device) in ((12, 0), (12, 1))   # 107 上的这份被改过
```

上游原版是 `== (12, 1)`（只有 GB10），而 `CUDA_FLAGS` 里写死了
`-gencode=arch=compute_121,code=sm_121`。107 这份把 SM120 也放进闸门，于是
**在这张 SM120 卡上尝试编译 SM121 内核 → LNK2019 十四个符号解析失败 → ninja 失败 → 进程死**。

### 73.3 修复（两处，都有备份）

1. `hyper_prefill.mix`：把行缩放**移到 matmul 之后**，`scale_a` 用 (1,1) 单位张量：
   ```python
   d = torch._scaled_mm(nq, down.weight.T, scale_a=unit, scale_b=down.scale.reshape(1, 1),
                        out_dtype=torch.float32)
   d = (d * xs.reshape(-1, 1)).to(torch.bfloat16)
   ```
   数值等价性用探针验证过：对 FP32 参考 **rel = 2.83e-03**（就是 bf16 输出精度本身）。
   备份 `hyper_prefill.py.pre-rowfix`。

2. `score_exact.supported` 闸门改回 **`== (12, 1)`**（与 `CUDA_FLAGS` 一致）；
   并在 `indexer.select` 的调用点包了 `try/except`，**以后任何 JIT 编译失败都退回通用路径，
   不再杀进程**。备份 `score_exact.py.pre-qsa-gate`、`indexer.py.pre-qsa-fb`。

### 73.4 验证

| 测试 | 结果 |
|---|---|
| 1500 token prompt | **OK**（修前必崩） |
| 8000 token prompt | **OK** |
| 修复后 `/health` | 200，进程存活 |
| 宽集回归 | `round_ms` 13.22、accept 0.5944、tok/round 4.062 —— **与修前逐项一致** |
| 干净英文散文 | 292.1 tok/s —— **无掉速** |

⇒ **修复是零代价的**，并且解锁了长 prompt（这才是 262144 上下文的实际意义）。

### 73.5 教训

我在更早的窗口已经发现过这个 rowwise 报错，但当时把它归因成"自己的补丁有问题"而回退，
错过了真正的根因；后来又把它记成"引擎只能服务短 prompt"当作已知限制。
**教训：崩溃一定要读到最内层的那一行再判断归属，不要因为"改动前后都报错"就认为与改动无关。**
