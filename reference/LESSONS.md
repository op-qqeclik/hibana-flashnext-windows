# 把 Linux/GB10 推理引擎移植到 Windows + RTX PRO 6000 的经验（2026-11-17）

> 适用面：任何「Linux + NVIDIA + 统一内存」上写过的高性能推理引擎，要搬到「Windows + 独立显存 + 更快的卡」。
> 全部结论都来自 hibana Flash-Next 在 107 上的实测，数字可复现。

## 一、先量，再改（最重要的一条）

1. **不要用「带宽不够」解释慢**。实测这张卡 bulk copy 1282 GB/s、read 1173 GB/s，而 hibana 的 3 比特 MoE 内核单跑
   **1536–1608 GB/s**（86% 标称）。内核效率早就达标，慢在别处。
2. **engine 的 wall 时间 ≠ GPU 忙碌时间**。用 CUDA event 跨度包住一步，会把「主机等 GPU 的空隙」也算成 gpu 时间（本次一开始就误判过）。
   要的是 **torch.profiler 的设备时间**（真忙碌）与 **self_cpu_time**（真主机）两张表，两者之差才是空隙。
3. **累计计数器会骗人**。`spec_stats` 的 `tokens_per_round` 是会话平均，混了单流和 16 并发；要按 load 取 `/metrics` 增量。
4. **探针要能在负载起来后再开窗**。`HIBANA_PROFILE_STEPS=N` 从启动计数，永远只抓到前几步；加一个哨兵文件
   （`HIBANA_PROFILE_SENTINEL`）在负载饱和时触发的窗口才有意义。

## 二、Windows 特有的三个坑（按损失排序）

1. **主机侧逐行 I/O**：`seek+read` 走 Python 线程池，2048 行要 36 ms（GIL 争用 + 每行 syscall）。
   用 **numpy 对 memmap 做 fancy-indexing**（页错误在 C 里做，不占 GIL）：2048 行 **0.67 ms，53×**。
   这是本次最大的一处收益（16 并发 791 → 1709 tok/s）。
2. **WDDM 的事件同步 ~1.85 ms/次**。但注意：实测轮询 `event.query()` 替代 `synchronize()` **中性**——说明那 1.85 ms 是真在等 GPU，
   不是系统调用开销。所以别在同步方式上做文章，要去**减少每轮的握手次数**（或让依赖并行）。
3. **热路径里绝不能调 `torch.cuda.memory_reserved()`**：它递归遍历分配器所有 segment，本机秒级；而准入检查每步都调，
   表现为「引擎线程 100% CPU 空转、请求永远不进来」。改用 `torch.cuda.mem_get_info()` + 250 ms TTL 缓存。

## 三、MSVC + CUDA 13 的硬门槛

1. **CUDA 13 不认 MSVC 2026**（`cudafe++` 0xC0000005）。钉住 VS2022 BuildTools（`call VsDevCmd.bat` + `HIBANA_MSVC_PATH`）。
2. **`-gencode` 必须是 family 架构 `compute_120f/sm_120f`**：`cvt.e2m1x2`（FP4）只在 120a/120f 存在，普通 `sm_120` 会让 `ptxas` 直接报错，
   而 tvm_ffi 的构建器只按设备推普通 120 —— 这是「编过但跑挂 / 编不过」的根。
3. **ninja 的 build 行盘符必须写 `C$:`**（官方 ninja 1.13.1 也一样），否则报 `expected build command name`，错误信息完全不提盘符。
4. **MSVC 不接受的 POSIX/C++ 习惯**：`__attribute__((always_inline))`、`ushort/uint/ulong`、`M_PI`（要 `_USE_MATH_DEFINES`）、
   `std::max(unsigned long, size_t)`（LLP64 下类型不匹配）、`std::optional::value_or` 在 device 代码里、`constexpr auto` 当模板参数。
   治法是 force-include 一个 `win_compat.h` + 逐处小改（不要试图整体重写头文件）。
5. **flashinfer 在 Windows 的钥匙是 tvm_ffi**：它自带的 JIT 生成器是 Linux 形状（`c++`/`-fPIC`/`.so`/`-MMD`），
   而它依赖的 `tvm_ffi` 自带跨平台构建器（cl + nvcc + `deps=msvc` + `.dll`）。把 `JitSpecNvcc` 的 build/load 重定向过去就通了。

## 四、性能工程：什么时候该动什么

1. **先分清「每轮固定开销」和「随行数增长的每步开销」**。前者（握手、启动、图重放间隙）压下去对**所有并发档**有利；
   后者（attention/GEMM/GDN/pack）只影响高并发档。二者混在一起调，会得出矛盾结论。
2. **投机解码的深度 = 每轮的握手次数**。浅草稿 + 快的取行是这对变量的最优组合；加深草稿是净亏
   （每个 draft step 是一次完整 MTP 头 forward，不是免费）。
3. **按并发档分别调深度**（本次 c2→6、c4→3、c8→2、c16→4），比一个全局值多拿 10–15%。
4. **split-K 的候选不该只取 2 的幂**。`chunks % (2·splits) == 0` 这条约束下，Flash-Next 的 K=2560（20 chunks）里 **5 和 10 也合法**，
   而原实现 walk 到 4 就停；放开除数后 block 数 80 → 200，单流 +5~14%。
5. **「更快硬件」会暴露被掩埋的开销**：Spark 上 21 ms 的权重流量把所有启动/握手开销都盖住了；换到 1.6 TB/s 的卡，
   这些开销变成主导。移植性能优化的第一件事就是重新做「设备时间 vs 主机时间」分解，而不是照搬原机的调参结论。
6. **有损杠杆要显式记录而不是悄悄打开**：`HIBANA_MOE_BUDGET` 能把专家流量砍掉一大半，但它改变路由（重归一化权重）。本次未启用，写在报告里备查。

## 五、工程纪律

1. **改引擎热路径前先确认能回退**。本次所有补丁都是「整文件 scp 覆盖 + 保留原文件」，且用哈希逐一校验存档与线上一致。
2. **一次只动一个变量，并且每次都用同一组基准**（同一题面、同一 max_tokens、best-of-3、记录 tok/round 与 round_ms）。
   本次靠这个在十几轮里分清了 8 个被否定的方向。
3. **被否定的方向同样要写下来**（PLE 批量取行、轮询等待、snapshots、nvfp4 KV、加深草稿、放大 SPLIT_TARGET、max-batch 32…），
   否则下一个人会重走一遍。
4. **不要在没有完整时间块时开「改内核 + 回归 + 回退验证」这种三段式工作**。半成品比不改更糟。

## 六、第二期（plan 缓存那一轮）新增的六条

1. **A/B 必须交错**。这台机器上同一份代码 15–30 分钟会漂 3–4%、一整天能漂 5–8%（温度/功耗墙 + 外部 co-tenant）。
   「改前 3 份 → 隔十分钟 → 改后 3 份」会把漂移算成收益：本次第一版 PLE 批量写成 +6%，交错重测是 **+2~3%**。
   正确做法：两臂背靠背轮流跑（`restart.ps1 -Bat` 切脚本 + 环境变量开关），各取 2–3 份取中位。
2. **主机/设备的分解要用引擎内计时，不要用 py-spy 的「阶段」桶**。py-spy 把
   `wait_event(...)` 记在被等待的那个函数页上，于是 `AcceptGraphs.round` 显示 1.24 ms/轮「主机工作」，
   引擎内 `HIBANA_PHASE_STATS=1` 打出来是 **wait 10.05 / 真实主机 0.38 ms**。
3. **profiler 窗口的绝对数字不能直接当预算**：同一步在 821.9 ms/30 步的窗口里是 27.4 ms，
   实际单步 13.6 ms。**先除以「窗口单步 ÷ 实测单步」= 2.01 再分类**，否则会把并发占用算成设备开销。
4. **主机侧优化会饱和，判据是 `round_ms ≈ device 时间`**。c1 现在是 13.6 ms 对 13.62 ms：主机已完全遮住，
   再改主机一行收益都没有（本次把 `step_pages` 的 cat 也干掉，实测零收益，已回退）。此时应该换武器。
5. **WDDM 下图启动成本随节点数线性涨**：draft 图（~50 节点）0.02–0.03 ms，verify 图（~500 节点）0.6–0.85 ms/轮。
   所以「融合每层小内核」不只是省 device 时间，也是在省图节点 —— 这是接下来唯一有量级意义的刀。
6. **「PLE 设备侧常驻 + 设备侧 hash/gather」这个方向的复核结论**：主机握手确实在 `fill` 里，
   但 c1 上它被 draft 链的等待遮住（fill 1.78 里大部分是 `events[j].synchronize()`），
   把批量取行做对（numpy 哈希 + 一次 gather）只有 **c4/c8 +2~3%**；真正的天花板是 MoE 带宽（占 device 52%），
   不是取行。别再为取行开新的战役。
