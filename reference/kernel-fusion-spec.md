# 内核层剩余空间的实地勘察（离线分析，改之前请先读完）

范围：HANDOFF §13 里 device 预算的「小内核堆 2.65 ms/步」（c1 的 19%）与 §9 第 ① 条。
以下结论全部来自读 `hibana/kernels/flashnext_fused.py` 源码 + 实测的 launch 计数，**不需要在机器上跑**。

## 1. device 预算里那几个 kernel 到底是谁

| profile 里的名字 | 次数/步 | 归一化 ms/步 | 实际来源 |
|---|---|---|---|
| `_inject_norm_kernel` | 94 | 0.31 | `inject_combine_norm()`：inject + combine + RMSNorm **已经是一个 kernel** |
| `_quant_small_kernel` | 101 | 0.50 | `quant_fp8_small()`：per-tensor 动态 FP8 量化 |
| `_budget_select_kernel` | 48 | 0.20 | qsa `select_pages`（indexer 选择） |
| `_grouped_norm_kernel` | — | — | `grouped_gemma_rms`，norm 的另一半 |
| `direct_copy` | 128+ | 0.40 | 散落的 D2D 小拷贝（tokens/positions/pages/slots/y/meta/relax/side） |

## 2. 「把 inject_norm + quant_small 融合」这条**做不到**（我上一版写在 §9 的建议要改）

`inject_combine_norm()` 的签名返回 `(y, normed, parts)`：`parts` 是**每个 block 的 |y| 最大值**，
给并行量化（`quant_fp8_parts`）用的。而 `quant_fp8_small()` 的 scale 是**整个张量的全局 max**

```
peak = max(|x| over all elements);  scale = max(peak / 448, 1e-12);  q = clamp(div_rn(x, scale), -448, 448)
```

全局 max 意味着「先算完所有元素、再做一次跨 program 的归约、然后才开始写 fp8」——
单靠一个 kernel 做不到（除非用 cooperative groups / persistent kernel 做 grid 内同步）。
所以这两步之间**必须**有一次物化 + 归约，现在这个结构已经是「没有 grid sync 的前提下最省的形式」。

## 3. 真正的浪费：`_quant_small_kernel` 是**单 CTA**

```python
_quant_small_kernel[(1,)](x, y, scale, alpha, wscale, n, BLOCK=4096, num_warps=8)   # 注意网格是 (1,)
```

它用**一个 program** 跑两遍（先求 max、再写 fp8），n = M*K（c1 时约 9×5120 ≈ 46K 个元素）。
单 SM 的带宽 ~50 GB/s ⇒ 46K × 2 字节 × 2 遍 ≈ 4 µs，加上固定开销约 10 µs；每步 48–100 次 ⇒ **0.5 ms/步**。
这是可以砍的：把 max 那一遍改成多 CTA + 两级归约（或直接复用现成的 `quant_fp8_parts` 的 block maxima），
第二遍并行写 fp8。

**为什么这样仍然逐位一致**：`scale` 是「全部元素绝对值的最大值」乘 1/448，并行归约得到的**是同一个 max**；
`div_rn`（round-to-nearest）对同一个 scale 逐元素相同 ⇒ 输出 fp8 字节完全相同。
（注意 `scale = tl.maximum(max * (1.0/448.0), 1e-12)` 里的乘倒数语义要保持：PyTorch 除以 Python 标量 = 乘 FP32 倒数。）

预期收益：0.5 → 0.2~0.25 ms/步 ⇒ **c1 +2%** 左右；对 c2–c8 同量级（这条是每步固定开销）。

## 4. `_budget_select_kernel` 48 次/步（0.20 ms/步）

只在 qsa 的「选择」路径上跑。短上下文时 `plan["dense"]` 为真、`select_pages` 不参与注意力——
但 profile 里确实有 48 次/步，说明 c1 的图捕获的是**选择型变体**。先确认再动：
抓一份 `variant()` 的日志（`HIBANA_PHASE_STATS` 或临时打印），如果 dense 轮次里它确实在算，
那么「dense 变体不捕获选择」能同时省 0.20 ms/步和 48 个图节点（WDDM 下图启动 0.6–0.85 ms/轮）。
**不要没确认就改**：选错变体会让注意力静默退回 dense（结果错）。

## 5. `direct_copy` 0.40 ms/步（128+ 次 4 KB 拷贝）

来源是每轮往静态张量里搬 tokens/positions/pages/slots/y/meta/relax 和 side inputs，每次 3–10 µs 的启动成本。
可行方向：把同一轮的常量（tokens/positions/pages/slots）打包进一个 pinned buffer、一次 `copy_`；
或者用 `torch._foreach_copy_` 合并。收益 ~0.2–0.3 ms/步，风险低但改动面广（每一处都要跟上布局）。

## 6. 顺序建议

1. `quant_fp8_small` 并行化（收益确定、语义清晰、逐位可证）——
2. `direct_copy` 打包（收益中等、改动面广）——
3. 确认后让 dense 变体跳过选择（需要先证明变体选择正确）——
4. 最后才是「减少 verify 图节点」这件更慢的活。

每条都必须过：`refcheck.py cmp base`（IDENTICAL）→ `accept_sampled.py <tag>`（采样五点）→ `soak.py`。


### 3b. 补充（round 43 复盘）：并行化 `quant_fp8_small` **不划算**，别做

`scale` 是全局 max ⇒ 并行版本最少要 3 个 launch（求 block parts / 归约出 scale / 写 fp8），
而现在的单 CTA 版本只要 **1 个 launch**、端到端 ~5 µs（46K 元素 × 2 遍 / 单 SM 50 GB/s ≈ 3.7 µs + 启动）。
12 个 CTA 并行只能把 3.7 µs 压到 ~0.3 µs，却要多付 2 次 WDDM 启动（各 2–3 µs）—— **净亏**。
若真要动它，唯一正确的方向是**把量化融进生产者**（例如 `silu_mul` 的输出直接写 fp8，scale 用上一层已知的值），
那属于逐点改图，不是改这个 kernel。
