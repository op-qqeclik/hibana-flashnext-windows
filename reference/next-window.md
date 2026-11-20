# 下一次开机后怎么接（107 关机中，本文件是可执行的清单）

## 0. 口径（2026-11-17 起变更，重要）

- **性能验收一律用采样档**（temperature 0.3，`accept_sampled.py`），不再用贪心五点曲线当吞吐指标。
- **质量闸门仍是贪心逐位一致**（temperature 0、seed 1234、200 token，`refcheck.py cmp base`）：
  吞吐只有在「输出 token 一个没变」时才值得上报。

## 1. 开机后先确认（2 分钟）

```
ssh pro6000 'powershell -NoProfile -ExecutionPolicy Bypass -File B:\hibana-win\state.ps1'
curl -s http://127.0.0.1:30004/health        # 必须 200 才发请求，不要 sleep 猜
```
若服务没起：`restart.ps1 -Bat B:\hibana-win\run-flashnext.bat`（它会轮询 health 到 200 再返回）。

## 2. 拷贝本轮新增的脚本（从 102）

```
scp accept_sampled.py drive_ab.py temp_cmp.py pro6000:B:/hibana-win/
```

## 3. 建立**采样档**基线（这是新的对照点）

```
cd B:\hibana-win && .venv\Scripts\python.exe accept_sampled.py base-sampled
```
它做五件事：先跑质量闸门（`refcheck.py cmp base`），再对 c1/2/4/8/16 各跑 2 次（每次 512 token/流、temp 0.3、
固定 seed、取中位），最后写 `results/ref_sampled_base-sampled.json`。
**注意**：c1 的采样吞吐只能与「同样的 512 token 预算」对比 —— 固定时长窗口不可比（快的手臂跑进更长的上下文，
round 时间从 15.9 ms 涨到 30.6 ms，会自我减速）。

## 4. 然后才是内核工作（唯一剩下的方向，需小时级 + 逐位回归）

先读 `kernel-fusion-spec.md`（离线读源码得到的结论，能省掉一轮白工）。按它对 §9 的修正，顺序是：

1. **并行化 `quant_fp8_small`**：`_quant_small_kernel[(1,)]` 是单 CTA、两遍扫 46K 元素 ≈10 µs/次 × 48–100 次/步
   = 0.5 ms/步。改成「多 CTA 求 block max → 两级归约 → 多 CTA 写 fp8」⇒ 0.2–0.25 ms/步（c1 +2%），
   且 `div_rn(x, scale)` 对同一个 scale 逐元素相同 ⇒ fp8 字节完全一致。
   **（注意：不要去做「inject_norm+quant_small 融合」——scale 是全局 max，没有 grid 内同步就融不了。）**
2. **打包 `direct_copy`**（0.40 ms/步、128+ 次 4 KB 拷贝）：同轮常量进一个 pinned buffer、一次 `copy_`。
3. **确认后让 dense 变体跳过选择**（`_budget_select_kernel` 48 次/步；先证明变体选择正确再动，选错会让注意力静默退回 dense）。
4. accept 图并进 verify 图（省一次大图启动：载波图 0.6–0.85 ms/轮，draft 图只要 0.02–0.03）。

每一步的验收：`refcheck.py cmp base`（必须 IDENTICAL）→ `accept_sampled.py <tag>`（采样五点）→ `soak.py`。
改前先 `copy` 一份 `.pre-<tag>` 备份；`HIBANA_PLE_BATCH_ROWS=0` / `HIBANA_PLAN_CACHE=0` 可一键回到本会话之前的行为。

## 5. 已知的坑（别重蹈）

- A/B **必须交错**（两臂背靠背各 2–3 次取中位）：这张卡 15–30 分钟漂 3–4%、一天漂 5–8%。
- 主机/设备分解用 `HIBANA_PHASE_STATS=1` 的引擎内计时（`verify cost:` / `accept cost:` / `fill cost:`），
  不要用 py-spy 的阶段桶（它把 `wait_event` 记成主机工作，会把 0.38 ms 看成 1.24 ms）。
- profiler 窗口的绝对数字要先除以「窗口单步 ÷ 实测单步」再当预算。
- 不改验收门限（`accept_top_k/margin/soft`）：那是拿质量换的，用户明确否掉。

## 下次窗口（2026-11-19 之后）的执行清单

### 第 0 步：恢复 107 并校验配置（**必做，因为关机时任务指针可能停在诊断配置上**）

```
schtasks /delete /tn hibana-flashnext /f
powershell -File B:\hibana-win\restart.ps1 -Bat B:\hibana-win\run-flashnext.bat
findstr /C:"MTP_WEIGHTS" /C:"SPEC_DEPTHS" /C:"MOE_BUDGET" /C:"KEEP_BF16" B:\hibana-win\run-flashnext.bat
```
必须看到 `mtp_weights.v16reason.safetensors` 和 `1:8,2:6,4:3,8:3,16:4`，
**且不得出现 `MOE_BUDGET` / `KEEP_BF16`**。然后 `refcheck.py cmp hcbf16` 应报 IDENTICAL（基准 `cb222a7f…`）。

### 第 1 步：补完被中断的 MoE budget 诊断消融

launcher 已备好：`run-flashnext-moeb2.bat` / `-moeb8.bat`（`HIBANA_MOE_BUDGET=2/8`，
**有损，仅诊断，绝不能固化**）。判据用 `eval_wide120.py` 的 `round_ms`：

```
<launcher> ; python eval_wide120.py moeb2
```

已知：MoE ≈ 2.2 ms/轮（18%），按 ~2.9 GB/轮、1.3 TB/s 算已到峰值 82%。若 `budget=2`
只能省 <0.5 ms，则 MoE 侧确认无空间，不必再投入。

### 第 2 步：按 bf16 原理继续逐组试（**只对 K/N 小的矩阵有效**）

已验证：K=320 的 hyper 矩阵 bf16 **+4.8%**；`o_proj`（大矩阵）bf16 **−2.6%**。
剩下的候选按"K 从小到大"排：
```
HIBANA_TRUNK_KEEP_BF16=in_proj_z        # GDN 的 z 投影
HIBANA_TRUNK_KEEP_BF16=indexer          # 若有
HIBANA_TRUNK_KEEP_BF16=mlp\.down_proj   # 对照组（大矩阵，预期负）
```
每组只跑一次 `eval_wide120.py`，`round_ms` 改善 ≥2% 才算数。**必须避开 231 个 3-bit 矩阵**
（`q_proj|k_proj|v_proj|gate_proj|up_proj|down_proj|in_proj_qkv`），否则 trunk 从 7 GB 涨到 14 GB。

### 第 3 步（真正的量级项，需要专门窗口）

- **trunk 231 次 3-bit GEMM**：`qkv` 与 `shared_gate_up` **引擎已经合并过**，
  231 = 48 层 × 4.8 已接近最小；要再降只能改权重布局、把剩余的同输入矩阵拼一起（作者级）
- **块并行草稿（DFlash2 在本机隐藏态重训）**：唯一能绕过「单 token 准确率 0.73 平线」的路；
  现成权重与本引擎特征 tap 不对齐（zh 接受率 0.066）

### 已固化的纪律（别再违反）

1. **裁决只用 `eval_wide120.py`（120 条）**；`c1_prose.py` 和 `head_eval_suite.py` 的 prompt
   **被训练语料覆盖**，只能当冒烟测试
2. **内核微基准必须用引擎内 CUPTI 复核**（Python 发射开销会把 12 µs 的 GPU 时间盖住）
3. 五点必须先暖机（图捕获按桶惰性发生）
4. 107 服务期间**不得跑任何生成任务**（并发 4 也会从 326 掉到 78 tok/s）
5. `schtasks /run` 在任务处于 Running 时是静默 no-op ⇒ `restart.ps1` 已改为先 end+delete

### 第 2b 步（已备好补丁，待测）：**强制 wide 内核跑 9 行**

**为什么可能有收益**：`csrc/w3_gemm.cu` 的窄内核 `kWarps=4, kBlockRows=128`，block 数 = `n // 128`。
对 (5120, 5120) 只有 **40 blocks × 4 warps = 160 warps**，而这张卡有 **188 SM** —— 不到 1 warp/SM，
无法隐藏 HBM 延迟，实测只跑到峰值的 **51%**（12.6 µs vs 理论 6.2 µs）。
`wide_for()` 目前对 `m <= 32` 一律返回 `None`（宽内核只在 33 行以上用），但宽内核按更少的行/warp 分块、
warp 数更多。

**补丁**：`wide-at-9rows.patch`（打到 `hibana/kernels/w3.py`，备份 `.pre-wide9`），
由 `HIBANA_W3_WIDE_M=<config[,splits]>` 控制，只在环境变量存在时才强制走宽内核，
**不打这个变量时行为与现在完全一致**（零风险）。

config = `MT*1000 + WM*100 + WN*10 + STAGES`，`rows = 16*MT*WM`，所以 9 行要用 `MT=1, WM=1`：
```bat
set HIBANA_W3_WIDE_M=1143
set HIBANA_W3_WIDE_M=1243
set HIBANA_W3_WIDE_M=1133
```
每组：`restart.ps1` → `eval_wide120.py <tag>`，看 `round_ms`。
基线是 **12.49–12.52 ms**；≤12.2 才算有效（≈+2%）。若全部更慢，说明窄内核的选择是对的，
这条也关闭。

⚠️ 注意：`WIDE` 表里没有的形状返回 `None`（保持原状），所以这个开关只影响表内的 8 个几何。

### 第 1b 步：用**分文体**评测确认主场景的真实速度

新增 `eval_wide120_domains.py`（120 条同样的 prompt，但**按 tech / reason / narrative 分开统计**接受率），
用法：`python eval_wide120_domains.py <tag> [max_tokens]`。

**为什么值得先做**：用户的绝对主场景是**推理型英文散文**。聚合接受率 0.5945 可能掩盖了
"reason 文体本来就更高"这件事（早前窄套件里 reason 域曾到 0.7175）。
如果 reason 分档确实明显更高，那么**对主场景应报的速度就比 325 tok/s 更高**，
而且后续训练/调参的判据也应该换成这个分档，而不是均值。

预期输出形如：
```
SUMMARY <tag> rounds=NNNN accept=0.59xx secs=NN
  tech    rounds=... accept=0.5xxx  round_ms=12.5x
  reason  rounds=... accept=0.6xxx  round_ms=12.5x
  narr    rounds=... accept=0.5xxx  round_ms=12.5x
```

拿到这张表之后，再决定：
- 若 reason ≥0.65：主场景已明显好于均值，应把「对 455 的达成率」按 reason 分档重算；
- 若三档接近：说明接受率与文体无关，只能动分母（wide@9 那一步）。
