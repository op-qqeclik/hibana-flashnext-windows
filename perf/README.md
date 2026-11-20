# perf/ — 引擎侧改动的参考材料

> **这里没有引擎源码。** 出于许可原因，本仓库不镜像引擎文件；下面给的是**变更清单 + 可读 diff**，
> 完整机制的说明在 `docs/02-tuning.md`，移植类改动（26 处）由 `port/winpatch.py` 用锚点字符串替换完成。

## 提供的 diff

| 文件 | 改了什么 | 实测 | 状态 |
|---|---|---|---|
| `hc-bf16.patch` | `model/flashnext/model.py`：三处 `HyperConnection.load(...)` 加 `precision="bf16"`（原 fp8） | 单流 **+4.8%**（每轮解码字节 5.38 → 6.01 GB，反而更快） | ✅ 固化 |
| `wide-at-9rows.patch` | `kernels/w3.py::wide_for`：加 `HIBANA_W3_WIDE_M` 实验开关，让 9 行也能强制走 wide 内核以便横扫配置 | 用于**证伪**（wide 路径净亏 1.5×） | ❌ 仅实验开关 |

`wide-at-9rows.patch` 保留的原因：它是"在这张卡上少读字节 ≠ 更快"这条结论的原始工具。
想复现 `docs/02` §2.7 的反直觉结果，用它扫一遍配置。

## 改了但只以"锚点替换"形式提供的六处

这六处改动跨多个文件、且需要与引擎版本绑定，所以以 `port/winpatch.py` 的锚点补丁 + 文字说明为准：

| 文件 | 改动 | 效果 |
|---|---|---|
| `proposers/mtp_graphs.py` | FA2 plan 按 `(split, 每请求 chunk 数)` 缓存；加 `STATS` 探针（copies/plan/launch 分项） | draft plan 0.21 → 0.12 ms/步 |
| `engine/spec_graphs.py` | verify 侧复用同一 plan 缓存；`HIBANA_PLE_BATCHED` 开关（默认关） | verify plan 0.29 → 0.07 ms/轮 |
| `engine/accept_graphs.py` | 接受路径接同一缓存；页表原地增长替代每轮 `torch.cat` | 同上 |
| `model/flashnext/extras.py` | 批量 PLE 行（`HIBANA_PLE_BATCH_ROWS`，b≥4 时 +2–6%） | c4/c8 小收益 |
| `engine/request.py` | 请求侧的准备与 plan 命中路径对齐 | 配合上面三项 |
| `cache/request_state.py` | 页表原地增长（不再每轮 `torch.cat` 复制） | 减掉每轮一次大拷贝 |

外加一个探针 `proposers/_planprof.py`（`HIBANA_PLAN_CACHE=0` 可回退，`HIBANA_PHASE_STATS=1` 打开引擎内分阶段计时）。

## 怎么用这些补丁

```powershell
# 1) 先确认你的引擎版本能命中锚点（winpatch 会列出未命中的锚点并中止）
python port\winpatch.py <engine_root>

# 2) diff 类改动手工应用（它们带上下文，锚点变了会失败得明显）
git -C <engine_root> apply --check ..\perf\hc-bf16.patch
git -C <engine_root> apply         ..\perf\hc-bf16.patch

# 3) 每次改完必过位一致闸门
python bench\refcheck.py cmp base      # 必须 IDENTICAL
```

## 纪律

- 任何"更快"的改动，先过 `refcheck` 的位一致闸门，再谈速度。
- 改一个文件前先 `copy` 一份 `.pre-<tag>`，服务必须保持可启动。
- 只碰你自己的引擎副本；本仓库的脚本默认只写 `HIBANA_HOME` 下的路径。
