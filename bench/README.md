# bench/ — 测量工具

全部脚本都假设服务在 `http://127.0.0.1:30004`（`HIBANA_PORT` 改了要同步改脚本里的 `BASE`）。

**先读 `docs/04-benchmarks.md` 的"测试纪律"再跑。** 没有暖机与交错的数字没有意义。

## 闸门三件套（每次改动必过）

| 顺序 | 命令 | 通过标准 |
|---|---|---|
| ① 位一致 | `python bench\refcheck.py cmp base` | 打印 `IDENTICAL`，否则直接回退 |
| ② 五点 | `python bench\sweep.py 1,2,4,8,16 700` | 与上一次基线逐点比，全部不退化 |
| ③ 浸泡 | `python bench\soak.py` | 4 分钟持续负载不掉速、不崩 |

`refcheck.py save base` 用来建立基线（第一次跑、或换权重/换精度后）。

## 脚本一览

| 脚本 | 参数 | 用途 |
|---|---|---|
| `refcheck.py` | `save\|cmp <tag>` | 贪心 200 token 的 md5 + 五点速度；位一致闸门 |
| `sweep.py` | `<并发,如 1,2,4,8,16> <tokens>` | 五点并发扫描（贪心） |
| `accept_sampled.py` | `<tag>` | 采样档（temperature 0.3）五点 |
| `c1_prose.py` | `<tag>` | 单流散文（英/中各 3 篇、900 token、best-of-2）——**窄集，会被训练语料覆盖** |
| `c1_domains.py` / `c1_probe.py` | `<tag>` | 分域单流（代码 / 中文口语 / 英中散文） |
| `eval_wide120.py` | `<tag>` | **120 条宽集**（技术 40 / 推理 40 / 叙事 40）：唯一未被污染的裁决口径 |
| `eval_wide40_long.py` | `<tag>` | 40 条技术/推理 × 700 token 交错 A/B 补充口径 |
| `soak.py` | — | 4 分钟浸泡 |
| `drive_ab.py` | — | 两臂背靠背交错 A/B 驱动 |
| `depth_sampled.py` | — | 采样档下的深度表扫描 |
| `ple_hash_test.py` / `ple_hash_check.py` | — | PLE 取行的离线回归（**不需要 GPU**），改 PLE 路径后先跑它 |
| `bench_final.py` / `bench_matrix.py` / `bench_c1_delta.py` | — | 早期矩阵式基准（保留作历史对照） |
| `ctx_speed.py` / `ctx_stream.py` / `chat_ttft.py` | — | 上下文长度 / 流式 / TTFT 专项 |
| `temp_cmp.py` / `stream_vs_not.py` | — | 温度对比、流式与非流式对比 |

## 草稿头怎么"选"

`eval_wide120.py` **不接 `--head`**：头是由服务加载的。
换头 = 改 `HIBANA_MTP_WEIGHTS` + `launch\restart.ps1`，然后跑 `eval_wide120.py <tag>`。
（这是故意的：只有走真实加载路径，测出来的接受率才作数。）

## 数据文件

- `eval_prompts_120.json` —— **120 条宽集，冻结不要改**。改了就和历史数字不可比。
- `eval_prompts_40long.json`、`eval_prompts_wide.json` —— 补充集。

> ⚠️ `c1_prose.py` 的三条英文 prompt 与蒸馏训练语料 `TECH_DOMAINS` 逐字重叠（`docs/03` §6.3）。
> 它是**窄集**，只用于"同一分布内的回归"，**不能用来裁决草稿头好坏**。
