# 03 · MTP 草稿头蒸馏与语料训练（完整流程 + 结果 + 为什么它没用）

> **English abstract** — The complete MTP draft-head distillation pipeline we ran: on-policy corpus
> generation with the model itself, top-32 logit extraction, a 4-step self-feeding KL+CE rollout loss with
> rank-32 LoRA on the MoE experts, export as a `safetensors` overlay, mount via one env var, and evaluation
> by `accept_len` then by engine throughput. Then the part that matters: **on a clean 120-prompt set, all
> five distilled heads landed within 2.3 % of the stock head.** The speedup we thought we had was prompt
> memorisation. Both the working pipeline and the retraction are here.

---

## 0. 为什么值得做（先算清账）

单流吞吐完全由**每轮吐多少 token** 决定，而 `tok/round` 又完全由**草稿链的逐位置接受率**决定：

| 深度 | tok/round 实测 | 逐位置接受率（反推） |
|---|---|---|
| d=1 | 1.85 | 第 1 个草稿 ≈ **0.85** |
| d=4 | 2.75 | 第 2–4 个 ≈ **0.30** |
| d=8 | 3.94 | 第 2–8 个 ≈ **0.30** |

要 455 tok/s（c1 的目标）需要 `tok/round ≈ 5.9`，也就是**第 2–8 个草稿的接受率从 0.30 抬到 ~0.76**。
引擎侧（分母 = 每轮的 device 时间）已经压到 12.6 ms 的固定成本，能再压的只有 10–20%。
**所以"提高接受率"是唯一还有量级空间的方向** —— 而提高接受率只能动草稿头的权重。

这就是这条线的全部动机。它的结论是负面的，但**流程是可复用的**，所以完整保留。

---

## 1. 管线总览（五步，每步都有脚本）

```
① 语料生成（on-policy，用正在服务的引擎自己写）
   train/src/generate_prose.py  →  corpus_*.jsonl（{domain, messages[, temperature]}）
        ↓
② 渲染成训练行
   train/src/mk_corpus.py       →  {input_ids, prompt_len, domain, thinking}
        ↓
③ 用目标模型抽特征（隐藏态 + top-32 logits）
   train/src/extract_mtp_features.py  →  shard_*.pt（每分片 60 条）
        ↓
④ 蒸馏训练（4 步自喂 rollout，CE + KL_top32，MoE 专家上挂 LoRA）
   train/src/train_mtp_kl.py    →  mtp_weights.safetensors（5.21 GB）
        ↓
⑤ 挂载 + 评测
   HIBANA_MTP_WEIGHTS=<path> + restart  →  bench/eval_wide120.py（唯一可信口径）
```

**为什么是"自喂（self-feeding）"**：草稿头吃的是**自己上一轮的输出位置**，不是教师强制的位置。
`rollout_steps=4` 表示在每条序列上取 64 个锚点，从锚点开始让草稿头自己滚 4 步，
每一步都和教师（目标模型）在同一位置的分布对齐。这样训练分布 = 推理分布。

---

## 2. 语料：用模型自己写，而不是找数据集

`generate_prose.py` 的 docstring 把意图写得很直白：

> The deployed workload is long English and Chinese prose (explanatory writing, essays), so the draft head
> must be distilled on that: long single-turn continuations at a low temperature on many distinct topics,
> not on chat turns or code.

机制：

1. 主题池是**组合式**的：`DOMAINS × TECH_ANGLES × TECH_FORMS`，`rng.choice` 取三元组并 `key = f"{d}|{a}|{f}"`
   去重 ⇒ 能凑出上千条互不重复的 prompt。
2. 采样温度**故意压低**：每 5 条里 4 条 `temp=0.3`、1 条 `temp=0.0`（贪心）—— 与线上部署同分布。
3. 逐条 POST 到本地 OpenAI 兼容端点（httpx 异步并发），全部 `err=0`。
4. 语言/文体分池：`EN_TOPICS` / `ZH_TOPICS`、`EN_STYLE` / `ZH_STYLE`（"约九百字"、"要具体有层次"、"不要提纲"）、
   `REASON_SUBJECTS`（一阶原理推导类）。

**实际产出的语料（dgx105 上的真实文件）**

| 文件 | 条数 | token | 来源 |
|---|---|---|---|
| `corpus_prose6.jsonl` | 1 400 | **1 185 008** | 长文散文（EN+ZH） |
| `corpus_reason_en.jsonl` | 1 600 | **1 612 098** | reasoning 语域 |
| `corpus_wide_zh.jsonl` | 1 200 | **1 187 471** | DOMAINS×ANGLES×FORMS |
| `corpus_skel_zh.jsonl` | 1 600 | **1 642 522** | 骨架/结构类中文 |
| **中文合并 `corpus_zh_all.jsonl`** | 2 050 | **2 055 036** | 上面中文侧合并（34 分片） |
| `corpus/balanced2.jsonl` | 1 380 | — | 早期多桶（chat/prose/roleplay/coding/reasoning/visual） |
| `corpus/balanced3.jsonl` | 1 644 | — | 同上，第二轮 |
| **合计** | **≈9 050** | **≈7.7 M** | |

> **教训 1**：语料量不是瓶颈。我们从 35 万 token 加到 122 万、再到 205 万，trainer 指标一路涨，
> **引擎侧收益一直是 0**（见 §5）。小语料能复现的现象，大语料救不回来。

---

## 3. 训练：超参、损失、LoRA

主脚本 `train/src/train_mtp_kl.py`（497 行）。以下是**实际跑过的启动参数**（`run_train.sh` / `run_train2.sh`）：

```bash
python training/train_mtp_kl.py \
  --data-dir ~/mtp_zh --out-dir ~/mtp_zh/out2 \
  --epochs 2 --lr 6e-6 --batch-size 2 --grad-accum 4 \
  --num-anchors 64 --lora-rank 32
```

| 超参 | 值 | 备注 |
|---|---|---|
| `rollout_steps` | **4** | 草稿头自喂 4 步 |
| 逐步损失权重 | **`[1.0, 0.8, 0.64, 0.5]`** | 越靠后的自喂步权重越低（误差累积） |
| 损失 | **`0.5 · CE + 0.5 · KL_top32`** | 见下 |
| `topk` | **32** | 教师 logits 只留 top-32，写进特征分片 |
| `num_anchors` | **64** | 每条序列取 64 个 rollout 锚点 |
| LoRA rank | **32** | **挂在 MoE 专家 `experts.gate_up_proj` / `experts.down_proj`** |
| 有效 batch | **2 × grad_accum 4 = 8** | 很小 ⇒ 单步 loss 噪声大（见 §4） |
| lr / epochs | `1e-5 / 1ep`（run1）；`6e-6 / 2ep`（run2） | 脚本默认 `6e-5 / 2ep`，实际压低了 |

**损失函数**（`compute_kl_ce()`）：

```python
ce = F.cross_entropy(logits, target_tok)                    # 硬标签：教师采样出来的 token
p_top = softmax(teacher_top_val)                            # 教师 top-32 分布
kl = Σ p_top · (log p_top − log q_top)                      # 学生在同 32 个位置上的分布对齐
loss = 0.5 * ce + 0.5 * kl
```

**LoRA 在导出时被合并成实权重**（不是运行时挂 adapter）：

```python
gu_merged = experts_gu + bmm(gu_B, gu_A)      # gate_up_proj
```

⇒ 导出物是**完整的 5 214 305 424 B safetensors**，与 checkpoint 里的 MTP 头逐张量同名，
所以挂载不需要任何转换步骤（§6）。

**算力成本**：122 万 token / 有效 batch 8 ≈ 244 步/epoch，1 epoch ≈ **18 分钟**（GB10 上单卡）。

---

## 4. 训练结果：loss 是噪声，`accept_len` 才是信号

`accept_len` = 验证集上每次投机能吐出多少个 token（含 1 个 verify token），是**唯一和引擎收益同向**的指标。

### run1：1 epoch、lr 1e-5、244 步 —— **没有净收益**

| step | loss | | step | loss |
|---|---|---|---|---|
| 10 | 1.2620 | | 130 | 0.7969 |
| 30 | 0.9454 | | 150 | 0.8474 |
| 50 | 0.8628 | | 170 | 0.9677 |
| 70 | 1.0275 | | 190 | 1.1892 |
| 90 | 1.2651 | | 210 | **1.3890** |
| 110 | 0.7029 | | 240 | 0.9894 |

```
[VAL init] accept_len=3.930  s0=0.789
[VAL ep1 ] accept_len=3.918  s0=0.788      ← 反而略降 0.012
```

### run2：2 epoch、lr 6e-6、488 步 —— **唯一有净收益的一次**

```
[VAL init] accept_len=3.885  s0=0.783
[VAL ep1 ] accept_len=3.932  s0=0.787      ← improved，存 out2/mtp_weights.safetensors
[VAL ep2 ] accept_len=3.939  s0=0.789      ← improved，存 mtp_weights.ep2.safetensors
```

单步 loss 同样在 **0.63–1.39 之间剧烈抖动、无下降趋势**（有效 batch 只有 8）。

> **教训 2**：小 batch 下 **单步 loss 不可用作训练信号**。判断一次 run 有没有用，只能看
> **held-out 上的 `accept_len`**（且必须是跨语料的留出集，见 §7）。
> 这条与 `AGENTS.md` 里"验收看行为、不看单步 loss"的口径一致。

### 跨语料留出集（这一步是对的，也是后来被自己推翻的起点）

v5 / v8 / v9 在**同一套 75 条散文留出序列**上：

| 头 | `accept_len` | 相对 v5 |
|---|---|---|
| v5 | 2.668 | — |
| v8 | 2.904 | +8.8% |
| **v9** | **3.017** | **+13.1%** |

看起来非常干净：单调上升、留出集、跨语料。**但留出集和训练集语料同源（都是"散文"），
所以它只证明了"在散文分布内更准"，没证明"在新题面上更准"。** 差别在引擎侧被放大成 +11%，
在宽集上被抹平为 0 —— 详见 §6。

---

## 5. 产物与挂载

### 5.1 产物

| 文件 | 大小 | md5 |
|---|---|---|
| `mtp_weights.v9.safetensors` | 5 214 305 424 B | **`44c8f960628498b679a6b3e382ea8825`** |
| `mtp_weights.v16reason.safetensors`（线上） | 5 214 305 424 B | `62cf963533e9a8244186f6bbfaadec8d` |
| `mtp_weights.orig.safetensors`（原始头，备份） | 5 214 305 424 B | — |
| `out_mtp11` / `out_mtp13`（早期两轮） | 5 214 305 424 B | `44cd8b47…` / `f292c5ca…` |

> **传权重必须校验 md5 和字节数**：`B:` 盘写满时 `scp` 会写出**半截文件且不明显报错**。
> 我们差点用 2 244 MB 的截断 safetensors 启动引擎（正常应为 5 214 305 424 B）。

### 5.2 挂载：一个环境变量 + 一个按名覆盖的只读代理层

```bat
set HIBANA_MTP_WEIGHTS=B:\hibana-models\mtp\mtp_weights.v16reason.safetensors
```

链路：`run-flashnext.bat` → `EngineConfig.mtp_weights` → `model/flashnext/mtp.py` 里
`FlashNextMtpHead.__init__` 的 `w = _Overlay(current().mtp_weights, w)`。

`_Overlay` 的语义（引擎 docstring 原文）：

> a distilled head's tensors override the checkpoint's, matched by name with or without the `mtp.` prefix.
> Everything else reads through to the snapshot, **so a partial export is legal**: the tensors it omits
> stay the checkpoint's.

⇒ **换头 = 改一行环境变量 + `restart.ps1`**，没有格式转换、没有注册步骤、缺张量自动回落。
加载时自动转 FP8，**零额外显存**（5.2 GB 的 fp32 头不会占显存）。

生效验证看启动日志：

```
speculation: mtp, 8 drafts, draft the model's own MTP head
acceptance: greedy 3:2.0, sampled s=0.5
```

---

## 6. ★ 结果：宽集上全部抹平（这是本文档最重要的一节）

### 6.1 窄集上看到的（**错误的**）

引擎侧成对交错，同一窗口：

| 项 | v5 | v9 | 增益 |
|---|---|---|---|
| 散文 en / zh | 283 / 276 | 315 / 300 | **+11.2%**（5/5 轮全正，最低 +9.3%） |
| c1 | 186–199 | 215–237 | +17% |
| c2 | 371–377 | 450–458 | +21% |
| c4 | 600–614 | 673–779 | +20% |
| c16 | 1680–1700 | 1714–1865 | +6% |

引擎内分解：en_prose 的 `tok/round` 3.31 → 4.26（+29%）、接受率 0.570 → 0.608。

### 6.2 换 120 条全新 prompt 之后（**决定性的**）

`bench/eval_wide120.py` + `bench/eval_prompts_120.json`：技术说理 40 + 推理 40 + 叙事 40，
**三域句式模板与任何训练语料都不同**；贪心、240 token/条；接受率取引擎累计计数器差分。

| 草稿头 | accept | tok/round | round_ms | tok/s |
|---|---|---|---|---|
| v16reason（声称 +11.2% 的那个） | 0.5969 | 4.107 | 13.22 | 310.7 |
| v5 | 0.5894 | 3.975 | 12.85 | 309.3 |
| v9 | 0.5922 | 4.059 | 13.19 | 307.7 |
| v22 | 0.5912 | 4.030 | 13.12 | 307.2 |
| **orig（从未训练过）** | 0.5912 | 3.986 | 13.12 | **303.8** |

**五个头全部落在 2.3% 以内。** 另用 40 条技术/推理 prompt × 700 token 交错 A/B：
orig 0.5980 / 3.934 vs v16reason 0.5931 / 3.863 —— **原头还略好**。

### 6.3 那 +11% 是怎么来的

`bench/c1_prose.py` 的 3 条英文 prompt 与训练语料 `TECH_DOMAINS` **逐字重叠**：

| `c1_prose.py` 的英文 prompt | 训练语料里 |
|---|---|
| "why memory bandwidth, not arithmetic throughput, sets the ceiling…" | **是**（第 2 条） |
| "how a modern OS scheduler balances latency and throughput…" | **是**（第 4 条） |
| "how distributed consensus protocols maintain safety…" | **是**（第 3 条） |

同窗口成对交错重测：orig 292.4 / 295.9，v16reason 348.8 / 349.5（+19%）。

**它测的是对这三道题的记忆，不是泛化能力。** 我们等于把验证集训进了训练集，然后为 +11% 高兴了一整天。

---

## 7. 结论（三条硬结论，写在这里供引用）

1. **MTP 草稿头的窄域蒸馏，泛化增益 ≈ 0。** 5 个版本（最多 2 epoch、约 200 万 token、多种体裁）
   在宽集上都与**原始未训练头**无差别（0.589–0.597）。
2. **原始头已接近该架构的泛化上限（≈0.5912）。** 瓶颈是草稿头的**单 token 准确率**——
   逐位置接受率在 0.73 → 0.567 之间几乎是**平的**，不是链式衰减问题。
   在"单层串行 MTP + 16 候选近似"这个结构下，0.73 很可能是上限。
3. **对外应报的诚实数字**：英文散文 **~305–310 tok/s**（宽集口径），对目标 455 是 **67–69%**。
   不是 +11% 之后那个 350。

**所以真正提升单流的是什么？** 是把 hyper-connection 从 fp8 换成 **bf16**（+4.8%，且多读字节反而更快，
见 `docs/02` §2.7），以及引擎侧的握手削减。**不是蒸馏。**

---

## 8. 如果重做一遍（给接手者的建议）

| 建议 | 理由 |
|---|---|
| **先建宽集，再训练** | 我们反过来做了：先训 23 个版本，再建宽集，结果全部作废。宽集（`eval_prompts_120.json`）应当在**第一次训练之前**就冻结 |
| **验证集必须与训练语料"句式不相交"** | 同语料留出集只能证明"分布内更准"，会给出虚假的 +13% |
| **只报引擎侧宽集数字，不报 trainer 指标** | `accept_len` 2.6→3.0 是真实的，但它不蕴含吞吐；两者在窄域内相关、在宽域内不相关 |
| **别加大语料** | 35 万 → 122 万 → 205 万 token，trainer 指标单调涨，引擎收益恒 0。这条路已证伪 |
| **要做就做块并行草稿（DFlash2 类）** | 唯一能绕开"单 token 准确率 0.73 平线"的结构性改动；现成权重与本引擎的特征 tap 不对齐（zh accept 0.066），**必须在本机隐藏态上重训** |
| **小 batch 下别看 loss** | 0.63–1.39 抖动、无趋势；2 epoch 那次 loss 没降，但 `accept_len` 涨了 |

---

## 9. 复现（脚本都在 `train/src/`）

```bash
# 0) 准备：一个能跑 hibana 的环境 + 目标模型权重（3-bit 档即可，训练时只读隐藏态）
export HIBANA_HF_CACHE=~/.cache/huggingface
export HIBANA_W3_CHECKPOINT=~/models/w3-flashnext
export HIBANA_FLASHNEXT_EXPERTS=w3

# 1) 起服务（语料生成要用它）
hibana serve flashnext --host 127.0.0.1 --port 30004 &

# 2) 生成语料（on-policy；约 1400 条 / 118 万 token，EN ~248 tok/s / ZH ~203 tok/s）
python train/src/generate_prose.py --n 1400 --out ~/mtp_gen/corpus_prose6.jsonl

# 3) 渲染训练行
python train/src/mk_corpus.py --in corpus_prose6.jsonl --out corpus8.jsonl

# 4) 抽特征（隐藏态 + top-32 logits，每分片 60 条）
python train/src/extract_mtp_features.py \
  --hibana-dir ~/hibana --corpus corpus8.jsonl --out-dir ~/mtp_train --shard-size 60

# 5) 蒸馏（2 epoch 是唯一有净收益的配置）
python train/src/train_mtp_kl.py --data-dir ~/mtp_train --out-dir ~/mtp_train/out \
  --epochs 2 --lr 6e-6 --batch-size 2 --grad-accum 4 --num-anchors 64 --lora-rank 32

# 6) 校验大小与 md5，拷到推理机
md5sum out/mtp_weights.safetensors     # 必须是 5 214 305 424 B
scp out/mtp_weights.safetensors pro6000:B:/hibana-models/mtp/mtp_weights.mine.safetensors

# 7) 挂载 + 重启 + 用宽集裁决（不要用窄集！）
#    set HIBANA_MTP_WEIGHTS=B:\hibana-models\mtp\mtp_weights.mine.safetensors
#    (the head is chosen by the server, not by the script) set HIBANA_MTP_WEIGHTS, restart, then:
python bench/eval_wide120.py mine
```

**判据**：宽集 `accept` 相对 `orig` 提升 < 2% ⇒ 判定无效，别上线。
这就是我们用一整天换来的那一条规则。

---

## 10. 已知的不确定（诚实清单）

1. **v9（md5 `44c8f960…`）的确切出处无法确认**：dgx105 上现存两份产物（`out_mtp11` / `out_mtp13`）md5 都不匹配，
   `mtp_zh/out2/`（训练日志明确写过保存）已被删除，另一台可能的训练机（102）SSH 不可达。
2. `data8/` 里的 `target_embed.pt` / `target_lm_head.pt` 已被删（符号链接断），
   所以"从零重跑特征抽取"要重新生成，不能复用。
3. 优化器段未逐行核对（脚本内固定，未见 `add_argument` 暴露）—— 复现时以 `train_mtp_kl.py` 源码为准。
4. 语料生成器 `generate_prose.py` 里的 `TECH_DOMAINS` 就是评测污染源（§6.3）。**复用时请把它从生成池里删掉**，
   否则你会重现我们那个 +11% 的假象。
