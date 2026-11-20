# train/ — MTP 草稿头蒸馏

完整方法论、超参、结果与**为什么它没用**见 [`../docs/03-mtp-distillation.md`](../docs/03-mtp-distillation.md)。
这里只放"怎么跑"。

```
src/generate_prose.py         ① on-policy 语料生成：用正在服务的引擎自己写长文（0.3 / 0.0 温度 4:1）
src/mk_corpus.py              ② 渲染成训练行 {input_ids, prompt_len, domain, thinking}
src/extract_mtp_features.py   ③ 用目标模型抽隐藏态 + top-32 logits（每分片 60 条）
src/train_mtp_kl.py           ④ 蒸馏训练（4 步自喂 rollout，0.5·CE + 0.5·KL_top32，MoE 上挂 LoRA）
src/run_extract.sh            ③ 的实际启动脚本（含 HIBANA_* 环境）
src/run_train.sh              ④ 1 epoch / lr 1e-5（实测：accept_len 3.930 → 3.918，无净收益）
src/run_train2.sh             ④ 2 epoch / lr 6e-6（实测：3.885 → 3.932 → 3.939，唯一有净收益的配置）
src/generate_corpus.py        早期多桶语料生成（chat/prose/roleplay/coding/reasoning/visual）
```

## 核心超参（实际跑过的）

| 项 | 值 |
|---|---|
| rollout 步数 | 4 |
| 逐步损失权重 | `[1.0, 0.8, 0.64, 0.5]` |
| 损失 | `0.5 · CE + 0.5 · KL_top32` |
| top-k | 32 |
| rollout 锚点 | 每序列 64 个 |
| LoRA | rank 32，挂 MoE 专家 `experts.gate_up_proj` / `experts.down_proj`，**导出时合并成实权重** |
| 有效 batch | 2 × grad_accum 4 = 8 |
| lr / epochs | 6e-6 / 2（或 1e-5 / 1） |

## 最小流程

```bash
python src/generate_prose.py --n 1400 --out ~/mtp_gen/corpus_prose6.jsonl
python src/mk_corpus.py --in ~/mtp_gen/corpus_prose6.jsonl --out ~/mtp_train/corpus8.jsonl
python src/extract_mtp_features.py --hibana-dir ~/hibana --corpus ~/mtp_train/corpus8.jsonl \
    --out-dir ~/mtp_train --shard-size 60
python src/train_mtp_kl.py --data-dir ~/mtp_train --out-dir ~/mtp_train/out \
    --epochs 2 --lr 6e-6 --batch-size 2 --grad-accum 4 --num-anchors 64 --lora-rank 32
```

产物 `mtp_weights.safetensors` 必须是 **5 214 305 424 B**；md5 与实际不符 ⇒ 传输出错，别上线。
挂载：`set HIBANA_MTP_WEIGHTS=<path>` + `launch\restart.ps1`。

## 三条硬规矩（用一天半换来的）

1. **先冻结宽集（`bench/eval_prompts_120.json`），再训第一个版本。** 我们反过来做，23 个版本全部作废。
2. **验证集必须与训练语料句式不相交。** 同分布留出集给出过虚假的 +13.1%。
3. **裁决只用引擎侧宽集 accept，不看 trainer loss。** 小 batch 下 loss 在 0.63–1.39 抖动、无趋势。

> ⚠️ `src/generate_prose.py` 里的 `TECH_DOMAINS` 是评测污染源（`docs/03` §6.3）。
> 复用时请把它从生成池删掉，否则你会重现我们那个 +11% 的假象。
