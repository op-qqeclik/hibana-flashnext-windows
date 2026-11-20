#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate an on-policy bilingual corpus (chat, prose, roleplay, coding, reasoning, visual prompts)
for multi-step MTP KL distillation from a running OpenAI-compatible Hibana server."""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer

DOMAINS = [
    ("zh_chat", False, 0.7, [
        "刚忙完一天工作回家，用自然口语和我聊几句放松一下，不要客套。",
        "用非常自然、有画面感的中文写一段深夜咖啡馆里两个老朋友叙旧的对话。",
        "把我发你的这句话改写得更口语、更有人味：『今天任务推进得不错，辛苦大家了。』",
        "假设我们正在一起逛夜市，你一边挑小吃一边吐槽今天的趣事，写一段生动口语。",
    ]),
    ("en_chat", False, 0.7, [
        "Hey, I just wrapped up a long day of debugging. Give me a warm, witty, natural spoken check-in.",
        "Write a grounded, vivid dialogue between two friends sharing takeout on the floor at midnight.",
        "Describe a rainy neon street scene in cinematic detail suitable for a video generation prompt.",
        "React naturally and playfully to someone claiming they were 'totally productive' all afternoon.",
    ]),
    ("zh_visual", False, 0.0, [
        "给出一段电影级雨夜街头镜头提示词，包含焦段、运镜、光线氛围与材质细节。",
        "写一段赛博朋克风格的城市高架桥航拍分镜描述，突出湿滑路面反光与车流光轨。",
    ]),
    ("code", False, 0.0, [
        "Write a clean Python asyncio rate-limited HTTP client pool using httpx with bounded retries and clean shutdown.",
        "Implement a thread-safe LRU cache with TTL expiration in Python without external dependencies.",
        "Write a PyTorch function computing grouped top-k KL divergence from sparse top-k teacher logits.",
    ]),
    ("en_think", True, 0.0, [
        "Analyze the memory bandwidth vs acceptance-rate tradeoffs of a 1-layer MoE autoregressive MTP head versus a 5-layer parallel block drafter on unified-memory hardware.",
        "Explain step by step how to verify prefix-cache eviction vs SSD tier restoration under slot pressure.",
    ]),
]


async def generate(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.snapshot_dir, trust_remote_code=True)
    tasks = []
    for i in range(args.count):
        dom, thinking, temp, prompts = DOMAINS[i % len(DOMAINS)]
        usr = f"{prompts[(i // len(DOMAINS)) % len(prompts)]} (var #{i})"
        tasks.append((dom, thinking, temp if i % 4 != 0 else 0.0, [{"role": "user", "content": usr}]))
    rng.shuffle(tasks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["idx"])

    todo = [(i, t) for i, t in enumerate(tasks) if i not in done]
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = {"ok": 0, "err": 0, "toks": 0, "t0": time.time()}

    async with httpx.AsyncClient(timeout=180.0) as client:
        with out_path.open("a", encoding="utf-8") as fout:
            async def one(idx: int, item: tuple) -> None:
                dom, thinking, temp, msgs = item
                async with sem:
                    body = {
                        "model": args.model,
                        "messages": msgs,
                        "max_tokens": args.max_tokens,
                        "temperature": temp,
                        "chat_template_kwargs": {"enable_thinking": thinking},
                    }
                    try:
                        r = await client.post(args.url, json=body)
                        r.raise_for_status()
                        content = r.json()["choices"][0]["message"].get("content") or ""
                        if not content:
                            stats["err"] += 1
                            return
                        p_ids = tok.apply_chat_template(
                            msgs, tokenize=True, add_generation_prompt=True, enable_thinking=thinking
                        )
                        if not isinstance(p_ids, list):
                            p_ids = list(p_ids["input_ids"])
                        f_ids = tok.apply_chat_template(
                            msgs + [{"role": "assistant", "content": content}],
                            tokenize=True,
                            add_generation_prompt=False,
                            enable_thinking=thinking,
                        )
                        if not isinstance(f_ids, list):
                            f_ids = list(f_ids["input_ids"])
                        if len(f_ids) <= len(p_ids) + 16:
                            stats["err"] += 1
                            return
                        row = {
                            "idx": idx,
                            "domain": dom,
                            "thinking": thinking,
                            "prompt_len": len(p_ids),
                            "input_ids": f_ids[: args.max_seq_len],
                        }
                        async with lock:
                            fout.write(json.dumps(row) + "\n")
                            stats["ok"] += 1
                            stats["toks"] += len(row["input_ids"]) - row["prompt_len"]
                            if stats["ok"] % 50 == 0:
                                fout.flush()
                                dt = max(1e-3, time.time() - stats["t0"])
                                print(f"[{stats['ok']}/{len(todo)}] toks={stats['toks']} rate={stats['toks']/dt:.1f} tok/s", flush=True)
                    except Exception as exc:
                        stats["err"] += 1
                        if stats["err"] <= 5:
                            print("ERR:", repr(exc)[:120], flush=True)

            await asyncio.gather(*(one(i, t) for i, t in todo))
    print(f"DONE ok={stats['ok']} err={stats['err']} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30004/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--snapshot-dir", required=True, help="Path to HF snapshot containing tokenizer")
    ap.add_argument("--out", default="corpus.jsonl")
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=20261115)
    asyncio.run(generate(ap.parse_args()))


if __name__ == "__main__":
    main()
