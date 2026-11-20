r"""Prefill throughput ladder: 4k / 8k / 16k / 32k / 64k / 128k, TTFT measured on the first streamed token.

Mirrors the SparkDash "上下文" card (docs/07-sparkdash.md): each size uses a **unique prefix** so the
prefix cache cannot inflate the number, and `stream_options.include_usage` gives the real prompt size.

    python bench\prefill_ladder.py [base_url] [size,size,...]

Notes
-----
* TTFT here includes request overhead (HTTP + tokenizer + scheduling). When that overhead is a large share
  of TTFT (short prompts, first call after start), the prefill rate reads low - the tool says so per size.
* Decode speed against context is a different script (`ctx_speed.py`); do not confuse the two.
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30004"
SIZES = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else "4096,8192,16384,32768,65536,131072")]
UNIT = ("The sparse attention indexer keeps one key per page, so a long conversation stays cheap to search "
        "while the resident set grows by a bounded amount per turn. ")
TASK = "\u7528\u4e00\u53e5\u8bdd\u8bf4\u660e\u5185\u5b58\u5e26\u5bbd\u4e3a\u4ec0\u4e48\u51b3\u5b9a\u63a8\u7406\u4e0a\u9650\u3002"


def build(size: int, nonce: str) -> str:
    """A prompt of roughly `size` tokens: a unique nonce first, then filler, then a short task."""
    target_chars = size * 4            # crude on purpose; the server reports the real token count
    body = (UNIT * (target_chars // len(UNIT) + 1))[:target_chars]
    return f"[run {nonce}]\n\n{body}\n\n(背景资料结束)\n\n{TASK}"


def stream(prompt: str):
    payload = {
        "model": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 24,
        "seed": 7,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    prompt_tokens = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                prompt_tokens = (obj["usage"] or {}).get("prompt_tokens", prompt_tokens)
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content") and ttft is None:
                    ttft = time.time() - t0
    wall = time.time() - t0
    return ttft, prompt_tokens, wall


def main() -> int:
    print(f"prefill ladder against {BASE}")
    print(f"{'size':>7} {'prompt_tok':>11} {'TTFT':>9} {'prefill tok/s':>14}   note")
    for i, size in enumerate(SIZES):
        nonce = f"{int(time.time())}-{i}-{size}"
        ttft, ntokens, wall = stream(build(size, nonce))
        if ttft is None:
            print(f"{size:>7} {'?':>11} {'-':>9} {'-':>14}   no streamed token; check the server log")
            continue
        tok = ntokens if ntokens else "?"
        rate = f"{ntokens / ttft:,.1f}" if (ntokens and ttft > 0) else "-"
        note = "low confidence: request overhead >30% of TTFT" if (ttft < 0.5) else ""
        print(f"{size:>7} {tok:>11} {ttft * 1000:>8.0f}ms {rate:>14}   {note}   (wall {wall:.2f}s)")
    print("\nprefill tok/s = prompt_tokens / TTFT. Unique prefix per size, so the prefix cache cannot help.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
