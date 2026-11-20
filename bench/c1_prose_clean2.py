"""Single-stream English prose, measured by counting the tokens the API actually returned.

Earlier versions derived tokens from the engine's cumulative counters; `tokens_per_round` is a
lifetime average, so mixing it with a per-run round delta inflated the rate by up to 30%. This
counts `usage.completion_tokens` per response -- exact, instrument-free, comparable across heads.
Prompts are drawn from eval_prompts_120.json (phrasing templates no corpus generator used).
"""
import json, sys, time, urllib.request

BASE = "http://127.0.0.1:30004"


def ask(prompt: str, n: int) -> int:
    body = json.dumps({"model": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": n, "seed": 7}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode())
    return int((d.get("usage") or {}).get("completion_tokens") or 0)


items = json.load(open("eval_prompts_120.json", encoding="utf-8"))
picks = [p for d, p in items if d == "tech"][:3] + [p for d, p in items if d == "reason"][:3]
tag = sys.argv[1] if len(sys.argv) > 1 else "clean"
ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 900
best = 0.0
for rep in range(3):
    t0 = time.time()
    total = sum(ask(p, ntok) for p in picks)
    secs = time.time() - t0
    rate = total / secs
    best = max(best, rate)
    print(f"  rep{rep + 1}: {total} tok in {secs:.1f}s -> {rate:.1f} tok/s", flush=True)
print(f"SUMMARY {tag} en_prose_exact={best:.1f}")
