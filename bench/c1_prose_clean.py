"""Single-stream English prose on prompts that are NOT in any training corpus.

c1_prose.py's three English prompts are literally TECH_DOMAINS entries, so a distilled draft head
scores them by memorisation (and the original head does not) -- that measurement is not comparable
across heads. This uses six long-form prompts drawn from eval_prompts_120.json (unseen: they are
phrased with templates no corpus generator used) at 900 tokens, greedy, three passes, best-of.
"""
import json, statistics, sys, time, urllib.request

BASE = "http://127.0.0.1:30004"


def stats():
    with urllib.request.urlopen(f"{BASE}/get_server_info", timeout=60) as r:
        return (json.loads(r.read()) or {}).get("spec_stats") or {}


def ask(prompt: str, n: int) -> None:
    body = json.dumps({"model": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": n, "seed": 7}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        r.read()


items = json.load(open("eval_prompts_120.json", encoding="utf-8"))
picks = [p for d, p in items if d == "tech"][:3] + [p for d, p in items if d == "reason"][:3]
tag = sys.argv[1] if len(sys.argv) > 1 else "clean"
best = 0.0
for rep in range(3):
    a = stats()
    t0 = time.time()
    for p in picks:
        ask(p, 900)
    b = stats()
    secs = time.time() - t0
    rounds = (b.get("rounds", 0) or 0) - (a.get("rounds", 0) or 0)
    tpr = (b.get("tokens_per_round", 0) or 0)
    # the prompts run SEQUENTIALLY, so the served tokens are rounds * tokens_per_round over all of them
    total_tok = rounds * tpr
    rate = total_tok / secs
    best = max(best, rate)
    print(f"  rep{rep + 1}: {rounds} rounds x {tpr:.3f} tok/round = {total_tok:.0f} tok in {secs:.1f}s -> {rate:.1f} tok/s"
          f"  (per stream {total_tok / len(picks):.0f} tok)", flush=True)
print(f"SUMMARY {tag} en_prose_clean={best:.1f}")
