"""120-prompt wide eval with a per-register breakdown (tech / reason / narrative).

The aggregate acceptance hides the thing that matters most here: the user's dominant workload is
reasoning-shaped English prose, and a per-register split shows whether that register is already
better or worse than the mean. Greedy, engine-accumulated counters, one delta per prompt.
"""
import json, sys, time, urllib.request

BASE = "http://127.0.0.1:30004"


def stats():
    with urllib.request.urlopen(f"{BASE}/get_server_info", timeout=60) as r:
        return (json.loads(r.read()) or {}).get("spec_stats") or {}


def ask(prompt: str, n: int) -> None:
    body = json.dumps({"model": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": n, "seed": 11}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        r.read()


items = json.load(open("eval_prompts_120.json", encoding="utf-8"))
tag = sys.argv[1] if len(sys.argv) > 1 else "run"
tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 240
by = {}
t0 = time.time()
prev = stats()
for dom, p in items:
    ask(p, tokens)
    cur = stats()
    # the engine reports accept_rate cumulatively, so the accepted count is accept_rate * rounds
    def acc_of(st):
        return (st.get("accept_rate", 0.0) or 0.0) * (st.get("rounds", 0) or 0)
    d_rounds = (cur.get("rounds", 0) or 0) - (prev.get("rounds", 0) or 0)
    d_acc = acc_of(cur) - acc_of(prev)
    if d_rounds > 0:
        e = by.setdefault(dom, [0, 0, 0.0])
        e[0] += d_rounds
        e[1] += d_acc
        e[2] = cur.get("round_ms", 0.0)
    prev = cur
total_r = sum(v[0] for v in by.values())
total_a = sum(v[1] for v in by.values())
print(f"SUMMARY {tag} rounds={total_r} accept={total_a / max(total_r, 1):.4f} secs={time.time() - t0:.0f}")
for dom in ("tech", "reason", "narr"):
    if dom not in by:
        continue
    r, a, ms = by[dom]
    acc = a / r
    print(f"  {dom:7s} rounds={r:5d} accept={acc:.4f}  round_ms={ms:.2f}  ({100 * r / total_r:.0f}% of prompts)", flush=True)
