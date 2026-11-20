"""120-prompt wide acceptance eval (40 tech / 40 reasoning / 40 narrative), greedy, engine counters."""
import json, sys, time, urllib.request

BASE = "http://127.0.0.1:30004"


def stats():
    with urllib.request.urlopen(f"{BASE}/get_server_info", timeout=60) as r:
        return (json.loads(r.read()) or {}).get("spec_stats") or {}


def ask(prompt: str, n: int = 240) -> None:
    body = json.dumps({"model": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": n, "seed": 11}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        r.read()


items = json.load(open("eval_prompts_120.json", encoding="utf-8"))
tag = sys.argv[1] if len(sys.argv) > 1 else "run"
a, b = stats(), None
t0 = time.time()
for _, p in items:
    ask(p)
b = stats()
dr = (b.get("rounds", 0) or 0) - (a.get("rounds", 0) or 0)
print(f"SUMMARY {tag} rounds={dr} accept={b.get('accept_rate', 0):.4f} "
      f"tok/round={b.get('tokens_per_round', 0):.3f} round_ms={b.get('round_ms', 0):.2f} "
      f"secs={time.time() - t0:.0f}", flush=True)
