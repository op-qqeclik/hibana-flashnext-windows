"""Acceptance helper: greedy reference text (determinism) + c1 speed + spec stats.

`python refcheck.py save <tag>` stores the greedy 200-token answer's text + md5 for later
comparison; `python refcheck.py cmp <tag>` re-generates it and diffs against the stored one
(identical text and identical completion_tokens == identical token ids for a deterministic
tokenizer). Both then report c1 tok/s (best of 3) and the engine's spec stats.
"""
import hashlib
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:30004"
PROMPT = ("Explain in detail how a paged KV cache works in an LLM serving engine, covering page "
          "tables, prefix caching, and eviction. About 600 words.")
BENCH = "Write a short technical paragraph about paged attention, 200 words."
OUT = r"B:\hibana-win\results"


def post(body, timeout=900):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def greedy(prompt, max_tokens=200):
    return post({"model": "hibana", "messages": [{"role": "user", "content": prompt}],
                 "max_tokens": max_tokens, "temperature": 0.0, "seed": 1234,
                 "chat_template_kwargs": {"enable_thinking": False}})


def key(out):
    text = out["choices"][0]["message"].get("content") or ""
    return out["usage"]["completion_tokens"], hashlib.md5(text.encode()).hexdigest(), text


def info():
    with urllib.request.urlopen(BASE + "/get_server_info", timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def bench_c1(n=3, max_tokens=400):
    best, speeds = None, []
    for _ in range(n):
        t0 = time.time()
        out = post({"model": "hibana", "messages": [{"role": "user", "content": BENCH}],
                    "max_tokens": max_tokens, "temperature": 0.0,
                    "chat_template_kwargs": {"enable_thinking": False}})
        el = time.time() - t0
        speeds.append(out["usage"]["completion_tokens"] / el)
        if best is None:
            best = out
    return speeds


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "save"
    tag = sys.argv[2] if len(sys.argv) > 2 else "base"
    path = f"{OUT}\\ref_{tag}.json"
    n, md5, text = key(greedy(PROMPT))
    print(f"greedy 200 tok: {n} tokens, md5 {md5}")
    driven = bench_c1()
    print("c1 tok/s: " + ", ".join(f"{s:.1f}" for s in driven) + f" (best {max(driven):.1f})")
    s = info().get("spec_stats") or {}
    print("spec:", {k: s.get(k) for k in ("accept_rate", "tokens_per_round", "round_ms", "rounds")})
    if mode == "save":
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"tokens": n, "md5": md5, "text": text}, fh)
        print("saved", path)
    else:
        with open(path, encoding="utf-8") as fh:
            ref = json.load(fh)
        same = ref["md5"] == md5 and ref["tokens"] == n
        print(f"reference {tag}: tokens {ref['tokens']} md5 {ref['md5']}")
        print("IDENTICAL" if same else "DIFFERENT")
        if not same:
            a, b = ref["text"], text
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f"  first diff at char {i}: ref {a[max(0, i - 40):i + 40]!r} / now {b[max(0, i - 40):i + 40]!r}")
                    break
    return 0


sys.exit(main())
