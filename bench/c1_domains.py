"""Single-stream throughput per domain (the metric that matters at c1): greedy, long enough to amortize prefill.

Usage: python c1_domains.py <tag>
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:30004"
CASES = {
    "en_prose": "Explain in detail how a paged KV cache works in an LLM serving engine, covering page tables, prefix caching, and eviction. About 600 words.",
    "zh_prose": "\u7528\u4e2d\u6587\u5199\u4e00\u6bb5\u5173\u4e8e\u6df1\u591c\u5496\u5561\u9986\u7684\u6563\u6587\uff0c\u5927\u7ea6\u516d\u767e\u5b57\u3002",
    "code": "Write a clean Python asyncio rate-limited HTTP client pool using httpx with bounded retries and clean shutdown. Include docstrings and a usage example.",
    "zh_chat": "\u521a\u5fd9\u5b8c\u4e00\u5929\u5de5\u4f5c\u56de\u5bb6\uff0c\u7528\u81ea\u7136\u53e3\u8bed\u548c\u6211\u804a\u51e0\u53e5\u653e\u677e\u4e00\u4e0b\uff0c\u4e0d\u8981\u5ba2\u5957\u3002\u591a\u8bf4\u70b9\u3002",
}
MAX_TOKENS = 600


def post(body):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode())


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "head"
    out = {}
    for name, prompt in CASES.items():
        best = 0.0
        toks = 0
        for _ in range(2):
            t0 = time.time()
            r = post({"model": "hibana", "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": MAX_TOKENS, "temperature": 0.0,
                      "chat_template_kwargs": {"enable_thinking": False}})
            el = time.time() - t0
            toks = r["usage"]["completion_tokens"]
            best = max(best, toks / el)
        out[name] = best
        print(f"{name:9s} {best:7.1f} tok/s (best of 2, {toks} tok)", flush=True)
    print("SUMMARY", tag, " ".join(f"{k}={v:.1f}" for k, v in out.items()),
          f"MEAN={sum(out.values()) / len(out):.1f}", flush=True)


main()
