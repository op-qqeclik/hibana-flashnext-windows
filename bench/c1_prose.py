"""Prose-only single-stream throughput: the two domains the user actually feels.

English and Chinese long-form prose, greedy, ~900 tokens each, several prompts per language so one
prompt's luck does not decide the verdict. Usage: python c1_prose.py <tag>
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:30004"
EN = [
    "Write a long, carefully reasoned essay about why memory bandwidth, not arithmetic throughput, "
    "sets the ceiling for large-model inference. Cover the roofline model, arithmetic intensity, "
    "batching, KV cache growth, and what changes when the model is quantized. Be thorough.",
    "Explain in depth how a modern operating system scheduler balances latency and throughput, "
    "including run queues, priority inheritance, and the effect of cache locality on wake-up order.",
    "Describe at length how distributed consensus protocols maintain safety across partitions, "
    "with attention to leader election, log replication, and the cost of fsync.",
]
ZH = [
    "\u5199\u4e00\u7bc7\u8f83\u957f\u7684\u6563\u6587\uff0c\u5199\u6df1\u591c\u7684\u57ce\u5e02\u3001\u6709\u4eba\u521a\u4e0b\u73ed\u3001"
    "\u6709\u4eba\u521a\u9162\u6765\uff0c\u5199\u51fa\u5149\u5f71\u3001\u58f0\u97f3\u548c\u5fc3\u91cc\u90a3\u70b9\u6ca1\u8bf4\u51fa\u6765\u7684\u8bdd\u3002",
    "\u7528\u4e2d\u6587\u8be6\u7ec6\u8bb2\u89e3\u4e3a\u4ec0\u4e48\u5927\u6a21\u578b\u63a8\u7406\u53d7\u9650\u4e8e\u5185\u5b58\u5e26\u5bbd\uff0c"
    "\u8bf7\u4ece\u7b97\u672f\u5f3a\u5ea6\u3001\u6279\u91cf\u5927\u5c0f\u3001KV \u7f13\u5b58\u589e\u957f\u4e0e\u91cf\u5316\u56db\u4e2a\u65b9\u9762\u5c55\u5f00\u3002",
    "\u5199\u4e00\u7bc7\u5173\u4e8e\u65c5\u884c\u4e0e\u4e61\u6101\u7684\u957f\u6587\uff0c\u4ece\u4e00\u4e2a\u5c0f\u9547\u7684\u8f66\u7ad9\u5199\u8d77\uff0c"
    "\u5199\u5230\u5f88\u4e45\u4ee5\u540e\u56de\u5934\u770b\u90a3\u4e00\u5929\u3002",
]
MAX_TOKENS = 900


def post(body):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read().decode())


def bench(prompts, reps=2):
    best = 0.0
    toks = 0
    for _ in range(reps):
        for p in prompts:
            t0 = time.time()
            r = post({"model": "hibana", "messages": [{"role": "user", "content": p}],
                      "max_tokens": MAX_TOKENS, "temperature": 0.0,
                      "chat_template_kwargs": {"enable_thinking": False}})
            el = time.time() - t0
            toks = r["usage"]["completion_tokens"]
            best = max(best, toks / el)
    return best, toks


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "head"
    en, en_t = bench(EN)
    zh, zh_t = bench(ZH)
    print(f"en_prose {en:7.1f} tok/s ({en_t} tok)   zh_prose {zh:7.1f} tok/s ({zh_t} tok)", flush=True)
    print(f"SUMMARY {tag} en_prose={en:.1f} zh_prose={zh:.1f} prose_mean={(en + zh) / 2:.1f}", flush=True)


main()
