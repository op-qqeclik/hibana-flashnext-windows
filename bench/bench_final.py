import json, time, urllib.request, concurrent.futures as cf
BASE = "http://127.0.0.1:30004"

def post(body, timeout=900):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def info():
    with urllib.request.urlopen(BASE + "/get_server_info", timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))

CASES = {
    "en_prose": "Explain in detail how a paged KV cache works in an LLM serving engine, covering page tables, prefix caching, and eviction. About 600 words.",
    "zh_prose": "\u7528\u4e2d\u6587\u5199\u4e00\u7bc7\u5173\u4e8e\u201c\u57ce\u5e02\u7684\u6df1\u591c\u201d\u7684\u6563\u6587\uff0c\u4e0d\u5c11\u4e8e 600 \u5b57\u3002",
    "code": "Write a Python class implementing a thread-safe LRU cache with TTL, with type hints and docstrings, then a short usage example. Be thorough.",
}
for name, prompt in CASES.items():
    speeds = []
    for _ in range(3):
        t0 = time.time()
        out = post({"model": "hibana", "messages": [{"role": "user", "content": prompt}], "max_tokens": 700,
                    "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}})
        el = time.time() - t0
        u = out["usage"]
        speeds.append(u["completion_tokens"] / el)
    print(f"{name}: {max(speeds):.1f} tok/s (best of 3, {u['completion_tokens']} tok)")

prompt = "Write a short technical paragraph about paged attention, 200 words."
def one(_):
    t0 = time.time()
    out = post({"model": "hibana", "messages": [{"role": "user", "content": prompt}], "max_tokens": 400,
                "temperature": 0.3, "chat_template_kwargs": {"enable_thinking": False}})
    return out["usage"]["completion_tokens"], time.time() - t0

for conc in (2, 4, 8, 16):
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(one, range(conc)))
    wall = time.time() - t0
    total = sum(c for c, _ in res)
    print(f"concurrency {conc}: {total/wall:.1f} tok/s aggregate, {total/wall/conc:.1f} per stream")

s = info()["spec_stats"]
print("spec:", {k: s[k] for k in ("accept_rate", "tokens_per_round", "round_ms")})
