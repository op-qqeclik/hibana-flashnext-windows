"""c1 in delta terms: one 400-token greedy request, with /metrics + /get_server_info counters taken
before and after (the cumulative averages mix loads, so only the delta counts)."""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:30004"
PROMPT = "Write a short technical paragraph about paged attention, 200 words."


def get(path, timeout=60):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.read().decode()


def metrics():
    out = {}
    for line in get("/metrics").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, val = line.partition(" ")
        try:
            out[name] = float(val)
        except ValueError:
            pass
    return out


def stats():
    d = json.loads(get("/get_server_info"))
    s = dict(d.get("spec_stats") or {})
    s["_mem"] = (d.get("memory_gib") or {})
    return s


def post(body, timeout=900):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    reqs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    for i in range(n):
        m0, s0 = metrics(), stats()
        t0 = time.time()
        out = {"usage": {"completion_tokens": 0}}
        for _ in range(reqs):
            o = post({"model": "hibana", "messages": [{"role": "user", "content": PROMPT}],
                      "max_tokens": max_tokens, "temperature": 0.0,
                      "chat_template_kwargs": {"enable_thinking": False}})
            out["usage"]["completion_tokens"] += o["usage"]["completion_tokens"]
        wall = time.time() - t0
        m1, s1 = metrics(), stats()
        tok = out["usage"]["completion_tokens"]
        dro = max(1.0, s1.get("rounds", 0) - s0.get("rounds", 0))
        dem = s1.get("emitted", 0) - s0.get("emitted", 0)
        dsteps = max(1.0, m1.get("hibana_decode_steps", 0) - m0.get("hibana_decode_steps", 0))
        dtok = m1.get("hibana_decode_tokens", 0) - m0.get("hibana_decode_tokens", 0)
        print(f"run {i + 1}: {tok} tok / {wall:.2f}s = {tok / wall:.1f} tok/s | round {1000 * wall / dro:.2f} ms "
              f"| rounds/s {dro / wall:.1f} | tok/round {dem / dro:.2f} | tok/step {dtok / dsteps:.2f} "
              f"| mem {s1['_mem'].get('allocated')} GiB")
        for k in sorted(s1):
            if not k.startswith("_") and k not in ("rounds", "emitted"):
                print(f"    {k}: {s1[k]}")
        break
    return 0


sys.exit(main())
