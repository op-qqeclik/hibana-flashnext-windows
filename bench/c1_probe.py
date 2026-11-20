"""Where does single-stream prose time go? Compare draft depth and context growth."""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:30004"
P = {
    "en": "Write a long, carefully reasoned essay about why memory bandwidth rather than arithmetic "
          "throughput sets the ceiling for large-model inference. About 900 words.",
    "zh": "\u5199\u4e00\u7bc7\u8f83\u957f\u7684\u6563\u6587\uff0c\u5199\u6df1\u591c\u7684\u57ce\u5e02\u3001"
          "\u6709\u4eba\u521a\u4e0b\u73ed\u3001\u6709\u4eba\u521a\u9162\u6765\u3002\u7ea6\u4e5d\u767e\u5b57\u3002",
}


def info():
    with urllib.request.urlopen(BASE + "/get_server_info", timeout=30) as r:
        return (json.loads(r.read().decode()) or {}).get("spec_stats") or {}


def run(name, prompt, n):
    a = info()
    t0 = time.time()
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                data=json.dumps({"model": "hibana",
                                                 "messages": [{"role": "user", "content": prompt}],
                                                 "max_tokens": n, "temperature": 0.0,
                                                 "chat_template_kwargs": {"enable_thinking": False}}).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=2400) as r:
        out = json.loads(r.read().decode())
    el = time.time() - t0
    b = info()
    d = {k: (b.get(k, 0) - a.get(k, 0)) for k in ("rounds", "proposed", "accepted", "emitted")}
    toks = out["usage"]["completion_tokens"]
    r_ = max(1, d["rounds"])
    print(f"{name:6s} n={n:4d} -> {toks:4d} tok / {el:5.1f}s = {toks/el:6.1f} tok/s | "
          f"tok/round {d['emitted']/r_:5.2f} | accept {d['accepted']/max(1,d['proposed']):.3f} | "
          f"round {el/d['rounds']*1000:5.1f} ms", flush=True)


tag = sys.argv[1] if len(sys.argv) > 1 else "x"
for n in (200, 600, 1200, 2000):
    for k, p in P.items():
        run(f"{tag}/{k}", p, n)
