"""Greedy vs sampled acceptance at one concurrency: same prompt, counter deltas only."""
import json
import sys
import time
import urllib.request

B = "http://127.0.0.1:30004"


def post(body):
    r = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t = time.time()
    o = json.loads(urllib.request.urlopen(r, timeout=600).read())
    return o["usage"]["completion_tokens"], time.time() - t


def stats():
    return json.loads(urllib.request.urlopen(B + "/get_server_info").read()).get("spec_stats") or {}


base = stats()
for tag, temp in (("greedy", 0.0), ("sampled0.3", 0.3)):
    n = el = 0.0
    for _ in range(3):
        c, t = post({"model": "hibana", "messages": [{"role": "user", "content": "Write a short technical paragraph about paged attention, 200 words."}],
                     "max_tokens": 400, "temperature": temp})
        n += c
        el += t
    s = stats()
    dr = s["rounds"] - base["rounds"]
    de = s["emitted"] - base["emitted"]
    dp = s["proposed"] - base["proposed"]
    da = s["accepted"] - base["accepted"]
    print(f"{tag:10s} {n/el:6.1f} tok/s | tok/round {de/max(1,dr):5.2f} | accept {da/max(1,dp):.3f} | rounds {dr} | "
          f"round_ms {(el*1000)/max(1,dr):.1f} (wall incl. prefill)", flush=True)
    base = s
