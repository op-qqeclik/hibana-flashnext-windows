"""Long-window throughput at a given concurrency and temperature, from counter deltas.

`python drive_ab.py <conc> <temp> <seconds>`: keeps `conc` requests in flight until the window ends, then reports
tokens/s, tokens per round, acceptance and round time over that window only (the engine's counters are cumulative,
so the deltas are the measurement). Long windows are what makes the sampled and greedy arms comparable: a single
burst is dominated by the prompt prefill.
"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.request

B = "http://127.0.0.1:30004"
PROMPT = "Write a detailed technical explanation of paged attention and its effect on serving throughput."
conc, temp, seconds = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])


def post(body, timeout=1800):
    r = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())["usage"]["completion_tokens"]


def stats():
    return json.loads(urllib.request.urlopen(B + "/get_server_info").read()).get("spec_stats") or {}


def one(_):
    return post({"model": "hibana", "messages": [{"role": "user", "content": PROMPT}],
                 "max_tokens": 3000, "temperature": temp})


base = stats()
t0 = time.time()
tok = 0
with cf.ThreadPoolExecutor(max_workers=conc) as ex:
    futs = [ex.submit(one, i) for i in range(conc)]
    while time.time() - t0 < seconds:
        time.sleep(1.0)
    for f in futs:
        f.cancel()
el = time.time() - t0
s = stats()
dr = s["rounds"] - base["rounds"]
de = s["emitted"] - base["emitted"]
dp = s["proposed"] - base["proposed"]
da = s["accepted"] - base["accepted"]
print(f"c{conc} temp {temp:3.1f}: {de/el:7.1f} tok/s | tok/round {de/max(1,dr):5.2f} | accept {da/max(1,dp):.3f} | "
      f"rounds {dr} | round {1000*el/max(1,dr):5.1f} ms", flush=True)
