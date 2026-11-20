"""Per-load engine metrics from counter deltas: tokens per spec round, tokens per decode step, steps/s, aggregate
tok/s — for greedy vs sampled, at several concurrencies. The cumulative counters mix loads, so only deltas count."""
import json, time, urllib.request, concurrent.futures as cf

BASE = "http://127.0.0.1:30004"


def metrics() -> dict:
    with urllib.request.urlopen(BASE + "/metrics", timeout=20) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, val = line.partition(" ")
        try:
            out[name] = float(val)
        except ValueError:
            pass
    return out


def stats() -> dict:
    with urllib.request.urlopen(BASE + "/get_server_info", timeout=20) as r:
        d = json.loads(r.read().decode())
    s = dict(d.get("spec_stats") or {})
    s["_mem"] = (d.get("memory_gib") or {}).get("allocated")
    return s


def post(body, timeout=900):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def case(label, conc, temp, max_tokens=400):
    m0, s0 = metrics(), stats()
    t0 = time.time()
    def one(_):
        return post({"model": "hibana", "messages": [{"role": "user", "content":
                    "Write a short technical paragraph about paged attention, 200 words."}],
                    "max_tokens": max_tokens, "temperature": temp,
                    "chat_template_kwargs": {"enable_thinking": False}})
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(one, range(conc)))
    wall = time.time() - t0
    m1, s1 = metrics(), stats()
    total = sum(r["usage"]["completion_tokens"] for r in res)
    dsteps = max(1.0, m1.get("hibana_decode_steps", 0) - m0.get("hibana_decode_steps", 0))
    dtok = m1.get("hibana_decode_tokens", 0) - m0.get("hibana_decode_tokens", 0)
    dro = max(1.0, s1.get("rounds", 0) - s0.get("rounds", 0))
    dem = s1.get("emitted", 0) - s0.get("emitted", 0)
    print(f"{label:10s} {total:5d} tok / {wall:5.2f}s = {total/wall:7.1f} tok/s "
          f"({total/wall/conc:5.1f}/stream) | tok/step {dtok/dsteps:5.2f} | steps/s {dsteps/wall:5.1f} "
          f"| tok/round {dem/dro:6.2f} | rounds/s {dro/wall:4.2f} | round {1000*dro/wall and wall/dro*1000:6.1f} ms")
    return total / wall


print("greedy (temp 0) vs sampled (temp 0.3), concurrency sweep")
for temp, tag in ((0.0, "greedy"), (0.3, "sampled")):
    for conc in (1, 2, 4, 8, 16):
        case(f"{tag} c{conc}", conc, temp)
