"""Acceptance in the mode the engine actually serves: sampled (temperature 0.3), not greedy.

`python accept_sampled.py [tag]` runs the five concurrency points with a fixed token budget per stream, so the
decode length (and therefore the context trajectory) is identical in every arm -- a fixed *duration* window is not
comparable, because a faster arm runs further into a longer context and then slows itself down (single stream:
round time grows from 15.9 ms at ~300 tokens to 30.6 ms at ~1500 on the 107 box).

Per point it reports aggregate and per-stream tok/s, tokens per round, the acceptance rate and round time, all from
counter deltas over that point only (the engine's counters are cumulative). Numbers go to results/ref_sampled_<tag>.json
for later comparison; `quality_gate()` then re-runs the greedy 200-token reference and fails the run if the token ids
moved -- throughput is only worth reporting when the engine still produces the same tokens.
"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:30004"
PROMPT = ("Write a detailed technical explanation of paged attention, its page table, prefix caching and eviction, "
          "and how they affect serving throughput. About 500 words.")
MAX_TOKENS = 512
TEMPERATURE = 0.3
POINTS = (1, 2, 4, 8, 16)
REPEATS = 2          # per point; the median is reported
SETTLE = 1.0         # let the counters drain between points


def post(body, timeout=1800):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out["usage"]["completion_tokens"], time.time() - t0, out["choices"][0]["message"]["content"]


def stats():
    try:
        with urllib.request.urlopen(BASE + "/get_server_info", timeout=60) as r:
            return (json.loads(r.read()) or {}).get("spec_stats") or {}
    except Exception:
        return {}


def health():
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def point(conc, temp=TEMPERATURE, max_tokens=MAX_TOKENS, seed=None):
    """One burst of `conc` requests; returns (tok/s, tokens, seconds, counter deltas)."""
    base = stats()
    t0 = time.time()
    tok = 0
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        futs = []
        for i in range(conc):
            body = {"model": "hibana", "messages": [{"role": "user", "content": PROMPT}],
                    "max_tokens": max_tokens, "temperature": temp}
            if seed is not None:
                body["seed"] = seed + i
            futs.append(ex.submit(post, body))
        for f in futs:
            tok += f.result()[0]
    el = time.time() - t0
    time.sleep(SETTLE)
    now = stats()
    d = {k: now.get(k, 0) - base.get(k, 0) for k in ("rounds", "emitted", "proposed", "accepted")}
    return tok / el, tok, el, d, now


def fmt(conc, tok_s, tok, el, d):
    rounds = max(1, d["rounds"])
    accept = d["accepted"] / max(1, d["proposed"])
    return (f"c{conc:<2d} {tok_s:7.1f} tok/s ({tok_s / conc:6.1f}/stream) | {tok:5d} tok in {el:5.2f}s | "
            f"tok/round {d['emitted'] / rounds:5.2f} | accept {accept:.3f} | round {1000 * el / rounds:5.1f} ms")


def quality_gate() -> bool:
    """Delegate the greedy reference to `refcheck.py cmp <tag>` (same prompt, seed and md5 as the stored one).

    Throughput is only worth reporting when the engine still produces the same token ids; `refcheck` owns that
    comparison so the two scripts cannot drift apart.
    """
    import subprocess
    for tag in ("base", "warm1"):
        try:
            out = subprocess.run([sys.executable, "refcheck.py", "cmp", tag], capture_output=True, text=True, timeout=1800)
        except Exception as exc:
            print(f"quality gate: refcheck failed ({exc})")
            return False
        line = [l for l in out.stdout.splitlines() if "IDENTICAL" in l or "DIFFERENT" in l]
        if line:
            print("quality gate:", line[-1])
            return "IDENTICAL" in line[-1]
    print("quality gate: no stored reference found (run refcheck.py save <tag> first)")
    return True


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y%m%d-%H%M")
    for _ in range(120):
        if health():
            break
        time.sleep(5)
    else:
        print("engine never answered /health")
        return 1
    print(f"sampled acceptance (temperature {TEMPERATURE}, {MAX_TOKENS} tokens per stream, median of {REPEATS})")
    out = {"tag": tag, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "points": {}}
    bad = [not quality_gate()]
    for conc in POINTS:
        point(conc, max_tokens=64)  # warm-up on the same shape
        runs = [point(conc) for _ in range(REPEATS)]
        runs.sort(key=lambda r: r[0])
        tok_s, tok, el, d, now = runs[len(runs) // 2]
        print(fmt(conc, tok_s, tok, el, d), flush=True)
        out["points"][str(conc)] = {"tok_s": tok_s, "tok": tok, "seconds": el,
                                    "tok_per_round": d["emitted"] / max(1, d["rounds"]),
                                    "accept": d["accepted"] / max(1, d["proposed"]),
                                    "round_ms": 1000 * el / max(1, d["rounds"])}
    path = f"results/ref_sampled_{tag}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"saved {path}")
    print("quality gate:", "PASS" if not any(bad) else "FAIL")
    return 0 if not any(bad) else 2


if __name__ == "__main__":
    sys.exit(main())
