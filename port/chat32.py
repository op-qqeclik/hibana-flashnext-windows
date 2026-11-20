import json, time, urllib.request
BASE = "http://127.0.0.1:30004"
body = {"model": "hibana", "messages": [{"role": "user", "content": "Say hello in Chinese, one short sentence."}],
        "max_tokens": 32, "temperature": 0.0}
req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.loads(r.read().decode())
    print("elapsed", round(time.time() - t0, 1))
    print(json.dumps(out.get("usage"), ensure_ascii=False))
    print(out["choices"][0]["message"]["content"][:300])
except Exception as exc:
    print("FAIL after", round(time.time() - t0, 1), type(exc).__name__, str(exc)[:200])
