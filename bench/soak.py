"""Soak: 4 minutes of sustained mixed load (concurrency 8, alternating prose/code, 200-500 tokens), checking the
service stays healthy and its throughput does not decay."""
import json, random, time, urllib.request, concurrent.futures as cf

BASE = "http://127.0.0.1:30004"
PROMPTS = [
    "Write a technical paragraph about paged attention, 200 words.",
    "Write a Python function that merges two sorted lists, with a docstring.",
    "\u7528\u4e2d\u6587\u5199\u4e00\u6bb5\u5173\u4e8e\u57ce\u5e02\u6df1\u591c\u7684\u6563\u6587\uff0c\u4e0d\u8d85\u8fc7 200 \u5b57\u3002",
]

def post(prompt, max_tokens):
    body = {"model": "hibana", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.3,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode())["usage"]["completion_tokens"]

def round_of(n):
    prompt = random.choice(PROMPTS)
    tok = random.choice([200, 300, 500])
    return post(prompt, tok)

start = time.time()
window = []
rounds = 0
while time.time() - start < 240:
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        got = list(ex.map(round_of, range(8)))
    wall = time.time() - t0
    window.append(sum(got) / wall)
    rounds += 1
    if rounds % 5 == 0:
        print(f"t={time.time()-start:5.0f}s rounds={rounds} last={window[-1]:.0f} tok/s", flush=True)

print(f"soak done: {rounds} rounds, first5 {sum(window[:5])/5:.0f} tok/s, "
      f"last5 {sum(window[-5:])/5:.0f} tok/s, min {min(window):.0f}, max {max(window):.0f}")
