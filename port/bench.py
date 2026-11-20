import json, time, urllib.request
BASE = "http://127.0.0.1:30004"
prompt = ("\u7528\u4e2d\u6587\u5199\u4e00\u6bb5\u5173\u4e8e\u201c\u672c\u5730\u5927\u6a21\u578b\u5728 Windows \u4e0a\u8dd1\u8d77\u6765\u201d\u7684\u6280\u672f\u968f\u7b14\uff0c"
          "\u4e0d\u8d85\u8fc7 300 \u5b57\uff0c\u8bb2\u6e05\u695a\u4e3a\u4ec0\u4e48\u96be\u3001\u96be\u5728\u54ea\u91cc\u3002")
body = {"model": "hibana", "messages": [{"role": "user", "content": prompt}], "max_tokens": 512,
        "temperature": 0.4, "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=600) as r:
    out = json.loads(r.read().decode("utf-8"))
elapsed = time.time() - t0
u = out["usage"]
text = out["choices"][0]["message"]["content"]
open(r"B:\hibana-win\results\zh_sample.txt", "w", encoding="utf-8").write(text)
print(f"latency {elapsed:.2f}s prompt {u['prompt_tokens']} completion {u['completion_tokens']} "
      f"speed {u['completion_tokens']/elapsed:.1f} tok/s")
print(repr(text[:120]))
