import json
import time
import urllib.request

BASE = "http://127.0.0.1:30004"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def post(path, payload, timeout=300):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


info = get("/get_server_info")
print("== server ==")
print("model:", info.get("model_path"))
print("context:", info.get("context_length"), "max_running:", info.get("max_running_requests"),
      "kv_pages:", info.get("kv_pages"), "state_slots:", info.get("state_slots"))
print("profile:", (info.get("config") or {}).get("profile"), "draft:", (info.get("speculative") or {}).get("draft"))
print("vision:", info.get("vision"), "memory:", info.get("memory_gib"))

cases = [
    ("zh", "\u7528\u4e09\u53e5\u8bdd\u4ecb\u7ecd\u4f60\u81ea\u5df1\uff0c\u5e76\u8bf4\u660e\u4f60\u8dd1\u5728\u4ec0\u4e48\u786c\u4ef6\u4e0a\u3002", 200),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.", 200),
]
for name, prompt, max_tokens in cases:
    body = {"model": "hibana", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.3, "chat_template_kwargs": {"enable_thinking": False}}
    started = time.time()
    out = post("/v1/chat/completions", body)
    elapsed = time.time() - started
    usage = out.get("usage") or {}
    text = out["choices"][0]["message"]["content"]
    print(f"== chat [{name}] latency {elapsed:.2f}s, prompt {usage.get('prompt_tokens')}, "
          f"completion {usage.get('completion_tokens')}, "
          f"{usage.get('completion_tokens', 0) / elapsed:.1f} tok/s ==")
    print(text.strip()[:600])
    print()
