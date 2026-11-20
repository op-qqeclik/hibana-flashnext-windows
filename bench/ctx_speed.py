"""Decode speed against context length: the same short question behind a padded history."""
import json, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30004"
unit = "The sparse attention indexer keeps a per-page key so a long conversation stays cheap to search. "
for target in (2000, 20000, 60000, 110000):
    pad = (unit * (target * 4 // len(unit) + 1))[: target * 4]
    msgs = [{"role": "user", "content": pad + "\n\n(背景资料结束)"},
            {"role": "assistant", "content": "收到。"},
            {"role": "user", "content": "用一句话说明内存带宽为什么决定推理上限。"}]
    body = json.dumps({"model": "RadixArk/Qwen3.8-Flash-Next-NVFP4", "messages": msgs,
                       "temperature": 0.0, "max_tokens": 64, "seed": 5,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode())
    dt = time.time() - t0
    u = d.get("usage") or {}
    n = u.get("completion_tokens") or 0
    print(f"  ctx={u.get('prompt_tokens'):>7} tok | wall={dt:6.2f}s | out={n:3d} | {n / dt:6.1f} tok/s")
