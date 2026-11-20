"""Streaming: separate TTFT (prefill) from steady-state decode tok/s, against context length."""
import json, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30004"
unit = "The sparse attention indexer keeps a per-page key so a long conversation stays cheap to search. "
for target in (2000, 20000, 60000, 110000):
    pad = (unit * (target * 4 // len(unit) + 1))[: target * 4]
    msgs = [{"role": "user", "content": pad + "\n\n(背景结束)"},
            {"role": "assistant", "content": "收到。"},
            {"role": "user", "content": "详细解释内存带宽为什么决定推理上限，至少三百字。"}]
    body = json.dumps({"model": "RadixArk/Qwen3.8-Flash-Next-NVFP4", "messages": msgs,
                       "temperature": 0.0, "max_tokens": 400, "seed": 5, "stream": True,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; n = 0; ptok = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            ch = (d.get("choices") or [{}])[0]
            piece = (ch.get("delta") or {}).get("content")
            u = d.get("usage") or {}
            if u.get("prompt_tokens"):
                ptok = u["prompt_tokens"]
            if piece:
                if first is None:
                    first = time.time()
                n += 1
    end = time.time()
    if first and n > 1:
        dec = (n - 1) / (end - first)
        print(f"  ctx={str(ptok):>7} | TTFT={(first - t0) * 1000:7.0f} ms | decode={dec:6.1f} tok/s | chunks={n}")
    else:
        print(f"  ctx={ptok} | no streamed content (chunks={n})")
