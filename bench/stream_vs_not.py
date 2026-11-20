"""Same prompt, same params: non-streaming vs streaming. DSH uses streaming; our benchmarks did not."""
import json, time, urllib.request

BASE = "http://127.0.0.1:30004"
PROMPT = ("Explain in detail why memory bandwidth sets the ceiling for large-model inference, "
          "covering weights streaming, KV-cache growth, arithmetic intensity, batching, and the "
          "difference between prefill and decode. Be thorough.")


def run(stream: bool) -> None:
    body = json.dumps({"model": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
                       "messages": [{"role": "user", "content": PROMPT}],
                       "temperature": 0.0, "max_tokens": 400, "seed": 5, "stream": stream,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; chunks = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        if not stream:
            d = json.loads(r.read().decode())
            dt = time.time() - t0
            u = d.get("usage") or {}
            n = u.get("completion_tokens") or 0
            print(f"  non-stream: {dt:6.2f}s  out={n:4d}  -> {n / dt:6.1f} tok/s")
            return
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
            if (ch.get("delta") or {}).get("content"):
                if first is None:
                    first = time.time()
                chunks += 1
            if (d.get("usage") or {}).get("completion_tokens"):
                out = d["usage"]["completion_tokens"]
    end = time.time()
    dec = (chunks - 1) / (end - first) if first and chunks > 1 else 0
    print(f"  stream:     {end - t0:6.2f}s  chunks={chunks:4d}  "
          f"ttft={(first - t0) * 1000:5.0f}ms  decode={dec:6.1f} tok/s")


run(False)
run(True)
run(False)
run(True)
