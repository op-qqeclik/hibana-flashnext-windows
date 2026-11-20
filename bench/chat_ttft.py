"""A growing conversation, the way DSH uses the engine: each turn reuses the previous prefix and adds one message."""
import json, sys, time, urllib.request

BASE = sys.argv[1]
unit = ("We keep finding that an instrument which looks authoritative is measuring the wrong thing. ")
history = [{"role": "user", "content": (unit * 220)[:8000] + "\n\nSummarise in one sentence."}]
for turn in range(6):
    body = json.dumps({"model": "RadixArk/Qwen3.8-Flash-Next-NVFP4", "messages": history,
                       "temperature": 0.0, "max_tokens": 1, "seed": 5,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode())
    print(f"  turn{turn}: {(time.time() - t0) * 1000:6.0f} ms  prompt_tokens={(d.get('usage') or {}).get('prompt_tokens')}")
    history += [{"role": "assistant", "content": "ok"},
                {"role": "user", "content": "and now? " + "x" * 40}]
