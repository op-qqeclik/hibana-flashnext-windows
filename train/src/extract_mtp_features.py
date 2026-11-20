#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract FlashNext target hyper-stream hidden states (out['norm']) and top-K target logits
from an on-policy corpus for multi-step MTP KL distillation."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hibana-dir", required=True, help="Path to hibana repository root")
    ap.add_argument("--corpus", required=True, help="Input corpus.jsonl path")
    ap.add_argument("--out-dir", required=True, help="Output directory for features and base tensors")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--shard-size", type=int, default=450)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.hibana_dir).resolve()))
    from hibana import target as targets
    from hibana.engine.context import scratch_prefill_context
    from hibana.weights.safetensors import SnapshotWeights

    out_dir = Path(args.out_dir)
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in Path(args.corpus).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(rows)} sequences from {args.corpus}", flush=True)

    tgt = targets.get("flashnext")
    snap = tgt.snapshot_dir()
    model = targets.load_model(tgt, snap, log=print)

    sw = SnapshotWeights(snap)
    if not (out_dir / "target_embed.pt").exists():
        torch.save(sw.get("model.language_model.embed_tokens.weight").cpu(), out_dir / "target_embed.pt")
    if not (out_dir / "target_lm_head.pt").exists():
        torch.save(sw.get("lm_head.weight").cpu(), out_dir / "target_lm_head.pt")
    if not (out_dir / "mtp_init.pt").exists():
        mtp_sd = {k: sw.get(k).cpu() for k in sorted(sw._map.keys()) if k.startswith("mtp.")}
        torch.save(mtp_sd, out_dir / "mtp_init.pt")
        print(f"Saved mtp_init.pt ({len(mtp_sd)} tensors)", flush=True)

    t0 = time.time()
    total_toks = 0
    for s_idx in range(0, len(rows), args.shard_size):
        sp = feat_dir / f"shard_{s_idx // args.shard_size:03d}.pt"
        if sp.exists():
            print(f"Skip existing {sp.name}", flush=True)
            continue
        items = []
        with torch.inference_mode():
            for r in rows[s_idx : s_idx + args.shard_size]:
                ids = r["input_ids"]
                t = len(ids)
                pl = int(r["prompt_len"])
                if t < pl + 16 or pl < 2:
                    continue
                ids_t = torch.tensor(ids, dtype=torch.int64, device=model.device)
                ctx = scratch_prefill_context(t, model.device)
                out = model.forward(ids_t, ctx)
                norm = out["norm"]
                pred_in = norm[pl - 1 : t - 1]
                mixed = model.mixer.mix(pred_in)[0]
                t_logits = model.head_logits(mixed)
                t_val, t_idx = t_logits.topk(args.topk, dim=-1)
                items.append({
                    "domain": r["domain"],
                    "thinking": bool(r.get("thinking", False)),
                    "prompt_len": pl,
                    "input_ids": torch.tensor(ids, dtype=torch.int32),
                    "norm": norm.cpu(),
                    "top_val": t_val.cpu(),
                    "top_idx": t_idx.to(torch.int32).cpu(),
                })
                total_toks += t
        torch.save(items, sp)
        dt = max(1e-3, time.time() - t0)
        print(f"Wrote {sp.name}: {len(items)} seqs, total_toks={total_toks} ({total_toks/dt:.0f} tok/s)", flush=True)


if __name__ == "__main__":
    main()
