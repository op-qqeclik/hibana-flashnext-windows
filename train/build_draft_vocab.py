"""Frequency-ranked draft vocabulary: the top-N ids a draft head may propose, from a real corpus.

`proposers/draft_vocab.py` ranks by the tokenizer's construction order (byte-level BPE id order), which
puts every CJK/multibyte token past the cut and collapses Chinese drafting. This ranks by observed
frequency in an on-policy corpus instead and always keeps the special/added tokens.
"""
import collections
import json
import sys

import torch


def main() -> None:
    corpus = sys.argv[1]
    out = sys.argv[2]
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 32768
    counts = collections.Counter()
    specials = set()
    for line in open(corpus, encoding="utf-8"):
        row = json.loads(line)
        counts.update(row["input_ids"])
    snap = "/home/qqelick/.cache/huggingface/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594"
    tok = json.loads(open(f"{snap}/tokenizer.json", encoding="utf-8").read())
    specials = {a["id"] for a in tok.get("added_tokens", [])}
    ranked = [i for i, _ in counts.most_common()]
    keep = sorted(specials | set(ranked[: max(0, size - len(specials))]))
    ids = torch.tensor(keep, dtype=torch.int64)
    torch.save(ids, out)
    print(f"corpus tokens={sum(counts.values())} distinct={len(counts)} specials={len(specials)}")
    print(f"shortlist={ids.numel()} of {len(tok.get('model', {}).get('vocab', {})) or 248320} rows -> {out}")
    print("coverage of corpus tokens:", f"{sum(counts[i] for i in keep) / sum(counts.values()):.4f}")


main()
