"""Unit check: the batched numpy PLE hash (`column_ids`, stacked per column) == the torch path (`ngram_ids`).

Both take a verification block: two history tokens then the pending token and the drafts. The engine's verify
rows must be identical either way, or the target model reads the wrong table rows.
"""
import random
import sys

import numpy as np
import torch

sys.path.insert(0, r"B:\hibana-win")


class Stub:
    """PleLayer's hash parameters, without loading a checkpoint."""
    from hibana.model.flashnext.ple import PleLayer
    ngram = 3
    heads_per_ngram = 8
    heads = 16
    ngram_ids = PleLayer.ngram_ids
    column_ids = PleLayer.column_ids
    _hash = PleLayer._hash


def make():
    s = Stub()
    g = torch.Generator().manual_seed(7)
    s.multipliers = torch.randint(1, 2**40, (3,), generator=g, dtype=torch.int64)
    s.vocab_sizes = torch.randint(10**5, 10**6, (s.heads,), generator=g, dtype=torch.int64)
    s.offsets = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int64), s.vocab_sizes[:-1]]), 0)
    s.eos = 151645
    return s


def torch_block(s, seq):  # seq: [T] python list, with 2 history tokens in front
    ids = torch.tensor(seq[2:], dtype=torch.int64)
    hist = torch.tensor(seq[:2], dtype=torch.int64)
    return s.ngram_ids(ids, hist, s.eos).numpy()


def numpy_block(s, seq):
    arr = np.asarray(seq, dtype=np.int64)
    k = len(seq) - 2
    cols = [s.column_ids(arr[2 + j:3 + j], arr[1 + j:2 + j], arr[0 + j:1 + j], s.eos) for j in range(k)]
    return np.stack(cols, axis=1).reshape(-1, s.heads)


def main():
    s = make()
    rng = random.Random(3)
    bad = 0
    for trial in range(200):
        k = rng.randint(1, 5)
        seq = [rng.randrange(100, 150000) for _ in range(k + 2)]
        if trial % 7 == 0:
            seq[rng.randrange(len(seq))] = s.eos  # exercise the EOS rewriting
        a, b = torch_block(s, seq), numpy_block(s, seq)
        if not np.array_equal(a, b):
            bad += 1
            if bad == 1:
                print("first mismatch", seq, "\n torch", a[0], "\n numpy", b[0])
    print(f"{200 - bad}/200 blocks identical (k = 1..5, EOS cases included)")
    return 0 if bad == 0 else 1


sys.exit(main())
