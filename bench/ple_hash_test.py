"""Offline (CPU, no GPU, no checkpoint) check that Flash-Next's two PLE row hashes agree.

`python ple_hash_test.py <path to ple.py>`: the engine hashes a verification block two ways -- `PleLayer.ngram_ids`
(the torch path) and `PleLayer.column_ids` (the numpy path the batched gather uses, hashed column by column and
stacked request-major). If they ever disagree, the target model reads the wrong table rows and every acceptance
number becomes meaningless, so this compares them bit for bit on random blocks (k = 1..5, with EOS rewriting).

Runs in a throwaway package shim: ple.py imports only numpy/torch at module level, so this needs nothing from the
engine -- which is what makes it usable as a regression test on a laptop or in CI.
"""
import os
import random
import shutil
import sys
import tempfile

import numpy as np
import torch


def load(ple_path: str):
    tmp = tempfile.mkdtemp(prefix="ple-hash-")
    pkg = os.path.join(tmp, "hibana", "model", "flashnext")
    os.makedirs(pkg)
    for d in (os.path.join(tmp, "hibana"), os.path.join(tmp, "hibana", "model"), pkg):
        open(os.path.join(d, "__init__.py"), "w").close()
    shutil.copy(ple_path, os.path.join(pkg, "ple.py"))
    sys.path.insert(0, tmp)
    from hibana.model.flashnext.ple import PleLayer
    return PleLayer


class Stub:
    """PleLayer's hash parameters without a checkpoint (any consistent values exercise the arithmetic)."""
    ngram = 3
    heads_per_ngram = 8
    heads = 16


def make_stub(PleLayer):
    s = Stub()
    # on the class, so they bind to the stub instance like the real methods do
    Stub.ngram_ids, Stub.column_ids, Stub._hash = PleLayer.ngram_ids, PleLayer.column_ids, PleLayer._hash
    g = torch.Generator().manual_seed(7)
    s.multipliers = torch.randint(1, 2**40, (3,), generator=g, dtype=torch.int64)
    s.vocab_sizes = torch.randint(10**5, 10**6, (s.heads,), generator=g, dtype=torch.int64)
    s.offsets = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int64), s.vocab_sizes[:-1]]), 0)
    s.eos = 151645
    return s


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    PleLayer = load(sys.argv[1])
    s = make_stub(PleLayer)
    rng = random.Random(3)
    bad = 0
    trials = 200
    for trial in range(trials):
        k = rng.randint(1, 5)
        seq = [rng.randrange(100, 150000) for _ in range(k + 2)]  # two history tokens, then the pending token + drafts
        if trial % 7 == 0:
            seq[rng.randrange(len(seq))] = s.eos  # exercise the EOS rewriting
        ids = torch.tensor(seq[2:], dtype=torch.int64)
        hist = torch.tensor(seq[:2], dtype=torch.int64)
        want = s.ngram_ids(ids, hist, s.eos).numpy()
        arr = np.asarray(seq, dtype=np.int64)
        cols = [s.column_ids(arr[2 + j:3 + j], arr[1 + j:2 + j], arr[0 + j:1 + j], s.eos) for j in range(k)]
        got = np.stack(cols, axis=1).reshape(-1, s.heads)
        if not np.array_equal(want, got):
            bad += 1
            if bad == 1:
                print("first mismatch for", seq)
    print(f"{trials - bad}/{trials} blocks identical (k = 1..5, EOS cases included)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
