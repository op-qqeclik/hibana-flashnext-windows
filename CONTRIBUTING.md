# Contributing

Thanks for looking. A few things about this repo that shape what a good contribution looks like:

## What is most useful

1. **A measurement that contradicts ours.** Open an issue with the command you ran, the date,
   the prompt set, greedy vs temp-0.3, and whether you warmed up. A contradicting number with
   conditions attached is worth more than a patch.
2. **A clean port to a third platform** (Linux + discrete GPU, WSL2 with GPU paravirt, a 32 GB card).
   The `port/winpatch.py` anchor list is a good map of every place the engine assumes its original host.
3. **A fix for one of the five open problems** in [`docs/06-postmortem.md`](docs/06-postmortem.md) §7.

## What will be declined

- Anything that relaxes decoding quality to gain speed (widening top-k, dropping rejection
  sampling, `--throughput`). This project's contract is "lossless or it doesn't count".
- Speed claims without a bit-exactness gate (`bench/refcheck.py cmp base` must print `IDENTICAL`).
- Speed claims measured on a *different prompt set* than the one being compared against.
  This is exactly the mistake documented in `docs/06` §2.1.

## Rules for benchmark PRs

```
1. python bench\refcheck.py cmp base      -> IDENTICAL
2. A/B interleaved, 2-3 runs each, median
3. report: GPU, driver, prompt set, greedy/temp, warm-up state, co-tenant state
4. if you touched the draft head: bench\eval_wide120.py only, never c1_prose.py
```

## Style

- Scripts: no new dependencies beyond what `port/setup_windows.py` installs.
- Docs: keep the "date + instrument + prompt set" convention on every number; mark retracted
  numbers as 【已作废】/【retracted】 instead of deleting them.
- Engine changes: propose them as an anchored string patch in `port/winpatch.py` (idempotent,
  fails loudly when the anchor moves) — not as a copy of the engine file.

## License

By contributing you agree your work is licensed under MIT (see `LICENSE`).
