# Hibana on Windows

**Running a 125B-A5B MoE (`Qwen3.8-Flash-Next`, NVFP4 trunk + 3-bit experts) natively on one RTX PRO 6000
— no WSL, no Docker — at 2035 tok/s aggregate, and everything we learned while getting there.**

![platform](https://img.shields.io/badge/platform-Windows%2010%2F11%20x64-0078D4)
![gpu](https://img.shields.io/badge/GPU-RTX%20PRO%206000%20Blackwell%2096GB-76B900)
![engine](https://img.shields.io/badge/engine-Hibana%200.1.0-black)
![license](https://img.shields.io/badge/license-MIT-blue)
![status](https://img.shields.io/badge/status-research%20port%20%C2%B7%20honest%20numbers-orange)

> **This is a port kit + engineering playbook, not a fork.** It contains no engine source and no model
> weights (see [NOTICE](NOTICE)). What it does contain: an idempotent Windows patcher, a one-command
> toolchain installer, the production launch script, the full measurement harness, the MTP draft-head
> distillation workflow, and 45 rounds of measured optimisation history — **including the ones that failed**.

---

## Results at a glance

Single machine: **RTX PRO 6000 Blackwell 96 GB · Windows 10 19045 · driver 616.92 · 191 GB RAM**.
Model: `Qwen3.8-Flash-Next-NVFP4` (125B total / ~5B active) + W3 3-bit experts (48 GB) + 47.7 GB host-resident PLE table.
Greedy decoding, 5-point concurrency sweep, `--max-batch 16`, depth table `1:8,2:6,4:3,8:3,16:4`.

| concurrency | tok/s (greedy) | tok/s (temp 0.3) | bandwidth-derived ceiling | achieved |
|---|---|---|---|---|
| c1  | **273.5** | 186.0 | 455  | 60 % |
| c2  | **527.7** | 324.0 | 700  | 75 % |
| c4  | **724–847** | 524.8 | 930  | 78–91 % |
| c8  | **1321.8** | 831.2 | 1570 | 84 % |
| c16 | **2035.5** | 1298.8 | 1690 | **120 %** (ceiling estimate was wrong — see docs/04) |

Prose single-stream (what you actually feel while chatting): **en 325.6 / zh 332.3 tok/s**.
Against the DGX Spark (GB10) production recipe this is **1.8–2.4× single-stream** and **~3.1× at 16-way**.

Cold start **~4 min** (47.7 GB PLE table must land in the Windows file cache), warm restart ~35 s.

> Every one of these numbers has a date, a prompt set and a measurement discipline attached.
> Read [docs/04-benchmarks.md](docs/04-benchmarks.md) before quoting them, and
> [docs/06-postmortem.md](docs/06-postmortem.md) before believing any *draft-head* speedup claim.

---

## What makes this hard (and why the repo exists)

The engine was built for a GB10 "Spark" box: aarch64 Linux, **unified memory**, `273 GB/s`,
where the per-token weight traffic (~6.5 GB) takes ~24 ms and *hides every other cost*.
The RTX PRO 6000 has `1792 GB/s` and **discrete** VRAM under WDDM — the same 6.5 GB takes ~3.6 ms,
so all the costs that were previously hidden become the bottleneck:

1. **Host↔device handshakes cost 1.85 ms each under WDDM** (10–30 µs on Linux), and the spec-decode loop
   does one per draft column. Fix: never fight the handshake — **reduce the number of them per token**.
2. **The PLE table (47.7 GB) lives on the host** and needs 16 random rows per token, computed *after*
   the draft token is known. Fix: `mmap` + fancy indexing instead of a per-call thread pool.
3. **~500–600 tiny kernel launches per decode step** were masked by 24 ms of bandwidth; here they dominate.
4. **Linux-only APIs** in the engine (`/proc/meminfo`, `statvfs`, `fcntl`, `preadv`, `fadvise`, `uname`),
   and **flashinfer's JIT assumes Linux** (ninja, `-fPIC`, `c++`, `.so`).

The solutions to all four are in [docs/01-porting.md](docs/01-porting.md) and [docs/02-tuning.md](docs/02-tuning.md).

---

## Hardware / software matrix (the "same GPU" contract)

| | Required |
|---|---|
| GPU | **RTX PRO 6000 Blackwell 96 GB** (SM 120, `compute_120a`) — the 3-bit profile needs ≥90 GB VRAM |
| OS | Windows 10 21H2+ / Windows 11, x64. **No WSL, no Docker.** |
| RAM | ≥128 GB (191 GB tested; 47.7 GB PLE table is page-cached) |
| Disk | ≥250 GB free on a **local NVMe** (135 GB trunk + 48 GB W3 + ~10 GB venv/cache). USB drives measured at 4–20 MB/s — unusable |
| CUDA | 13.0 (a `C:\cuda13` junction is created to dodge the space in `Program Files`) |
| MSVC | **VS 2019/2022 Build Tools (14.4x)**. CUDA 13 refuses the 2026 toolset (`cudafe++` → `0xC0000005`) |
| Python | 3.12.11 via `uv` |

Anything else (RTX 5090 32 GB, L40S, A100, GB10 itself) will not fit the 3-bit profile as configured.
Porting notes for a 32 GB card are in [docs/05-troubleshooting.md](docs/05-troubleshooting.md#smaller-cards).

---

## Quick start

```powershell
# 0) You must already have, legally:
#    - a Hibana 0.1.0 checkout you are licensed to patch   ->  $env:HIBANA_SRC
#    - Hugging Face access to the Qwen3.8-Flash-Next weights
git clone https://github.com/<you>/hibana-flashnext-windows.git
cd hibana-flashnext-windows

copy setup\config.example.ps1 setup\config.ps1   # edit paths once
powershell -ExecutionPolicy Bypass -File setup\preflight.ps1   # 12 checks, tells you exactly what's missing
powershell -ExecutionPolicy Bypass -File setup\bootstrap.ps1   # toolchain + patches + weights + warm-up
launch\run-flashnext.bat                                        # -> http://127.0.0.1:30004
```

`bootstrap.ps1` is idempotent: re-running it skips finished stages, re-applies patches only when the
target string is still unpatched, and never touches files outside the paths in `config.ps1`.

Verify:

```powershell
python port\verify.py            # /health, /v1/models, /get_server_info, one zh + one code completion
python bench\refcheck.py cmp base  # bit-exactness gate; must print IDENTICAL
python bench\sweep.py 1,2,4,8,16 700   # the 5-point concurrency table above
```

---

## Repository map

```
setup/     preflight.ps1 · bootstrap.ps1 · config.example.ps1 · install stage scripts
port/      winpatch.py (24 idempotent engine patches) · _plat.py (POSIX shim) · win_compat.h
           setup_windows.py (toolchain) · verify.py
launch/    run-flashnext.bat (production) · run-flashnext-*.bat (A/B variants) · restart.ps1
perf/      reference diffs for the six hot files + hc-bf16 / wide-9rows patches
bench/     refcheck.py (bit-exact) · sweep.py · accept_sampled.py · soak.py · c1_prose.py
           eval_prompts_120.json (the only non-contaminated prompt set) ...
train/     build_corpus.py · distill recipe · eval_accept.py (accept_len on held-out prose)
docs/      01 porting · 02 tuning · 03 MTP distillation · 04 benchmarks · 05 troubleshooting · 06 postmortem
reference/ HANDOFF.md (3 000 lines of raw round-by-round log) · LESSONS.md · kernel-fusion-spec.md
```

---

## The distillation story, in one paragraph (because it is the most useful negative result here)

We ran the full 4-step self-feeding KL distillation pipeline from
[`SAA-Brain-3.8-flash-next-Hibana`](https://github.com/CharAzinabulu0079/SAA-Brain-3.8-flash-next-Hibana)
on our own prose corpus (0.35 M → 1.22 M → 1.37 M tokens, 8-step self-feeding rollout, CE + KL_top32, rank-16 LoRA on
512 experts; the trainer we ship has that constant at 4 — set it to 8 to reproduce the v-series), produced 23 draft heads (`v2 … v23`), and watched the in-trainer metric climb
(`accept_len` 2.668 → 3.017, **+13.1 %**) on a *held-out* prose set. On the engine it looked like
**+11 % single-stream** — and then we built a fresh 120-prompt set with sentence patterns disjoint from the
training corpus, and all five heads landed within **2.3 %** of the stock head
(0.5912 vs 0.5912 accept, 3.986 vs 4.030 tok/round).

**Conclusion: narrow-domain distillation of the MTP head generalises ≈ nothing.** The stock head is already
close to the architecture's generalisation limit; what actually moved the needle was making the
hyper-connections **bf16 instead of fp8** (+4.8 % single-stream, and it *reads more bytes*).
The whole workflow, the numbers, and the methodology that caught the contamination are in
[docs/03-mtp-distillation.md](docs/03-mtp-distillation.md) and [docs/06-postmortem.md](docs/06-postmortem.md).

---

## Composition of the speedup (what to copy if you port to a third platform)

| # | Change | Mechanism | Measured gain | Kept? |
|---|---|---|---|---|
| 1 | PLE gather via `mmap` fancy-indexing | 2048 rows 36 ms → 0.67 ms | c16 ×2.1 | ✅ |
| 2 | Per-concurrency spec-depth table | handshakes ∝ depth | c16 492→903 | ✅ |
| 3 | Host-side PLE thread-pool fix | pool+handle reuse, no per-call executor | single-stream 1.5–1.7× | ✅ |
| 4 | `HIBANA_PLAN_CACHE` (FA2 schedule cache) | plan 0.29→0.07 ms/round | +4 % c8/c16 | ✅ |
| 5 | split-K divisor relaxation | 80→200 blocks for K=2560 | zh +14 %, c4 +13 % | ✅ |
| 6 | SM 12.1 hard-gate widened to 12.0 | graph replay of 66 fused kernels | code +8 %, accept 0.517→0.563 | ✅ |
| 7 | bf16 hyper-connections | fp8 `_scaled_mm` tile is inefficient on SM120 | **+4.8 %** single-stream | ✅ |
| 8 | PLE preload into file cache | cold pages → resident | c8 +33 % | ✅ |
| 9 | batched PLE rows | one gather instead of 9 | −9 % c4, −12 % c16 | ❌ reverted |
| 10 | `verify_state=snapshots` | removes reconstruct | doesn't fit in 96 GB | ❌ reverted |
| 11 | NVFP4 KV cache | fewer bytes | FP4-KV module won't compile in this port | ❌ |
| 12 | narrow-domain MTP distillation | higher accept | ≈0 % on a clean 120-prompt set | ❌ (see docs/06) |

Arrow `→` numbers are all from the same machine, A/B interleaved, median of 2–3 runs, with a
bit-exactness gate (`refcheck.py`) in front of every accepted change.

---

## Citing / attribution

The distillation recipe and the "lossless speedup" methodology come from
`CharAzinabulu0079/SAA-Brain-3.8-flash-next-Hibana`. The 3-bit W3 kernel layout, PLE design and MTP
proposer come from the Hibana engine. This repo only documents how they behave on a discrete-GPU Windows
host, plus the glue that makes it run there.

If you use the measured numbers, please link this repo **and** quote the conditions (GPU, driver,
prompt set, greedy vs temp-0.3). They are not comparable across prompt sets — that is precisely the
lesson in [docs/06-postmortem.md](docs/06-postmortem.md).

MIT for everything original here — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
