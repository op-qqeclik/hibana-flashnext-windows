#!/usr/bin/env python3
"""On-policy long-form prose corpus for MTP distillation.

The deployed workload is long English and Chinese prose (explanatory writing, essays), so the draft head
must be distilled on that: long single-turn continuations at a low temperature on many distinct topics,
not on chat turns or code. Sibling of `generate_corpus.py`; same output schema (input_ids only, so
`extract_mtp_features.py` accepts it unchanged).
"""
import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer

EN_TOPICS = [
    "how a paged KV cache works in an LLM serving engine, covering page tables, prefix caching and eviction",
    "why memory bandwidth rather than arithmetic throughput sets the ceiling for large-model inference",
    "how distributed consensus protocols keep logs consistent across network partitions",
    "how a modern operating system scheduler balances latency against throughput",
    "what happens inside a GPU when a kernel's working set stops fitting in L2",
    "how speculative decoding trades extra compute for lower per-token latency",
    "the history and engineering tradeoffs of virtual memory and address translation",
    "how TCP congestion control adapts to loss, delay and bufferbloat",
    "why quantization changes a transformer's numerics and where the error accumulates",
    "how compilers schedule instructions around memory latency",
]
ZH_TOPICS = [
    "深夜的城市街道、灯光和下班回家的人，写成一篇完整散文",
    "一个南方小镇的车站，以及很多年以后回头看那一天",
    "为什么大模型推理受限于内存带宽，而算力并不总是瓶颈",
    "老屋的院子、夏天的蝉声和搬走之后的空落",
    "一次长途火车上的陌生人，与那些没说完的话",
    "海岸边的风、退潮后留下的东西，和一个人独自散步的下午",
    "写一段关于手艺人的长文，写他的一天和他手上留下的痕迹",
    "冬天的早晨、窗户上的雾气和屋里正在烧的水",
    "一个人搬家时翻出的旧物件，以及它们背后的那些年",
    "山里的雨季、泥路和屋檐下的声音，写成一篇完整的散文",
]
EN_STYLE = [
    "Write a long, carefully reasoned essay. About 900 words. Be concrete and thorough.",
    "Write a detailed technical explanation, roughly 900 words, with a clear structure.",
    "Write an in-depth piece for an engineering audience. About 900 words.",
]
ZH_STYLE = [
    "写一篇约九百字的完整文章。要具体、有细节、有层次。",
    "写一篇详细的说明性长文，约九百字，结构清晰。",
    "写一篇长篇散文，约九百字，语言自然，不要提纲。",
]


# --- wide combinatorial topics: domains x angles x forms (thousands of distinct prompts) ---
DOMAINS = [
    "paged attention and KV cache management in an inference engine", "memory bandwidth limits in large-model inference",
    "distributed consensus and log replication", "operating-system CPU scheduling", "GPU L2 cache behaviour",
    "speculative decoding tradeoffs", "quantisation of transformer weights", "tensor-parallel communication patterns",
    "flash attention tiling strategies", "mixture-of-experts routing and load balance", "continual learning and catastrophic forgetting",
    "the physics of heat transfer in data centres", "photolithography and process nodes", "solid-state battery chemistry",
    "CRISPR and off-target effects", "protein folding prediction", "gut microbiome and immune signalling",
    "sleep architecture and memory consolidation", "the economics of semiconductor supply chains", "central bank interest-rate transmission",
    "auction design and market microstructure", "the history of the printing press", "the collapse of the Bronze Age",
    "the Meiji restoration and industrial policy", "medieval trade routes and the Silk Road", "the invention of Writing and record keeping",
    "urban planning and street networks", "architecture and structural load paths", "water treatment and public health",
    "atmospheric circulation and monsoons", "glacier dynamics and sea-level rise", "coral reef ecology",
    "bird migration and magnetic navigation", "octopus neurobiology", "fungal networks and forest nutrient exchange",
    "the philosophy of causation", "epistemic humility in science", "the nature of mathematical proof",
    "game theory and cooperative equilibria", "moral luck and responsibility", "the aesthetics of minimalism",
    "learning a musical instrument as an adult", "long-distance walking and attention", "cooking as applied chemistry",
    "repairing old mechanical watches", "the craft of translation", "keeping a commonplace book",
    "wilderness solitude and reflection", "the soundscape of a city at dawn", "the texture of winter light",
    "railway travel and the perception of distance", "library architecture and silence", "the ritual of morning coffee",
    "a conversation overheard on a night train", "a family workshop closing down", "a lighthouse keeper's routine",
    "a museum guard's afternoons", "an engineer explaining a failure to a customer", "two colleagues disagreeing about a design",
]
ANGLES = [
    "explain the mechanism step by step", "trace the historical development", "compare two competing explanations",
    "argue for a contrarian position", "describe the common misconceptions", "explain what the numbers actually show",
    "walk through a concrete worked example", "discuss the failure modes", "explain the tradeoffs honestly",
    "describe how a practitioner would approach it", "narrate a specific incident", "reflect on what it changed in you",
    "describe the sensory detail of a particular moment", "interview an imagined expert", "explain it to a smart beginner",
    "critique the mainstream account", "describe the state of the art and its limits", "explain the economic incentives",
    "describe the ethical tensions", "reconstruct a decision that went wrong",
]
FORMS = [
    "as a long essay of about 900 words", "as an in-depth explainer for an engineering audience",
    "as a personal narrative", "as a structured analysis with clear sections",
    "as a piece of long-form journalism", "as a reflective meditation",
    "as a technical deep dive", "as a short story built around the subject",
]


def _wide_tasks(count: int, seed: int, lang: str) -> list[tuple[str, float, list[dict]]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[tuple[str, float, list[dict]]] = []
    guard = 0
    while len(out) < count and guard < count * 40:
        guard += 1
        d, a, f = rng.choice(DOMAINS), rng.choice(ANGLES), rng.choice(FORMS)
        key = f"{d}|{a}|{f}"
        if key in seen:
            continue
        seen.add(key)
        dom = "en_prose" if lang == "en" else "zh_prose"
        if lang == "en":
            usr = f"Write about {d}. {a.capitalize()}, {f}."
        else:
            usr = f"请围绕「{d}」写一篇长文。{a}，{f}。用中文自然书写，不要写成条目。"
        out.append((dom, 0.3 if len(out) % 5 else 0.0, [{"role": "user", "content": usr}]))
    return out


# --- technical / reasoning register: many subjects, same form as the deployed workload ---
TECH_DOMAINS = [
    "how a paged KV cache works in an LLM serving engine", "why memory bandwidth sets the inference ceiling",
    "distributed consensus and log replication", "how an OS scheduler balances latency and throughput",
    "what happens when a GPU working set leaves L2", "how speculative decoding trades compute for latency",
    "how transformer weights are quantised to 4 bits", "tensor-parallel communication patterns",
    "flash attention tiling and shared-memory reuse", "mixture-of-experts routing and load balance",
    "the arithmetic intensity of attention versus GEMM", "how a CUDA graph removes launch overhead",
    "why WDDM adds overhead that Linux avoids", "KV cache eviction and prefix reuse policies",
    "how continuous batching keeps a GPU saturated", "roofline analysis for a decode step",
    "why small matrix multiplies waste tensor cores", "how NVLink changes the parallelisation strategy",
    "the cost of a reduction across 188 SMs", "why an fp8 matmul still needs a scale factor",
    "how a compiler fuses elementwise kernels", "the difference between latency- and bandwidth-bound kernels",
    "how a speculative draft head is distilled", "why per-token acceptance decays along a draft chain",
    "the tradeoffs of block-parallel drafts", "how a page table maps logical to physical KV blocks",
    "why an inference engine keeps a persistent workspace", "how tensor cores handle a 9-row GEMM",
    "the cost of a host-device synchronisation", "how memory fragmentation accumulates in a KV pool",
    "how expert parallelism differs from tensor parallelism", "why a 3-bit weight format needs calibration",
    "the physics of HBM stacks and refresh overhead", "how a GPU L2 slices addresses across partitions",
    "what a warp scheduler does when a kernel is latency bound", "why occupancy is not the same as efficiency",
    "how a fused kernel changes the roofline", "why caching a plan helps an attention kernel",
    "how graph capture interacts with dynamic shapes", "the cost of recompiling a kernel at runtime",
    "how a serving engine decides a request is done", "why prefill and decode have different bottlenecks",
    "how quantisation error accumulates across layers", "the role of the residual stream in depth scaling",
    "why a model needs a separate draft head at all", "how speculation handles rejection correctly",
    "the statistics of the acceptance rate under sampling", "why temperature changes acceptance",
    "how a KV cache interacts with long context", "what a request scheduler optimises for",
    "how memory allocators avoid fragmentation on a GPU", "why a small batch leaves SMs idle",
    "how a memory-bound kernel hides latency", "the role of shared memory in a fast softmax",
    "how to reason about the cost of a token", "why two identical models can run at different speeds",
    "how a thermal limit shows up in throughput", "what a co-tenant does to measured bandwidth",
    "how to design an experiment that is not fooled by drift", "why a benchmark must be interleaved",
    "how to separate host overhead from device time", "why a profiler changes what it measures",
    "how to validate a cached plan across configurations", "why a numerical change needs a different gate",
    "how to reason from a budget to a ceiling", "what makes a performance claim falsifiable",
    "how to read a kernel-level profile", "why averaging hides a bimodal distribution",
    "how to attribute time to a subsystem", "why a microbenchmark can mislead about bandwidth",
    "how a bottleneck moves when a component improves", "the economics of one more layer of caching",
    "why a system should be measured at its real batch size", "how to decide a optimisation is not worth it",
]
TECH_ANGLES = [
    "explain the mechanism from first principles", "walk through a concrete worked example",
    "compare two competing explanations", "explain what the numbers actually show",
    "discuss the failure modes and limits", "explain the tradeoffs honestly",
    "describe how a practitioner would approach it", "critique the mainstream account",
    "explain the state of the art and its limits", "explain the economic incentives",
    "explain it to a smart beginner", "reconstruct a decision that went wrong",
    "argue for a contrarian position", "trace how the idea developed historically",
    "explain which measurement would settle it", "describe the common misconceptions",
]
TECH_FORMS = [
    "as a long technical essay of about 900 words", "as an in-depth explainer for an engineering audience",
    "as a structured analysis with clear sections", "as a reasoned argument with a conclusion",
    "as a design review write-up", "as a research-style discussion",
    "as a technical deep dive", "as a careful step-by-step derivation",
]


def _tech_tasks(count: int, seed: int, lang: str) -> list[tuple[str, float, list[dict]]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[tuple[str, float, list[dict]]] = []
    guard = 0
    while len(out) < count and guard < count * 40:
        guard += 1
        d, a, f = rng.choice(TECH_DOMAINS), rng.choice(TECH_ANGLES), rng.choice(TECH_FORMS)
        key = f"{d}|{a}|{f}"
        if key in seen:
            continue
        seen.add(key)
        dom = "en_prose" if lang == "en" else "zh_prose"
        if lang == "en":
            usr = f"Write about {d}. {a.capitalize()}, {f}."
        else:
            usr = f"请围绕「{d}」写一篇长文。{a}，{f}。用中文自然书写，不要写成条目。"
        out.append((dom, 0.3 if len(out) % 5 else 0.0, [{"role": "user", "content": usr}]))
    return out


# --- reasoning register: first-principles derivations, logic, causal chains, proofs, decisions ---
REASON_SUBJECTS = [
    "why the sum of the first n odd numbers is n squared", "why a sorting lower bound is n log n",
    "whether a greedy algorithm is optimal for interval scheduling", "why the halting problem is undecidable",
    "why a cached plan can be reused across shapes", "whether two hypotheses can be distinguished by one experiment",
    "why the acceptance rate of a draft chain is a product of conditional probabilities",
    "whether a memory-bound kernel can be improved by more warps", "why a 9-row GEMM wastes tensor cores",
    "why the roofline changes with batch size", "whether a 5% speedup is statistically real given drift",
    "why the mean of a bimodal distribution misleads", "whether a co-tenant explains a throughput drop",
    "why per-token cost grows with context length", "whether quantisation error compounds across layers",
    "why a sampling temperature changes the acceptance rate", "whether a draft head can be improved by more data",
    "why a numerical change requires a different equivalence gate", "whether the ceiling is bandwidth or latency",
    "why an expert layer is read in full even for one token", "whether a page table can be grown in place",
    "why a graph launch costs more with more nodes", "whether a kernel fusion always helps",
    "why a small benchmark underestimates bandwidth", "whether caching helps when the key is unique",
    "why a proof by contradiction is valid", "whether a correlation implies a mechanism",
    "why an estimate can be right for the wrong reason", "whether a decision rule is consistent under uncertainty",
    "why a discounted reward changes the optimal policy", "whether a contract is enforceable without a witness",
    "why a differential diagnosis must be narrowed by priors", "whether a failure mode can be reproduced deterministically",
    "why a system becomes unstable past a load threshold", "whether a queue is stable under bursty arrivals",
    "why a bisection converges faster than a scan", "whether a hash collision can be forced",
    "why an entropy bound limits compression", "whether a consensus protocol needs a leader",
]
REASON_TASKS = [
    "derive the result from first principles, showing every step",
    "prove or disprove the claim, and say which assumption is load-bearing",
    "find the flaw in the following reasoning and repair it",
    "trace the causal chain, distinguishing correlation from causation",
    "decide the question under the stated constraints and justify each step",
    "estimate the quantity with an explicit quantitative argument",
    "formalise the argument and check whether it is valid",
    "reason about the limit cases and say what they reveal",
    "identify the hidden assumptions and test each one",
    "construct a counterexample, then explain why it works",
    "compare two competing hypotheses and design the test that separates them",
    "step through the algorithm on a concrete input and check the invariant",
    "reduce the problem to something already solved, and justify the reduction",
    "bound the answer from above and below, then close the gap",
    "explain the underlying logic so a reader could rederive it unaided",
    "separate what is proven from what is merely plausible",
]
REASON_FORMS = [
    "as a step-by-step derivation", "as a rigorous written argument",
    "as a worked solution with all intermediate steps", "as a reasoning log that shows false starts",
    "as a proof sketch followed by the gaps", "as a quantitative estimate with stated uncertainties",
    "as a structured analysis that ends in a definite conclusion", "as a dialogue where an objection is answered",
]


def _reason_tasks(count: int, seed: int, lang: str) -> list[tuple[str, float, list[dict]]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[tuple[str, float, list[dict]]] = []
    guard = 0
    while len(out) < count and guard < count * 40:
        guard += 1
        d, a, f = rng.choice(REASON_SUBJECTS), rng.choice(REASON_TASKS), rng.choice(REASON_FORMS)
        key = f"{d}|{a}|{f}"
        if key in seen:
            continue
        seen.add(key)
        dom = "en_reason" if lang == "en" else "zh_reason"
        if lang == "en":
            usr = f"Take up this question: {d}. {a.capitalize()}, {f}. Think through it carefully rather than summarising."
        else:
            usr = f"请处理这个问题：{d}。{a}，{f}。要真正一步步想清楚，而不是概述。"
        out.append((dom, 0.3 if len(out) % 5 else 0.0, [{"role": "user", "content": usr}]))
    return out


TECH_DOMAINS += [
    "how a compiler schedules instructions around a memory stall", "why a branch predictor changes a kernel's cost",
    "how a filesystem journals a metadata update", "why an SSD write amplifies under random small writes",
    "how a database chooses a join order", "why a query planner misestimates cardinality",
    "how a lock-free queue avoids a false sharing cliff", "why a garbage collector pauses under a large heap",
    "how a network stack merges small packets", "why congestion control backs off multiplicatively",
    "how a TLS handshake proves freshness", "why a hash table degrades under adversarial keys",
    "how a consensus round survives a slow follower", "why a clock skew breaks an ordering assumption",
    "how a scheduler avoids priority inversion", "why a cache coherence protocol needs invalidations",
    "how a vector instruction handles a misaligned access", "why a branchless rewrite helps a divergent warp",
    "how an allocation pattern causes fragmentation", "why a page fault costs what it costs",
    "how a memory controller reorders requests", "why a refresh cycle steals bandwidth",
    "how a tensor core multiplies two fp8 tiles", "why a scale factor must be applied after the accumulation",
    "how a softmax is stabilised numerically", "why a layernorm placement changes training stability",
    "how a residual connection keeps gradients scaled", "why a depth-dependent norm needs a different constant",
    "how a speculative chain computes its expected length", "why a rejection sampling step is exact",
    "how a verification batch shares one forward pass", "why an acceptance rate changes with temperature",
    "how a draft block differs from a serial chain", "why a block-parallel draft needs a mask token",
    "how a KV page is allocated and freed", "why a prefix cache hit saves a prefill",
    "how a serving engine batches ragged sequences", "why a p99 latency hides behind a queue",
    "how a request timeout interacts with a retry policy", "why an idempotency key prevents a double write",
    "how a distributed cache stays consistent", "why a two-phase commit blocks on a coordinator",
    "how a lock upgrade can deadlock", "why a seqlock is safe for a single writer",
    "how a benchmark avoids a warm-up artifact", "why an interleaved A/B is required under drift",
    "how a profiler changes the workload it measures", "why a per-kernel table can mislead about a step",
    "how a host-device copy shows up in a step trace", "why a graph launch scales with node count",
    "how a numerical change forces a different acceptance gate", "why a token-level md5 is not a quality metric",
    "how to bound an optimisation's remaining upside", "why a measured ceiling can be mistaken for a physical one",
    "how to tell a latency bound from a bandwidth bound", "why a small batch wastes a wide machine",
    "how a fusion trades bytes for launches", "why an occupancy hint can hurt a bandwidth kernel",
    "how to choose between a GEMV and a GEMM", "why a matrix shape decides the tensor-core path",
    "how an expert routing decision scales with batch", "why a shared expert is read every step",
    "how a quantisation scale is calibrated per group", "why a 3-bit format needs a reconstruction step",
    "how a weight layout affects a vectorised read", "why a transposed layout changes coalescing",
    "how an activation checkpoint saves memory", "why a recomputation cost can be cheaper than a store",
    "how a pipeline parallel schedule hides a bubble", "why a micro-batch count decides a bubble size",
    "how a tensor parallel reduction is synchronised", "why a ring all-reduce saturates a link",
]


# --- skeleton register: a fixed set of REASONING SKELETONS rendered under many PROSE SURFACES ---
# Theory (user): the reasoning structure is the bone; adjectives and descriptive language are installed on it.
# So the same skeleton must appear as an essay, a scene, a script, a dialogue, a news feature...
SKELETONS = [
    ("claim_then_decompose", "state a claim, break it into its necessary conditions, then test each condition and conclude"),
    ("mechanism_chain", "trace a causal mechanism link by link from an input to an observable output, and name the rate-limiting step"),
    ("assumption_audit", "list the hidden assumptions behind a belief, and show which one is load-bearing by removing it"),
    ("counterexample", "propose an intuitive rule, construct the case that breaks it, then repair the rule"),
    ("bound_and_estimate", "bound a quantity from above and below with explicit reasoning, then narrow the gap"),
    ("two_hypotheses", "present two explanations, derive a distinguishing prediction, and say what the observation would look like for each"),
    ("decision_under_constraints", "lay out the constraints, eliminate infeasible options, and justify the remaining choice step by step"),
    ("incentive_analysis", "identify who gains, who pays, and how the incentives produce the observed behaviour"),
    ("failure_reconstruction", "reconstruct how a plan went wrong, separating the trigger from the underlying weakness"),
    ("scale_limit", "reason about what breaks when a quantity grows, and locate the transition point"),
    ("definition_precision", "show that a vague term hides two different questions, define both, and answer each"),
    ("evidence_weighing", "weigh conflicting evidence, state what would change the verdict, and commit to one"),
    ("analogy_and_its_limits", "explain by analogy, then state exactly where the analogy fails and why it still helps"),
    ("invariant", "identify what stays constant while everything else changes, and use the invariant to predict"),
    ("tradeoff_frontier", "show that two goals trade off, sketch the frontier, and argue for a point on it"),
    ("retrospective_lesson", "describe an outcome, then extract the transferable principle and the condition under which it holds"),
]
SURFACES = [
    ("a technical essay for an engineering audience", "en_prose"),
    ("a worked solution written the way a good teacher would write it", "en_prose"),
    ("a long-form magazine feature", "en_prose"),
    ("a chapter of a novel in which a character works the problem out in their head", "en_prose"),
    ("a scene of two people arguing the question out loud, with dialogue", "en_prose"),
    ("a screenplay scene with action lines and spoken lines", "en_prose"),
    ("a personal essay that thinks through the problem while telling a small story", "en_prose"),
    ("a lab-notebook style reasoning log with false starts kept in", "en_prose"),
]
SUBJECT_POOL = [
    "why a familiar habit is harder to break than it looks", "whether a rumour could have been stopped earlier",
    "why a well-planned trip fell apart", "how a cook decides when a dish is ready",
    "why two equally good teams got different results", "how a city decides where to put a bridge",
    "why an old building is still standing", "how a chess player chooses between two moves",
    "why a market price moved the way it did", "how a doctor narrows a diagnosis with few tests",
    "why a machine failed on the day it did", "how an editor cuts a long draft down",
    "why a species survives in one valley and not the next", "how a teacher spots a misunderstanding",
    "why a song works and a similar one does not", "how a negotiator finds the room to move",
    "why a memory of a place is more vivid than the photograph", "how a sailor reads weather from the sky",
    "why a community rebuilt faster than its neighbour", "how a programmer decides a bug is in the tool and not the code",
    "why a translation loses something specific", "how a farmer decides when to harvest",
    "why a rumour spread faster than the correction", "how a runner paces a long race",
    "why an argument convinced one person and alienated another", "how a museum chooses what to keep in storage",
    "why a shortcut turned out longer", "how a parent decides a child is ready",
    "why a design that felt right failed in use", "how a journalist verifies a single anonymous claim",
]


def _skeleton_tasks(count: int, seed: int, lang: str) -> list[tuple[str, float, list[dict]]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[tuple[str, float, list[dict]]] = []
    guard = 0
    while len(out) < count and guard < count * 40:
        guard += 1
        sk, sk_desc = rng.choice(SKELETONS)
        surface, dom = rng.choice(SURFACES)
        subj = rng.choice(SUBJECT_POOL)
        key = f"{sk}|{surface}|{subj}"
        if key in seen:
            continue
        seen.add(key)
        if lang == "en":
            usr = (f"Write {surface}. Build it on this question: {subj}. "
                   f"The spine of the piece is: {sk_desc}. "
                   f"Let the reasoning carry it; do not summarise, think it through as you write.")
            d = dom
        else:
            usr = (f"请写成{surface}。围绕这个问题展开：{subj}。"
                   f"全篇的骨架是：{sk_desc}。让推理带着文字走，不要概述，边写边想清楚。")
            d = "zh_prose"
        out.append((d, 0.3 if len(out) % 5 else 0.0, [{"role": "user", "content": usr}]))
    return out


def tasks(count: int, seed: int) -> list[tuple[str, float, list[dict]]]:
    rng = random.Random(seed)
    out = []
    for i in range(count):
        if i % 2 == 0:
            topic = EN_TOPICS[(i // 2) % len(EN_TOPICS)]
            style = EN_STYLE[(i // 20) % len(EN_STYLE)]
            usr = f"{style} Topic: {topic}. (var #{i})"
            dom = "en_prose"
        else:
            topic = ZH_TOPICS[(i // 2) % len(ZH_TOPICS)]
            style = ZH_STYLE[(i // 20) % len(ZH_STYLE)]
            usr = f"{style} 题目：{topic}。（var #{i}）"
            dom = "zh_prose"
        out.append((dom, 0.3 if i % 5 else 0.0, [{"role": "user", "content": usr}]))
    rng.shuffle(out)
    return out


async def generate(args: argparse.Namespace) -> None:
    tok = AutoTokenizer.from_pretrained(args.snapshot_dir, trust_remote_code=True)
    if getattr(args, "skeleton", False):
        items = _skeleton_tasks(args.count, args.seed, args.lang)
    elif getattr(args, "reason", False):
        items = _reason_tasks(args.count, args.seed, args.lang)
    elif getattr(args, "tech", False):
        items = _tech_tasks(args.count, args.seed, args.lang)
    elif getattr(args, "wide", False):
        items = _wide_tasks(args.count, args.seed, args.lang)
    else:
        items = tasks(args.count, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    if out_path.exists():
        done = sum(1 for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip())
    lock = asyncio.Lock()
    stats = {"ok": 0, "err": 0, "toks": 0}
    tempo = time.time()
    sem = asyncio.Semaphore(args.concurrency)
    fh = out_path.open("a", encoding="utf-8")

    async with httpx.AsyncClient(timeout=900) as client:
        async def one(idx: int, item) -> None:
            dom, temp, msgs = item
            async with sem:
                body = {"model": args.model, "messages": msgs, "max_tokens": args.max_tokens,
                        "temperature": temp, "chat_template_kwargs": {"enable_thinking": False}}
                try:
                    r = await client.post(args.url, json=body)
                    r.raise_for_status()
                    msg = r.json()["choices"][0]["message"]
                    content = msg.get("content") or msg.get("reasoning_content") or ""
                    if not content:
                        stats["err"] += 1
                        return
                    p_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                    enable_thinking=False)
                    if not isinstance(p_ids, list):
                        p_ids = list(p_ids["input_ids"])
                    f_ids = tok.apply_chat_template(msgs + [{"role": "assistant", "content": content}],
                                                    tokenize=True, add_generation_prompt=False,
                                                    enable_thinking=False)
                    if not isinstance(f_ids, list):
                        f_ids = list(f_ids["input_ids"])
                    if len(f_ids) <= len(p_ids) + 200:
                        stats["err"] += 1
                        return
                    row = {"idx": idx, "domain": dom, "thinking": False, "prompt_len": len(p_ids),
                           "input_ids": f_ids[: args.max_seq_len]}
                    async with lock:
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        stats["ok"] += 1
                        stats["toks"] += len(row["input_ids"])
                        if stats["ok"] % 25 == 0:
                            el = time.time() - tempo
                            print(f"[{stats['ok']}/{args.count}] toks={stats['toks']} "
                                  f"rate={stats['toks'] / el:.1f} tok/s", flush=True)
                except Exception as exc:  # noqa: BLE001
                    stats["err"] += 1
                    if stats["err"] <= 3:
                        print("ERR:", type(exc).__name__, str(exc)[:160], flush=True)

        await asyncio.gather(*(one(i, it) for i, it in enumerate(items[done:], start=done)))
    fh.close()
    print(f"DONE ok={stats['ok']} err={stats['err']} toks={stats['toks']} -> {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30004/v1/chat/completions")
    ap.add_argument("--model", default="RadixArk/Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--snapshot-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=1200)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--wide", action="store_true", help="combinatorial domain x angle x form topics")
    ap.add_argument("--tech", action="store_true", help="technical/reasoning register (matches the deployed workload)")
    ap.add_argument("--reason", action="store_true", help="reasoning/logic register: derivations, proofs, causal chains")
    ap.add_argument("--skeleton", action="store_true", help="reasoning skeletons rendered under many prose surfaces")
    ap.add_argument("--lang", default="en", choices=("en", "zh"))
    generate_args = ap.parse_args()
    asyncio.run(generate(generate_args))


if __name__ == "__main__":
    main()
