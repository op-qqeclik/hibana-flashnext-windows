#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Multi-step self-feeding KL + CE distillation for FlashNext's native autoregressive MTP head.

Uses torch._grouped_mm to train rank-16 LoRA adapters across all 512 routed experts alongside
full BF16 training of the core MTP projections, hyper-connections, attention, router, and shared expert,
then merges LoRA weights cleanly back into the 31 native mtp.* tensors (zero serving overhead).
"""

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

HC = 4
H = 2560
EPS = 1e-6
NUM_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
ROTARY_DIM = 64
ROPE_THETA = 10_000_000.0
NUM_EXPERTS = 512
TOPK_EXPERTS = 10
MOE_INTER = 640


def gemma_rms(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).to(x.dtype)


def grouped_gemma_rms(x: torch.Tensor, w: torch.Tensor, hc: int = HC, hidden: int = H, eps: float = EPS) -> torch.Tensor:
    xf = x.float().view(*x.shape[:-1], hc, hidden)
    xn = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).flatten(-2)
    return (xn * (1.0 + w.float())).to(x.dtype)


class TrainableHyperConnection(nn.Module):
    def __init__(self, combine: bool = True):
        super().__init__()
        self.hc_norm = nn.Parameter(torch.zeros(HC * H, dtype=torch.bfloat16))
        self.down = nn.Linear(HC * H, 320, bias=False, dtype=torch.bfloat16)
        self.up = nn.Linear(320, HC * H, bias=False, dtype=torch.bfloat16)
        self.inject = nn.Linear(HC * H, HC, bias=False, dtype=torch.bfloat16) if combine else None

    def mix(self, hyper: torch.Tensor):
        normed = grouped_gemma_rms(hyper, self.hc_norm)
        gate = torch.sigmoid(self.up(F.silu(self.down(normed) / HC)))
        mixed = (gate.view(*hyper.shape[:-1], HC, H) * normed.view(*hyper.shape[:-1], HC, H)).mean(-2)
        return mixed.to(hyper.dtype), (hyper, normed)

    def combine(self, block_out: torch.Tensor, res):
        hyper, normed = res
        inj = 2.0 * torch.sigmoid(self.inject(normed) / HC)
        out = hyper.view(*hyper.shape[:-1], HC, H) + block_out.unsqueeze(-2) * inj.unsqueeze(-1)
        return out.flatten(-2).to(hyper.dtype)


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rot, rest = x[..., :ROTARY_DIM], x[..., ROTARY_DIM:]
    x1, x2 = rot.float().chunk(2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat([o1.to(x.dtype), o2.to(x.dtype), rest], dim=-1)


class TrainableFlashNextMtp(nn.Module):
    def __init__(self, sd: dict, lora_rank: int = 16):
        super().__init__()
        p = "mtp."
        l0 = "mtp.layers.0."
        self.fc_embedding = nn.Linear(H, H, bias=False, dtype=torch.bfloat16)
        self.fc_hidden = nn.Linear(H, H, bias=False, dtype=torch.bfloat16)
        self.norm_e = nn.Parameter(sd[f"{p}pre_fc_norm_embedding.weight"].clone())
        self.norm_h = nn.Parameter(sd[f"{p}pre_fc_norm_hidden.weight"].clone())
        self.fc_embedding.weight.data.copy_(sd[f"{p}fc_embedding.weight"])
        self.fc_hidden.weight.data.copy_(sd[f"{p}fc_hidden.weight"])

        self.attn_hc = TrainableHyperConnection(combine=True)
        self.mlp_hc = TrainableHyperConnection(combine=True)
        self.mixer = TrainableHyperConnection(combine=False)
        for mod, pref in ((self.attn_hc, f"{l0}attn_hyper_connection."),
                          (self.mlp_hc, f"{l0}mlp_hyper_connection."),
                          (self.mixer, f"{p}hyper_connection_mixer.")):
            mod.hc_norm.data.copy_(sd[pref + "hc_norm.weight"])
            mod.down.weight.data.copy_(sd[pref + "input_mix_weight_down.weight"])
            mod.up.weight.data.copy_(sd[pref + "input_mix_weight_up.weight"])
            if mod.inject is not None:
                mod.inject.weight.data.copy_(sd[pref + "block_inject_weight.weight"])

        ap = l0 + "self_attn."
        self.q_proj = nn.Linear(H, NUM_HEADS * 2 * HEAD_DIM, bias=False, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(H, NUM_KV_HEADS * HEAD_DIM, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(H, NUM_KV_HEADS * HEAD_DIM, bias=False, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, H, bias=False, dtype=torch.bfloat16)
        self.q_proj.weight.data.copy_(sd[ap + "q_proj.weight"])
        self.k_proj.weight.data.copy_(sd[ap + "k_proj.weight"])
        self.v_proj.weight.data.copy_(sd[ap + "v_proj.weight"])
        self.o_proj.weight.data.copy_(sd[ap + "o_proj.weight"])
        self.q_norm = nn.Parameter(sd[ap + "q_norm.weight"].clone())
        self.k_norm = nn.Parameter(sd[ap + "k_norm.weight"].clone())

        mp = l0 + "mlp."
        self.router = nn.Linear(H, NUM_EXPERTS, bias=False, dtype=torch.bfloat16)
        self.router.weight.data.copy_(sd[mp + "gate.weight"])
        self.shared_gate_proj = nn.Linear(H, MOE_INTER, bias=False, dtype=torch.bfloat16)
        self.shared_up_proj = nn.Linear(H, MOE_INTER, bias=False, dtype=torch.bfloat16)
        self.shared_down_proj = nn.Linear(MOE_INTER, H, bias=False, dtype=torch.bfloat16)
        self.shared_gate_proj.weight.data.copy_(sd[mp + "shared_expert.gate_proj.weight"])
        self.shared_up_proj.weight.data.copy_(sd[mp + "shared_expert.up_proj.weight"])
        self.shared_down_proj.weight.data.copy_(sd[mp + "shared_expert.down_proj.weight"])
        self.shared_gate = nn.Parameter(sd[mp + "shared_expert_gate.weight"].clone())

        self.register_buffer("experts_gu", sd[mp + "experts.gate_up_proj"].clone(), persistent=False)
        self.register_buffer("experts_dn", sd[mp + "experts.down_proj"].clone(), persistent=False)
        self.lora_rank = lora_rank
        self.gu_A = nn.Parameter(torch.randn(NUM_EXPERTS, lora_rank, H, dtype=torch.bfloat16) * (1.0 / math.sqrt(H)))
        self.gu_B = nn.Parameter(torch.zeros(NUM_EXPERTS, 2 * MOE_INTER, lora_rank, dtype=torch.bfloat16))
        self.dn_A = nn.Parameter(torch.randn(NUM_EXPERTS, lora_rank, MOE_INTER, dtype=torch.bfloat16) * (1.0 / math.sqrt(MOE_INTER)))
        self.dn_B = nn.Parameter(torch.zeros(NUM_EXPERTS, H, lora_rank, dtype=torch.bfloat16))

        self._extra_sd = {
            ap + "indexer.index_qk_proj.weight": sd[ap + "indexer.index_qk_proj.weight"].clone(),
            ap + "indexer.k_layernorm.weight": sd[ap + "indexer.k_layernorm.weight"].clone(),
            ap + "indexer.q_layernorm.weight": sd[ap + "indexer.q_layernorm.weight"].clone(),
        }

        inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotary(self, pos: torch.Tensor):
        angles = pos.float().unsqueeze(-1) * self.inv_freq
        return angles.cos().unsqueeze(-2), angles.sin().unsqueeze(-2)

    def fuse(self, tokens: torch.Tensor, hidden: torch.Tensor, embed_w: torch.Tensor) -> torch.Tensor:
        e = self.fc_embedding(gemma_rms(F.embedding(tokens, embed_w), self.norm_e))
        nh = gemma_rms(hidden, self.norm_h)
        h = self.fc_hidden(nh.view(*hidden.shape[:-1], HC, H))
        return (h + e.unsqueeze(-2)).flatten(-2)

    def qkv_proj(self, h: torch.Tensor, pos: torch.Tensor):
        b, l, _ = h.shape
        qg = self.q_proj(h).view(b, l, NUM_HEADS, 2 * HEAD_DIM)
        q, gate = qg[..., :HEAD_DIM], qg[..., HEAD_DIM:]
        k = self.k_proj(h).view(b, l, NUM_KV_HEADS, HEAD_DIM)
        v = self.v_proj(h).view(b, l, NUM_KV_HEADS, HEAD_DIM)
        q = gemma_rms(q, self.q_norm)
        k = gemma_rms(k, self.k_norm)
        cos, sin = self.rotary(pos)
        q = apply_partial_rope(q, cos, sin)
        k = apply_partial_rope(k, cos, sin)
        return q, k, v, gate

    def moe(self, x: torch.Tensor) -> torch.Tensor:
        t, _ = x.shape
        probs = F.softmax(self.router(x).float(), dim=-1)
        top_w, top_ids = probs.topk(TOPK_EXPERTS, dim=-1)
        top_w = (top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)).to(x.dtype)

        flat_ids = top_ids.reshape(-1)
        x_rep = x.unsqueeze(1).expand(-1, TOPK_EXPERTS, -1).reshape(-1, H)
        order = torch.argsort(flat_ids)
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(order.numel(), device=x.device)
        s_ids = flat_ids[order]
        s_x = x_rep[order].contiguous()
        offs = torch.bincount(s_ids, minlength=NUM_EXPERTS).cumsum(0, dtype=torch.int32)

        gu = torch._grouped_mm(s_x, self.experts_gu.transpose(1, 2), offs=offs) + torch._grouped_mm(
            torch._grouped_mm(s_x, self.gu_A.transpose(1, 2), offs=offs),
            self.gu_B.transpose(1, 2),
            offs=offs,
        )
        g, u = gu.chunk(2, dim=-1)
        mid = (F.silu(g.float()).to(x.dtype) * u).contiguous()
        out_sorted = torch._grouped_mm(mid, self.experts_dn.transpose(1, 2), offs=offs) + torch._grouped_mm(
            torch._grouped_mm(mid, self.dn_A.transpose(1, 2), offs=offs),
            self.dn_B.transpose(1, 2),
            offs=offs,
        )

        routed = (out_sorted[inv_order].view(t, TOPK_EXPERTS, H) * top_w.unsqueeze(-1)).sum(dim=1)
        shared = self.shared_down_proj(
            F.silu(self.shared_gate_proj(x).float()).to(x.dtype) * self.shared_up_proj(x)
        )
        sg = torch.sigmoid(F.linear(x, self.shared_gate).float()).to(x.dtype)
        return routed + sg * shared

    def forward_prefill(self, tokens: torch.Tensor, hidden: torch.Tensor, pos: torch.Tensor, embed_w: torch.Tensor):
        b, l = tokens.shape
        u = self.fuse(tokens, hidden, embed_w)
        h, attn_res = self.attn_hc.mix(u)
        q, k, v, gate = self.qkv_proj(h, pos)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, enable_gqa=True
        ).transpose(1, 2).contiguous()
        attn_out = self.o_proj((attn * torch.sigmoid(gate.float()).to(attn.dtype)).reshape(b, l, -1))
        u = self.attn_hc.combine(attn_out, attn_res)
        h_m, mlp_res = self.mlp_hc.mix(u)
        m_out = self.moe(h_m.reshape(b * l, H)).view(b, l, H)
        d = self.mlp_hc.combine(m_out, mlp_res)
        return d, k.transpose(1, 2), v.transpose(1, 2)

    def forward_rollout_step(self, tokens: torch.Tensor, hidden: torch.Tensor, pos: torch.Tensor,
                             k_ctx: torch.Tensor, v_ctx: torch.Tensor, attn_mask: torch.Tensor, embed_w: torch.Tensor):
        b, m = tokens.shape
        u = self.fuse(tokens, hidden, embed_w)
        h, attn_res = self.attn_hc.mix(u)
        q, k_new, v_new, gate = self.qkv_proj(h, pos)
        k_all = torch.cat([k_ctx, k_new.transpose(1, 2)], dim=2)
        v_all = torch.cat([v_ctx, v_new.transpose(1, 2)], dim=2)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k_all, v_all, attn_mask=attn_mask, enable_gqa=True
        ).transpose(1, 2).contiguous()
        attn_out = self.o_proj((attn * torch.sigmoid(gate.float()).to(attn.dtype)).reshape(b, m, -1))
        u = self.attn_hc.combine(attn_out, attn_res)
        h_m, mlp_res = self.mlp_hc.mix(u)
        m_out = self.moe(h_m.reshape(b * m, H)).view(b, m, H)
        d = self.mlp_hc.combine(m_out, mlp_res)
        return d, k_new.transpose(1, 2), v_new.transpose(1, 2)

    def export_state_dict(self) -> dict:
        p = "mtp."
        l0 = "mtp.layers.0."
        ap = l0 + "self_attn."
        mp = l0 + "mlp."
        out = {
            f"{p}fc_embedding.weight": self.fc_embedding.weight.detach().cpu().contiguous(),
            f"{p}fc_hidden.weight": self.fc_hidden.weight.detach().cpu().contiguous(),
            f"{p}pre_fc_norm_embedding.weight": self.norm_e.detach().cpu().contiguous(),
            f"{p}pre_fc_norm_hidden.weight": self.norm_h.detach().cpu().contiguous(),
            f"{mp}gate.weight": self.router.weight.detach().cpu().contiguous(),
            f"{mp}shared_expert.gate_proj.weight": self.shared_gate_proj.weight.detach().cpu().contiguous(),
            f"{mp}shared_expert.up_proj.weight": self.shared_up_proj.weight.detach().cpu().contiguous(),
            f"{mp}shared_expert.down_proj.weight": self.shared_down_proj.weight.detach().cpu().contiguous(),
            f"{mp}shared_expert_gate.weight": self.shared_gate.detach().cpu().contiguous(),
            f"{ap}q_proj.weight": self.q_proj.weight.detach().cpu().contiguous(),
            f"{ap}k_proj.weight": self.k_proj.weight.detach().cpu().contiguous(),
            f"{ap}v_proj.weight": self.v_proj.weight.detach().cpu().contiguous(),
            f"{ap}o_proj.weight": self.o_proj.weight.detach().cpu().contiguous(),
            f"{ap}q_norm.weight": self.q_norm.detach().cpu().contiguous(),
            f"{ap}k_norm.weight": self.k_norm.detach().cpu().contiguous(),
        }
        for mod, pref in ((self.attn_hc, f"{l0}attn_hyper_connection."),
                          (self.mlp_hc, f"{l0}mlp_hyper_connection."),
                          (self.mixer, f"{p}hyper_connection_mixer.")):
            out[pref + "hc_norm.weight"] = mod.hc_norm.detach().cpu().contiguous()
            out[pref + "input_mix_weight_down.weight"] = mod.down.weight.detach().cpu().contiguous()
            out[pref + "input_mix_weight_up.weight"] = mod.up.weight.detach().cpu().contiguous()
            if mod.inject is not None:
                out[pref + "block_inject_weight.weight"] = mod.inject.weight.detach().cpu().contiguous()
        out.update({k: v.cpu().contiguous() for k, v in self._extra_sd.items()})
        with torch.no_grad():
            gu_merged = self.experts_gu + torch.bmm(self.gu_B, self.gu_A).to(torch.bfloat16)
            dn_merged = self.experts_dn + torch.bmm(self.dn_B, self.dn_A).to(torch.bfloat16)
            out[mp + "experts.gate_up_proj"] = gu_merged.cpu().contiguous()
            out[mp + "experts.down_proj"] = dn_merged.cpu().contiguous()
        return out


def compute_kl_ce(hidden_d: torch.Tensor, target_tok: torch.Tensor, top_val: torch.Tensor, top_idx: torch.Tensor,
                  mixer: TrainableHyperConnection, head_w: torch.Tensor):
    mixed = mixer.mix(hidden_d)[0]
    logits = F.linear(mixed, head_w)
    ce = F.cross_entropy(logits, target_tok, reduction="none")
    lse = (ce + logits.gather(-1, target_tok.unsqueeze(-1)).squeeze(-1).float()).unsqueeze(-1)
    q_top_logp = logits.gather(-1, top_idx.long()).float() - lse
    p_top = F.softmax(top_val.float(), dim=-1)
    p_top_logp = F.log_softmax(top_val.float(), dim=-1)
    kl = (p_top * (p_top_logp - q_top_logp)).sum(dim=-1)
    return 0.5 * ce + 0.5 * kl, logits.argmax(dim=-1)


def step_batch(model: TrainableFlashNextMtp, embed_w: torch.Tensor, head_w: torch.Tensor,
               batch: list[dict], device: torch.device, num_anchors: int = 64, rollout_steps: int = 4):
    b = len(batch)
    max_t = max(int(x["input_ids"].shape[0]) for x in batch)
    ids = torch.zeros((b, max_t), dtype=torch.long, device=device)
    norms = torch.zeros((b, max_t, HC * H), dtype=torch.bfloat16, device=device)
    pls = [int(x["prompt_len"]) for x in batch]
    lens = [int(x["input_ids"].shape[0]) for x in batch]
    for i, x in enumerate(batch):
        t = lens[i]
        ids[i, :t] = x["input_ids"].to(device=device, dtype=torch.long)
        norms[i, :t] = x["norm"].to(device=device)

    l_ctx = max_t - 1
    in_tok, in_hid = ids[:, 1:], norms[:, :-1]
    pos0 = torch.arange(1, max_t, device=device).unsqueeze(0).expand(b, -1)
    d0, k0, v0 = model.forward_prefill(in_tok, in_hid, pos0, embed_w)

    anchors_list = []
    for i in range(b):
        avail = list(range(max(2, pls[i]), lens[i] - rollout_steps))
        if len(avail) > num_anchors:
            avail = sorted(random.sample(avail, num_anchors))
        anchors_list.append(avail)
    m = max(len(a) for a in anchors_list)
    if m == 0:
        return None

    topk = int(batch[0]["top_val"].shape[-1])
    anch = torch.ones((b, m), dtype=torch.long, device=device)
    amask = torch.zeros((b, m), dtype=torch.bool, device=device)
    top_val_b = torch.zeros((b, m, rollout_steps, topk), dtype=torch.bfloat16, device=device)
    top_idx_b = torch.zeros((b, m, rollout_steps, topk), dtype=torch.int64, device=device)
    tgt_tok_b = torch.zeros((b, m, rollout_steps), dtype=torch.long, device=device)
    for i, av in enumerate(anchors_list):
        if not av:
            continue
        k_n = len(av)
        anch[i, :k_n] = torch.tensor(av, dtype=torch.long, device=device)
        amask[i, :k_n] = True
        pl = pls[i]
        tv = batch[i]["top_val"].to(device)
        ti = batch[i]["top_idx"].to(device=device, dtype=torch.long)
        for s in range(rollout_steps):
            a_t = anch[i, :k_n] + s
            tgt_tok_b[i, :k_n, s] = ids[i, a_t]
            top_val_b[i, :k_n, s] = tv[a_t - pl]
            top_idx_b[i, :k_n, s] = ti[a_t - pl]

    idx0 = (anch - 2).clamp(min=0)
    b_idx = torch.arange(b, device=device).unsqueeze(1).expand(b, m)
    cur_d = d0[b_idx, idx0]

    step_weights = [1.0, 0.8, 0.64, 0.5]
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    w_sum = 0.0
    preds = []
    k_roll, v_roll = [], []
    ctx_indices = torch.arange(l_ctx, device=device).view(1, 1, l_ctx)

    for s in range(rollout_steps):
        if s == 0:
            step_d = cur_d
        else:
            tok_s = tgt_tok_b[:, :, s - 1]
            pos_s = anch + s - 1
            ctx_keep = ctx_indices <= (anch - 2).unsqueeze(-1)
            eye_m = torch.eye(m, dtype=torch.bool, device=device).unsqueeze(0).expand(b, -1, -1)
            if s == 1:
                am = torch.cat([ctx_keep, eye_m], dim=-1).unsqueeze(1)
                k_cat, v_cat = k0, v0
            else:
                prev_keep = torch.cat([eye_m] * s, dim=-1)
                am = torch.cat([ctx_keep, prev_keep], dim=-1).unsqueeze(1)
                k_cat = torch.cat([k0, *k_roll], dim=2)
                v_cat = torch.cat([v0, *v_roll], dim=2)
            step_d, ks, vs = model.forward_rollout_step(tok_s, cur_d, pos_s, k_cat, v_cat, am, embed_w)
            k_roll.append(ks)
            v_roll.append(vs)
            cur_d = step_d

        l_vec, pred_s = compute_kl_ce(
            step_d[amask], tgt_tok_b[:, :, s][amask], top_val_b[:, :, s][amask], top_idx_b[:, :, s][amask],
            model.mixer, head_w
        )
        total_loss = total_loss + step_weights[s] * l_vec.mean()
        w_sum += step_weights[s]
        p_full = torch.zeros((b, m), dtype=torch.long, device=device)
        p_full[amask] = pred_s
        preds.append(p_full)

    with torch.no_grad():
        pred_stack = torch.stack(preds, dim=-1)
        eq = (pred_stack == tgt_tok_b) & amask.unsqueeze(-1)
        acc_s0 = float(eq[:, :, 0][amask].float().mean().item())
        acc_len = float((1.0 + eq.long().cumprod(dim=-1).sum(dim=-1).float())[amask].mean().item())
    return {"loss": total_loss / w_sum, "acc_s0": acc_s0, "accept_len": acc_len}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory with mtp_init.pt, target_*.pt, and features/")
    ap.add_argument("--out-dir", required=True, help="Directory for exported mtp_weights.safetensors")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--num-anchors", type=int, default=64)
    ap.add_argument("--lora-rank", type=int, default=16)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    sd = torch.load(data_dir / "mtp_init.pt", map_location="cpu", weights_only=True)
    model = TrainableFlashNextMtp(sd, lora_rank=args.lora_rank).to(device)
    embed_w = torch.load(data_dir / "target_embed.pt", map_location=device, weights_only=True)
    head_w = torch.load(data_dir / "target_lm_head.pt", map_location=device, weights_only=True)

    items = []
    for p in sorted((data_dir / "features").glob("shard_*.pt")):
        items.extend(torch.load(p, map_location="cpu", weights_only=False))
    rng = random.Random(42)
    rng.shuffle(items)
    val_n = max(60, len(items) // 20)
    val_items, train_items = items[:val_n], items[val_n:]
    print(f"Train seqs={len(train_items)}, Val seqs={len(val_items)}", flush=True)

    decay_p, nodecay_p = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (nodecay_p if (p.ndim < 2 or "norm" in n or "_A" in n or "router" in n) else decay_p).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay_p, "weight_decay": 0.01}, {"params": nodecay_p, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    steps_per_ep = math.ceil(len(train_items) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_ep * args.epochs
    warmup_steps = max(20, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda st: (st + 1) / warmup_steps
        if st < warmup_steps
        else 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * (st - warmup_steps) / max(1, total_steps - warmup_steps))),
    )

    @torch.no_grad()
    def evaluate(tag: str) -> float:
        model.eval()
        by_dom = {}
        tot_len = tot_s0 = tot_n = 0.0
        for item in val_items:
            res = step_batch(model, embed_w, head_w, [item], device, num_anchors=96)
            if res is None:
                continue
            tot_len += res["accept_len"]
            tot_s0 += res["acc_s0"]
            tot_n += 1.0
            d = item["domain"].split("_")[0]
            s = by_dom.setdefault(d, [0.0, 0.0, 0.0])
            s[0] += res["accept_len"]
            s[1] += res["acc_s0"]
            s[2] += 1.0
        dom_str = " ".join(f"{k}:len={v[0]/v[2]:.3f}/s0={v[1]/v[2]:.3f}" for k, v in sorted(by_dom.items()))
        mean_len = tot_len / max(1.0, tot_n)
        print(f"[VAL {tag}] accept_len={mean_len:.3f} s0={tot_s0/max(1.0,tot_n):.3f} | {dom_str}", flush=True)
        model.train()
        return mean_len

    best_len = evaluate("init")
    save_file(model.export_state_dict(), str(out_dir / "mtp_weights.safetensors"))

    global_step = 0
    for ep in range(1, args.epochs + 1):
        sorted_items = sorted(train_items, key=lambda x: int(x["input_ids"].shape[0]) + rng.randint(-32, 32))
        batches = [sorted_items[i : i + args.batch_size] for i in range(0, len(sorted_items), args.batch_size)]
        rng.shuffle(batches)
        model.train()
        opt.zero_grad(set_to_none=True)
        ep_t0 = time.time()
        accum = 0
        run_loss = 0.0
        for batch in batches:
            res = step_batch(model, embed_w, head_w, batch, device, num_anchors=args.num_anchors)
            if res is None:
                continue
            (res["loss"] / args.grad_accum).backward()
            run_loss += float(res["loss"].item())
            accum += 1
            if accum == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0:
                    print(f"ep={ep} step={global_step}/{total_steps} loss={run_loss/args.grad_accum:.4f} elapsed={time.time()-ep_t0:.0f}s", flush=True)
                accum = 0
                run_loss = 0.0
        if accum > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        val_len = evaluate(f"ep{ep} ({time.time() - ep_t0:.0f}s)")
        if val_len > best_len:
            best_len = val_len
            save_file(model.export_state_dict(), str(out_dir / "mtp_weights.safetensors"))
            print(f"Saved improved MTP weights -> {out_dir / 'mtp_weights.safetensors'}", flush=True)


if __name__ == "__main__":
    main()
