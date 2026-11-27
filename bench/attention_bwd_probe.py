"""Usage: python bench/attention_bwd_probe.py  (STEPS=300 by default; needs ~/nvfp4_data; REAL=0: random data only)

Probe: which products of the attention backward can run in 8 bits at the gradient quality of the exact
backward (PyTorch's flash backward on the dequantized forward operands, attention(backward="bf16"), which trains
at BF16 attention's loss). Every backward recipe is emulated in fp32 on the forward's quantized operands
(INT8 q / centred k per token, MXFP8 v), with the forward's output and log-sum-exp, and its gradients are
compared with exact fp32 attention on the original inputs. Recipe fields (dp / dv / dk / dq / fwd):

  dp   dO V^T:  bf16 (dO in BF16, the forward's V)  |  i8 (INT8 dO and V per token, as backward="8bit")
  dv   P^T dO:  bf16  |  k8 (e4m3(256 P) x MXFP8 dO along the queries, as backward="8bit")  |  p2 (P in two
                e4m3 terms)  |  i8x2 (P in two INT8 terms at the fixed scale 127, dO in INT8 per (channel, 32
                queries))
  dk   dS^T Q:  bf16  |  k8 (MXFP8 dS and Q along the queries)  |  p2 (dS in two e4m3 terms)
  dq   dS K:    bf16 (BF16 dS x centred BF16 K, minus r B as the 8-bit kernel)  |  f16 (FP16 dS scaled per
                row by the dO scale, FP16 K, FP16 accumulation over each 16 keys, fp32 across tiles)
  fwd  forward P V:  p2 (two e4m3 terms, today)  |  i8x2 (two INT8 terms, V in INT8 per (channel, 32 keys))
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from nvfp4 import reference as R  # noqa: E402

E4 = torch.float8_e4m3fn


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def flash_bwd(do, q, k, v, o, lse, causal, scale):
    rng = torch.zeros(2, dtype=torch.uint64, device=q.device)
    return torch.ops.aten._scaled_dot_product_flash_attention_backward(
        do, q, k, v, o, lse, None, None, q.shape[2], k.shape[2], 0.0, causal, rng,
        torch.zeros((), dtype=torch.uint64, device=q.device), scale=scale)


def int8_block_fake(x, dim):
    """INT8 with one scale per 32 elements along ``dim`` (amax / 127)."""
    xm = x.movedim(dim, -1)
    bl = xm.reshape(*xm.shape[:-1], xm.shape[-1] // 32, 32)
    am = bl.abs().amax(-1, keepdim=True)
    sc = am / 127.0
    out = torch.where(am > 0, torch.round(bl / sc.clamp(min=1e-30)) * sc, torch.zeros_like(bl))
    return out.reshape(xm.shape).movedim(-1, dim)


def e4(x):
    return x.to(E4).float()


def two_e4m3(x):
    """x ~ e4m3(x) + e4m3(16 r) / 16 (x already in e4m3 range)."""
    hi = e4(x)
    return hi + e4((x - hi) * 16) / 16


def mx_two(x, dim):
    hi = R.mxfp8_fake(x, dim=dim)
    return hi + R.mxfp8_fake(x - hi, dim=dim)


def i8x2_fixed(p):
    """P (in [0, 1]) in two INT8 terms at the fixed scale 127."""
    hi = torch.round(p * 127)
    return (hi + torch.round((p * 127 - hi) * 127) / 127) / 127


def f16_acc_matmul(a, b, chunk=16):
    """a [.., M, K] @ b [K, D] with FP16 operands and an FP16 accumulator over each 64-k tile (rounded after
    every `chunk` of k, as an m16n8k16 FP16-accumulate MMA chain), tiles summed in fp32."""
    a16, b16 = a.half().float(), b.half().float()
    K = a.shape[-1]
    total = torch.zeros(*a.shape[:-1], b.shape[-1], device=a.device)
    for t0 in range(0, K, 64):
        acc = torch.zeros_like(total)
        for c0 in range(t0, min(K, t0 + 64), chunk):
            acc = (acc + a16[..., c0:c0 + chunk] @ b16[c0:c0 + chunk]).half().float()
        total += acc
    return total


def f16_tile_matmul(a, b, row_scale=False, tile=32, chunk=16):
    """a [H, M, K] @ b [H, K, D]: per K-tile of `tile`, FP16 operands (a divided by its per-row tile maximum
    if row_scale), FP16 accumulation rounded after every `chunk`, the tile result (times the row scale) added in
    fp32."""
    H, M, K = a.shape
    out = torch.zeros(H, M, b.shape[-1], device=a.device)
    for t0 in range(0, K, tile):
        at, bt = a[..., t0:t0 + tile], b[:, t0:t0 + tile]
        sc = at.abs().amax(-1, keepdim=True).clamp(min=1e-30) if row_scale else None
        a16 = (at / sc if row_scale else at).half().float()
        b16 = bt.half().float()
        acc = torch.zeros_like(out)
        for c0 in range(0, a16.shape[-1], chunk):
            acc = (acc + a16[..., c0:c0 + chunk] @ b16[:, c0:c0 + chunk]).half().float()
        out += acc * sc if row_scale else acc
    return out


def forward_emul(qf, kf, vX, scale, causal, mode):
    N = qf.shape[-2]
    mask = torch.ones(N, N, dtype=torch.bool, device=qf.device).triu(1)
    s = (qf @ kf.transpose(-1, -2)) * scale
    if causal:
        s = s.masked_fill(mask, float("-inf"))
    m = s.amax(-1, keepdim=True)
    p = torch.exp(s - m)
    l = p.sum(-1, keepdim=True)
    lse = (m + torch.log(l)).squeeze(-1)
    if mode == "p2":
        pq = two_e4m3(p * 128) / 128
    else:  # i8x2
        pq = i8x2_fixed(p)
    return pq @ vX / l, lse


RECIPES = {
    "fused": ("i8", "f16t", "f16t", "bf16nc", "p2"),
    "fused +corr": ("i8", "f16t", "f16t", "bf16", "p2"),
    "fused dq=f16": ("i8", "f16t", "f16t", "f16nc", "p2"),
    "fused fwd=i8x2": ("i8", "f16t", "f16t", "bf16nc", "i8x2"),
    "k8 nocorr": ("i8", "k8", "k8", "bf16nc", "p2"),
    "k8": ("i8", "k8", "k8", "bf16", "p2"),
    "k8 dv=p2": ("i8", "p2", "k8", "bf16", "p2"),
    "k8 dk=p2": ("i8", "k8", "p2", "bf16", "p2"),
    "i8 all": ("i8", "i8x2", "p2", "f16", "i8x2"),
    "bf16 all(i8x2 fwd)": ("bf16", "bf16", "bf16", "bf16", "i8x2"),
}


def bwd_emul(q, k, v, do, qf, kc, kf, scale, causal, recipe):
    """One (batch) slice [H, N, D]; returns dq, dk, dv (fp32)."""
    dpm, dvm, dkm, dqm, fwd = recipe
    N = q.shape[-2]
    if fwd == "i8x2":
        vX = int8_block_fake(R._pad_seq(v.float()), -2)[..., :N, :]
    else:
        vX = R.mxfp8_fake(R._pad_seq(v.float()), dim=-2)[..., :N, :]
    o, lse = forward_emul(qf, kf, vX, scale, causal, "p2" if fwd == "p1" else fwd)
    if fwd == "p1":  # one-term e4m3 P
        s = (qf @ kf.transpose(-1, -2)) * scale
        if causal:
            s = s.masked_fill(torch.ones(N, N, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
        p = torch.exp(s - s.amax(-1, keepdim=True))
        o = (e4(p * 128) / 128) @ vX / p.sum(-1, keepdim=True)
        del s, p
    o = o.to(q.dtype).float()
    dof = do.float()
    s = (qf @ kf.transpose(-1, -2)) * scale
    if causal:
        s = s.masked_fill(torch.ones(N, N, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    p = torch.exp(s - lse.unsqueeze(-1))
    del s
    delta = (dof * o).sum(-1, keepdim=True)
    if dpm == "bf16":
        dp = dof @ vX.transpose(-1, -2)
    else:
        dp = R.int8_token_fake(dof) @ R.int8_token_fake(v.float()).transpose(-1, -2)
    ds = p * (dp - delta)
    del dp
    # dV = P^T dO (blocks along the queries: dim -2 of P and of dO)
    if dvm == "bf16":
        dv = p.transpose(-1, -2) @ dof
    elif dvm == "k8":
        dv = (e4(p * 256) / 256).transpose(-1, -2) @ R.mxfp8_fake(R._pad_seq(dof), dim=-2)[..., :N, :]
    elif dvm == "p2":
        dv = (two_e4m3(p * 256) / 256).transpose(-1, -2) @ R.mxfp8_fake(R._pad_seq(dof), dim=-2)[..., :N, :]
    elif dvm == "f16t":  # P (<= 1) in FP16, dO / max|dO| in FP16, FP16 accumulation per 32 queries
        gm = dof.abs().amax().clamp(min=1e-30)
        dv = f16_tile_matmul(p.transpose(-1, -2), dof / gm) * gm
    else:  # i8x2
        dv = i8x2_fixed(p).transpose(-1, -2) @ int8_block_fake(R._pad_seq(dof), -2)[..., :N, :]
    # dK = dS^T Q (blocks along the queries)
    if dkm == "bf16":
        dk = ds.transpose(-1, -2) @ qf
    elif dkm == "f16t":  # dS^T / its per-(key, 32 queries) maximum in FP16, q / max|q| in FP16
        qm = qf.abs().amax().clamp(min=1e-30)
        dk = f16_tile_matmul(ds.transpose(-1, -2), qf / qm, row_scale=True) * qm
    else:
        qm = R.mxfp8_fake(R._pad_seq(q.float()), dim=-2)[..., :N, :]
        dsp = R._pad_seq(ds)  # pad the query dim
        dsq = R.mxfp8_fake(dsp, dim=-2)[..., :N, :] if dkm == "k8" else mx_two(dsp, -2)[..., :N, :]
        dk = dsq.transpose(-1, -2) @ qm
    dk = dk * scale
    # dQ = dS K
    rsum = ds.sum(-1, keepdim=True)
    bq = (e4(p * 256) / 256) @ R.mxfp8_fake(R._pad_seq(kc), dim=-2)[..., :N, :]
    if dqm.startswith("bf16"):
        dq = ds.bfloat16().float() @ kc.bfloat16().float()
    else:
        gsc = dof.abs().amax(-1, keepdim=True) / 127.0 * 16  # the dO row scale (x 16: FP16 headroom)
        gsc = torch.where(gsc > 0, gsc, torch.ones_like(gsc))
        dq = torch.cat([f16_acc_matmul(ds[h] / gsc[h], kc[h]) for h in range(ds.shape[0])]).view_as(qf) * gsc
    dq = (dq - rsum * bq) * scale if not dqm.endswith("nc") else dq * scale
    return dq, dk, dv


def run(q, k, v, do, causal=True):
    B, H, N, D = q.shape
    scale = D ** -0.5
    dt = q.dtype
    res = {}
    qe, ke, ve = (t.float().requires_grad_() for t in (q, k, v))
    oe = F.scaled_dot_product_attention(qe, ke, ve, is_causal=causal)
    oe.backward(do.float())
    ex = (qe.grad, ke.grad, ve.grad)
    del qe, ke, ve, oe
    qb, kb, vb = (t.clone().requires_grad_() for t in (q, k, v))
    F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal).backward(do)
    res["sdpa bf16"] = (qb.grad, kb.grad, vb.grad)
    qf = R.int8_token_fake(q.float())
    kc = k.float() - k.float().mean(2, keepdim=True)
    kf = R.int8_token_fake(kc)
    vf = R.mxfp8_fake(R._pad_seq(v.float()), dim=-2)[..., :N, :]
    # the exact backward as attention(backward="bf16") runs it (emulated p2 forward output, exact lse)
    o2, lse2 = zip(*[forward_emul(qf[b], kf[b], vf[b], scale, causal, "p2") for b in range(B)])
    o2, lse2 = torch.stack(o2).to(dt), torch.stack(lse2)
    res["flash (default)"] = flash_bwd(do, qf.to(dt), kf.to(dt), vf.to(dt), o2, lse2.contiguous(), causal, scale)
    del o2, lse2
    for name, recipe in RECIPES.items():
        outs = [bwd_emul(q[b], k[b], v[b], do[b], qf[b], kc[b], kf[b], scale, causal, recipe) for b in range(B)]
        res[name] = tuple(torch.stack(x) for x in zip(*outs))
        del outs
        torch.cuda.empty_cache()
    return {name: tuple(rel(a, b) for a, b in zip(r, ex)) for name, r in res.items()}


ALL = {}


def show(tag, out):
    print(tag, flush=True)
    for n, e in out.items():
        print(f"  {n:22s} dq {e[0]:.4f} dk {e[1]:.4f} dv {e[2]:.4f}", flush=True)
        ALL.setdefault(n, []).append(e)


torch.manual_seed(0)
for D in (64, 128):
    N = 2048
    q, k, v, do = (torch.randn(2, 4, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    k = k + 0.5 * torch.randn(1, 4, 1, D, device="cuda", dtype=torch.bfloat16)
    show(f"random D={D} N={N}", run(q, k, v, do))

if os.environ.get("REAL", "1") == "1":
    import train_gpt as tg  # noqa: E402

    ALL.clear()
    STEPS = int(os.environ.get("STEPS", "300"))
    T = 4096
    data = np.memmap(os.path.expanduser("~/nvfp4_data/train.bin"), dtype=np.uint16, mode="r")
    model = tg.GPT(50304, 768, 12, 6, T).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    it = tg.batches(data, 2, T, torch.Generator().manual_seed(0))
    for s in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = 6e-4 * min(1.0, (s + 1) / 100)
        x, y = next(it)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)[1]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if s % 100 == 0:
            print(f"step {s} loss {loss.item():.3f}", flush=True)
    caps = []

    def capture_sdpa(self, q, k, v):
        q, k, v = (t.detach().clone().requires_grad_() for t in (q, k, v))
        rec = {"q": q.detach(), "k": k.detach(), "v": v.detach()}
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out.register_hook(lambda g, rec=rec: rec.__setitem__("do", g.detach().clone()))
        caps.append(rec)
        return out

    tg.Block.sdpa = capture_sdpa
    x, y = next(it)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, y)[1]
    loss.backward()
    del model, opt
    torch.cuda.empty_cache()
    for li, c in enumerate(caps[:12]):
        q, k, v, do = (c[n][:1].contiguous() for n in ("q", "k", "v", "do"))  # one sequence per layer
        show(f"real layer {li} N={T}", run(q, k, v, do))
    print("== real layers: median / max over layers", flush=True)
    for n, es in ALL.items():
        a = torch.tensor(es)
        med, mx = a.median(0).values, a.max(0).values
        print(f"  {n:22s} dq {med[0]:.4f}/{mx[0]:.4f} dk {med[1]:.4f}/{mx[1]:.4f} dv {med[2]:.4f}/{mx[2]:.4f}",
              flush=True)
