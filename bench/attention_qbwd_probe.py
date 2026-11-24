"""Usage: python bench/attention_qbwd_probe.py  (STEPS=300 by default; needs ~/nvfp4_data)

Probe: 8-bit forward + PyTorch's exact flash backward run on the *quantized* inputs (int8 q / centred k and
MXFP8 v, dequantized to BF16), so the backward's recomputed scores match the forward's log-sum-exp. Gradient
errors vs exact fp32 attention, on random data and on real GPT-124M activations, for:
  bf16      SDPA in BF16 (the floor)
  8bitbwd   nvfp4.attention(backward="8bit")
  bf16bwd   nvfp4.attention(backward="bf16") (the kernels of qb_p2)
  qb        flash backward(q', k', v', our one-term-P o, our lse)
  qb_lse    flash backward(q', k', v', exact o of (q', k', v'), our lse)   -- isolates the output (Delta) error
  qb_p1/p2  flash backward(q', k', v', emulated o with P in one / two e4m3 terms, our lse)
(docs/DESIGN.md §7: why the one-term forward's o breaks dQ and the two-term one does not.)
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from nvfp4 import _library as _lib  # noqa: E402
from nvfp4 import reference as R  # noqa: E402
from nvfp4.attention import attention  # noqa: E402

LN2 = math.log(2.0)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def flash_bwd(do, q, k, v, o, lse, causal, scale):
    rng = torch.zeros(2, dtype=torch.uint64, device=q.device)
    return torch.ops.aten._scaled_dot_product_flash_attention_backward(
        do, q, k, v, o, lse, None, None, q.shape[2], k.shape[2], 0.0, causal, rng,
        torch.zeros((), dtype=torch.uint64, device=q.device), scale=scale)


def quantized_inputs(q, k, v):
    N = q.shape[2]
    qf = R.int8_token_fake(q.float())
    kf = R.int8_token_fake(k.float() - k.float().mean(2, keepdim=True))
    vf = R.mxfp8_fake(R._pad_seq(v.float()), dim=-2)[..., :N, :]
    return qf, kf, vf


def emul_fwd(qf, kf, vf, scale, causal, terms):
    """Forward with P = e4m3(128 p) (terms=1, as the kernel) or + its e4m3 residual (terms=2), fp32 otherwise."""
    outs = []
    N = qf.shape[2]
    mask = torch.ones(N, N, dtype=torch.bool, device=qf.device).triu(1)
    for b in range(qf.shape[0]):
        s = (qf[b] @ kf[b].transpose(-1, -2)) * scale
        if causal:
            s = s.masked_fill(mask, float("-inf"))
        p = torch.exp(s - s.amax(-1, keepdim=True))
        l = p.sum(-1, keepdim=True)
        hi = (p * 128).to(torch.float8_e4m3fn).float()
        pq = hi
        if terms == 2:
            pq = hi + ((p * 128 - hi) * 16).to(torch.float8_e4m3fn).float() / 16
        outs.append((pq / 128) @ vf[b] / l)
        del s, p, hi, pq
    return torch.stack(outs)


def run(q, k, v, do, causal=True):
    B, H, N, D = q.shape
    scale = D ** -0.5
    dt = q.dtype
    res = {}
    # exact
    qe, ke, ve = (t.float().requires_grad_() for t in (q, k, v))
    oe = F.scaled_dot_product_attention(qe, ke, ve, is_causal=causal)
    oe.backward(do.float())
    ex = (oe.detach(), qe.grad, ke.grad, ve.grad)
    # bf16 SDPA
    qb, kb, vb = (t.clone().requires_grad_() for t in (q, k, v))
    ob = F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal)
    ob.backward(do)
    res["bf16"] = (ob.detach(), qb.grad, kb.grad, vb.grad)
    # the 8-bit backward
    q1, k1, v1 = (t.clone().requires_grad_() for t in (q, k, v))
    o1 = attention(q1, k1, v1, causal=causal, backward="8bit")
    o1.backward(do)
    res["8bitbwd"] = (o1.detach(), q1.grad, k1.grad, v1.grad)
    q2, k2, v2 = (t.clone().requires_grad_() for t in (q, k, v))
    o2 = attention(q2, k2, v2, causal=causal, backward="bf16")
    o2.backward(do)
    res["bf16bwd"] = (o2.detach(), q2.grad, k2.grad, v2.grad)
    # our forward's o and lse
    kmean = k.mean(dim=2, dtype=torch.float32)
    q8, qs, _, _, _, _, _ = _lib.attn_quant(q, None, None, True, False, False)
    k8, ks, _, _, _, _, _ = _lib.attn_quant(k, kmean, None, True, False, False)
    _, _, vt, vts, _, _, _ = _lib.attn_quant(v, None, None, False, True, False)
    o8, lse8 = _lib.attn_fwd(q8, qs, k8, ks, vt, vts, N, causal, scale, dt)
    o8 = o8.transpose(1, 2)
    lse_nat = (lse8[:, :, :N] * LN2).contiguous()
    qf, kf, vf = quantized_inputs(q, k, v)
    qq, kq, vq = (t.to(dt) for t in (qf, kf, vf))
    # exact attention of the quantized inputs (o in input dtype, lse natural)
    oq, lseq = torch.ops.aten._scaled_dot_product_flash_attention(qq, kq, vq, 0.0, causal, False, scale=scale)[:2]
    outs = {"qb": (o8, lse_nat), "qb_lse": (oq, lse_nat)}
    for terms in (1, 2):
        outs[f"qb_p{terms}"] = (emul_fwd(qf, kf, vf, scale, causal, terms).to(dt), lse_nat)
    for name, (o, lse) in outs.items():
        dq, dk, dv = flash_bwd(do, qq, kq, vq, o, lse, causal, scale)
        res[name] = (o, dq, dk, dv)
    out = {}
    for name, r in res.items():
        out[name] = tuple(rel(a, b) for a, b in zip(r, ex))
    lse_err = (lse_nat - lseq).abs().max().item()
    return out, lse_err


def show(tag, out, lse_err):
    s = " | ".join(f"{n} o {e[0]:.4f} dq {e[1]:.4f} dk {e[2]:.4f} dv {e[3]:.4f}" for n, e in out.items())
    print(f"{tag} | max|lse8 - lse(q')| {lse_err:.2e} | {s}", flush=True)


torch.manual_seed(0)
for D in (64, 128):
    for N in (1024, 4096):
        q, k, v, do = (torch.randn(2, 4, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(4))
        k = k + 0.5 * torch.randn(1, 4, 1, D, device="cuda", dtype=torch.bfloat16)  # channel offsets
        show(f"random D={D} N={N}", *run(q, k, v, do))

if os.environ.get("REAL", "1") == "1":
    import train_gpt as tg  # noqa: E402

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
    for li, c in enumerate(caps[:10]):
        q, k, v, do = (c[n].contiguous() for n in ("q", "k", "v", "do"))
        show(f"real layer {li} N={T}", *run(q, k, v, do))
