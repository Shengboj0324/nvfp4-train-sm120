"""Usage: python bench/attention_pv_probe.py  (STEPS=300 by default; needs ~/nvfp4_data; REAL=0: random data only)

Probe: cheaper forms of the forward's two-term P V for the exact backward (attention(backward="bf16")). The
second term e4m3(residual) V costs one more MXFP8 MMA per P V MMA (the forward's 1.5-2.1x over SDPA drops to
1.2-1.6x). Candidates, each emulated in fp32 and fed to PyTorch's flash backward on the quantized q / k / v
(as bench/attention_qbwd_probe.py), errors vs exact fp32 attention:
  p1        P in one e4m3 term (no residual)
  p2        P in two e4m3 terms (the kernel today)
  r4f<s>    residual in E2M1 at a fixed power-of-two scale 2^s (s = 1, 2: no per-block amax), times V in
            NVFP4 (E2M1, one E4M3 scale per 16 tokens of a channel): one kind::mxf4nvf4 MMA at twice the
            MXFP8 rate instead of a second MXFP8 MMA
  r4b       residual in NVFP4 too (one E4M3 scale per row and 16 keys, from the block amax), V in NVFP4
  r4b_v8    residual in NVFP4, times the MXFP8 V (isolates the residual's own rounding)
  i8x2      P in two INT8 terms at a fixed scale (127 / row maximum, then 127 x the residual), V in INT8 with one
            scale per channel (the backward then runs on that V): s8 MMAs instead of MXFP8
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

LN2 = math.log(2.0)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def flash_bwd(do, q, k, v, o, lse, causal, scale):
    rng = torch.zeros(2, dtype=torch.uint64, device=q.device)
    return torch.ops.aten._scaled_dot_product_flash_attention_backward(
        do, q, k, v, o, lse, None, None, q.shape[2], k.shape[2], 0.0, causal, rng,
        torch.zeros((), dtype=torch.uint64, device=q.device), scale=scale)


def e2m1_fake(x):
    return R.e2m1_decode(R.e2m1_round_rne(x)).to(x.dtype)


def nvfp4_fake(x, dim):
    """E2M1 with one E4M3 scale (amax / 6) per 16 elements along ``dim`` (no tensor scale: values are O(1))."""
    xm = x.movedim(dim, -1)
    bl = xm.reshape(*xm.shape[:-1], xm.shape[-1] // 16, 16)
    sc = (bl.abs().amax(-1, keepdim=True) / 6.0).clamp(min=2.0 ** -9).to(torch.float8_e4m3fn).float()
    sc = torch.where(sc == 0, torch.ones_like(sc), sc)
    out = e2m1_fake(bl / sc) * sc
    return out.reshape(xm.shape).movedim(-1, dim)


def quantized_inputs(q, k, v):
    N = q.shape[2]
    qf = R.int8_token_fake(q.float())
    kf = R.int8_token_fake(k.float() - k.float().mean(2, keepdim=True))
    vf = R.mxfp8_fake(R._pad_seq(v.float()), dim=-2)[..., :N, :]
    return qf, kf, vf


def emul_fwd(qf, kf, vf, scale, causal, mode, v4=None):
    outs = []
    N = qf.shape[2]
    mask = torch.ones(N, N, dtype=torch.bool, device=qf.device).triu(1)
    for b in range(qf.shape[0]):
        s = (qf[b] @ kf[b].transpose(-1, -2)) * scale
        if causal:
            s = s.masked_fill(mask, float("-inf"))
        p = torch.exp(s - s.amax(-1, keepdim=True))
        del s
        l = p.sum(-1, keepdim=True)
        if mode == "i8x2":
            hi = torch.round(p * 127)
            lo = torch.round((p * 127 - hi) * 127)
            outs.append(((hi + lo / 127) / 127) @ vf[b] / l)
            continue
        hi = (p * 128).to(torch.float8_e4m3fn).float()
        if mode == "p1":
            o = hi @ vf[b]
        else:
            r = p * 128 - hi
            del p
            if mode == "p2":
                o = (hi + (r * 16).to(torch.float8_e4m3fn).float() / 16) @ vf[b]
            elif mode.startswith("r4f"):
                sc = 2.0 ** int(mode[3:])
                o = hi @ vf[b] + (e2m1_fake(r / sc) * sc) @ v4[b]
            elif mode == "r4b":
                o = hi @ vf[b] + nvfp4_fake(r, -1) @ v4[b]
            elif mode == "r4b_v8":
                o = hi @ vf[b] + nvfp4_fake(r, -1) @ vf[b]
            else:
                raise ValueError(mode)
        outs.append(o / 128 / l)
        del hi, o
    return torch.stack(outs)


MODES = ("p1", "p2", "r4f1", "r4f2", "r4b", "r4b_v8", "i8x2")


def run(q, k, v, do, causal=True):
    B, H, N, D = q.shape
    scale = D ** -0.5
    dt = q.dtype
    res = {}
    qe, ke, ve = (t.float().requires_grad_() for t in (q, k, v))
    oe = F.scaled_dot_product_attention(qe, ke, ve, is_causal=causal)
    oe.backward(do.float())
    ex = (oe.detach(), qe.grad, ke.grad, ve.grad)
    del qe, ke, ve, oe
    qb, kb, vb = (t.clone().requires_grad_() for t in (q, k, v))
    ob = F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal)
    ob.backward(do)
    res["bf16"] = (ob.detach(), qb.grad, kb.grad, vb.grad)
    kmean = k.mean(dim=2, dtype=torch.float32)
    q8, qs, _, _, _, _, _, _ = _lib.attn_quant(q, None, None, True, False, False)
    k8, ks, _, _, _, _, _, _ = _lib.attn_quant(k, kmean, None, True, False, False)
    _, _, vt, vts, _, _, _, _ = _lib.attn_quant(v, None, None, False, True, False)
    _, lse8 = _lib.attn_fwd(q8, qs, k8, ks, vt, vts, N, causal, scale, dt)
    lse_nat = (lse8[:, :, :N] * LN2).contiguous()
    qf, kf, vf = quantized_inputs(q, k, v)
    qq, kq, vq = (t.to(dt) for t in (qf, kf, vf))
    v4 = nvfp4_fake(R._pad_seq(vf), -2)[..., :N, :]
    vi8 = R.int8_token_fake(v.float().transpose(-1, -2)).transpose(-1, -2)
    for mode in MODES:
        vin = vi8 if mode == "i8x2" else vf
        o = emul_fwd(qf, kf, vin, scale, causal, mode, v4).to(dt)
        vb = vi8.to(dt) if mode == "i8x2" else vq
        dq, dk, dv = flash_bwd(do, qq, kq, vb, o, lse_nat, causal, scale)
        res[mode] = (o, dq, dk, dv)
        del o, dq, dk, dv
    return {name: tuple(rel(a, b) for a, b in zip(r, ex)) for name, r in res.items()}


def show(tag, out):
    print(tag, flush=True)
    for n, e in out.items():
        print(f"  {n:8s} o {e[0]:.4f} dq {e[1]:.4f} dk {e[2]:.4f} dv {e[3]:.4f}", flush=True)


torch.manual_seed(0)
for D in (64, 128):
    for N in (1024, 4096):
        q, k, v, do = (torch.randn(2, 4, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(4))
        k = k + 0.5 * torch.randn(1, 4, 1, D, device="cuda", dtype=torch.bfloat16)
        show(f"random D={D} N={N}", run(q, k, v, do))

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
    del model, opt
    torch.cuda.empty_cache()
    for li, c in enumerate(caps[:12]):
        q, k, v, do = (c[n].contiguous() for n in ("q", "k", "v", "do"))
        show(f"real layer {li} N={T}", run(q, k, v, do))
