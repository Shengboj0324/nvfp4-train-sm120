"""Usage: python bench/attention_skip_probe.py  (STEPS=300 by default; needs ~/nvfp4_data)

Probe for skipping the residual P V product of the two-term forward (DESIGN.md §7.1) on key blocks whose
probabilities are all negligible: on the activations of a GPT-124M trained briefly at 4096 tokens, for each
threshold tau (in the kernel's units, where the running row maximum is 2^7), the fraction of (16-row,
32-key) MMA tiles that would skip, and the gradient errors of flash attention's backward when the forward's
residual is dropped on those tiles (emulated; the running maximum per 64-key tile as in the kernel, without
the lazy-rescale slack)."""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import train_gpt as tg  # noqa: E402

from nvfp4 import _library as _lib  # noqa: E402
from nvfp4 import reference as R  # noqa: E402

LN2 = math.log(2.0)
TAUS = (0.0, 2.0 ** -4, 2.0 ** -2, 1.0, 4.0)  # 0: no skipping
ELEM = []


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def flash_bwd(do, q, k, v, o, lse, scale):
    rng = torch.zeros(2, dtype=torch.uint64, device=q.device)
    return torch.ops.aten._scaled_dot_product_flash_attention_backward(
        do, q, k, v, o, lse, None, None, q.shape[2], k.shape[2], 0.0, True, rng,
        torch.zeros((), dtype=torch.uint64, device=q.device), scale=scale)


def emul(qf, kf, vf, scale, tau):
    """Two-term-P forward (fp32) with the residual dropped on tiles whose p (x 2^7, relative to the running
    maximum) stays below tau; returns (o, fraction of computed tiles skipped)."""
    B, H, N, D = qf.shape
    outs, skipped, total = [], 0, 0
    mask = torch.ones(N, N, dtype=torch.bool, device=qf.device).triu(1)
    for b in range(B):
        for h in range(H):
            s = (qf[b, h] @ kf[b, h].T) * (scale / LN2)  # log2 domain
            s = s.masked_fill(mask, float("-inf"))
            # running maximum after each 64-key tile (the kernel's m when it quantizes that tile's p)
            tmax = s.view(N, N // 64, 64).amax(-1)
            run = torch.cummax(tmax, dim=1).values.repeat_interleave(64, dim=1)
            p = torch.exp2(s - run + 7)  # the kernel's scaled p (<= 2^7 here)
            p = torch.nan_to_num(p, nan=0.0)
            hi = p.to(torch.float8_e4m3fn).float()
            lo = ((p - hi) * 16).to(torch.float8_e4m3fn).float() / 16
            tiles = p.view(N // 16, 16, N // 32, 32).amax(dim=(1, 3))  # [row tile, key block]
            r = torch.arange(N // 16, device=qf.device)[:, None] * 16 + 15
            c = torch.arange(N // 32, device=qf.device)[None, :] * 32
            live = c <= r  # tiles the causal kernel computes
            skip = (tiles < tau) & live
            skipped += skip.sum().item()
            total += live.sum().item()
            if b == 0 and h == 0 and tau > 0:  # element-level view: how many live p are below tau at all
                cm = ~mask
                ELEM.append((tau, ((p < tau) & cm).sum().item() / cm.sum().item()))
            keep = (~skip).repeat_interleave(16, 0).repeat_interleave(32, 1)
            pq = hi + lo * keep
            # exact normalizer of the unquantized p, relative to the row maximum
            fin = run[:, -1:]
            l = (p * torch.exp2(run - fin)).sum(-1, keepdim=True)
            o = (pq * torch.exp2(run - fin)) @ vf[b, h] / l
            outs.append(o)
            del s, run, p, hi, lo, pq, keep
    return torch.stack(outs).view(B, H, N, D), skipped / total


STEPS = int(os.environ.get("STEPS", "300"))
T = 4096
torch.manual_seed(0)
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

print("layer | tau: skipped tiles, dq dk dv error (vs exact fp32 attention)")
for li, c in enumerate(caps[:10]):
    q, k, v, do = (c[n].contiguous() for n in ("q", "k", "v", "do"))
    N, D = q.shape[2], q.shape[3]
    scale = D ** -0.5
    qe, ke, ve = (t.float().requires_grad_() for t in (q, k, v))
    F.scaled_dot_product_attention(qe, ke, ve, is_causal=True).backward(do.float())
    kmean = k.mean(dim=2, dtype=torch.float32)
    q8, qs, _, _, _, _, qd = _lib.attn_quant(q, None, None, True, False, False, True)
    k8, ks, _, _, _, _, kd = _lib.attn_quant(k, kmean, None, True, False, False, True)
    _, _, vt, vts, _, _, vd = _lib.attn_quant(v, None, None, False, True, False, True)
    _, lse8 = _lib.attn_fwd(q8, qs, k8, ks, vt, vts, N, True, scale, q.dtype, True)
    lse = (lse8[:, :, :N] * LN2).contiguous()
    qd, kd, vd = qd[:, :, :N], kd[:, :, :N], vd[:, :, :N]
    parts = []
    for tau in TAUS:
        o, frac = emul(qd.float(), kd.float(), vd.float(), scale, tau)
        dq, dk, dv = flash_bwd(do, qd, kd, vd, o.to(q.dtype), lse, scale)
        parts.append(f"{tau:g}: {frac:.2f} {rel(dq, qe.grad):.4f} {rel(dk, ke.grad):.4f} {rel(dv, ve.grad):.4f}")
    print(f"{li:2d} | " + " | ".join(parts), flush=True)
    print("    elements below tau (batch 0, head 0): " + ", ".join(f"{t:g}: {f:.3f}" for t, f in ELEM), flush=True)
    ELEM.clear()
