"""Usage: python bench/attention_real_errors.py  (STEPS=800 by default; needs ~/nvfp4_data)

8-bit attention errors on real activations: train GPT-124M (6 heads of 128) briefly at 4096 context with BF16
attention, capture q, k, v and dO of every layer, and compare nvfp4.attention to exact attention (fp32), at 4096
and at 512 tokens, with an emulation breakdown of the forward error by quantization source."""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import train_gpt as tg  # noqa: E402

from nvfp4 import reference as R  # noqa: E402
from nvfp4.attention import attention  # noqa: E402

STEPS = int(os.environ.get("STEPS", "800"))
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
    if s % 200 == 0:
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


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def exact(q, k, v, do):
    q, k, v = (t.float().requires_grad_() for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    o.backward(do.float())
    return o.detach(), q.grad, k.grad, v.grad


def ours(q, k, v, do):
    q, k, v = (t.clone().requires_grad_() for t in (q, k, v))
    o = attention(q, k, v, causal=True)
    o.backward(do)
    return o.detach(), q.grad, k.grad, v.grad


def emul(q, k, v, qk="int8", vq=True, pq=True):
    """Forward emulation with chosen quantizations (fp32)."""
    D = q.shape[-1]
    qf, kf, vf = q.float(), k.float() - k.float().mean(2, keepdim=True), v.float()
    if qk == "int8":
        qf, kf = R.int8_token_fake(qf), R.int8_token_fake(kf)
    if vq:
        vf = R.mxfp8_fake(R._pad_seq(vf), dim=-2)[..., :v.shape[2], :]
    s = qf @ kf.transpose(-1, -2) / math.sqrt(D)
    n = s.shape[-1]
    s = s.masked_fill(torch.ones(n, n, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
    p = torch.exp(s - s.amax(-1, keepdim=True))
    l = p.sum(-1, keepdim=True)
    if pq:
        p = (p * 128).clamp(max=448).to(torch.float8_e4m3fn).float() / 128
    return p @ vf / l


print("layer | N | o err | dq dk dv err | fwd emulation: qk-int8 only, v-fp8 only, p-e4m3 only | "
      "q / k outlier ratio (max|ch| / rms) | attn entropy (nats)")
for li, c in enumerate(caps[:10]):  # the blocks that would use 8-bit attention (hp_last = 2)
    for N in (4096, 512):
        q, k, v, do = (c[n][:, :, :N] for n in ("q", "k", "v", "do"))
        q, k, v, do = (t.contiguous() for t in (q, k, v, do))
        eo, edq, edk, edv = exact(q, k, v, do)
        oo, odq, odk, odv = ours(q, k, v, do)
        e_qk = rel(emul(q, k, v, "int8", False, False), eo)
        e_v = rel(emul(q, k, v, None, True, False), eo)
        e_p = rel(emul(q, k, v, None, False, True), eo)
        out_q = (q.float().abs().amax(-1) / q.float().pow(2).mean(-1).sqrt()).mean().item()
        out_k = ((k.float() - k.float().mean(2, keepdim=True)).abs().amax(-1)
                 / (k.float() - k.float().mean(2, keepdim=True)).pow(2).mean(-1).sqrt()).mean().item()
        with torch.no_grad():
            s = q.float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])
            s = s.masked_fill(torch.ones(N, N, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
            pr = torch.softmax(s, -1)
            ent = -(pr * torch.log(pr.clamp_min(1e-30))).sum(-1).mean().item()
        print(f"{li:2d} | {N:4d} | {rel(oo, eo):.4f} | {rel(odq, edq):.4f} {rel(odk, edk):.4f} {rel(odv, edv):.4f} | "
              f"{e_qk:.4f} {e_v:.4f} {e_p:.4f} | {out_q:.1f} / {out_k:.1f} | {ent:.2f}", flush=True)
