"""8-bit flash attention for sm_120, a drop-in for causal or full ``F.scaled_dot_product_attention`` in
training.

The forward runs Q K^T on INT8 tensor cores with per-token scales and P V on block-scaled FP8 (E4M3 with one
power-of-two scale per 32 elements, MXFP8), with FP32 accumulation, softmax statistics and online rescaling.
The default backward (``backward="bf16"``) is PyTorch's flash-attention backward of exactly that forward: it
runs on the dequantized 8-bit operands with the forward's log-sum-exp, and the forward then computes P V with P
in two e4m3 terms so that its output gives a consistent Delta (docs/DESIGN.md §7.1). ``backward="8bit"`` is
this package's deterministic backward: dO V^T in INT8, P^T dO and dS^T Q in MXFP8, dQ = dS K (a small
difference of large terms) in BF16. K is centred by its per-channel mean over the sequence before
quantization (``smooth_k``), which leaves the attention output unchanged (a per-row constant in the scores)
and its gradients too, but takes the channel offsets typical of keys out of the quantization range.

    from nvfp4.attention import attention
    o = attention(q, k, v, causal=True)      # q [B, H, N, D], k / v [B, Hkv, N, D], D in {64, 128}

Inputs that the kernels don't cover (other head dims, fp32, attention masks, dropout) fall back to PyTorch's
scaled_dot_product_attention. The returned tensor is a [B, H, N, D] view of [B, N, H, D] memory, so the usual
``o.transpose(1, 2).reshape(B, N, H * D)`` is free (``backward="8bit"`` also returns the input gradients that
way).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from nvfp4 import _library as _lib


def supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """True if the MXFP8 kernels handle these inputs (otherwise ``attention`` falls back to SDPA)."""
    if _lib._C is None or not q.is_cuda or q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return False
    B, H, N, D = q.shape
    if D not in (64, 128) or q.dtype not in (torch.bfloat16, torch.float16) or N == 0:
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype or k.shape != v.shape:
        return False
    if k.shape[0] != B or k.shape[2] != N or k.shape[3] != D or k.shape[1] == 0 or H % k.shape[1]:
        return False
    try:
        _lib.require(q)
    except RuntimeError:
        return False
    return True


class _AttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal: bool, scale: float, smooth_k: bool):
        N = q.shape[2]
        grad = any(ctx.needs_input_grad[:3])
        kmean = k.mean(dim=2, dtype=torch.float32) if smooth_k else None
        q8, qs, qt, qts, _, _, _ = _lib.attn_quant(q, None, None, True, grad, False)
        # K: INT8 rows, plus (for dQ = dS K, done in BF16) a BF16 copy, both centred by the mean
        k8, ks, kt, kts, _, k16, _ = _lib.attn_quant(k, kmean, None, True, grad, grad)
        v8, vs, vt, vts, _, _, _ = _lib.attn_quant(v, None, None, grad, True, False)
        o, lse = _lib.attn_fwd(q8, qs, k8, ks, vt, vts, N, causal, scale, q.dtype)
        o = o.transpose(1, 2)
        if grad:
            ctx.save_for_backward(q8, qs, qt, qts, k8, ks, kt, kts, k16, v8, vs, o, lse)
        ctx.cfg = (N, causal, scale, list(q.stride()), list(k.stride()), q.dtype)
        return o

    @staticmethod
    def backward(ctx, do):
        q8, qs, qt, qts, k8, ks, kt, kts, k16, v8, vs, o, lse = ctx.saved_tensors
        N, causal, scale, q_stride, k_stride, dtype = ctx.cfg
        g8, gs, gt, gts, delta, _, _ = _lib.attn_quant(do.to(dtype), None, o, True, True, False)
        dq, dk, dv = _lib.attn_bwd(q8, qs, qt, qts, k8, ks, kt, kts, k16, v8, vs, g8, gs, gt, gts, lse, delta, N,
                                   causal, scale, dtype, q_stride, k_stride)
        return dq, dk, dv, None, None, None


_LN2 = 0.6931471805599453
_FLASH_BWD: bool | None = None  # PyTorch's flash-attention backward is available (None: not tried yet)


def _math_backward(do, q, k, v, o, lse, causal: bool, scale: float, rows: int = 512):
    """The flash-attention backward's math, for PyTorch builds without flash attention: in fp32, one block of
    query rows at a time (scores [B, H, rows, N]), P recomputed from the forward's natural-log lse.
    dV = P^T dO, dS = P (dO V^T - rowsum(dO O)), dQ = scale dS K, dK = scale dS^T Q (deterministic)."""
    B, H, N, D = q.shape
    qf, kf, vf = q.float(), k.float(), v.float()
    delta = (do.float() * o.float()).sum(-1)
    dq = torch.empty_like(qf)
    dk, dv = torch.zeros_like(kf), torch.zeros_like(vf)
    cols = torch.arange(N, device=q.device)
    for r0 in range(0, N, rows):
        r1 = min(N, r0 + rows)
        s = torch.matmul(qf[:, :, r0:r1], kf.transpose(-1, -2)) * scale
        if causal:
            s.masked_fill_(cols[None, :] > torch.arange(r0, r1, device=q.device)[:, None], float("-inf"))
        p = torch.exp(s - lse[:, :, r0:r1, None])
        g = do[:, :, r0:r1].float()
        dv += torch.matmul(p.transpose(-1, -2), g)
        ds = p * (torch.matmul(g, vf.transpose(-1, -2)) - delta[:, :, r0:r1, None])
        dq[:, :, r0:r1] = torch.matmul(ds, kf) * scale
        dk += torch.matmul(ds.transpose(-1, -2), qf[:, :, r0:r1]) * scale
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class _AttentionBf16BwdFn(torch.autograd.Function):
    """8-bit forward, exact backward of it: PyTorch's flash-attention backward (input precision) run on the
    dequantized 8-bit operands (int8 q / centred k, MXFP8 v), which are exactly what the forward multiplied, with
    the forward's log-sum-exp and output. The backward's recomputed scores then agree with the forward's
    softmax statistics, and P V with P carried to ~2^-8 (two e4m3 terms) keeps Delta = rowsum(dO O)
    consistent. The gradients are those of the quantized forward (straight-through for the quantizers)."""

    @staticmethod
    def forward(ctx, q, k, v, causal: bool, scale: float, smooth_k: bool):
        N = q.shape[2]
        grad = any(ctx.needs_input_grad[:3])
        kmean = k.mean(dim=2, dtype=torch.float32) if smooth_k else None
        q8, qs, _, _, _, _, qd = _lib.attn_quant(q, None, None, True, False, False, grad)
        k8, ks, _, _, _, _, kd = _lib.attn_quant(k, kmean, None, True, False, False, grad)
        _, _, vt, vts, _, _, vd = _lib.attn_quant(v, None, None, False, True, False, grad)
        o, lse = _lib.attn_fwd(q8, qs, k8, ks, vt, vts, N, causal, scale, q.dtype, True)
        o = o.transpose(1, 2)
        if grad:
            lse_nat = (lse[:, :, :N] * _LN2).contiguous()  # log2 domain -> natural log, as flash attention keeps it
            ctx.save_for_backward(qd[:, :, :N], kd[:, :, :N], vd[:, :, :N], o, lse_nat)
        ctx.cfg = (causal, scale)
        return o

    @staticmethod
    def backward(ctx, do):
        global _FLASH_BWD
        qd, kd, vd, o, lse = ctx.saved_tensors
        causal, scale = ctx.cfg
        rep = qd.shape[1] // kd.shape[1]
        ke, ve = (kd.repeat_interleave(rep, 1), vd.repeat_interleave(rep, 1)) if rep > 1 else (kd, vd)
        do = do.to(qd.dtype)
        dq = None
        if _FLASH_BWD is not False:
            zero = torch.zeros((), dtype=torch.uint64, device=qd.device)
            try:
                dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward(
                    do, qd, ke, ve, o, lse, None, None, qd.shape[2], kd.shape[2], 0.0, causal,
                    torch.zeros(2, dtype=torch.uint64, device=qd.device), zero, scale=scale)
                _FLASH_BWD = True
            except RuntimeError as e:  # PyTorch built without flash attention (its Windows wheels)
                if _FLASH_BWD or "FLASH" not in str(e).upper():
                    raise
                _FLASH_BWD = False
        if dq is None:
            dq, dk, dv = _math_backward(do, qd, ke, ve, o, lse, causal, scale)
        if rep > 1:
            B, Hk, N, D = kd.shape
            dk = dk.view(B, Hk, rep, N, D).sum(2)
            dv = dv.view(B, Hk, rep, N, D).sum(2)
        return dq, dk, dv, None, None, None


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False,
              scale: float | None = None, smooth_k: bool = True, backward: str = "bf16") -> torch.Tensor:
    """softmax(scale * q k^T [+ causal mask]) v with 8-bit tensor-core GEMMs (see the module docstring).

    q: [B, H, N, D]; k, v: [B, Hkv, N, D] with H a multiple of Hkv (grouped-query attention); bf16 or fp16;
    D in {64, 128}; any strides with a contiguous last dim. scale defaults to D ** -0.5.

    backward="bf16" (default): the exact gradient of the 8-bit forward, from PyTorch's flash-attention backward
    in the input precision (as fast as BF16 attention's backward; not bitwise deterministic unless
    torch.use_deterministic_algorithms is on; PyTorch builds without flash attention, such as its Windows wheels,
    get the same math in fp32, a block of query rows at a time). backward="8bit": this package's 8-bit backward (deterministic,
    2-4x larger dK / dV errors) with a slightly faster forward (P V in one e4m3 term)."""
    if backward not in ("bf16", "8bit"):
        raise ValueError(f"backward must be 'bf16' or '8bit', got {backward!r}")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if not (supported(q, k, v) and scale > 0):
        rep = q.shape[1] // max(1, k.shape[1])
        if rep > 1:
            k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
    if backward == "bf16":
        return _AttentionBf16BwdFn.apply(q, k, v, causal, float(scale), smooth_k)
    return _AttentionFn.apply(q, k, v, causal, float(scale), smooth_k)
