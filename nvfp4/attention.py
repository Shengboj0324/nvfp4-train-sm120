"""MXFP8 flash attention for sm_120 (forward and a deterministic backward), a drop-in for causal or full
``F.scaled_dot_product_attention`` in training.

Q K^T and dO V^T run on INT8 tensor cores with per-token scales; P V, P^T dO and dS^T Q on block-scaled FP8
(E4M3 with one power-of-two scale per 32 elements, MXFP8); dQ = dS K, which cancels to a small difference of
large terms, in BF16. All with FP32 accumulation; softmax statistics, the online rescaling and every reduction
stay in FP32. K is centred by its per-channel mean over the sequence before quantization (``smooth_k``), which
leaves the attention output unchanged (a per-row constant in the scores) and its gradients too, but takes the
channel offsets typical of keys out of the quantization range.

    from nvfp4.attention import attention
    o = attention(q, k, v, causal=True)      # q [B, H, N, D], k / v [B, Hkv, N, D], D in {64, 128}

Inputs that the kernels don't cover (other head dims, fp32, attention masks, dropout) fall back to PyTorch's
scaled_dot_product_attention. The returned tensor is a [B, H, N, D] view of [B, N, H, D] memory, so the usual
``o.transpose(1, 2).reshape(B, N, H * D)`` is free, and the input gradients come back the same way.
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
        q8, qs, qt, qts, _, _ = _lib.attn_quant(q, None, None, True, grad, False)
        # K: INT8 rows, plus (for dQ = dS K, done in BF16) a BF16 copy, both centred by the mean
        k8, ks, kt, kts, _, k16 = _lib.attn_quant(k, kmean, None, True, grad, grad)
        v8, vs, vt, vts, _, _ = _lib.attn_quant(v, None, None, grad, True, False)
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
        g8, gs, gt, gts, delta, _ = _lib.attn_quant(do.to(dtype), None, o, True, True, False)
        dq, dk, dv = _lib.attn_bwd(q8, qs, qt, qts, k8, ks, kt, kts, k16, v8, vs, g8, gs, gt, gts, lse, delta, N,
                                   causal, scale, dtype, q_stride, k_stride)
        return dq, dk, dv, None, None, None


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False,
              scale: float | None = None, smooth_k: bool = True) -> torch.Tensor:
    """softmax(scale * q k^T [+ causal mask]) v with MXFP8 tensor-core GEMMs (see the module docstring).

    q: [B, H, N, D]; k, v: [B, Hkv, N, D] with H a multiple of Hkv (grouped-query attention); bf16 or fp16;
    D in {64, 128}; any strides with a contiguous last dim. scale defaults to D ** -0.5."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if not (supported(q, k, v) and scale > 0):
        rep = q.shape[1] // max(1, k.shape[1])
        if rep > 1:
            k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
    return _AttentionFn.apply(q, k, v, causal, float(scale), smooth_k)
