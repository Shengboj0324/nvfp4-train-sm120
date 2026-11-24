"""MXFP8 flash attention for sm_120 (forward and a deterministic backward), a drop-in for causal or full
``F.scaled_dot_product_attention`` in training.

All five attention matrix products (Q K^T and P V forward; dO V^T, P^T dO, dS^T Q, dS K and the recomputed
Q K^T backward) run on the block-scaled FP8 tensor-core path: E4M3 operands with one power-of-two scale per 32
elements of the reduction dimension (MXFP8), FP32 accumulation. On an RTX 5090 that path has 3.6x the
throughput of BF16 with FP32 accumulation. Softmax statistics, the online rescaling and every reduction stay
in FP32. K is centred by its per-channel mean over the sequence before quantization (``smooth_k``), which
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
        q8, qs, qt, qts, _ = _lib.attn_quant(q, None, None, True, grad)
        k8, ks, kt, kts, _ = _lib.attn_quant(k, kmean, None, True, grad)
        v8, vs, vt, vts, _ = _lib.attn_quant(v, None, None, grad, True)
        o, lse = _lib.attn_fwd(q8, qs, k8, ks, vt, vts, N, causal, scale, q.dtype)
        o = o.transpose(1, 2)
        if grad:
            ctx.save_for_backward(q8, qs, qt, qts, k8, ks, kt, kts, v8, vs, o, lse)
        ctx.cfg = (N, causal, scale, list(q.stride()), list(k.stride()))
        return o

    @staticmethod
    def backward(ctx, do):
        q8, qs, qt, qts, k8, ks, kt, kts, v8, vs, o, lse = ctx.saved_tensors
        N, causal, scale, q_stride, k_stride = ctx.cfg
        g8, gs, gt, gts, delta = _lib.attn_quant(do.to(o.dtype), None, o, True, True)
        dq, dk, dv = _lib.attn_bwd(q8, qs, qt, qts, k8, ks, kt, kts, v8, vs, g8, gs, gt, gts, lse, delta, N, causal,
                                   scale, o.dtype, q_stride, k_stride)
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
