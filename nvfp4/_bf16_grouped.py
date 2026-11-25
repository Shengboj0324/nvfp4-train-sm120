"""Grouped BF16 / FP32 GEMMs for the MoE layer's high-precision path, without host synchronization.

The NVFP4 MoE layer lays its routed tokens out in a worst-case-sized buffer whose layout lives on the GPU
(nvfp4.moe). These Triton kernels run the high-precision experts (before ``to_nvfp4``, or during a BF16
tail under ``ops.set_high_precision``) on the same buffer and tile table, so that path is also free of host
synchronization: CUDA-graph capturable and traced by ``torch.compile`` without graph breaks.
(``torch._grouped_mm`` would do, but on sm_120 it falls back to a loop that reads the group sizes on the
host.) Deterministic: no atomics, a fixed reduction order per output tile.

  gmm_m(x, w, tiles, ...)  ragged M: y[rows of tile t] = x[rows] @ w[g]^T (or @ w[g]), g = tiles[t]
  gmm_k(x, y, segs, ...)   ragged K: out[g] = x[k0:k0+len]^T @ y[k0:k0+len] per group (weight gradients)

Without Triton (PyTorch's Windows wheels ship none) ``available()`` is False and the MoE layer keeps its
per-expert loop.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

try:  # pragma: no cover - depends on the platform
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

ROWS = 128  # = the MoE layer's tile (nvfp4.moe._ROWS): every group's rows start on a tile


def available() -> bool:
    return triton is not None


if triton is not None:
    @triton.jit
    def _gmm_m_kernel(a, b, d, bias, tile_group, tile_rows, M, N, K, s_am, s_ak, s_be, s_bk, s_bn, s_dm, s_bias,
                      HAS_BIAS: tl.constexpr, PRECISION: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                      BK: tl.constexpr):
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(0)
        g = tl.load(tile_group + pid_m)
        ge = tl.maximum(g, 0)
        k_end = tl.where(g >= 0, K, 0)  # an unused tile reads nothing and stores zeros
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptr = a + rm[:, None].to(tl.int64) * s_am + rk[None, :] * s_ak
        b_ptr = b + ge.to(tl.int64) * s_be + rk[:, None] * s_bk + rn[None, :] * s_bn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, k_end, BK):
            kk = k + rk
            x = tl.load(a_ptr, mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
            w = tl.load(b_ptr, mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
            acc = tl.dot(x, w, acc, input_precision=PRECISION)
            a_ptr += BK * s_ak
            b_ptr += BK * s_bk
        if HAS_BIAS:  # fp32 add, one rounding (as cuBLAS's bias epilogue); padding rows get none
            row_end = pid_m * BM + tl.where(g >= 0, tl.load(tile_rows + pid_m), 0)
            bv = tl.load(bias + ge.to(tl.int64) * s_bias + rn, mask=rn < N, other=0.0).to(tl.float32)
            acc = tl.where(rm[:, None] < row_end, acc + bv[None, :], acc)
        tl.store(d + rm[:, None].to(tl.int64) * s_dm + rn[None, :], acc.to(d.dtype.element_ty),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    @triton.jit
    def _gmm_k_kernel(x, y, d, segs, P, Q, s_xm, s_xp, s_ym, s_yq, ACCUM: tl.constexpr, PRECISION: tl.constexpr,
                      BP: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        g = tl.program_id(1)
        tiles_q = tl.cdiv(Q, BQ)
        rp = (pid // tiles_q) * BP + tl.arange(0, BP)
        rq = (pid % tiles_q) * BQ + tl.arange(0, BQ)
        k0 = tl.load(segs + 3 * g)
        length = tl.load(segs + 3 * g + 1)
        r0 = tl.load(segs + 3 * g + 2)
        rk = tl.arange(0, BK)
        x_ptr = x + (k0 + rk)[None, :].to(tl.int64) * s_xm + rp[:, None] * s_xp
        y_ptr = y + (k0 + rk)[:, None].to(tl.int64) * s_ym + rq[None, :] * s_yq
        acc = tl.zeros((BP, BQ), dtype=tl.float32)
        for k in range(0, length, BK):  # lengths are multiples of the tile (128): no k mask
            xt = tl.load(x_ptr, mask=rp[:, None] < P, other=0.0)
            yv = tl.load(y_ptr, mask=rq[None, :] < Q, other=0.0)
            acc = tl.dot(xt, yv, acc, input_precision=PRECISION)
            x_ptr += BK * s_xm
            y_ptr += BK * s_ym
        out = d + (r0 + rp)[:, None].to(tl.int64) * Q + rq[None, :]
        mask = (rp[:, None] < P) & (rq[None, :] < Q)
        if ACCUM:
            acc += tl.load(out, mask=mask, other=0.0)
        tl.store(out, acc, mask=mask)


def _precision(dt: torch.dtype) -> str:
    return "ieee" if dt == torch.float32 else "tf32"  # (tf32 is ignored for 16-bit inputs)


def _gmm_m_impl(x: Tensor, w: Tensor, tiles: Tensor, transpose: bool, bias: Optional[Tensor],
                tile_rows: Optional[Tensor]) -> Tensor:
    """y [M, N] = x [M, K] @ w[g]^T (transpose; w [G, N, K]) or x @ w[g] (w [G, K, N]) per 128-row tile."""
    M, K = x.shape
    N = w.shape[1] if transpose else w.shape[2]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return out
    s_bk, s_bn = (w.stride(2), w.stride(1)) if transpose else (w.stride(1), w.stride(2))
    has_bias = bias is not None
    b = bias.contiguous() if has_bias else x
    tr = tile_rows if tile_rows is not None else tiles
    grid = (triton.cdiv(N, 128), tiles.numel())
    _gmm_m_kernel[grid](x, w, out, b, tiles, tr, M, N, K, x.stride(0), x.stride(1), w.stride(0), s_bk, s_bn,
                        out.stride(0), b.stride(0) if has_bias else 0, HAS_BIAS=has_bias,
                        PRECISION=_precision(x.dtype), BM=ROWS, BN=128, BK=32, num_warps=8, num_stages=3)
    return out


def _gmm_k_impl(x: Tensor, y: Tensor, segs: Tensor, groups: int) -> Tensor:
    """out [G, P, Q] fp32 with out[g] = x[k0:k0+len]^T @ y[k0:k0+len] (x [M, P], y [M, Q])."""
    P, Q = x.shape[1], y.shape[1]
    out = torch.empty(groups, P, Q, device=x.device, dtype=torch.float32)
    if out.numel() == 0:
        return out
    grid = (triton.cdiv(P, 128) * triton.cdiv(Q, 128), groups)
    _gmm_k_kernel[grid](x, y, out, segs, P, Q, x.stride(0), x.stride(1), y.stride(0), y.stride(1), ACCUM=False,
                        PRECISION=_precision(x.dtype), BP=128, BQ=128, BK=32, num_warps=8, num_stages=3)
    return out


if triton is not None:
    _gmm_m_op = torch.library.custom_op("nvfp4::bf16_gmm_m", _gmm_m_impl, mutates_args=())
    _gmm_k_op = torch.library.custom_op("nvfp4::bf16_gmm_k", _gmm_k_impl, mutates_args=())

    @_gmm_m_op.register_fake
    def _(x, w, tiles, transpose, bias, tile_rows):
        return x.new_empty((x.shape[0], w.shape[1] if transpose else w.shape[2]))

    @_gmm_k_op.register_fake
    def _(x, y, segs, groups):
        return x.new_empty((groups, x.shape[1], y.shape[1]), dtype=torch.float32)


def gmm_m(x, w, tiles, transpose, bias=None, tile_rows=None):
    if torch.compiler.is_compiling():
        return _gmm_m_op(x, w, tiles, transpose, bias, tile_rows)
    return _gmm_m_impl(x, w, tiles, transpose, bias, tile_rows)


def gmm_k(x, y, segs, groups):
    if torch.compiler.is_compiling():
        return _gmm_k_op(x, y, segs, groups)
    return _gmm_k_impl(x, y, segs, groups)


def expert_sums(partial: Tensor, tile_group: Tensor, groups: int) -> Tensor:
    """Per-group sums of per-tile partial sums (fp32 [tiles, C], tile i in group tile_group[i], -1: none):
    fp32 [groups, C]. A one-hot matrix product in fp64, so the order of the additions does not matter at fp32
    precision and nothing is read on the host."""
    onehot = (tile_group.long()[None, :] == torch.arange(groups, device=partial.device)[:, None]).double()
    return (onehot @ partial.double()).float()


class GroupedLinear(torch.autograd.Function):
    """y = x[rows of tile t] @ w[g]^T + b[g] (``w`` [G, N, K], ``b`` [G, N] or None) on the MoE buffer, with
    the grouped backward: dx (ragged M), dw (ragged K over ``segs``), db (per-group row sums)."""

    @staticmethod
    def forward(ctx, x, w, b, tiles, tile_rows, segs):
        ctx.save_for_backward(x, w, tiles, tile_rows, segs)
        ctx.has_bias = b is not None
        return gmm_m(x, w, tiles, True, b, tile_rows)

    @staticmethod
    def backward(ctx, dy):
        x, w, tiles, tile_rows, segs = ctx.saved_tensors
        dy = dy.contiguous()
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = gmm_m(dy, w, tiles, False)
        if ctx.needs_input_grad[1]:
            dw = gmm_k(dy, x, segs, w.shape[0]).to(w.dtype)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            g = dy.float().view(-1, ROWS, dy.shape[1])
            if tile_rows is not None:  # the rows that got the bias
                g = g * (torch.arange(ROWS, device=g.device)[None, :] < tile_rows[:, None]).unsqueeze(-1)
            db = expert_sums(g.sum(1), tiles, w.shape[0]).to(w.dtype)
        return dx, dw, db, None, None, None


__all__ = ["available", "gmm_m", "gmm_k", "expert_sums", "GroupedLinear"]
