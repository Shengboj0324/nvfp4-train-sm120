"""torch.library registrations of the CUDA kernels used inside the autograd Functions.

Eager calls go straight to the pybind11 extension (no dispatcher overhead). Under torch.compile the
same functions are reached through ``torch.ops.nvfp4.*`` custom ops with fake (meta) implementations,
so Dynamo traces NVFP4 layers into the graph instead of breaking the graph at each one.

(This module deliberately has no ``from __future__ import annotations``: torch.library infers the
operator schemas from the annotations.)
"""
import os
from typing import List, Optional, Tuple

import torch
from torch import Tensor

try:
    from nvfp4 import _C  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CPU-only installs
    _C = None

_DUAL = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
_SUPPORTED_CAPABILITY = (12, 0)
_device_ok: dict = {}

PTX_JIT_MESSAGE = (
    "CUDA_FORCE_PTX_JIT is set in the environment: the driver then ignores compiled GPU code and JIT-compiles "
    "PTX instead, but this extension (sm_120a arch-specific code) and PyTorch's own wheels ship no PTX for "
    "sm_120, so every kernel launch fails with 'no kernel image is available for execution on the device'. "
    "Unset CUDA_FORCE_PTX_JIT (it is sometimes left behind by old TensorFlow / CUDA setup guides).")


def ptx_jit_forced() -> bool:
    return os.environ.get("CUDA_FORCE_PTX_JIT", "0").strip() not in ("", "0")


def require(t: Tensor) -> None:
    """Fail early with a clear message instead of a CUDA launch error deep inside a kernel."""
    if _C is None:
        raise RuntimeError("nvfp4 CUDA extension not built (pip install -e . on a machine with CUDA >= 12.8); "
                           "use NVFP4Linear(..., backend='emulate') to run the recipe without it")
    if not t.is_cuda:
        raise RuntimeError(f"nvfp4 CUDA ops need CUDA tensors, got a tensor on {t.device}")
    if ptx_jit_forced():
        raise RuntimeError(PTX_JIT_MESSAGE)
    idx = t.device.index if t.device.index is not None else torch.cuda.current_device()
    ok = _device_ok.get(idx)
    if ok is None:
        ok = _device_ok[idx] = torch.cuda.get_device_capability(idx) == _SUPPORTED_CAPABILITY
    if not ok:
        cap = torch.cuda.get_device_capability(idx)
        raise RuntimeError(f"nvfp4 kernels run on compute capability {_SUPPORTED_CAPABILITY} (RTX 50-series) only; "
                           f"cuda:{idx} is {cap}. Use NVFP4Linear(..., backend='emulate') elsewhere")


def aligned(t: Tensor) -> Tensor:
    """Contiguous and 16-byte aligned (the kernels use 16-byte vector loads)."""
    if not t.is_contiguous():
        t = t.contiguous()
    return t if t.data_ptr() % 16 == 0 else t.clone()


# --- implementations (eager) -----------------------------------------------------------------------

def _quantize_dual_impl(x: Tensor, do_row: bool, do_col: bool, block_2d: bool, row_sr: bool, row_seed: int,
                        col_sr: bool, col_seed: int, col_rht: bool, rht_seed: int, col_pad: int,
                        row_seed_t: Optional[Tensor], col_seed_t: Optional[Tensor],
                        rht_seed_t: Optional[Tensor]) -> _DUAL:
    require(x)
    return tuple(_C.quantize_dual(aligned(x), do_row, do_col, block_2d, row_sr, row_seed, col_sr, col_seed, col_rht,
                                  rht_seed, col_pad, row_seed_t, col_seed_t, rht_seed_t))


def _quantize_dual_parts_impl(parts: List[Tensor], do_row: bool, do_col: bool, row_sr: bool, row_seed: int,
                              col_sr: bool, col_seed: int, col_rht: bool, rht_seed: int, col_pad: int,
                              row_seed_t: Optional[Tensor], col_seed_t: Optional[Tensor],
                              rht_seed_t: Optional[Tensor], col_sums: bool) -> List[Tensor]:
    require(parts[0])
    return list(_C.quantize_dual_parts([aligned(p) for p in parts], do_row, do_col, row_sr, row_seed, col_sr,
                                       col_seed, col_rht, rht_seed, col_pad, row_seed_t, col_seed_t, rht_seed_t,
                                       col_sums))


def _layernorm_quantize_dual_impl(x: Tensor, weight: Tensor, bias: Tensor, eps: float, do_row: bool, do_col: bool,
                                  row_sr: bool, row_seed: int, col_sr: bool, col_seed: int, col_rht: bool,
                                  rht_seed: int, col_pad: int, row_seed_t: Optional[Tensor],
                                  col_seed_t: Optional[Tensor], rht_seed_t: Optional[Tensor]
                                  ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    require(x)
    return tuple(_C.layernorm_quantize_dual(aligned(x), weight.float().contiguous(), bias.float().contiguous(), eps,
                                            do_row, do_col, row_sr, row_seed, col_sr, col_seed, col_rht, rht_seed,
                                            col_pad, row_seed_t, col_seed_t, rht_seed_t))


def _bias(b: Optional[Tensor]) -> Optional[Tensor]:
    return None if b is None else b.float().contiguous()


def _gelu_quantize_dual_impl(h: Tensor, do_row: bool, do_col: bool, row_sr: bool, row_seed: int, col_sr: bool,
                             col_seed: int, col_rht: bool, rht_seed: int, col_pad: int,
                             row_seed_t: Optional[Tensor], col_seed_t: Optional[Tensor],
                             rht_seed_t: Optional[Tensor], bias: Optional[Tensor], exact: bool) -> _DUAL:
    require(h)
    return tuple(_C.gelu_quantize_dual(aligned(h.to(torch.bfloat16)), do_row, do_col, row_sr, row_seed, col_sr,
                                       col_seed, col_rht, rht_seed, col_pad, row_seed_t, col_seed_t, rht_seed_t,
                                       _bias(bias), exact))


def _gelu_backward_quantize_dual_impl(dy: Tensor, h: Tensor, do_row: bool, do_col: bool, row_sr: bool, row_seed: int,
                                      col_sr: bool, col_seed: int, col_rht: bool, rht_seed: int, col_pad: int,
                                      row_seed_t: Optional[Tensor], col_seed_t: Optional[Tensor],
                                      rht_seed_t: Optional[Tensor], bias: Optional[Tensor], exact: bool,
                                      col_sums: bool) -> List[Tensor]:
    require(dy)
    return list(_C.gelu_backward_quantize_dual(aligned(dy.to(torch.bfloat16)), aligned(h.to(torch.bfloat16)),
                                               do_row, do_col, row_sr, row_seed, col_sr, col_seed, col_rht, rht_seed,
                                               col_pad, row_seed_t, col_seed_t, rht_seed_t, _bias(bias), exact,
                                               col_sums))


def _rmsnorm_quantize_dual_impl(x: Tensor, weight: Tensor, eps: float, do_row: bool, do_col: bool, row_sr: bool,
                                row_seed: int, col_sr: bool, col_seed: int, col_rht: bool, rht_seed: int, col_pad: int,
                                row_seed_t: Optional[Tensor], col_seed_t: Optional[Tensor],
                                rht_seed_t: Optional[Tensor]
                                ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    require(x)
    return tuple(_C.rmsnorm_quantize_dual(aligned(x), weight.float().contiguous(), eps, do_row, do_col, row_sr,
                                          row_seed, col_sr, col_seed, col_rht, rht_seed, col_pad, row_seed_t,
                                          col_seed_t, rht_seed_t))


def _swiglu_quantize_dual_impl(g: Tensor, u: Tensor, do_row: bool, do_col: bool, row_sr: bool, row_seed: int,
                               col_sr: bool, col_seed: int, col_rht: bool, rht_seed: int, col_pad: int,
                               row_seed_t: Optional[Tensor], col_seed_t: Optional[Tensor],
                               rht_seed_t: Optional[Tensor], bias_g: Optional[Tensor],
                               bias_u: Optional[Tensor]) -> _DUAL:
    require(g)
    return tuple(_C.swiglu_quantize_dual(aligned(g.to(torch.bfloat16)), aligned(u.to(torch.bfloat16)), do_row, do_col,
                                         row_sr, row_seed, col_sr, col_seed, col_rht, rht_seed, col_pad, row_seed_t,
                                         col_seed_t, rht_seed_t, _bias(bias_g), _bias(bias_u)))


def _swiglu_backward_quantize_dual_impl(dy: Tensor, g: Tensor, u: Tensor, gate: bool, do_row: bool, do_col: bool,
                                        row_sr: bool, row_seed: int, col_sr: bool, col_seed: int, col_rht: bool,
                                        rht_seed: int, col_pad: int, row_seed_t: Optional[Tensor],
                                        col_seed_t: Optional[Tensor], rht_seed_t: Optional[Tensor],
                                        bias_g: Optional[Tensor], bias_u: Optional[Tensor],
                                        col_sums: bool) -> List[Tensor]:
    require(dy)
    return list(_C.swiglu_backward_quantize_dual(aligned(dy.to(torch.bfloat16)), aligned(g.to(torch.bfloat16)),
                                                 aligned(u.to(torch.bfloat16)), gate, do_row, do_col, row_sr,
                                                 row_seed, col_sr, col_seed, col_rht, rht_seed, col_pad, row_seed_t,
                                                 col_seed_t, rht_seed_t, _bias(bias_g), _bias(bias_u), col_sums))


def _norm_backward_impl(dy: Tensor, x: Tensor, mean: Optional[Tensor], rstd: Tensor, w: Tensor,
                        dres: Optional[Tensor], need_dx: bool, need_dw: bool, need_db: bool) -> List[Tensor]:
    require(x)
    return list(_C.norm_backward(aligned(dy.float()), aligned(x), mean, rstd, w.float().contiguous(),
                                 None if dres is None else aligned(dres.to(x.dtype)), need_dx, need_dw, need_db))


def _gemm_impl(a: Tensor, sfa: Tensor, gdec_a: Tensor, b: Tensor, sfb: Tensor, gdec_b: Tensor, out_dtype: torch.dtype,
               impl: int, splits: int, bias: Optional[Tensor] = None, residual: Optional[Tensor] = None,
               bias_round: bool = False) -> Tensor:
    require(a)
    return _C.gemm(a, sfa, gdec_a, b, sfb, gdec_b, out_dtype, None, False, impl, splits, bias, residual, bias_round)


def _gemm_out_impl(a: Tensor, sfa: Tensor, gdec_a: Tensor, b: Tensor, sfb: Tensor, gdec_b: Tensor, out: Tensor,
                   accumulate: bool, impl: int, splits: int) -> None:
    require(a)
    _C.gemm(a, sfa, gdec_a, b, sfb, gdec_b, out.dtype, out, accumulate, impl, splits)


def _attn_quant_impl(x: Tensor, mean: Optional[Tensor], o: Optional[Tensor], want_row: bool,
                     want_col: bool) -> List[Tensor]:
    require(x)
    return list(_C.attn_quant(x, mean, o, want_row, want_col))


def _attn_fwd_impl(q: Tensor, qs: Tensor, k: Tensor, ks: Tensor, vt: Tensor, vts: Tensor, N: int, causal: bool,
                   scale: float, dtype: torch.dtype) -> List[Tensor]:
    require(q)
    return list(_C.attn_fwd(q, qs, k, ks, vt, vts, N, causal, scale, dtype))


def _attn_bwd_impl(q: Tensor, qs: Tensor, qt: Tensor, qts: Tensor, k: Tensor, ks: Tensor, kt: Tensor, kts: Tensor,
                   v: Tensor, vs: Tensor, g: Tensor, gs: Tensor, gt: Tensor, gts: Tensor, lse: Tensor, delta: Tensor,
                   N: int, causal: bool, scale: float, dtype: torch.dtype, dq_stride: List[int],
                   dk_stride: List[int]) -> List[Tensor]:
    require(q)
    return list(_C.attn_bwd(q, qs, qt, qts, k, ks, kt, kts, v, vs, g, gs, gt, gts, lse, delta, N, causal, scale,
                            dtype, dq_stride, dk_stride))


# --- custom ops (torch.compile) ----------------------------------------------------------------------

_quantize_dual_op = torch.library.custom_op("nvfp4::quantize_dual", _quantize_dual_impl, mutates_args=())
_quantize_dual_parts_op = torch.library.custom_op("nvfp4::quantize_dual_parts", _quantize_dual_parts_impl,
                                                  mutates_args=())
_layernorm_quantize_dual_op = torch.library.custom_op("nvfp4::layernorm_quantize_dual", _layernorm_quantize_dual_impl,
                                                      mutates_args=())
_gelu_quantize_dual_op = torch.library.custom_op("nvfp4::gelu_quantize_dual", _gelu_quantize_dual_impl,
                                                 mutates_args=())
_gelu_backward_quantize_dual_op = torch.library.custom_op("nvfp4::gelu_backward_quantize_dual",
                                                          _gelu_backward_quantize_dual_impl, mutates_args=())
_rmsnorm_quantize_dual_op = torch.library.custom_op("nvfp4::rmsnorm_quantize_dual", _rmsnorm_quantize_dual_impl,
                                                    mutates_args=())
_swiglu_quantize_dual_op = torch.library.custom_op("nvfp4::swiglu_quantize_dual", _swiglu_quantize_dual_impl,
                                                   mutates_args=())
_swiglu_backward_quantize_dual_op = torch.library.custom_op("nvfp4::swiglu_backward_quantize_dual",
                                                            _swiglu_backward_quantize_dual_impl, mutates_args=())
_norm_backward_op = torch.library.custom_op("nvfp4::norm_backward", _norm_backward_impl, mutates_args=())
_gemm_op = torch.library.custom_op("nvfp4::gemm", _gemm_impl, mutates_args=())
_gemm_out_op = torch.library.custom_op("nvfp4::gemm_out", _gemm_out_impl, mutates_args=("out",))
_attn_quant_op = torch.library.custom_op("nvfp4::attn_quant", _attn_quant_impl, mutates_args=())
_attn_fwd_op = torch.library.custom_op("nvfp4::attn_fwd", _attn_fwd_impl, mutates_args=())
_attn_bwd_op = torch.library.custom_op("nvfp4::attn_bwd", _attn_bwd_impl, mutates_args=())


def _dual_fake(x: Tensor, do_row: bool, do_col: bool, col_pad: int) -> _DUAL:
    R, C = x.shape
    r_pad = (R + col_pad - 1) // col_pad * col_pad
    u8, f32 = torch.uint8, torch.float32
    return (x.new_empty((R if do_row else 0, C // 2), dtype=u8), x.new_empty((R if do_row else 0, C // 16), dtype=u8),
            x.new_empty((1 if do_row else 0,), dtype=f32), x.new_empty((C if do_col else 0, r_pad // 2), dtype=u8),
            x.new_empty((C if do_col else 0, r_pad // 16), dtype=u8), x.new_empty((1 if do_col else 0,), dtype=f32))


@_quantize_dual_op.register_fake
def _(x, do_row, do_col, block_2d, row_sr, row_seed, col_sr, col_seed, col_rht, rht_seed, col_pad, *seed_tensors):
    return _dual_fake(x, do_row, do_col, col_pad)


def _colsum_fake(x: Tensor, rows: int, do_col: bool, col_pad: int, C: int) -> Tensor:
    rows = (rows + col_pad - 1) // col_pad * col_pad if do_col else rows
    return x.new_empty((max(1, (rows + 63) // 64), C), dtype=torch.float32)


@_quantize_dual_parts_op.register_fake
def _(parts, do_row, do_col, row_sr, row_seed, col_sr, col_seed, col_rht, rht_seed, col_pad, row_seed_t, col_seed_t,
      rht_seed_t, col_sums):
    R, Cp = parts[0].shape
    C = Cp * len(parts)
    outs = list(_dual_fake(parts[0].new_empty((R, C)), do_row, do_col, col_pad))
    if col_sums:
        outs.append(_colsum_fake(parts[0], R, do_col, col_pad, C))
    return outs


@_layernorm_quantize_dual_op.register_fake
def _(x, weight, bias, eps, do_row, do_col, row_sr, row_seed, col_sr, col_seed, col_rht, rht_seed, col_pad,
      *seed_tensors):
    stats = x.new_empty((x.shape[0],), dtype=torch.float32)
    return (stats, torch.empty_like(stats), *_dual_fake(x, do_row, do_col, col_pad))


@_gelu_quantize_dual_op.register_fake
def _(h, do_row, do_col, row_sr, row_seed, col_sr, col_seed, col_rht, rht_seed, col_pad, row_seed_t, col_seed_t,
      rht_seed_t, bias, exact):
    return _dual_fake(h, do_row, do_col, col_pad)


@_gelu_backward_quantize_dual_op.register_fake
def _(dy, h, do_row, do_col, row_sr, row_seed, col_sr, col_seed, col_rht, rht_seed, col_pad, row_seed_t, col_seed_t,
      rht_seed_t, bias, exact, col_sums):
    outs = list(_dual_fake(dy, do_row, do_col, col_pad))
    if col_sums:
        outs.append(_colsum_fake(dy, dy.shape[0], do_col, col_pad, dy.shape[1]))
    return outs


@_rmsnorm_quantize_dual_op.register_fake
def _(x, weight, eps, do_row, do_col, row_sr, row_seed, col_sr, col_seed, col_rht, rht_seed, col_pad, *seed_tensors):
    return (x.new_empty((x.shape[0],), dtype=torch.float32), *_dual_fake(x, do_row, do_col, col_pad))


@_swiglu_quantize_dual_op.register_fake
def _(g, u, do_row, do_col, row_sr, row_seed, col_sr, col_seed, col_rht, rht_seed, col_pad, row_seed_t, col_seed_t,
      rht_seed_t, bias_g, bias_u):
    return _dual_fake(g, do_row, do_col, col_pad)


@_swiglu_backward_quantize_dual_op.register_fake
def _(dy, g, u, gate, do_row, do_col, row_sr, row_seed, col_sr, col_seed, col_rht, rht_seed, col_pad, row_seed_t,
      col_seed_t, rht_seed_t, bias_g, bias_u, col_sums):
    outs = list(_dual_fake(dy, do_row, do_col, col_pad))
    if col_sums:
        outs.append(_colsum_fake(dy, dy.shape[0], do_col, col_pad, dy.shape[1]))
    return outs


@_norm_backward_op.register_fake
def _(dy, x, mean, rstd, w, dres, need_dx, need_dw, need_db):
    R, C = x.shape
    return [x.new_empty((R, C) if need_dx else (0, 0)), x.new_empty((C,) if need_dw else (0, C), dtype=torch.float32),
            x.new_empty((C,) if need_db else (0, C), dtype=torch.float32)]


@_gemm_op.register_fake
def _(a, sfa, gdec_a, b, sfb, gdec_b, out_dtype, impl, splits, bias=None, residual=None, bias_round=False):
    return a.new_empty((a.shape[0], b.shape[0]), dtype=out_dtype)


@_gemm_out_op.register_fake
def _(a, sfa, gdec_a, b, sfb, gdec_b, out, accumulate, impl, splits):
    return None


def _pad128(n: int) -> int:
    return (n + 127) // 128 * 128


@_attn_quant_op.register_fake
def _(x, mean, o, want_row, want_col):
    B, H, N, D = x.shape
    Np, u8 = _pad128(N), torch.uint8
    e = x.new_empty((0,), dtype=u8)
    return [x.new_empty((B, H, Np, D), dtype=u8) if want_row else e,
            x.new_empty((B, H, Np) if want_row else (0,), dtype=torch.float32),
            x.new_empty((B, H, D, Np), dtype=u8) if want_col else e,
            x.new_empty((B, H, Np // 64, 2 * D), dtype=u8) if want_col else e,
            x.new_empty((B, H, Np) if o is not None else (0,), dtype=torch.float32)]


@_attn_fwd_op.register_fake
def _(q, qs, k, ks, vt, vts, N, causal, scale, dtype):
    B, H, Np, D = q.shape
    return [q.new_empty((B, N, H, D), dtype=dtype), q.new_empty((B, H, Np), dtype=torch.float32)]


def _dense_strides(size: List[int], stride: List[int]) -> bool:
    """True if (size, stride) is a non-overlapping layout without gaps (any dim order), as C++'s
    is_non_overlapping_and_dense (not exposed in Python)."""
    expect = 1
    for n, st in sorted(((n, st) for n, st in zip(size, stride) if n != 1), key=lambda x: x[1]):
        if st != expect:
            return False
        expect *= n
    return True


def _attn_grad_fake(q: Tensor, B: int, H: int, N: int, D: int, stride: List[int], dtype: torch.dtype) -> Tensor:
    if len(stride) == 4 and stride[3] == 1 and _dense_strides([B, H, N, D], stride):
        return torch.empty_strided((B, H, N, D), stride, dtype=dtype, device=q.device)
    return q.new_empty((B, N, H, D), dtype=dtype).transpose(1, 2)


@_attn_bwd_op.register_fake
def _(q, qs, qt, qts, k, ks, kt, kts, v, vs, g, gs, gt, gts, lse, delta, N, causal, scale, dtype, dq_stride, dk_stride):
    B, H, Np, D = q.shape
    Hk = k.shape[1]
    return [_attn_grad_fake(q, B, H, N, D, dq_stride, dtype), _attn_grad_fake(q, B, Hk, N, D, dk_stride, dtype),
            _attn_grad_fake(q, B, Hk, N, D, dk_stride, dtype)]


# --- dispatch: plain call in eager, custom op while compiling -------------------------------------------

def _pick(impl, op):
    def call(*args):
        if torch.compiler.is_compiling():
            return op(*args)
        return impl(*args)
    call.__name__ = impl.__name__.removesuffix("_impl").lstrip("_")
    return call


quantize_dual = _pick(_quantize_dual_impl, _quantize_dual_op)
quantize_dual_parts = _pick(_quantize_dual_parts_impl, _quantize_dual_parts_op)
norm_backward = _pick(_norm_backward_impl, _norm_backward_op)
layernorm_quantize_dual = _pick(_layernorm_quantize_dual_impl, _layernorm_quantize_dual_op)
gelu_quantize_dual = _pick(_gelu_quantize_dual_impl, _gelu_quantize_dual_op)
gelu_backward_quantize_dual = _pick(_gelu_backward_quantize_dual_impl, _gelu_backward_quantize_dual_op)
rmsnorm_quantize_dual = _pick(_rmsnorm_quantize_dual_impl, _rmsnorm_quantize_dual_op)
swiglu_quantize_dual = _pick(_swiglu_quantize_dual_impl, _swiglu_quantize_dual_op)
swiglu_backward_quantize_dual = _pick(_swiglu_backward_quantize_dual_impl, _swiglu_backward_quantize_dual_op)
gemm = _pick(_gemm_impl, _gemm_op)
gemm_out = _pick(_gemm_out_impl, _gemm_out_op)
attn_quant = _pick(_attn_quant_impl, _attn_quant_op)
attn_fwd = _pick(_attn_fwd_impl, _attn_fwd_op)
attn_bwd = _pick(_attn_bwd_impl, _attn_bwd_op)
