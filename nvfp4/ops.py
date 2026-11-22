"""Thin Python API over the CUDA kernels. Each op has a pure-PyTorch twin in ``nvfp4.reference``."""
from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from nvfp4 import _C  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CPU-only installs
    _C = None


def available() -> bool:
    return _C is not None and torch.cuda.is_available() and torch.cuda.get_device_capability() >= (12, 0)


@dataclass
class NVFP4Tensor:
    """A 2D tensor quantized along its last dim: value = e2m1(data) * e4m3(scales) * gdec."""

    data: torch.Tensor    # uint8 [R, C/2]
    scales: torch.Tensor  # uint8 [R, C/16]
    gdec: torch.Tensor    # fp32 [1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[0], self.data.shape[1] * 2

    def as_tuple(self):
        return self.data, self.scales, self.gdec.reshape(())

    def as_storage(self) -> list[torch.Tensor]:
        """[data, scales, gdec], e.g. for ctx.save_for_backward; NVFP4Tensor(*storage) rebuilds it."""
        return [self.data, self.scales, self.gdec]

    def dequantize(self) -> torch.Tensor:
        from nvfp4 import reference
        return reference.dequantize(self.data, self.scales, self.gdec.reshape(()))


def quantize(x: torch.Tensor, *, transpose: bool = False, block_2d: bool = False, stochastic: bool = False,
             sr_seed: int = 0, rht_seed: int | None = None) -> NVFP4Tensor:
    """Quantize ``x`` (or ``x.T`` if ``transpose``) along the last dim. ``rht_seed`` enables the 16-point RHT."""
    if _C is None:
        raise RuntimeError("nvfp4 CUDA extension not built")
    if not x.is_contiguous():
        x = x.contiguous()
    d, s, g = _C.quantize(x, transpose, block_2d, stochastic, int(sr_seed), rht_seed is not None,
                          int(rht_seed or 0))
    return NVFP4Tensor(d, s, g)


def quantize_dual(x: torch.Tensor, *, row: bool = True, col: bool = True, block_2d: bool = False,
                  row_sr_seed: int | None = None, col_sr_seed: int | None = None, rht_seed: int | None = None,
                  col_pad: int = 16) -> tuple[NVFP4Tensor | None, NVFP4Tensor | None]:
    """One read of ``x`` [R, C] -> (``x`` quantized along C, ``x.T`` quantized along R).

    The second output is padded with zero rows of ``x`` to a multiple of ``col_pad`` and optionally
    random-Hadamard transformed (``rht_seed``). ``*_sr_seed`` enables stochastic rounding for that
    output. With ``block_2d`` both use 16x16 tiles, so the col output is the exact transpose of the row
    output. Bit-identical to ``reference.quantize_dual``.
    """
    if _C is None:
        raise RuntimeError("nvfp4 CUDA extension not built")
    if not x.is_contiguous() or x.data_ptr() % 16:
        x = x.contiguous() if not x.is_contiguous() else x.clone()
    rd, rs, rg, cd, cs, cg = _C.quantize_dual(
        x, row, col, block_2d, row_sr_seed is not None, int(row_sr_seed or 0), col_sr_seed is not None,
        int(col_sr_seed or 0), rht_seed is not None, int(rht_seed or 0), int(col_pad))
    return (NVFP4Tensor(rd, rs, rg) if row else None, NVFP4Tensor(cd, cs, cg) if col else None)


def _aligned(t: torch.Tensor) -> torch.Tensor:
    if not t.is_contiguous():
        t = t.contiguous()
    return t if t.data_ptr() % 16 == 0 else t.clone()


def _dual_flags(row, col, row_sr_seed, col_sr_seed, rht_seed, col_pad):
    return (row, col, row_sr_seed is not None, int(row_sr_seed or 0), col_sr_seed is not None, int(col_sr_seed or 0),
            rht_seed is not None, int(rht_seed or 0), int(col_pad))


def _pair(outs, row, col):
    rd, rs, rg, cd, cs, cg = outs
    return (NVFP4Tensor(rd, rs, rg) if row else None, NVFP4Tensor(cd, cs, cg) if col else None)


# Producer-fused variants of quantize_dual: the op that produces the quantized tensor runs inside the
# quantizer, so its output is never written to memory. Same keyword arguments as quantize_dual
# (without block_2d); each is bit-identical to its reference.* twin.

def layernorm_quantize_dual(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float, *, row: bool = True,
                            col: bool = True, row_sr_seed: int | None = None, col_sr_seed: int | None = None,
                            rht_seed: int | None = None, col_pad: int = 16):
    """Quantize ``LayerNorm(x)`` [R, C] (fp32 weight / bias [C]) without writing it.

    Returns ``(mean, rstd, row, col)``: the per-row statistics ([R] fp32, what the LayerNorm backward
    needs) and the two quantize_dual outputs of ``((x - mean) * rstd) * weight + bias`` in fp32.
    """
    if _C is None:
        raise RuntimeError("nvfp4 CUDA extension not built")
    outs = _C.layernorm_quantize_dual(_aligned(x), weight.float().contiguous(), bias.float().contiguous(), float(eps),
                                      *_dual_flags(row, col, row_sr_seed, col_sr_seed, rht_seed, col_pad))
    return (outs[0], outs[1], *_pair(outs[2:], row, col))


def gelu_quantize_dual(h: torch.Tensor, *, row: bool = True, col: bool = True, row_sr_seed: int | None = None,
                       col_sr_seed: int | None = None, rht_seed: int | None = None, col_pad: int = 16):
    """Quantize ``gelu(h)`` (tanh approximation, bf16 ``h``) without writing it."""
    if _C is None:
        raise RuntimeError("nvfp4 CUDA extension not built")
    outs = _C.gelu_quantize_dual(_aligned(h.to(torch.bfloat16)),
                                 *_dual_flags(row, col, row_sr_seed, col_sr_seed, rht_seed, col_pad))
    return _pair(outs, row, col)


def gelu_backward_quantize_dual(dy: torch.Tensor, h: torch.Tensor, *, row: bool = True, col: bool = True,
                                row_sr_seed: int | None = None, col_sr_seed: int | None = None,
                                rht_seed: int | None = None, col_pad: int = 16):
    """Quantize the GELU input gradient ``dy * gelu'(h)`` (bf16 ``dy``, ``h``) without writing it."""
    if _C is None:
        raise RuntimeError("nvfp4 CUDA extension not built")
    outs = _C.gelu_backward_quantize_dual(_aligned(dy.to(torch.bfloat16)), _aligned(h.to(torch.bfloat16)),
                                          *_dual_flags(row, col, row_sr_seed, col_sr_seed, rht_seed, col_pad))
    return _pair(outs, row, col)


def gemm(a: NVFP4Tensor, b: NVFP4Tensor, out_dtype: torch.dtype = torch.bfloat16, *,
         out: torch.Tensor | None = None, accumulate: bool = False, impl: str = "auto",
         splits: int = 0) -> torch.Tensor:
    """D[M, N] = A[M, K] @ B[N, K]^T with both operands in NVFP4 (block-scaled tensor-core MMA).

    ``out`` writes into an existing contiguous [M, N] tensor; ``accumulate=True`` adds to it instead
    (fp32 only), e.g. to accumulate a weight gradient in place across micro-batches.
    ``impl``: "auto" (TMA main loop when K % 256 == 0, else cp.async), "tma" or "cp.async".
    ``splits``: number of K splits (0 = chosen from the grid size); the split-K reduction is
    deterministic but sums in a different order than an unsplit GEMM.
    """
    if _C is None:
        raise RuntimeError("nvfp4 CUDA extension not built")
    if out is not None:
        out_dtype = out.dtype
    impl_id = {"auto": 0, "cp.async": 1, "tma": 2}[impl]
    return _C.gemm(a.data, a.scales, a.gdec, b.data, b.scales, b.gdec, out_dtype, out, accumulate, impl_id, splits)
