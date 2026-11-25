"""Thin Python API over the CUDA kernels. Each op has a pure-PyTorch twin in ``nvfp4.reference``."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Union

import torch

from nvfp4 import _library as _lib
from nvfp4._library import _C, require as _require  # noqa: F401  (re-exported for callers and tests)

_DEFAULT_SPLITS = int(os.environ.get("NVFP4_GEMM_SPLITS", "0"))  # 1 disables split-K (ablations)


# The kernels are compiled for the arch-specific target sm_120a, whose code runs on compute
# capability 12.0 only (RTX 50-series); 'a' targets are not forward compatible.
SUPPORTED_CAPABILITY = _lib._SUPPORTED_CAPABILITY

# A seed is an int, or a one-element int64 tensor on the GPU (read by the kernel: no host sync, and
# CUDA graphs / torch.compile see a new value on every replay / call), or None (feature off).
Seed = Union[int, torch.Tensor, None]


def available(device: int | torch.device | None = None) -> bool:
    """True if the extension is built and ``device`` (default: current) is an sm_120 GPU (and the
    environment doesn't force PTX JIT compilation, which makes the kernels unlaunchable; see
    ``nvfp4._library.PTX_JIT_MESSAGE``)."""
    if _C is None or _lib.ptx_jit_forced() or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device) == SUPPORTED_CAPABILITY


_FOUR_SIX = [False]  # mirrors the extension's switch (also used without the extension, by the emulation)


def set_four_six(on: bool) -> None:
    """Recipe switch: "Four Over Six" adaptive block scales (Cook et al., arXiv 2512.02010) on every
    round-to-nearest NVFP4 output (activations, weights, their transposes; not the stochastically rounded
    gradients): each 16-element block keeps whichever of max -> 6 and max -> 4 gives the smaller squared error,
    with the tensor scale 6 * 256 / amax. Read when a quantizer is launched (so it also applies inside
    torch.compile'd and CUDA-graph-captured code captured after the call)."""
    _FOUR_SIX[0] = bool(on)
    if _C is not None:
        _C.set_four_six(bool(on))


def four_six() -> bool:
    """Whether Four Over Six block scales are on (``set_four_six``)."""
    return _FOUR_SIX[0]


_WEIGHT_BLOCKS = ["2d"]


def set_weight_blocks(mode: str) -> None:
    """Recipe switch for the NVFP4 layers' weights: "2d" (default; 16x16 blocks, so W and W^T are one
    quantization, NVIDIA's recipe) or "1d" (1x16 blocks along K for fprop, which 4/6 can improve, and for
    dgrad the dequantized fprop weight re-quantized along N; Quartet II / Mosaic NVFP4)."""
    if mode not in ("2d", "1d"):
        raise ValueError(f"weight blocks must be '2d' or '1d', got {mode!r}")
    _WEIGHT_BLOCKS[0] = mode


def weight_blocks() -> str:
    """The weight block mode (``set_weight_blocks``)."""
    return _WEIGHT_BLOCKS[0]


_HIGH_PRECISION = [False]


def set_high_precision(on: bool) -> None:
    """Precision switch for the NVFP4 layers (NVIDIA, arXiv 2509.25149 §4.4: training the last ~10-20% of
    tokens in BF16 recovers most of NVFP4's loss gap). While on, NVFP4Linear and the nvfp4.fused sub-blocks
    compute exactly what the modules they replace would (nn.Linear / LayerNorm / activation in the caller's
    dtype, autocast respected), with ordinary autograd; their weights, optimizer state and module structure
    are unchanged, so it can be flipped at any step (and back). The 8-bit attention is not affected."""
    _HIGH_PRECISION[0] = bool(on)


def high_precision() -> bool:
    """Whether the NVFP4 layers currently compute in high precision (``set_high_precision``)."""
    return _HIGH_PRECISION[0]


_MS_EDEN = [False]
_MS_EDEN_DGRAD = [False]
ROW_RHT_SALT = 0x1E3779B97F4A7C15  # reference.ROW_RHT_SALT, csrc kRowRhtSalt


def set_ms_eden(on: bool, *, dgrad: bool = False) -> None:
    """Recipe switch: MS-EDEN for the weight-gradient GEMM's operands (Quartet II, arXiv 2601.22813). The dual
    quantizer's RHT outputs (X^T and dY^T, transformed along the tokens with one seed so the rotation cancels in
    dW = dY^T X) use a 64-point randomized Hadamard transform instead of 16-point, and the stochastically
    rounded one (dY^T) is quantized with round-to-nearest elements and an EDEN-corrected, stochastically
    rounded block scale instead of element-wise stochastic rounding: unbiased, ~half the error. Needs
    col_pad % 64 == 0 (NVFP4Linear uses 128).

    ``dgrad=True`` (needs ``on``) also applies it to the input-gradient GEMM dX = dY W: every stochastically
    rounded row output of the dual quantizer (dY, the GELU / SwiGLU input gradients: the dgrad A operands) is
    rotated along its row by a 64-point RHT whose signs come from its row seed (``row_rht_seed``) and quantized
    with EDEN instead of element SR, and the layers quantize the dgrad weight operand W^T from the
    dequantized forward weight under the same rotation (``rotated_weight_t``): (R dY)(R W^T)^T = dY W. Quartet
    II's backward; it replaces the 16x16-block transpose for dgrad by a second (rotated) weight quantization
    per backward."""
    if dgrad and not on:
        raise ValueError("MS-EDEN on dgrad needs MS-EDEN on (on=True)")
    _MS_EDEN[0], _MS_EDEN_DGRAD[0] = bool(on), bool(dgrad)
    if _C is not None:
        _C.set_ms_eden(bool(on))
        _C.set_ms_eden_dgrad(bool(dgrad))


def ms_eden() -> bool:
    """Whether MS-EDEN is on (``set_ms_eden``)."""
    return _MS_EDEN[0]


def ms_eden_dgrad() -> bool:
    """Whether MS-EDEN also covers the dgrad operands (``set_ms_eden(True, dgrad=True)``)."""
    return _MS_EDEN_DGRAD[0]


def row_rht_seed(row_sr_seed):
    """The RHT seed of a row output quantized with ``row_sr_seed`` under MS-EDEN dgrad (int or int64 tensor;
    no host sync for a tensor)."""
    return row_sr_seed ^ ROW_RHT_SALT


def rotated_weight_t(wq: NVFP4Tensor, row_sr_seed, groups: int = 1) -> NVFP4Tensor:
    """The dgrad weight operand under MS-EDEN dgrad: the dequantized forward weight wq [N, K] transposed and
    quantized along N (round-to-nearest) after the 64-point RHT that the row output with ``row_sr_seed``
    received, [K, N] (``groups`` > 1: a stacked [G N, K] weight -> [G K, N], see stack_transposed)."""
    _, wt = quantize_dual(wq.dequantize(), row=False, col=True, rht_seed=row_rht_seed(row_sr_seed), col_pad=128)
    return stack_transposed(wt, groups) if groups > 1 else wt


class four_six_scales:
    """Context manager: ``with ops.four_six_scales(True): ...`` sets the switch and restores it on exit."""

    def __init__(self, on: bool = True):
        self.on = bool(on)

    def __enter__(self):
        self.prev = four_six()
        set_four_six(self.on)
        return self

    def __exit__(self, *exc):
        set_four_six(self.prev)
        return False


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
    _require(x)
    if not x.is_contiguous():
        x = x.contiguous()
    d, s, g = _C.quantize(x, transpose, block_2d, stochastic, int(sr_seed), rht_seed is not None,
                          int(rht_seed or 0))
    return NVFP4Tensor(d, s, g)


def quantize_dual(x: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...], *, row: bool = True,
                  col: bool = True, block_2d: bool = False, row_sr_seed: Seed = None, col_sr_seed: Seed = None,
                  rht_seed: Seed = None, col_pad: int = 16, bias_grad: bool = False):
    """One read of ``x`` [R, C] -> (``x`` quantized along C, ``x.T`` quantized along R).

    The second output is padded with zero rows of ``x`` to a multiple of ``col_pad`` and optionally
    random-Hadamard transformed (``rht_seed``). ``*_sr_seed`` enables stochastic rounding for that
    output. With ``block_2d`` both use 16x16 tiles, so the col output is the exact transpose of the row
    output. Bit-identical to ``reference.quantize_dual`` (with the seeds' values).

    ``x`` may also be a list of 1-3 same-shaped tensors: the input is then their concatenation along
    dim 1, read in place (e.g. the q / k / v gradients of a fused qkv projection; not with ``block_2d``).
    ``bias_grad=True`` also returns the column sums of ``x`` (fp32 [C], the gradient of a bias added to the
    layer output ``x`` is the gradient of), from the same read: ``(row, col, bias_grad)``;
    ``bias_grad="tiles"`` returns the per-64-row-tile sums instead (see _colsums).
    """
    if isinstance(x, (list, tuple)) or bias_grad:
        parts = list(x) if isinstance(x, (list, tuple)) else [x]
        if block_2d:
            raise ValueError("block_2d is not supported for multi-part inputs / bias_grad")
        outs = _lib.quantize_dual_parts(parts, row, col, *_seed_flags(row_sr_seed, col_sr_seed, rht_seed),
                                        int(col_pad), *_seed_tensors(row_sr_seed, col_sr_seed, rht_seed),
                                        bool(bias_grad))
        pair = _pair(outs[:6], row, col)
        return (*pair, _colsums(outs[6], bias_grad)) if bias_grad else pair
    outs = _lib.quantize_dual(x, row, col, block_2d, *_seed_flags(row_sr_seed, col_sr_seed, rht_seed), int(col_pad),
                              *_seed_tensors(row_sr_seed, col_sr_seed, rht_seed))
    return _pair(outs, row, col)


def _colsums(partial: torch.Tensor, bias_grad) -> torch.Tensor:
    """The quantizers' column sums: summed over rows (``bias_grad=True``), or ``bias_grad="tiles"``: the
    per-tile partial sums, fp32 [ceil(R / 64), C] (tile i covers rows [64 i, 64 i + 64)), e.g. for the
    per-expert bias gradients of a mixture of experts."""
    if bias_grad == "tiles":
        return partial
    if bias_grad is not True:
        raise ValueError(f"bias_grad must be True, False or 'tiles', got {bias_grad!r}")
    return partial.sum(0)


def _seed_flags(row_sr_seed: Seed, col_sr_seed: Seed, rht_seed: Seed):
    """(row_sr, row_seed, col_sr, col_seed, col_rht, rht_seed) for the kernels' scalar arguments."""
    out = []
    for s in (row_sr_seed, col_sr_seed, rht_seed):
        out += [s is not None, 0 if s is None or isinstance(s, torch.Tensor) else int(s)]
    return out


def _seed_tensors(*seeds: Seed):
    return [s if isinstance(s, torch.Tensor) else None for s in seeds]


def _pair(outs, row, col):
    rd, rs, rg, cd, cs, cg = outs
    return (NVFP4Tensor(rd, rs, rg) if row else None, NVFP4Tensor(cd, cs, cg) if col else None)


# Producer-fused variants of quantize_dual: the op that produces the quantized tensor runs inside the
# quantizer, so its output is never written to memory. Same keyword arguments as quantize_dual
# (without block_2d); each is bit-identical to its reference.* twin.

def layernorm_quantize_dual(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float, *, row: bool = True,
                            col: bool = True, row_sr_seed: Seed = None, col_sr_seed: Seed = None,
                            rht_seed: Seed = None, col_pad: int = 16):
    """Quantize ``LayerNorm(x)`` [R, C] (fp32 weight / bias [C]) without writing it.

    Returns ``(mean, rstd, row, col)``: the per-row statistics ([R] fp32, what the LayerNorm backward
    needs) and the two quantize_dual outputs of ``((x - mean) * rstd) * weight + bias`` in fp32.
    """
    outs = _lib.layernorm_quantize_dual(x, weight, bias, float(eps), row, col,
                                        *_seed_flags(row_sr_seed, col_sr_seed, rht_seed), int(col_pad),
                                        *_seed_tensors(row_sr_seed, col_sr_seed, rht_seed))
    return (outs[0], outs[1], *_pair(outs[2:], row, col))


def _exact(approximate: str) -> bool:
    if approximate not in ("tanh", "none"):
        raise ValueError(f"approximate must be 'tanh' or 'none', got {approximate!r}")
    return approximate == "none"


def gelu_quantize_dual(h: torch.Tensor, *, bias: torch.Tensor | None = None, approximate: str = "tanh",
                       row: bool = True, col: bool = True, row_sr_seed: Seed = None, col_sr_seed: Seed = None,
                       rht_seed: Seed = None, col_pad: int = 16):
    """Quantize ``gelu(h + bias)`` (bf16 ``h``, optional per-column ``bias``; ``approximate`` as in
    ``torch.nn.functional.gelu``: "tanh" or "none" for the exact erf form) without writing it."""
    outs = _lib.gelu_quantize_dual(h, row, col, *_seed_flags(row_sr_seed, col_sr_seed, rht_seed), int(col_pad),
                                   *_seed_tensors(row_sr_seed, col_sr_seed, rht_seed), bias, _exact(approximate))
    return _pair(outs, row, col)


def gelu_backward_quantize_dual(dy: torch.Tensor, h: torch.Tensor, *, bias: torch.Tensor | None = None,
                                approximate: str = "tanh", bias_grad: bool = False, row: bool = True,
                                col: bool = True, row_sr_seed: Seed = None, col_sr_seed: Seed = None,
                                rht_seed: Seed = None, col_pad: int = 16):
    """Quantize the GELU input gradient ``dy * gelu'(h + bias)`` (bf16 ``dy``, ``h``) without writing it.

    With ``bias_grad=True`` also returns its sum over rows (fp32 [C], the gradient of ``bias``), computed
    from the same read: ``(row, col, bias_grad)``.
    """
    outs = _lib.gelu_backward_quantize_dual(dy, h, row, col, *_seed_flags(row_sr_seed, col_sr_seed, rht_seed),
                                            int(col_pad), *_seed_tensors(row_sr_seed, col_sr_seed, rht_seed), bias,
                                            _exact(approximate), bool(bias_grad))
    pair = _pair(outs[:6], row, col)
    return (*pair, _colsums(outs[6], bias_grad)) if bias_grad else pair


def rmsnorm_quantize_dual(x: torch.Tensor, weight: torch.Tensor, eps: float, *, row: bool = True, col: bool = True,
                          row_sr_seed: Seed = None, col_sr_seed: Seed = None, rht_seed: Seed = None,
                          col_pad: int = 16):
    """Quantize ``RMSNorm(x)`` [R, C] (fp32 weight [C]) without writing it.

    Returns ``(rstd, row, col)``: the per-row ``1 / sqrt(mean(x^2) + eps)`` ([R] fp32, what the RMSNorm
    backward needs) and the two quantize_dual outputs of ``(x * rstd) * weight`` in fp32.
    """
    outs = _lib.rmsnorm_quantize_dual(x, weight, float(eps), row, col,
                                      *_seed_flags(row_sr_seed, col_sr_seed, rht_seed), int(col_pad),
                                      *_seed_tensors(row_sr_seed, col_sr_seed, rht_seed))
    return (outs[0], *_pair(outs[1:], row, col))


def swiglu_quantize_dual(g: torch.Tensor, u: torch.Tensor, *, bias_g: torch.Tensor | None = None,
                         bias_u: torch.Tensor | None = None, row: bool = True, col: bool = True,
                         row_sr_seed: Seed = None, col_sr_seed: Seed = None, rht_seed: Seed = None, col_pad: int = 16):
    """Quantize ``silu(g + bias_g) * (u + bias_u)`` (bf16 gate / up projections, Llama's SwiGLU MLP; the
    optional per-column biases are added in fp32) without writing it."""
    outs = _lib.swiglu_quantize_dual(g, u, row, col, *_seed_flags(row_sr_seed, col_sr_seed, rht_seed), int(col_pad),
                                     *_seed_tensors(row_sr_seed, col_sr_seed, rht_seed), bias_g, bias_u)
    return _pair(outs, row, col)


def swiglu_backward_quantize_dual(dy: torch.Tensor, g: torch.Tensor, u: torch.Tensor, *, gate: bool,
                                  bias_g: torch.Tensor | None = None, bias_u: torch.Tensor | None = None,
                                  bias_grad: bool = False, row: bool = True, col: bool = True,
                                  row_sr_seed: Seed = None, col_sr_seed: Seed = None, rht_seed: Seed = None,
                                  col_pad: int = 16):
    """Quantize the gradient of ``silu(g) * u`` w.r.t. ``g`` (``gate=True``: ``dy * u * silu'(g)``) or w.r.t.
    ``u`` (``gate=False``: ``dy * silu(g)``) from its output gradient ``dy`` (all bf16), without writing it.
    ``g`` / ``u`` get the biases added first, as in swiglu_quantize_dual; ``bias_grad=True`` also returns the
    gradient's column sums (the gradient of that projection's bias): ``(row, col, bias_grad)``."""
    outs = _lib.swiglu_backward_quantize_dual(dy, g, u, bool(gate), row, col,
                                              *_seed_flags(row_sr_seed, col_sr_seed, rht_seed), int(col_pad),
                                              *_seed_tensors(row_sr_seed, col_sr_seed, rht_seed), bias_g, bias_u,
                                              bool(bias_grad))
    pair = _pair(outs[:6], row, col)
    return (*pair, _colsums(outs[6], bias_grad)) if bias_grad else pair


def norm_backward(dy: torch.Tensor, x: torch.Tensor, mean: torch.Tensor | None, rstd: torch.Tensor,
                  weight: torch.Tensor, *, dres: torch.Tensor | None = None, need_dx: bool = True,
                  need_dw: bool = True, need_db: bool = False):
    """LayerNorm (``mean`` given) or RMSNorm (``mean=None``) backward from the statistics the fused
    quantizers return. ``dy``: gradient of the normalized output (fp32 [R, C]); ``dres`` (optional, x's
    dtype) is added to dx, e.g. the gradient that reaches x through a residual connection. Returns
    ``(dx, dweight, dbias)`` (None where not requested). Deterministic."""
    dx, dw, db = _lib.norm_backward(dy, x, mean, rstd, weight, dres, bool(need_dx), bool(need_dw), bool(need_db))
    return dx if need_dx else None, dw if need_dw else None, db if need_db else None


def gemm(a: NVFP4Tensor, b: NVFP4Tensor, out_dtype: torch.dtype = torch.bfloat16, *,
         out: torch.Tensor | None = None, accumulate: bool = False, impl: str = "auto",
         splits: int | None = None, bias: torch.Tensor | None = None,
         residual: torch.Tensor | None = None) -> torch.Tensor:
    """D[M, N] = A[M, K] @ B[N, K]^T with both operands in NVFP4 (block-scaled tensor-core MMA).

    ``out`` writes into an existing contiguous [M, N] tensor; ``accumulate=True`` adds to it instead
    (fp32 only), e.g. to accumulate a weight gradient in place across micro-batches.
    ``impl``: "auto" (TMA main loop when K % 256 == 0, else cp.async), "tma" or "cp.async".
    ``splits``: number of K splits (0 = chosen from the grid size; None = $NVFP4_GEMM_SPLITS, default 0).
    The split-K reduction is deterministic but sums in a different order than an unsplit GEMM.

    ``bias`` ([N]) / ``residual`` ([M, N], bf16 or fp32): the linear-layer epilogue, computed in the GEMM's
    store, bit-identical to ``residual + (D_bf16.to(out_dtype) + bias.to(out_dtype))`` in PyTorch, where
    ``D_bf16`` is the bf16 GEMM output (so ``out_dtype`` is the layer's output dtype; the result has dtype
    ``promote(residual.dtype, out_dtype)``). Saves the separate bias / residual-add passes.
    """
    impl_id = {"auto": 0, "cp.async": 1, "tma": 2}[impl]
    if splits is None:
        splits = _DEFAULT_SPLITS
    if bias is not None or residual is not None:
        if out is not None or accumulate:
            raise ValueError("bias / residual cannot be combined with out / accumulate")
        dt = out_dtype if residual is None else torch.promote_types(residual.dtype, out_dtype)
        if dt not in (torch.bfloat16, torch.float32):
            raise ValueError(f"unsupported epilogue dtype {dt}")
        b32 = None if bias is None else bias.to(out_dtype).float().contiguous()
        res = None if residual is None else residual.contiguous()
        if res is not None and res.dtype not in (torch.bfloat16, torch.float32):
            res = res.to(dt)
        return _lib.gemm(a.data, a.scales, a.gdec, b.data, b.scales, b.gdec, dt, impl_id, splits, b32, res,
                         out_dtype == torch.bfloat16)
    if out is None:
        return _lib.gemm(a.data, a.scales, a.gdec, b.data, b.scales, b.gdec, out_dtype, impl_id, splits)
    _lib.gemm_out(a.data, a.scales, a.gdec, b.data, b.scales, b.gdec, out, accumulate, impl_id, splits)
    return out


def _scales_ld16(t: torch.Tensor) -> torch.Tensor:
    """Scales [R, C/16] with a row stride that is a multiple of 16 bytes (a padded copy if needed)."""
    c = t.shape[1]
    if c % 16 == 0 and t.is_contiguous():
        return t
    return torch.nn.functional.pad(t, (0, (-c) % 16))[:, :c]


def gemm_grouped(a: NVFP4Tensor, b: NVFP4Tensor, mode: str, meta: torch.Tensor, *, group_rows: int = 0,
                 out_dtype: torch.dtype = torch.bfloat16, out: torch.Tensor | None = None,
                 accumulate: bool = False, bias: torch.Tensor | None = None,
                 tile_rows: torch.Tensor | None = None) -> torch.Tensor:
    """Grouped NVFP4 GEMMs on the TMA main loop (mixture of experts; K a multiple of 128).

    ``mode="m"`` (ragged M: forward and input gradient): A [M, K] holds the groups' rows back to back, each
    group starting on a 128-row tile; ``meta`` (int32 [ceil(M / 128)]) gives the group of every tile (-1: an
    unused tile, whose output rows are stored as zeros without reading A or B) and
    ``b`` stacks the groups' [group_rows, K] matrices. Returns D [M, group_rows] with
    D[rows of g] = A[rows of g] @ B_g^T.

    ``mode="k"`` (ragged K: weight gradients): ``meta`` (int32 [G, 3]) = (k0, length, out_row0) per group,
    k0 and length multiples of 128 (length 0: the group's output tiles get zeros, or are left unchanged with
    ``accumulate``); ``out[out_row0 + i, j] (+)= sum_{k0 <= k < k0 + length} A[i, k] B[j, k]``
    for i < A's rows. ``out`` is required; with ``accumulate`` (fp32) it is added to.

    ``bias`` ([G, group_rows], mode "m", no ``out``): the groups' biases, added in the store as in ``gemm``
    (bit-identical to ``D_bf16.to(out_dtype) + bias[g].to(out_dtype)``; as there, every row of the output is
    then based on the bf16 product ``D_bf16``). ``tile_rows`` (int32, one per tile) limits the bias to each
    tile's first ``tile_rows[t]`` rows, so a group's padding rows stay zero; unused tiles get no bias.
    """
    m = {"m": 1, "k": 2}[mode]
    sfa, sfb = _scales_ld16(a.scales), _scales_ld16(b.scales)
    if sfa.stride(0) != sfb.stride(0):
        raise ValueError("A and B must have the same K")
    if out is not None and out.dtype != out_dtype:
        raise RuntimeError(f"out is {out.dtype}, out_dtype {out_dtype}")
    if bias is not None:
        if m != 1 or out is not None:
            raise ValueError("bias needs mode 'm' and no `out`")
        if out_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"unsupported epilogue dtype {out_dtype}")
        b32 = bias.reshape(-1).to(out_dtype).float().contiguous()
        return _lib.gemm_grouped(a.data, sfa, a.gdec, b.data, sfb, b.gdec, m, meta, int(group_rows), out_dtype,
                                 b32, tile_rows, out_dtype == torch.bfloat16)
    if out is None:
        return _lib.gemm_grouped(a.data, sfa, a.gdec, b.data, sfb, b.gdec, m, meta, int(group_rows), out_dtype)
    _lib.gemm_grouped_out(a.data, sfa, a.gdec, b.data, sfb, b.gdec, m, meta, int(group_rows), out, accumulate)
    return out


def stack_transposed(t: NVFP4Tensor, groups: int) -> NVFP4Tensor:
    """[K, G * N] -> [G * K, N]: the transposed quantization of G stacked [N, K] matrices (``quantize_dual`` of
    a [G * N, K] tensor, column output) rearranged so that group g's [K, N] block is rows [g K, (g + 1) K), the
    layout gemm_grouped(mode="m") reads (bytes only: blocks never straddle groups when N % 16 == 0)."""
    k, gn = t.shape
    if gn % (groups * 16):
        raise ValueError(f"{gn} columns do not split into {groups} groups of multiples of 16")
    n = gn // groups
    data = t.data.view(k, groups, n // 2).transpose(0, 1).reshape(groups * k, n // 2)
    scales = t.scales.view(k, groups, n // 16).transpose(0, 1).reshape(groups * k, n // 16)
    return NVFP4Tensor(data, scales, t.gdec)
