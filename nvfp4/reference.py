"""Pure-PyTorch reference implementation of the NVFP4 format and every transform used in training.

This module is the *specification*: the CUDA kernels in ``csrc/`` are tested to be bit-exact against
it (quantization) or to match it to fp32 accumulation tolerance (GEMM). It runs on CPU or GPU and
needs no compiled code, so CI can run it on machines without a GPU.

NVFP4 layout used throughout (row-major, quantized along the last dimension = the GEMM K dim):

* ``data``   uint8  ``[R, C/2]``  two E2M1 codes per byte, element ``2j`` in the low nibble.
* ``scales`` uint8  ``[R, C/16]`` one FP8 E4M3 (bit pattern) decode scale per 16-element block.
* ``gdec``   fp32   scalar        per-tensor decode scale.

Value of element ``(r, c)`` = ``e2m1(data) * e4m3(scales[r, c // 16]) * gdec``.

Recipe (matches NVIDIA's published NVFP4 recipe, two-level scaling):

    genc  = (448 * 6) / amax(tensor)          (1.0 if amax == 0, NaN if amax is NaN);  gdec = 1 / genc
    s_b   = e4m3_rn( min(amax(block) / 6 * genc, 448) )
    q     = round_e2m1( x * (genc / s_b) )     (q = 0 where s_b == 0)

All arithmetic is fp32 with IEEE round-to-nearest, in exactly this order, so a CUDA kernel doing the
same operations reproduces it bit for bit.
"""
from __future__ import annotations

import math

import torch

BLOCK = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
# Magnitudes of the 8 non-negative E2M1 codes 0..7.
E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

_M32 = 0xFFFFFFFF


# --------------------------------------------------------------------------------------------------
# Scalar formats
# --------------------------------------------------------------------------------------------------
def e2m1_round_rne(v: torch.Tensor) -> torch.Tensor:
    """Round fp32 values to E2M1 codes (0..15, bit 3 = sign), round-to-nearest-even, saturating.

    Ties go to the even code: 0.25->0, 0.75->1.0, 1.25->1.0, 1.75->2, 2.5->2, 3.5->4, 5->4.
    NaN is not expected (inputs are finite activations/weights/gradients); it maps to 0.
    """
    a = v.abs()
    code = torch.zeros_like(a, dtype=torch.int32)
    # Each comparison adds 1 when the magnitude passes the rounding boundary into the next code.
    code += (a > 0.25).int()
    code += (a >= 0.75).int()
    code += (a > 1.25).int()
    code += (a >= 1.75).int()
    code += (a > 2.5).int()
    code += (a >= 3.5).int()
    code += (a > 5.0).int()
    sign = (v < 0).int() & (code > 0).int()  # -0 is encoded as +0
    return code | (sign << 3)


def e2m1_decode(code: torch.Tensor) -> torch.Tensor:
    code = code.to(torch.int64)
    mag = E2M1_GRID.to(code.device)[code & 7]
    return torch.where((code & 8) != 0, -mag, mag)


def e4m3_from_float(x: torch.Tensor) -> torch.Tensor:
    """fp32 (non-negative, <= 448) -> E4M3 bit pattern as uint8, round-to-nearest-even."""
    return x.clamp(0.0, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def e4m3_to_float(b: torch.Tensor) -> torch.Tensor:
    return b.view(torch.float8_e4m3fn).to(torch.float32)


# --------------------------------------------------------------------------------------------------
# Counter-based RNG (identical integer hash in CUDA: see csrc/nvfp4_common.cuh)
# --------------------------------------------------------------------------------------------------
def hash_u32(x: torch.Tensor) -> torch.Tensor:
    """'lowbias32' integer hash (Chris Wellons). Input/output: int64 tensors holding uint32 values."""
    x = x & _M32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & _M32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & _M32
    x = x ^ (x >> 16)
    return x


def sr_key(seed: int) -> int:
    """32-bit key of a 64-bit stochastic-rounding seed: hash(lo ^ hash(hi))."""
    hi = hash_u32(torch.tensor([(seed >> 32) & _M32], dtype=torch.int64))
    return int(hash_u32(hi ^ (seed & _M32)).item())


def sr_bits(seed: int, index: torch.Tensor) -> torch.Tensor:
    """8 random bits (int64 0..255) per linear element index: elements 4q .. 4q + 3 take bytes 0 .. 3 of
    hash(q + key(seed)) (one hash per four elements, as NVIDIA's hardware SR feeds 8 bits per E2M1 element)."""
    index = index.to(torch.int64)
    h = hash_u32((index >> 2) + sr_key(seed))
    return (h >> (8 * (index & 3))) & 0xFF


def e2m1_round_stochastic(v: torch.Tensor, r8: torch.Tensor) -> torch.Tensor:
    """Stochastic rounding of fp32 values to E2M1 codes (saturating at 6) with 8 random bits per element.

    Integer-domain, as the kernels: magnitudes below 1 are lifted into [1, 2) (where E2M1's 0.5 spacing is one
    mantissa bit), then the random byte plus half a unit of it (the unbiased "SRF" few-bit rounding of
    Fitzgibbon & Felix, ARITH 2025) is added below the one kept mantissa bit and the rest is truncated. The
    probability of rounding up is round(256 f) / 256 for the fractional position f in the grid step (exact on
    average; within 2^-9 of f)."""
    a = v.float().abs()
    off = (a < 1.0).float()
    z = a + off
    b = z.view(torch.int32).to(torch.int64) + (r8.to(torch.int64) << 14) + (1 << 13)
    b = (b & 0xFFC00000).to(torch.int32).view(torch.float32)
    r = torch.minimum(b - off, torch.full_like(b, E2M1_MAX))
    return e2m1_round_rne(torch.where(v < 0, -r, r))


# --------------------------------------------------------------------------------------------------
# Random Hadamard transform on 16-element blocks
# --------------------------------------------------------------------------------------------------
def hadamard16() -> torch.Tensor:
    h = torch.ones(1, 1)
    for _ in range(4):
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h  # entries +-1, H H^T = 16 I


def rht_signs(seed: int, n: int = 16) -> torch.Tensor:
    """``n`` random signs derived from ``seed`` with the same hash used on the GPU (the first 16 do not
    depend on n)."""
    bits = hash_u32(torch.arange(n, dtype=torch.int64) ^ (seed & _M32)) & 1
    return 1.0 - 2.0 * bits.to(torch.float32)


def rht_matrix(seed: int) -> torch.Tensor:
    """Orthogonal 16x16 matrix M = diag(s) H / 4, so (A M)(B M)^T == A B^T for block-diagonal use."""
    return rht_signs(seed).unsqueeze(1) * hadamard16() * 0.25


def apply_rht(x: torch.Tensor, seed: int, n: int = 16) -> torch.Tensor:
    """Apply M = diag(s) H / 4 to every 16-element block along the last dim: y = x_blk @ M.

    Computed as sign flip -> fast Walsh-Hadamard butterflies (h = 1, 2, 4, 8) -> * 0.25, in fp32, in
    exactly the order the CUDA kernel uses, so results are bit-identical (a dense matmul would not be).
    Mathematically equal to ``x_blk @ rht_matrix(seed)`` (H is symmetric).

    n = 64 (MS-EDEN): the 64-point RHT diag(s64) H64 / 8 on every 64-element block, as the dual quantizer
    computes it: the 16-point butterflies of each 16-block with its slice of the 64 signs (* 0.25), then the
    4-point Hadamard across the four 16-blocks (* 0.5), H64 = H4 (x) H16.
    """
    if n == 64:
        *lead, c = x.shape
        assert c % 64 == 0
        v = x.float().reshape(-1, 64) * rht_signs(seed, 64).to(x.device)
        v = _wht16(v.reshape(-1, BLOCK)).reshape(-1, 4, BLOCK)
        w0, w1, w2, w3 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        a, d01, e, d23 = w0 + w1, w0 - w1, w2 + w3, w2 - w3
        return (torch.stack([a + e, d01 + d23, a - e, d01 - d23], 1) * 0.5).reshape(*lead, c)
    assert n == 16
    *lead, c = x.shape
    assert c % BLOCK == 0
    v = x.float().reshape(-1, BLOCK) * rht_signs(seed).to(x.device)
    return _wht16(v).reshape(*lead, c)


def _wht16(v: torch.Tensor) -> torch.Tensor:
    """Butterflies h = 1, 2, 4, 8 and * 0.25 on [n, 16] (the kernel's order)."""
    h = 1
    while h < BLOCK:
        v = v.reshape(-1, BLOCK // (2 * h), 2, h)
        a, b = v[:, :, 0, :], v[:, :, 1, :]
        v = torch.stack([a + b, a - b], 2)
        h *= 2
    return v.reshape(-1, BLOCK) * 0.25


# --------------------------------------------------------------------------------------------------
# Tensor quantization
# --------------------------------------------------------------------------------------------------
FOUR_SIX_E4M3_MAX = 256.0  # tensor-scale constant with 4/6 block scales (a block mapped to 4 needs 1.5x)


EDEN_S = 6.07210636138916        # fp32 of (1 / 0.93) * 6 * 16 / 17: MS-EDEN's block-max grid point
EDEN_NUM = 1554.459228515625      # fp32 of EDEN_S * 256: MS-EDEN's tensor-scale numerator


def global_scales(amax: torch.Tensor, four_six: bool = False, eden: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """genc = 6 * 448 / amax (6 * 256 / amax with 4/6 block scales, EDEN_S * 256 / amax for MS-EDEN),
    gdec = 1 / genc."""
    # NB: `scalar / tensor` in PyTorch is reciprocal(tensor) * scalar (two roundings); use true
    # IEEE division so this matches the kernel's __fdiv_rn bit for bit.
    amax = amax.float()
    num = torch.full_like(amax, EDEN_NUM if eden else (FOUR_SIX_E4M3_MAX if four_six else E4M3_MAX) * E2M1_MAX)
    genc = torch.where(amax > 0, torch.div(num, torch.where(amax > 0, amax, torch.ones_like(amax))),
                       torch.ones_like(amax))
    # A NaN anywhere must not quantize to finite values: it makes the decode scale (and so every
    # GEMM output using this tensor) NaN. An inf amax gives genc 0 and gdec inf, likewise non-finite.
    genc = torch.where(torch.isnan(amax), torch.full_like(amax, float("nan")), genc)
    return genc, torch.div(torch.ones_like(genc), genc)


def _rtn_candidate(blocks, bamax, div: float, genc):
    """Round-to-nearest quantization of the blocks with the block max mapped to ``div`` (6 or 4): codes, scale
    bytes, and the squared error in the encoded domain, sum((|v| - q)^2) * sf^2, summed in element order
    (as the kernel, so that the 4/6 choice matches it bit for bit)."""
    s_target = torch.div(bamax, torch.full_like(bamax, div)) * genc
    s = e4m3_from_float(torch.minimum(s_target, torch.tensor(E4M3_MAX, device=blocks.device)))
    sf = e4m3_to_float(s)
    enc = torch.where(sf > 0, genc / torch.where(sf > 0, sf, torch.ones_like(sf)), torch.zeros_like(sf))
    v = blocks * enc.unsqueeze(-1)
    codes = e2m1_round_rne(v)
    d = v.abs() - e2m1_decode(codes & 7).float()
    e = torch.zeros_like(sf)
    for i in range(BLOCK):
        e = e + d[..., i] * d[..., i]
    return codes, s, e * (sf * sf)


def _eden_blocks(blocks, bamax, genc, seed: int):
    """MS-EDEN (Quartet II): RTN codes at the block scale bmax / EDEN_S * genc, then the scale replaced by an
    E4M3 stochastic rounding of S * sf, S = sum(y^2) / sum(y q) per block (element order; 1 if all codes are 0).
    The random byte of block k is sr_bits(seed, k); below E4M3's normal range (2^-6) the scale is RN."""
    s_target = torch.div(bamax, torch.full_like(bamax, EDEN_S)) * genc
    sf = e4m3_to_float(e4m3_from_float(torch.minimum(s_target, torch.tensor(E4M3_MAX, device=blocks.device))))
    enc = torch.where(sf > 0, genc / torch.where(sf > 0, sf, torch.ones_like(sf)), torch.zeros_like(sf))
    y = blocks * enc.unsqueeze(-1)
    codes = e2m1_round_rne(y)
    q = e2m1_decode(codes).float()
    yy, yq = torch.zeros_like(sf), torch.zeros_like(sf)
    for i in range(BLOCK):
        yy = yy + y[..., i] * y[..., i]
        yq = yq + y[..., i] * q[..., i]
    corr = torch.where(yq > 0, yy / torch.where(yq > 0, yq, torch.ones_like(yq)), torch.ones_like(yq))
    t = corr * sf
    r8 = sr_bits(seed, torch.arange(t.numel(), device=t.device, dtype=torch.int64)).reshape(t.shape)
    bits = t.view(torch.int32).to(torch.int64) + (r8 << 12) + (1 << 11)
    t_sr = (bits & 0xFFF00000).to(torch.int32).view(torch.float32)
    t = torch.where(t >= 0.015625, t_sr, t)
    s = e4m3_from_float(torch.minimum(t, torch.tensor(E4M3_MAX, device=t.device)))
    return codes, s


def quantize(x: torch.Tensor, *, block_2d: bool = False, stochastic: bool = False, seed: int = 0,
             amax: torch.Tensor | None = None, four_six: bool = False, eden: bool = False):
    """Quantize a 2D tensor along its last dim. Returns (data uint8 [R,C/2], scales uint8 [R,C/16], gdec fp32).

    block_2d: one scale per 16x16 tile (rows r..r+15 share scales), so that quantize(W) and
              quantize(W.T) represent exactly the same values (NVIDIA recipe for weights).
    stochastic: unbiased stochastic rounding of the E2M1 step (gradients).
    four_six: "Four Over Six" adaptive block scales (round-to-nearest only; ignored with ``stochastic``):
              each block is quantized with its max mapped to 6 and to 4 and keeps the one with the smaller
              squared error, and the tensor scale uses 256 instead of 448 so the 1.5x scale fits E4M3
              (Cook et al., arXiv 2512.02010).
    eden: MS-EDEN instead of element-wise stochastic rounding (with ``stochastic``; Quartet II, arXiv 2601.22813):
          RTN elements, the block scale stochastically rounded with the EDEN correction (see _eden_blocks).
          Unbiased when ``x`` was randomly rotated (the dual quantizer's 64-point RHT).
    """
    assert x.dim() == 2
    r, c = x.shape
    assert c % BLOCK == 0, f"last dim {c} must be a multiple of {BLOCK}"
    xf = x.float()
    if amax is None:
        amax = xf.abs().max()
    eden = eden and stochastic
    four_six = four_six and not stochastic
    genc, gdec = global_scales(amax, four_six, eden)
    blocks = xf.reshape(r, c // BLOCK, BLOCK)
    bamax = blocks.abs().amax(-1)  # [R, C/16]
    if block_2d:
        assert r % BLOCK == 0, f"rows {r} must be a multiple of {BLOCK} for 2D blocks"
        bamax = bamax.reshape(r // BLOCK, BLOCK, c // BLOCK).amax(1, keepdim=True).expand(-1, BLOCK, -1).reshape(r, c // BLOCK)
    if eden:
        codes, s = _eden_blocks(blocks, bamax, genc, seed)
        codes = codes.reshape(r, c).to(torch.uint8)
        data = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
        return data.contiguous(), s.contiguous(), gdec.reshape(()).float()
    if four_six:
        c6, s6, e6 = _rtn_candidate(blocks, bamax, E2M1_MAX, genc)
        c4, s4, e4 = _rtn_candidate(blocks, bamax, 4.0, genc)
        pick4 = e4 < e6
        codes = torch.where(pick4.unsqueeze(-1), c4, c6).reshape(r, c).to(torch.uint8)
        s = torch.where(pick4, s4, s6)
        data = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
        return data.contiguous(), s.contiguous(), gdec.reshape(()).float()
    # NB: `tensor / python_scalar` on CUDA is computed as tensor * (1/scalar), which is not IEEE division
    # and differs at E4M3 rounding ties; divide by a full tensor so CPU, CUDA and the kernels agree.
    s_target = torch.div(bamax, torch.full_like(bamax, E2M1_MAX)) * genc
    s = e4m3_from_float(torch.minimum(s_target, torch.tensor(E4M3_MAX, device=x.device)))
    sf = e4m3_to_float(s)
    enc = torch.where(sf > 0, genc / torch.where(sf > 0, sf, torch.ones_like(sf)), torch.zeros_like(sf))
    v = blocks * enc.unsqueeze(-1)
    if stochastic:
        idx = torch.arange(r * c, device=x.device, dtype=torch.int64).reshape(r, c // BLOCK, BLOCK)
        codes = e2m1_round_stochastic(v, sr_bits(seed, idx))
    else:
        codes = e2m1_round_rne(v)
    codes = codes.reshape(r, c).to(torch.uint8)
    data = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return data.contiguous(), s.contiguous(), gdec.reshape(()).float()


ROW_RHT_SALT = 0x1E3779B97F4A7C15  # csrc kRowRhtSalt


def quantize_dual(x: torch.Tensor, *, row: bool = True, col: bool = True, block_2d: bool = False,
                  row_sr_seed: int | None = None, col_sr_seed: int | None = None, rht_seed: int | None = None,
                  col_pad: int = 16, four_six: bool = False, ms_eden: bool = False, ms_eden_dgrad: bool = False):
    """Spec of the fused dual quantizer: (quantize(x), quantize(RHT(pad_rows(x, col_pad).T))).

    Either output is None when disabled. A seed of None means round-to-nearest-even. ``ms_eden_dgrad``: a
    stochastically rounded row output is the 64-point RHT of x along its rows with seed row_sr_seed ^
    ROW_RHT_SALT, quantized with EDEN (the dgrad operand of MS-EDEN).
    """
    xf = x.float()
    r_out = c_out = None
    if row:
        if ms_eden_dgrad and row_sr_seed is not None and not block_2d:
            r_out = quantize(apply_rht(xf, row_sr_seed ^ ROW_RHT_SALT, 64), stochastic=True, seed=row_sr_seed,
                             eden=True)
        else:
            r_out = quantize(xf, block_2d=block_2d, stochastic=row_sr_seed is not None, seed=row_sr_seed or 0,
                             four_six=four_six)
    if col:
        pad = (-xf.shape[0]) % col_pad
        y = torch.nn.functional.pad(xf, (0, 0, 0, pad)).T.contiguous()
        eden = ms_eden and rht_seed is not None
        if rht_seed is not None:
            y = apply_rht(y, rht_seed, 64 if eden else 16)
        c_out = quantize(y, block_2d=block_2d, stochastic=col_sr_seed is not None, seed=col_sr_seed or 0,
                         four_six=four_six, eden=eden)
    return r_out, c_out


# --------------------------------------------------------------------------------------------------
# Producer ops fused into the quantizer (csrc/quantize_dual.cu "Pro"). Each is written one rounded
# fp32 op at a time, in the kernel's order. Run them on the same device as the kernel for bit-exact
# comparisons: tanh is libdevice tanhf on CUDA but a different implementation on CPU.

GELU_BETA = 0.7978845608028654   # sqrt(2 / pi); used as fp32, like the kernel's constants
GELU_KAPPA = 0.044715
GELU_3KAPPA = 0.134145


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """tanh-approximate GELU in fp32: (x * 0.5) * (tanh((x + x^3 * kappa) * beta) + 1)."""
    x = x.float()
    x3 = (x * x) * x
    inner = (x + x3 * GELU_KAPPA) * GELU_BETA
    return (x * 0.5) * (torch.tanh(inner) + 1.0)


def gelu_tanh_backward(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dy * gelu'(x) in fp32, with gelu as in gelu_tanh."""
    dy, x = dy.float(), x.float()
    x2 = x * x
    t = torch.tanh((x + (x2 * x) * GELU_KAPPA) * GELU_BETA)
    left_d = (t + 1.0) * 0.5
    right_d = (((x * 0.5) * (1.0 - t * t)) * GELU_BETA) * (x2 * GELU_3KAPPA + 1.0)
    return dy * (left_d + right_d)


SQRT1_2 = 0.7071067811865476      # 1 / sqrt(2)
INV_SQRT_2PI = 0.3989422804014327  # 1 / sqrt(2 pi)


def gelu_erf(x: torch.Tensor) -> torch.Tensor:
    """Exact GELU in fp32: (x * 0.5) * (1 + erf(x / sqrt(2)))."""
    x = x.float()
    return (x * 0.5) * (torch.erf(x * SQRT1_2) + 1.0)


def gelu_erf_backward(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dy * gelu'(x) in fp32 for the exact GELU: dy * (cdf + x * pdf)."""
    dy, x = dy.float(), x.float()
    cdf = (torch.erf(x * SQRT1_2) + 1.0) * 0.5
    pdf = torch.exp((x * -0.5) * x) * INV_SQRT_2PI
    return dy * (cdf + x * pdf)


def _gelu_input(h, bias):
    return h.float() if bias is None else h.float() + bias.float()


def layernorm_stats(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row mean and 1 / sqrt(biased var + eps) (the kernel's reduction order differs: compare with a tolerance)."""
    x = x.double()
    mean = x.mean(-1)
    var = ((x - mean[:, None]) ** 2).mean(-1)
    return mean.float(), (1.0 / torch.sqrt(var + eps)).float()


def layernorm_apply(x: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, weight: torch.Tensor,
                    bias: torch.Tensor) -> torch.Tensor:
    """((x - mean) * rstd) * weight + bias in fp32, with the per-row statistics given."""
    return ((x.float() - mean.float()[:, None]) * rstd.float()[:, None]) * weight.float() + bias.float()


def layernorm_quantize_dual(x, weight, bias, mean, rstd, **kw):
    """Spec of ops.layernorm_quantize_dual given its statistics: quantize_dual(layernorm_apply(...))."""
    return quantize_dual(layernorm_apply(x, mean, rstd, weight, bias), **kw)


def gelu_quantize_dual(h, *, bias=None, approximate="tanh", **kw):
    """Spec of ops.gelu_quantize_dual."""
    x = _gelu_input(h, bias)
    return quantize_dual(gelu_tanh(x) if approximate == "tanh" else gelu_erf(x), **kw)


def gelu_backward(dy, h, *, bias=None, approximate="tanh"):
    """dy * gelu'(h + bias) in fp32."""
    x = _gelu_input(h, bias)
    return gelu_tanh_backward(dy, x) if approximate == "tanh" else gelu_erf_backward(dy, x)


def gelu_backward_quantize_dual(dy, h, *, bias=None, approximate="tanh", **kw):
    """Spec of ops.gelu_backward_quantize_dual (without the bias gradient: gelu_backward(...).sum(0))."""
    return quantize_dual(gelu_backward(dy, h, bias=bias, approximate=approximate), **kw)


def rmsnorm_stats(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-row 1 / sqrt(mean(x^2) + eps) (the kernel's reduction order differs: compare with a tolerance)."""
    x = x.double()
    return (1.0 / torch.sqrt((x * x).mean(-1) + eps)).float()


def rmsnorm_apply(x: torch.Tensor, rstd: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """(x * rstd) * weight in fp32, with the per-row statistic given."""
    return (x.float() * rstd.float()[:, None]) * weight.float()


def rmsnorm_quantize_dual(x, weight, rstd, **kw):
    """Spec of ops.rmsnorm_quantize_dual given its statistic: quantize_dual(rmsnorm_apply(...))."""
    return quantize_dual(rmsnorm_apply(x, rstd, weight), **kw)


def silu(g: torch.Tensor) -> torch.Tensor:
    """g / (1 + exp(-g)) in fp32 (exp is libdevice expf on CUDA, like the kernel)."""
    g = g.float()
    return g / (torch.exp(-g) + 1.0)


def swiglu(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """silu(g) * u in fp32 (Llama's MLP activation)."""
    return silu(g) * u.float()


def swiglu_backward_gate(dy: torch.Tensor, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """d(silu(g) * u)/dg times dy: (dy * u) * (s * (1 + g * (1 - s))), s = sigmoid(g)."""
    dy, g, u = dy.float(), g.float(), u.float()
    s = 1.0 / (torch.exp(-g) + 1.0)
    return (dy * u) * (s * (g * (1.0 - s) + 1.0))


def swiglu_backward_up(dy: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """d(silu(g) * u)/du times dy: dy * silu(g)."""
    return dy.float() * silu(g)


def swiglu_quantize_dual(g, u, *, bias_g=None, bias_u=None, **kw):
    """Spec of ops.swiglu_quantize_dual (biases added in fp32)."""
    return quantize_dual(swiglu(_gelu_input(g, bias_g), _gelu_input(u, bias_u)), **kw)


def swiglu_backward(dy, g, u, *, gate: bool, bias_g=None, bias_u=None):
    """The gradient of silu(g + bias_g) * (u + bias_u) w.r.t. g (gate) or u, in fp32."""
    g, u = _gelu_input(g, bias_g), _gelu_input(u, bias_u)
    return swiglu_backward_gate(dy, g, u) if gate else swiglu_backward_up(dy, g)


def swiglu_backward_quantize_dual(dy, g, u, *, gate: bool, bias_g=None, bias_u=None, **kw):
    """Spec of ops.swiglu_backward_quantize_dual."""
    return quantize_dual(swiglu_backward(dy, g, u, gate=gate, bias_g=bias_g, bias_u=bias_u), **kw)


def norm_backward(dy, x, mean, rstd, weight, dres=None):
    """Spec of ops.norm_backward (fp32): LayerNorm if ``mean`` is given, else RMSNorm. Returns (dx, dw, db)."""
    dy, xf, w = dy.float(), x.float(), weight.float()
    n = (xf if mean is None else xf - mean.float()[:, None]) * rstd.float()[:, None]
    g = dy * w
    proj = n * (g * n).mean(-1, keepdim=True)
    dx = rstd.float()[:, None] * (g - proj if mean is None else g - g.mean(-1, keepdim=True) - proj)
    if dres is not None:
        dx = dx + dres.float()
    return dx.to(x.dtype), (dy * n).sum(0), dy.sum(0)


def linear_epilogue(y_bf16, out_dtype, bias=None, residual=None):
    """Spec of ops.gemm's bias / residual epilogue: residual + (y.to(out_dtype) + bias.to(out_dtype))."""
    y = y_bf16.to(torch.bfloat16).to(out_dtype)
    if bias is not None:
        y = y + bias.to(out_dtype)
    return y if residual is None else residual + y


def unpack_codes(data: torch.Tensor) -> torch.Tensor:
    lo = data & 0xF
    hi = data >> 4
    return torch.stack([lo, hi], -1).reshape(data.shape[0], data.shape[1] * 2)


def dequantize(data: torch.Tensor, scales: torch.Tensor, gdec: torch.Tensor) -> torch.Tensor:
    vals = e2m1_decode(unpack_codes(data))
    sf = e4m3_to_float(scales).repeat_interleave(BLOCK, dim=1)
    return vals * sf * gdec


def gemm(a: tuple, b: tuple) -> torch.Tensor:
    """Reference D[M,N] = A[M,K] @ B[N,K]^T for NVFP4 operands (each a (data, scales, gdec) tuple)."""
    da, db = dequantize(*a).double(), dequantize(*b).double()
    return (da @ db.T).float()


def fake_quant(x: torch.Tensor, **kw) -> torch.Tensor:
    """quantize + dequantize, same dtype/shape as x. Useful for numerics studies."""
    return dequantize(*quantize(x, **kw)).to(x.dtype)


def nvfp4_bytes(r: int, c: int) -> int:
    return r * c // 2 + r * c // BLOCK


def snr_db(ref: torch.Tensor, approx: torch.Tensor) -> float:
    err = (ref.double() - approx.double()).pow(2).mean().item()
    sig = ref.double().pow(2).mean().item()
    return 10.0 * math.log10(sig / err) if err > 0 else float("inf")


# --- MXFP8 attention (nvfp4.attention) ----------------------------------------------------------------------

def _sigma(k: int) -> int:
    return 16 * (k >> 4) + 8 * ((k & 3) >> 1) + 2 * ((k & 15) >> 2) + (k & 1)


ATTN_SIGMA = [_sigma(k) for k in range(32)]  # token at k-position k of a 32-block in the col layout


def mxfp8_scale_exp(amax: torch.Tensor) -> torch.Tensor:
    """Biased UE8M0 exponent of the smallest power of two 2^E with amax / 2^E <= 448, clamped to [1, 253]."""
    bits = (amax.float() * (1.0 / 448.0)).view(torch.int32)
    e = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return e.clamp(1, 253)


def mxfp8_quantize(x: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 along ``dim`` (blocks of 32): returns (E4M3 codes as torch.float8_e4m3fn, biased UE8M0 exponents
    [..., n/32] with ``dim`` moved last). Value = code * 2^(e - 127)."""
    xm = x.float().movedim(dim, -1)
    blocks = xm.reshape(*xm.shape[:-1], xm.shape[-1] // 32, 32)
    e = mxfp8_scale_exp(blocks.abs().amax(-1))
    inv = torch.exp2((127 - e).float()).unsqueeze(-1)
    codes = (blocks * inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return codes.reshape(xm.shape), e


def mxfp8_fake(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Quantize-dequantize (fp32) along ``dim`` with MXFP8 blocks of 32."""
    codes, e = mxfp8_quantize(x, dim)
    xm = codes.float().reshape(*codes.shape[:-1], codes.shape[-1] // 32, 32)
    xm = xm * torch.exp2((e - 127).float()).unsqueeze(-1)
    return xm.reshape(codes.shape).movedim(-1, dim)


def int8_token_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 with one scale per row (last dim): codes rn(x * (127 / amax)), scale amax / 127."""
    xf = x.float()
    am = xf.abs().amax(-1, keepdim=True)
    # correctly rounded fp32 divisions, as in the kernel (PyTorch divides by a scalar via its reciprocal)
    inv = torch.where(am > 0, (127.0 / am.double()).float(), torch.zeros_like(am))
    codes = torch.round(xf * inv).clamp(-127, 127).to(torch.int8)
    return codes, (am.double() / 127.0).float().squeeze(-1)


def int8_token_fake(x: torch.Tensor) -> torch.Tensor:
    codes, s = int8_token_quantize(x)
    return codes.float() * s.unsqueeze(-1)


def _pad_seq(x: torch.Tensor, mult: int = 128) -> torch.Tensor:
    n = x.shape[-2]
    pad = (n + mult - 1) // mult * mult - n
    return torch.nn.functional.pad(x, (0, 0, 0, pad)) if pad else x


def attention_8bit(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False,
                   scale: float | None = None, smooth_k: bool = True) -> torch.Tensor:
    """fp32 emulation of nvfp4.attention's forward ([B, H, N, D] inputs, K/V heads may divide H): Q and K
    (minus its per-channel mean over the sequence) quantized to INT8 per token, V to MXFP8 along the sequence,
    as the kernels do, and the probabilities rounded to E4M3 as 256 p. The kernel takes p relative to the
    running maximum of its 64-key tiles and this emulation relative to the row maximum, so individual P values
    can round differently (both are unbiased to within E4M3 precision)."""
    B, H, N, D = q.shape
    scale = D ** -0.5 if scale is None else scale
    kk = k.float()
    if smooth_k:
        kk = kk - kk.mean(dim=2, keepdim=True)
    qf = int8_token_fake(q)
    kf = int8_token_fake(kk)
    vf = mxfp8_fake(_pad_seq(v.float()), dim=-2)[..., :N, :]
    rep = H // k.shape[1]
    kf, vf = kf.repeat_interleave(rep, 1), vf.repeat_interleave(rep, 1)
    s = (qf @ kf.transpose(-1, -2)) * scale
    if causal:
        s = s.masked_fill(torch.ones(N, N, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    m = s.amax(-1, keepdim=True)
    p = torch.exp(s - m)
    p8 = (p * 256).clamp(max=448.0).to(torch.float8_e4m3fn).float() / 256
    return (p8 @ vf) / p.sum(-1, keepdim=True)
