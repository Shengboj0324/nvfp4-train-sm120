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

    genc  = (448 * 6) / amax(tensor)          (1.0 if amax == 0);  gdec = 1 / genc
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


def sr_uniform(seed: int, index: torch.Tensor) -> torch.Tensor:
    """Uniform [0, 1) fp32 for each linear element index; 24-bit resolution."""
    h = hash_u32(hash_u32(index.to(torch.int64) ^ (seed & _M32)) + ((seed >> 32) & _M32))
    return (h >> 8).to(torch.float32) * (1.0 / 16777216.0)


def e2m1_round_stochastic(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Unbiased stochastic rounding to E2M1 (saturating at 6). ``u`` ~ U[0,1) per element."""
    a = v.abs().clamp(max=E2M1_MAX)
    grid = E2M1_GRID.to(v.device)
    # lo code: largest grid point <= a
    lo = (a.unsqueeze(-1) >= grid).sum(-1) - 1  # 0..7
    lo = lo.clamp(0, 7)
    hi = (lo + 1).clamp(max=7)
    glo, ghi = grid[lo], grid[hi]
    width = ghi - glo
    p_up = torch.where(width > 0, (a - glo) / torch.where(width > 0, width, torch.ones_like(width)),
                       torch.zeros_like(a))
    code = torch.where(u < p_up, hi, lo).int()
    sign = (v < 0).int() & (code > 0).int()
    return code | (sign << 3)


# --------------------------------------------------------------------------------------------------
# Random Hadamard transform on 16-element blocks
# --------------------------------------------------------------------------------------------------
def hadamard16() -> torch.Tensor:
    h = torch.ones(1, 1)
    for _ in range(4):
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h  # entries +-1, H H^T = 16 I


def rht_signs(seed: int) -> torch.Tensor:
    """16 random signs derived from ``seed`` with the same hash used on the GPU."""
    bits = hash_u32(torch.arange(16, dtype=torch.int64) ^ (seed & _M32)) & 1
    return 1.0 - 2.0 * bits.to(torch.float32)


def rht_matrix(seed: int) -> torch.Tensor:
    """Orthogonal 16x16 matrix M = diag(s) H / 4, so (A M)(B M)^T == A B^T for block-diagonal use."""
    return rht_signs(seed).unsqueeze(1) * hadamard16() * 0.25


def apply_rht(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Apply M = diag(s) H / 4 to every 16-element block along the last dim: y = x_blk @ M.

    Computed as sign flip -> fast Walsh-Hadamard butterflies (h = 1, 2, 4, 8) -> * 0.25, in fp32, in
    exactly the order the CUDA kernel uses, so results are bit-identical (a dense matmul would not be).
    Mathematically equal to ``x_blk @ rht_matrix(seed)`` (H is symmetric).
    """
    *lead, c = x.shape
    assert c % BLOCK == 0
    v = x.float().reshape(-1, BLOCK) * rht_signs(seed).to(x.device)
    h = 1
    while h < BLOCK:
        v = v.reshape(-1, BLOCK // (2 * h), 2, h)
        a, b = v[:, :, 0, :], v[:, :, 1, :]
        v = torch.stack([a + b, a - b], 2)
        h *= 2
    return (v.reshape(-1, BLOCK) * 0.25).reshape(*lead, c)


# --------------------------------------------------------------------------------------------------
# Tensor quantization
# --------------------------------------------------------------------------------------------------
def global_scales(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # NB: `scalar / tensor` in PyTorch is reciprocal(tensor) * scalar (two roundings); use true
    # IEEE division so this matches the kernel's __fdiv_rn bit for bit.
    amax = amax.float()
    num = torch.full_like(amax, E4M3_MAX * E2M1_MAX)
    genc = torch.where(amax > 0, torch.div(num, torch.where(amax > 0, amax, torch.ones_like(amax))),
                       torch.ones_like(amax))
    return genc, torch.div(torch.ones_like(genc), genc)


def quantize(x: torch.Tensor, *, block_2d: bool = False, stochastic: bool = False, seed: int = 0,
             amax: torch.Tensor | None = None):
    """Quantize a 2D tensor along its last dim. Returns (data uint8 [R,C/2], scales uint8 [R,C/16], gdec fp32).

    block_2d: one scale per 16x16 tile (rows r..r+15 share scales), so that quantize(W) and
              quantize(W.T) represent exactly the same values (NVIDIA recipe for weights).
    stochastic: unbiased stochastic rounding of the E2M1 step (gradients).
    """
    assert x.dim() == 2
    r, c = x.shape
    assert c % BLOCK == 0, f"last dim {c} must be a multiple of {BLOCK}"
    xf = x.float()
    if amax is None:
        amax = xf.abs().max()
    genc, gdec = global_scales(amax)
    blocks = xf.reshape(r, c // BLOCK, BLOCK)
    bamax = blocks.abs().amax(-1)  # [R, C/16]
    if block_2d:
        assert r % BLOCK == 0, f"rows {r} must be a multiple of {BLOCK} for 2D blocks"
        bamax = bamax.reshape(r // BLOCK, BLOCK, c // BLOCK).amax(1, keepdim=True).expand(-1, BLOCK, -1).reshape(r, c // BLOCK)
    s = e4m3_from_float(torch.minimum(bamax / E2M1_MAX * genc, torch.tensor(E4M3_MAX, device=x.device)))
    sf = e4m3_to_float(s)
    enc = torch.where(sf > 0, genc / torch.where(sf > 0, sf, torch.ones_like(sf)), torch.zeros_like(sf))
    v = blocks * enc.unsqueeze(-1)
    if stochastic:
        idx = torch.arange(r * c, device=x.device, dtype=torch.int64).reshape(r, c // BLOCK, BLOCK)
        codes = e2m1_round_stochastic(v, sr_uniform(seed, idx))
    else:
        codes = e2m1_round_rne(v)
    codes = codes.reshape(r, c).to(torch.uint8)
    data = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return data.contiguous(), s.contiguous(), gdec.reshape(()).float()


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
