# Design

This document explains how the NVFP4 training stack in this repository works, why it is shaped the
way it is for consumer Blackwell (sm_120), and how each piece was verified on real hardware.

## 1. Why consumer Blackwell needs its own kernels

RTX 50-series GPUs (compute capability 12.0, `sm_120`) have 5th-generation tensor cores with
hardware block-scaled FP4/FP6/FP8 support. However, they don't share the datacenter Blackwell
(`sm_100`, B200) programming model:

| | sm_100 (B200) | **sm_120 (RTX 50)** | sm_90 (H100) |
|---|---|---|---|
| Tensor-core interface | `tcgen05.mma` (async, operands in shared/tensor memory) | **`mma.sync` (warp-level, operands in registers)** | `wgmma` |
| Tensor memory (TMEM) | yes | **no** | no |
| Shared memory per SM | 228 KB | **~99 KB usable per block** | 228 KB |
| FP4 block-scaled MMA | `tcgen05.mma ... block_scale` | **`mma.sync ... kind::mxf4nvf4.block_scale`** | — |

So kernels written for B200, such as FlashAttention-4, cuDNN's Blackwell kernels and DeepGEMM, can't
just be recompiled for sm_120. They have to be redesigned around warp-level MMAs and a third of the
shared memory. Mainstream training stacks also don't yet run NVFP4 training on sm_120 (see
[`docs/research/`](research/)). This repository fills that gap for linear layers.

The instruction used by every GEMM here is

```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

It must be compiled for the arch-specific target `sm_120a`. In SASS it appears as
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`, and CI checks that the instruction is present.

## 2. Number format and recipe

**NVFP4** stores each element as E2M1, a 4-bit float with values ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}. Each
block of 16 consecutive elements along the GEMM K dimension shares one FP8 E4M3 scale, and each
tensor has one FP32 scale on top of that:

```
value = e2m1(code) * e4m3(block_scale) * tensor_decode_scale

genc  = (448 * 6) / amax(tensor)            tensor_decode_scale = 1 / genc
s_b   = e4m3_rn(min(amax(block) / 6 * genc, 448))
code  = round_e2m1(x * (genc / s_b))        round-to-nearest-even, or stochastic
```

`nvfp4/reference.py` is the executable specification: a pure-PyTorch implementation of the format,
the 16-point random Hadamard transform (RHT) and stochastic rounding. The CUDA quantizer is tested to
be bit-identical to it in every mode.

**Training recipe.** This follows NVIDIA's published NVFP4 pretraining recipe.

| GEMM | Computation | A operand | B operand |
|---|---|---|---|
| fprop | `Y = X W^T` | X, 1×16 blocks, RNE | W, **16×16 2D blocks**, RNE |
| dgrad | `dX = dY W` | dY, 1×16 blocks, **stochastic rounding** | W^T, same 2D blocks |
| wgrad | `dW = dY^T X` | dY^T, **RHT** along tokens, 1×16 blocks, SR | X^T, same RHT, 1×16 blocks |

- **2D weight blocks:** with 16×16 tiles, `quantize(W)` and `quantize(W^T)` represent exactly the
  same values, so fprop and dgrad see the same weights (tested in `test_block_2d_is_transpose_invariant`).
- **Stochastic rounding on gradients:** it makes the 4-bit gradient an unbiased estimator. Tests
  check that the error of the SR average falls as more samples are averaged.
- **RHT on the wgrad operands:** it spreads outliers across each 16-element block before
  quantization. Because the transform is orthogonal and applied to both operands along the
  contraction dimension, `(A·M)(B·M)^T = A·B^T` holds exactly in exact arithmetic.
- **Kept in BF16:** embeddings, the LM head, LayerNorms, attention and the last N transformer blocks.

All three GEMMs run on a **single "TN" kernel** (`D = A·B^T`, with both operands quantized along
the contraction dimension). This works because the quantizer can emit transposed outputs directly.

## 3. Quantizers (`csrc/quantize_dual.cu`, `csrc/quantize.cu`)

Every NVFP4 quantization makes two passes: an amax reduction, then the quantization itself. Global
amax scaling needs the whole tensor's maximum before any block can be scaled.

### 3.1 The fused dual quantizer (used by `NVFP4Linear`)

Each training operand is needed in two layouts: X for fprop and RHT(X^T) for wgrad, W for fprop and
W^T for dgrad, dY for dgrad and RHT(dY^T) for wgrad. `quantize_dual` produces both from **one read**:

```
quantize_dual(X [R, C]) -> ( Q(X) along C            [R, C/2]          row output
                             Q(RHT(pad(X)^T)) along R [C, R_pad/2]      col output )
```

- **Options:** each output has its own stochastic-rounding seed. The col output can take the RHT and
  is zero-padded to a multiple of `col_pad` tokens (128 for the wgrad GEMM). With 16×16 2D blocks
  (weights), the col output is the exact transpose of the row output.
- **Tiling:** a CTA stages a 64×64 tile of X in shared memory with coalesced 16-byte loads. The row
  phase then has thread (b, r) quantize the block (row r, block column b). The col phase has thread
  (b, c) quantize the block (column c, block row b).
- **Conflict-free shared memory:** both phases read shared memory without bank conflicts (row stride
  65 floats). In both, a warp's lanes are 32 consecutive rows or columns, so a 16×16 tile's scale is a
  4-step `__shfl_xor` max.
- **Staged stores:** each phase stages its codes and scales in shared memory and writes whole 32-byte
  output rows with 16-byte stores. Writing directly from the quantizing threads meant 8-byte and
  1-byte stores scattered over 32 rows per warp instruction, with four warps completing each sector.
  That alone made each phase 2.3× slower.
- **Amax:** it is a plain vectorized max, unless the col output uses the RHT. Then one tile kernel
  computes both amax(X) and amax(RHT(X^T)). Both amax kernels are persistent (about 4 CTAs per SM)
  and use **one `atomicMax` per CTA**. Same-address atomics serialize at the L2, and one per warp over
  thousands of CTAs cost more than reading the tensor.
- **Bit-exact:** each output is bit-identical to `reference.quantize_dual`, and so to two separate
  single-output calls (`tests/test_quantize_dual_cuda.py`).

`NVFP4Linear` therefore quantizes X in the forward pass in both layouts. It saves only the
**NVFP4 copies of X^T and W^T** for backward. That is about 3.5× less than the BF16 activation
v0.1 saved, and backward no longer re-reads it.

### 3.2 Arithmetic: fast and bit-exact

The quantize pass was ALU-bound with stochastic rounding. Every change below keeps the output
bit-identical to the reference:

- **Round-to-nearest-even uses the hardware convert** `cvt.rn.satfinite.e2m1x2.f32`, available on
  sm_100a/sm_120a. One instruction converts two values. The hardware returns −0 for negative values
  that round to zero, so a 3-instruction SWAR fix-up (`clear_negative_zeros`) clears the sign of
  zero-magnitude nibbles, matching the reference.
- **Stochastic rounding** has no hardware path on sm_120a: `ptxas` rejects `cvt.rs` (".rs not
  supported on sm_120a"). It is done in software with no division and no int→float conversion:
  - the 24-bit uniform `(h >> 8) · 2^-24` is assembled from mantissa bits;
  - the lower grid point is `a` with its mantissa truncated to 1 bit (for `a ≥ 1`);
  - the step's reciprocal comes from the exponent bits, a power of two, so `(a − lo) / step` is exact
    as a multiply;
  - the chosen grid value is then converted exactly by the hardware convert.

  Together these made the SR quantize pass about 1.5× faster than v0.1's version.
- **The RHT sign mask** (16 bits, derived from the seed) is computed once on the host. Sign flips are
  XORs of the float sign bit, identical to multiplying by −1.
- **Explicit rounding:** every remaining operation uses an explicit IEEE intrinsic (`__fdiv_rn`,
  `__fmul_rn`, ...), and the Hadamard transform is a butterfly in the same order as the reference.
- **Reference fix (v0.2):** in PyTorch on CUDA, `tensor / python_scalar` is computed as
  `tensor * (1/scalar)`, which is not IEEE division. The reference's block-scale target
  `amax_b / 6 · genc` used it, and it differs at E4M3 rounding ties. For example,
  6.25 / 6 · 192 = 200 is exactly the midpoint of 192 and 208. The reference now uses true division,
  so CPU and CUDA references and the kernels agree. `test_scale_rounding_tie` and
  `test_block_scale_uses_ieee_division` cover it. In PyTorch, `2688.0 / tensor` is likewise
  `reciprocal(tensor) * 2688`, so `global_scales` uses `torch.div`.
- **Stochastic rounding is reproducible:** it uses a counter-based hash (`lowbias32`) of
  (seed, element index), so SR is seedable and identical between CPU and GPU.

### 3.3 The single-output quantizer

`quantize()` (`csrc/quantize.cu`) is the general single-output kernel: rowwise, transposed or 2D, with
optional RHT and SR. It shares the per-block math above (`quantize16` in `nvfp4_common.cuh`). One
thread owns one 16-element block. In rowwise 1D mode a warp covers 32 consecutive blocks of a row
with 16-byte loads. Otherwise a warp covers 32 consecutive rows of one block column.

## 4. GEMM (`csrc/gemm.cu`)

| Parameter | Value |
|---|---|
| CTA tile | 128 × 128 × 128 (K in elements, i.e. 64 bytes per row) |
| Warps | 8: 2 along M × 4 along N, each computing a 64×32 output tile |
| MMAs per warp per 64-wide K-step | 16 (4 m16 tiles × 4 n8 tiles) |
| Pipeline | 4-stage `cp.async`, 18 KB per stage = 72 KB shared memory |
| Registers | 150 per thread, no spills (one 256-thread CTA per SM) |
| Epilogue | `alpha = gdec_A · gdec_B` applied in fp32, then bf16 or fp32 store, or `D += alpha·acc` into an existing fp32 tensor |

**Accumulating epilogue.** `gemm(..., out=g, accumulate=True)` adds into `g`. `NVFP4Linear(...,
fuse_wgrad_accumulation=True)` uses it to accumulate weight gradients across micro-batches in place.
The add is two separate IEEE ops (no FMA contraction), so the result is bit-identical to autograd's
`grad + dW`. The old values for one m16 row group are loaded together before any store. A load can't
be moved past a store to the same array, so a load-add-store loop serialized 32 memory latencies per
warp: with it, the fused version was slower than the separate add it replaced.

**Fragment layouts.** These come from CUTLASS `MMA_Traits<SM120_16x8x64_TN_VS>` and were confirmed
on the GPU by the GEMM tests. `g = lane / 4`, `t = lane % 4`.

- **A** (16×64, 4-bit): `a0 = A[g][8t .. 8t+7]`, `a1 = A[g+8][8t ..]`, `a2 = A[g][32+8t ..]`,
  `a3 = A[g+8][32+8t ..]`, eight nibbles per register with the lowest nibble first.
  - This is byte-for-byte the int8 `m16n8k32` layout, so a single `ldmatrix.x4` loads it.
- **B** (64×8, stored as N×K): `b0 = B[n=g][k=8t ..]`, `b1 = B[g][32+8t ..]`.
  - One `ldmatrix.x4` covers two n8 tiles.
- **C/D**: the usual `m16n8` f32 layout. `c0, c1` are at (g, 2t..2t+1) and `c2, c3` at (g+8, 2t..2t+1).
- **Scale factors** (`scale_vec::4X`, selectors `bid = tid = 0`):
  - Each lane passes a 32-bit register whose byte j is the UE4M3 scale of K-block j of the current
    64-wide K step.
  - For A, the register is for row `g + 8·(lane & 1)` of the m16 tile. For B, it is for column `g`.
  - `tests/test_gemm_cuda.py::test_varied_block_scales` checks that every scale byte reaches the
    right block. It uses per-block magnitudes spanning six orders of magnitude.

**Shared-memory swizzle.** Rows are 64 bytes, made of four 16-byte chunks. The swizzle is
`chunk ^ ((row >> 1) & 3)`, which puts the eight row addresses of every `ldmatrix` phase on
distinct banks.

**Bounds.** `M` and `N` can be any size: loads use zero-filled `cp.async` and stores are guarded.
`K` must be a multiple of 128. `NVFP4Linear` zero-pads the token dimension for wgrad.

## 5. Verification

| Layer | Test | Criterion |
|---|---|---|
| Format spec | `tests/test_reference.py` (CPU, in CI) | RNE ties, saturation, SNR, 2D transpose invariance, RHT orthogonality, SR unbiasedness |
| Quantizer | `tests/test_quantize_cuda.py` | **bit-exact** vs reference: bf16/fp32, transpose, 2D, SR, RHT, edge values, every E2M1 rounding tie, E4M3 scale ties, −0 |
| Dual quantizer | `tests/test_quantize_dual_cuda.py` | **bit-exact** vs `reference.quantize_dual` for all 6 configurations × 7 shapes × 2 dtypes; == two separate calls; 2D col == exact transpose |
| GEMM | `tests/test_gemm_cuda.py` | vs fp64 reference: max error ≤ 1e-5 · max\|ref\|, ragged shapes, extreme scales |
| Layer | `tests/test_linear.py` | CUDA == emulated recipe (same seeds); cosine with full precision > 0.97–0.98; SR noise averages out; only NVFP4 tensors saved for backward (exact byte count); no-grad / frozen-weight / no-input-grad paths; fused wgrad accumulation == autograd bit for bit |
| Hardware ISA | CI `cuda-compile` job | `OMMA.SF.16864...UE4M3.4X` present in SASS |
| End to end | `experiments/train_gpt.py` | loss curves vs BF16 (see [RESULTS.md](RESULTS.md)) |

Every benchmark checks correctness against the reference before it times anything.

## 6. Known limitations and next steps

Done in v0.2: a single fused read per operand (§3.1), NVFP4-saved activations, and in-place wgrad
accumulation (opt-in).

1. **Quantization is not fused into the producing op.** Each operand still costs an amax pass plus a
   quantize pass over a tensor that the preceding LayerNorm, GELU or attention just wrote. Fusing
   amax and quantization into those producers would remove most of the remaining ~0.6 ms per block
   step at d=768.
2. **Stochastic rounding is ALU-bound.** It needs two 32-bit hashes per element, because sm_120a has
   no `cvt.rs`. A cheaper generator, such as one hash per two elements, would need a spec change.
3. **Large-K GEMM efficiency.** The kernel reaches about 85% of cuBLASLt NVFP4 on long-K shapes, and
   beats it on short K. Candidates: 128×256 tiles, warp-specialized producer warps, stream-K.
4. **wgrad with a small M×N** has no split-K yet. At d=768, the 768×768 projection wgrad fills only
   36 CTAs on 82 SMs.
5. **Weight quantization is redone per micro-batch.** It could be cached per optimizer step, but a
   safe invalidation rule is needed: `param.data` writes don't bump `_version`.
6. **Platform.** Linux/WSL2 only. The native Windows build hasn't been tried.
