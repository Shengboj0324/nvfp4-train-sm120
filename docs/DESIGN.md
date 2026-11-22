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

## 3. Quantizer (`csrc/quantize.cu`)

The quantizer makes two passes: an amax reduction, then the quantization itself. Global amax
scaling needs the whole tensor's maximum before any block can be scaled.

- **Work assignment:** one thread owns one 16-element block, and there are two lane mappings.
  - **Rowwise 1D:** a warp covers 32 consecutive blocks of one row, using 16-byte vector loads.
    It reaches about 725–760 GB/s on 8192×2048 bf16, against the laptop's 896 GB/s peak.
  - **Transposed and 2D:** a warp covers 32 consecutive rows of one block column. Reads are
    coalesced across input columns, and each 16×16 tile sits in 16 adjacent lanes, so its scale is
    a 4-step `__shfl_xor` max.
- **Bit-exact arithmetic:** every operation uses an explicit IEEE intrinsic (`__fdiv_rn`,
  `__fmul_rn`, ...), and the Hadamard transform is an explicit butterfly in the same order as the
  reference. Results match the reference bit for bit.
  - The reference had one subtle trap. In PyTorch, `2688.0 / tensor` is computed as
    `reciprocal(tensor) * 2688`, which rounds twice and is 1 ulp off IEEE division. The reference
    uses `torch.div` instead.
- **Stochastic rounding:** it uses a counter-based hash (`lowbias32`) of (seed, element index).
  This makes SR reproducible, seedable and identical between CPU and GPU.

## 4. GEMM (`csrc/gemm.cu`)

| Parameter | Value |
|---|---|
| CTA tile | 128 × 128 × 128 (K in elements, i.e. 64 bytes per row) |
| Warps | 8: 2 along M × 4 along N, each computing a 64×32 output tile |
| MMAs per warp per 64-wide K-step | 16 (4 m16 tiles × 4 n8 tiles) |
| Pipeline | 4-stage `cp.async`, 18 KB per stage = 72 KB shared memory |
| Registers | 130 per thread, no spills |
| Epilogue | `alpha = gdec_A · gdec_B` applied in fp32, then bf16 or fp32 store |

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
| Quantizer | `tests/test_quantize_cuda.py` | **bit-exact** vs reference: bf16/fp32, transpose, 2D, SR, RHT, edge values |
| GEMM | `tests/test_gemm_cuda.py` | vs fp64 reference: max error ≤ 1e-5 · max\|ref\|, ragged shapes, extreme scales |
| Layer | `tests/test_linear.py` | CUDA == emulated recipe (same seeds); cosine with full precision > 0.97–0.98; SR noise averages out |
| Hardware ISA | CI `cuda-compile` job | `OMMA.SF.16864...UE4M3.4X` present in SASS |
| End to end | `experiments/train_gpt.py` | loss curves vs BF16 (see [RESULTS.md](RESULTS.md)) |

Every benchmark checks correctness against the reference before it times anything.

## 6. Known limitations and next steps

1. **Quantization overhead.** Two passes per operand, with no fusion into the producing op. Fusing
   amax and quantization into the preceding LayerNorm, GELU or residual would remove most of it.
2. **Large-K GEMM efficiency.** The kernel reaches about 85% of cuBLASLt NVFP4 on long-K shapes,
   and beats it on short K. Candidates: 2 CTAs per SM (2–3 stages), 128×256 tiles, a TMA producer
   warp.
3. **wgrad with small M×N.** There is no split-K yet.
4. **Memory.** Backward re-quantizes the saved BF16 activations. Saving the NVFP4 copies instead
   would cut activation memory by about 3.5×.
5. **Weight quantization is redone per micro-batch.** It could be cached per optimizer step.
6. **Platform.** Linux/WSL2 only. The native Windows build hasn't been tried.
