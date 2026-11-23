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
- **Kept in higher precision:** embeddings, the LM head, attention and the last N transformer blocks
  stay BF16; LayerNorm and GELU run in fp32 (inside the quantizers when fused, §3.4).

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
- **The RHT sign mask** (16 bits, derived from the seed) is computed once on the host for an integer
  seed. For a seed held in GPU memory (what the layers use, §3.5) each warp builds it with one hash per
  lane and a ballot. Sign flips are XORs of the float sign bit, identical to multiplying by −1.
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

### 3.4 Producer fusion: LayerNorm and GELU inside the quantizer (v0.3)

In a pre-LN block the NVFP4 operands are produced by elementwise ops: qkv and fc1 quantize
`LayerNorm(x)`, fc2 quantizes `gelu(h)`, and fc1's backward quantizes `dY · gelu'(h)`. Unfused, each
of these is a separate kernel that writes its output (the LayerNorm output in fp32 under autocast,
plus a bf16 copy) only for the quantizer to read it twice (amax, quantize).

The dual quantizer's tile loader can instead apply the producer op itself (`Pro` in
`quantize_dual.cu`), in both the amax pass and the quantize pass:

| Op | Reads | Computes per element (fp32, each op IEEE-rounded) |
|---|---|---|
| LayerNorm | x, mean[r], rstd[r], γ, β | `((x − mean) · rstd) · γ + β` |
| GELU (tanh) | h | `(x · 0.5) · (tanh((x + x³ · κ) · β) + 1)` |
| GELU backward | dY, h | `dY · gelu'(h)` |

- **LayerNorm statistics** come from a small kernel (`ln_stats_kernel`: one warp per row, two passes,
  deterministic shuffle-tree sums). The LayerNorm backward then runs PyTorch's
  `native_layer_norm_backward` with these statistics, so forward and backward agree.
- **Zero padding stays zero:** the padded token rows of the wgrad operand are not passed through the
  op (LayerNorm of a zero row would be β).
- **Bit-exact:** every op is written one rounded operation at a time, without FMA contraction, and
  `nvfp4/reference.py` mirrors it with separate torch ops. Run on the GPU, `torch.tanh` is the same
  libdevice `tanhf`, so `tests/test_fused_producers_cuda.py` checks bit-exactness against reference
  op + `reference.quantize_dual`.
- **What changes numerically:** the fused path quantizes the fp32 LayerNorm/GELU outputs directly
  instead of first rounding them to bf16, and uses its own LayerNorm statistics.

`nvfp4/fused.py` builds the two pre-LN sub-blocks on top: `layernorm_linear` (LN → qkv) and
`layernorm_mlp` (LN → fc1 → GELU → fc2), each one autograd Function. They save what the unfused graph
saves (x, the statistics, h, the NVFP4 transposed operands). One subtlety: everything a single
`backward()` references stays alive until it returns, unlike between separate autograd nodes, so the
functions drop each temporary as soon as it is consumed and the LayerNorm-gradient GEMM writes fp32
directly. Without that, peak memory of the d=768 block rose 20%; with it, it is unchanged.

### 3.5 Seeds on the GPU, and what that buys (v0.5)

Each layer call draws its stochastic-rounding and RHT seeds as one small int64 tensor with
`torch.randint(..., device="cuda")`: three per linear (dY row SR, dY^T SR, RHT), six per fused MLP. The
quantizers take each seed either as an integer kernel argument or as a pointer to such a tensor
(`Params::*_seed_ptr`). With pointers:

- **No host synchronization** anywhere in a training step (v0.4 drew seeds with `.item()` on the CPU
  generator; `tests/test_integration_cuda.py` runs a step under `torch.cuda.set_sync_debug_mode("error")`).
- **CUDA graphs** can capture whole training steps: PyTorch's CUDA generator is graph-safe, so every
  replay draws new seeds (v0.4 would have frozen them into the graph).
- **torch.compile** can trace the layers (the random draw is an ordinary op), and with inductor's
  `fallback_random` the compiled step is bit-identical to eager.
- **Determinism** is unchanged in kind: `torch.manual_seed` fixes the CUDA generator, and activation
  checkpointing restores its state before recomputation.

In the kernels, threads 0–2 of each CTA load the three seeds at entry and park them in shared memory
after issuing their tile loads; the barrier that follows the tile load publishes them, and each warp
builds the RHT sign mask with one hash per lane. Per-kernel GPU time equals the integer-seed path. (A
first version that hashed all sixteen mask bits in every thread was compiled into predicated code that
ran even without device seeds and made the quantizers ~10% slower.) A whole block step is within ~1%
of v0.4 (A/B in one session: +0.5–1.5% at d = 768–2048, from the extra `randint` launches and Python
work per layer).

**torch.compile without graph breaks.** `nvfp4/_library.py` registers every kernel entry point as a
`torch.library` custom op with a fake (shape-only) implementation. Eager calls go straight to the
pybind11 function (no dispatcher overhead); only under `torch.compiler.is_compiling()` do the layers
call `torch.ops.nvfp4.*`. Glue that inspects data pointers (16-byte alignment) lives inside the ops.

**Fused gradient accumulation that keeps autograd's contract.** v0.2's fused wgrad accumulation added
the new gradient into `weight.grad` inside the GEMM and returned `None` to autograd, so the weight's
AccumulateGrad node never ran: per-parameter hooks did not fire, which DDP's bucketing relies on. Now
the backward adds into `.grad`, detaches `.grad` from the parameter, and returns that same tensor.
AccumulateGrad finds no `.grad` and adopts the returned tensor without a copy (its "steal" path), and
every hook fires. The backward only does this when the parameter is a leaf whose AccumulateGrad node
this backward pass will run (`torch._C._will_engine_execute_node`); `torch.autograd.grad`, non-fp32
gradients and compiled graphs take the ordinary path. Tests check the in-place property (the
gradient's data pointer is unchanged across micro-batches) and DDP / FSDP2 equality.

**Outputs are not views.** The layers return `aten._unsafe_view` of their 2D result, as `F.linear`
does internally, instead of a reshape view. FSDP2 warns about modules that return views, because an
in-place op on one would bypass its pre-backward hook.

## 4. GEMM (`csrc/gemm.cu`, `csrc/gemm_tma.cu`, `csrc/gemm_common.cuh`)

Two main loops share one epilogue. Both use 128×128 CTA tiles and 8 MMA warps (2 along M × 4 along N,
each computing a 64×32 output tile, 16 MMAs per 64-wide K step), one CTA per SM.

| | TMA loop (`gemm_tma.cu`, default for K ≥ 1024) | cp.async loop (`gemm.cu`) |
|---|---|---|
| K tile | 256 (128 bytes per row) | 128 (64 bytes per row) |
| Loads | TMA (`cp.async.bulk.tensor`), issued by one thread; full/empty mbarriers | every thread issues `cp.async` |
| Buffers | 2 × 32 KB A/B stages + 2 × 16 KB scale buffers = 96 KB | 4 × 18 KB stages = 72 KB |
| Swizzle | TMA 128B for A/B, 64B for scales | XOR `chunk ^ ((row >> 1) & 3)` |
| Registers | 220, no spills | 180, no spills |
| Requires | K % 256 == 0, 16-byte aligned operands | K % 128 == 0 |

Below K = 1024 the cp.async loop is used: the 2-stage TMA loop needs a few k-tiles to reach steady
state (8192×3072×768: 273 vs 268 TFLOPS).

**Epilogue** (`finish_tile`): `alpha = gdec_A · gdec_B` in fp32, then a bf16 or fp32 store, or
`D += alpha·acc` into an existing fp32 tensor (`accumulate=True`, used by fused wgrad accumulation).
The add is two separate IEEE ops (no FMA contraction), so it is bit-identical to autograd's
`grad + dW`. The old values of one m16 row group are loaded together before any store: a
load-add-store loop serialized 32 memory latencies per warp and made the fused version slower than
the separate add it replaced.

### 4.1 Closing the gap to cuBLASLt (v0.3)

v0.2's cp.async loop reached about 86% of cuBLASLt's NVFP4 kernel on long-K shapes. With GPU
performance counters (Nsight Compute, at its locked ~1.05 GHz clock, so power limits don't
interfere), 8192³ showed:

| Kernel | Time | Tensor pipe active | L2 requests | Main stalls |
|---|---|---|---|---|
| cuBLASLt (CUTLASS `128x128x256`, 384 threads) | 3.4–3.5 ms | 90% | 39 M | math-pipe throttle, sleeping |
| v0.2 cp.async loop | 5.4 ms | 56% | 116 M | MIO throttle (shared-memory instruction queue), wait |
| TMA loop, first version | 5.3 ms | 57% | 74 M | branch resolving (spinning waits), long scoreboard |
| + per-lane swizzled addresses, compile-time stage index, no by-reference lambdas | 4.85 ms | 62% | | |
| *experiment: scale loads disabled (wrong results)* | *4.05 ms* | *77%* | | |
| **+ scales loaded 4 k-tiles at a time (final)** | **4.4–4.55 ms** | **69%** | **48 M** | wait |

What mattered, in order:

1. **TMA instead of per-thread copies.** The cp.async loop issued 5 copy instructions per thread per
   k-tile, which saturated the MIO queue that `ldmatrix` also needs.
2. **Scale-factor traffic.** One k-tile's scales are a 16-byte slice of each of 128 rows (the scale
   tensor is row-major `[rows, K/16]`), so their TMA boxes cost as many L2 requests as the 16 KB of
   data and fetch half-empty sectors. Disabling them (an experiment with wrong results) recovered 18%
   of the kernel time. Loading 64-byte rows (4 k-tiles) into separate double-buffered scale buffers
   with their own mbarrier cuts those requests 4×.
3. **Register budget.** A ninth (producer) warp puts 3 warps on one scheduler, which caps every
   thread at 168 registers (3 × 32 × R ≤ 16K per scheduler). The MMA warps then spilled, and ptxas
   moved each k-step's `ldmatrix` next to its MMAs, undoing the fragment double-buffering.
   `setmaxnreg` would rebalance this (it compiles for sm_120a), but ptxas drops it ("ignored to
   maintain compatibility into extern call") because taking the address of a `__grid_constant__`
   tensor map inserts a call. So there is no producer warp: thread 0 issues the TMA copies once a
   stage's empty barrier completes, and the 8 MMA warps get up to 255 registers.
4. **Local-memory traffic.** Lambdas capturing locals by reference forced those locals onto the
   stack (`LDL` in the main loop); the loop now uses plain functions and precomputed per-lane
   addresses, with the stage index a template parameter.
5. **Waiting warps sleep:** `mbarrier.try_wait` with a suspend-time hint, not a spin loop.

**Robustness.** Lane 0 of warp 0 refills stages, so it can wait on an empty barrier while the rest
of its warp moves on to `ldmatrix` / `mma.sync`, which are `.aligned` (the whole warp must execute
them together). Independent thread scheduling doesn't guarantee reconvergence after such a branch,
so an explicit `__syncwarp()` follows every divergent refill and barrier arrival. Without it, a
d=2048 training run hung after ~600 steps (surfacing in WSL as "CUDA error: unknown error", i.e. a
display-driver watchdog reset). In addition, every barrier wait is bounded: after 1 s (below the
2 s Windows watchdog) it executes `trap`, so a pipeline bug fails as an ordinary CUDA error instead
of a driver reset.

Measured the same and dropped: four stages of 128-deep k-tiles in the same 96 KB (4.44 vs 4.55 ms),
releasing a stage right after its last `ldmatrix` instead of after its MMAs, and B-reuse or
serpentine MMA orders. The remaining locked-clock gap (69% vs 90% tensor-pipe activity with the same
`ldmatrix` count as cuBLASLt, so the same warp tiling) is in instruction scheduling: the top stall
is "wait", fixed-latency dependencies between tensor instructions.

**At real clocks the gap is closed.** Training runs at the 175 W board limit, where kernels compete
for power rather than issue slots. There the TMA loop measures 404–433 TFLOPS on long-K shapes, equal
to or above cuBLASLt (401–426), and it stays far ahead on short K (RESULTS.md §1).

### 4.2 Split-K for small output grids

A wgrad GEMM with a small M×N, such as the 768×768 attention projection at d=768 (36 CTA tiles on 82
SMs), leaves most of the GPU idle. With `splits = S`, CTA z covers k-tiles
`[KT·z/S, KT·(z+1)/S)` and writes its fp32 partial to a workspace; the **last CTA of each tile to
arrive** (atomic counter) sums all S partials in split order 0..S−1 and runs the normal epilogue.

- **Deterministic:** the sum order doesn't depend on arrival order. Unsplit, the TMA and cp.async
  loops are bit-identical to each other; split results differ from unsplit only by fp32 reassociation.
- **Accumulate stays exact:** `old + alpha · Σ partials` uses the same ops as the unsplit epilogue, so
  fused wgrad accumulation remains bit-identical to autograd.
- **When:** only for grids that leave more than half the SMs idle, into as many splits as still fit
  in one wave (≥ 8 k-tiles each). On this power-limited laptop GPU a partly idle GPU clocks its busy
  SMs higher (36 CTAs reached 6.4 TFLOPS per SM vs about 5.0 with all SMs busy), so splitting into a
  second wave never paid off. For 768×768×8192, two splits cut the kernel from 41.7 to 37.5 µs
  under the profiler and lift it from 212 to 266 TFLOPS in the benchmark (RESULTS.md §1); in the
  d=768 block step that is about 1% of the step.
- `NVFP4_GEMM_SPLITS=1` disables automatic splitting (for ablations).

### 4.3 Fragment layouts, swizzles, bounds

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

**Shared-memory swizzles.** In both loops the swizzle term of every row a lane addresses is a
per-lane constant, so the eight row addresses of every `ldmatrix` phase land on distinct banks:
`chunk ^ (row & 7)` for 128-byte rows (TMA 128B swizzle) and `chunk ^ ((row >> 1) & 3)` for 64-byte
rows.

**Bounds.** `M` and `N` can be any size: TMA zero-fills out-of-bounds rows, the cp.async loop uses
zero-filled copies, and stores are guarded. `K` must be a multiple of 128 (256 for the TMA loop).
`NVFP4Linear` zero-pads the token dimension for wgrad.

## 5. Verification

| Layer | Test | Criterion |
|---|---|---|
| Format spec | `tests/test_reference.py` (CPU, in CI) | RNE ties, saturation, SNR, 2D transpose invariance, RHT orthogonality, SR unbiasedness |
| Quantizer | `tests/test_quantize_cuda.py` | **bit-exact** vs reference: bf16/fp32, transpose, 2D, SR, RHT, edge values, every E2M1 rounding tie, E4M3 scale ties, −0 |
| Dual quantizer | `tests/test_quantize_dual_cuda.py` | **bit-exact** vs `reference.quantize_dual` for all 6 configurations × 7 shapes × 2 dtypes; == two separate calls; 2D col == exact transpose |
| GEMM | `tests/test_gemm_cuda.py` | vs fp64 reference: max error ≤ 1e-5 · max\|ref\|, ragged shapes, extreme scales; TMA loop == cp.async loop bit for bit; split-K deterministic across runs, accumulate == add |
| Fused producers | `tests/test_fused_producers_cuda.py` | **bit-exact** vs reference LayerNorm / GELU / GELU-backward + `quantize_dual` (4 configurations × 5 shapes × 2 dtypes); LayerNorm statistics vs fp64 |
| Fused sub-blocks | `tests/test_fused_modules.py` | forward == explicit op composition bit for bit; every gradient's cosine with fp32 ≥ the unfused NVFP4 path's − 0.01; seeded determinism; partial gradients; fused wgrad accumulation |
| Layer | `tests/test_linear.py` | CUDA == emulated recipe (same seeds); cosine with full precision > 0.97–0.98; SR noise averages out; only NVFP4 tensors saved for backward (exact byte count); no-grad / frozen-weight / no-input-grad paths; fused wgrad accumulation == autograd bit for bit |
| Robustness | `tests/test_robustness_cuda.py` | NaN / ±inf anywhere → non-finite outputs (quantize, dual, fused producers, GEMM, layers, gradients); clear errors on misuse; any leading shape incl. empty; fp32/bf16/fp16 and non-contiguous inputs; bf16 weights; state dict ↔ `nn.Linear`; activation checkpointing bit-exact; `torch.compile` == eager; CUDA-graph replay == eager; concurrent streams with split-K; DDP |
| Integration | `tests/test_integration_cuda.py` | GPU seed tensors == integer seeds bit for bit (all four quantizers); no host sync in a training step; `torch.compile(fullgraph=True)` with no graph breaks, bit-identical to eager; CUDA-graph capture of fwd+bwd (fresh seeds per replay, noise averages out) and of a full Adam step (loss falls); fused accumulation in place with hooks firing, and under `torch.autograd.grad`; outputs not views; **two-process DDP (plain and bucket views) and FSDP2** over gloo, fused and unfused, bit-exact vs a single-process reference |
| Memory / race safety | `tools/sanitize.sh` | compute-sanitizer memcheck, racecheck, synccheck, initcheck over every kernel variant (ragged shapes, TMA and cp.async loops, split-K, accumulate, all producer ops, GPU-resident seeds, a layer forward + backward): 0 findings |
| Long-run stability | `tools/soak.py` | 8 minutes: 18,401 fused + unfused training iterations, 3,682 bit-exact gradient replays, 92,005 GEMMs re-checked bit for bit, no non-finite values, no hang |
| Packaging | wheel built and installed into an isolated directory | imports and trains from outside the repository |
| Hardware ISA | CI `cuda-compile` job | `OMMA.SF.16864...UE4M3.4X` present in SASS |
| End to end | `experiments/train_gpt.py` | loss curves vs BF16 (see [RESULTS.md](RESULTS.md)) |

Every benchmark checks correctness against the reference before it times anything.

## 6. Known limitations and next steps

Done in v0.5: GPU-resident seeds, no graph breaks under `torch.compile`, CUDA-graph capture of
training steps, fused accumulation compatible with DDP / FSDP2, FSDP2 and multi-process tests, a
three-seed quality study ([RESULTS.md §4.1](RESULTS.md)).
Done in v0.4: NaN propagation (a NaN used to be quantized to finite values), per-device kernel
attributes, exact-architecture checks, sanitizer / soak / packaging verification (§5).
Done in v0.3: the TMA GEMM main loop (§4.1), split-K (§4.2) and LayerNorm/GELU fused into
quantization (§3.4). Done in v0.2: one fused read per operand, NVFP4-saved activations, in-place wgrad
accumulation.

1. **Locked-clock GEMM efficiency.** At equal clocks the TMA loop keeps the tensor pipe 69% busy vs
   cuBLASLt's 90%; the difference is instruction scheduling inside the MMA warps (§4.1). It doesn't
   show at this laptop's power limit, but it would on a GPU with more power headroom (a desktop RTX
   5090). A tile-contiguous scale layout, which is what cuBLASLt's "blocked" scale format provides,
   would remove the rest of the scale traffic; it changes the NVFP4 tensor format, so it was left out.
2. **Attention-side producers.** The attention output (the projection's input) and the residual adds
   are not fused; attention itself is PyTorch's BF16 SDPA.
3. **Stochastic rounding is ALU-bound.** It needs two 32-bit hashes per element, because sm_120a has
   no `cvt.rs`. A cheaper generator, such as one hash per two elements, would need a spec change.
4. **Weight quantization is redone per micro-batch** (about 0.5 ms per block step at d=2048). It
   could be cached per optimizer step, but a safe invalidation rule is needed: `param.data` writes
   don't bump `_version`.
5. **Fused sub-blocks** cover pre-LN blocks with bias-free linears and tanh-GELU.
6. **Framework coverage.** FSDP1 (`FullyShardedDataParallel`, flat parameters) is untested; FSDP2
   (`fully_shard`) is tested. Multi-process tests run two ranks on one GPU over gloo: the NCCL path
   across several GPUs is exercised only with one rank. `DTensor.full_tensor()` over gloo with CUDA
   tensors crashes in PyTorch 2.14 (also for plain `nn.Linear`); the tests gather shards on the CPU.
7. **Platform.** Linux/WSL2 only, one GPU model tested (RTX 5090 Laptop). The native Windows build
   hasn't been tried.
