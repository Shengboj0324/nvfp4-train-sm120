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
  - one 32-bit hash per **pair** of elements gives both their uniforms, 16 bits each
    (`(h & 0xFFFF) · 2^-16`, `(h >> 16) · 2^-16`), assembled from mantissa bits (v0.8; until v0.7
    every element hashed twice for a 24-bit uniform). The hash is most of SR's cost: SR now adds
    34–43% to the plain dual quantizer instead of 48–61%. 16 bits bound the rounding bias at 2^-17 of
    a grid step;
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
- **Stochastic rounding is reproducible:** it uses a counter-based hash (`lowbias32`, applied twice
  so the seed's high half is mixed in after the first round: different seeds do not give shifted
  copies of one stream) of (seed, element-pair index), so SR is seedable and identical between CPU and
  GPU. (The SR noise differs from v0.7's for the same seed.)

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
| GELU (tanh) | h, optional bias[c] | `(x · 0.5) · (tanh((x + x³ · κ) · β) + 1)`, x = h + bias |
| GELU backward | dY, h, optional bias[c] | `dY · gelu'(h + bias)` |
| GELU (exact, v0.7) | h, optional bias[c] | `(x · 0.5) · (1 + erf(x / √2))` and its backward `dY · (Φ + x · φ)` |
| RMSNorm (v0.7) | x, rstd[r], γ | `(x · rstd) · γ` |
| SwiGLU (v0.7) | g, u, optional biases b_g[c], b_u[c] (v0.8) | `silu(g) · u` on g + b_g, u + b_u; silu(g) = `g / (1 + exp(−g))` |
| SwiGLU backward (v0.7) | dY, g, u, optional biases | gate: `(dY · u) · (s · (1 + g · (1 − s)))`, s = sigmoid(g); up: `dY · silu(g)` |
| Multi-part input (v0.8) | 1–3 tensors [R, C/n] | their concatenation along C, read in place (the q, k, v gradients) |

**Bias gradients without writing the gradient (v0.7).** fc1's bias sits before the GELU, so its
gradient is the column sum of the GELU input gradient, which the fused path never writes. The
quantize kernel already holds each 64×64 tile of it in shared memory; when asked, 64 threads sum the
tile's columns in a fixed order and write one row of per-tile partial sums ([tiles along R, C]),
which the host sums. Deterministic (no atomics), and the gradient costs one small extra write. v0.8
uses the same column sums for the SwiGLU gate / up biases and, from the plain output-gradient
quantizer, for every bias added in a GEMM epilogue (§3.6).

- **LayerNorm statistics** come from a small kernel (`ln_stats_kernel`: one warp per row, two passes,
  deterministic shuffle-tree sums). The backward uses the same statistics, so forward and backward
  agree (v0.3–v0.7: PyTorch's `native_layer_norm_backward`; v0.8: this library's kernel, §3.6).
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

**Llama-style blocks (v0.7).** `rmsnorm_swiglu_mlp` quantizes RMSNorm(x) once; that operand (and
its RHT-transposed copy for the weight gradients) feeds both the gate and the up GEMM. The SwiGLU
producer reads both projections and the backward runs the gate and up halves as two quantizer
calls whose RHT seed matches x^T's, and accumulates both dgrad GEMMs into one fp32 buffer. The
RMSNorm backward was PyTorch's fused kernel (`aten._fused_rms_norm_backward`) fed with our statistic
in v0.7, and is this library's norm-backward kernel since v0.8 (§3.6).

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

**Weight-quantization cache (v0.7).** Weights use deterministic 16×16 blocks, so their NVFP4 copies
can be reused while the weight is unchanged. The cache key is (optimizer-step counter, storage
address, autograd version counter, dtype, shape, device). The version counter alone is not enough:
measured on PyTorch 2.14, `AdamW(fused=True)` updates parameters in place *without* bumping it
(foreach and single-tensor optimizers do). So a global `register_optimizer_step_post_hook` counts
every `torch.optim` step, fused or not, including subclasses. What remains invisible, in-place writes
through `param.data`, is documented with an explicit `invalidate_weight_caches`. During CUDA-graph
capture the cache switches itself off for good, since replays change weights without running Python.

**Outputs are not views.** The layers return `aten._unsafe_view` of their 2D result, as `F.linear`
does internally, instead of a reshape view. FSDP2 warns about modules that return views, because an
in-place op on one would bypass its pre-backward hook.

### 3.6 Residual connections, biases and the q / k / v hand-off (v0.8)

With the producers fused, what remained around the GEMMs of a block was glue: the residual adds, the
sum of the two gradients of every residual-stream tensor, the bias adds, and the q / k / v plumbing.
In a d=4096 Llama block step (0.7 fused path, 39.4 ms of GPU time) the profiler attributed ~3.4 ms
(8.5%) to it: the stack of dq, dk, dv into the qkv gradient (0.85 ms) and the copy that made it
contiguous (0.64 ms), the two forward residual adds (0.79 ms), the two gradient sums (0.79 ms) and the
bf16 cast of the fp32 residual gradient for the bf16 branch (0.25 ms).

- **GEMM epilogue** (`Args::bias`, `Args::res`): the store computes `y = bf16(alpha · acc)`, adds
  `bias[col]` (rounding to bf16 when the layer outputs bf16) and then `res[row, col]`, writing the
  result in the residual's dtype. That is exactly `res + (gemm_bf16.to(dt) + bias.to(dt))` in PyTorch,
  bit for bit (`tests/test_epilogue_cuda.py`, every main loop and split-K). `NVFP4Linear` uses it for
  its bias and for `forward(x, residual=r)`; the MLPs use it for the down projection's bias and the
  residual (`residual=True`). A bias's gradient comes from the column sums of the output-gradient
  quantizer (§3.4), so the bias costs neither a forward nor a backward pass of its own.
- **Residual gradients summed in the norm backward.** In `x + mlp(norm(x))` the gradient of x is
  `dout + norm_backward(...)`. The MLP Function owns both paths, so its backward passes `dout` to the
  norm-backward kernel, which adds it while writing dx. For the attention half the residual add
  happens in o_proj's epilogue, outside the qkv function; `return_residual=True` therefore returns x
  itself as an extra output (autograd makes it a view, the "residual handle"). Passed as o_proj's
  residual, its gradient (the residual gradient, passed through as is) comes back to the qkv
  function's backward, which adds it in its norm backward the same way. No elementwise pass is left.
- **fp32 residual gradients are quantized as they are**, without the bf16 cast the unfused graph
  inserts before a bf16 branch (one pass less, and slightly more accurate).
- **q / k / v hand-off.** `chunks=3` returns q, k and v as views of the GEMM output. PyTorch's flash
  attention allocates its input gradients contiguous in (b, t, h, d) order, so after the view
  backwards the three gradients reach the qkv function as contiguous [tokens, d] tensors, and the
  output-gradient quantizer reads them in place as one [tokens, 3d] tensor (`quantize_dual` of a list,
  the `kParts` loader). The unfused path stacked them into a new tensor and copied it once more.
  (Indexing an array of part pointers in the kernel-parameter struct at run time made the compiler
  copy the struct to local memory in every plain quantizer: 72-byte stack frames and up to 80% slower
  row-only quantization. The multi-part loader is its own template variant with named pointers.)
- **The norm-backward kernel** (`csrc/norm.cu`, LayerNorm and RMSNorm) replaces PyTorch's:
  one CTA per 16 rows; each row (C ≤ 8192) is read once into registers (C/8 threads rounded to a warp,
  up to 256, 8 elements per chunk), the two row sums are warp-shuffle trees summed over warps in a
  fixed order, dx is written with the residual gradient added, and the thread's dw (and db) partial
  sums accumulate over the group's rows in registers, one [groups, C] partial row per CTA (summed on
  the host). Deterministic. At d=4096 an RMSNorm backward with dw and the residual gradient takes
  0.69 ms, where PyTorch's dx and dw kernels took 0.69 + 0.32 ms plus 0.4 ms for the separate add.
  Longer rows fall back to a two-pass warp kernel plus a column-sum kernel.
- **Post-norm blocks** (BERT's original layout: `LN(h + mlp(h))`) need the normalized tensor itself
  as the residual, so there is no norm to fuse into the MLP's input; `layernorm_mlp(h, None, ...)`
  (or `gelu_mlp`, `swiglu_mlp`) with `residual=True` fuses everything up to the user's LayerNorm.

## 4. GEMM (`csrc/gemm.cu`, `csrc/gemm_tma.cu`, `csrc/gemm_common.cuh`)

Two main loops share one epilogue. Both use 128×128 CTA tiles and 8 MMA warps (2 along M × 4 along N,
each computing a 64×32 output tile, 16 MMAs per 64-wide K step), one CTA per SM.

| | TMA loop (`gemm_tma.cu`) | cp.async loop (`gemm.cu`) |
|---|---|---|
| Used for | K ≥ 1024, or K ≥ 512 with ≥ 8 output tiles per SM | the rest |
| Grid | persistent (v0.8): one CTA per SM looping over output tiles (§4.2) | one CTA per tile |
| K tile | 256 (128 bytes per row) | 128 (64 bytes per row) |
| Loads | TMA (`cp.async.bulk.tensor`), issued by the last warp to release a stage; full/empty mbarriers | every thread issues `cp.async` |
| Buffers | 2 × 32 KB A/B stages + 2 × 16 KB scale buffers = 96 KB | 4 × 18 KB stages = 72 KB |
| Swizzle | TMA 128B for A/B, 64B for scales | XOR `chunk ^ ((row >> 1) & 3)` |
| Registers | 245, no spills | 180, no spills |
| Requires | K % 256 == 0, 16-byte aligned operands | K % 128 == 0 |

**Epilogue** (`finish_tile`): `alpha = gdec_A · gdec_B` in fp32, then a bf16 or fp32 store, the
bias / residual epilogue of §3.6, or `D += alpha·acc` into an existing fp32 tensor
(`accumulate=True`, used by fused wgrad accumulation). The add is two separate IEEE ops (no FMA
contraction), so it is bit-identical to autograd's `grad + dW`. All of a thread's old values are
loaded before any store (a load cannot be moved past a store to the same array, so a load-add-store
loop serializes the latencies), and the TMA loop starts a bulk L2 prefetch of the tile's old values
(or of the residual) when it starts the tile.

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
serpentine MMA orders. v0.3 left a locked-clock gap of 69% vs 90% tensor-pipe activity; §4.2 is what
v0.8 recovered of it.

**At real clocks the gap is closed.** Training runs at the 175 W board limit, where kernels compete
for power rather than issue slots. There the TMA loop measures 404–433 TFLOPS on long-K shapes, equal
to or above cuBLASLt (401–426), and it stays far ahead on short K (RESULTS.md §1).

### 4.2 Persistent tiles and early refills (v0.8)

Measured at locked clocks (8192×8192×K, Nsight Compute), v0.7's TMA loop took 0.41M + 0.586M cycles
per 1024 of K: **~8K cycles of overhead per output tile** (pipeline fill, CTA launch and barrier setup,
epilogue), 8% of the time at K = 8192 and 40% at K = 1024, on top of a main loop at ~70% of the tensor
pipe. cuBLASLt's NVFP4 kernel is persistent and warp-specialized (384 threads, 168 registers on
average, 88 KB of shared memory, one CTA per SM) and took 3.65M cycles at 8192³ to v0.7's 5.10M.

- **Persistent CTAs.** One CTA per SM loops over work units (an output tile, or one K split of it) in
  the order the hardware would have launched them. The data stages and scale buffers form one ring
  across a CTA's units, so while the warps run a tile's last k-steps and its epilogue, the next tile's
  first k-tiles and scales are already loading. (The first version re-decoded the unit — with a 64-bit
  division, i.e. a subroutine call — on every refill and was slower than v0.7; the cursors now keep
  the decoded unit and use 32-bit arithmetic.)
- **Refills start when a stage is free.** Stall sampling showed warps sleeping on the full barriers
  for 11.6% of the samples: only thread 0 refilled stages, and only after issuing its own MMAs for the
  k-tile. Now each warp counts its release of a stage with a shared-memory atomic, and the warp that
  releases it last issues the refill (after the empty-barrier wait, which then completes at once and
  orders the TMA writes after every warp's reads). The sequence of loads is fixed, so every warp keeps
  its own copy of the load cursors and advances it at the same points; nothing else is shared. (A
  shared cursor, handed from one last warp to the next through fences and the atomic, is correct under
  the PTX memory model but invisible to racecheck, which reported it.) Sleep fell to 6.8%.
- **The accumulating epilogue** lost 35–40% inside the unit loop, where loading the old gradient one
  row group at a time exposed its DRAM latency four times per tile. Loading all of a thread's old
  values first and starting a bulk L2 prefetch of the tile when the unit starts made it 3–28% faster
  than v0.7's.

Result at locked clocks, against v0.7: 17% fewer cycles at K = 1024, 15% at 2048, 8–9% from 4096 to
16384; tensor-pipe activity at 8192³ 64% → 71%. cuBLASLt still needs 21% fewer cycles there (3.65M vs
4.64M). The rest is per-warp issue: ptxas spaces each warp's OMMAs 16 cycles apart (an 11-cycle stall
plus a filler instruction; the filler's "wait" is 23% of the stall samples), and with two MMA warps
per scheduler any other instruction or stall leaves the tensor pipe idle. cuBLASLt's layout (one
producer and two consumer warpgroups, registers rebalanced with `setmaxnreg` to roughly 40 / 232)
would need `setmaxnreg`, which ptxas ignores here because taking the address of a `__grid_constant__`
tensor map inserts a call (§4.1); the descriptors would have to live in global memory instead. Left
for a GPU with power headroom, where it would show (RESULTS.md §1).

**Ablation (v0.9): bigger warp tiles do not fit.** The obvious way to space each warp's OMMAs further
apart without a second warp on the scheduler is a 64 × 64 warp tile (4 MMA warps, one per scheduler, 32
independent MMAs per k-step instead of 16, a quarter less shared-memory traffic per MMA). Built with
`NVFP4_NVCC_FLAGS=-DNVFP4_WARPS_N=2`: 128 fp32 accumulators + double-buffered fragments + the scales of a
k-tile need more than the 255 registers a thread can have, ptxas spills 1.3–1.6 KB per thread, and the
kernel runs at half speed (8192³ at locked clocks: 10.9 ms, or 8.8 ms with 128-deep k-tiles, vs 4.4 ms;
236–294 vs 457 TFLOPS at real clocks). `setmaxnreg` cannot raise a thread above 255 registers either, so
with `mma.sync` and fp32 accumulation the 64 × 32 warp tile is the largest that keeps its fragments
double-buffered; what cuBLASLt gains at locked clocks comes from instruction scheduling, not from a
larger tile. At real clocks the gap does not exist: the kernel is ahead of cuBLASLt on every
benchmarked shape on the desktop RTX 5090 (RESULTS.md §6).

### 4.3 Split-K for small output grids

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

### 4.4 Fragment layouts, swizzles, bounds

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
| Fused sub-blocks | `tests/test_fused_modules.py`, `tests/test_fused_llama.py` | GPT / BERT style (incl. biases, exact GELU) and Llama style (RMSNorm, SwiGLU; norm-module variants): forward == explicit op composition bit for bit; every gradient's cosine with fp32 ≥ the unfused NVFP4 path's − 0.01; seeded determinism; partial gradients (incl. bias-only); fused wgrad accumulation; compile |
| Epilogue, multi-part input, norm backward (v0.8) | `tests/test_epilogue_cuda.py` | bias / residual epilogue == `reference.linear_epilogue` bit for bit (TMA, cp.async, split-K; bf16 / fp32 outputs and residuals); multi-part quantization == quantizing the concatenation bit for bit (incl. device seeds), column sums == sums; SwiGLU biases bit-exact; norm backward == reference and PyTorch autograd for 1–4 register chunks and the long-row path, deterministic |
| Residual fusion (v0.8) | `tests/test_fused_residual.py` | Llama and GPT blocks with chunks + residual handles + fused residual adds: forward == the 0.7 composition bit for bit, gradients as close to fp32 as unfused NVFP4; chunk gradients bit-identical to the concatenated path; residual-handle gradient summed exactly; `NVFP4Linear(residual=)` bit-exact; norm-free MLPs; SwiGLU biases incl. partial / bias-only training; fused accumulation + cache bit-identical; compile (fullgraph) |
| Weight cache | `tests/test_weight_cache_cuda.py` (+ two-process DDP / FSDP2) | bit-identical to uncached over optimizer steps (fused / foreach AdamW, SGD); reused within a step; every update path noticed (and the documented `.data` exception); CUDA-graph capture bypasses it; checkpointing, compile, deepcopy / pickle |
| Layer | `tests/test_linear.py` | CUDA == emulated recipe (same seeds); cosine with full precision > 0.97–0.98; SR noise averages out; only NVFP4 tensors saved for backward (exact byte count); no-grad / frozen-weight / no-input-grad paths; fused wgrad accumulation == autograd bit for bit |
| Robustness | `tests/test_robustness_cuda.py` | NaN / ±inf anywhere → non-finite outputs (quantize, dual, fused producers, GEMM, layers, gradients); clear errors on misuse; any leading shape incl. empty; fp32/bf16/fp16 and non-contiguous inputs; bf16 weights; state dict ↔ `nn.Linear`; activation checkpointing bit-exact; `torch.compile` == eager; CUDA-graph replay == eager; concurrent streams with split-K; DDP |
| Integration | `tests/test_integration_cuda.py` | GPU seed tensors == integer seeds bit for bit (all four quantizers); no host sync in a training step; `torch.compile(fullgraph=True)` with no graph breaks, bit-identical to eager; CUDA-graph capture of fwd+bwd (fresh seeds per replay, noise averages out) and of a full Adam step (loss falls); fused accumulation in place with hooks firing, and under `torch.autograd.grad`; outputs not views; **two-process DDP (plain and bucket views), FSDP1 (flat and `use_orig_params`) and FSDP2** over gloo, fused and unfused, bit-exact vs a single-process reference; deterministic mode + flash attention reproduces a block's gradients bit for bit |
| 8-bit attention (v0.9) | `tests/test_attention_cuda.py` | operand quantizer **bit-exact** vs `reference.int8_token_quantize` / `mxfp8_quantize` (both layouts, σ order, scale layout, mean subtraction, padding, strided inputs, Delta); forward vs `reference.attention_8bit` (< 3%) and exact attention (< 5%); gradients vs exact attention (< 7%) for causal / full, GQA, N = 1…2048, D = 64 / 128, bf16 / fp16; backward bitwise deterministic; gradient layouts; NaN propagation; SDPA fallback; `torch.compile(fullgraph=True)` |
| Memory / race safety | `tools/sanitize.sh` | compute-sanitizer memcheck, racecheck, synccheck, initcheck over every kernel variant (ragged shapes, TMA and cp.async loops, persistent grids with more units than SMs, split-K, accumulate, the bias / residual epilogue, all producer ops, multi-part inputs, the norm backward, GPU-resident seeds, layer and v0.8 block forward + backward, 8-bit attention forward + backward): 0 findings |
| Long-run stability | `tools/soak.py` | 8 minutes (v0.7): 18,401 fused + unfused training iterations, 3,682 bit-exact gradient replays, 92,005 GEMMs re-checked bit for bit; 5 minutes (v0.8, residual-fused blocks): 14,474 iterations, 2,896 replays, 72,370 GEMMs; no non-finite values, no hang |
| Packaging | wheel built and installed into an isolated directory | imports and trains from outside the repository |
| Hardware ISA | CI `cuda-compile` job | `OMMA.SF.16864...UE4M3.4X` present in SASS |
| End to end | `experiments/train_gpt.py` | loss curves vs BF16 (see [RESULTS.md](RESULTS.md)) |

Every benchmark checks correctness against the reference before it times anything.

## 6. Known limitations and next steps

Done in v0.9: 8-bit attention (§7; item 2 of v0.8), the warp-tile ablation of the locked-clock gap
(§4.2, item 1).
Done in v0.8: residual adds, biases and the q / k / v hand-off fused (§3.6, item 2 of v0.7),
SwiGLU gate / up biases and norm-free MLPs for post-norm blocks (item 5), a persistent GEMM with early
refills that recovers about half of the locked-clock gap (§4.2, item 1), stochastic rounding at half
the hashing cost (item 3), and weight-cache hits under FSDP2 when the all-gathered weight returns to
the same address within a step (item 4).
Done in v0.7: weight-quantization cache (item 4 of v0.6), Llama-style fused blocks (RMSNorm,
SwiGLU), biases and exact GELU in the fused GPT blocks (item 5), two-GPU NCCL tests and a
multi-GPU session script (run once a multi-GPU machine is available).
Done in v0.6: native Windows build and tests, FSDP1, bitwise-reproducible training with flash
attention (deterministic mode), CI compile job on Windows.
Done in v0.5: GPU-resident seeds, no graph breaks under `torch.compile`, CUDA-graph capture of
training steps, fused accumulation compatible with DDP / FSDP2, FSDP2 and multi-process tests, a
three-seed quality study ([RESULTS.md §4.1](RESULTS.md)).
Done in v0.4: NaN propagation (a NaN used to be quantized to finite values), per-device kernel
attributes, exact-architecture checks, sanitizer / soak / packaging verification (§5).
Done in v0.3: the TMA GEMM main loop (§4.1), split-K (§4.3) and LayerNorm/GELU fused into
quantization (§3.4). Done in v0.2: one fused read per operand, NVFP4-saved activations, in-place wgrad
accumulation.

1. **Locked-clock GEMM efficiency.** At equal clocks the TMA loop keeps the tensor pipe 71% busy vs
   cuBLASLt's ~90% (v0.7: 64%); what remains is per-warp OMMA issue spacing with two MMA warps per
   scheduler (§4.2). In practice it does not show: on a desktop RTX 5090 at full power the kernel is
   1.03–1.71× cuBLASLt on every benchmarked shape, 1.03 PFLOPS at 8192³ (RESULTS.md §6). The larger warp
   tile that would space the OMMAs out does not fit in 255 registers (v0.9 ablation, §4.2).
2. **Attention** (v0.9: 8-bit, §7). The forward is 1.6–2.2× flash attention; in training, forward +
   backward gains only 1.14–1.18× at D = 128 once dQ is computed accurately, and at 4096 tokens the loss
   is 0.5% higher than with BF16 attention. Ways forward: an 8-bit forward with an exact BF16 backward
   (most of the speed of the forward, little of the error), V in INT8 per token (V in MXFP8 is the
   largest forward error), FlashAttention-4's polynomial exp2 on the FMA pipe, and a deterministic dQ
   without the recomputed products (ordered accumulation, as in FlashAttention-3's deterministic mode).
   Masks other than causal, packed sequences, dropout and head dims other than 64 / 128 fall back to
   SDPA.
3. **Stochastic rounding is still software** (sm_120a has no `cvt.rs`): one 32-bit hash per element
   pair, +34–43% on the plain dual quantizer.
4. **Weight cache limits.** In-place writes through `param.data` outside an optimizer step cannot be
   detected (`.data` deliberately gives the tensor a fresh autograd version counter); call
   `invalidate_weight_caches`. Under FSDP2 the cache hits only when the all-gathered weight lands at
   the same address within a step (the allocator's choice; correct either way).
5. **Fused sub-blocks** cover GPT / BERT / ViT-style and Llama-style blocks, pre-norm or (via the
   norm-free MLPs) post-norm, with biases anywhere. Mixture-of-experts layers (grouped GEMMs) are not
   covered.
6. **Framework coverage.** DDP, FSDP1 and FSDP2 are tested with two ranks on one GPU over gloo and
   (v0.8) with NCCL across two physical RTX 5090s, bit-identical to a single process. Data-parallel
   *speed* depends on the host: on the rented two-GPU machine all-reduce ran at 2 GB/s (no P2P), so DDP
   was communication-bound for BF16 and NVFP4 alike (RESULTS.md §6). `DTensor.full_tensor()` over gloo
   with CUDA tensors crashes in PyTorch 2.14 (also for plain `nn.Linear`); the tests gather shards on
   the CPU.
7. **Platform.** Linux, WSL2 and native Windows build and pass the tests on the RTX 5090 Laptop, and
   Linux on the desktop RTX 5090 (CUDA 12.8, PyTorch 2.11). The smaller RTX 50 parts (5080 / 5070 /
   5060) share the architecture but have fewer SMs; the tile / split-K heuristics were tuned on the
   82- and 170-SM parts (a rental attempt on 2026-11-24 found no RTX 5070 Ti / 5060 Ti / 5070 capacity
   and five RTX 5080 hosts whose containers never started). The attention kernels need nothing
   SM-count specific: two forward CTAs per SM fit the 100 KB of shared memory every sm_120 SM has. On
   Windows, inductor needs Triton, which PyTorch's Windows wheels don't include.

## 7. 8-bit attention (`csrc/attention.cu`, `nvfp4/attention.py`, v0.9)

With the linear layers in NVFP4, attention became the largest BF16 cost: in a fused NVFP4 Llama block
(d = 2048) PyTorch's flash attention took 13% of the training step at 512 tokens and 51% at 8192, running
at 31–88 TFLOPS. The tensor-core rates that bound what an attention kernel can do on this GPU (measured,
`mma.sync` throughput with every SM busy):

| Instruction (m16n8k·) | TFLOPS / TOPS | vs BF16 |
|---|---|---|
| BF16 / FP16, FP32 accumulate | 121 | 1× |
| FP16 / E4M3, FP16 accumulate; E4M3 unscaled, FP32 accumulate | 240 | 2× |
| **E4M3 block-scaled (`kind::mxf8f6f4`, UE8M0 per 32), FP32 accumulate** | **438** | 3.6× |
| **INT8, INT32 accumulate** | **473** | 3.9× |
| NVFP4 / MXFP4 block-scaled | 914 | 7.5× |

Consumer Blackwell runs FP8 with FP32 accumulation at full rate only on the block-scaled path, and
INT8 at full rate. Attention probabilities and score gradients span many binades, so FP4 is no option
for them; the design uses the two 8-bit paths, each where its number format fits:

- **Q K^T and dO V^T in INT8, one FP32 scale per token** (over the whole head dim). These products
  produce the scores and dP, whose absolute errors go straight into the softmax / dS. With operands
  scaled to their row maximum, INT8's 7 bits carry 4× less error than E4M3's 3-bit mantissa (emulated
  on random and peaked inputs: 0.8–1.9% vs 3.4–8.0% relative error in the output, the largest of all
  error sources). K is centred by its per-channel mean over the sequence first, as in SageAttention:
  a per-row constant in the scores, so the output is unchanged (and dK; dQ changes by
  Σ_j dS_ij · mean(K) = 0), while channel offsets typical of keys leave the quantization range.
- **P V, P^T dO, dS^T Q in MXFP8** (E4M3, UE8M0 scale per 32 elements of the reduction dim). P is
  quantized as e4m3(2^7 p) with p ≤ 2^1.5 relative to the running row maximum (a fixed scale, as in
  FlashAttention-3's FP8 path: values too small to represent are negligible next to the row maximum);
  dS with a per-(row, 32-block) scale from its amax across the quad.
- **dQ = dS K in BF16** (v0.9.1), against a mean-centred BF16 copy of K. Every row of dS sums to zero
  (softmax), so dQ is a small difference of large terms; dK = dS^T Q sums over queries, where no such
  constraint holds. On activations of a trained model at 4096 tokens, dQ in MXFP8 had 7–24% relative
  error (dS and K quantization each 5–18% alone, emulated), dK and dV 2–3%; in BF16 the product itself
  adds 0.5–1.4%. A second effect: Delta = rowsum(dO ∘ O) uses the 8-bit forward's O (~2% off), so the
  rows of dS = P (dP − Delta) sum to r_i ≠ 0, which dQ, a sum over all keys, amplified to another 4–14%.
  The dQ kernel therefore also accumulates r_i and B_i = Σ_j P_ij K_j (one cheap MXFP8 product; r is small,
  so B needs little precision) and returns dQ_i − r_i B_i, exactly the result of the consistent
  Delta' = rowsum(P dP) without a second pass over the keys. dQ error on those activations: 0.5–1.5%.

**Layouts.** Each operand is quantized once (`attn_quant_kernel`, one read of the BF16 tensor) into a
row layout (INT8 [N, D] + per-token scale) and, where a product reduces over the sequence, a col layout
(E4M3 [D, N] with UE8M0 scales per 32 tokens). The forward quantizes Q, K, V into the layouts both passes
need and saves them for backward (as much memory as the BF16 q, k, v SDPA saves); backward quantizes
dO and computes Delta = rowsum(dO ∘ O) in the same pass; K also gets a mean-centred BF16 copy for the dQ
product. P and dS never leave registers: the C fragment
of an m16n8 MMA gives thread t columns {2t, 2t+1} of each 8-column tile, while the A fragment of the
k32 8-bit MMA wants k-indices {4t..4t+3, 16+4t..16+4t+3}. The sequence order inside a 32-block is free as
long as the other operand follows it, so the col layout stores at position k the token
σ(k) = 16⌊k/16⌋ + 8⌊(k mod 4)/2⌋ + 2⌊(k mod 16)/4⌋ + (k mod 2) — the token whose value the thread holds
there. The col scales of a 64-token tile are stored so that each lane's are contiguous, and the MMA's
byte selector picks the right one (no per-MMA scale loads).

**Kernels.** Forward: CTAs of 128 queries, 4 warps of 32 rows, two CTAs per SM; each B fragment loaded
from shared memory feeds two MMAs (at 16 rows per warp the operand loads took about half the
shared-memory bandwidth the 8-bit MMAs leave), and the two CTAs' independent barriers let one's softmax
overlap the other's tensor work (8–20% over one 8-warp CTA). Online softmax in the log2 domain with a
lazily updated maximum (O is rescaled only when a row maximum grows by more than 1.5). The per-token
scales are applied in the FMA that already forms the exponent, and INT32 → FP32 uses the magic-number
add (an integer add on the full-rate pipe; I2F runs at a quarter of the FMA rate). Backward: two
kernels without atomics, so gradients are **bitwise deterministic**: one per 128 queries
(S, dP, dS, dQ in BF16, P K) and one per 128 keys (S^T, dP^T, dV, dK, looping over the query heads of a
GQA group, so dK / dV of shared heads are summed in registers). That costs three products more than
FlashAttention-2's atomic dQ (8 instead of 5, one of them BF16).

**Accuracy and convergence.** Against exact attention on random inputs: output 3.3%, gradients
4.5–5% relative error (BF16 flash attention: 0.2–0.3%); on a trained model's activations (GPT-124M, 4096
tokens): output 1.1–2.5%, dQ 0.5–1.5%, dK 2.6–3.9%, dV 2.3–3.0%. GPT-124M on TinyStories, 3000 steps
(RESULTS.md §7):
- at 512 tokens per sequence (12 heads of 64, seed 0, before the dQ fix): the same validation loss as
  BF16 attention (1.4892 vs 1.4892 with BF16 linears; 1.5095 vs 1.5073 with NVFP4 linears);
- at 4096 tokens (6 heads of 128, NVFP4 linears, seeds 0 and 1): **+0.52%** (1.6088 / 1.6033 vs 1.6027 /
  1.5928), down from +0.72% before the dQ fix. The rest is spread over the forward (V in MXFP8 is its
  largest term) and dK / dV; no single source dominates any more.

So at long context 8-bit attention is not free in quality: it adds about half the loss gap the NVFP4
linears themselves cost (+1.3%, §2 / RESULTS.md §4.1). Use it where its speed matters and validate.

**Speed** (RESULTS.md §7). Forward 1.6–2.2× SDPA from 4k tokens (D = 128; D = 64: 1.4–1.9×): the forward
is where 8-bit attention pays (inference, evaluation). Forward + backward 1.14–1.18× (D = 128) and about
even at D = 64: the BF16 dQ product and the correction cost the backward most of its 8-bit advantage
(before the fix: 1.4–1.5×, with the accuracy problem above). The softmax arithmetic, exponentials and
conversions that BF16 MMAs hide behind their own latency are exposed at 3.6× the MMA rate, and the 8-bit
kernels reach ~45% of the 8-bit tensor peak where FlashAttention-2 reaches ~75% of BF16's. A fused NVFP4
Llama block at d = 2048 trains 1.06× faster at 8192 tokens and 1.11× at 16384 with it (1.88× BF16);
GPT-124M at 4096 tokens trains at the same speed.

**Coverage.** Causal or full attention, GQA / MQA, any sequence length (padded internally to 128),
head dims 64 and 128, bf16 / fp16, any strides with a unit stride along the head dim (gradients are
returned in the inputs' layout when it is dense), `torch.compile` (custom ops). Other head dims,
attention masks, dropout and fp32 fall back to SDPA. NaN in an input row propagates to that row.
