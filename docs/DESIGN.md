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

### 2.1 Recipe options from the 2026 literature (v0.11)

The recipe above leaves GPT-124M +1.3% above BF16 in validation loss (RESULTS.md §4.1). The 2026 papers
attribute most of such gaps to forward-pass quantization error, and propose fixes
(docs/research/nvfp4-open-problems-2026-11/ surveys them). Four are implemented as process-wide switches
(`nvfp4.ops`, USAGE.md §3.5), bit-exact against `nvfp4/reference.py` like the rest of the quantizer, and
measured with a three-seed ablation harness (`experiments/run_ablation.sh`, paired by seed against the
baseline: same initialization, data order and rounding seeds, so seed noise largely cancels).

- **Four Over Six** (`set_four_six`, arXiv 2512.02010). E2M1 has no value between 4 and 6, so a block
  whose maximum maps to 6 represents values near 5 badly. Each round-to-nearest 16-block is quantized
  twice, with its maximum mapped to 6 and to 4, and keeps the candidate with the smaller squared error
  (summed in the same order as the reference, so ties resolve identically). Mapping to 4 needs block
  scales up to 448 · 6/4, so the tensor scale uses 256 instead of 448 as its E4M3 ceiling. Applies to
  all RTN outputs (X, W, and X^T, W^T); stochastically rounded gradients are unchanged.
- **MS-EDEN** (`set_ms_eden`, Quartet II, arXiv 2601.22813) for the wgrad operands. The RHT along the
  tokens is widened from 16 to 64 points: H64 = H4 ⊗ H16, i.e. the existing per-lane 16-point butterfly
  (on a slice of a 64-bit sign mask) followed by a 4-point butterfly across lanes through shared
  memory, scaled by ½. The gradient operand dY^T is then quantized with round-to-nearest elements, and
  each block's scale is corrected by EDEN's factor S = Σy² / Σy·q (y the scaled inputs, q the RTN codes)
  and stochastically rounded to E4M3, which makes the product unbiased **over the random rotation**
  rather than per element. With a 16-point transform the correction was measurably biased on
  heavy-tailed inputs; with 64 points the bias test passes (`tests/test_ms_eden_cuda.py`). Scales follow the
  paper's constants: a block maximum maps to 6.07 instead of 6, and the tensor scale uses 256 as its E4M3
  ceiling, leaving room for the correction to raise a scale.
  It removes the per-element random numbers from the gradient operand and halves its error.
- **MS-EDEN on dgrad** (`set_ms_eden(True, dgrad=True)`, v0.12; Quartet II's backward). dX = dY W
  contracts over N, so both operands are rotated along N by the same 64-point RHT. The A operand is the
  stochastically rounded row output of the dual quantizer (dY, or the GELU / SwiGLU input gradients the
  producers compute): its row phase applies `rht16` per thread and the 4-point stage across the four
  16-blocks of a tile row through shared memory (a 64×64 tile row is exactly one 64-element block), then
  quantizes with EDEN. The column phase, which needs the unrotated tile, reads its values into registers
  first; the amax pass computes amax of the rotated rows as a third maximum. Both are a separate
  instantiation of each kernel (template parameter), so the other paths are unchanged (A/B identical).
  The rotation's signs come from the row SR seed XOR a fixed salt, so no new parameters thread through the
  producers. The B operand cannot be the 16×16-block transpose any more (a rotation along N mixes the
  2D blocks): the layers dequantize the fprop weight and quantize it along N under the same rotation
  (round-to-nearest, `linear.dgrad_weight_operand`), so dgrad sees the forward's weights up to a second
  rounding, as Quartet II and Mosaic do with 1×16 weights. Cost: the gradient quantizer about doubles
  (both of its outputs rotate), plus one weight re-quantization per backward (not cacheable: the
  rotation changes every call).
- **1×16 weights with re-quantization** (`set_weight_blocks("1d")`, Quartet II / Mosaic). fprop uses W
  in 1×16 blocks along K (lower error than 16×16 blocks, and Four Over Six can choose per row); dgrad
  uses the dequantized fprop weight re-quantized in 1×16 blocks along N, so both GEMMs see the same
  weights up to a second rounding. Costs one dequantize and one extra quantization per weight per
  micro-batch (amortized by `cache_weight_quantization` under gradient accumulation).
- **BF16 tail** (`set_high_precision`, NVIDIA arXiv 2509.25149 §4.4): switch the NVFP4 layers to BF16 for
  the last part of training. The layers keep their weights and optimizer state and fall back to the
  modules they replace (fused sub-blocks to the unfused composition), so it can be flipped at any step.

**Measured** (GPT-124M, TinyStories, 3000 steps, three seeds; RESULTS.md §4.2):

| option | vs BF16 | vs baseline, paired | tokens/s cost |
|---|---|---|---|
| baseline (v0.10 recipe) | +1.32% | | |
| Four Over Six | +1.30% | −0.03% ± 0.02 | −2% |
| MS-EDEN | +1.24% | −0.08% ± 0.05 | −3% |
| MS-EDEN + Four Over Six | +1.21% | −0.12% ± 0.05 | −4% |
| 1×16 weights | +1.20% | −0.12% ± 0.09 | −12% |
| 1×16 weights + Four Over Six | +1.20% | −0.13% ± 0.20 | −13% |
| 1×16 weights + MS-EDEN + Four Over Six | +1.13% | −0.19% ± 0.11 | −17% |
| MS-EDEN on wgrad and dgrad (v0.12) | +1.24% | −0.08% ± 0.15 | −15% |
| MS-EDEN on wgrad and dgrad + Four Over Six (v0.12) | +1.12% | −0.20% ± 0.17 | −16% |
| fc2 kept in BF16 (precision budget) | +1.05% | −0.27% ± 0.06 | −12% |
| **BF16 tail (last 10% of steps)** | **+0.62%** | **−0.70% ± 0.04** | −4% |
| BF16 tail + MS-EDEN + Four Over Six | +0.55% | −0.76% ± 0.20 | −7% |

Every quantizer option lowers the loss on average, and MS-EDEN's and Four Over Six's reductions have the
same sign on every seed, but none removes more than a tenth of the gap at this scale: the forward-error
fixes that gave the papers 3–25% gap reductions on 0.3–7B models give 2–10% here. The BF16 tail halves
the gap on every seed for 4% of the run's time, and on top of it the quantizer options add nothing
resolvable. So the quantizer options stay off by default (they change numerics and cost 2–13% throughput
for ~0.1% loss), and the tail is the recommended addition (it is a training-loop decision, so it is a
switch the loop flips, not a default). A tail-trained model should be served in BF16: evaluated with its
NVFP4 layers it is no better than the baseline (RESULTS.md §4.2). Not implemented: the low-rank BF16
weight subspace of Full-Stack FP4 (+4% parameters) and oscillation damping (ALTO; aimed at gaps that grow
with training, which ours does not).

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
  supported on sm_120a"; PTX ISA 9.5 allows it only on sm_100a / sm_103a / sm_107a). It is done in
  software in the integer domain (v0.10), with 8 random bits per element, as NVIDIA's hardware SR uses:
  - one 32-bit hash per **four** elements, keyed once per block by the seed (`hash(q + key)`,
    `key = hash(lo ^ hash(hi))`), gives each element one byte;
  - magnitudes below 1 are lifted into [1, 2) (`a + 1`), where E2M1's 0.5 spacing is one mantissa bit;
    the random byte plus half a unit of it is added just below the one kept mantissa bit and the rest
    is masked off: a carry rounds up (into the next binade on its own), and the result is an exact
    E2M1 grid value that the hardware convert passes through. The half-unit offset is Fitzgibbon &
    Felix's unbiased few-bit rounding ("SRF", ARITH 2025): without it few-bit SR rounds down 2^-9 of a
    step too often (TE's software fallback has that bias), with it P(up) = round(256 f) / 256 for the
    position f in the step, exact on average and within 2^-9 everywhere (`test_eight_bit_stochastic_
    rounding_bias_is_bounded` checks all 256 bytes for 8000 values).
  - v0.8–0.9 used one hash per pair, 16-bit uniforms and float-domain arithmetic (28.6 extra SASS
    instructions per element vs ~12 now; docs/research/nvfp4-open-problems-2026-11/). Measured on the
    dual quantizer with SR on both outputs (bf16 gradients 8192 × 2304 / 8192 × 8192 / 16384 × 3072):
    152 / 602 / 536 µs before, 119 / 485 / 481 µs after (−10 to −22%); GPT-124M's NVFP4 validation loss
    over three seeds is unchanged (+1.32% ± 0.16 vs BF16, v0.5: +1.28%; RESULTS.md §4.2).
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
- **Stochastic rounding is reproducible:** it uses a counter-based hash (`lowbias32`) of the element's
  quad index plus a key that mixes both halves of the 64-bit seed, so SR is seedable and identical
  between CPU and GPU. (The SR noise differs from v0.9's for the same seed.)

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

### 4.5 Grouped GEMMs for mixture of experts (v0.12, `ops.gemm_grouped`)

A MoE layer runs every projection of every expert in one launch of the persistent TMA kernel. The kernel's
work-unit decoder (`unit_of`) is the only place that changes; the pipeline, the MMA loop and the epilogue are
the dense kernel's. Two modes:

- **Ragged M** (forward and input gradient). The routed tokens sit in one buffer sorted by expert, each
  expert's rows padded to a multiple of 128 (one tile), and are quantized once. An int32 table gives
  the expert of every 128-row tile. A unit reads A rows `m0` as usual and B rows
  `expert * N + n0` from the stacked weights. The dgrad operand of 16×16-block weights is the column output
  of the stacked weight's quantization, [K, E N], rearranged bytewise to [E K, N] (`ops.stack_transposed`;
  blocks never straddle experts).
- **Ragged K** (weight gradients). Each expert's output tiles reduce over its own token range of the
  column-quantized operands, `[k0, k0 + len)`, and write rows `expert * R + m0` of the stacked gradient
  (accumulating into `.grad` in place). Ranges are multiples of 128 tokens, but the k-tile is 256 deep and
  TMA needs 16-byte-aligned box starts in the scale rows. So a unit starts at `floor(k0 / 256) * 256`, and
  the MMAs of the first k-tile's first half or the last k-tile's second half are skipped when they lie
  outside the range (the data loaded for them is never multiplied).
- **K not a multiple of 256** (e.g. d_ff = 1408). The last k-tile reads zeros past K: the data tensor
  map's out-of-bounds fill, with the scale rows padded to a 16-byte stride (`ops._scales_ld16`). Dense GEMMs
  still require K % 256 for the TMA loop.

The grouped code is a separate instantiation of the kernel (template parameter): the dense kernel compiles
exactly as before. That matters: with runtime branches the dense GEMM lost 2–5%, without them it measures
the same as v0.11 (463 vs 464 TFLOPS at 8192³, interleaved A/B). Grouped results are bit-identical to the
dense TMA GEMM of each group where that one runs the same loop (`tests/test_grouped_gemm_cuda.py`).

The MoE layer around it (`nvfp4/moe.py`):

- **Dispatch** is one gather into the padded buffer (in bf16, which the quantizer reads anyway).
- **Combine** is a gather of each token's k rows and a k-term weighted sum (a batched matmul).
- **Backward** of the dispatch sums each token's k rows by gather. Neither direction uses scatter-add
  atomics, so the layer is deterministic.
- **Per step** that is a fixed number of kernels whatever the number of experts: 258 for a forward and
  backward of a 32-expert layer, vs 1,903 with per-expert launches (v0.11). The GEMMs are about a third
  of it; quantizers, the weight quantization and the routing gathers are most of the rest.
- **No host synchronization (v0.13).** v0.12 read the expert sizes on the host to size the buffer and
  build the tile table. Now:
  - The buffer has the worst-case size for T tokens: `T k + E (128 − 1)` rows, rounded up to 256, since
    each expert pads to at most 127 extra rows.
  - The tile table, the ragged-K segments and the dispatch indices are computed on the GPU from the
    device-side counts (cumulative sums and a `searchsorted`). The counts come from a `scatter_add` of
    integers, which is exact and so deterministic in any order. `bincount` has a data-dependent output
    size.
  - Two kinds of work units have no k-tiles: tiles past the used rows (tile group −1) and the segments of
    experts without tokens (length 0). The consumer warps run them as zero tiles (or leave an
    accumulated output unchanged). The load cursors skip them, so the producer and consumer sequences
    stay in step (`skip_empty`, grouped instantiation only).
  - The tail costs one zero store per unused tile. The numerics match v0.12 in distribution, not bit for
    bit: same rows, tensor scales and reduction ranges. The wider buffer changes the stochastic-rounding
    counter of the column (wgrad) operands, column × padded rows, so they draw other random bits.
  - With that, a step is capturable in a CUDA graph, and one graph serves any routing of the same token
    count. The layer also traces under `torch.compile(fullgraph=True)`: `gemm_grouped` is a
    `torch.library` custom op like the others.
- **Expert biases (v0.14).** The store of a ragged-M unit adds `bias[g, col]` when the layer passes one:
  the epilogue of the dense linear layers (v0.8, `linear_epilogue`), with the bias row taken from the tile's
  group. A second int32 table gives each tile's real (non-padding) rows, and only those get the bias.
  Biased padding rows would be harmless in the forward (their outputs are never gathered) but not in the
  wgrad: the column operands are quantized in 16-token blocks and rotated by the RHT along the tokens, so
  a nonzero padding row would share a block scale with, and be mixed into, an expert's real tokens. The
  check costs nothing in the dense kernel (it is inside the epilogue's bias branch, taken once per tile:
  457–463 TFLOPS at 8192³). Bias gradients are per-expert row sums of the gradients the quantizers read
  anyway: they already write per-64-row column sums for the dense layers' biases (§3.6); here those
  partial sums stay per tile (`bias_grad="tiles"`) and a one-hot [E, tiles] product in fp64 adds them per
  expert. That is deterministic and reads nothing on the host. Padding rows and unused tiles carry zero
  gradient, since nothing downstream reads their outputs. When the weights are frozen, the quantizers
  still run for those sums.
- **High-precision experts without host synchronization (v0.14).** Before conversion, or under
  `ops.set_high_precision` (a BF16 tail), the experts used to run one after another, which reads the
  expert sizes on the host. `torch._grouped_mm` would avoid that on Hopper; on sm_120 PyTorch 2.14 runs it
  as a loop that synchronizes. So the layer has its own grouped BF16 GEMMs, written in Triton
  (`nvfp4/_bf16_grouped.py`) because they need no FP4 machinery. They work on the same padded buffer and
  the same tables: a ragged-M kernel (one program per 128-row tile and 128 columns; unused tiles store
  zeros, the bias is added in fp32 with one rounding to the tile's real rows) for the forward and the input
  gradient, and a ragged-K kernel for the weight gradients. Both have no atomics and a fixed reduction
  order, so they are deterministic, and both are `torch.library` custom ops. On the benchmarked layers
  they are 2–7% faster than the loop (RESULTS.md §8). The point is not speed: a model with MoE layers is now
  CUDA-graph capturable and compiles without graph breaks in BF16 too, including the BF16 blocks of an
  NVFP4 model and a BF16 tail. Without Triton (native Windows) the loop remains.
- **Capacity, shared expert, expert parallelism (v0.15).** All three reuse one planner (`_plan`): given
  S slots, each with a source row and an expert (−1 for a dropped slot), it builds the padded buffer's
  tables on the GPU. The planner is the v0.13 layout, generalized.
  - **Capacity** (`capacity_factor`): expert e keeps the first C = ⌈cf·T k / E⌉ of its slots in token
    order, as in the Switch Transformer. A slot's rank within its expert comes from the same stable sort
    by expert and the per-expert offsets. A dropped slot's combine weight is zeroed, and it points at a
    buffer row that is always zero (one extra row, past every used tile). The buffer shrinks to the
    worst case min(T k, E C) + E·127 rows.
  - **Shared expert** (`shared_d_ff`): a dense MLP every token passes through with weight 1. That is
    the planner with one group, so the same fused NVFP4 kernels run it (or the Triton / loop paths in
    high precision). n shared experts of hidden h are exactly one of hidden n h (their outputs are
    summed with weight 1), so one is all the layer needs.
  - **Expert parallelism** (`ep_group`, W ranks, E / W experts each): slots travel in fixed-size
    buffers [W, E / W, C, d] (expert e's kept slots in rows e C + rank). The kept counts go along in a
    small integer all-to-all. The receiving rank plans its E / W experts over the W C rows per expert,
    marking unused rows −1. It runs them like any other layout, and the outputs return by the reverse
    all-to-all (a differentiable wrapper: its backward is the reverse exchange).
  - **Dropless expert parallelism (v0.16)**: without a capacity a rank may receive up to W T k rows, so
    a worst-case buffer would cost W times the memory. Instead the counts go first (an integer
    all-to-all) and are read on the host once per forward, and the rows, sorted by expert and so by
    destination rank, travel in variable-size all-to-alls; the receiver plans its E / W experts over
    exactly the rows it got. This is what the dropless dispatchers of Megatron-LM and DeepSpeed do, and
    like them it cannot be captured in a CUDA graph.
  - What we gain: the layer reads nothing on the host; with NCCL a step stays capturable. With fp32
    experts the result equals the single-process layer on each rank's tokens, and the local experts'
    gradients equal the sum over every rank's tokens (tested with two processes over gloo on one GPU).
    With NVFP4 experts the activation's tensor scale spans every source rank's rows, so the numerics are
    equivalent, not bitwise.
  - **Overlap (v0.16)**: an exchange is two autograd functions, one that starts the all-to-all
    (`async_op`) and one that waits for it. Whatever runs between them overlaps with the transfer in the
    forward. In the backward the waiting function's backward starts the reverse exchange and the starting
    function's backward waits for it; autograd runs ready nodes with the highest sequence number first,
    so the nodes created between the two in the forward run, in reverse, while the gradients travel.
    The layer puts the shared expert there, during the return of the expert outputs (a test checks the
    call order).
  - **Variable-size exchanges (v0.17)** replace v0.16's dropless path (and serve a capacity too, with
    `ep_dispatch="variable"`). Three changes, each aimed at the exchange that v0.16 measured as the
    whole cost of expert parallelism on PCIe machines (RESULTS.md §8.1):
    - *Once per rank.* v0.16 sent every slot, so a token whose k experts sit on one rank crossed the
      link k times, and its k outputs came back separately. Now a token travels once to each rank that
      holds one of its experts, with those experts' ids and routing weights (fp32) in the same message,
      and the rank returns one row: the weighted sum of its experts' outputs (the layout of DeepSeek's
      DeepEP). With top-4 of 32 experts on two ranks a token reaches 1.9 ranks on average instead of
      sending 4 rows. A router (`_route`) computes, on the GPU, one sent row per (token, member) pair
      grouped by member, the member-local ids, and for the backward the row each (token, choice) maps
      to (its member's first choice; the others point at a zero row, so each token's gradient is a
      gather-sum over distinct rows: deterministic). The weights' gradients come back through the same
      reverse message; at most one sent row carries a nonzero weight per choice, so their gather-sums
      add exact zeros.
    - *Counts once.* The receivers' sizes depend on routing, so they are read on the host, but once per
      forward: one integer all-to-all carries, per chunk and destination, the slots per expert there,
      the rows to it, to its node and to each rank of its node, and the input's amax (for the NVFP4
      wire). Every message of every chunk and stage is then sized from that one read.
    - *Byte messages.* A payload (rows, weights, ids) is packed into one uint8 row per token, so each
      exchange is one all-to-all; a start function (async) and a wait function as in v0.16, now carrying
      a scalar handle through autograd because the bytes are not differentiable.
  - **Nodes (v0.17, `ep_node_size` L).** With W = N L ranks in nodes of L consecutive ranks, a token
    crosses to another node once per node it needs (DeepEP's node-limited layout), to the rank with the
    sender's position in that node: the "rails" (one per position, N ranks each) carry only inter-node
    traffic, which on rail-optimized clusters is one NIC per position. The receiving rank routes the
    token again, once per rank of its node holding one of its experts, and on the way back sums its
    node's rows, so the return crosses nodes once per (token, node) too. The stages run over rail and node
    sub-groups (created once per (group, L), shared by all layers). Running both over the whole group's
    communicator instead, with zero-size splits to the other ranks, was tried and measured 3x slower on the
    four-GPU host (284 vs 96 ms per step; its 4-rank all-to-alls are slow even when half the pairs send
    nothing). With chunks the two sub-groups' collectives are in flight at the same time next to the
    persistent GEMMs; on the final measurement that was 2.4x slower than unchunked (233 vs 96 ms; an earlier
    run with the unfused combine had it 8% faster), so `ep_chunks` stays at 1 with nodes. Tested with four
    processes as two nodes of two on one GPU (gloo) and over NCCL on four GPUs against the single-process
    layer. No multi-node machine was available; it was timed on one host whose GPUs sit in pairs behind
    separate PCIe host bridges, the pairs as nodes (RESULTS.md §8.2).
  - **Chunks (v0.17, `ep_chunks` n).** The routed experts used to wait for all their rows. Now the tokens
    go in n contiguous chunks, each a generator that yields after starting each exchange, run in a
    wavefront: at each step the later chunks start their next exchange before the earlier ones compute,
    so chunk c + 1's rows travel while chunk c's experts run, and chunk c's outputs return while chunk
    c + 1's run. In the backward the same structure falls out of autograd's order: the waits' backward
    nodes (which start the reverse exchanges) have the highest sequence numbers and all become ready at
    once, so every chunk's output gradients leave first, and each chunk's input gradients leave as soon
    as its experts' backward is done. Each chunk's experts form their own buffer (their hidden
    activation's tensor scale is per chunk, the weight gradients accumulate over chunks).
  - **SM margin (v0.17, `ops.set_sm_margin`).** The persistent GEMMs launch one CTA per SM with ~99 KB of
    shared memory each, so an NCCL kernel issued during a GEMM finds no SM: it waits, or, if it started
    first, the GEMM CTAs that could not become resident run their statically assigned tiles only after
    the transfer ends. A margin leaves k SMs to concurrent kernels; GEMM results are bit-identical with
    any margin (tested).
  - **NVFP4 messages (v0.17, `ep_wire`, `ep_wire_grad`; NVFP4 experts).** An NVFP4 row is 9 d / 16 bytes
    of codes and block scales plus 8 bytes (the tensor decode scale and amax), 3.6× fewer than bf16 at
    d = 1024. The receiver dequantizes (a new kernel, bit-identical to `reference.dequantize`, reading
    strided message rows with per-row tensor scales) and the experts quantize again. That is lossless
    for the forward because of two things. `quantize_dual(amax_floor=)` lets a tensor's scale come from
    max(amax, floor), and the expert-parallel layer passes the group's input amax (from the counts
    message) both to the sender and to the experts' quantizer, so the tensor scale is the same everywhere.
    And re-quantizing a dequantized NVFP4 row with the same tensor scale reproduces its codes and
    block scales: each block's largest element maps back to 6 (or 4 under Four Over Six) times the same
    e4m3 scale, and bf16 rounding of the dequantized values (2^-9 relative) is far from any rounding
    boundary of the 1-mantissa-bit codes. So the experts' forward and input gradient are bit-identical to
    bf16 messages (tested, also through the node stage, where the rail rank re-encodes). Only the weight
    gradients see a difference: their token-wise operand is quantized from the dequantized rows, a
    second rounding. Gradient messages (`ep_wire_grad`) are stochastically rounded with a tensor scale
    agreed over the group (a one-element all-to-all in the backward) and carried along for re-encoding
    at a rail rank. The experts' input gradients are then rounded twice, both times unbiased: the
    receiver multiplies the rows by the routing weights before its own quantization, so the codes are
    not reproduced. The combine direction (expert outputs, input gradients) stays bf16, as in DeepSeek-V3:
    those tensors are otherwise unquantized.
  - **Fused gather / combine (v0.17, `csrc/moe_combine.cu`).** The combine y[t] = Σ_j w[t, j] ys[pos[t, j]]
  was an index_select into a [T, k, d] buffer and a batched GEMM of T 1 × k by k × d products, whose
  backward is a scatter-add (atomics). Profiled on expert-parallel steps (RESULTS.md §8.2) those kernels
  and their copies took 15–20 ms of a 70 ms step. Now one kernel reads each of a token's k rows once and
  accumulates in fp32 in choice order (one rounding; indices outside the rows read nothing, so no zero row
  is concatenated). The backward is two gathers: each row's gradient comes from the one choice that reads
  it (the layer's layouts guarantee that, or the choice has weight zero; the reader is found with an
  integer scatter-max, deterministic), and the weights' gradients are per-choice dot products, a warp per
  choice. The same kernel is the token dispatch's backward (a gather-sum with unit weights) and the
  expert-parallel partial sums. The weights are rounded to the rows' dtype first, as the batched GEMM took
  them, so compiled and eager steps stay bit-identical. `NVFP4_MOE_FUSED_COMBINE=0` restores the old
  path. The single-GPU MoE layer gets 8–15% faster in NVFP4 (3–7% in BF16) and uses ~2 GB less at 16k
  tokens (RESULTS.md §8).
- **With data parallelism.** The experts' gradients already sum every rank's tokens, so data
    parallelism over the same ranks must skip them (`ignore_expert_parameters_in_ddp`) and scale them by
    1 / W to match DDP's average (`scale_expert_grads`); a two-rank DDP test checks every gradient
    against one process training on both batches, and `train_gpt.py --moe-ep` trains that way.

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
| Grouped GEMMs, MoE (v0.12–v0.15) | `tests/test_grouped_gemm_cuda.py`, `tests/test_moe_cuda.py`, `tests/test_moe_integration_cuda.py` | ragged M / K vs fp64 and bit-identical to the per-group dense TMA GEMM; unused tiles and empty segments store zeros (or leave accumulated outputs unchanged) between real units; per-group bias epilogue == `D_bf16 + bias[g]` bit for bit on each tile's real rows; MoE (with and without biases, NVFP4 / grouped BF16 / loop) vs a per-token loop, BF16 experts and a per-group fp64 GEMM; bias gradients == routed gradient sums, bias-only training bit-identical; grouped BF16 == loop at bf16 level, deterministic; no host sync in forward + backward; CUDA-graph forward replays == eager bit for bit under changing routing; captured Adam steps learn; `torch.compile(fullgraph=True)` == eager (bit-identical outputs, bias and NVFP4 weight gradients with `emulate_precision_casts`); checkpointing bit-exact; **two-process DDP, FSDP1, FSDP2** bit-exact, fused and unfused; capacity and the shared expert vs a per-token reference (dropped-slot counts exact; a capacity no slot reaches == dropless bit for bit); **two-rank expert parallelism** == the single-process layer per rank (fp32), local expert gradients == the sum over ranks |
| Expert parallelism, wire, combine (v0.17) | `tests/test_moe_ep_cuda.py` | dequantize **bit-exact** vs `reference.dequantize` (bf16 / fp32, strided message rows, per-row tensor scales); `amax_floor` == quantizing with an extra row at the floor (plain and multi-part paths); NVFP4 message round trip reproduces codes and scales (also with Four Over Six); the router's (row, member) pairs, ids, masks and positions on a worked example; persistent GEMMs bit-identical with an SM margin; **two- and four-process expert parallelism** (flat and two nodes of two; dropless and capacity; 1–7 chunks) == the single-process layer in fp32, every gradient; NVFP4: forward and input gradient with NVFP4 messages **bit-identical** to bf16 messages (also through the node stage), weight gradients and NVFP4 gradient messages by cosine; the same over NCCL on 2 / 4 GPUs; DDP + expert parallelism == one process on both batches (every gradient); fused combine == fp32 reference **bit for bit** (bf16 / fp16 / fp32, vector and scalar paths, strided rows), its gradients == autograd, the layer == the bmm path |
| Memory / race safety | `tools/sanitize.sh` | compute-sanitizer memcheck, racecheck, synccheck, initcheck over every kernel variant (ragged shapes, TMA and cp.async loops, persistent grids with more units than SMs, split-K, accumulate, the bias / residual epilogue, all producer ops, multi-part inputs, the norm backward, GPU-resident seeds, layer and v0.8 block forward + backward, 8-bit attention forward + backward): 0 findings |
| Long-run stability | `tools/soak.py` | 8 minutes (v0.7): 18,401 fused + unfused training iterations, 3,682 bit-exact gradient replays, 92,005 GEMMs re-checked bit for bit; 5 minutes (v0.8, residual-fused blocks): 14,474 iterations, 2,896 replays, 72,370 GEMMs; no non-finite values, no hang |
| Packaging | wheel built and installed into an isolated directory | imports and trains from outside the repository |
| Hardware ISA | CI `cuda-compile` job | `OMMA.SF.16864...UE4M3.4X` present in SASS |
| End to end | `experiments/train_gpt.py` | loss curves vs BF16 (see [RESULTS.md](RESULTS.md)) |

Every benchmark checks correctness against the reference before it times anything.

## 6. Known limitations and next steps

Done in v0.17: expert parallelism with each token sent once per rank (or node, two-stage), chunked
exchanges overlapping the experts, NVFP4 messages, an SM margin for the persistent GEMMs, expert parallelism
under DDP, a fused gather / combine for every MoE path (§4.5, item 5); measured on four RTX PRO 4500
Blackwell (a second sm_120 model: tests, sanitizers, soak, benchmarks; item 7).
Done in v0.16: dropless expert parallelism, the shared expert overlapped with the expert parallel
exchange, expert parallelism measured over NCCL on two RTX 5090s, and native Windows tests on PyTorch 2.11
(§4.5, items 5 and 7).
Done in v0.15: capacity-limited routing, a shared expert and expert parallelism in the MoE layer
(§4.5, item 5).
Done in v0.14: per-expert biases in the MoE layer (added in the grouped GEMM's store; §4.5) and
grouped BF16 experts without host synchronization (Triton), so every MoE path is capturable (item 5).
Done in v0.13: the NVFP4 MoE layer without host synchronization (CUDA-graph capturable, `torch.compile`
without graph breaks; §4.5, item 5), and MoE under DDP / FSDP1 / FSDP2, compile and activation
checkpointing tested.
Done in v0.12: grouped GEMMs for mixture-of-experts layers (§4.5; item 5), MS-EDEN on the dgrad operands
(§2.1).
Done in v0.11: the recipe options of the 2026 literature with a three-seed ablation (§2.1), a BF16-tail
precision switch, cheaper stochastic rounding (§3.2, item 3), mixture-of-experts layers (item 5), and a
research survey of the open problems (docs/research/nvfp4-open-problems-2026-11/).
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
2. **Attention** (v0.9: 8-bit, §7; v0.10: exact backward of the 8-bit forward, §7.1). Training with
   it now matches BF16 attention's loss at 4096 tokens, but gains only 1.07–1.12× per attention call at
   D = 128 (1.04–1.07× at D = 64): the backward runs at BF16 speed, and the two-term P V the exact
   backward needs costs the forward a quarter of its speedup (1.2–1.6× instead of 1.5–2.1×). The
   0.9 8-bit backward (`backward="8bit"`) is 1.12–1.18× but +0.5% loss at 4096 tokens. Tried and
   rejected: skipping the residual P V product on (16-row, 32-key) MMA tiles whose probabilities are all
   negligible. On a GPT-124M's activations at 4096 tokens (`bench/attention_skip_probe.py`) no tile
   qualifies even at a threshold of 2^-5 of the running row maximum (where in most layers under 1% of the
   individual probabilities fall below 2^-7 and at most 26% below 2^-5): attention there is too diffuse,
   and the kernel's running maximum makes early tiles look larger still. Also tried and rejected
   (v0.11, `bench/attention_qbwd_probe.py`, variant `qb_int8pv`): a single-term INT8 P V (P at a fixed scale
   127 / row maximum, V per channel, as SageBwd) under the same exact backward; on the peaked layers of the 4096-token model
   dQ errors reach 3.6–10.6% (the two-term MXFP8 product: 0.4–2.0%), because small probabilities round
   to zero. FlashAttention-4's polynomial exp2 was not pursued: the forward is bound by the MMA pipe
   (INT8 and MXFP8 at 473 / 438 dense TFLOPS on this part), not by MUFU. Remaining way forward: V in INT8
   per channel with P in two terms. Masks other than causal, packed
   sequences, dropout and head dims other than 64 / 128 fall back to SDPA.
3. **Stochastic rounding is still software** (sm_120a has no `cvt.rs`, and PTX ISA 9.5 keeps it off every
   sm_12x target): v0.10's integer-domain rounder with 8 random bits per element costs 10–22% less than
   v0.9's; MS-EDEN (§2.1) removes element-wise random numbers from the wgrad gradient operand.
4. **Weight cache limits.** In-place writes through `param.data` outside an optimizer step cannot be
   detected (`.data` deliberately gives the tensor a fresh autograd version counter); call
   `invalidate_weight_caches`. Under FSDP2 the cache hits only when the all-gathered weight lands at
   the same address within a step (the allocator's choice; correct either way).
5. **Fused sub-blocks** cover GPT / BERT / ViT-style and Llama-style blocks, pre-norm or (via the
   norm-free MLPs) post-norm, with biases anywhere. Mixture-of-experts MLPs (`nvfp4.moe.MoE`, dropless
   top-k; v0.11, grouped GEMMs since v0.12, sync-free since v0.13, biases and a sync-free BF16 path since
   v0.14, capacity, shared expert and expert parallelism since v0.15, §4.5) cover SwiGLU and GELU experts.
   Expert parallelism (fixed-size or variable-size exchanges, nodes, chunks, NVFP4 messages; v0.15–v0.17)
   is verified with two and four gloo processes on one GPU and over NCCL on two RTX 5090s and four RTX PRO
   4500 (RESULTS.md §8.1–8.2). Not timed: a real multi-node cluster (the two-stage exchanges ran over pairs
   of GPUs behind separate PCIe host bridges of one host). Chunks overlap the exchange, but on that host the
   expert GEMMs run ~1.7x slower while NCCL copies run beside them (memory contention), and an SM margin does
   not change that; on fast links chunking costs a few percent. The combine direction (expert outputs,
   input gradients) always travels in bf16.
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
   and five RTX 5080 hosts whose containers never started; a second one later that day found none of the
   four available on Runpod's Secure or Community cloud). The attention kernels need nothing
   SM-count specific: two forward CTAs per SM fit the 100 KB of shared memory every sm_120 SM has. On
   Windows, inductor needs Triton, which PyTorch's Windows wheels don't include. v0.16 reran the suite
   natively on Windows with PyTorch 2.11.0+cu128 and CUDA 12.9 (919 passed). That wheel has no flash
   attention: the 8-bit attention's BF16 backward then runs the same math in fp32, a block of query rows
   at a time (tested against the flash backward on Linux). Its gloo crashes on all_reduce of CUDA
   tensors (an access violation, even with one rank; all_to_all works), so DDP / FSDP over gloo on
   Windows with that PyTorch is unusable and those tests skip (they passed with 2.7.1).
   v0.17 ran the suite, the sanitizers and the soak on an RTX PRO 4500 Blackwell (82 SMs, 200 W, 32 GB;
   PyTorch 2.11, CUDA 12.8), the first sm_120 part besides the RTX 5090s: all pass (after two fixes for
   PyTorch 2.11's torch.compile: the attention backward's flash check is now decided at import, and the MoE
   compile test compares numerically before 2.14, whose inductor rounds bf16 casts like eager). The GEMM
   reaches 491 TFLOPS there, 1.0–2.0x cuBLASLt's NVFP4, and the training-step gains match the larger parts
   (RESULTS.md §9). The 8-bit attention does not pay there: 0.77–1.01x SDPA forward + backward (it was
   tuned on the 5090's 170 SMs and power budget).

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

### 7.1 The exact backward of the 8-bit forward (v0.10, the default)

The 8-bit backward above leaves 2–4% error in dK / dV and costs +0.5% loss at 4096 tokens; it is also no
faster than flash attention's BF16 backward (≈80 TFLOPS both, at D = 128 from 4k tokens: the 8-bit
kernels are bound by element-wise work and shared-memory traffic, not the tensor pipe). So the default
backward is now PyTorch's flash-attention backward, in the input precision, *of the function the 8-bit
forward computes*:

- **Same operands.** The quantizer also writes the dequantized operands (`attn_quant(..., want_deq)`):
  int8 code × per-token scale for q and centred k, e4m3 × UE8M0 scale for v, rounded to BF16 / FP16
  (exact for v; 2^-9 for q, k). Flash attention's backward runs on these, so its recomputed scores are the
  forward's scores up to BF16 rounding and the forward's log-sum-exp matches them (max difference 0.005
  nats on random inputs, ≤ 0.14 on the most peaked trained layers). The first attempt passed the original
  q, k, v instead, and the 8-bit scores' lse differed by up to 0.2 nats (P no longer summed to one: 6%
  gradient error on random inputs, over 100% on trained activations).
- **Consistent Delta.** Flash attention takes Delta = rowsum(dO ∘ O) from the O it is given, and the
  exact backward needs Delta = rowsum(P ∘ dP) of its own exact P: an O computed with P rounded to e4m3
  (relative error up to 2^-4) left rows of dS summing to r_i ≠ 0, which dQ amplified to 6–8% on the
  peaked layers 8–9 of a trained GPT-124M at 4096 tokens (`bench/attention_qbwd_probe.py`). The forward
  therefore computes P V as e4m3(P) V + e4m3(16 (P − e4m3(P))) V / 16: the residual comes from the
  f16x2 difference of P and its e4m3 value (exact), one more MXFP8 MMA per P V k-step with A scale
  2^-4, P carried to ~2^-8. dQ error on those layers: 1.2–2.0%.
- **Straight-through.** The gradients are the exact gradients of attention at the quantized operands,
  used as the gradients of the unquantized ones (the tests check them against fp32 autograd at the
  dequantized inputs: within 1.2%, BF16 flash attention's level).

Errors against exact attention on a trained model's activations at 4096 tokens (layers 0–9; 8-bit
backward in brackets): output 0.6–1.8% (0.7–2.3%), dQ 0.4–2.0% (0.3–0.6%), dK 1.2–6.9% (2.8–7.1%; about
half on most layers), dV 0.2–2.1% (2.5–3.1%). GPT-124M at 4096 tokens, 2 seeds: validation loss 1.5979 /
1.5949 against BF16 attention's 1.6027 / 1.5928 (−0.09% on average: the gap is gone) and the 8-bit
backward's 1.6088 / 1.6033, at the same training speed (66.9k tokens/s vs 66.2k for SDPA).

The cost is in the forward: the extra MMA (not the residual arithmetic: issuing it with the residual
replaced by the main term runs as slow) takes the forward from 1.5–2.1× flash attention to 1.2–1.6×
(D = 128, 4k–8k tokens), and forward + backward is 1.07–1.12× (D = 64: 1.04–1.07×). Determinism is that
of PyTorch's flash backward: bitwise reproducible under `torch.use_deterministic_algorithms`, else not
(dQ accumulated with atomics); `backward="8bit"` keeps the deterministic kernels and the one-term
forward. Memory: the dequantized q, k, v (input dtype) plus O and lse, as SDPA saves.
