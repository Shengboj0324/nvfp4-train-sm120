# Cheap software stochastic rounding (SR) to FP4 (E2M1) and FP8 on GPUs without a hardware SR convert (state as of 2026-11-24)

Scope: what hardware offers (NVIDIA PTX, AMD CDNA, Graphcore), how production stacks generate and consume random bits, how many random bits SR needs and what bias the cheap variants carry, alternatives to element-wise SR, and the cost per element of each technique on sm_120a. The cost numbers are static SASS instruction counts. This research compiled them with CUDA 13.0 (`nvcc -O3 -arch=sm_120a`) from a probe file of 16-element-per-thread kernels. The probe sources are in [../code/](../code/) (`sr_variants.cu`, `count.sh`, `bias_check.py`); the load-bearing code is also reproduced below.

## Q1. Does sm_120 / sm_120a support any form of cvt.rs or another RNG-assisted convert?

### Takeaway
No. In the current PTX ISA (the live docs are version 9.5), the `.rs` rounding mode and every `cvt.rs.*` form are restricted to `sm_100a`, `sm_103a` and `sm_107a`. That list has no family (`f`) targets, so `sm_110a`, `sm_120a`, `sm_121a` and `sm_120f` are all excluded. PTX has no RNG instruction. The only FP4 converts that sm_120a/sm_120f has are round-to-nearest ones: `cvt.rn.satfinite.e2m1x2.f32`, and, from PTX 9.1, `e2m1x2` from `f16x2`/`bf16x2`. The newer `.rz` mode for FP8/FP6/FP4 converts (PTX 9.4) is also gated, to `sm_107f` or higher in that family, so sm_120 cannot use it for truncate-after-noise either.

### Cited Findings
- **`.rs` targets.** PTX ISA "Target ISA Notes" for `cvt` (current docs, PTX ISA 9.5) say ".rs rounding mode is supported on following architectures: sm_100a sm_103a sm_107a". `cvt.rs{.e2m1x4/.e4m3x4/.e5m2x4/.e3m2x4/.e2m3x4}.f32` has the same three-target list. Neither list has a family-specific (`f`) variant. — [PTX ISA, cvt](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
- **Version history.** ".rs rounding mode introduced in PTX ISA version 8.7." So do the `cvt.rs` e2m1x4/e4m3x4/e5m2x4/e3m2x4/e2m3x4 forms. The `sm_107`/`sm_107a` target strings arrived in PTX ISA 9.4. The PTX docs list `sm_107` among the "sm_10x target architectures". — [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- **Forms and semantics.** The forms are `cvt.rs{.relu}{.satfinite}.f16x2.f32 d, a, b, rbits`, the same for `.bf16x2`, and `cvt.rs{.relu}.satfinite.{f8x4,f6x4,f4x4}type.f32 d, {a,b,e,f}, rbits`. The result "is rounded in the direction toward zero or away from zero based on the carry out of the integer addition of the supplied random bits (rbits) to the truncated off (discarded) bits of mantissa." — [PTX ISA, cvt](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
- **Random bits per element in hardware.** e2m1x4 gets 8 bits per element ("lower 8-bits from both 16-bits half of rbits are used for operands e, f and upper 8-bits ... for operands a, b"). e4m3x4/e5m2x4/e2m3x4/e3m2x4 get 16 bits per pair of elements: one 16-bit half for a,b and the other for e,f. f16x2 gets 13 bits per element, and bf16x2 gets 16 bits per element. — [PTX ISA, cvt](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
- **RN FP4 converts.** `cvt.rn.satfinite{.relu}.{e2m1x2/e2m3x2/e3m2x2/ue8m0x2}.f32` is supported on sm_100a, sm_101a/sm_110a and sm_120a, and from PTX 8.8 on the families sm_100f, sm_110f and sm_120f. `cvt.rn.satfinite{.relu}{.e2m3x2/.e3m2x2/.e2m1x2}{.f16x2/.bf16x2}` (PTX 9.1) is family-specific: sm_100f, sm_110f and sm_120f. — [PTX ISA, cvt](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
- **`.rz` is gated too.** ".rz rounding mode on cvt.{e4m3x2, e5m2x2, e2m3x2, e3m2x2, e2m1x2}.{f32/f16x2/bf16x2}" was introduced in PTX ISA 9.4 and is supported only on "sm_107f or higher in the same family". — [PTX ISA, cvt](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
- **Compiler gating.** LLVM/Clang gate the `cvt.rs` intrinsics on PTX ≥ 87 and `__CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM103_ALL`. MLIR NVVM requires `NVVMRequiresSMa<[100, 103]>`. — [LLVM PR #160494](https://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20250922/1740988.html); [MLIR PR #165736](https://lists.llvm.org/pipermail/mlir-commits/2025-November/138850.html)
- **Transformer Engine's flag.** TE defines `ARCH_HAS_STOCHASTIC_ROUNDING` as `ArchSpecific<100>, ArchSpecific<103>, ArchSpecific<107>`. Its FP4 cast comments say "No cvt.rs on this architecture (e.g. SM110): use the software rounder." (main branch, commit a6393d0, 2026-11-24.) — [TE util/ptx.cuh](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/util/ptx.cuh); [TE hadamard_transform_cast_fusion.cu](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/hadamard_transform/hadamard_transform_cast_fusion.cu)

### Inferences
- **Fastest sm_120a route.** It is "integer noise + truncate in fp32 bits, then an exact `cvt.rn.satfinite.e2m1x2.f32`". The convert is exact because the value is already on the E2M1 grid (see Q5). Using `cvt.rn` on a noised value only gives an unbiased result when the noise is centred on a uniform-spacing interval. Truncation in integer space avoids the binade-boundary special cases.
- **What NVIDIA considers sufficient.** Because NVIDIA's hardware SR feeds only 8 random bits per FP4 element, 8 bits per E2M1 element is the budget NVIDIA's own validated recipe trains with (see Q3). The project's current 16 bits per element is twice that.

### Gaps
- PTX does not say how the 8 e2m1 rbits line up against the 22 discarded fp32 mantissa bits, or whether hardware adds a half-LSB offset. So it is unverified whether `cvt.rs` is the biased "SRFF" variant of Q3.
- Nothing found says whether a future sm_12x revision (for example a later family member) will get `.rs`.

## Q2. How do Transformer Engine, torchao, Quartet/QuTLASS, Graphcore and AMD implement SR (bits per element, RNG choice)?

### Takeaway
The two NVIDIA stacks are the same design:
- **NVIDIA (TE, torchao).** Philox4x32 counter-based RNG, 10 rounds by default and configurable at build time. One `generate4()` (128 bits) feeds 16 FP4 elements at 8 bits each, and one 32-bit word covers 4 elements. Since 2026, TE also ships a float-domain software SR fallback using the same 8 bits per element, for architectures without `cvt.rs`.

The other platforms:
- **AMD MI300 (CDNA3).** It has FP8 SR converts that add 20 random bits to the mantissa.
- **AMD MI355 (CDNA4).** It adds FP4/FP6/FP16/BF16 SR converts. The second element of a packed FP4 pair takes its randomness from a single Galois-LFSR step (`V_PRNG_B32`) of the first element's 32-bit seed. That is a very cheap, low-quality decorrelation step.
- **Graphcore IPU.** It has a per-tile hardware PRNG (xoroshiro128aox, 64 bits per cycle).
- **Quartet/QuTLASS.** No SR kernel source was found in QuTLASS.

### Cited Findings
- **TE NVFP4 quantize kernels.** They hold `philox4x32_native_state<NVTE_BUILD_NUM_PHILOX_ROUNDS> rng`, call `rng.generate4()` (a uint4), and hand out one 32-bit `rbits` word per `cvt.rs.satfinite.e2m1x4.f32`, i.e. 4 elements, refilling every 4 words. — [TE cast/nvfp4/core_nvfp4.cuh](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/cast/nvfp4/core_nvfp4.cuh); [TE quantize_transpose_nvfp4.cuh](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh)
- **TE's Philox round count.** It is a build option: `NVTE_BUILD_NUM_PHILOX_ROUNDS`, default "10", with the CMake message "Philox rounds for stochastic rounding". — [TE common/CMakeLists.txt](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/CMakeLists.txt)
- **TE's software SR (`stochastic_round_fp4_e2m1`).** Its comment: "Takes 8 random bits per element, which is what cvt.rs.satfinite.e2m1x4.f32 takes from its rbits operand". The algorithm:
  - u = (rbits & 0xFF)/256, a = |x|.
  - step = 0.5, 1 or 2 by region; t = fma(u, step, a).
  - Recompute the step from t, then q = min(floor(t/step_t)·step_t, 6), and copysign.
  - `stochastic_round_fp4_e2m1x4` uses bits [8i, 8i+8) for value i.
  - It is used in the Hadamard-transform+cast fusions when `ARCH_HAS_STOCHASTIC_ROUNDING` is false.

  — [TE util/ptx.cuh](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/util/ptx.cuh)
- **Where TE applies SR.** The TE NVFP4 docs say SR is enabled only for gradients and can be disabled with `disable_stochastic_rounding=True`. They describe it as hardware-accelerated on Blackwell and list training support on SM 10.0/10.3. — [TE NVFP4 user guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
- **torchao NVFP4 MoE training kernels (CuTe DSL / Triton).** They use a caller-owned Philox state and emit `cvt.rs.satfinite.e2m1x4.f32`, "each consuming one 32-bit random word". The row-wise and column-wise paths take distinct seeds/offsets for "fully independent Philox streams for the two quantization paths" ("uncorrelated RS between column and row quantization"). (main, commit 0a49cd7, 2026-11-23.) — [torchao nvfp4_training/_cutedsl_kernels_impl.py](https://github.com/pytorch/ao/blob/main/torchao/prototype/moe_training/nvfp4_training/_cutedsl_kernels_impl.py); [torchao hadamard_quantize_row_col_triton.py](https://github.com/pytorch/ao/blob/main/torchao/prototype/moe_training/nvfp4_training/hadamard_quantize_row_col_triton.py)
- **torchao BF16 optimizer SR (`_fp32_to_bf16_sr`).** It draws a 16-bit random integer per element with `torch.randint(0, 1<<16)` and rounds away from zero when `rand_16bit < (x_bits & 0xFFFF)`. That is N = D = 16, a full-width compare. — [torchao optim/quant_utils.py](https://github.com/pytorch/ao/blob/main/torchao/optim/quant_utils.py)
- **AMD CDNA3 (MI300) `V_CVT_SR_FP8_F32` / `V_CVT_SR_BF8_F32`.** Pseudocode: `m = mantissa(S0) + S1[31:12]`, i.e. 20 random bits added to the 23-bit mantissa, "wrap around on overflow". Then RNE convert to FP8. The random value is a full VGPR operand. — [AMD CDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- **AMD CDNA4 (MI355, gfx950) SR converts.** It adds `CVT_SR_FP16_F32`, `CVT_SR_BF16_F32` and the scaled `V_CVT_SCALEF32_SR_PK_FP4_F32` (plus FP6/FP8 variants). The FP4 pseudocode:
  - `randomVal = S1.u32; tmp0 = f32_to_fp4_sr_scale(S0[31:0], randomVal, scale)`
  - `randomVal = v_prng_b32(randomVal); tmp1 = ...` (second element)
  - The ISA says that for multipass conversions "an updated random number is generated every pass ... created internally via V_PRNG_B32".

  — [AMD CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- **AMD `V_PRNG_B32` (CDNA4).** It is a one-step 32-bit Galois LFSR: `D = (in << 1) ^ (in[31] ? 197 : 0)`, period 2^32−1. It is not present in the CDNA3 ISA text. — [AMD CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- **HIP SR APIs.** HIP exposes them with an integer seed argument, e.g. `__amd_cvt_float_to_fp8_sr(float, interp, unsigned int seed)`, `__amd_cvt_floatx2_to_fp4x2_sr_scale` and `__amd_cvt_bf16x8_to_fp4x8_sr_scale`. "gfx950 provides hardware acceleration for hip extensions." — [ROCm HIP low-precision types](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html)
- **Graphcore IPU.** It supports SR in hardware, sharing a global seed with the TF stateful RNGs. — [Graphcore TF user guide](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/2.1.0/rand_and_fp.html). Each tile's PRNG is xoroshiro128aox, "capable of producing 64 bits of random data every cycle". — [Felix et al., arXiv 2203.04058](https://ar5iv.arxiv.org/html/2203.04058)
- **Philox-4×32 round counts and speed.** Full Crush-resistance was verified at 7 rounds (Philox-4×32-7). The authors recommend Philox-4×32-10, i.e. "three extra rounds of safety margin", for GPUs. Philox-4×32-7 produced 202 GB/s on a GTX580. — [Salmon et al., SC11 Random123](https://users.cs.utah.edu/~hari/teaching/bigdata/random123sc11.pdf)
- **GPU hash comparison.** Jarzynski & Olano (JCGT 2020) tested more than 100 GPU hashes against TestU01 BigCrush. They recommend `pcg` as "a good default choice" for 1D→1D and `pcg3d`/`pcg4d` for multi-output. `lcg` is fastest but "fails 90% of the BigCrush tests". The pcg4d code: `v = v*1664525u + 1013904223u; v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z; v ^= v >> 16u;` then the same multiply-add round again. — [Jarzynski & Olano, Hash Functions for GPU Rendering](https://jcgt.org/published/0009/03/02/paper.pdf)
- **Quartet II.** It reports speedups on RTX 5090 and B200 but does not describe its SR RNG or bit usage, and does not discuss hardware SR on sm_120. The cloned QuTLASS repo (commit e74319e, 2026-07-20) has no SR / RNG / `cvt.rs` code (grep). — [Quartet II, arXiv 2601.22813](https://arxiv.org/html/2601.22813); [QuTLASS](https://github.com/IST-DASLab/qutlass)

### Inferences
- **The 4-elements-per-word design.** Industry practice (TE, torchao, and the NVIDIA hardware rbits layout) is 8 bits per FP4 element, one 32-bit word per 4 elements, from a full-strength counter RNG. The project's current lowbias32-twice-per-pair at 16 bits per element spends about 4× more hash work per element than that design: 2 hashes per 2 elements versus 1 per 4.
- **AMD's LFSR step.** AMD's CDNA4 uses one LFSR step between paired FP4 elements, which shows a vendor accepting very weak per-element decorrelation inside a pair. It is not evidence of sufficient quality on its own, because no published quality study was found.

### Gaps
- No CDNA4 doc states how many of the 32 seed bits `f32_to_fp4_sr_scale` actually consumes.
- No source gives the minimum Philox round count that still passes BigCrush below 7. The Random123 paper only states that 7 passes.
- How many BigCrush tests `pcg4d` fails is read off a plot (it appears at or near 0) and was not confirmed in a table.

## Q3. How many random bits does SR need, and what are the bias implications?

### Takeaway
How the few random bits are added matters more than how many there are.
- **Naive N-bit SR ("SRFF").** Round up if δ + n·2^-N ≥ 1. It has a systematic bias of −2^-(N+1) ulp (toward zero, i.e. shrinking magnitudes). This matters in training: SRFF diverged in a GPT-2-scale QAT test.
- **The fix (SRF).** Add half an LSB: n + ½. It costs nothing, and the mean bias becomes zero.
- **Recommended budget.** With SRF, 8 bits per FP4 element leaves a worst-case per-value bias of 2^-9 step. Only values within 1/512 of a step of a grid point are always rounded deterministically. 8 bits per element is also what NVIDIA's hardware uses and what TE/torchao feed it.
- **Fewer bits.** 3–4 bits with SRF/SRC trained acceptably in the only published ML test (FP8-like weights, nanoGPT). But at 4 bits, any |δ| below 1/32 step can never round up, which defeats SR's purpose for small gradient entries.

### Cited Findings
- **Fitzgibbon & Felix (Graphcore), "On Stochastic Rounding with Few Random Bits" (ARITH 2025, arXiv 2504.20634).** Three few-bit variants:
  - SRFF: round up if δ + n·2^-N ≥ 1.
  - SRF: round up if δ + (n+½)·2^-N ≥ 1.
  - SRC: first round δ·2^N to nearest, then apply SRFF.

  Biases for infinite-precision input: SRFF −2^-(N+1); SRF 0. For finite-precision input with D excess bits: SRFF bias ≤ (2^-D − 2^-N)/2, which is zero when N = D; SRF bias ≤ 2^-(D+1); SRC ≈ 0. — [arXiv 2504.20634](https://arxiv.org/html/2504.20634v1)
- **The same paper's ML experiments.** Quantization-aware nanoGPT training rounded weights to Binary8P4 (precision equivalent to E4M3) with 3 random bits; 4 would be "ideal".
  - Small model (Shakespeare): RTNE stagnated, SRFF diverged, SRF/SRC converged.
  - GPT-2-scale model (354M parameters, 30k iterations): final loss SRFF 4.06, SRF 3.14, SRC 3.14, Binary16 2.74.
  - The authors state "even with 4 bits of randomness, the bias can significantly affect deep learning training".
  - They also note that almost all current implementations use N = number of trailing (discarded) significand bits.

  — [arXiv 2504.20634](https://arxiv.org/html/2504.20634v1); [code: graphcore-research/arith25-stochastic-rounding](https://arxiv.org/abs/2504.20634v1)
- **El Arar, Fasi, Filip & Mikaitis, "Probabilistic error analysis of limited-precision stochastic rounding" (arXiv 2408.03069, v3 Feb 2025).** Limited-precision SR with r random bits is biased, the bias depends on r, and it vanishes as r grows. They give probabilistic error bounds as a function of r. The abstract states no recommended r. — [arXiv 2408.03069](https://arxiv.org/abs/2408.03069)
- **Gupta et al. 2015.** 16-bit fixed point with SR trains close to fp32. Below 8 fractional bits even SR "is unable to fully prevent the loss of gradient information". This is about format precision, not random-bit count. — [Gupta et al., ICML 2015](https://proceedings.mlr.press/v37/gupta15.pdf)
- **NVIDIA's NVFP4 pretraining paper (arXiv 2509.25149).**
  - It recommends "stochastic rounding for gradients and round-to-nearest-even for weights and activations".
  - For the 12B model, SR "must be applied to gradients going into both Dgrad and Wgrad".
  - SR on forward tensors "is detrimental, as it amplifies quantization error relative to nearest rounding", and on activations/weights (1.2B model) "causes divergence".
  - It cites Blackwell's native SR for FP4 converts, which Q1 shows takes 8 rbits per e2m1 element.

  — [arXiv 2509.25149](https://arxiv.org/html/2509.25149)
- **This research's exact bias check (enumerating all 256 random bytes for 200k values uniform in [−6, 6]).**
  - Integer-domain 8-bit SR with the half-LSB offset (SRF): mean bias 0.000000 step, max per-value bias 0.001953 step (= 2^-9).
  - The same without offset (SRFF): mean bias −0.001956 step (toward zero), max 0.003906 step.
  - TE's software fallback: mean −0.001956 step, max 0.003906 step, i.e. statistically SRFF.
  - All outputs were on the E2M1 grid.

  — [bias_check.py](../code/bias_check.py)

### Inferences
- **What −2^-9 step means.** An SRFF bias of −2^-9 step is about −0.2% of a step. It is a magnitude shrink (toward zero, following sign(x)), not a direction error. In a GEMM-K reduction it acts roughly like a tiny, value-dependent gradient-scale reduction. Fitzgibbon & Felix show this bias class can hurt training, but their failure case was N = 3. At N = 8 the effect is 32× smaller and probably benign, but it is free to remove, so the SRF offset should be used.
- **Deterministic dead zone.** With N random bits and SRF, any fractional position δ < 2^-(N+1) step can never round up. At N = 8 the dead zone is 1/512 step, at N = 4 it is 1/32 step, and at the project's current N = 16 it is 2^-17. Gradient elements far below a block's step are exactly the ones SR exists to preserve in expectation. So going below about 6–8 bits for FP4 gradients is not recommended without an experiment, even though Fitzgibbon & Felix's 3-bit SRF worked for weight QAT.
- **Bits needed for other targets.** For FP8 E4M3/E5M2 the hardware uses 16 bits per element (Q1) and CDNA3 uses 20. The same arguments suggest 8 bits with SRF is also enough for FP8 gradients, but no study tests it.

### Gaps
- No paper was found that measures convergence of NVFP4 gradient SR as a function of random-bit count (for example 4, 6, 8 or 16 bits). All evidence for 8 bits is that NVIDIA's validated hardware recipe uses 8 bits.
- It is unverified whether NVIDIA's `cvt.rs` is SRFF or SRF/SRC internally.

## Q4. Alternatives: dithering with fixed or blue-noise patterns, shared randomness, SR only on some tensors

### Takeaway
- **SR only on gradients.** NVIDIA's recipe applies SR only to backward-pass gradient operands. SR on forward tensors is harmful, so the forward quantizations should be RN with no RNG cost.
- **Shared or structured randomness.** Sharing one random threshold across many elements, per token or as a fixed pattern, introduces a correlated offset that measurably degrades convergence unless explicitly corrected (S3R/S3R-EC, TPU INT8).
- **MS-EDEN (Quartet II).** It needs SR only on the per-16-element FP8 scales, combined with a randomized Hadamard transform, with RN on the FP4 elements. That is 16× fewer random draws and gives about 2.5× lower quadratic error than element-wise SR.
- **Blue-noise / ordered dither.** No training study was found.

### Cited Findings
- **NVIDIA NVFP4 recipe.** SR for gradients only. Forward SR "amplifies quantization error"; SR on weights/activations "causes divergence" at 1.2B. — [arXiv 2509.25149](https://arxiv.org/html/2509.25149)
- **Quartet II (arXiv 2601.22813).** MS-EDEN is "the only place where we require stochastic rounding is for the group scales". Quadratic error on N(0,1) (×10^-3): SR (1×16) 23.5; SR + 4/6 17.5; MS-EDEN 9.4; RTN 9.0. MS-EDEN has "more than 2x lower quantization error than SR". Variance from SR on the 8-bit scales is "an order of magnitude smaller" than the 4-bit quantization error. — [Quartet II](https://arxiv.org/html/2601.22813)
- **S3R (Wang et al., arXiv 2610.36587, Oct 2026).** It shares one threshold per token across the hidden dimension, reusing one quantization of the upstream gradient for both dgrad and wgrad. Uncorrected sharing "degrades convergence" through a common offset that mainly hurts via the activation gradient. The S3R-EC correction matches independent SR's validation loss and gives a 1.11× step speedup over independent SR (INT8 on TPU v5p). — [arXiv 2610.36587](https://arxiv.org/abs/2610.36587)
- **Dither as SR.** Dithering (adding uniform noise, then round-to-nearest) is a standard way to implement SR. Deterministic "dither rounding" has similar cost to SR but needs index bookkeeping. — [Dither computing, arXiv 2102.10732](https://arxiv.org/abs/2102.10732)
- **Fixed-pattern cases in shipped stacks.** None. TE/torchao use a fresh Philox counter per element per step, and torchao deliberately keeps the row- and column-quantization random streams independent. — [torchao hadamard_quantize_row_col_triton.py](https://github.com/pytorch/ao/blob/main/torchao/prototype/moe_training/nvfp4_training/hadamard_quantize_row_col_triton.py)

### Inferences
- **Fixed patterns across steps.** A dither pattern fixed across steps (same noise for the same tensor position every iteration) turns SR into a deterministic, position-dependent rounding. For slowly varying gradients the per-position error then stops averaging out over steps. A seed that changes every step is necessary, and the project already has one (a GPU-resident seed).
- **Blue noise within a step.** Spatially decorrelated noise (for example a 16-element permutation of 16 evenly spaced thresholds per block, a stratified "Latin" dither) keeps per-element marginals uniform, so each element stays unbiased if the pattern's offset is re-randomized each step. It also lowers the block-sum variance. This is plausible but no ML evidence was found. S3R's result warns that correlated thresholds across a reduction dimension need care.
- **Sharing randomness across the dual quantizer's outputs.** One word could drive both the row-wise (dgrad operand) and column-wise (wgrad operand) quantization of the same element. That halves RNG cost while keeping each output unbiased, because the two outputs use different block scales, so the rounding decisions differ even with the same u. It does correlate the errors of the dgrad and wgrad operands. No study evaluates it, and torchao explicitly avoids it.
- **MS-EDEN fits this codebase.** The project already applies a randomized Hadamard transform (RHT) per 16-element block (`rht_sign_mask` in `csrc/nvfp4_common.cuh`). That makes MS-EDEN-style "RN elements + SR only on scales + correction" a strong candidate for removing per-element RNG almost entirely: 1 random draw per 16 elements. It needs Quartet II's correction math, not just a rounding swap.

### Gaps
- No training study of blue-noise, stratified or ordered dither for NVFP4/FP8 gradients was found.
- No study evaluates sharing random bits between the row- and column-quantized copies of a gradient.

## Q5. Technique catalogue: cost per element on sm_120a and quality

### Takeaway
On sm_120a the project's current SR adds about 28.6 SASS instructions per element over the RN path:
- about 9.5 for the RNG (two nested lowbias32 per pair, 16-bit uniforms);
- about 19 for the float-domain SR math.

Two changes stack:
1. **RNG.** 8 bits per element from one lowbias32 per 4 elements (≈2.8 instr/elt), pcg4d per 16 elements (≈1.3 instr/elt) or Philox4x32-7/-10 per 16 elements (≈2.4 / ≈3.5 instr/elt).
2. **SR math.** An integer-domain "add random bits to the discarded fp32 mantissa bits, mask, exact cvt.rn" rounder costs about 9 instr/elt, down from about 19.

Combined they cut the SR overhead from about 28.6 to about 10–12 instructions per element, roughly 2.4–2.8× less. The half-LSB offset keeps the mean bias at zero, and quality equals hardware-style 8-bit SR. These are static instruction counts, not timings.

### Cited Findings
- **SASS counts** (CUDA 13.0.88, `-O3 -arch=sm_120a`). Each kernel loads 16 fp32 values and writes 8 bytes. The table shows total SASS instructions per kernel, and extra instructions per element over the RN kernel (41 total). — [sr_variants.cu / count.sh](../code/sr_variants.cu)

| Kernel (RNG + SR math) | Total | Extra / element | IMAD/IMUL count |
|---|---|---|---|
| RN only (`cvt.rn.satfinite.e2m1x2.f32`) | 41 | 0 | 6 |
| **Current project**: 2× lowbias32 per pair, 16-bit u, float SR (`e2m1_sr_value`) | 498 | 28.6 | 38 |
| Current float SR, no RNG (isolates SR math) | 346 | 19.1 | 6 |
| TE software SR, no RNG | 389 | 21.8 | 6 |
| Integer-domain SR (`int_sr`), no RNG | 186 | 9.1 | 6 |
| Current RNG + `int_sr` | 338 | 18.6 | 46 |
| 1× lowbias32 per 4 elements (8 bits) + current float SR | 391 | 21.9 | 16 |
| 1× lowbias32 per 4 elements (8 bits) + `int_sr` | 231 | 11.9 | 16 |
| pcg4d per 16 elements (8 bits) + `int_sr` | 207 | 10.4 | 17 |
| Philox4x32-7 per 16 elements + `int_sr` | 224 | 11.4 | 23 |
| Philox4x32-10 per 16 elements + `int_sr` | 242 | 12.6 | 29 |
| Philox4x32-10 + TE software SR (TE's sm_110 path) | 437 | 24.8 | 25 |

- **RNG cost per element by difference** (approximate, since the "no RNG" kernels keep a few bit-extract shifts): current ≈ 9.5; 1× lowbias32 per quad ≈ 2.8; pcg4d ≈ 1.3; Philox-7 ≈ 2.4; Philox-10 ≈ 3.5. On sm_120a Philox compiles to one `IMAD.WIDE.U32` (both hi and lo) per multiply: 29 IMAD for 10 rounds plus addressing. — [count.sh output](../code/count.sh)
- **The integer-domain rounder** whose SASS was counted and whose bias was checked exactly (SRF: mean bias 0, per-value bias ≤ 2^-9 step, outputs always on the E2M1 grid). About 9 SASS instructions per element: FSET.BF, FADD, SHL+LOP3 (byte extract), IADD3 (adds the random bits and the 0x2000 offset in one instruction), LOP3 mask, FADD, FMNMX, LOP3 copysign. — [sr_variants.cu](../code/sr_variants.cu); [bias_check.py](../code/bias_check.py)

```cuda
// x already divided by the block scale. r8: 8 random bits. Output is an exact E2M1 grid value,
// so cvt.rn.satfinite.e2m1x2.f32 afterwards is exact. |x|<1 is lifted into [1,2), where keeping
// 1 fp32 mantissa bit gives spacing 0.5 = E2M1's spacing on [0,2); carries cross binades correctly.
__device__ __forceinline__ float int_sr(float x, uint32_t r8) {
  const float a = fabsf(x);
  const float off = a < 1.0f ? 1.0f : 0.0f;
  const float z = a + off;                                  // exact to 2^-23
  uint32_t b = __float_as_uint(z) + ((r8 & 0xFFu) << 14)   // 8 random bits on top of the 22 dropped bits
                                  + (1u << 13);              // +1/2 LSB of the random field (SRF: zero mean bias)
  b &= 0xFFC00000u;                                         // keep sign/exponent/1 mantissa bit
  const float q = fminf(__uint_as_float(b) - off, 6.0f);    // fminf optional: cvt .satfinite clamps
  return copysignf(q, x);
}
```

- **Quality of the RNG options.** Philox4x32-7 is Crush-resistant and -10 is the recommended GPU default ([Random123](https://users.cs.utah.edu/~hari/teaching/bigdata/random123sc11.pdf)). pcg and pcg4d are on Jarzynski & Olano's speed/quality Pareto frontier and are their default recommendations; LCGs fail about 90% of BigCrush ([JCGT 2020](https://jcgt.org/published/0009/03/02/paper.pdf)). The project's lowbias32 is a multiply-xorshift integer hash. The JCGT study did not test it, so its BigCrush standing comes from no source here.

### Inferences
- **Recommended drop-in for the project's quantizer:**
  1. Fold the 64-bit seed into a per-thread key once, not per pair.
  2. Use one 32-bit hash per 4 elements, or one pcg4d / Philox-7 call per 16-element NVFP4 block, taking 8 bits per element.
  3. Replace `e2m1_sr_value` with `int_sr` (SRF offset).

  Expected SR overhead falls from about 28.6 to about 10–12 instructions per element. If the dual quantizer's +34–43% runtime is roughly proportional to added ALU work (plausible, since the hash is IMAD-heavy and the RN kernel is memory-bound), this would cut the overhead to roughly a third or less. It must be measured, though: this is a static count, not a timing.
- **It changes bits.** It changes the bit-exact output relative to the project's 16-bit reference (`reference.sr_uniform2` / `e2m1_sr_value`), so the Python reference must switch to the same 8-bit SRF rule.
- **Further trims.** Drop `fminf` (the convert's `.satfinite` already clamps values above 6). Generate random bytes already positioned (PRMT/LEA) to save about 1 instruction per element. Reuse one random word for the row and column outputs (halves RNG cost; see Q4 caveat). Or use MS-EDEN to remove per-element RNG entirely.
- **Below 8 bits.** With N = 4 bits per element, one 32-bit word covers 8 elements. That saves only about 1 instruction per element more with pcg4d, while widening the deterministic dead zone to 1/32 step (Q3). The cost/benefit does not favour going below 8 bits.
- **FP8 targets.** The same integer trick works, with only normal/subnormal boundary handling differing: E4M3 keeps 3 mantissa bits, so shift the random field to the top of the 20 dropped bits. Use 8–16 random bits.

### Gaps
- No wall-clock measurement on the RTX 50 laptop was done in this research (static counts only). SASS pipe mix matters: IMAD issue rate on sm_120 versus FP32/LOP3 was not found in NVIDIA documentation.
- No public measured cost was found for TE's software SR path, Philox-10 on consumer Blackwell, or the hardware `cvt.rs` throughput on sm_100a, to compare against.
