# Grouped GEMM for MoE training in NVFP4 / MXFP4 / FP8 on consumer (sm_120, sm_89) and datacenter Blackwell — state as of Nov 2026

Scope note: researched 2026-11-24. Most numbers are vendor- or author-reported and measured on B200/B300/GB200 (sm_100/sm_103), not sm_120. Where a claim comes from a single community issue report it is flagged as such.

## 1. How do existing systems implement low-precision MoE grouped GEMMs (TE, DeepGEMM, CUTLASS, Megablocks/ScatterMoE/SonicMoE, torchao, vLLM/SGLang/FlashInfer)?

### Takeaway
Every production system converged on one pattern: sort/permute tokens so each expert's rows are contiguous, describe groups with a device-side offsets tensor (no host sync), and run a single persistent grouped kernel where only M varies per group (forward/dgrad) or only K varies per group (wgrad). Low-precision variants add per-group quantization kernels that write scale factors in the tensor-core "blocked/swizzled" layout per group. For sm_120 specifically, upstream support for low-precision grouped GEMM arrived late and is still uneven: CUTLASS added sm_120 NVFP4/MX grouped kernels in the profiler in 4.4.2 (Mar 2026) and FP8 tensor/token-scaled ptr-array grouped GEMM for sm_120 only in 4.7.0 (Aug 2026), while vLLM/FlashInfer FP4 MoE on sm_120 was reported broken or slow for much of 2026.

### Cited Findings

**Transformer Engine (NVIDIA)**
- `GroupedLinear` (in `transformer_engine.pytorch.ops`) takes `num_groups, in_features, out_features, bias, dtype, device`; each expert keeps its own weight tensor; forward takes the token batch plus per-expert token counts `split_sizes` and "submits the local experts through the TE grouped GEMM path", cutting per-expert launch overhead — [NVIDIA blog: Efficient MoE training for biological foundation models](https://developer.nvidia.com/blog/efficient-moe-training-for-biological-foundation-models/)
- `Sequential(GroupedLinear(gate_up), ScaledSwiGLU(), GroupedLinear(down))` is pattern-matched and replaced by a fused `ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8` op (plus fused backward); `ScaledSwiGLU` applies routing probabilities; the fused path avoids materializing intermediates; Blackwell required for the fused MXFP8 GroupedMLP kernel — [NVIDIA blog](https://developer.nvidia.com/blog/efficient-moe-training-for-biological-foundation-models/)
- TE v2.16 release notes: "Expanded grouped GEMM support with NVFP4 on Blackwell and FP8 block scaling on Hopper" (#2971); fixed cuBLAS grouped GEMM when weight dims not divisible by 128 (#2954); variable grouped swizzle support for grouped tensor scale layouts (#2914); CUDA Graph capture for GroupedLinear/grouped MoE ops (#2923); reduced GroupedLinear CPU overhead (#2900, #2957, #2666); row-scaled NVFP4 forward recipe (#2931); `moe_permute` int32 overflow fix (#2907); fused top-k routing op (#2890). No release-note item mentions SM120/RTX 50/GeForce — [TE 2.16 release notes](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.16/release-notes/_sources/index.md.txt)
- TE 2.18 user guide documents an NVFP4 training recipe, `is_nvfp4_available()`, and MoE permute helpers such as `moe_permute()` — [TE 2.18 low-precision training docs](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.18/user-guide/features/low_precision_training/index.html)
- Megatron Core runs all local experts in a single grouped GEMM via `TEGroupedMLP` (TE implementation with FP8 and FP4 support); router can run in FP32 (`--moe-router-dtype fp32`); paper has sections titled "fusion of padding and unpadding", "grouped quantization with grouped GEMM", and "NVFP4 quantization fusion" (body of those sections not retrievable this session) — [Megatron Core MoE report, arXiv 2603.07685](https://arxiv.org/html/2603.07685)

**DeepGEMM (DeepSeek)**
- Three grouped layouts: `m_grouped_fp8_gemm_{nt,nn}_contiguous` (training fwd / prefill; tokens of all experts concatenated; "DeepGEMM groups only the M-axis, while N and K must remain fixed"); `m_grouped_fp8_gemm_nt_masked` (CUDA-graph decode; mask says which rows are valid; consumes DeepEP low-latency output); `k_grouped_fp8_gemm_tn_contiguous` (MoE weight backward, M and N fixed, K grouped) — [DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM)
- Each expert segment in the contiguous layout must be padded to the GEMM M block size, queried with `get_mk_alignment_for_contiguous_layout()` (also `set_...` and `get_theoretical_...` variants); LHS scale factors must be TMA-aligned and transposed — [DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM)
- Arch support: SM90 and SM100 only; SM120 not mentioned. SM90 scales are FP32; SM100 scales are packed UE8M0 (4 per int32). FP8xFP4 "Mega MoE" added 2026-04-16 (PR #304); wgrad kernels for dense and MoE added 2025-05-14 (PR #95) — [DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM)

**CUTLASS**
- 4.2.0 (2025-09-15): example 92 MoE grouped GEMM uses TMA for weights and cp.async for tokens with only one dimension varying across groups; "Support Blackwell SM120 mixed input blockscaled grouped GEMM"; SM120 blockwise GEMM example 87; SM121 (DGX Spark) shares code with SM120 — [CUTLASS CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)
- 4.3.0/4.3.1 (Nov 2025): "Ragged Contiguous Grouped gemm" kernel and simplified MoE GEMM API in example 92, then its blockscaled variant "for all microscaling types"; GEMM_K=0 allowed in grouped GEMM; async TMA descriptor update for grouped GEMM — [CUTLASS CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)
- 4.4.0 (2026-02-14): fixes "missing SMEM alignment in Blackwell SM120 scale factors" and "memory fence for clc scheduler in Blackwell SM120 pingpong kernel" (i.e. SM120 kernels use the CLC dynamic tile scheduler); grouped GEMM PDL and cluster-swizzle fixes; CuTe DSL grouped GEMM supports device-only problem shapes — [CUTLASS CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)
- 4.4.2 (2026-03-13): "Enable Blackwell SM120f compilation of examples and exposes NVFP4/MX Grouped GEMM in the CUTLASS Profiler" — [CUTLASS CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)
- 4.5.0 (2026-05-01): 128x32xK and 128x64xK tile sizes for SM120 blockscaled MMA collective builders ("up to 30% gain on SM121"); CuTe DSL block-scaled MMA for SM120 works on Spark; CuTe DSL torch-`grouped_mm`-style MoE grouped GEMM with B200 speedups vs torch 2.10 cu13: mxfp8 2Dx3D avg 1.29x, mxfp8 2Dx2D avg 1.41x, nvfp4 2Dx3D avg 1.11x, nvfp4 2Dx2D avg 1.12x (worst 0.98x) — [CUTLASS CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)
- 4.5.1 (2026-05-15): "Fixed issues for SM120 blockscaled MMAs" (missing MXF8F6F4 MMA ops for sm120 in CuTe DSL) — [CUTLASS CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)
- 4.7.0 (2026-08-04): "Add ptr-array TMA collective for tensor/token-scaled FP8 grouped GEMM Blackwell SM120/SM121 kernels"; tileN = 8, 16 for SM120 blockscale GEMM — [CUTLASS CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)
- 4.8.0 (2026-09-17): CuTe extensions "Unscaled and blockscaled grouped GEMM", GeForce pingpong dense GEMM example; Operator API adds `IndexPtrGroupedGemmArguments`, deprecates `GroupedGemmArguments` — [CUTLASS CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)
- 4.9.0 (2026-11-13): Operator API "block-scaled grouped GEMMs for MXFP8, MXFP4, and NVFP4, including per-expert global scales for NVFP4", plus "MoE weight-gradient GEMM" and row/column-major layouts; example notebook `010_moe_grouped_gemm.ipynb` (the changelog does not say whether this Operator API path covers SM120) — [CUTLASS CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)
- A vLLM issue reports CUTLASS 4.5 ships SM120 examples for blockwise FP8, NVFP4 and MXFP8 grouped GEMM (79d_blackwell_geforce_nvfp4_grouped_gemm.cu, 87c blockwise FP8 grouped) — single-source claim — [vLLM issue #43507](https://github.com/vllm-project/vllm/issues/43507)

**torchao (PyTorch)**
- `_to_mxfp8_then_scaled_grouped_mm(A, B.transpose(-2,-1), offs)` is a differentiable drop-in for `torch._grouped_mm`; `offs` are token-group end offsets produced by the router/permute; model conversion via `quantize_` with `MXFP8TrainingOpConfig` filtered to `"experts"`; TorchTitan flag `--model.converters="quantize.grouped_mm.mx"` — [torchao moe_training README](https://github.com/pytorch/ao/tree/main/torchao/prototype/moe_training)
- Requirements: MXFP8 needs CUDA 12.8+ and SM100+; FP8 rowwise needs CUDA 12.4+ and SM89+; torchao 0.14+. An `nvfp4_training/` directory and `nvfp4_grouped_mm.py` exist — [torchao moe_training README](https://github.com/pytorch/ao/tree/main/torchao/prototype/moe_training)
- Kernels: Triton `triton_to_mxfp8_dim0`, `triton_mx_block_rearrange_*`; CUDA `mx_block_rearrange_2d_M_groups_cuda`, `to_mxfp8_dim1_cuda`, `mxfp8_quantize_cuda_3d`; GEMMs via `torch._scaled_grouped_mm` — [torchao moe_training README](https://github.com/pytorch/ao/tree/main/torchao/prototype/moe_training)
- NVFP4 MoE plan: grouped GEMMs 1x16 @ 16x16 for output and dgrad, 1x16 @ 16x1 for wgrad; 16x16 scaling for 3D expert weights; per-group conversion to blocked swizzled scale format for groups along M, along K, and for 3D weights (reusing MXFP8 kernels); RHT on wgrad inputs at 16x16; SR for gradients. Tracker closed 2026-10-07 as complete via torchtitan #4821; MoE GroupedExperts on Blackwell in ao #4598; CuTe DSL RHT kernels in ao #4487. No SM120 target mentioned — [torchao issue #3293](https://github.com/pytorch/ao/issues/3293)
- NeMo AutoModel's MXFP8 MoE path requires compute capability >= 10 and torchao; otherwise callers fall back to BF16 `torch._grouped_mm` — [NeMo AutoModel mxfp8 docs](https://docs.nvidia.com/nemo/automodel/nemo-automodel/nemo_automodel/components/moe/mxfp8)

**SonicMoE (Tri Dao group) / QuACK**
- Forward uses varlen-M grouped GEMM (ragged tokens per expert, shared N/K); weight gradient uses varlen-K grouped GEMM (reduce over tokens, fixed M/N); single producer-consumer mainloop with customizable epilogue, built on QuACK (CuTe DSL) — [SonicMoE Blackwell blog](https://tridao.me/blog/2026/sonicmoe-blackwell/)
- Token gather is fused into the GMEM->SMEM load (cp.async or TMA gather4, autotuned, within ~2% of each other), so gathered X and dO are never materialized; better L2 locality than a separate gather kernel (2.20 GB vs 2.68 GB HBM loads in one ablation) — [SonicMoE Blackwell blog](https://tridao.me/blog/2026/sonicmoe-blackwell/)
- 8 kernels total for fwd+bwd (up-proj, down-proj, aggregation in fwd; activation-grad and weight-grad in bwd); 45% less activation memory and 1.86x compute throughput vs ScatterMoE BF16 on H100 for a fine-grained 7B MoE; "token rounding" routing rounds per-expert token counts to multiples of the grouped-GEMM tile (e.g. 128) — [SonicMoE arXiv 2512.14080](https://arxiv.org/html/2512.14080v1)
- Only BF16 kernels described; MXFP8/MXFP4 listed as future work. SM120 status is inconsistent in the blog (both "on the way" and "added via a ~500-line base GEMM class") — [SonicMoE Blackwell blog](https://tridao.me/blog/2026/sonicmoe-blackwell/)
- PyPI release 2026-04-19 adds Blackwell SM100 support built on QuACK grouped GEMM — [sonic-moe on PyPI](https://pypi.org/project/sonic-moe/)

**Megablocks / ScatterMoE** (not re-fetched this session; from paper abstracts)
- MegaBlocks reformulates MoE as block-sparse matmuls so no tokens are dropped and no capacity padding is needed — [MegaBlocks, arXiv 2211.15841](https://arxiv.org/abs/2211.15841)
- ScatterMoE fuses the gather/scatter of tokens into Triton GEMM kernels ("ParallelLinear"), avoiding padding and extra copies of the input — [ScatterMoE, arXiv 2403.08245](https://arxiv.org/abs/2403.08245)

**vLLM / SGLang / FlashInfer on sm_120 (inference, but the same kernels)**
- FlashInfer issue (Mar 2026, 4x RTX PRO 6000, CUTLASS 4.2.1/4.4.1): vLLM native CUTLASS grouped block-scaled FP4 MoE GEMM gives garbage output while dense FP4 GEMM works; pingpong schedule with `compute_120a` segfaults; building FlashInfer CUTLASS MoE for `compute_120f` with CUDA 13.0 enabled TMA grouped tactics: 39 tok/s vs 14.6 tok/s with `compute_120a`; Marlin W4A16 still 46–49 tok/s and recommended. Closed 2026-03-14 citing PR #2725. Root cause not confirmed — [FlashInfer issue #2723](https://github.com/flashinfer-ai/flashinfer/issues/2723)
- A forum post attributes SM120/SM121 grouped-GEMM problems to SMEM (99 KB vs 228 KB on SM100) and proposes K=64 tile shapes (tested only on SM121), citing CUTLASS PR #3120 / issue #3096 — [NVIDIA forum: state of NVFP4 on DGX Spark](https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069/131)
- vLLM (issue opened 2026-05-23, still open/stale 2026-11-13): FP8 tensor/token-scaled MoE on SM120/121 falls back to Triton because vLLM only has `grouped_mm_c3x_sm90.cu` and `grouped_mm_c3x_sm100.cu`; NVFP4 `--moe-backend cutlass` ran a blockscaled FP4 grouped kernel on an RTX 5090 Laptop with a locally modified build (Aug 2026); FlashInfer MXFP4 MoE runner is used on SM120 in SGLang; a fix commit "[Bugfix][SM120] Enable CUTLASS grouped GEMM (MoE) for SM_120/SM_121" (2026-11-16) was linked but not shown merged — [vLLM issue #43507](https://github.com/vllm-project/vllm/issues/43507)

### Inferences
- The "contiguous M-grouped" layout (DeepGEMM contiguous, CUTLASS example 92 ragged contiguous, torch `_grouped_mm` 2D x 3D with `offs`) is the common denominator; pointer arrays (CUTLASS ptr-array) are only needed when experts' weights are not in one strided 3D tensor or when N/K differ per group, which is not the case in standard MoE.
- No system I found ships an sm_120 NVFP4 *training* MoE path (fwd+dgrad+wgrad) end to end. torchao NVFP4 MoE targets "Blackwell" (B200 benchmarks); TE does not list sm_120 for grouped NVFP4; DeepGEMM and SonicMoE are SM90/SM100. A custom sm_120 implementation is likely necessary, with CUTLASS 4.4.2+ sm_120 NVFP4 grouped kernels and CUTLASS 4.9 MoE wgrad GEMM as the closest references.
- The sm_120 failures reported in 2026 (garbage output, segfaults) are consistent with smem-tight tile configs and pingpong/CLC scheduler bugs that CUTLASS fixed incrementally (4.4.0 smem-alignment and CLC-fence fixes, 4.5.0 128x32/128x64 tiles). Anyone building on CUTLASS sm_120 grouped kernels should use >= 4.5 and build `sm_120f`/`compute_120f` with CUDA 13.

### Gaps
- Could not confirm whether the CUTLASS 4.9 Operator API block-scaled grouped GEMM and MoE wgrad GEMM run on SM120 or only SM100/SM103.
- Could not read the Megatron Core MoE report's Section 5 (FP8/FP4 recipes, padding fusion, NVFP4 quantization fusion) — HTML truncated and PDF too large.
- No source found on TE GroupedLinear NVFP4 actually running on sm_120 (RTX 50).

## 2. Layout choices: contiguous+padding vs pointer arrays, persistent tile scheduler, and low-precision wgrad per expert

### Takeaway
Use contiguous per-expert segments with a device-side offsets tensor, padded so every expert segment starts on a scale-factor/tile boundary (128 rows is the common choice because the block-scaled scale layout is tiled in 128x4 blocks). Forward and dgrad are "2D x 3D" (groups along M); wgrad is "2D x 2D" with groups along the contraction (token) dimension, which forces a second quantization of X and dY along the token axis (1x16 along tokens for NVFP4, 1x32 for MXFP8) and per-group variable-stride scale layouts. wgrad is consistently the least efficient of the three GEMMs.

### Cited Findings
- Tokens are shuffled so groups are sorted in expert-weight order; the shuffle produces an offsets tensor with the end index of each group — [PyTorch blog: MXFP8 MoE training](https://pytorch.org/blog/mxfp8-training-for-moes-1-3x-training-speedup-vs-bf16-for-llama4-scout-on-gb200-cluster-using-torchao-and-torchtitan/)
- Scale factors must be in the blocked layout: pad each token group's scales to multiples of 128x4 tiles, rearrange each 128x4 tile to ((32,4),4), write tiles "row of blocks" major; must be done per group because the data and scales of all groups live in one tensor but each must be individually valid for the MMA — [PyTorch blog](https://pytorch.org/blog/mxfp8-training-for-moes-1-3x-training-speedup-vs-bf16-for-llama4-scout-on-gb200-cluster-using-torchao-and-torchtitan/)
- Group sizes are known only on device; reading them on host would need a host-device sync, so the per-group layout conversion is a custom device-side kernel — [PyTorch blog](https://pytorch.org/blog/mxfp8-training-for-moes-1-3x-training-speedup-vs-bf16-for-llama4-scout-on-gb200-cluster-using-torchao-and-torchtitan/)
- dgrad = dO @ W is the same 2D x 3D grouped GEMM; dO is quantized like the forward activations, and weights are written to per-expert column-major — [PyTorch blog](https://pytorch.org/blog/mxfp8-training-for-moes-1-3x-training-speedup-vs-bf16-for-llama4-scout-on-gb200-cluster-using-torchao-and-torchtitan/)
- wgrad dW = dO^T @ X is a 2D x 2D grouped GEMM (N, total_M) @ (total_M, K) with groups along the contracting dimension; with row-major scales, slicing along K makes strides dynamic per group (stride = stride per tile x number of tiles along a row in that group) — [PyTorch blog](https://pytorch.org/blog/mxfp8-training-for-moes-1-3x-training-speedup-vs-bf16-for-llama4-scout-on-gb200-cluster-using-torchao-and-torchtitan/)
- torchao: wgrad path quantizes grad_out and A along M (token dim); 2D-2D wgrad speedups are much lower than output/dgrad (52.1% of peak TFLOPS); large `total_M` recommended — [torchao moe_training README](https://github.com/pytorch/ao/tree/main/torchao/prototype/moe_training)
- NVFP4 recipe: 1x16 scaling for LHS activations, 16x1 for RHS activations (wgrad), 16x16 for weights; output/dgrad = 1x16 @ 16x16, wgrad = 1x16 @ 16x1; RHT on wgrad inputs — [torchao issue #3293](https://github.com/pytorch/ao/issues/3293)
- Nemotron 3 NVFP4 pretraining applies RHT to wgrad inputs, 2D block scaling to weights, and stochastic rounding to gradients — [Nemotron 3 report, arXiv 2512.20856](https://arxiv.org/pdf/2512.20856)
- DeepGEMM provides a separate `k_grouped_fp8_gemm_tn_contiguous` kernel for MoE weight backward (M, N fixed; K grouped) — [DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM)
- SonicMoE uses varlen-K grouped GEMM for weight grads — [SonicMoE Blackwell blog](https://tridao.me/blog/2026/sonicmoe-blackwell/)
- CUTLASS added "MoE weight-gradient GEMM" to the Operator API in 4.9.0 and "Enable GEMM_K = 0 in grouped gemm" in 4.3.0 (needed for experts with zero tokens in a K-grouped wgrad) — [CUTLASS CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)
- Persistent scheduling: SonicMoE uses a static persistent scheduler on Hopper and Cluster Launch Control (hardware dynamic scheduler, no global atomics) on Blackwell; CLC + 2CTA MMA gives ~10% higher varlen-M throughput than DeepGEMM and the Triton example — [SonicMoE Blackwell blog](https://tridao.me/blog/2026/sonicmoe-blackwell/). CUTLASS's SM120 pingpong kernel also uses a CLC scheduler (4.4.0 fence fix) — [CUTLASS CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)
- miniTE uses contiguous buffers with offsets instead of per-expert tensor lists (avoids slicing and H2D metadata); zero-token experts and padding handled in the same grouped path; its production per-token path requires `m_splits` multiples of 128 — [miniTE blog](https://lmsys.org/blog/2026-10-25-minite)
- SonicMoE "token rounding" rounds per-expert token counts to tile multiples (e.g. 128) at routing time instead of padding — [SonicMoE arXiv](https://arxiv.org/html/2512.14080v1)
- CUTLASS 4.9.0 grouped NVFP4 supports "per-expert global scales" (the NVFP4 second-level FP32 tensor scale, one per expert) — [CUTLASS CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)

### Inferences
- For a single-GPU sm_120 library: one 3D weight tensor [E, N, K] (quantized once per step with 16x16 2D blocks so the same FP4 copy serves forward (K-major) and dgrad (N-major) after a cheap transposed repack) plus a contiguous [total_M_padded, K] activation tensor and an int32 offsets[E+1] tensor is sufficient; pointer arrays add no value.
- Padding each expert segment to 128 rows lets forward/dgrad reuse the existing dense NVFP4 kernel's scale layout per tile unchanged and makes wgrad's K-groups 16-aligned for the 16x1 token-dim scales. Cost: up to 127 wasted rows per expert, which matters with many experts and few tokens per expert on one GPU (e.g. 64 experts x 127 rows); a tile list built on device from offsets (expert_id, m_tile) lets the kernel skip all-padding tiles.
- Because 16x1 token-dim quantization of X and dY for wgrad is a second pass over those tensors, the cheapest place to produce it is the same kernel that produces the 1x16 (K-dim) copy (read once, emit both), or the GEMM epilogue that writes dY/X; with RHT on wgrad inputs, this kernel also applies the 16x16 Hadamard.
- The per-expert NVFP4 global (FP32) scale requires a per-expert amax; computing it during permutation (see section 3) avoids a separate amax pass.

### Gaps
- No published sm_120-specific numbers for 2D x 2D (K-grouped) NVFP4 wgrad efficiency.
- No source quantified the padding waste vs a non-padded (masked-tile) approach at small per-expert token counts.

## 3. Fusions: permute + quantize, unpermute + combine, routing — which give the biggest wins?

### Takeaway
Measured wins come mostly from (a) fusing token gather into the GEMM load or into the quantizer so the permuted BF16 copy is never materialized, (b) fusing activation (SwiGLU, routing-weight scaling) and amax/quantization into the FC1 epilogue, and (c) keeping quantization/scale-layout kernels fast enough not to eat the low-precision GEMM speedup. Fusing scatter/combine into the GEMM epilogue was found not worth it on Hopper; a separate bandwidth-bound gather-and-sum kernel is used instead.

### Cited Findings
- SonicMoE: gather fused into GMEM->SMEM load; SwiGLU/dSwiGLU applied in registers in the epilogue; aggregation (unpermute+combine) is a separate Triton gather-and-sum kernel reaching >6.5 TB/s on B300 (~85% of peak) because fusing scatter into the GEMM epilogue with synchronous `st.global` cost ~20% TFLOPS on Hopper; backward dH kernel computes dH, dS and A' together with an algebraic reordering so dS needs neither cached Y nor dY — [SonicMoE Blackwell blog](https://tridao.me/blog/2026/sonicmoe-blackwell/)
- SonicMoE reduces activation memory 45% vs ScatterMoE — [SonicMoE arXiv](https://arxiv.org/html/2512.14080v1)
- TE fuses gate-up GroupedLinear + ScaledSwiGLU (with routing-probability scaling) + down GroupedLinear into `ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8` — [NVIDIA blog](https://developer.nvidia.com/blog/efficient-moe-training-for-biological-foundation-models/)
- miniTE: rowwise amax accumulated during token dispatch/permutation, then padding processes activations and amax together so quantization does not rescan; FC2 input amax fused with weighted SwiGLU (`_weighted_swiglu_row_amax_kernel`); a CuTeDSL SM100 prototype that avoids writing the FC1 intermediate (accumulating FC2 partials under a running row scale) was ~4–8x faster than the unfused TE two-level chain on a B200 shape test. However, skipping the amax pass alone was neutral to slightly slower (~0.81–1.11x) in a module simulation — [miniTE blog](https://lmsys.org/blog/2026-10-25-minite)
- miniTE: host-side overhead matters — cuBLASLt heuristic calls ~4.6x higher (148,000 vs 32,000) and `nvjet` kernel calls ~25x in one profile of the TE NVFP4 path — [miniTE blog](https://lmsys.org/blog/2026-10-25-minite)
- torchao: when GEMM dims are small (smaller MoE models), quantization overhead can match or exceed the MXFP8 GEMM speedup — [PyTorch blog: DeepSeek-V3 MXFP8 + DeepEP](https://pytorch.org/blog/enabling-up-to-41-faster-pre-training-mxfp8-and-deepep-for-deepseek-v3-on-b200-with-torchtitan/)
- torchao: scale-layout transformation "will need to be done very carefully to avoid incurring excessive overhead that will kill our net speedup (or worse, cause a slowdown!)" — [PyTorch blog](https://pytorch.org/blog/mxfp8-training-for-moes-1-3x-training-speedup-vs-bf16-for-llama4-scout-on-gb200-cluster-using-torchao-and-torchtitan/)
- Megatron Core lists "fusion of padding and unpadding" and "NVFP4 quantization fusion" as MoE-specific optimizations (details not retrieved) — [Megatron Core MoE report](https://arxiv.org/html/2603.07685)
- TE added a fused top-k routing op (v2.16, #2890) and an improved fused MoE aux-loss kernel (#2758) — [TE 2.16 release notes](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.16/release-notes/_sources/index.md.txt)
- ScatterMoE fuses gather/scatter into its Triton GEMMs to avoid padding and copies — [ScatterMoE, arXiv 2403.08245](https://arxiv.org/abs/2403.08245)

### Inferences
- For NVFP4 specifically (unlike BF16 SonicMoE), gather-in-GEMM is less natural: the GEMM consumes already-quantized FP4 + swizzled scales, so the gather must happen at quantization time. The equivalent fusion is "permute + quantize": a kernel that reads BF16 tokens by the sorted index, computes per-16 block scales (and per-expert amax for the global scale), writes FP4 data into the expert-contiguous padded layout and scales directly in the swizzled per-group layout. This replaces three passes (permute copy, amax, quantize) with one read of X.
- Highest expected win on one consumer GPU: (1) permute+quantize fused (removes a BF16 copy of top-k x tokens x hidden), (2) SwiGLU + routing-weight + FC2-input quantization in the FC1 epilogue (removes the largest intermediate), (3) a separate gather-and-sum combine kernel (bandwidth-bound, simple). Fusing combine into the FC2 epilogue is lower priority given SonicMoE's 20% finding.
- CPU/launch overhead (miniTE's heuristic-call finding, TE's CPU-overhead fixes and CUDA-graph support) argues for one persistent kernel per GEMM type with device-side offsets and CUDA-graph capture.

### Gaps
- No sm_120 measurements of any of these fusions.
- Relative contribution of each fusion in NVFP4 (as opposed to MXFP8/BF16) training is not quantified in any source found.

## 4. Reported speedups of FP8/FP4 MoE training vs BF16, and accuracy considerations

### Takeaway
MXFP8 MoE training is the mature path: ~1.4–1.8x on the grouped GEMM op, ~1.15–1.43x per MoE layer, ~1.2–1.3x end to end on B200/GB200 with BF16-equivalent loss. NVFP4 grouped GEMMs give smaller realized gains (CUTLASS's own NVFP4 grouped vs torch BF16 avg ~1.11x on B200; miniTE ~1.33–1.37x end to end only when backward is left in BF16). Accuracy is acceptable for MXFP8; NVFP4 MoE shows a measurable gap in RL fine-tuning.

### Cited Findings
- torchao MXFP8: `_to_mxfp8_then_scaled_grouped_mm` ~1.8x faster than compiled BF16 for Llama4 Scout shapes; 1.43x faster MoE layer; 1.2x e2e with routed experts only, 1.3x with shared experts too; +30.2% tokens/s on 256 GB200 (5317 -> 6921), ~81% of the 1.37x roofline; loss curves "virtually identical" to BF16 over 3k steps; output projection kept in BF16 (too sensitive), Wk/Wv excluded (too small) — [PyTorch blog](https://pytorch.org/blog/mxfp8-training-for-moes-1-3x-training-speedup-vs-bf16-for-llama4-scout-on-gb200-cluster-using-torchao-and-torchtitan/)
- Microbenchmarks (B200, fwd+bwd): (M=128000, N=8192, K=5120) G=1: 1.808x; G=2: 1.690x; G=4: 1.637x; G=8: 1.700x — [PyTorch blog](https://pytorch.org/blog/mxfp8-training-for-moes-1-3x-training-speedup-vs-bf16-for-llama4-scout-on-gb200-cluster-using-torchao-and-torchtitan/)
- torchao README: kernel-level 1.4–1.8x (Llama4 Scout shapes), 1.19–1.6x (DeepSeekV3 671B shapes); single MoE layer fwd+bwd 1.431x (Llama4) and 1.148x (DeepSeekV3) — [torchao moe_training README](https://github.com/pytorch/ao/tree/main/torchao/prototype/moe_training)
- DeepSeek-V3 16B with MXFP8 + DeepEP: "up to 41%" (title); expert-parallel tutorial reports 10–25% tokens/s speedup — [PyTorch blog](https://pytorch.org/blog/enabling-up-to-41-faster-pre-training-mxfp8-and-deepep-for-deepseek-v3-on-b200-with-torchtitan/); [torchao EP tutorial](https://docs.pytorch.org/ao/stable/eager_tutorials/mxfp8_expert_parallel_training.html)
- TE Mixtral-8x7B on 8xB200: HF BF16 4,096 tok/s/GPU; TE BF16 4,447; TE EP + MXFP8 9,050 (2.21x vs HF baseline; mixes parallelism and precision changes) — [NVIDIA blog](https://developer.nvidia.com/blog/efficient-moe-training-for-biological-foundation-models/)
- CUTLASS CuTe DSL MoE grouped GEMM vs torch 2.10 cu13 on B200: nvfp4 2Dx3D avg 1.11x, nvfp4 2Dx2D avg 1.12x (worst 0.98x), mxfp8 1.29x/1.41x (comparison is against torch's own kernels of the same dtype per the changelog wording; dtype of the baseline not fully explicit) — [CUTLASS CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)
- miniTE (Qwen3-30B-A3B, 8xB300, RL): full learner update ~1.33x (per-token) / ~1.37x (CUTLASS) vs BF16 with NVFP4 forward + BF16 backward; upstream TE NVFP4 ~0.96x; module F+B slower than BF16 when backward is MXFP8, ~1.2–1.3x faster when backward is BF16; peak reserved memory ~133 GB vs ~120 GB BF16 — [miniTE blog](https://lmsys.org/blog/2026-10-25-minite)
- miniTE accuracy: forward GroupedLinear vs BF16 cosine ~0.991 (attributed to FP4 information loss, not GEMM error); MXFP8 backward gradient cosine ~0.9993, rel-L2 ~0.038, sign agreement ~98.8%; long run (step 999, 5-task mean): BF16 58.84%, miniTE 56.35%, TE NVFP4 54.57% — [miniTE blog](https://lmsys.org/blog/2026-10-25-minite)
- Megatron Core reports 1,233/1,048 TFLOPS/GPU (GB300/GB200) for DeepSeek-V3-685B; precision not stated in the abstract — [Megatron Core MoE report](https://arxiv.org/html/2603.07685)
- SonicMoE (BF16) on B300: +54% fwd / +35% bwd TFLOPS vs a DeepGEMM-based pipeline averaged over 6 configs — [SonicMoE Blackwell blog](https://tridao.me/blog/2026/sonicmoe-blackwell/); ICLR version: 28.7% fwd / 22.1% bwd vs DeepGEMM on OLMoE-sized 7B — [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10010074)
- A 2026 comparative study of NVFP4 pretraining recipes (RHT, selective high-precision layers, WGRAD definition) exists — [What Matters for NVFP4 Training?, arXiv 2610.24343](https://arxiv.org/pdf/2610.24343)

### Inferences
- A well-engineered BF16 grouped kernel (SonicMoE-class) already beats several low-precision pipelines; the MoE speedup from NVFP4 is gated by quantization overhead and wgrad efficiency more than by GEMM peak. For our sm_120 library, an NVFP4 forward + dgrad with FP8 or BF16 wgrad may be the practical sweet spot (miniTE's best result kept backward in BF16).
- Expect smaller gains on a single consumer GPU than the B200 numbers: per-expert M is small, so the "large total_M" condition torchao recommends is hard to meet.
- RTX 4090 (sm_89) has FP8 tensor cores but no FP4; the only applicable low-precision path there is FP8 (torchao lists FP8 rowwise at SM89+).

### Gaps
- No published NVFP4 MoE *pretraining* convergence curve vs BF16 (only RL fine-tuning in miniTE; torchao #3293 claims completion but its validation results were not retrieved).
- No published consumer-GPU (RTX 5090 / 4090) MoE training throughput numbers in any precision.

## 5. PyTorch APIs: torch._grouped_mm / torch._scaled_grouped_mm in 2026 and sm_120 support

### Takeaway
`torch.nn.functional.grouped_mm` and `torch.nn.functional.scaled_grouped_mm` are documented public functional APIs in PyTorch 2.11–2.15 (the underscore ops still exist as what torchao wraps). BF16 grouped_mm is documented for SM80+ (so it should run on sm_120 and sm_89). The scaled version's docs do not state an architecture matrix; I found no evidence of sm_120 support for its MX/NVFP4 recipes, and torchao gates MXFP8 MoE to SM100+.

### Cited Findings
- `grouped_mm` docs (2.11–2.15, main): inputs expected to be bf16 on CUDA devices with SM >= 80; main-branch docs add that FP8 inputs need row-major `mat_a` and column-major `mat_b`, and mat_a/mat_b must be 2D or 3D — [PyTorch docs: grouped_mm 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.grouped_mm.html); [main](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.grouped_mm.html)
- `scaled_grouped_mm(mat_a, mat_b, scale_a, scale_recipe_a, scale_b, scale_recipe_b, swizzle_a=None, swizzle_b=None, bias=None, offs=None, output_dtype=torch.bfloat16, contraction_dim=(), use_fast_accum=False)`; `offs` = "optional offsets into the source tensors denoting group start indices"; scale recipes and swizzles are enums; `use_fast_accum` "Hopper-GPUs only"; no SM100/SM120 statement — [PyTorch docs: scaled_grouped_mm 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_grouped_mm.html)
- torchao MXFP8 MoE: SM100+ only; FP8 rowwise SM89+ — [torchao moe_training README](https://github.com/pytorch/ao/tree/main/torchao/prototype/moe_training)
- NeMo AutoModel falls back to `torch._grouped_mm` when compute capability < 10 — [NeMo AutoModel mxfp8 docs](https://docs.nvidia.com/nemo/automodel/nemo-automodel/nemo_automodel/components/moe/mxfp8)
- A third-party Triton package `grouped-mm-bf16` reimplements bf16 grouped_mm semantics for CUDA and HIP — [PyPI grouped-mm-bf16](https://pypi.org/project/grouped-mm-bf16/0.1.2/)

### Inferences
- For our library, `torch.nn.functional.grouped_mm` (BF16, SM80+) is the natural correctness reference and fallback on sm_120/sm_89, and its `(mat_a 2D, mat_b 3D, offs)` signature is the de facto API shape to mirror (torchao, CUTLASS CuTe DSL "torch-grouped_mm-style", NeMo all mirror it).
- Mirroring `scaled_grouped_mm`'s argument structure (separate scale recipe + swizzle enums, `contraction_dim` to express 2Dx2D wgrad) would make a future swap to upstream kernels easy if PyTorch adds sm_120 NVFP4 support.

### Gaps
- Could not find PyTorch source/release notes stating which `scale_recipe` values `scaled_grouped_mm` supports on which architectures, or any explicit sm_120 statement. Needs a check of `aten/src/ATen/native/cuda/` (grouped blockwise/scaled grouped mm dispatch) or empirical testing on the local RTX 50 card.
- Whether BF16 `grouped_mm` on sm_120 uses a CUTLASS kernel or a slow fallback path is not documented.

## 6. What a minimal but efficient sm_120 NVFP4 MoE implementation looks like (synthesis)

### Takeaway
A minimal design reuses the existing dense TMA + mma.sync block-scaled kernel as the inner tile loop and adds (1) a device-built tile list over (expert, m_tile) from an offsets tensor, (2) a fused permute+quantize kernel, (3) a K-grouped wgrad kernel (or a BF16/FP8 wgrad fallback), and (4) a gather-sum combine kernel — matching the structure all surveyed systems converged on.

### Cited Findings
- sm_120 constraints: ~99 KB SMEM vs 228 KB on SM100 limits tile shapes; K=64 tile shapes proposed — [NVIDIA forum](https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069/131); CUTLASS added 128x32xK / 128x64xK SM120 blockscaled tiles (up to 30% gain on SM121) in 4.5.0 and tileN = 8/16 in 4.7.0 — [CUTLASS CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)
- Standard structure: contiguous M-grouped fwd/dgrad, K-grouped wgrad — [DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM); [SonicMoE Blackwell blog](https://tridao.me/blog/2026/sonicmoe-blackwell/); [PyTorch blog](https://pytorch.org/blog/mxfp8-training-for-moes-1-3x-training-speedup-vs-bf16-for-llama4-scout-on-gb200-cluster-using-torchao-and-torchtitan/)
- NVFP4 scaling per GEMM: 1x16 @ 16x16 for out/dgrad, 1x16 @ 16x1 for wgrad, RHT on wgrad inputs, SR on gradients — [torchao issue #3293](https://github.com/pytorch/ao/issues/3293)
- Per-expert NVFP4 global scales — [CUTLASS CHANGELOG 4.9.0](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)
- Building for `compute_120f` with CUDA 13 enabled TMA grouped tactics that `compute_120a` did not (39 vs 14.6 tok/s) — [FlashInfer issue #2723](https://github.com/flashinfer-ai/flashinfer/issues/2723)

### Inferences (design sketch, not sourced as a whole)
- Routing: router GEMM in BF16/FP32 (Megatron allows FP32 router); device-side top-k, histogram per expert, exclusive scan to get padded offsets (round each count up to 128), and a sorted token-index array. No host sync; capture in a CUDA graph.
- Forward FC1: fused permute+quantize kernel reads X by sorted index, emits FP4 + E4M3 block scales (1x16 along hidden) in swizzled per-128-row layout, accumulates per-expert amax for the FP32 global scale (or uses a delayed/previous-step amax to stay single-pass). Grouped GEMM against [E, 2*ffn, hidden] FP4 weights (16x16-block quantized once per optimizer step). Epilogue: SwiGLU, multiply by routing weight (if applied pre-FC2), quantize to NVFP4 for FC2 and also save what backward needs.
- Forward FC2: same grouped kernel; output BF16 in expert order; separate combine kernel gathers top-k rows per token and sums with routing weights.
- Backward dgrad (FC2 and FC1): same 2Dx3D grouped kernel using the weight's transposed FP4 copy (free if weights use 16x16 2D scaling); dY quantized with SR, fused with the unpermute-gather of dY (the backward of combine is a permute).
- Backward wgrad: K-grouped kernel per expert dW_e = dY_e^T X_e with both operands quantized 16x1 along tokens (after 16x16 RHT), padding guarantees 16-aligned (and 128-aligned) K-groups; empty experts produce zero dW (GEMM_K=0). Pragmatic first version: wgrad in BF16 via `torch.nn.functional.grouped_mm`, or FP8, since miniTE found BF16 backward gave the best end-to-end speed and wgrad is the least efficient low-precision GEMM in torchao.
- Scheduling: one persistent kernel per GEMM type, CTAs pull tiles from a device-built (expert, m_tile, n_tile) list or use CLC (supported by CUTLASS SM120 pingpong kernels); skip fully padded tiles. Order tiles expert-major so each expert's weight tile stays L2-resident across its m_tiles.
- Validation baseline: `torch.nn.functional.grouped_mm` (BF16, SM80+) for correctness; for performance compare against a BF16 grouped kernel, not a Python loop, since SonicMoE-class BF16 kernels are a strong baseline.

### Gaps
- No public sm_120 NVFP4 MoE training kernel exists to benchmark against; all efficiency expectations for consumer GPUs above are extrapolations from B200/B300 data.
- Optimal padding granularity (128 vs 64 vs 16) on sm_120 given its smaller tiles is untested in any source found.
