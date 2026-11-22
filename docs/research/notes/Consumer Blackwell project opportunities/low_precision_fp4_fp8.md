# Low-precision (FP4/FP6/FP8 block-scaled) compute on consumer Blackwell (sm_120, RTX 50-series), status as of Nov 2026

Research date: 2026-11-21. Notes on verification: items marked **[fetched]** were read from the page itself; items marked **[snippet]** come only from search-result summaries and were not opened, so their status may be out of date or slightly wrong. Information dated before ~mid-2026 should be treated as possibly outdated, because this area changes month to month.

## Q1. What PTX/SASS does sm_120 expose for block-scaled low precision, and how does it differ from sm_100 (tcgen05)?

### Takeaway
sm_120 does expose block-scaled FP4/FP6/FP8 tensor-core MMA. It is a synchronous, warp-level `mma.sync ... .block_scale` instruction with operands and accumulators in registers. It has none of sm_100's `tcgen05.mma`, tensor memory (TMEM), or 2-SM/cluster-multicast machinery, and it has about 99 KB of shared memory per SM instead of 228 KB. So every datacenter-Blackwell (SM100) kernel has to be rewritten, not recompiled. The feature needs the arch-specific `sm_120a` target or the family `sm_120f` target (PTX ISA 8.8+).

### Cited Findings
- PTX ISA 8.7 (CUDA 12.8) added the sm_120 target and extended `mma` / `mma.sp::ordered_metadata` with types `.e3m2/.e2m3/.e2m1` and the qualifiers `.kind`, `.block_scale`, `.scale_vec_size` **[snippet]** — [PTX ISA 8.7 (CUDA 12.8.2 archive)](https://docs.nvidia.com/cuda/archive/12.8.2/parallel-thread-execution/index.html)
- PTX ISA 8.8 (CUDA 12.9) added the `sm_120f` family target. The snippet lists the `.scale_vec::1X/2X/4X` modifiers as sm_120f features **[snippet]** — [PTX ISA 8.8 (CUDA 12.9.2 archive)](https://docs.nvidia.com/cuda/archive/12.9.2/parallel-thread-execution/)
- PTX ISA 9.1 adds `.scale_vec::4X` with `.ue8m0` as the scale type for `.kind::mxf4nvf4` on `mma`/`mma.sp`. Before that, 4X was tied to ue4m3 (NVFP4). This allows an "MXFP4 with 16-element blocks" variant **[snippet]** — [PTX ISA 9.1 (CUDA 13.1.2 archive)](https://docs.nvidia.com/cuda/archive/13.1.2/parallel-thread-execution/index.html)
- The exact NVFP4 instruction on SM12x is `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`. m16n8k64 is the only shape for NVFP4 block scaling. `scale_vec::4X` gives 4 scale factors per K=64 (one per 16-element micro-block), and `.ue4m3` is the required scale type. Scale operands are `scale-a-data`/`scale-b-data` with `{byte-id, thread-id}` selectors. CUTLASS hard-codes threads 0–1 of each quad to supply SFA and thread 0 to supply SFB **[fetched]** — [Colfax Research, "CUTLASS Tutorial: NVFP4 block-scaled GEMM on RTX PRO Blackwell (SM12x)", 2026-06-20](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/)
- SM10x vs SM12x, per Colfax **[fetched]**:
  - SM10x issues asynchronous `tcgen05.mma` from a single thread, with operands in SMEM/TMEM and accumulators in TMEM.
  - SM12x uses synchronous, warp-collective `mma.sync` with operands and accumulators in registers, and has no tcgen05 or TMEM.
  - SM10x GEMM kernels will not run on SM12x. SM12x kernels may run on SM10x but perform poorly.
  - Warpgroup register re-allocation is available on sm120a/sm120f.

  — [Colfax Research, 2026-06-20](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/)
- A community patch set lists the hardware gaps against SM100 **[fetched]**:
  - SM120 lacks TMEM, UMMA/tcgen05, and cluster multicast.
  - Shared memory is 99 KB per SM, against 228 KB on SM100.
  - `sm_100f` cubins fail to load on SM120, and `sm_120a` rejects `tcgen05.*`.
  - DeepGEMM, upstream CUTLASS SM100 collectives, and WGMMA-based FlashAttention are incompatible.

  — [lna-lab/blackwell-geforce-nvfp4-gemm](https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm)
- ptxas warns that `.multicast::cluster` gives "substantially reduced performance on sm_120a". In one kernel, cluster multicast was functionally correct but far slower and was dropped. The native E2M1 decode instruction `cvt.rn.f16x2.e2m1x2` is available on sm_100a+/sm_120a **[fetched]** — [EfficientMoE/MoE-Infinity PR #327, merged 2026-11-05](https://github.com/EfficientMoE/MoE-Infinity/pull/327)
- LLVM NVPTX gained NVVM intrinsics for dense and sparse block-scaled MMA (PR #163561, Nov 2025), with support rules keyed on kind, scale_vec size, and stype. This is relevant to Triton and other MLIR/LLVM compilers **[snippet]** — [llvm-commits: NVPTX block-scaling MMA intrinsics (#163561)](https://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20251117/1782260.html)
- A developer forum thread (2026) reports that the register fragment layout for FP4 E2M1 block-scaled `m16n8k64` on SM120 (including where the scales live) is not clearly documented in PTX ISA 9.2 **[snippet]** — [NVIDIA Developer Forums: fragment layout for mma.sync m16n8k64 FP4 block scaling on SM120](https://forums.developer.nvidia.com/t/fragment-layout-for-mma-sync-aligned-m16n8k64-with-fp4-e2m1-and-block-scaling-on-sm120/364020)
- NVIDIA's RTX Blackwell whitepaper says the RTX Blackwell tensor cores "add new support for FP4 and FP6 Tensor Core operations, and the new Second-Generation FP8 Transformer Engine, similar to our datacenter-class Blackwell GPUs" **[fetched, PDF text]** — [NVIDIA RTX Blackwell GPU Architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)

### Inferences
- The programming model on sm_120 is "Ampere-style `mma.sync` plus block scales". Of the Hopper/datacenter-Blackwell features, it keeps TMA but has no WGMMA, tcgen05, or TMEM. Kernels must keep accumulators in registers, so tile sizes are limited by registers and the 99 KB of SMEM, not by TMEM.
- Because the block-scaled MMA needs `sm_120a` or `sm_120f`, binaries built for plain `sm_120` or PTX-JIT'd from `compute_120` cannot use it. Several real bugs below come from build-flag mismatches (`120a` vs `120f`, or a stale nvcc).
- FP6 (e3m2/e2m3) shares the `kind::mxf8f6f4` path with FP8. I found no framework that ships FP6 GEMMs for sm_120. In practice FP6 is unused.

### Gaps
- I did not retrieve the PTX ISA tables directly. The exact `kind::mxf8f6f4` shape on sm_120 (I recall m16n8k32 with `scale_vec::1X` and `ue8m0`, but did not verify it here) and the full list of legal (kind, scale_vec, stype) combinations per PTX version should be checked in the current PTX ISA (9.x) `mma` block-scale section.
- I found no SASS-level microbenchmark of the sm_120 block-scaled MMA (latency, issue rate, register-file pressure) published in 2026. The arXiv paper 2507.10789 (RTX 5080 vs H100, July 2025) covers 5th-gen tensor cores, but only its abstract was read.

## Q2. CUTLASS / CuTe support for sm_120: GEMM examples, grouped GEMM / MoE, attention, CuTe DSL

### Takeaway
CUTLASS has had sm_120 ("Blackwell GeForce") block-scaled GEMM since 3.9.0 (examples 79a–d: NVFP4, NVFP4 output with scale-factor generation, mixed MXFP8/MXFP6, and NVFP4 grouped GEMM). It has expanded through 4.x with mixed-input grouped GEMM, blockwise-scaled GEMM (example 87), and more tile sizes. CuTe also has SM120 MMA traits and a `blackwell_geforce` block-scaled GEMM example used by Colfax. There is no first-party sm_120 FP4 attention kernel, and grouped-GEMM (MoE) correctness on sm_120 has had real bugs.

### Cited Findings
- CUTLASS 3.9.0 introduced SM120 GeForce kernels in the 3.x API, with block-scaled dense and sparse GEMM and grouped GEMM with NVFP4 **[snippet]** — [CUTLASS 3.9.0 release discussion #2262](https://github.com/NVIDIA/cutlass/discussions/2262); [Release v3.9.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.9.0)
- The changelog maps the examples **[snippet]**:
  - 79a: block-scaled GEMM, NVFP4 input, BF16 output.
  - 79b: NVFP4 to NVFP4, with scale-factor generation.
  - 79c: mixed MXFP8/MXFP6 input, BF16 output.
  - 79d: grouped GEMM with NVFP4.

  — [CUTLASS CHANGELOG](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html); [raw CHANGELOG.md](https://raw.githubusercontent.com/NVIDIA/cutlass/main/CHANGELOG.md)
- CUTLASS 4.2 added "Blackwell SM120 mixed input blockscaled grouped GEMM" and an SM120 blockwise GEMM (example 87). Later entries add tileN = 8 and 16 for SM120 block-scale GEMMs, and 128x32xK / 128x64xK tiles for the SM120 block-scaled collective builders, with gains noted on SM121 **[snippet]** — [nvidia-cutlass on PyPI](https://pypi.org/project/nvidia-cutlass/); [CUTLASS CHANGELOG](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html)
- CUTLASS 4.9's Operator API added block-scaled grouped GEMMs for MXFP8, MXFP4, and NVFP4, with per-expert global scales for NVFP4. The snippet does not say whether this covers SM120 **[snippet]** — [CUTLASS Overview](https://docs.nvidia.com/cutlass/latest/overview.html)
- The CUTLASS 4.6.0 docs (updated July 2026) have a "Blackwell SM120 GEMMs" page covering cluster size, tensor layout, pingpong vs cooperative schedules, epilogue schedule, and tile size **[snippet]** — [CUTLASS 4.6.0 Blackwell docs](https://docs.nvidia.com/cutlass/4.6.0/media/docs/cpp/blackwell.html)
- Colfax's tutorial uses CuTe's `mma_traits_sm120.hpp`, the `blackwell_geforce/kernel/blockscaled_gemm` examples, and `dense_gemm.py`, which indicates CuTe/CuTe DSL-level SM12x examples exist. Limits it notes **[fetched]**:
  - bM and bN are capped at 128 by the CUTLASS scale-factor helpers.
  - The MMA atom is fixed at 16x8x64.

  — [Colfax Research, 2026-06-20](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/)
- CUTLASS issue #3096 reports that NVFP4 MoE through CUTLASS grouped GEMM produces garbage output on SM120. It was fixed via FlashInfer SM120 patches plus `compute_120f` on CUDA 13.0, reaching "39 tok/s native FP4" **[snippet]** — [NVIDIA/cutlass#3096](https://github.com/NVIDIA/cutlass/issues/3096)
- Community patches that were needed to run NVFP4 MoE on SM120 **[fetched]**:
  - Limit FlashInfer grouped-GEMM MoE tiles to 99 KB SMEM ("M128/M256 tiles crash without it").
  - Use pre-built SM120 TMA FMHA kernels (64×128, 64×64 tiles).
  - Route MLA through the `xqa` backend instead of `trtllm-gen`.
  - Write an SM120 FlashAttention subclass on SM80 MMA with 99 KB tile sizing.

  — [lna-lab/blackwell-geforce-nvfp4-gemm](https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm)
- A FlashInfer feature request (2026) proposes using the block-scaled MMA with unit scales for SM120 FP8 varlen attention, described as "bit-identical, ~1.3x on RTX 5090". This suggests the FP8 attention path on SM120 was not yet using the block-scaled MMA **[snippet]** — [flashinfer-ai/flashinfer#5963](https://github.com/flashinfer-ai/flashinfer/issues/5963)

### Inferences
- GEMM coverage in CUTLASS C++ is reasonably complete for sm_120: dense, grouped, mixed-input, and NVFP4/MXFP8/MXFP6. The gaps are tuning (small 128×128 tiles and SMEM limits), fused attention, and integration correctness, not missing primitives.
- No vendor ships a FlashAttention-3/4-class kernel for sm_120, because FA3 is Hopper-only (WGMMA) and FA4/cuDNN-Blackwell target tcgen05. Low-precision attention on sm_120 therefore relies on third-party work (SageAttention3).

### Gaps
- I could not confirm from a primary source exactly which CuTe DSL (Python) examples officially support sm_120, or whether CuTe DSL block-scaled grouped GEMM works on sm_120. The CUTLASS `examples/python/CuTe_DSL` directory should be checked.
- I did not find a published CUTLASS profiler table of TFLOPS for examples 79a–d on an RTX 5090.

## Q3. Which frameworks actually run NVFP4/MXFP4/MXFP8 natively on sm_120?

### Takeaway
Native FP4 inference on sm_120 now works in several stacks: TensorRT-LLM, vLLM via FlashInfer/CUTLASS (often opt-in), llama.cpp (NVFP4/MXFP4 W4A4 MMQ, merged April 2026), cuBLASLt via `torch._scaled_mm`, and Triton FP8 `dot_scaled`. The defaults often silently fall back to weight-only Marlin (W4A16) or a BF16 decomposition, and many paths carry correctness bugs. Training support (Transformer Engine MXFP8/NVFP4) on sm_120 was explicitly blocked as of 2025 and is only now being patched.

### Cited Findings
**PyTorch / torchao**
- `torch._scaled_mm` with MXFP8 on sm_120 accepted only the TN layout up to cuBLAS 13.5. NN and NT fail with `CUBLAS_STATUS_NOT_SUPPORTED` from `cublasLtMatmulAlgoGetHeuristic` **[fetched]**:
  - Rejected on cuBLAS 13.1.1.3 (the version shipped with torch 2.14.0+cu130 wheels), 13.4.x, and 13.5.1.
  - Fixed in cuBLAS 13.6.0.2 (CUDA 13.3 Update 1) and 13.8.0.4 (CUDA 13.4).
  - Issue opened 2026-09-22 and still open. PR #198188 (version-gated error) is still open.

  — [pytorch/pytorch#198126](https://github.com/pytorch/pytorch/issues/198126)
- Because cuBLAS is TN-only on sm_120, MXFP8 training needs duplicate transposed quantized data. torchao asked for an orientation-flexible MXFP8 GEMM, and a CuTe DSL MXFP8 GEMM stopgap was planned. A follow-up notes cuBLAS 13.6 lifts the restriction, making single-copy 32x32 2D weight blocks viable **[snippet]** — [pytorch/ao#4932](https://github.com/pytorch/ao/issues/4932); [pytorch/ao#4938](https://github.com/pytorch/ao/issues/4938)
- A community benchmarking project's notes (Chinese) say the following **[snippet; secondary]**:
  - torch 2.7.1 lacks the FP4 dtype, and FP8 row-wise scaling errors on sm_120 in that version.
  - torch 2.9.1 and 2.14.1 include NVFP4 GEMM via cuBLASLt.
  - MXFP4 `_scaled_mm` in 2.14.1 is restricted to compute capability 10.x, so it excludes sm_120.

  — [meiyou666/SDC#37](https://github.com/meiyou666/SDC/issues/37)
- `torch._scaled_grouped_mm` has no NVFP4 support, so grouped NVFP4 (MoE) workloads are forced into N separate `_scaled_mm` launches **[snippet]** — [pytorch/pytorch#187095](https://github.com/pytorch/pytorch/issues/187095)
- The torchao docs still list NVFP4 dynamic quantization as a prototype **[snippet]** — [torchao 0.17 Quantized Inference docs](https://docs.pytorch.org/ao/stable/workflows/inference.html)

**TensorRT / TensorRT-LLM / ModelOpt**
- The official TensorRT-LLM quantization hardware matrix marks Blackwell sm120 as supported for NVFP4, MXFP4, and FP8 per-tensor, and lists GPT-OSS with MXFP4. FP8 block scaling and NVFP4 KV cache are not marked for sm120. A third-party page claims otherwise, which conflicts with the official table **[snippet]** — [TensorRT-LLM Quantization docs](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html); [TensorRT-LLM Release Notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html)
- The TensorRT-LLM release notes list a known issue: NVFP4 W4A16 MoE models are unsupported on SM120 and abort during kernel preparation **[snippet]** — [TensorRT-LLM Release Notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html)

**vLLM**
- vLLM v0.10.2 (Sept 2025) failed on NVFP4 MoE on RTX 5090 with "no cutlass_scaled_mm kernel" for capability 120. PR #24968 enabled the CUTLASS MoE kernels for SM120 at build time **[snippet]** — [vllm#24921](https://github.com/vllm-project/vllm/issues/24921); [vllm PR #24968](https://github.com/vllm-project/vllm/pull/24968)
- Issue #31085 "Add SM120 support for native NVFP4 MoE kernels" was opened 2025-12-20, is still open, and was labelled stale on 2026-10-29. It has 12 heart reactions and about 20 comments **[fetched]**:
  - Native SM120 NVFP4 MoE works only when opted in with `--moe-backend flashinfer_b12x`. Auto-selection excludes it "until the upstream CUTLASS SM121 MMA op guard is resolved".
  - W4A16 checkpoints still land on Marlin.
  - PR #47577 (auto-select B12X on SM120, opened 2026-07-03) appeared unmerged as of the last comment.
  - Mismatched `flashinfer-python` and `flashinfer-cubin` versions, or a stale nvcc, can silently force Marlin.
  - Measured: Qwen3.6-35B-A3B-NVFP4 on RTX PRO 6000 reached parity at batch size 1 and about +10% output tokens/s at batch size 8 versus Marlin. gpt-oss-120b on RTX 6000 Max-Q reached about 185 tok/s with Marlin vs about 140 with Triton.

  — [vllm#31085](https://github.com/vllm-project/vllm/issues/31085)
- Other open vLLM SM120 reports **[snippet]**:
  - #51525: explicitly selecting the VLLM_CUTLASS NVFP4 MoE backend silently produces empty completions on an RTX 5090 Laptop. Marlin works.
  - #48898: NVFP4 garbage output under torch.compile when per-rank K is not 32-aligned (TP8). Padded block-scale columns read stale memory.
  - #47749: a ModelOpt mixed NVFP4 checkpoint falls back to Marlin W4A16 with a "no native FP4" warning.
  - #35065 and #34452: Nemotron-3-Nano NVFP4 fails to start because no NVFP4 MoE backend supports the configuration.
  - #32826: MiniMax-M2.1 NVFP4 fails with expert parallel on RTX PRO 6000.

  — [vllm#51525](https://github.com/vllm-project/vllm/issues/51525); [vllm#48898](https://github.com/vllm-project/vllm/issues/48898); [vllm#47749](https://github.com/vllm-project/vllm/issues/47749); [vllm#35065](https://github.com/vllm-project/vllm/issues/35065); [vllm#34452](https://github.com/vllm-project/vllm/issues/34452); [vllm#32826](https://github.com/vllm-project/vllm/issues/32826)
- vLLM PR #41028 includes SM 12.x in the OAITritonExperts MXFP4 range (gpt-oss). The snippet says the Triton MXFP4 kernels initially failed on a real RTX 5090 because of insufficient shared memory **[snippet]** — [vllm PR #41028](https://github.com/vllm-project/vllm/pull/41028)
- An NVFP4 KV-cache PR for SM120 (RTX 5090) exists **[snippet]** — [vllm PR #50288](https://github.com/vllm-project/vllm/pull/50288)
- A Hugging Face discussion on nvidia/Qwen3.6-35B-A3B-NVFP4 reports three failure modes on one RTX 5090 **[snippet]**:
  - An illegal instruction, tied to FlashInfer compiling for `compute_120a`. The workaround is setting the FlashInfer arch list to `12.0f`.
  - A CUDA-graph-replay crash.
  - A silent engine hang, mitigated but not fixed.

  — [HF discussion #9](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4/discussions/9)

**SGLang**
- Secondary sources only **[snippet]**:
  - SGLang uses FlashInfer CUTLASS kernels for NVFP4 on SM120.
  - A release (cited as 0.32.0) added "OAI Triton MXFP4 MoE on SM12x" and NVFP4 KV cache on SM8x/SM12x.

  — [eurokommerz.at blog](https://eurokommerz.at/blog/blackwell-rtx-sm120-software-support/). Not verified against SGLang's own release notes.

**llama.cpp**
- Mainline PR #22196 "ggml-cuda: Blackwell native NVFP4 support" (by michaelw9999) was merged 2026-04-28 **[fetched]**:
  - Adds a Blackwell NVFP4 MMQ kernel and one templated `mma_block_scaled_fp4` covering MXFP4 and NVFP4.
  - Activations are quantized to FP4 (W4A4) through `quantize_mmq_fp4_cuda`.
  - On DGX Spark (GB10, sm_121) with Nemotron 3 Super 120B, pp2048 went from 506 to 611 t/s (about 1.2x). tg128 was unchanged (12.0 vs 11.9).
  - Perplexity on Nemotron 3 Super 120B: Q4_K 4.6237, NVFP4 with Q8 activations 4.6283, NVFP4 W4A4 4.6577. NVIDIA reviewers noted that 4-bit activations "undeniably hurt quality".
  - The FP4 activation quantize kernel took about 8% of time. Its optimization was deferred.

  — [ggml-org/llama.cpp#22196](https://github.com/ggml-org/llama.cpp/pull/22196)
- The ik_llama.cpp fork has native W4A4 NVFP4 MMQ / MMQ-ID for sm_120/sm_121 and MXFP4 through the same native FP4 kernels (`--nvfp4-precision w4a4`, PR 2880) **[snippet]** — [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
- A blog reports +43% to +68% prefill on a 27B Qwen NVFP4 GGUF on RTX 5090 versus the previous build **[snippet; unverified blog]** — [boredconsultant.com, 2026-07-18](https://boredconsultant.com/2026/07/18/45-tok-s-on-a-Laptop-Getting-Qwen3-6-27B-s-NVFP4-Quants-Running-on-llama-cpp/). A September 2026 guide warned that earlier merged CUDA FP4 paths were generic integer paths without the tensor-core speedup **[snippet]** — [insiderllm.com](https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/)

**Transformer Engine (training)**
- TE raised "MXFP8 (for all gemm layouts) is not supported on 12.0+ architectures yet" on RTX Pro 6000 / sm_120 and GB10 (sm_121). The forum thread is from about late 2025 **[snippet]** — [NVIDIA Forums: TE and GB10, MXFP8/MXFP4 training not yet supported](https://forums.developer.nvidia.com/t/transformer-engine-and-gb10-mxfp8-and-mxfp4-training-not-yet-supported/351220); [TransformerEngine#2304 (supported datatypes per CC)](https://github.com/NVIDIA/TransformerEngine/issues/2304)
- TE PR #3300 (about Aug 2026) adds a fused NVFP4 RHT (random Hadamard transform) columnwise quantize path for SM120/SM121. I could not confirm it shipped in a release **[snippet]** — [TransformerEngine PR #3300](https://github.com/NVIDIA/TransformerEngine/pull/3300)
- The TE 2.18 docs say Blackwell supports MXFP8 and NVFP4 but do not single out sm_120 **[snippet]** — [Transformer Engine docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/)

**Triton**
- Issue #7550 reports that `tl.dot_scaled` "always compiles to bf16 mma on the 5090 RTX" (mxfp8 or mxfp4) **[snippet]** — [triton#7550](https://github.com/triton-lang/triton/issues/7550)
- PR #7918 adds native MXFP FP8 scaled_dot for SM120 and lists MXFP4×MXFP4 (scale_vec 2X) as future work **[snippet]** — [triton PR #7918](https://github.com/triton-lang/triton/pull/7918); [author's blog, 2025-08-29](https://ita9naiwa.github.io/mlsys/2025/08/29/triton-first-pr.html)
- PR #11776 fixes small-K FP4 scaled-dot lowering on SM120: the native FP4 path needs K ≥ 64, and smaller K falls back to decomposition. This implies a native FP4 path exists by then **[snippet]** — [triton PR #11776](https://github.com/triton-lang/triton/pull/11776)
- PR #12383 (about Oct 2026) moves sm_12x to the sm_80/sm_90 TTGIR pipeline **[snippet]**:
  - A 128x128x256 MXFP4 tile needs 131,080 B of SMEM, more than the 99 KB available.
  - The warp-specialization pass hurts sm_120.

  — [triton PR #12383](https://github.com/triton-lang/triton/pull/12383)
- Issue #9678 says FP4 `dot_scaled` operands need `rhs_k_pack=True`, which is poorly documented **[snippet]** — [triton#9678](https://github.com/triton-lang/triton/issues/9678)

**ExLlamaV3**
- No relevant source found (see Gaps).

### Inferences
- The pattern across stacks is that the primitives exist but the integration is fragile. Backend auto-selection excludes sm_120, build flags (`120a` vs `120f`) cause illegal-instruction crashes, SMEM-sized tiles from SM100 crash, and layout restrictions (TN-only) force copies. Many users end up on weight-only Marlin W4A16, which gives the memory savings but not FP4 tensor-core compute.
- MXFP4 (gpt-oss) on sm_120 mostly runs through Marlin or a Triton BF16-decompose path rather than native `kind::mxf4` MMA. The MoE-Infinity PR (Q4) shows a hand-tuned dequant-to-BF16 path beating Triton `dot_scaled`.

### Gaps
- ExLlamaV3: I found no source on whether its EXL3 kernels use sm_120 FP4/FP8 tensor cores. It most likely does not, since EXL3 is a trellis/codebook weight-only format, but that is unverified.
- I could not confirm whether TE PR #3300 merged or whether a TE release enables MXFP8/NVFP4 recipes on CC 12.0.
- The SGLang sm_120 status is unverified from primary sources.
- I could not confirm the exact current merge status of Triton PRs #7918, #11776, and #12383, or of vLLM PR #47577.

## Q4. Achieved vs theoretical throughput on RTX 5090 (and siblings); known correctness bugs and hangs

### Takeaway
The RTX 5090's official dense peaks are 1676 TFLOPS FP4 (FP32 accumulate), 419 TFLOPS FP8 with FP32 accumulate (838 with FP16 accumulate), and 209.5 TFLOPS BF16 with FP32 accumulate. The FP8 FP32-accumulate rate is half the FP16-accumulate rate, as on Ada. I found no rigorous, published RTX 5090 NVFP4 GEMM TFLOPS measurement. The best hard data comes from siblings:
- An unoptimized CUTLASS NVFP4 GEMM reaches about 60% of FP4 peak on the RTX PRO 6000.
- cuBLASLt MXFP8 reaches about 372 TFLOPS on the RTX 5080, against a 450 TFLOPS FP16-accumulate peak.

There are multiple confirmed wrong-result bugs in NVFP4 GEMM and MoE paths on sm_120.

### Cited Findings
- Official RTX 5090 (GB202, 170 SMs, 2407 MHz boost, 575 W) peaks from the whitepaper Appendix A, dense / sparse **[fetched, PDF text extracted locally]**:

  | Precision | Dense / sparse |
  |---|---|
  | FP4 tensor, FP32 accumulate | 1676 / 3352 TFLOPS |
  | FP8, FP16 accumulate | 838 / 1676 TFLOPS |
  | FP8, FP32 accumulate | 419 / 838 TFLOPS |
  | FP16, FP16 accumulate | 419 / 838 TFLOPS |
  | FP16/BF16, FP32 accumulate | 209.5 / 419 TFLOPS |
  | INT8 | 838 / 1676 TOPS |

  The same pattern holds for RTX 5080 (FP4 900.4 dense; FP8 450.2 with FP16 accumulate / 225.1 with FP32 accumulate) and RTX 5070 Ti (FP4 703 dense). Marketing "3352 AI TOPS" for the 5090 is the sparse FP4 figure — [NVIDIA RTX Blackwell GPU Architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- RTX PRO 6000 Blackwell Server Edition (SM120), CUDA 13.2: a tutorial (unoptimized) CUTLASS NVFP4 GEMM is about 3x cuBLAS FP16 GEMM for most shapes and tops out at about 60% of the 2000 TFLOPS dense FP4 peak. No absolute TFLOPS were given, so about 1.2 PFLOPS is implied. Quantization time is excluded. Wave quantization: a 2k×2k problem with 128×128 tiles gives 256 CTAs on 188 SMs, about 1.36 waves **[fetched]** — [Colfax Research, 2026-06-20](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/)
- cuBLASLt MXFP8 TN via `torch._scaled_mm`, 4096x3072x3072 **[fetched; TFLOPS derived from the reported µs as 77.3 GFLOP / time]**:
  - RTX 5080: 208 µs, about 372 TFLOPS, vs BF16 726 µs, about 106 TFLOPS (3.5x).
  - RTX 5060 Ti: 503 µs, about 154 TFLOPS, vs BF16 1610 µs, about 48 TFLOPS (3.2x).
  - The 5080's 372 TFLOPS exceeds the 225 TFLOPS FP8 FP32-accumulate peak and is about 83% of the 450 TFLOPS FP16-accumulate peak. This suggests the block-scaled MXFP8 path is not subject to the GeForce half-rate FP32-accumulate penalty, or that cuBLAS uses reduced-precision accumulation. That is an inference to verify.

  — [pytorch/pytorch#198126](https://github.com/pytorch/pytorch/issues/198126)
- A stock RTX 5090 at 575 W measured 206.6 and 209.0 TFLOPS. The precision was not stated, but it matches the 209.5 BF16/FP16 FP32-accumulate peak at about 99% **[snippet]** — [tokenwatt-check PR #1](https://github.com/tokenwatt/tokenwatt-check/pull/1)
- RTX PRO 6000 Max-Q (sm_120, 188 SMs, 128 MiB L2), gpt-oss MXFP4 MoE weights with BF16 compute (W4A16 dequant) **[fetched]**:
  - Triton `tl.dot_scaled` reached 167 TF at M=6400 and used 254 registers.
  - An inline-asm bf16x2 decode reached 216 TF.
  - The new prefill kernel reached 214–227 TF, 72–84% of the 269 TF BF16 roof.
  - Bugs found: scale byte 0 decoded to 0 instead of the reference's 2^-126 clamp, and e8m0 255 (NaN) produced inf/NaN.

  — [MoE-Infinity PR #327, merged 2026-11-05](https://github.com/EfficientMoE/MoE-Infinity/pull/327)
- A self-reported custom FP4 GEMM on DGX Spark (sm_121) reached 85–129 TFLOPS, with 143 claimed in the README **[snippet; unverified forum claim]** — [NVIDIA Forums: Custom FP4 CUDA kernel 129 TFLOPS on DGX Spark](https://forums.developer.nvidia.com/t/custom-fp4-cuda-kernel-129-tflops-on-dgx-spark-with-pre-quantized-weight-cache/361600)
- SageAttention3 FP4 attention kernel on RTX 5090: about 1038 TOPS (8K noncausal), against about 212 TOPS for FlashAttention2, about 5x. HunyuanVideo end-to-end time fell from 489 s to 164 s **[snippet]** — [SageAttention3, arXiv 2505.11594 (NeurIPS 2025)](https://arxiv.org/abs/2505.11594); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/118711)
- End-to-end inference with SM120 patches for vLLM, FlashInfer, and CUTLASS, single request **[fetched]**:

  | Model (RTX PRO 6000, NVFP4) | tok/s |
  |---|---|
  | Qwen3.6-35B-A3B MoE | 175 (168 elsewhere on the same page) |
  | Gemma4-26B-A4B MoE | 160 |
  | Qwen3.5-27B dense | 57 |
  | Gemma4-31B dense | 51 |

  No TFLOPS were reported — [lna-lab/blackwell-geforce-nvfp4-gemm](https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm)
- Correctness and stability bugs:
  - TensorRT-LLM #14154: `nvfp4_gemm` gives incorrect numerical results on SM120 (RTX 5060 Ti). The reporter says the CUTLASS, cuBLASLt, and CuTe DSL backends all show the error, and the output is repeated tokens or gibberish. A fix (PR #4821 / TRT-LLM v1.4) is referenced but unconfirmed **[snippet]** — [NVIDIA/TensorRT-LLM#14154](https://github.com/NVIDIA/TensorRT-LLM/issues/14154)
  - CUTLASS grouped-GEMM NVFP4 MoE garbage output on SM120 **[snippet]** — [NVIDIA/cutlass#3096](https://github.com/NVIDIA/cutlass/issues/3096)
  - vLLM empty completions (#51525), stale-memory garbage under torch.compile (#48898), and the Qwen3.6 illegal instruction, CUDA-graph crash, and hang (HF #9) **[snippet]** — see the Q3 links
  - cuBLASLt NN/NT MXFP8 layouts unsupported before cuBLAS 13.6 **[fetched]** — [pytorch#198126](https://github.com/pytorch/pytorch/issues/198126)
  - Triton silently using BF16 MMA for `dot_scaled` on the 5090 **[snippet]** — [triton#7550](https://github.com/triton-lang/triton/issues/7550)
- One community project's native NVFP4 and MXFP8 kernel gates passed on an RTX 5090 with an `sm_120a` build (driver 595.99, CUDA 13.2, Oct–Nov 2026): `nvfp4_native` 51 PASS / wrong=0 and `mxfp8_w8a8` 32 PASS. No TFLOPS were given; the claimed 2–13% end-to-end gain is a projection from DGX Spark, not a measurement **[fetched]** — [midagedev/bloomery#926](https://github.com/midagedev/bloomery/issues/926)

### Inferences
- From the measurements above, achievable dense FP4 on a 5090 with a tuned CUTLASS kernel plausibly falls between about 1.0 and 1.3 PFLOPS. That is 60–80% of 1676, extrapolated from the RTX PRO 6000 60% figure. This is an extrapolation, not a measurement, and the 5090's lower 575 W power limit and clocks may reduce it under sustained load.
- Decode and small-batch LLM inference on a 5090 is memory-bound (1.79 TB/s), so FP4 tensor-core compute helps mainly prefill and batched serving. This matches llama.cpp's unchanged tg and +20–68% pp.
- An MXFP8/FP4 path that avoids the GeForce half-rate FP32-accumulate FP8 penalty would be a real reason to prefer block-scaled formats over per-tensor FP8 on consumer cards. This needs a microbenchmark to confirm.

### Gaps
- No rigorous public RTX 5090 NVFP4/MXFP4 GEMM roofline (TFLOPS vs M/N/K, power, and clocks) was found. Searches for cuBLASLt or CUTLASS 79a numbers on a 5090 returned nothing. This is itself an opportunity.
- I did not find SASS-level confirmation of the accumulate precision used by sm_120 block-scaled MMA.

## Q5. Demand signals: NVFP4 training/fine-tuning on consumer GPUs, FP4 attention, MoE FP4 grouped GEMM; and what a single developer could build

### Takeaway
Demand is clear and mostly unmet:
- Many vLLM, TRT-LLM, and CUTLASS issues cover NVFP4 MoE on sm_120.
- Papers have appeared on NVFP4 LoRA on an RTX 5080 (PACT, Oct 2026) and NVFP4 pretraining with a 5090 benchmark (Quartet II).
- SageAttention3 shows 5x attention gains on the 5090.

Remaining gaps are an upstream, well-tested sm_120 NVFP4/MXFP4 grouped-GEMM MoE kernel with correct auto-dispatch, orientation-flexible MXFP8/NVFP4 GEMMs for training, sm_120 FP8/FP4 attention in mainstream engines, and a credible public benchmark suite.

### Cited Findings
- vLLM #31085 (native SM120 NVFP4 MoE): open since Dec 2025, 12 reactions, about 20 comments, and marked stale in Oct 2026 with no maintainer resolution **[fetched]** — [vllm#31085](https://github.com/vllm-project/vllm/issues/31085)
- The PACT paper (Gao, Entezari Zarch, Jiang, Annavaram; arXiv 2610.25048, 2026-10-18) describes NVFP4 data-flow LoRA fine-tuning on a 16 GB RTX 5080:
  - Native NVFP4 GEMMs for frozen projections.
  - Quantized RMSNorm outputs reused, and gate/up activations stored in NVFP4 for the backward pass.
  - GPTQ adapted to NVFP4 scales.
  - The abstract claims lower memory, faster fine-tuning, and competitive accuracy, without numbers **[fetched]**.

  — [arXiv 2610.25048](https://arxiv.org/abs/2610.25048)
- Quartet II (arXiv 2601.22813) covers NVFP4 pretraining with improved unbiased gradient estimation. It includes a single-RTX 5090 benchmark, limited to about 1.1B parameters by 32 GB, with NVFP4 at about 185% of BF16 training throughput **[snippet]** — [arXiv 2601.22813](https://arxiv.org/pdf/2601.22813)
- SageAttention3 (Tsinghua, NeurIPS 2025): FP4 microscaling attention with 1×16 group scales, about 1038 TOPS on RTX 5090. Its README warns it does not guarantee lossless acceleration for all models **[snippet]** — [arXiv 2505.11594](https://arxiv.org/abs/2505.11594)
- An arXiv practical guide benchmarks NVFP4 inference on consumer Blackwell (RTX 5090) for SME deployment **[snippet]** — [arXiv 2601.09527](https://arxiv.org/html/2601.09527v1)
- A blog on RTX 5090 NVFP4 quantization for video diffusion (T2V/I2V) reports quality differs markedly between modes. The title says this; the details are unverified **[snippet]** — [zenn.dev/toki_mwc](https://zenn.dev/toki_mwc/articles/rtx5090-nvfp4-quantization-reality?locale=en)
- Grouped NVFP4 in PyTorch is missing from `_scaled_grouped_mm` **[snippet]** — [pytorch#187095](https://github.com/pytorch/pytorch/issues/187095). torchao asked for orientation-flexible MXFP8 on sm_120 for training **[snippet]** — [pytorch/ao#4932](https://github.com/pytorch/ao/issues/4932)
- The community is maintaining out-of-tree SM120 patch sets (vLLM, FlashInfer, CUTLASS, Inductor) for 5090/5080/5070 Ti/RTX PRO 6000 **[fetched]** — [lna-lab/blackwell-geforce-nvfp4-gemm](https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm). Other examples are the club-3090 Marlin NVFP4 crash reports **[snippet]** — [noonghunna/club-3090#828](https://github.com/noonghunna/club-3090/issues/828)

### Inferences
Candidate single-developer projects, ranked by gap size and feasibility:
1. **sm_120 NVFP4/MXFP4 MoE grouped GEMM with a 99 KB SMEM tile set, plus a correctness harness.** Upstream it into FlashInfer or vLLM's auto-dispatch. The evidence is the open #31085, the wrong-output bugs #51525 and #3096, and the TRT-LLM W4A16 MoE abort. Build on CUTLASS 79d and the mixed-input grouped GEMM.
2. **Public sm_120 low-precision roofline and bug-regression suite.** Cover cuBLASLt, CUTLASS 79a–d, Triton `dot_scaled`, and `_scaled_mm` across layouts, formats, shapes, power limits, and CUDA versions, with bit-exact checks against an FP64 reference. Nothing rigorous exists publicly for the 5090. Such a suite would also catch silent fallbacks (Triton #7550) and NaN/zero-scale edge cases (MoE-Infinity).
3. **Orientation-flexible MXFP8/NVFP4 GEMM (CuTe DSL) for training on CUDA versions before 13.6**, plus TE recipe enablement for CC 12.0. This would enable 5090 LoRA, QLoRA, and small pretraining runs in the style of PACT and Quartet II.
4. **FP8/FP4 block-scaled attention for sm_120 inside FlashInfer/vLLM.** Productize SageAttention3-style kernels, or the FlashInfer #5963 unit-scale MXFP8 idea, for serving engines rather than only diffusion.
5. **Fast fused activation quantizer (BF16 to NVFP4 with scales) for llama.cpp's W4A4 path.** PR #22196 measured about 8% of time spent there and deferred the work.

Risk: NVIDIA, FlashInfer, and vLLM maintainers are actively landing sm_120 work, including CUTLASS tile additions and B12X. Any project should target an upstream contribution rather than a standalone fork.

### Gaps
- I found no GitHub reaction counts for the most-wanted issues beyond vLLM #31085. Counts for TRT-LLM #14154, CUTLASS #3096, and pytorch #198126 were not available.
- I found no 2026 paper benchmarking SageAttention3 against an fp8/fp4 FlashInfer attention on the 5090 in a serving engine.
- PACT's quantitative speed and memory numbers on the RTX 5080 were not retrieved, since only the abstract was read.

## Q6. Academic papers and blogs (2025–2026) benchmarking FP4 on RTX 50-series

### Takeaway
A small set of credible sources exists: SageAttention3 (attention, 5090), Quartet II (training, 5090), PACT (LoRA, 5080), Colfax (GEMM, RTX PRO 6000), and the arXiv Blackwell microbenchmark paper (RTX 5080). Most other "RTX 5090 NVFP4" content is vendor or SEO blog material with unverified numbers.

### Cited Findings
- Jarmusch, Graddon, Chandrasekaran, "Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks" (arXiv 2507.10789, July 2025). It compares an RTX 5080 with an H100 PCIe, covers 5th-gen tensor cores including FP4/FP6, and reports generational improvements and regressions. Only the abstract was read, so specific numbers were not retrieved; possibly outdated with respect to drivers and CUDA **[fetched abstract]** — [arXiv 2507.10789](https://arxiv.org/abs/2507.10789)
- SageAttention3 (arXiv 2505.11594, v1 May 2025, v3 Jan 2026; NeurIPS 2025): about 1038 TOPS FP4 attention on the RTX 5090 **[snippet]** — [arXiv 2505.11594](https://arxiv.org/abs/2505.11594)
- Quartet II (arXiv 2601.22813, Jan 2026): NVFP4 pretraining at about 1.85x BF16 throughput on one RTX 5090, about 1.1B parameters **[snippet]** — [arXiv 2601.22813](https://arxiv.org/pdf/2601.22813)
- PACT (arXiv 2610.25048, Oct 2026): NVFP4 LoRA on an RTX 5080 **[fetched abstract]** — [arXiv 2610.25048](https://arxiv.org/abs/2610.25048)
- Colfax Research NVFP4 GEMM tutorial for SM12x (June 2026): about 60% of FP4 peak on an RTX PRO 6000 with an unoptimized kernel; an optimization follow-up was promised **[fetched]** — [Colfax Research](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/)
- An arXiv guide to private LLM inference on consumer Blackwell (2601.09527, Jan 2026) includes NVFP4 inference on the RTX 5090 **[snippet]** — [arXiv 2601.09527](https://arxiv.org/html/2601.09527v1)
- An arXiv paper on low-rank GEMM with FP8 acceleration (2511.18674) appeared in RTX 5090 FP4/FP8 search results, but its relevance was not checked **[snippet]** — [arXiv 2511.18674](https://arxiv.org/pdf/2511.18674)
- Lower-credibility blogs (vendor or SEO, numbers unverified):
  - [computingforgeeks: NVFP4 vs FP8 on RTX 5090 with TRT-LLM](https://computingforgeeks.com/nvfp4-quantization-tensorrt-llm/)
  - [specpicks: llama.cpp NVFP4 on RTX 50](https://specpicks.com/reviews/llama-cpp-nvfp4-rtx-5090-inference-2026)
  - [X article "NVFP4 on Blackwell: what 4-bit actually delivers"](https://x.com/Mayhem4Markets/article/2081909466606305656)
  - [datapnt: Gemma 4 26B A4B on RTX 5090](https://datapnt.com/blog/deploying-gemma-4-26b-a4b-on-rtx-5090)

  One source mislabels the RTX 5090 as SM100.

### Inferences
- Peer-reviewed FP4 GEMM characterization specifically on the RTX 5090 is lacking. The academic work uses the 5090 as a platform for attention or training methods rather than characterizing the block-scaled MMA itself.

### Gaps
- I did not read the full texts of Quartet II, PACT, or 2507.10789, so their kernel choices (cuBLASLt vs CUTLASS vs custom) and exact tables are not captured here.
