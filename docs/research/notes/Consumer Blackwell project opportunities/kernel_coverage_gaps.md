# Kernel and Library Coverage Gaps for Consumer Blackwell (sm_120 / sm_121), as of November 2026

Method note: reaction/comment counts and open/closed states below come from the GitHub REST search API, queried 2026-11-21 (unauthenticated, sorted by reactions; "r" = total reactions, "c" = comments). The search API reports "closed" for both merged and unmerged-closed PRs, so a "closed" PR is not proof of a merge unless stated. Several queries (Triton, SageAttention, causal-conv1d, DeepEP, CUTLASS, PyTorch core) hit the rate limit and are listed under Gaps.

## 1. FlashAttention (FA2 / FA3 / FA4-CuTe DSL) on sm_120

### Takeaway
FA2 (the C++/CUDA mma.sync path) is the only officially working FlashAttention on sm_120. FA3 (wgmma, Hopper-only) and the FA4 sm_100 fast path (tcgen05/TMEM, 2-CTA) cannot run on sm_120 silicon. A separate sm_120 CuTe-DSL path (SM80-style mma.sync plus TMA loads) is being built mostly by community and NVIDIA-adjacent contributors. Pieces have landed upstream, but the consolidating PR #2634 (fwd+bwd, paged KV, fp8 decode) is still open with no maintainer review visible. Head dim above 128 is the sharpest functional hole: Gemma 4 needs D=512 and Qwen3.5 needs D=256.

### Cited Findings
- Hardware/ISA: an RTX 5090 SM120a SASS listing contains only `HMMA.16816.F32.BF16`, a register-to-register op, and zero `UTC*`/`UTMA*` (tcgen05/TMEM) opcodes. ptxas (CUDA 13.1) rejects `tcgen05.alloc`, `.cta_group::2` and `tcgen05.mma` for sm_120a. An sm_100a cubin fails with `cudaErrorNoKernelImageForDevice`. The author concludes FA4 "cannot and will not run" on the 5090. Context: one individual's investigation dated 2026-03-06, tested with nvidia-cutlass-dsl 4.4.1, no maintainer statements. — [solatticus gist](https://gist.github.com/solatticus/aab6ec3a0436748b021cbbdd12e8c739)
- PR #2634 says sm_120 "keeps the Ampere-style warp-level mma.sync programming model" and lacks WGMMA and tcgen05. Its kernels reuse SM80-base code plus a TMA forward path. — [FA PR #2634](https://github.com/Dao-AILab/flash-attention/pull/2634)
- Community FA4-sm120 bundle (Hugging Face `SecondNatureComputing/flash-attn-4-sm120`) packages six upstream PRs: #2336 split-KV, #2348 paged KV, #2349 TMA fwd with warp specialization, #2389 block-sparse, #2439 dropout, #2484 GQA/runtime fix.
  - Correctness: fwd 64/64 and bwd 40/40 configurations pass.
  - Limits it lists: 99 KB smem (vs 163 KB on SM80); "D > 128 is not supported on SM120"; split-KV clamped to 1; GQA uses the non-packed path.
  - Speed on DGX Spark (sm_121a): "within 4 to 10 percent of FA2" at S=512–4096, i.e. slower, not faster (S=4096: 0.976 ms vs 0.886 ms for vLLM FA2).
  - The author says to prefer upstream once the PRs merge.
  - Source: [HF model card](https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120)
- PR #2634, "FA4 consumer Blackwell (sm_120) integration":
  - Opened 2026-06-08 by thad0ctor with commits from blake-snc (Second Nature Computing). 37 commits; still open as of 2026-11-21; 2 reactions, 8 comments, last updated 2026-08-10.
  - Adds fwd+bwd for dense, causal, sliding-window, GQA/MQA, varlen, paged-KV, block-sparse, score_mod/mask_mod and learnable sink, plus an fp8 (e4m3/e5m2) KV decode path.
  - Author-reported FA4/FA2 speed ratios on the 5090: D256 GQA fwd about 1.11×; D128 fwd 1.03–1.07×; bwd about parity; fp8 decode 1.4–2.2× at GQA r2/r4 with batch ≥64, but 0.3–0.5× at GQA r8.
  - No maintainer comments are visible.
  - Source: [FA PR #2634](https://github.com/Dao-AILab/flash-attention/pull/2634)
- Other sm_120 PRs in Dao-AILab/flash-attention:
  - Closed (merge status not verified): #2268 "Add SM120 (Blackwell GeForce / DGX Spark) flash attention" (closed 2026-03-12); #2349 TMA fwd (closed 2026-03-14); #2439 dropout (closed); #2474 Spark fwd/bwd regressions (closed); #2671 compile-arg fix (closed 2026-07-07).
  - Still open: #2404 SM120 fwd/bwd + atomic add (open since 2026-03-28); #2406 optimized TMA fwd (open); #2599 "Optimize sm120 fwd kernel performance" (open since 2026-05-28); #2814 pack_gqa crash fix (open since 2026-08-24); #2862 varlen OOB fix (open since 2026-09-05).
  - Sources: [#2268](https://github.com/Dao-AILab/flash-attention/pull/2268), [#2349](https://github.com/Dao-AILab/flash-attention/pull/2349), [#2404](https://github.com/Dao-AILab/flash-attention/pull/2404), [#2406](https://github.com/Dao-AILab/flash-attention/pull/2406), [#2599](https://github.com/Dao-AILab/flash-attention/pull/2599), [#2814](https://github.com/Dao-AILab/flash-attention/pull/2814), [#2862](https://github.com/Dao-AILab/flash-attention/pull/2862)
- Conflict on merge status: the HF card (published about 48 days before an Oct–Nov 2026 search snapshot) calls the six PRs "open upstream PRs," but the search API shows #2349 and #2439 as closed. Whether they were merged or closed unmerged is not verified. — [HF card](https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120); [#2349](https://github.com/Dao-AILab/flash-attention/pull/2349)
- Demand signals (issues still open):
  - #1987 "Support for NVIDIA Blackwell (sm_120) — any planned release date?": 30r, 11c, opened 2025-11-05. [#1987](https://github.com/Dao-AILab/flash-attention/issues/1987)
  - #1683 "Full RTX compatible version here… RTX 50 Blackwell included": 28r. [#1683](https://github.com/Dao-AILab/flash-attention/issues/1683)
  - #1969 sm121 / GB10: 19r, 21c. [#1969](https://github.com/Dao-AILab/flash-attention/issues/1969)
  - #2427 "Support head_dim=512 for Gemma 4 global attention layers": 12r, 12c, updated 2026-11-12. [#2427](https://github.com/Dao-AILab/flash-attention/issues/2427)
  - #1549 "FA3 support for 50 series (sm_120)": 3r, opened 2025-03-23. [#1549](https://github.com/Dao-AILab/flash-attention/issues/1549)
  - #1763 RTX 5090 kernel crash with Voxtral: 21c, updated 2026-05-16. [#1763](https://github.com/Dao-AILab/flash-attention/issues/1763)
  - Build pain: #1038 "Build flash-attn takes a lot of time" (34r, 77c); #1560 "6 hours building on 5090 Windows". [#1038](https://github.com/Dao-AILab/flash-attention/issues/1038), [#1560](https://github.com/Dao-AILab/flash-attention/issues/1560)
- Maintainer and contributor statements in #1987:
  - No core Dao-AILab maintainer has replied.
  - Contributor johnnynunez wrote that "Flash Attention is Hardware Aware Programming": FA3 targets Hopper and FA4 targets B200/B300. He said better GeForce support is waiting on CUTLASS 4.4–4.5 and PyTorch on CUDA 13.2 (2026-02-05).
  - Commenter weireweire noted that FA2 lacks FP8 QKV support and "performs poorly on Blackwell."
  - Source: [#1987](https://github.com/Dao-AILab/flash-attention/issues/1987)
- vLLM's attention-backend table (latest docs, dated Nov 22, 2026) lists FLASH_ATTN FA3 as 9.x and FA4 as ≥10.0. It defines priority lists only for SM 8.x–9.x and SM 10.x, with no 12.x list. — [vLLM attention backends doc](https://docs.vllm.ai/en/latest/design/attention_backends/)
- NVIDIA CUTLASS PR #3030 adds CuTe-DSL "Flash Attention v2 for SM120 (Blackwell GeForce)": forward only, FP16/BF16, causal. — [cutlass #3030](https://github.com/NVIDIA/cutlass/pull/3030)
- Community wheels:
  - mjun0812/flash-attention-prebuild-wheels: 1,801 stars, pushed 2026-11-22, actively maintained. [repo](https://github.com/mjun0812/flash-attention-prebuild-wheels)
  - kingbri1/flash-attention (Windows wheels): 940 stars, last push 2025-12-09, likely stale. [repo](https://github.com/kingbri1/flash-attention)
  - gau-nernst/learn-cuda (educational Blackwell FA in C++): 534 stars. [repo](https://github.com/gau-nernst/learn-cuda)

### Inferences
- The fp8-KV decode path and D>128 (D=256/512) support on sm_120 are the concrete FA holes. They are being worked on but are not shipped in official wheels. Because PR #2634 has gone months without maintainer review, upstream bandwidth for sm_120 is the bottleneck, not technical feasibility.
- Near-parity numbers (FA4-sm120 ≈ FA2 ± 10%) suggest the headroom left on sm_120 attention is in low precision (fp8/FP4 QK/PV via block-scaled mma.sync), not in the bf16 dense path.

### Gaps
- I could not confirm whether #2268, #2349 and #2439 were merged, or whether any official flash-attn release ships sm_120 CuTe kernels in its wheel.
- No primary Tri Dao statement on sm_120 prioritization was found.
- FA3 has no sm_120 path and no evidence one is planned.

## 2. Per-library status: vLLM, SGLang, FlashInfer, TRT-LLM, DeepGEMM/DeepEP, FlashMLA, Triton, torchao, bitsandbytes, Unsloth, Liger, Mamba, xformers, SageAttention

### Takeaway
Dense bf16 and FP8 inference mostly works on sm_120 by late 2026. The gaps cluster around four areas:
- **MLA and sparse MLA attention.** FlashMLA, CUTLASS_MLA and the FlashInfer MLA backends are sm_90/sm_100-only.
- **DeepSeek-specific kernels.** DeepGEMM upstream refuses sm_120; FlashMLA sparse has no upstream sm_120 path.
- **Native FP4 MoE auto-selection.** It falls back to Marlin by default.
- **Datacenter-only features.** These include trtllm-gen FMHA, NVFP4 KV cache, attention sinks on early builds, and the large-smem Mamba3 backward.

A new sm_120-specific CuTe-DSL library, b12x, became the de facto fix path in vLLM and FlashInfer during Aug–Nov 2026.

### Cited Findings
**vLLM**
- In the latest docs, these backends exclude 12.x: FLASHINFER native (8.x–9.x), XQA (9.0), trtllm-gen (10.x), FLASHMLA/FLASHMLA_SPARSE (9.x–10.x), FLASH_ATTN_MLA (9.x), FLASHINFER_MLA, CUTLASS_MLA and TOKENSPEED_MLA (10.x).
- These include 12.x: FLASHINFER_MLA_SPARSE_SM120, FLASHINFER_MLA_SPARSE_DSV4/DSV41 (10.x and 12.x), and B12X ("causal decoder attention on NVIDIA SM120 and SM121"; head sizes 128/192/256; no sparse, no MLA).
- Source: [vLLM docs](https://docs.vllm.ai/en/latest/design/attention_backends/)
- B12X integration:
  - PR #51696 "[Kernel] Add native B12X linear, MoE, and causal attention backends" by Luke Alonso: 76r, the highest-reacted sm120 item in vLLM. Opened 2026-08-10 and closed 2026-08-12 unmerged; mgoin asked for it to be split.
  - It was superseded by #52016 (linear), #52017 (causal paged attention) and #52018 (FP4 MoE), reported as merged.
  - Gains on an RTX PRO 6000 Max-Q (single-request decode): +4.3% dense FP8, +4.3% NVFP4 MoE, +1.2–1.4% attention vs FlashInfer.
  - The PR defers MLA, sparse MLA and DSV4 kernels to later PRs.
  - Source: [vLLM #51696](https://github.com/vllm-project/vllm/pull/51696)
- Sparse MLA (DeepSeek V3.2/V4 DSA, GLM-5.x):
  - Issue #38006 "Implement TRITON_MLA_SPARSE backend for sm80/120/121": 10r, open.
  - PR #38476 TRITON_MLA_SPARSE: 12r, 220c, open since 2026-03-29, last updated 2026-09-11.
  - Takeover PR #47629: 5r, 25c, open.
  - Sources: [#38006](https://github.com/vllm-project/vllm/issues/38006), [#38476](https://github.com/vllm-project/vllm/pull/38476), [#47629](https://github.com/vllm-project/vllm/pull/47629)
- An RTX PRO 6000 user running GLM-5.1-FP8 hit "No valid attention backend": TRITON_MLA was rejected as "sparse not supported" and FLASHMLA as the wrong compute capability. — [NVIDIA forum](https://forums.developer.nvidia.com/t/running-glm-5-1-fp8-on-rtx-pro-6000/366892)
- DeepSeek V4 on SM12x: PR #41834 ("Add SM12x support for DeepSeek V4 Flash"; 46r, 446c; closed 2026-05-06) states that SM12x lacks TMEM/tcgen05, so the DeepGEMM, FlashMLA and Marlin FP8 paths fail at link time. One search summary said its SM12x path depended on unreleased FlashInfer/DeepGEMM branches. — [vLLM #41834](https://github.com/vllm-project/vllm/pull/41834)
- NVFP4 MoE:
  - Issue #31085 "Add SM120 support for native NVFP4 MoE kernels": 12r, 21c, open; marked stale by a bot on 2026-10-29.
  - Logs show "Your GPU does not have native support for FP4 computation… Using Marlin backend."
  - `--moe-backend flashinfer_b12x` works but is not auto-selected; a code comment says it is excluded "until the upstream CUTLASS SM121 MMA op guard is resolved." Dense FP4 layers still fall back to Marlin.
  - PR #47577 wires B12X ahead of Marlin; its merge status was not shown.
  - No vLLM maintainer has replied in the thread.
  - Source: [vLLM #31085](https://github.com/vllm-project/vllm/issues/31085)
- Other open SM120 items:
  - #49011 "nvfp4 KV cache on SM120 — flashinfer ships the kernels, vLLM isn't wired to them": 15r, 34c, open since 2026-07-18. [#49011](https://github.com/vllm-project/vllm/issues/49011)
  - #47130 DeepGEMM "Unknown recipe" assertion on sm_120 (a regression in v0.24.0): 8r, open, updated 2026-11-13. [#47130](https://github.com/vllm-project/vllm/issues/47130)
  - #40756 MTP speculative-decode illegal memory access (Qwen3.6-27B-FP8): 18r, 46c, open, updated 2026-11-20. [#40756](https://github.com/vllm-project/vllm/issues/40756)
- Historical (likely resolved):
  - #14452 "Steps to run vLLM on your RTX5080 or 5090!" (44r, 139c, closed). [#14452](https://github.com/vllm-project/vllm/issues/14452)
  - #22279 gpt-oss "Sinks are only supported in FlashAttention 3" on the 5090 (21r, 97c, closed 2026-01). [#22279](https://github.com/vllm-project/vllm/issues/22279)

**SGLang**
- Tracking issue #19637 "SM120 Performance Optimization Plan" (20r, 25c):
  - Opened 2026-03-02 by b8zhong and labeled high priority. Closed by a bot on 2026-10-12 for inactivity, even though the close reason reads "completed."
  - Gaps it listed: "DSA was not currently supported"; MLA prefill/decode (fmha_v2, XQA MLA); MoE grouped GEMM ("highest importance"); NVFP4, FP8 blockwise and MXFP8 MoE/dense GEMM; AllReduce over PCIe.
  - Only MXFP8 enablement was marked done.
  - Benchmarks on an RTX PRO 5000 (Qwen3.6-27B-FP8): `flashinfer_cutlass` FP8 GEMM was slower than plain `cutlass` in 75/75 operator cases, with E2E output of 296 vs 375 tok/s.
  - masahi (NVIDIA Triton team) said Triton's `triton_kernels` on SM120 "is actually doing dequantization and fp16 MMA."
  - Source: [SGLang #19637](https://github.com/sgl-project/sglang/issues/19637)
- Still-open SM120 items:
  - #37813 GLM-5.3-Flash on SM120 required fixes (open). [#37813](https://github.com/sgl-project/sglang/issues/37813)
  - #42367 DeepSeek-V4.1-Flash CUDA graph capture fails with FlashInfer 0.7.0 on SM120. [#42367](https://github.com/sgl-project/sglang/issues/42367)
  - PR #51890 FP32-state FlashInfer GDN decode on SM120 (opened 2026-11-20). [#51890](https://github.com/sgl-project/sglang/pull/51890)
  - PR #51911 SM120 Triton MoE fusion (2026-11-20). [#51911](https://github.com/sgl-project/sglang/pull/51911)
- Historical: #8879, ptxas "invalid architecture sm_120a" in Triton for GPT-OSS (11r, closed). — [#8879](https://github.com/sgl-project/sglang/issues/8879)

**FlashInfer**
- Initial SM120/SM121 support: PR #1608 (2025-08-30). — [#1608](https://github.com/flashinfer-ai/flashinfer/pull/1608)
- SM120 FP4 work:
  - #2738 MXFP4/NVFP4 grouped GEMMs on "GeForce and Spark" (9r, closed 2026-03). [#2738](https://github.com/flashinfer-ai/flashinfer/pull/2738)
  - #2786 K=64 block-scaled MoE GEMM for SM120 (7r). [#2786](https://github.com/flashinfer-ai/flashinfer/pull/2786)
  - #2577 "NVFP4 mm_fp4 GEMM broken on SM120 — all backends fail" (5r, 22c, closed 2026-07). [#2577](https://github.com/flashinfer-ai/flashinfer/issues/2577)
- Still open:
  - #3463 native MXFP8 W8A8 grouped MoE for SM120/121 (open since 2026-05-29). [#3463](https://github.com/flashinfer-ai/flashinfer/pull/3463)
  - #3628 RFC "MXFP8 block-scaled prefill attention for SM120a" (0 comments). [#3628](https://github.com/flashinfer-ai/flashinfer/issues/3628)
  - #4071 `flashinfer.experimental.sm12x` (open). [#4071](https://github.com/flashinfer-ai/flashinfer/pull/4071)
  - #4495 "B12x Direct low-token quantized MoE" (open). [#4495](https://github.com/flashinfer-ai/flashinfer/pull/4495)
  - #3346 DeepSeek V4 kernel tracker (6r, open). [#3346](https://github.com/flashinfer-ai/flashinfer/issues/3346)
- Toolchain gotcha: `-arch=sm_120a` does not set the ptxas target to sm_120a, so block-scaled instructions fail. You must use `-gencode arch=compute_120a,code=sm_120a`. — [FlashInfer issue #2723 (via search)](https://github.com/flashinfer-ai/flashinfer/issues/2723)

**TensorRT-LLM**
- #11799 "Compile trtllm-gen FMHA cubins for SM120/SM121": closed 2026-07; resolution not verified. [#11799](https://github.com/NVIDIA/TensorRT-LLM/issues/11799)
- #10241 "Add NVFP4 KV cache support for SM120 in trtllm-gen": 3r, 9c, open. [#10241](https://github.com/NVIDIA/TensorRT-LLM/issues/10241)
- #7484 "Only SM100 is supported by FP4 block scale MOE": open since 2025-09. [#7484](https://github.com/NVIDIA/TensorRT-LLM/issues/7484)
- #6892 gpt-oss on the 5090, "attention sinks is only supported on SM90": closed. [#6892](https://github.com/NVIDIA/TensorRT-LLM/issues/6892)
- PR #16511 "Support nvfp4 w4a16 on sm120": closed 2026-07-16. [#16511](https://github.com/NVIDIA/TensorRT-LLM/pull/16511)
- PR #15928 cuda_scaled_mm FP8 fast path for SM121: open. [#15928](https://github.com/NVIDIA/TensorRT-LLM/pull/15928)
- Demand in TRT-LLM is low (≤4 reactions per issue).

**DeepGEMM / FlashMLA (DeepSeek)**
- DeepGEMM #236 "Support sm_120 (5090 and Blackwell 6000 Pro)": 10r, 29c, open.
  - Maintainer LyricZhao (2025-12-05): "Sorry, as we don't have SM120 devices, we don't have plans or people to support this." He cited GEMM-heuristic maintenance cost and suggested a community fork.
  - Blockers cited in the thread: sm_100 kernels expect about 168 KB smem against the 99 KB limit; TMEM/UMMA ops would have to move to smem; the sm_90 path needs WGMMA; host dispatch only handles arch major 9 and 10.
  - Source: [DeepGEMM #236](https://github.com/deepseek-ai/DeepGEMM/issues/236)
- Community DeepGEMM ports:
  - PR #318 "Support SM120": 12r, 18c, open since 2026-04-25. [#318](https://github.com/deepseek-ai/DeepGEMM/pull/318)
  - PR #324: closed 2026-07-10. [#324](https://github.com/deepseek-ai/DeepGEMM/pull/324)
  - PR #447 (targets the `nv_dev` branch, 2026-09-16, lucifer1004): dense/grouped BF16/FP8/FP4 GEMM, mixed FP8×FP4, HC prenorm, and dense, paged and sparse FP8/MXFP4 MQA logits; 741 SM120 tests. It excludes SM120 MegaMoE, Mega Gate and Mega mHC. [#236 thread](https://github.com/deepseek-ai/DeepGEMM/issues/236)
- Recent DeepGEMM SM120 PRs (#431, #475, #501, #513; Sep–Nov 2026) and open issue #493 "SM120 support for the mega fused MoE path" (opened 2026-11-03) show active, NVIDIA-branch-driven work. — [#493](https://github.com/deepseek-ai/DeepGEMM/issues/493), [#501](https://github.com/deepseek-ai/DeepGEMM/pull/501), [#513](https://github.com/deepseek-ai/DeepGEMM/pull/513)
- DeepGEMM #317: on SM120, DeepSeek-V4 is missing `tf32_hc_prenorm_gemm` and `paged_mqa_logits` (7r, closed 2026-04-30). — [#317](https://github.com/deepseek-ai/DeepGEMM/issues/317)
- FlashMLA:
  - Issues #124 "support for rtx 6000 sm120" (open, updated 2026-10-01) and #119 Windows+sm_120 community build (open). [#124](https://github.com/deepseek-ai/FlashMLA/issues/124), [#119](https://github.com/deepseek-ai/FlashMLA/issues/119)
  - PRs #146 SM120 support and #189 "SM120 (RTX 5090) sparse decode kernel": both closed; merge not verified. [#146](https://github.com/deepseek-ai/FlashMLA/pull/146), [#189](https://github.com/deepseek-ai/FlashMLA/pull/189)
  - Community fork IISuperluminaLII/FlashMLA_Windows_Linux_sm120: 8 stars; dense only, no sparse kernels. [repo](https://github.com/IISuperluminaLII/FlashMLA_Windows_Linux_sm120), [DeepGEMM #236](https://github.com/deepseek-ai/DeepGEMM/issues/236)
- DeepEP: no data (rate-limited; see Gaps).

**b12x (the main gap-filler)**
- local-inference-lab/b12x: 270 stars, last push 2026-10-06.
- It is an "SM120/SM121 CuTe DSL kernel library for (primarily) NVFP4 LLM inference," pure Python and JIT-compiled, needing torch ≥ 2.12.
- Kernels: fused FP4 TP MoE (micro-kernel decode, dynamic persistent-grid, W4A16), FP8/MXFP8/NVFP4 dense, paged MHA/GQA.
- An older README says it does not intend to target SM100.
- Sources: [repo](https://github.com/local-inference-lab/b12x), [PyPI](https://pypi.org/project/b12x/), [vLLM #51696](https://github.com/vllm-project/vllm/pull/51696)
- lna-lab/blackwell-geforce-nvfp4-gemm (vLLM/FlashInfer SM120 NVFP4 patches): 29 stars, last push 2026-04-27. — [repo](https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm)

**Triton**
- `tt.dot_scaled` supports the sm_120 block-scaled MMA, added in 2025. — [ita9naiwa write-up](https://ita9naiwa.github.io/mlsys/2025/08/29/triton-first-pr.html)
- FP4 `dot_scaled` requires `rhs_k_pack=True` on Blackwell, which is undocumented; tested on a 5070 Ti with Triton 3.6.0. — [triton #9678](https://github.com/triton-lang/triton/issues/9678)
- A DGX Spark forum post reports a software-emulation fallback on SM121. — [NVIDIA forum](https://forums.developer.nvidia.com/t/custom-fp4-cuda-kernel-129-tflops-on-dgx-spark-with-pre-quantized-weight-cache/361600)
- `triton_kernels` (the MoE path used by gpt-oss) dequantizes and runs fp16 MMA on SM120, per masahi of NVIDIA. — [SGLang #19637](https://github.com/sgl-project/sglang/issues/19637)
- Triton's Gluon tcgen05 and 2-CTA tutorials are sm_100-only. — [Triton Gluon tcgen05 tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/tcgen05-mma-scaled.html)

**torchao**
- PR #2285 "Build mxfp4 kernel for sm120a" (5r, closed 2025-06). [#2285](https://github.com/pytorch/ao/pull/2285)
- Issue #2376 "int8 w8a8 doesn't work on 5090": open since 2025-06-14, updated 2026-09-01. [#2376](https://github.com/pytorch/ao/issues/2376)
- NVFP4 Training Tracker #3293: closed 2026-10. [#3293](https://github.com/pytorch/ao/issues/3293)
- #4376: Float8DynamicActivationFloat8WeightConfig is inference-only (open). [#4376](https://github.com/pytorch/ao/issues/4376)

**bitsandbytes**
- Blackwell binaries since PR #1491 (2025-01). [#1491](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1491)
- Issue #1851: NF4 "significant energy efficiency penalty on Blackwell (RTX 5090) for small m" (open). [#1851](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1851)
- PR #2098 "Optimize NF4 single-token GEMV on Blackwell" (open since 2026-10-08). [#2098](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/2098)
- PR #1949 new 4-bit GEMM kernels (closed 2026-07). [#1949](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1949)
- Demand is low (≤1 reaction per item).

**Unsloth**
- Blackwell issues are installer and packaging items, not kernels:
  - #6971 "Install Flash Attention on Blackwell when a CUDA ≥12.8 wheel exists" (closed 2026-07-08). [#6971](https://github.com/unslothai/unsloth/pull/6971)
  - #6379 prefer native cu13 on Blackwell (closed). [#6379](https://github.com/unslothai/unsloth/pull/6379)
  - #4849 llama.cpp CUDA 13.2 gibberish for IQ quants (7r, closed). [#4849](https://github.com/unslothai/unsloth/issues/4849)

**Liger-Kernel**
- No sm_120-specific bugs found; Liger is Triton-based. Open #1205 proposes a cuTile backend with B200 data. — [#1205](https://github.com/linkedin/Liger-Kernel/issues/1205)

**Mamba / causal-conv1d**
- mamba #710 "Add Support for NVIDIA RTX 50 Series": 8r, open since 2025-03. [#710](https://github.com/state-spaces/mamba/issues/710)
- PR #735 "full CUDA Support for Blackwell": 5r, open with 0 comments since 2025-05-28. [#735](https://github.com/state-spaces/mamba/pull/735)
- Issue #1044 "Mamba3 MIMO backward cannot launch on a lower-shared-memory Blackwell SKU (220 KB requested vs 99 KB limit)": open, updated 2026-11-17. [#1044](https://github.com/state-spaces/mamba/issues/1044)
- causal-conv1d: no data (rate-limited).

**xformers**
- Open issues:
  - #1251 memory_efficient_attention fails in half precision on the 5090 (8r, 14c). [#1251](https://github.com/facebookresearch/xformers/issues/1251)
  - #1342 50-series compatibility (7r). [#1342](https://github.com/facebookresearch/xformers/issues/1342)
  - #1329 FA-compiled xformers fails on RTX 5000 series. [#1329](https://github.com/facebookresearch/xformers/issues/1329)
  - #1395 "Blackwell Support" (3r, open since 2026-04-29, 0 comments). [#1395](https://github.com/facebookresearch/xformers/issues/1395)
- Historical: #1231 sm90 sparse24 GEMM blocks the 5090 build (closed 2025-03). [#1231](https://github.com/facebookresearch/xformers/issues/1231)

**SageAttention**
- SageAttention3 (MXFP4 attention) was designed for sm_120: the paper reports 1,038 TOPS on the 5090, about 5× FlashAttention. [arXiv 2505.11594](https://arxiv.org/abs/2505.11594)
- thu-ml/SageAttention: 4,019 stars but last push 2026-01-17, i.e. upstream is quiet. [repo](https://github.com/thu-ml/SageAttention)
- Fork woct0rdho/SageAttention (Windows wheels): 1,068 stars, pushed 2026-10-09. [repo](https://github.com/woct0rdho/SageAttention)
- A community HF wheel exists for SA3 on cu130 + torch 2.10 + py3.12. [HF](https://huggingface.co/Seryoger/Sageattention-3-cu130-5090-endpoint)

### Inferences
- The most sm_90/sm_100-locked kernel families on sm_120 in late 2026:
  1. MLA decode/prefill, especially sparse MLA/DSA for DeepSeek V3.2/V4, GLM-5.x and Kimi-type models. Only FlashInfer's SM120 sparse MLA and DSV4 backends cover 12.x; TRITON_MLA_SPARSE has sat as an unmerged PR since March.
  2. DeepGEMM mega-fused MoE and mHC paths.
  3. NVFP4 KV cache. FlashInfer has kernels, but vLLM and TRT-LLM are not wired to them.
  4. FP4 MoE auto-selection, which defaults to Marlin's slow path.
  5. Large-smem training backward kernels such as Mamba3 MIMO.
- Demand is driven by RTX PRO 6000 (96 GB) users running DeepSeek, GLM and Qwen MoE models. Most benchmark numbers come from PRO 6000 and DGX Spark, not 24–32 GB GeForce cards. For a 24 GB 5090 Laptop, the relevant gaps are smaller-model ones: FP8/FP4 attention, D=256/512 heads, NVFP4 KV cache, small-batch FP4 GEMV and speculative-decoding stability.

### Gaps
- Rate-limited, no issue data: Triton (sm120/5090 issues), SageAttention sm120 issues, causal-conv1d, DeepEP, NVIDIA/cutlass sm120, PyTorch core.
- Unconfirmed merge status: vLLM #52016/#52017/#52018 and #47577; FlashMLA #146/#189; TRT-LLM #11799 resolution.
- DeepEP on sm_120 is unresearched. DeepEP mainly targets NVLink/IB multi-GPU, so it matters little to a single laptop GPU.
- No authoritative NVIDIA document was fetched confirming the sm_120 smem limit (99 KB per block / about 100 KB per SM) or the full ISA list (mma.sync with block-scale, TMA, no wgmma/tcgen05). The 99 KB figure appears consistently across the HF card, DeepGEMM #236 and mamba #1044.

## 3. Highest-demand unresolved gaps (late 2026)

### Takeaway
By reaction counts, the loudest still-open sm_120 asks are:
- FA sm_120 official support (#1987, 30r)
- FA on GB10/sm121 (#1969, 19r)
- vLLM nvfp4 KV cache on SM120 (#49011, 15r)
- vLLM native NVFP4 MoE (#31085, 12r)
- vLLM TRITON_MLA_SPARSE (#38476, 12r and 220 comments)
- FA head_dim=512 for Gemma 4 (#2427, 12r)
- DeepGEMM sm_120 (#236 and #318, 10–12r)

Absolute counts are modest (tens, not hundreds). The highest-reaction sm120-tagged items overall (vLLM B12X PR, 76r; SGLang DeepSeek V4, 216r) are model-enablement PRs, so demand concentrates on "run the newest MoE/MLA model fast on consumer/workstation Blackwell."

### Cited Findings
- FA #1987 (30r), #1969 (19r), #2427 (12r): all open. — [#1987](https://github.com/Dao-AILab/flash-attention/issues/1987), [#1969](https://github.com/Dao-AILab/flash-attention/issues/1969), [#2427](https://github.com/Dao-AILab/flash-attention/issues/2427)
- vLLM #49011 (15r), #31085 (12r; stale-flagged 2026-10-29), #38476 (12r, 220c), #38006 (10r), #47130 (8r): all open. — [#49011](https://github.com/vllm-project/vllm/issues/49011), [#31085](https://github.com/vllm-project/vllm/issues/31085), [#38476](https://github.com/vllm-project/vllm/pull/38476)
- vLLM B12X PR #51696: 76r, the most-reacted SM120 kernel PR. — [#51696](https://github.com/vllm-project/vllm/pull/51696)
- SGLang's SM120 plan #19637 (20r) was auto-closed for inactivity with items unchecked. — [#19637](https://github.com/sgl-project/sglang/issues/19637)
- DeepGEMM #236 (10r, 29c) and #318 (12r): open. — [#236](https://github.com/deepseek-ai/DeepGEMM/issues/236), [#318](https://github.com/deepseek-ai/DeepGEMM/pull/318)
- Forum demand: NVIDIA developer-forum threads on GLM-5.1-FP8 on the RTX PRO 6000 and on DeepSeek V4 Flash across 2× DGX Spark (at least 6 pages). — [GLM thread](https://forums.developer.nvidia.com/t/running-glm-5-1-fp8-on-rtx-pro-6000/366892), [DSV4 Spark thread](https://forums.developer.nvidia.com/t/deepseek-v4-flash-official-fp8-running-across-2x-dgx-spark-tp-2-mtp-200k-ctx-recipe-numbers/370309?page=6)

### Inferences
- The pattern is "kernels exist in someone's fork or in FlashInfer, but not wired or merged upstream." Examples: nvfp4 KV, B12X MoE auto-select, TRITON_MLA_SPARSE, FA #2634. Integration and upstreaming may be as valuable as new kernel authoring.
- Bot auto-closes and stale flags (SGLang #19637, vLLM #31085) show that maintainers are not treating sm_120 as a priority.

### Gaps
- r/LocalLLaMA thread counts were not gathered.
- Reaction counts understate demand: duplicate issues are spread across repos and many users comment rather than react.

## 4. Community forks and prebuilt-wheel repos

### Takeaway
Wheel-packaging repos are popular and active, notably mjun0812 (1.8k stars) and woct0rdho SageAttention (1.1k). Kernel-level sm_120 forks are small (b12x at 270 stars; FlashMLA-sm120 at 8; lna-lab at 29) but increasingly upstreamed via vLLM and FlashInfer. The FA4-sm120 HF bundle (Second Nature Computing) is the main FA-side fork.

### Cited Findings
- mjun0812/flash-attention-prebuild-wheels: 1,801 stars, pushed 2026-11-22. [repo](https://github.com/mjun0812/flash-attention-prebuild-wheels)
- kingbri1/flash-attention: 940 stars, pushed 2025-12-09. [repo](https://github.com/kingbri1/flash-attention)
- woct0rdho/SageAttention: 1,068 stars, pushed 2026-10-09; upstream thu-ml has 4,019 stars, pushed 2026-01-17. [fork](https://github.com/woct0rdho/SageAttention), [upstream](https://github.com/thu-ml/SageAttention)
- local-inference-lab/b12x: 270 stars, pushed 2026-10-06; PyPI 1.2.x. The yatesdr/b12x mirror has 0 stars. [repo](https://github.com/local-inference-lab/b12x), [PyPI](https://pypi.org/project/b12x/)
- SecondNatureComputing/flash-attn-4-sm120: an HF kernels-hub package. [HF](https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120)
- IISuperluminaLII/FlashMLA_Windows_Linux_sm120: 8 stars, pushed 2026-09-05. [repo](https://github.com/IISuperluminaLII/FlashMLA_Windows_Linux_sm120)
- lna-lab/blackwell-geforce-nvfp4-gemm: 29 stars, pushed 2026-04-27. [repo](https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm)
- eous/vllm-sm120 (a Docker image with FlashMLA sparse and DeepGEMM hacks, cited in DeepGEMM #236): the GitHub repo returned 404 on 2026-11-21. [DeepGEMM #236](https://github.com/deepseek-ai/DeepGEMM/issues/236)
- A Seryoger HF SageAttention3 wheel exists (cu130/py3.12). [HF](https://huggingface.co/Seryoger/Sageattention-3-cu130-5090-endpoint)
- FA issue #2016 "[Info Sharing] Success Combination for Blackwell (RTX5090) / NO BUILD TIME (using whl)" (7r) shows users rely on third-party wheels. [#2016](https://github.com/Dao-AILab/flash-attention/issues/2016)

### Inferences
- There is a clear market for "it just installs" packaging, especially on Windows and with CUDA 12.8/13.x and new torch versions. That gap is being filled actively.
- The less-crowded niche is kernel coverage, specifically sparse MLA, fp8/FP4 attention and large-head-dim attention for sm_120.

### Gaps
- No download counts (PyPI or HF) were gathered.
- Star counts for the Second Nature HF package were not available.

## 5. Why maintainers deprioritize sm_120

### Takeaway
The stated reasons are:
- Lack of hardware: DeepSeek says "we don't have SM120 devices."
- Per-architecture maintenance cost: DeepGEMM's GEMM heuristics.
- A fundamentally different tensor-core ISA: sm_120 has warp-level mma.sync (including block-scaled FP4/FP8) and TMA, but no wgmma (sm_90) and no tcgen05/TMEM or 2-CTA (sm_100), so datacenter kernels cannot be retargeted.
- About 99 KB of shared memory, which breaks tile shapes designed for 160–228 KB.

### Cited Findings
- DeepGEMM maintainer LyricZhao: "as we don't have SM120 devices, we don't have plans or people to support this," citing maintenance cost of GEMM heuristics. — [DeepGEMM #236](https://github.com/deepseek-ai/DeepGEMM/issues/236)
- FA contributor johnnynunez: "Flash Attention is Hardware Aware Programming"; FA3 targets Hopper and FA4 targets B200/B300. — [FA #1987](https://github.com/Dao-AILab/flash-attention/issues/1987)
- vLLM PR #41834: SM12x lacks TMEM/tcgen05, so DeepGEMM, FlashMLA and Marlin FP8 paths fail at link time. — [vLLM #41834](https://github.com/vllm-project/vllm/pull/41834)
- Shared memory: 99 KB vs 163 KB (SM80) limits D>128 attention tiles [HF card](https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120). DeepGEMM sm_100 kernels need about 168 KB [DeepGEMM #236](https://github.com/deepseek-ai/DeepGEMM/issues/236). Mamba3 backward requests 220 KB against the 99 KB limit [mamba #1044](https://github.com/state-spaces/mamba/issues/1044).
- Triton on SM120 falls back to dequant + fp16 MMA for MoE (masahi, NVIDIA). — [SGLang #19637](https://github.com/sgl-project/sglang/issues/19637)
- vLLM docs define backend priority lists only for SM 8.x–9.x and 10.x, so 12.x has no curated default ordering. — [vLLM docs](https://docs.vllm.ai/en/latest/design/attention_backends/)

### Inferences
- Because sm_120's MMA model is "Ampere-plus" (mma.sync with new FP4/FP6/FP8 block-scaled variants plus TMA), most good sm_120 kernels are re-tiled SM80-style kernels with low-precision MMA. That is well within reach of one CUDA/CuTe developer, unlike tcgen05 kernels. The binding constraint is the 99 KB smem budget and register pressure.

### Gaps
- There is no official NVIDIA statement about sm_120 software prioritization.
- An NVIDIA CUDA Programming Guide or PTX ISA page listing sm_120 features (smem per SM/block, block-scaled mma.sync shapes, TMA/multicast, cluster limits) was not fetched in this pass.
