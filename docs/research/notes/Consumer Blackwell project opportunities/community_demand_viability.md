# Community Demand and Project Viability for a Solo Consumer-Blackwell (sm_120) Developer — Nov 2026

Research scope note: about 18 search/fetch calls. Reddit threads could not be retrieved directly. Searches restricted to reddit.com returned no r/LocalLLaMA or r/StableDiffusion posts, so "loudest complaints" evidence comes from GitHub issues, project READMEs, vendor/NVIDIA pages and secondary blogs. Treat blog content (willitrunai, openclawdc, localllm.in, aliteq and similar) as anecdotal or lower-authority. Star counts are as displayed on GitHub pages fetched on 2026-11-21 (rounded by GitHub).

## 1. Top recurring needs and complaints from RTX 50 / 24 GB / laptop owners (2025–2026)

### Takeaway
The recurring pain points are (1) software and kernel support lagging on sm_120: missing or broken NVFP4/MoE kernels in serving stacks, plus wheels that don't match torch/CUDA/Python, worst on Windows; (2) VRAM ceilings: MoE expert offloading and KV-cache size for long context on 24–32 GB; and (3) video diffusion (Wan 2.2, LTX-2) spilling past VRAM into slow system-memory offload. The basic "PyTorch doesn't support sm_120" problem from early 2025 is largely solved.

### Cited Findings
**sm_120 kernel gaps in serving stacks (measured: GitHub issues)**
- vLLM issue #33333: the NVFP4 MoE backend `FLASHINFER_CUTLASS` "does not support the deployment configuration since kernel does not support current device" on sm_120 — [vLLM #33333](https://github.com/vllm-project/vllm/issues/33333)
- vLLM v0.10.2: Qwen3-30B-A3B-NVFP4 on a 5090 failed with `no cutlass_scaled_mm kernel` (required capability 90 or 100). A community PR (#24968, by MrVolts) enabled the existing CUTLASS MoE kernels for SM_120 through build-flag changes. This is an example of a small, high-leverage outside contribution — [vLLM #24921](https://github.com/vllm-project/vllm/issues/24921); [vLLM PR #24968](https://github.com/vllm-project/vllm/pull/24968)
- vLLM feature request #31085 asks for "SM120 (RTX 6000/5000 Blackwell) support for native NVFP4 MoE kernels". The MXFP4 backend logic doesn't recognize SM120 and falls back to Marlin — [vLLM #31085](https://github.com/vllm-project/vllm/issues/31085)
- A community repo issue reports the `nvfp4_marlin` kernel crashing on consumer Blackwell (sm_120) for dense models while working for some MoE models — [club-3090 #828](https://github.com/noonghunna/club-3090/issues/828)
- An NVIDIA developer forum thread reports random `cudaErrorIllegalInstruction` crashes on SM120 under load with NVFP4 MoE models. It references vLLM fix #38423 and a CUTLASS bump to v4.4.2 for SM120-family compilation — [NVIDIA forums: state of NVFP4 support](https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069?page=9)
- Older official vLLM Docker images (v0.9.0) shipped no sm_120 kernels, which caused "no kernel image is available" errors — [vLLM forums](https://discuss.vllm.ai/t/docker-image-vllm-vllm-openai-v0-9-0-doesnt-work-on-5090/761/2)

**Wheel/ABI hell, especially on Windows (measured: many community wheel repos)**
- Many separate GitHub repos exist only to ship prebuilt SageAttention wheels for sm_120: mobcat40/sageattention-blackwell, ziggyxp/…-pt211-cu130-cp313, jthickma/sageattention-blackwell-wheels (CUDA 12.9), k1n0F/sageattention3-blackwell-wsl2, Rogala/AI_Attention (xFormers + SageAttention + FlashAttention for ComfyUI on Windows, SM120), and kasparovabi Docker setup — [mobcat40](https://github.com/mobcat40/sageattention-blackwell); [ziggyxp](https://github.com/ziggyxp/sageattention-blackwell-pt211-cu130-cp313); [jthickma release](https://github.com/jthickma/sageattention-blackwell-wheels/releases/tag/v2.2.0-cu129); [k1n0F](https://github.com/k1n0F/sageattention3-blackwell-wsl2); [Rogala/AI_Attention](https://github.com/Rogala/AI_Attention); [kasparovabi](https://github.com/kasparovabi/ComfyUI-SageAttention-sm120-Blackwell-Docker-Setup)
- mobcat40's README says the official SageAttention wheels are built against older PyTorch and fail with DLL load errors on PyTorch 2.11 — [mobcat40 README](https://github.com/mobcat40/sageattention-blackwell/blob/main/README.md)
- ComfyUI discussion #11583 (user-reported benchmark) claims SageAttention on a 5090 is about 30–35% faster, e.g. Qwen Image Edit at 40 steps went from 14m30s to 9m30s — [ComfyUI Discussion #11583](https://github.com/Comfy-Org/ComfyUI/discussions/11583)
- The Nunchaku install guidance stresses that the backend wheel must match your CUDA, PyTorch and Python versions. RTX 50 needs PyTorch ≥2.7 and CUDA ≥12.8, and a mismatched wheel is the common cause of load failures (low-authority blogs) — [cloudgpu.app guide](https://cloudgpu.app/guides/how-to-install-nunchaku-on-a-cloud-gpu); [localaimaster](https://localaimaster.com/blog/comfyui-quantization-format-errors)
- Early sm_120 issues (PyTorch failing with sm_120 errors at launch) now look outdated. Unsloth docs list RTX 5060–5090 as supported, needing triton ≥3.3.1 — [Unsloth Blackwell docs](https://docs.unsloth.ai/basics/training-llms-with-blackwell-rtx-50-series-and-unsloth); [Unsloth issue #1679](https://github.com/unslothai/unsloth/issues/1679)

**VRAM ceilings: MoE offload and long context (mostly blogs and guides)**
- llama.cpp `--n-cpu-moe N` (or `-ot ".ffn_.*_exps.=CPU"`) keeps expert weights in system RAM. Guides disagree on whether it counts from the first or the highest-numbered layers, and recent builds auto-size offload (`--fit on`) — [HF blog, Doctor-Shotgun](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide); [openclawdc](https://openclawdc.com/blog/llama-cpp-moe-offload-flags-explained/); [aliteq](https://aliteq.com/llama-cpp-moe-offloading-run-bigger-models-less-vram)
- Anecdotal: Qwen3.6-35B-A3B runs at 64K context on 24 GB with `--n-cpu-moe 12`, about 20 t/s at 64K decode with `--n-cpu-moe 24`. Separately, an RTX 3060 12GB with 32 GB RAM reports about 51–53 t/s at IQ4_NL / 64K ctx with `--n-cpu-moe 25` — [localllm.in](https://localllm.in/blog/best-local-llms-24gb-vram); [aliteq](https://aliteq.com/n-cpu-moe-llama-cpp-what-it-actually-does)
- On 24 GB, Q4 27–35B models fit comfortably, and 70B at Q4 only barely, with almost no room for KV cache — [llmconfigurator RTX 5090 Laptop](https://llmconfigurator.com/en/best-models/rtx-5090-laptop); [willitrunai](https://willitrunai.com/gpus/rtx-5090-laptop-24gb)
- A llama.cpp research issue on KV-cache quantization (KVarN, variance-normalized KV quant, #24139, opened 2026-06-04) shows ongoing interest in shrinking KV cache — [llama.cpp issues by reactions](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc)

**Video diffusion speed and VRAM**
- NVIDIA's LTX-2 guide: on a 5090, a 720p 4s clip took about 25 s, but an 8 s clip that spilled past 32 GB took about 3 minutes because of weight streaming to system memory — [NVIDIA LTX-2 ComfyUI guide](https://www.nvidia.com/en-sg/geforce/news/rtx-ai-video-generation-guide/)
- Wan 2.2 14B's reference single-GPU setup asks for ≥80 GB VRAM, so consumer users run fp8 or GGUF with RAM offload. The 5B model runs in about 8 GB with ComfyUI native offloading — [Hivenet Wan 2.2 guide](https://www.hivenet.com/post/wan-2-2-cloud-gpu-comfyui)
- ComfyUI issue #9177 (2025): extreme slowdowns at KSampler on a 5090, attributed to spill into slow shared GPU memory when VRAM is nearly full — [ComfyUI #9177](https://github.com/Comfy-Org/ComfyUI/issues/9177)
- NVIDIA (CES 2026) says ComfyUI added native NVFP4/FP8 support, claiming about 3x performance and 60% lower VRAM with NVFP4 on RTX 50 — [NVIDIA blog (Spanish)](https://la.blogs.nvidia.com/blog/rtx-ai-garage-ces-2026-open-models-video-generation)

**Fine-tuning on 24 GB**
- 24 GB is considered workable for QLoRA on 7B to about 32B models with Unsloth. Sources disagree on the floor (16 GB vs 24 GB) — [jakeinsight 2026-08](https://jakeinsight.com/buying-guide/2026-08-09-fine-tune-llm-laptop-gpu-worth-2026/); [computingforgeeks](https://computingforgeeks.com/best-laptop-for-local-llm/)
- Unsloth claims "2× faster with 70% less VRAM" across many models, including gpt-oss-20B GRPO at 80% less. These are vendor claims without methodology on the page — [unslothai/unsloth](https://github.com/unslothai/unsloth)

**NVFP4 on consumer Blackwell (measured, academic)**
- arXiv 2601.09527 (Jan 2026) benchmarked RTX 5060 Ti, 5070 Ti and 5090 desktop cards: NVFP4 gives 1.6x throughput over BF16, 41% energy reduction and 2–4% quality loss. It did not cover laptops or llama.cpp — [arXiv 2601.09527](https://arxiv.org/html/2601.09527v1)

### Inferences
- Ranked by evidence volume and how well it fits a CUDA developer: (1) NVFP4/FP4 kernel coverage on sm_120 (vLLM/FlashInfer/CUTLASS MoE paths, Marlin crashes, dense vs MoE), (2) attention kernels plus prebuilt wheels for sm_120 on Windows and new torch versions, (3) MoE expert offload and CPU/GPU split efficiency on 24 GB, (4) KV-cache compression for long context, (5) video-diffusion memory and offload efficiency, (6) 24 GB fine-tuning (largely served by Unsloth already).
- sm_120 is the "orphan" Blackwell target. Datacenter sm_100 gets kernels first (tcgen05/TMEM paths don't exist on sm_120), so consumer-Blackwell ports of datacenter FP4 kernels are a recurring gap. This inference is consistent with "Required capability: 90 or 100" errors.
- Wheel-repo proliferation (6+ near-duplicate SageAttention-for-Blackwell repos) signals real demand, but also low differentiation and quick obsolescence: each repo goes stale with the next torch release.

### Gaps
- No direct Reddit data (upvote counts, thread frequency) was retrievable. The ranking is inferred from GitHub and guides.
- No reaction counts for vLLM, ComfyUI, ollama, diffusers or Unsloth issues were retrieved.
- No SGLang-specific sm_120 evidence was found.
- Hugging Face forum complaints were found only incidentally (an RTX 5060 Ti SadTalker thread, not analyzed).

## 2. Unresolved high-reaction feature requests in local-inference tools relevant to GPU kernels

### Takeaway
In llama.cpp, the most-reacted open issues are mostly non-CUDA: new backends (Apple ANE, AMD XDNA, DirectML), Mamba GPU support, server features. CUDA-relevant open asks are Mamba/SSM GPU support, KV-cache quantization research, disaggregated prefill/decode and new-model support. The clearer kernel-level gaps for an sm_120 CUDA developer are in vLLM (NVFP4/MXFP4 MoE on SM120).

### Cited Findings
- llama.cpp open issues sorted by 👍 (page 1). The fetched counts appear to be comments, not reactions:
  - ANE backend #10453
  - XDNA backend #21725 (2026-04)
  - DirectML backend #7772
  - Sparse MoE models #5365
  - Metal vs Vulkan perf #10982
  - OpenAI v1/responses #14702
  - GPU support for Mamba models #6758 (labelled Nvidia GPU)
  - grammar sampling speed-up #4218
  - disaggregated prefill/decode #21266 (2026-04)
  - KVarN KV-cache quant #24139 (2026-06)
  - GLM5.3 support #27922 (2026-08)
  - Vulkan device lost #27076 (2026-08)

  None mention Blackwell, NVFP4 or MTP — [llama.cpp issues by reactions](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc)
- vLLM SM120 NVFP4 MoE: #31085 (feature), #33333 (bug), #35065 (Nemotron-3-Nano-30B-A3B-NVFP4 fails on 5090; the page text may be partially spammy) — [#31085](https://github.com/vllm-project/vllm/issues/31085); [#33333](https://github.com/vllm-project/vllm/issues/33333); [#35065](https://github.com/vllm-project/vllm/issues/35065)

### Inferences
- llama.cpp's CUDA backend is mature and heavily contributed to, so a newcomer's marginal impact there is in narrow PRs (e.g., FP4 matmul paths for sm_120, MoE offload overlap, Mamba/SSM kernels), not a standalone project.
- vLLM/FlashInfer SM120 FP4 MoE is a concrete, upstreamable niche, and an outside contributor has already landed a fix (#24968). Upstreaming earns credibility but not a standalone star count.

### Gaps
- Reaction-sorted lists for ollama, ComfyUI, diffusers and Unsloth were not fetched, so actual reaction numbers are unknown.

## 3. Case studies: solo/small-team gap-filler projects (stars, speed, upstreaming, maintenance)

### Takeaway
The biggest wins pair a novel kernel or quantization technique with easy install for consumer workflows (ComfyUI or HF notebooks): Unsloth (~79.9k stars), SageAttention (~4.0k), Nunchaku (~4.0k), triton-windows (~1.9k, later adopted by the triton-lang org). Pure wheel-repackaging repos have very low stars and go stale. The maintenance load is dominated by keeping up with torch/CUDA/Python ABI churn and new model architectures.

### Cited Findings
- **Unsloth** (Daniel and Michael Han, small team): 79.9k stars, 7.4k forks, 15,112 commits, 200 open issues and 350 open PRs as of the 2026-11 fetch. Its pitch is "2× faster, 70% less VRAM" fine-tuning, and it supports RTX 30/40/50, Blackwell and DGX Spark — [unslothai/unsloth](https://github.com/unslothai/unsloth). NVIDIA now co-markets it on its developer blog — [NVIDIA dev blog](https://developer.nvidia.com/blog/train-an-llm-on-an-nvidia-blackwell-desktop-with-unsloth-and-scale-it/)
- **SageAttention** (Tsinghua thu-ml, academic group, not solo): 4.0k stars, 561 forks, 179 open issues, 43 open PRs.
  - Timeline: RTX 5090 compile support 2025-02-15; SageAttention2++ 2025-07-01; SageAttention3 (FP4 microscaling, `sageattention3_blackwell`) NeurIPS 2025 Spotlight, 2025-09-27.
  - Claims "2–5x" over FlashAttention, and 560 TOPS on RTX 5090, 2.7x faster than FA2.
  - The README doesn't mention Windows wheels, and that gap was filled by community wheel repos.

  — [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention); [SageAttention #107 (Blackwell compile)](https://github.com/thu-ml/SageAttention/issues/107)
- **Nunchaku / SVDQuant** (MIT Han Lab lineage; the repo now redirects to the nunchux-ai org): about 4.0k stars, 289 forks, 5 open issues, 26 open PRs, 567 commits.
  - Release cadence: NVFP4 on RTX 5090 2025-02-20 (~3x over BF16); v0.3.0 2025-06; v1.0.0 2025-09-04; v1.2.0 2026-01-12.
  - Frequent 4-bit model drops (FLUX Kontext/Krea, Qwen-Image, Qwen-Image-Edit-2509, Z-Image-Turbo).
  - Distributed through a separate ComfyUI-nunchaku plugin.
  - ICLR 2025 Spotlight.

  — [nunchaku repo](https://github.com/nunchaku-tech/nunchaku)
- Nunchaku's latest dated news on the fetched page is 2026-01-12. Combined with 26 open PRs, momentum may have slowed in 2026 (inference). Separately, ComfyUI's native NVFP4 support (CES 2026) partly overlaps its value — [NVIDIA blog](https://la.blogs.nvidia.com/blog/rtx-ai-garage-ces-2026-open-models-video-generation)
- **triton-windows** (woct0rdho, with credited earlier contributors): about 1.9k stars, 98 forks.
  - It filled the "Triton doesn't run on Windows" gap that blocked SageAttention, torch.compile, ComfyUI and FramePack on Windows.
  - Archived 2026-02-18 after development moved to the official `triton-lang/triton-windows` repo, maintained by @woct0rdho and @jammm. This is a clear case of a community gap-filler being adopted by the upstream org.
  - Supports sm120 with Triton ≥3.3, PyTorch ≥2.7 and CUDA ≥12.8.

  — [woct0rdho/triton-windows](https://github.com/woct0rdho/triton-windows)
- **ExLlamaV3** (turboderp, essentially solo): EXL3 format based on QTIP (Cornell RelaxML). Star count is uncertain: about 1,300 per a 2026 secondary article vs 756 on an older listing. A 2026 update reportedly added MoE CPU offloading and self-calibrated quantization (secondary source) — [exllamav3 README](https://raw.githubusercontent.com/turboderp-org/exllamav3/master/README.md); [promptquorum](https://www.promptquorum.com/es/power-local-llm/exllamav2-explained); [sourcepulse](https://www.sourcepulse.org/projects/2147040); [0dev post](https://0dev.duckdns.org/en/posts/exllamav3-update-adds-moe-cpu-offloading-and-self-calibrated-quantization)
- **Community sm_120 wheel repos** (mobcat40, ziggyxp, jthickma, Rogala, k1n0F): star counts weren't shown in search excerpts. Their value is tied to specific torch versions (e.g., "PyTorch 2.11 nightly, CUDA 12.8/13.0, cp313") — [mobcat40](https://github.com/mobcat40/sageattention-blackwell); [Rogala/AI_Attention](https://github.com/Rogala/AI_Attention)

### Inferences
- Success pattern:
  - A measurable speedup or VRAM cut (≥1.3x end-to-end, or a fit/no-fit change) on popular models (FLUX, Qwen-Image, Wan, Llama/Qwen).
  - Delivery through the dominant UI or workflow (ComfyUI node, HF/Colab notebook, pip wheel).
  - Fast support for each new hot model.
  - Often a paper or academic backing (SageAttention, Nunchaku) or relentless release cadence (Unsloth).
- Speed: the NVFP4 5090 support in Nunchaku and SageAttention came within weeks of the Jan 2025 RTX 50 launch (Feb 2025), so first-mover timing on new hardware matters.
- Maintenance burden is high and ongoing: Unsloth carries 350 open PRs, SageAttention 179 open issues, and wheel repos must rebuild for every torch/CUDA/Python combination. Repackaging-only projects get superseded (by upstream wheels or official orgs) within months. triton-windows avoided abandonment by being absorbed upstream.
- For a solo developer, "build-matrix wheels" alone is a low-moat, high-churn project. A kernel with a defensible technique, upstreamed or packaged as a ComfyUI/llama.cpp integration, has the better impact profile.

### Gaps
- No star-history time series (star-history.com) was retrieved, so growth rates per month aren't quantified.
- Unsloth's early-days growth (2023–2024 star trajectory) wasn't researched.
- Individual llama.cpp CUDA contributor case studies (e.g., JohannesGaessler's CUDA work) weren't researched.
- Whether the community SageAttention wheel repos were made obsolete by official wheels wasn't verified.

## 4. Laptop-specific constraints (RTX 5090 Laptop, 24 GB, sm_120)

### Takeaway
The RTX 5090 Laptop is a GB203 (desktop-5080-class) chip with 10,496 CUDA cores, 24 GB GDDR7 on a 256-bit bus at 896 GB/s, and a 150 W TGP (+25 W Dynamic Boost). The desktop 5090 has 21,760 cores, 1.79 TB/s and 575 W. Measured AI throughput is roughly 2x lower than desktop when plugged in and up to 4x lower on battery, so memory-bandwidth-bound decode and VRAM capacity dominate.

### Cited Findings
- Specs: 5090 Mobile 24 GB, 256-bit, 896.0 GB/s, 150 W TDP; desktop 5090 32 GB, 512-bit, 1.79 TB/s, 575 W; 4090 Mobile 16 GB, 576 GB/s — [StorageReview, 2025-03-27](https://www.storagereview.com/review/nvidia-geforce-rtx-5090-mobile-review-ai-power-meets-efficiency)
- Measured Procyon LLM tokens/s (TensorRT), battery / plugged in / desktop 5090:
  - Phi: 98.6 / 163.1 / 314.4
  - Mistral: 67.0 / 125.6 / 255.9
  - Llama3: 58.4 / 107.9 / 214.3
  - Llama2: 30.2 / 61.3 / 134.5

  — [StorageReview](https://www.storagereview.com/review/nvidia-geforce-rtx-5090-mobile-review-ai-power-meets-efficiency)
- Measured SDXL FP16 time per image: 29.9 s on battery, 13.4 s plugged in, 5.2 s desktop. SD1.5 INT8: 1.006 / 0.860 / 0.394 s. Razer Blade 16 runs the GPU at 135 W in performance mode, with an estimated GPU draw of about 137.5 W — [StorageReview](https://www.storagereview.com/review/nvidia-geforce-rtx-5090-mobile-review-ai-power-meets-efficiency)
- Notebookcheck: the fastest configurations are 175 W (150 W TGP + 25 W Dynamic Boost). The 5090 Laptop performs almost identically to a desktop RTX 4070 Ti Super in gaming. Spec INT8 is 1334 TOPS (laptop) vs 1824 (desktop). OEM power limits range from about 120–175 W — [Notebookcheck analysis](https://www.notebookcheck.net/Nvidia-GeForce-RTX-5090-Laptop-Analysis-Faster-but-there-is-more-to-come.988519.0.html); [Notebookcheck vs 4070 Ti Super](https://www.notebookcheck.net/GeForce-RTX-5090-laptops-perform-almost-identically-to-a-desktop-RTX-4070-Ti-Super.993212.0.html); [Notebookcheck GPU list](https://www.notebookcheck.net/mobile-graphics-cards-benchmark-list.844.0.html)
- Reviewers call 5090 laptops poor value vs 5080 laptops at the same die — [TechRadar](https://www.techradar.com/computing/gpu/new-tests-cast-a-disappointing-light-on-nvidias-rtx-5090-laptop-gpu-suggesting-that-at-todays-prices-rtx-5080-notebooks-are-a-far-better-buy); [Windows Central](https://www.windowscentral.com/hardware/laptops/nvidia-rtx-5090-laptops-terrible-value)

### Inferences
- At 896 GB/s, single-stream decode of a dense 30B Q4 model (~17 GB of weights) is capped around 50 t/s theoretical. Weight compression (FP4/NVFP4, EXL3-style trellis quants) and KV-cache compression pay off proportionally more on the laptop than on the desktop.
- The 150–175 W power cap makes energy-efficient kernels (FP4 tensor cores, reduced data movement) matter. The arXiv result of NVFP4 cutting energy by 41% is especially relevant for laptops ([arXiv 2601.09527](https://arxiv.org/html/2601.09527v1)).
- The 24 GB laptop sits in an awkward class: more VRAM than the desktop 5080 (16 GB) but desktop-5080 compute. Workloads that matter most are therefore capacity-bound: MoE offload, long-context KV and video diffusion memory. A developer on this machine can credibly target the large 16–24 GB user segment (4090 desktop, 5080/5090 laptop, 3090) and not just 5090 desktops.
- Testing caveat: benchmarks from a laptop vary with OEM power limit and thermals, so the developer should report power mode and TGP with results.

### Gaps
- No measured sustained (multi-hour) AI training or inference throttling data for the 5090 Laptop was found.
- No llama.cpp t/s measurements specific to the 5090 Laptop were found.

## 5. Framework landscape: TensorFlow's decline vs PyTorch/JAX/Keras 3 — is anything TF-related still worth building?

### Takeaway
Google's own TensorFlow 2.21 announcement (2026-03-06) says TF will "exclusively focus on" security and critical bug fixes, dependency updates and reviewing community bug fixes. It recommends Keras 3, JAX and PyTorch for new generative-AI work, and LiteRT for on-device. TF still has about 12M monthly PyPI downloads (legacy/production), but there is little evidence of users demanding TF on new consumer GPUs. A TF sm_120 build project has weak impact potential.

### Cited Findings
- The TF 2.21 blog (2026-03-06) says TensorFlow "continues to provide stability for production" and "Going forward, we will exclusively focus on" security/critical bug fixes, dependency updates (incl. new Python releases) and review of community bug fixes. It recommends Keras 3, JAX and PyTorch for new GenAI projects. LiteRT is "the universal on-device inference framework," 1.4x faster GPU than TFLite. There is no mention of CUDA versions or new GPU support — [Google Developers Blog: What's new in TF 2.21](https://developers.googleblog.com/whats-new-in-tensorflow-221/)
- Secondary coverage frames this as TF entering maintenance mode. The official post doesn't use that phrase; it is a paraphrase of the "exclusively focus on" fixes language — [MarkTechPost 2026-03-06](https://www.marktechpost.com/2026/03/06/google-launches-tensorflow-2-21-and-litert-faster-gpu-performance-new-npu-acceleration-and-seamless-pytorch-edge-deployment-upgrades/)
- TF 2.20 (Aug 2025) had already started deprecating `tf.lite` in favor of LiteRT, moving it to a separate repository — [Hackster](https://www.hackster.io/news/tensorflow-2-20-sets-the-stage-for-tensorflow-lite-s-deprecation-supplantation-by-litert-next-4a4efe68d03c); [TF blog 2.20](https://blog.tensorflow.org/2025/08/whats-new-in-tensorflow-2-20.html?hl=hi)
- PyPI (measured, pypistats, fetched 2026-11-21): tensorflow had 295,051 downloads last day, 2,439,336 last week and 12,247,917 last month — [pypistats tensorflow](https://pypistats.org/packages/tensorflow)
- Secondary/unverified figures from web-search summaries:
  - torch about 59.0M, tensorflow 11.9M, keras 11.4M, jax 8.7M in the 30 days to 2026-10-28.
  - TF monthly downloads fell 25% from April to August 2026 (16.7M in August); torch peaked at 98.6M in July.
  - Keras downloads ≈ TF, partly because Keras 3 runs on JAX and PyTorch.

  The original sources couldn't be pinned; candidates are [Second Talent](https://www.secondtalent.com/resources/pytorch-vs-tensorflow-usage-popularity-and-performance/) and [tech-insider](https://tech-insider.org/pytorch-vs-tensorflow-2026/), and these should be treated cautiously. Download counts include CI runs, and PyTorch's CUDA builds served from download.pytorch.org aren't counted by PyPI, which undercounts torch.
- Keras 3 is multi-backend (JAX, TensorFlow, PyTorch) — [Wikipedia: Keras](https://en.wikipedia.org/wiki/Keras)

### Inferences
- With Google committed only to fixes and dependency updates, official TF GPU wheels may lag new CUDA/arch support. In principle that leaves a gap for TF on sm_120. But the demand evidence is nil: no TF complaints showed up anywhere in the RTX 50 community material (ComfyUI, llama.cpp, vLLM, Unsloth issues). Consumer-GPU users are almost all on PyTorch and GGUF stacks.
- Remaining TF users are mostly legacy production, TFX/Serving and edge (LiteRT). Few of them run on RTX 50 laptops, and those who do can often use Keras 3 with a PyTorch or JAX backend, which already supports sm_120. The marginal impact of a TF sm_120 custom build is therefore low.
- If anything TF-adjacent is worth building, the stronger bets are Keras 3 backend-agnostic ops, or a LiteRT/PyTorch conversion path. Even these sit outside the consumer-GPU pain points identified in Q1–Q4.

### Gaps
- Stack Overflow Developer Survey 2025/2026 framework shares weren't retrieved.
- Papers with Code framework share wasn't retrieved (the site was sunset in 2025, so no current series may exist).
- Whether TF 2.21 wheels include sm_120 SASS/PTX wasn't verified from the release notes.
- PyPI numbers for torch, jax and keras weren't fetched directly from pypistats.
