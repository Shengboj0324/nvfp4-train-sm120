# GPU-Architecture Packaging and Compatibility Failures in the Python ML Ecosystem (as of 2026-11-21)

Scope: "no kernel image is available" failures, wheels missing SASS/PTX for new GPUs (sm_120 RTX 50, sm_121 DGX Spark/GB10), CUDA 12 to 13 transitions and dropped arches, broken shared-library search paths (RUNPATH/dlopen), existing diagnostic tooling, and whether there is an unmet need for an arch-coverage scanner / CI check / compatibility database.

Method note: Issue counts below came from the unauthenticated GitHub search API (`api.github.com/search/issues`), queried on 2026-11-21. GitHub's search endpoint counts issues and PRs together unless `is:issue` is given. The search matches title or body text, so counts are an upper-bound proxy for "threads that mention the term", not a count of distinct bugs. Some queries hit the anonymous rate limit (10/min) and are marked as missing.

## Q1. How common are these failures for sm_120 / sm_121 / dropped-arch users?

### Takeaway
The failure is widespread and long-running. Across major ML repos there are hundreds of GitHub threads mentioning sm_120 and "no kernel image is available" (PyTorch: 116 "no kernel image" and 164 sm_120 threads; vLLM: 133 / 560, plus 378 sm_121). GitHub-wide there are 3,501 "no kernel image is available" issues, 1,499 of them opened since 2025-01-01. The pain has moved from PyTorch core, which has shipped sm_120 since 2.7/cu128 in April 2025, to the long tail of third-party extension wheels: flash-attn, xformers, SageAttention, FlashInfer, spconv, onnxruntime-gpu, and community wheel repos. sm_121 (DGX Spark/GB10) is the newest gap: stock PyTorch wheels compile up to sm_120 only. Older GPUs now fail in the other direction as CUDA 13 builds drop sm_50 to sm_70.

### Cited Findings
**GitHub thread counts (issues+PRs mentioning the term, retrieved 2026-11-21 via GitHub search API):**

| Repo | "no kernel image is available" | sm_120 | sm_121 |
|---|---|---|---|
| pytorch/pytorch | 116 | 164 | 28 |
| tensorflow/tensorflow | 9 | 17 | 2 |
| jax-ml/jax | 15 | 16 | 2 |
| Dao-AILab/flash-attention | 16 | 55 | 15 |
| bitsandbytes-foundation/bitsandbytes | 20 | 4 | 2 |
| facebookresearch/xformers | 7 | 10 | 1 |
| abetlen/llama-cpp-python | 0 | 5 | 0 |
| microsoft/onnxruntime | (query failed: "Validation Failed") | (failed) | (failed) |
| cupy/cupy | 6 | 13 | 1 |
| vllm-project/vllm | 133 | 560 | 378 |

**GitHub-wide (all public repos, `is:issue` only):** "no kernel image is available" = **3,501 issues** total, of which **1,499 (~43%) were created on or after 2025-01-01**, i.e. in the Blackwell/CUDA 12.8+ era. "sm_120" = **6,487 issues**.

— Source: GitHub REST search API, e.g. [api.github.com/search/issues?q=repo:pytorch/pytorch "no kernel image is available"](https://api.github.com/search/issues?q=repo:pytorch/pytorch+%22no+kernel+image+is+available%22); [global query](https://api.github.com/search/issues?q=%22no+kernel+image+is+available%22+is:issue+created:%3E=2025-01-01)
- vLLM is the hotspot (560 sm_120 and 378 sm_121 mentions). Its many custom CUDA kernels and its popularity on DGX Spark make it the single largest concentration of arch-coverage threads among the repos sampled — same API source.

**Representative sm_120 (RTX 50) failures in third-party wheels:**
- flash-attn on RTX 5090: "no kernel image is available" from flash_fwd_launch_template.h with flash-attn 2.8.3 / CUDA 12.8; a separate report ties it to xformers 0.0.30 + torch 2.7.0+cu128 — [ostris/ai-toolkit #475](https://github.com/ostris/ai-toolkit/issues/475); [flash-attention #1763](https://github.com/Dao-AILab/flash-attention/issues/1763); [flash-attention #2168 (RTX 5090, WSL2)](https://github.com/Dao-AILab/flash-attention/issues/2168); [flash-attention #1638 "Support for RTX 5090 (CC 12.0)"](https://github.com/Dao-AILab/flash-attention/issues/1638)
- onnxruntime-gpu on an RTX 5090 Laptop: `cudaErrorNoKernelImageForDevice` during inference (ComfyUI/Wan2.2 DWPreprocessor) — [microsoft/onnxruntime #26181](https://github.com/microsoft/onnxruntime/issues/26181)
- dots.ocr on RTX 5090: "no kernel image is available" — [studio-dots-ai/dots.ocr #173](https://github.com/studio-dots-ai/dots.ocr/issues/173)
- SageAttention cu130/Torch 2.10 wheel shipped without sm_120: the arch list for that entry was "8.0 9.0" and did not override the global list the way the 2.9 and 2.11 entries did — [Comfy-Org/wheels #22](https://github.com/Comfy-Org/wheels/issues/22)
- A long-running PyTorch forum thread on sm_120 support ran to 4+ pages; posts confirm cu128+ wheels contain sm_120 and cu126 and older do not — [PyTorch Forums "Pytorch support for sm120" p.4](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099?page=4); [RTX 5060 thread](https://discuss.pytorch.org/t/pytorch-support-for-sm-120-nvidia-geforce-rtx-5060/220941); [RTX 5070 Ti thread](https://discuss.pytorch.org/t/nvidia-geforce-rtx-5070-ti-with-cuda-capability-sm-120/221509)
- PyTorch issue "Add official support for CUDA sm_120 (RTX 5090 / Blackwell architecture)" — [pytorch/pytorch #159207](https://github.com/pytorch/pytorch/issues/159207)
- Community "fixed wheel" repos exist specifically to work around this: prebuilt spconv v2.3.8 wheels "with native Blackwell (RTX 50-series, sm_120) kernels" — [fafraob/spconv-wheels](https://github.com/fafraob/spconv-wheels); SageAttention/FlashAttention Blackwell cu130 wheels on HF — [JahJedi HF repo](https://huggingface.co/JahJedi/sageattention-flashattn-blackwell-cu130-torch211-cp312); an RTX 5070 Ti laptop sm_120 how-to repo — [Timothyoverton repo](https://github.com/Timothyoverton/NVIDIA-GeForce-RTX-5070-Ti-Laptop-GPU-sm_120-py-support)
- An SEO "fix" content industry has formed around the error: e.g. "CUDA Error: No Kernel Image Is Available? Fix the PyTorch sm_120 and Old-GPU Mismatch (2026)" — [runaihome.com](https://runaihome.com/blog/cuda-no-kernel-image-available-pytorch-fix-2026/); "No Kernel Image Is Available on RTX 50-Series: The sm_120 Fix" — [localaimaster.com](https://localaimaster.com/blog/no-kernel-image-available-rtx-50)
- Downstream projects add manual guards and docs: a robotics project's PR "Document the sm_120 kernels a policy image needs" suggests checking `torch.cuda.get_arch_list()` contains sm_120 — [Positronic PR #1012](https://github.com/Positronic-Robotics/positronic/pull/1012); VoiceStudio corrected a mistaken claim that torch lacked sm_120 (torch 2.8.0+cu128 does include it) — [VoiceStudio PR #2084](https://github.com/debpalash/VoiceStudio/pull/2084). These show that even maintainers misdiagnose arch coverage.

**sm_121 (DGX Spark / GB10):**
- Pre-built PyPI PyTorch wheels (cu128 stable and cu130 nightly) compile kernels up to sm_120, and sm_121 is "not included", which is why custom builds circulate — [HF Qanatpharma/pytorch-sm121-gb10 README](https://huggingface.co/Qanatpharma/pytorch-sm121-gb10/blob/main/README.md). An NVIDIA-forum user on a Dec 2025 NGC container reported compute_120 and sm_100 to sm_120 but no sm_121 (via search summary of [NVIDIA forums "Effective PyTorch and CUDA"](https://forums.developer.nvidia.com/t/effective-pytorch-and-cuda/348230?page=2); not independently fetched).
- Note: sm_121 can run sm_120 SASS only if built as family target 120f, or via PTX JIT. Plain sm_120 cubins are not guaranteed binary-compatible with 12.1. See the Q2 family-specific section.
- FlashInfer "DGX Spark (SM121) Current Support Audit": wheel build arch targets lacked 12.0a/12.1a variants, so wheels lacked NVFP4/MXFP4 MMA without JIT. Fix PR #3907 adds 12.1a to the aarch64 jit-cache wheel targets — [flashinfer #3170](https://github.com/flashinfer-ai/flashinfer/issues/3170)
- FlashInfer #3294 (opened 2026-05-11, closed 2026-05-21): user reported `flashinfer-cubin` 0.6.11 shipped only Sm100a/Sm100f/Sm103a cubins and no sm_120/121. The reporter diagnosed with `find -name "*.cubin" | wc -l` (12,681 files), filename grepping, then `cuobjdump --list-elf` on the .so files. The maintainer explained that closed-source cubin backends do not support sm_12x, and that open-source sm_12x kernels ship in `flashinfer-jit-cache` built as 120f. The reporter partly retracted. This is a clear example of users hand-rolling an arch audit and getting it partly wrong — [flashinfer #3294](https://github.com/flashinfer-ai/flashinfer/issues/3294)

**Older GPUs (Maxwell/Pascal/Volta) after CUDA 13:**
- PyTorch 2.8+ cu128 wheels drop sm_50 and sm_60. CUDA 13.0 starts supported architectures at Turing (sm_75) — [Qualiteg (updated 2026-09-23)](https://journal.qualiteg.com/pytorch_and_supported_gpu_version/)
- Measured PyTorch 2.13.0 arch lists (2026-08-24): cu126 = sm_50, 60, 70, 75, 80, 86, 90; cu130 and cu132 = sm_75, 80, 86, 90, 100, 120. So cu13x drops sm_50/60/70, and cu126 lacks sm_120 — [VRAM Lab](https://vramlab.com/posts/pytorch-2-13-rtx-5060-ti-cuda-wheels/)
- Consumer-oriented articles on Pascal/Volta owners after CUDA 13 — [localaimaster "GTX 1080 Ti and Tesla P40 After CUDA 13 Dropped Pascal"](https://localaimaster.com/blog/pascal-gpus-after-cuda-13); [eurokommerz "Replacing V100 After CUDA 13"](https://eurokommerz.at/blog/replacing-v100-after-cuda-13/)
- Older-GPU breakage is not only about SASS. TF 2.21's `and-cuda` extra allows `nvidia-cudnn-cu12>=9.3.0.75,<10`, so pip pulled cuDNN 9.26.0.51, whose convolutions failed on a GTX 1060 (sm_61) with "Profiling failure on CUDNN engine ... status: 5003". Pinning `nvidia-cudnn-cu12==9.3.0.75` worked (reported 2026-09-16) — [GretaLupi/hamlet-toolkit #1](https://github.com/GretaLupi/hamlet-toolkit/issues/1)

### Inferences
- PyTorch core is no longer the main sm_120 offender (cu128/cu130/cu132 all include sm_120). Most current failures come from (a) users on the wrong index (cu126), and (b) third-party CUDA extensions whose arch lists are set independently of PyTorch's. A scanner that inspects every installed package, not just torch, targets exactly this gap.
- sm_121 is the next wave. It is a small but high-value user base (DGX Spark buyers) where the sm_120 vs 120f vs 121a distinction matters and is poorly understood.
- Dependency-range drift (cuDNN 9.26 vs 9.3 on Pascal) shows that "compatibility" also covers library-version vs arch matrices, not only fatbin contents.
- The proliferation of "fix" blogs and community wheel repos signals real demand, but users mostly need a diagnosis ("which package is the culprit?") more than a one-size fix.

### Gaps
- Could not retrieve counts for onnxruntime (GitHub search returned "Validation Failed" twice, possibly a repo-qualifier issue). Per-repo counts are mention-based and include PRs, with no deduplication. Global counts are issues only.
- No Stack Overflow frequency data was gathered (no SO API query was run).
- No quantitative data on sm_121 user population or DGX Spark sales.

## Q2. CUDA 13 changes, family-specific targets, PyTorch wheel arch lists, wheel variants (PEP 817/825)

### Takeaway
CUDA 13.0 removed offline compilation and library support for Maxwell, Pascal and Volta (cc < 7.5). Family-specific targets (`sm_100f`, `sm_120f`, available since CUDA 12.9) let one cubin cover sm_120 and sm_121, but not across families (100f does not run on 120). Arch-specific `a` targets give no forward compatibility. PyTorch's cu130/cu132 wheels cover sm_75 to sm_120 (+PTX), and cu126 still covers sm_50 to sm_90. Wheel variants shipped for PyTorch 2.10 (Feb 2026) through an experimental variant-enabled uv. In Sept 2026 PEP 817 was being reworked into an Informational umbrella, with PEP 825 for format, and uv was vendoring an NVIDIA variant provider (Oct 2026). Variants select by CUDA/driver version, not by verified per-package arch coverage.

### Cited Findings
**CUDA 13 dropped architectures:**
- CUDA 13.0 release notes (Update 3, PDF dated Apr 13, 2026): offline compilation and library support for Maxwell, Pascal and Volta removed in CUDA 13.0. 12.x toolkits can still build for them — [NVIDIA CUDA 13.0.3 Release Notes PDF](https://docs.nvidia.com/cuda/archive/13.0.3/pdf/CUDA_Toolkit_Release_Notes.pdf)
- Secondary: CUDA 13 drops sm_50 to sm_72, starts at sm_75 — [Qualiteg](https://journal.qualiteg.com/pytorch_and_supported_gpu_version/); [michaelbommarito.com CUDA 13 overview](https://michaelbommarito.com/wiki/programming/tools/cuda-13/); [imec iLab.t GPU software compatibility](https://doc.ilabt.imec.be/ilabt/gpulab/gpu-software-compatibility.html)
- CUDA 13.4 adds Rubin and requires an R615+ driver (per Qualiteg, no date given) — [Qualiteg](https://journal.qualiteg.com/pytorch_and_supported_gpu_version/). Current NVIDIA binary-utilities docs are at 13.4 (Oct 12, 2026) and 13.5 — [CUDA Binary Utilities 13.4 PDF](https://docs.nvidia.com/cuda/pdf/CUDA_Binary_Utilities.pdf); [13.5 docs](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)

**Family-specific / arch-specific targets:**
- NVIDIA Programming Guide: family-specific targets were introduced with CC 10.0 devices. compute_120f is compatible with 12.0 and 12.1. Family-target code runs only on GPUs in that family — [CUDA Programming Guide, Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html); [13.4.3 archive](https://docs.nvidia.com/cuda/archive/13.4.3/cuda-programming-guide/05-appendices/compute-capabilities.html)
- sm_120f/compute_120f is new in CUDA 12.9 and covers sm_120 and sm_121. sm_100f code does not run on sm_120 and vice versa — [Arnon Shimoni](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
- FlashInfer CUDA-version to target mapping: below 12.9 use 12.0a; from 12.9 use 12.0f (covers SM120+SM121, includes dense NVFP4/MXFP4 MMA). Most SM12x features are family-specific from PTX ISA 8.8 — [flashinfer #3170](https://github.com/flashinfer-ai/flashinfer/issues/3170)
- ONNX Runtime PR fixing CUDA arch gating for SM12x, family/arch-specific targets and `-virtual` entries (90/100/110/120 require native code for their family). This shows the build-system logic is error-prone even in major projects — [microsoft/onnxruntime PR #33373](https://github.com/microsoft/onnxruntime/pull/33373)
- Arch-specific `sm_120a` gives no compatibility with future chips; prefer `f` when available — [agirault gist](https://gist.github.com/agirault/d0d66c12c85dbc2662c95569ef039a1d)
- The current nvcc allowed `-arch` list runs from sm_75 to sm_121 with `a`/`f` variants (e.g. sm_100a, sm_100f ... sm_121f) — [CUDA Binary Utilities docs](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) (via search summary)
- A PyPI helper `nvidia-arch` (8.0.0) lists 120f/121f family targets and generates `-gencode arch=compute_XXXf,code=sm_XXXf` flags (search-snippet only; the PyPI page failed to load) — [PyPI nvidia-arch](https://pypi.org/project/nvidia-arch/8.0.0/)

**PyTorch wheel arch lists:**
- 2.7.0 (Apr 2025, first Blackwell support, cu128): SM 50, 60, 70, 75, 80, 86, 89, 90, 100, 120. 2.8.0+ cu128: SM 70, 75, 80, 86, 89, 90, 100, 120 — [Qualiteg](https://journal.qualiteg.com/pytorch_and_supported_gpu_version/)
- 2.10 (Jan 2026) offered CUDA 12.6, 12.8 and 13.0 builds — [Qualiteg](https://journal.qualiteg.com/pytorch_and_supported_gpu_version/); [PyTorch dev-discuss 2.10 variants](https://dev-discuss.pytorch.org/t/wheel-variants-for-pytorch-release-2-10-published/3305)
- 2.13.0 measured `get_arch_list()`: cu126 = sm_50/60/70/75/80/86/90. cu130 = sm_75/80/86/90/100/120. cu132 = sm_75/80/86/90/100/120. The post says the 2.13 release blog calls CUDA 13.0 "the default" (2026-08-24) — [VRAM Lab](https://vramlab.com/posts/pytorch-2-13-rtx-5060-ti-cuda-wheels/)
- No sm_121 or sm_103/110 in stock PyTorch wheels per the community sm121 build README — [HF Qanatpharma](https://huggingface.co/Qanatpharma/pytorch-sm121-gb10/blob/main/README.md)
- No reliable source found for cu129 arch lists.

**Wheel variants / WheelNext / PEP 817 / PEP 825:**
- PEP 817 "Wheel Variants: Beyond Platform Tags" PR (by Jonathan Dekhtiar, NVIDIA) merged into python/peps on 2026-01-23 as Draft, Standards Track — [python/peps PR #4740](https://github.com/python/peps/pull/4740); [PEP 817 text](https://peps.python.org/pep-0817/); [discuss.python.org thread](https://discuss.python.org/t/pep-817-wheel-variants-beyond-platform-tags/105860)
- Sept 2026: PR by rgommers to switch PEP 817 to Informational, an umbrella for separate Standards Track PEPs (merge status not confirmed) — [python/peps PR #5149](https://github.com/python/peps/pull/5149). PEP 825 covers the package format — [pydevtools "What Are Wheel Variants?"](https://pydevtools.com/handbook/explanation/what-are-wheel-variants/) (Draft as of ~Apr 2026)
- wheelnext repo PR linking to the published PEP merged 2026-10-12 — [wheelnext/wheelnext PR #116](https://github.com/wheelnext/wheelnext/pull/116)
- NVIDIA variant provider plugin queries the driver and returns the max CUDA version. It can force variants like `cu129` — [wheelnext/nvidia-variant-provider](https://github.com/wheelnext/nvidia-variant-provider)
- PyTorch 2.10.0 wheel variants published 2026-02-05 for CUDA 12.6/12.8/13.0 (Linux x86, aarch64, Windows), ROCm 7.0/7.1, XPU and CPU. Install uses a variant-enabled uv from `wheelnext.astral.sh/v0.0.3`. Built by `release-wheel-variants.yml` in pytorch/test-infra. No per-variant arch lists are published — [PyTorch dev-discuss](https://dev-discuss.pytorch.org/t/wheel-variants-for-pytorch-release-2-10-published/3305). Earlier: 2.8 introduced variant wheels with experimental uv — [Quansight/Medium](https://medium.com/quansight/python-wheels-from-tags-to-variants-db3bd2aab938); [Astral blog](https://astral.sh/blog/wheel-variants). 2.9 extended to ROCm (experimental) — [AMD](https://www.amd.com/en/developer/resources/technical-articles/2025/pytorch-2-9-wheel-variant-support-expands-to-rocm.html)
- uv PR (Oct 2026, preview) vendoring an NVIDIA variant provider. A smoke test converting a torch wheel to the new format passed on a real GPU — [astral-sh/uv PR #22690](https://github.com/astral-sh/uv/pull/22690). Mainline uv docs still use per-CUDA index URLs (e.g. pytorch-cu130) — [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)

### Inferences
- Variant selection keys on driver/CUDA version, not on "does this wheel contain SASS or PTX for my exact cc". A variant can therefore select a cu130 wheel for a GTX 1080 (no sm_61 code) or for an sm_121 box (no sm_121 SASS). The variant metadata is a natural place for arch-coverage claims, but nothing verifies them. A verifier could slot into this ecosystem as a variant-provider companion or a CI check.
- Family targets add a third dimension (exact sm, family f, arch-specific a, plus PTX). Users and even maintainers misread `cuobjdump` output (FlashInfer #3294), which argues for a tool that encodes NVIDIA's compatibility rules: sm_X.Y SASS runs on same-major higher-minor? 120f covers 121; 100f does not cover 120; `a` is exact-only; PTX JITs forward.
- Each CUDA major transition (12 to 13) forces every extension maintainer to re-pick arch lists. This is recurring work, so a CI arch-coverage check has a durable reason to exist.

### Gaps
- Final status of PEP 817 / PEP 825 as of Nov 2026 is unconfirmed (whether the Informational switch merged, or whether either is Accepted).
- No official per-variant arch list for PyTorch cu128/cu129 found. The cu129 index contents are unverified.
- The exact rule for whether plain sm_120 SASS runs on sm_121 (same-major binary compat) was not verified from NVIDIA docs in this pass. Check the Programming Guide's binary compatibility section.

## Q3. TensorFlow 2.21.0 `[and-cuda]` RUNPATH bug (libcusolver.so.11 not found)

### Takeaway
Yes, this is reported upstream as **tensorflow/tensorflow #113541** ("TF 2.21.0 doesn't find GPU, but 2.20.0 does"), opened 2026-03-30. Root cause: change #102733 / commit 630956b removed `cuda_solver_context` from `all_runtime`, which dropped the `nvidia/cusolver/lib` RPATH entry from `libtensorflow_framework.so.2`. A community fix PR (#112973) went stale and was closed unmerged. Maintainers closed #113541 on 2026-10-09 after a fix landed on master (commit b7ba7a1) that adds a wheel verification check. The fix is slated for TF 2.22.0, with no 2.21.x patch mentioned. Workarounds: LD_LIBRARY_PATH, ctypes preload, or a symlink. 2.20.0 did not have this specific bug.

### Cited Findings
- Issue #113541 "TF 2.21.0 doesn't find GPU, but 2.20.0 does": opened 2026-03-30 by thephet on Arch Linux with pip `tensorflow[and-cuda]`, RTX 2070 Super and RTX 4090, reproduced on two machines and in tf-nightly. Log: "Cannot dlopen some GPU libraries" then "Skipping registering GPU devices...". Labels: 2.21.0, comp:gpu, type:bug. Closed as completed 2026-10-09 by dmiltr3. The maintainer reply (vruotolo, 2026-10-09) says the fix was committed to master (b7ba7a1), adds a **wheel verification check**, is in tf-nightly, and is slated for **TensorFlow 2.22.0**. Engagement: 7 thumbs-up on the OP and roughly 6 human comments. A birdnetR commit on 2026-11-20 references it — [tensorflow/tensorflow #113541](https://github.com/tensorflow/tensorflow/issues/113541)
- In that thread, piernov identified the missing `nvidia/cusolver/lib` in `libtensorflow_framework.so.2` RUNPATH and suggested diagnosing with `LD_DEBUG=libs` and `TF_CPP_MAX_VLOG_LEVEL=3`. Workarounds posted: ctypes preload of `libcusolver.so.11` (amylizzle), symlinking cusolver into `nvidia/cublas/lib` (guptadivyansh; must be redone after reinstall), and LD_LIBRARY_PATH (some dispute over a WSL2 variant) — [#113541](https://github.com/tensorflow/tensorflow/issues/113541)
- PR #112973 "[XLA:GPU] Restore cusolver rpath in all_runtime" by vladfi1, opened 2026-03-22. It explains that #102733 removed `cuda_solver_context` from `all_runtime`, "inadvertently" dropping the cusolver rpath from `libtensorflow_framework.so.2`, while TF linalg ops (SVD, eig) still use cusolver. It was approved on 2026-08-06, then got a changes request on 2026-08-30 because the file is mirrored from openxla/xla and must be submitted upstream. Marked stale 2026-09-15 and closed unmerged 2026-09-29. The author's workaround: `export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cusolver/lib:$LD_LIBRARY_PATH"` — [tensorflow/tensorflow PR #112973](https://github.com/tensorflow/tensorflow/pull/112973)
- Downstream independent reproduction (2026-09-16): TF 2.21.0, Keras 3.15.1, GTX 1060, driver 580.173.02. Bisecting LD_LIBRARY_PATH entries showed only `nvidia/cusolver/lib` matters. The cusolver entries are in `libtensorflow_cc.so.2` RUNPATH but not `libtensorflow_framework.so.2`. The project shipped a permanent ctypes `RTLD_GLOBAL` preload shim in v0.1.2 — [GretaLupi/hamlet-toolkit #1](https://github.com/GretaLupi/hamlet-toolkit/issues/1)
- Related TF 2.20/2.21 loader issue: the runtime is loaded with RTLD_GLOBAL, which exposes bundled BoringSSL/curl process-wide. 2.22.0rc0 is RTLD_LOCAL again. Note that RTLD_LOCAL in 2.22 could change how the ctypes-preload workaround behaves — [tensorflow/tensorflow #129050](https://github.com/tensorflow/tensorflow/issues/129050)
- Historical (older, pre-2.16): "Could not load dynamic library libcusolver.so.11" / "Cannot dlopen some GPU libraries" was a perennial TF symptom in the CUDA 10/11 era, typically from system CUDA/LD_LIBRARY_PATH mismatches — [tensorflow #50842](https://github.com/tensorflow/tensorflow/issues/50842); [tensorflow #44878](https://github.com/tensorflow/tensorflow/issues/44878); [NVIDIA/tensorflow #110](https://github.com/NVIDIA/tensorflow/issues/110) (2024; dlerror pointed at a missing libnvJitLink.so.12, a different dependency); [NVIDIA forums](https://forums.developer.nvidia.com/t/tensorflow-gpu-not-working/184575)

### Inferences
- This bug class (a RUNPATH entry silently dropped by a build-graph refactor) went unnoticed through a full TF release. Users then waited about 6.5 months (Mar 30 to Oct 9) for an upstream fix that ships only in the next minor. The maintainers' own fix is a "wheel verification check", direct evidence that maintainers see value in automated post-build checks on wheel RUNPATH/dlopen closure.
- TF fails silently: it falls back to CPU with a single warning. A user-side tool that resolves every `dlopen` target of installed GPU packages against their RUNPATH and the pip `nvidia-*` layout would have pinpointed this in seconds, versus the manual LD_DEBUG and bisection users did.
- The 2.20 to 2.21 regression and the 2.21 to 2.22 RTLD_GLOBAL to RTLD_LOCAL change show that the loader surface changes from release to release, so regression checks need to be continuous.

### Gaps
- Could not confirm whether TF 2.16 to 2.19 had analogous RUNPATH omissions. The 2.21 thread says 2.20.0 works. No source found on 2.16 to 2.19 pip `and-cuda` RUNPATH problems specifically (anecdotally, early 2.16 had pip CUDA lib discovery problems, but this was not verified in this pass).
- Did not verify whether the b7ba7a1 fix restores the RPATH, adds a cusolver dependency, or only adds the check. Also did not verify whether 2.22.0 final has shipped as of 2026-11-21.

## Q4. Existing diagnostic tools and what they lack

### Takeaway
No end-to-end tool was found that scans all installed Python packages' CUDA fatbins against the local GPU (SASS vs PTX-JIT vs fail) and also checks dlopen/RUNPATH closure. Nor was a CI action found that asserts wheel arch coverage. Today's tooling is fragmented and per-framework: torch's `get_arch_list()` and collect_env, `python -m xformers.info`, `python -m bitsandbytes`, raw `cuobjdump --list-elf/-lptx`, and LD_DEBUG. It requires expert knowledge of NVIDIA's compatibility rules (f/a targets, PTX forward-compat, driver-JIT limits).

### Cited Findings
- `torch.cuda.get_device_capability()` and `torch.cuda.get_arch_list()` report torch's own compiled arches only, not those of other extensions — [quantacloud guide](https://quantacloud.net/guides/check-gpu-driver-cuda-version)
- Per-package checks recommended by fix guides: `python -m xformers.info`, `python -m bitsandbytes`, `cuobjdump -lelf` on a compiled `.so` — [localaimaster sm_120 fix](https://localaimaster.com/blog/no-kernel-image-available-rtx-50)
- `cuobjdump` extracts ELF, SASS and PTX from host binaries and cubins. `--list-elf` lists embedded SM targets; `--dump-ptx`/`--dump-elf` dump the code — [NVIDIA CUDA Binary Utilities 13.x](https://docs.nvidia.com/cuda/cuda-binary-utilities/); [Blackwell GPU Wiki: CUDA pipeline](https://0xsero.github.io/blackwell-gpu-wiki/fundamentals/cuda-pipeline/)
- cuobjdump and nvdisasm ship as pip wheels (`nvidia-cuda-cuobjdump`, `nvidia-cuda-nvdisasm`), usable in CI without a system CUDA install — (search summary of [The Software Frontier, "How CUDA Binaries Actually Work"](https://www.thesoftwarefrontier.com/p/how-cuda-binaries-actually-work); not independently fetched)
- A Go package wraps the installed cuobjdump to inspect device code without a GPU — [pkg.go.dev tmc/lib/nvidia/x/cudaobjdump](https://pkg.go.dev/github.com/tmc/lib/nvidia/x/cudaobjdump); a web UI for cuobjdump/nvdisasm on HF Spaces — [ejschwartz/cuobjdump](https://huggingface.co/spaces/ejschwartz/cuobjdump)
- Third-party SASS tools lag new toolkits (e.g. a SASS assembler for sm_50 to sm_80 that fails on CUDA 13.4 cubins due to an ABI check) — [damien1141/secret_sauce](https://github.com/damien1141/secret_sauce)
- Searches for "scan python wheels cuobjdump arch coverage", "auditwheel CUDA arch", "GitHub Action CUDA arch coverage" and "CLI check installed packages CUDA arch compatible with my GPU" returned no dedicated tool. Results were only docs, how-to blogs and one-off scripts — searches via WebSearch 2026-11-21 (no single URL; see the sources above)
- The wheelnext NVIDIA variant provider detects driver/CUDA version for install-time selection, but does not inspect binaries — [wheelnext/nvidia-variant-provider](https://github.com/wheelnext/nvidia-variant-provider)
- `nvidia-arch` (PyPI 8.0.0) generates gencode flags and family-target lists. It is a build-side helper, not a scanner (snippet only) — [PyPI](https://pypi.org/project/nvidia-arch/8.0.0/)
- Users debugging RUNPATH issues resort to `LD_DEBUG=libs`, bisecting LD_LIBRARY_PATH, and `readelf`-style RUNPATH inspection — [tensorflow #113541](https://github.com/tensorflow/tensorflow/issues/113541); [hamlet-toolkit #1](https://github.com/GretaLupi/hamlet-toolkit/issues/1)
- Hand-rolled audits are error-prone: the FlashInfer #3294 reporter's filename-grep audit wrongly concluded that jit-cache lacked sm_120 kernels — [flashinfer #3294](https://github.com/flashinfer-ai/flashinfer/issues/3294)

**What existing tools lack (synthesis of the above):**
1. Whole-environment scope: no tool walks site-packages and inspects every `.so` (torch, xformers, flash-attn, bitsandbytes, onnxruntime, cupy, jaxlib, TF, custom ops).
2. Compatibility-rule engine: none maps the fatbin contents (sm_XY, sm_XYa, sm_XYf, PTX version) plus the local cc plus driver version to a verdict (native SASS / PTX JIT with driver-version caveat / will fail).
3. Loader closure: none checks that every DT_NEEDED and known dlopen target (libcusolver, libcudnn, libnvJitLink, etc.) resolves via RUNPATH/$ORIGIN to the pip `nvidia-*` wheels.
4. CI form factor: no GitHub Action or `auditwheel`-style "assert arch coverage = {75..120 + PTX}" gate.
5. Public database: no per-package, per-version arch-coverage table. Users rely on blogs and forum posts.

### Inferences
- The building blocks are mature and pip-installable (cuobjdump wheels, pyelftools for RUNPATH, fatbin header parsing). The missing piece is integration plus encoding NVIDIA's rules, which makes this a tractable OSS project.
- A pure-Python fatbin parser, with no cuobjdump dependency, would allow scanning on machines without a CUDA toolkit, and on Windows .pyd/.dll files, which none of the found tools address.

### Gaps
- Did not specifically verify `torch.utils.collect_env` output fields, `pip-cuda-check`, conda's `__cuda` virtual package, or NVIDIA cuda-python utilities in this pass. No sources were fetched for these. (General knowledge, unverified here: conda's `__cuda` exposes driver CUDA version, not GPU cc or binary contents. collect_env reports torch/CUDA/cuDNN versions and the GPU name, not extension arch coverage.)
- No evidence found for an existing commercial service offering a compatibility database.

## Q5. Evidence that maintainers want automated arch-coverage / packaging checks in CI

### Takeaway
There is moderate, concrete evidence. Releases or wheel entries shipped without sm_120 (SageAttention cu130/Torch 2.10 in Comfy-Org/wheels). Build-gating logic needed fixes (ONNX Runtime PR #33373, FlashInfer adding 12.1a). TensorFlow's fix for the 2.21 RUNPATH regression was explicitly a "wheel verification check". At least one project's packaging example already runs a checker that flags missing sm_80/86/120. Demand exists but is mostly met with ad-hoc, per-project scripts.

### Cited Findings
- TensorFlow's fix for #113541 added a wheel verification check on master (b7ba7a1), slated for 2.22.0 — [tensorflow #113541](https://github.com/tensorflow/tensorflow/issues/113541)
- SageAttention cu130/Torch 2.10 wheel shipped missing sm_120 because one matrix entry did not override the arch list — [Comfy-Org/wheels #22](https://github.com/Comfy-Org/wheels/issues/22)
- FlashInfer: wheel build arch targets lacked 12.0a/12.1a; fix PR #3907 added 12.1a to the aarch64 jit-cache targets — [flashinfer #3170](https://github.com/flashinfer-ai/flashinfer/issues/3170). User-filed wheel-content audit #3294 (May 2026) — [flashinfer #3294](https://github.com/flashinfer-ai/flashinfer/issues/3294)
- ONNX Runtime needed a PR to fix CUDA architecture gating for SM12x family/arch-specific targets — [onnxruntime PR #33373](https://github.com/microsoft/onnxruntime/pull/33373)
- A project issue: "The CUDA release packaging example builds only sm_121" and fails a checker for missing sm_80, sm_86 and sm_120. This shows checkers of this kind are starting to appear in individual projects — [ashhart/TensorFold #3763](https://github.com/ashhart/TensorFold/issues/3763)
- Downstream apps add manual sm_120 arch guards and documentation — [Positronic PR #1012](https://github.com/Positronic-Robotics/positronic/pull/1012); [VoiceStudio PR #2084](https://github.com/debpalash/VoiceStudio/pull/2084)
- PyTorch's own variant pipeline uses `validate-binaries.yml` with forced `NV_VARIANT_PROVIDER_FORCE_*` env vars (validation exists in-house at PyTorch scale) — [PyTorch dev-discuss](https://dev-discuss.pytorch.org/t/wheel-variants-for-pytorch-release-2-10-published/3305)
- The TF RUNPATH fix PR stalled because the file lives in openxla/xla (mirroring friction), so regressions can persist for months — [PR #112973](https://github.com/tensorflow/tensorflow/pull/112973)

### Inferences
- Opportunity sizing (qualitative): the strongest unmet need is a reusable `check-cuda-wheel` CI action/CLI for the long tail of extension maintainers (flash-attn, SageAttention, spconv, xformers, FlashInfer, bitsandbytes, custom-op authors, community wheel builders like Comfy-Org/wheels). These maintainers ship many matrix entries (torch × CUDA × python × OS) and regress arch lists by accident. PyTorch, TF and ONNX Runtime have in-house validation, but even they regress (TF 2.21).
- The user-side scanner ("doctor" CLI) addresses the largest audience (RTX 50 / DGX Spark / Pascal hobbyists in ComfyUI, LLM and vision stacks), where the symptom shows up far from the cause.
- A public compatibility database is the hardest to sustain. It could be generated automatically by running the same scanner over PyPI/HF wheel indexes, and could feed wheel-variant metadata.
- Competitive risk: NVIDIA/WheelNext could add arch-coverage metadata or validation to the variant tooling, and PyTorch/TF could expand in-house checks. No evidence was found that either is planned as of Nov 2026.

### Gaps
- No maintainer survey or explicit "we want an arch-coverage GitHub Action" feature request was found in a major repo.
- No download or usage metrics to quantify the addressable user base (e.g. PyPI downloads of cu126 vs cu13x wheels on Blackwell machines).
