# Build the verifier consumer Blackwell still lacks

**Recommendation: build a whole-environment CUDA compatibility verifier.** It should ship as a "doctor" CLI plus a CI gate for wheel builders. It reads the GPU code embedded in every installed Python package and checks each package's shared-library search paths. Then it gives one verdict per package for the GPU and driver actually present: runs natively, runs only through PTX JIT, or will fail and why. Of all the candidates, this is the only one where the research found **no existing tool**. The failure it diagnoses is the most common one in the GPU Python ecosystem: GitHub has **3,501 "no kernel image is available" issues, 1,499 of them opened since January 2025**. Spend one to two weeks first on a quick win: the first public block-scaled (NVFP4/MXFP8) GEMM roofline for the RTX 5090 Laptop, which answers questions nobody has measured yet. The stretch goal is a low-precision conformance suite for sm_120 that catches silent fallbacks and wrong results across cuBLASLt, CUTLASS, Triton, FlashInfer and llama.cpp, used to land fixes upstream. Two directions are deliberately not recommended as the main project. New standalone FP4 kernels have real but modest demand (tens of reactions per issue), and NVIDIA-adjacent teams are closing those gaps faster than one person could. TensorFlow on sm_120 is already solved on this exact machine: stock TF 2.21 passed every eager, cuDNN, XLA and Keras test once `LD_LIBRARY_PATH` was fixed. All three recommendations share a property the deleted repo lacked: every claim is checked against real hardware. That matters after a project that failed on fabricated patches.

## The kernel gaps are real, but well-staffed teams are closing them

The research shows that plain sm_120 support is no longer the problem. PyTorch has included sm_120 in its cu128, cu130 and cu132 wheels since 2.7 ([VRAM Lab](https://vramlab.com/posts/pytorch-2-13-rtx-5060-ti-cuda-wheels/)). On this laptop, stock TF 2.21 reaches about 73.6 TFLOPS bf16 through cuBLAS. TensorFlow's own roadmap now says it will "exclusively focus on" security fixes, bug fixes and dependency updates ([Google Developers Blog](https://developers.googleblog.com/whats-new-in-tensorflow-221/)). The remaining problems fall into two groups. The first is missing or fragile low-precision kernels, because sm_120 is an architecture of its own. The second is packaging and loader failures, which surface far from their cause.

**The kernel gap is structural.** Consumer Blackwell uses the warp-level `mma.sync ... block_scale` instruction with accumulators in registers. It has no `tcgen05`, no tensor memory (TMEM) and only about 99 KB of shared memory per SM, compared with 228 KB on sm_100. So datacenter Blackwell kernels have to be rewritten, not just recompiled ([Colfax Research](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/); [lna-lab patch set](https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm)). The symptoms show up across the stack:

- In vLLM, native NVFP4 MoE on SM120 runs only if you opt in with `--moe-backend flashinfer_b12x`. Otherwise it silently falls back to Marlin. The issue tracking this (#31085) has **12 reactions**, no maintainer reply, and was flagged stale in October 2026 ([vLLM #31085](https://github.com/vllm-project/vllm/issues/31085)).
- FlashAttention has no official sm_120 path for head dimensions above 128. Gemma 4 needs D=512 ([FA #2427](https://github.com/Dao-AILab/flash-attention/issues/2427)).
- The consolidating sm_120 pull request (#2634) has been open since June with **no visible maintainer review** ([FA PR #2634](https://github.com/Dao-AILab/flash-attention/pull/2634)).
- DeepGEMM's maintainer said plainly: "as we don't have SM120 devices, we don't have plans or people to support this" ([DeepGEMM #236](https://github.com/deepseek-ai/DeepGEMM/issues/236)).

Demand here is real but small. The loudest open sm_120 kernel ask is FlashAttention #1987, with **30 reactions** ([FA #1987](https://github.com/Dao-AILab/flash-attention/issues/1987)). Supply is also arriving fast, which matters more for a solo developer:

- **b12x**, a CuTe-DSL kernel library for sm_120/121, became the de facto fix path in 2026. Its vLLM integration PR drew **76 reactions** before being split into smaller merged PRs ([vLLM #51696](https://github.com/vllm-project/vllm/pull/51696); [b12x](https://github.com/local-inference-lab/b12x)).
- A DeepGEMM port on NVIDIA's branch already carries **741 SM120 tests** ([DeepGEMM #236](https://github.com/deepseek-ai/DeepGEMM/issues/236)).
- llama.cpp merged native Blackwell NVFP4/MXFP4 MMQ kernels in April 2026 ([llama.cpp #22196](https://github.com/ggml-org/llama.cpp/pull/22196)).

A solo developer who builds a fresh FP4 MoE GEMM in Q1 2027 would be competing with these teams, and would probably arrive after the gap had closed.

**The packaging gap has no one on it.** The research found **3,501 GitHub issues** containing "no kernel image is available", **1,499 created since 2025-01-01** ([GitHub search API](https://api.github.com/search/issues?q=%22no+kernel+image+is+available%22+is:issue+created:%3E=2025-01-01)). It also found 164 sm_120 threads in PyTorch alone and 560 in vLLM, plus 378 for sm_121 ([GitHub search API](https://api.github.com/search/issues?q=repo:pytorch/pytorch+%22no+kernel+image+is+available%22)). The trouble has moved away from PyTorch core. Most failures now come from third-party extensions such as flash-attn, xformers, SageAttention, FlashInfer, spconv and onnxruntime-gpu, whose architecture lists are set independently of PyTorch ([onnxruntime #26181](https://github.com/microsoft/onnxruntime/issues/26181); [Comfy-Org/wheels #22](https://github.com/Comfy-Org/wheels/issues/22)). Two shifts keep this failure class growing:

- CUDA 13 removed support for Maxwell, Pascal and Volta, so older GPUs now fail in the opposite direction ([CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.3/pdf/CUDA_Toolkit_Release_Notes.pdf)).
- The stock PyTorch wheels stop at sm_120, so DGX Spark owners on sm_121 hit the same wall ([HF sm121 build README](https://huggingface.co/Qanatpharma/pytorch-sm121-gb10/blob/main/README.md)).

Searches for an arch-coverage scanner, an `auditwheel`-style CUDA check or a GitHub Action found only how-to blogs and one-off scripts. The tools that do exist each cover one framework: `torch.cuda.get_arch_list()` reports only torch's own architectures, `python -m xformers.info` covers only xformers, and raw `cuobjdump` leaves the interpretation to you ([localaimaster guide](https://localaimaster.com/blog/no-kernel-image-available-rtx-50); [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)).

## Ten candidates ranked: verification beats new kernels for one person

The scores combine the evidence in the notes with judgment about this specific developer and machine. The release-time column is an estimate for one person working roughly full-time.

| Rank | Candidate | Impact | Unmet / urgent | Feasible on 5090 Laptop | First useful release | Maintenance | Obsolescence risk | Evidence strength |
|---|---|---|---|---|---|---|---|---|
| **1 (primary)** | Whole-environment CUDA arch + loader verifier (CLI + CI gate) | High: every GPU Python user, all vendors' stacks | High: no tool exists; failures recur with each CUDA and architecture transition | Very high: mostly Python; GPU used for ground truth | 3–4 weeks | Low–medium: rule table changes per CUDA release | Low–medium: WheelNext could add arch metadata, but nothing verifies it | Moderate: API issue counts, several concrete regressions, no explicit feature request |
| **2 (quick win)** | Public RTX 5090 Laptop block-scaled GEMM roofline | Medium: credibility, answers open questions | High: no rigorous public 5090 NVFP4 roofline found | Very high: this is the hardware | 1–2 weeks | None (a dated report) | Low | Moderate: absence of data is itself the evidence |
| **3 (stretch)** | sm_120 low-precision conformance suite feeding upstream fixes | High per user, narrower audience | High: wrong-output and silent-fallback bugs across 5+ stacks | High; model-level tests limited to ≤24 GB | 6–8 weeks | Medium: tracks backend APIs | Low for the suite, high for any single bug | Mixed: several key bugs are snippet-only |
| 4 | Fused BF16→NVFP4 activation quantizer for llama.cpp W4A4 | Low–medium (~8% of op time) | Medium: explicitly deferred | Very high | 2–3 weeks | Low (upstream owns it) | Medium | Strong (fetched PR) |
| 5 | vLLM NVFP4 KV-cache wiring on SM120 | Medium: long context in 24 GB | Medium: 15 reactions, a PR may exist | Medium | 4–6 weeks | Upstream | High | Moderate |
| 6 | vLLM/FlashInfer NVFP4 MoE auto-dispatch and correctness | Medium | Medium: b12x and PR #47577 already in flight | Medium: 24 GB caps model choice | 4–8 weeks | Upstream | High | Moderate (fetched #31085) |
| 7 | FlashAttention sm_120 D=256/512 and fp8 paths | High | High (30 + 12 reactions) | Low: D=512 against 99 KB SMEM is hard | Months | High | Medium | Strong |
| 8 | Prebuilt Windows wheels (SageAttention/FA) | Medium | Low: crowded | High | Days | Very high (per torch release) | Very high | Strong, but crowded |
| 9 | Sparse MLA / DeepGEMM sm_120 ports | High for 96 GB users | Medium | Very low: target models don't fit 24 GB | Months | High | High (NVIDIA branch active) | Strong |
| 10 | Anything TensorFlow-on-sm_120 | Very low | None | — | — | — | Already solved | Strong (verified locally) |

**Why the verifier comes first.** It ranks first because it does well on every axis at once.

- **It is unmet.** No end-to-end tool was found.
- **It is urgent and recurring.** Every CUDA major release forces every extension maintainer to choose architecture lists again. The ONNX Runtime and FlashInfer build systems both needed fixes for sm_12x family and arch-specific targets ([onnxruntime PR #33373](https://github.com/microsoft/onnxruntime/pull/33373); [flashinfer #3170](https://github.com/flashinfer-ai/flashinfer/issues/3170)).
- **Experts get it wrong by hand.** A FlashInfer user audited the wheel manually with filename greps and `cuobjdump`, wrongly concluded that sm_120 kernels were missing, and later partly retracted ([flashinfer #3294](https://github.com/flashinfer-ai/flashinfer/issues/3294)).
- **Real releases have shipped without sm_120.** One SageAttention cu130/Torch 2.10 matrix entry shipped with arch list "8.0 9.0" because it did not override the global list ([Comfy-Org/wheels #22](https://github.com/Comfy-Org/wheels/issues/22)).
- **Maintainers already reach for this fix.** TensorFlow's RUNPATH regression, the one hit on this laptop, took **six and a half months** to fix (opened 2026-03-30, closed 2026-10-09). The fix maintainers chose was a **"wheel verification check"**, which shows they value exactly this kind of post-build gate ([tensorflow #113541](https://github.com/tensorflow/tensorflow/issues/113541)).

**Why kernels rank lower despite stronger engineering appeal.** Kernel work has a different risk profile. Items 5–7 and 9 depend on maintainer bandwidth that the notes show is scarce: FA #2634 has no review, SGLang's SM120 plan was bot-closed with items unchecked ([SGLang #19637](https://github.com/sgl-project/sglang/issues/19637)), and vLLM #31085 was stale-flagged. They also face fast-moving, better-resourced competition. Pure wheel repackaging (item 8) is the opposite trap. It has proven demand, with mjun0812's FA wheels at **1,801 stars** and woct0rdho's SageAttention fork at **1,068** ([mjun0812](https://github.com/mjun0812/flash-attention-prebuild-wheels); [woct0rdho/SageAttention](https://github.com/woct0rdho/SageAttention)). But the field is crowded, and each wheel goes stale with the next torch release.

**Where the evidence is weak.** The issue counts are mention-based upper bounds that include PRs and duplicates. No maintainer was found explicitly asking for an arch-coverage GitHub Action. No Reddit data could be retrieved. Several of the correctness bugs that motivate the stretch goal are known only from search snippets: vLLM #51525, TRT-LLM #14154, CUTLASS #3096 and Triton #7550. Treat them as leads to reproduce, not as facts.

The strongest counter-argument to the verifier is that it is "just tooling". Kernels are where a CUDA developer's skills are rarest. The quick win and the stretch are there to keep that skill central: they use the same verification approach but point it at the tensor cores.

## Primary pick: a fatbin-and-loader verifier with hardware ground truth

### Architecture: five layers, each testable on its own

The working name is `cuda-doctor`; check that the name is free on PyPI before using it. The core is pure Python and does not need a GPU, so it runs in CI. An optional probe mode uses the real GPU to confirm its predictions. There are five layers.

**1. Discovery.** Walk every `site-packages` on `sys.path`, and also scan conda environments and `.whl` archives given on the command line. Collect ELF `.so` files on Linux/WSL2 and PE `.pyd`/`.dll` files on Windows. Map each file to its distribution through `importlib.metadata` RECORD files, so a verdict reads "flash-attn 2.8.3" rather than a bare path. Standalone `.cubin` and `.ptx` files go into the same index. Packages such as `flashinfer-cubin` ship thousands of these, and they are exactly where the FlashInfer #3294 reporter got confused.

**2. Extraction.** Read the embedded fatbin sections. On ELF that is `.nv_fatbin`, read with pyelftools. On Windows it is the PE equivalent, read with pefile. Parse the community-documented fatbin container to list each image's kind (SASS cubin or PTX), target (`sm_XY`, `sm_XYa`, `sm_XYf`), PTX ISA version and compression flag. The container format is undocumented and compressed entries are tricky. So every parser result must be checked by differential testing against `cuobjdump --list-elf --list-ptx`, and the tool should fall back to `cuobjdump` (also distributed as a pip wheel, per search summaries) whenever its own parser is unsure.

**3. Rule engine.** Map (images present, device compute capability, driver version) to one of three verdicts:

- **NATIVE**: compatible SASS is present.
- **JIT**: only PTX can serve this GPU. Warn about first-run compile cost, and check that the driver can actually compile that PTX ISA version.
- **FAIL**: nothing can serve this GPU.

Encode the rules from NVIDIA's documentation:

- `a` targets match one exact compute capability.
- `compute_120f` covers 12.0 and 12.1, but `100f` code never runs on 12.x ([CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html); [Arnon Shimoni](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)).
- PTX JIT-compiles forward to newer GPUs.

The notes flag one rule as unverified: whether plain `sm_120` SASS loads on sm_121. Encode it from NVIDIA's binary-compatibility section, and label any sm_121 verdict "rule-derived, not hardware-verified" until someone with a DGX Spark confirms it. Add a hint layer for the most common real-world cause, a torch from the cu126 index on an sm_120 GPU. Measured arch lists show cu126 stops at sm_90 ([VRAM Lab](https://vramlab.com/posts/pytorch-2-13-rtx-5060-ti-cuda-wheels/)).

**4. Loader closure.** For each ELF, resolve `DT_NEEDED` entries against `RPATH`/`RUNPATH` (with `$ORIGIN` expanded), then `LD_LIBRARY_PATH`, the ld.so cache and the default directories. On WSL2, also include `/usr/lib/wsl/lib`, where the driver's `libcuda` lives. The key technical addition is finding dlopen targets. TensorFlow loads `libcusolver.so.11` with `dlopen`, not through `DT_NEEDED`. For `dlopen`, glibc consults the **calling object's** RUNPATH, so the bug comes down to a single missing entry in `libtensorflow_framework.so.2` ([PR #112973](https://github.com/tensorflow/tensorflow/pull/112973); [hamlet-toolkit #1](https://github.com/GretaLupi/hamlet-toolkit/issues/1)). Extract `lib*.so.N` string literals from each library's read-only data as candidate dlopen targets. Resolve each one against that library's own RUNPATH and against the pip `nvidia-*` package layout. Report candidates that do not resolve, along with the exact `LD_LIBRARY_PATH` or preload fix. Windows DLL search-order checking is deferred past the MVP.

**5. Probe mode** (`--probe`, the hardware ground truth). In a subprocess, load each extracted cubin and PTX image with `cuModuleLoadData` through cuda-python. Record whether it loads or returns `CUDA_ERROR_NO_BINARY_FOR_GPU`, and compare with the rule engine's prediction. Any mismatch is a bug in the rules, not something to explain away. Probe mode is what keeps the project honest, and it needs a real sm_120 GPU like this one.

**Outputs.** A human-readable table, JSON, and CI-friendly exit codes. A `scan-wheel dist/*.whl --require 80,86,89,90,100,120 --require-ptx` mode can fail a build before release. That single check would have caught Comfy-Org/wheels #22.

### MVP (v0.1) scope and milestones

v0.1 has a deliberately narrow promise. It gives a correct NATIVE/JIT/FAIL verdict for every embedded or standalone CUDA image in an environment, on Linux, WSL2 and Windows, and complete loader closure on Linux/WSL2. Runtime-JIT systems are not judged. Triton, NVRTC users and `flashinfer-jit-cache` builds get a "runtime-compiled; depends on toolkit at run time" label rather than a guessed verdict.

| Week | Milestone | Exit criterion |
|---|---|---|
| 1 | Fixture corpus and fatbin parser | Tiny kernels built with local nvcc 12.9 in every gencode mix: sm_86 only; compute_80 PTX only; sm_120; sm_120a; sm_120f; compute_120 PTX; sm_100a. Parser matches `cuobjdump` on all of them. |
| 2 | Rule engine and probe mode | On this 5090 Laptop, `--probe` agrees with the prediction for 100% of fixtures. The sm_100a fixture must FAIL; sm_120a/sm_120f/compute_120 must load. |
| 3 | Environment scan and loader closure on WSL2 | Reproduces the TF 2.21 cusolver bug (see validation) and agrees with `cuobjdump` across a real environment holding torch, TF, xformers, bitsandbytes, flash-attn and onnxruntime-gpu. |
| 4 | Windows PE support; **v0.1 on PyPI** | Scans a Windows ComfyUI-style environment, including the large torch CUDA DLLs, in under about 30 s. JSON schema is frozen. |
| 5–6 | `scan-wheel` and a GitHub Action | First external PR adding the check to a wheel-builder's CI. |
| 7–10 | Public coverage database | Nightly job scans current wheels of the top CUDA packages on PyPI and publishes a static per-package, per-version table. |

### Validating correctness on this exact machine

This laptop is an unusually good test bench. It has a real sm_120 GPU, both Windows and WSL2 driver paths, and a known, documented loader bug already reproduced on it. Validation has four tiers.

1. **Synthetic fixtures with probe ground truth.** Required: zero disagreements between prediction and `cuModuleLoadData`.
2. **Differential testing against `cuobjdump`** on every binary in several real environments. Required: zero parser disagreements; any unsure parse falls back to `cuobjdump`.
3. **Known-regression replays.**
   - Run TF 2.21 in WSL2 with `LD_LIBRARY_PATH` unset. The doctor must name `libcusolver.so.11` as unresolved from `libtensorflow_framework.so.2` and print the fix. With the path set it must pass. Cross-check against an `LD_DEBUG=libs` trace, the method users used by hand in #113541.
   - Install a cu126 torch. The doctor must report FAIL for sm_120, and the actual CUDA call must fail the same way.
   - Fetch `flashinfer-cubin` 0.6.11. The doctor must correctly separate closed-source sm_100-only cubins from the 120f JIT-cache path. That is the distinction the #3294 reporter missed ([flashinfer #3294](https://github.com/flashinfer-ai/flashinfer/issues/3294)).
   - If the SageAttention entry from Comfy-Org/wheels #22 can still be downloaded, it must FAIL on sm_120.
4. **Performance.** Scan time and memory on a large Windows environment, with results cached by file hash so repeat runs take seconds.

### Distribution and upstream path

Ship through four channels:

- **PyPI**, as a pure-Python wheel with a `[probe]` extra for cuda-python, so `pipx run cuda-doctor` works in one line.
- **conda-forge**.
- **A GitHub Marketplace action** for wheel builders.
- **The coverage database**, hosted on GitHub Pages and rebuilt by a scheduled workflow.

Upstream adoption is the real goal. The first targets are the wheel builders whose regressions are on record: Comfy-Org/wheels, mjun0812's flash-attention wheels and woct0rdho's SageAttention fork. FlashInfer is a natural second, given #3170 and #3294.

The longer-term home is WheelNext. Wheel variants pick a build by driver and CUDA version, not by verified per-package arch coverage, and uv is now vendoring an NVIDIA variant provider ([wheelnext/nvidia-variant-provider](https://github.com/wheelnext/nvidia-variant-provider); [uv PR #22690](https://github.com/astral-sh/uv/pull/22690)). The verifier is the natural companion check for whatever arch claims variant metadata ends up making. That is also the main obsolescence risk, and the best defence is to become the tool those systems call.

There is a precedent for this kind of absorption. triton-windows, another single-maintainer gap-filler, reached about 1.9k stars and was then adopted into the official `triton-lang` organisation ([woct0rdho/triton-windows](https://github.com/woct0rdho/triton-windows)).

## Quick win: publish the first RTX 5090 Laptop block-scaled roofline

Before starting the verifier, spend one to two weeks producing a dated, reproducible report on block-scaled GEMM throughput, power and correctness on this machine. Two things make it worth doing. First, no rigorous public NVFP4/MXFP4 roofline exists for any RTX 5090, desktop or laptop. The best available numbers come from siblings: an unoptimized CUTLASS NVFP4 GEMM reaches about **60% of FP4 peak** on the RTX PRO 6000 ([Colfax Research](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/)), and cuBLASLt MXFP8 reaches about **372 TFLOPS on an RTX 5080** ([pytorch #198126](https://github.com/pytorch/pytorch/issues/198126)). Second, it builds the measurement harness the stretch goal reuses.

**Set expectations from first principles.** The whitepaper's desktop RTX 5090 dense peaks are 1,676 TFLOPS FP4, 419 FP8 with FP32 accumulate, and 209.5 BF16 with FP32 accumulate, across 170 SMs at 2,407 MHz ([NVIDIA RTX Blackwell whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)). The laptop part has 10,496 CUDA cores against 21,760 on the desktop, which is 82 SMs, or about 48% ([StorageReview](https://www.storagereview.com/review/nvidia-geforce-rtx-5090-mobile-review-ai-power-meets-efficiency)). Scaling per SM gives about **101 TFLOPS BF16 and about 808 TFLOPS dense FP4 at desktop clocks**. The 73.6 TFLOPS bf16 measured on this laptop implies an effective clock of roughly 1.75 GHz if cuBLAS were near peak. That projects to a practical FP4 ceiling of roughly **350–500 TFLOPS** at 60–80% efficiency. All of these are inferences, and producing real numbers is the point of the exercise.

**The roofline answers three specific open questions.**

1. **How close a tuned NVFP4 GEMM gets to the laptop's ceiling under a 150–175 W power cap.**
2. **Whether block-scaled MXFP8 avoids GeForce's half-rate FP32-accumulate penalty.** The 5080's 372 TFLOPS beats its 225 TFLOPS FP32-accumulate FP8 peak, which suggests it does. On this laptop, MXFP8 running clearly above about 150 TFLOPS would confirm that, a result useful to anyone choosing between per-tensor FP8 and block-scaled FP8 on consumer cards.
3. **Joules per FLOP against BF16.** The desktop-card claim of 41% energy savings comes from a measurement on desktop RTX 50 GPUs ([arXiv 2601.09527](https://arxiv.org/html/2601.09527v1)), and on a laptop energy matters even more.

**How to run it.**

- **Backends:**
  - `torch._scaled_mm` (cuBLASLt) for MXFP8, TN layout. NN and NT fail on cuBLAS versions before 13.6; record that as a finding ([pytorch #198126](https://github.com/pytorch/pytorch/issues/198126)).
  - CUTLASS examples 79a (NVFP4) and 79c (MXFP8/MXFP6), built in WSL2 with `-gencode arch=compute_120a,code=sm_120a`. Plain `-arch=sm_120a` silently fails to target the block-scaled instructions ([FlashInfer #2723](https://github.com/flashinfer-ai/flashinfer/issues/2723)).
  - Triton `tl.dot_scaled`.
  - A BF16 cuBLAS baseline.
- **Shapes:** sweep M, N and K from 512 to 16,384, plus M=1–64 to model decode.
- **Telemetry:** log SM clock, power draw and the OEM power mode for every run, on AC power. Laptop results vary with TGP and thermals ([Notebookcheck](https://www.notebookcheck.net/Nvidia-GeForce-RTX-5090-Laptop-Analysis-Faster-but-there-is-more-to-come.988519.0.html)).
- **Correctness:** check each output against an FP64 reference built from the dequantized inputs, so the report covers correctness as well as speed.

Publish the code and a short write-up. Then post the MXFP8 accumulate finding in the relevant PyTorch and torchao threads ([pytorch/ao #4932](https://github.com/pytorch/ao/issues/4932)). That first public, verifiable result also starts to replace the credibility lost with the deleted repo.

## Stretch: a conformance suite for consumer Blackwell low precision

Once the verifier reaches v0.2, the stretch goal is a conformance suite for sm_120 low-precision compute. It extends the roofline harness into an automated test matrix: backend × format (NVFP4, MXFP4, MXFP8) × layout × shape × adversarial input. Each cell answers three questions: is the result correct, did it actually use the FP4/FP8 tensor cores, and how fast and how energy-efficient was it.

The case for it is that the stack's failures are mostly not missing primitives but **fragile integration**:

- **Silent fallbacks.** vLLM drops to Marlin W4A16 when the `flashinfer-python` and `flashinfer-cubin` versions are mismatched ([vLLM #31085](https://github.com/vllm-project/vllm/issues/31085)). NVIDIA's own Triton engineer said `triton_kernels` on SM120 "is actually doing dequantization and fp16 MMA" ([SGLang #19637](https://github.com/sgl-project/sglang/issues/19637)).
- **Edge-case numerics.** In MoE-Infinity's FP4 path, a scale byte of 0 decoded to 0 instead of the reference 2^-126 clamp, and the e8m0 value 255 (NaN) produced inf or NaN outputs ([MoE-Infinity PR #327](https://github.com/EfficientMoE/MoE-Infinity/pull/327)).
- **Wrong or empty outputs**, reported only in search snippets so far:
  - an RTX 5090 **Laptop** producing empty completions on vLLM's CUTLASS NVFP4 MoE backend ([vLLM #51525](https://github.com/vllm-project/vllm/issues/51525));
  - stale padded scale columns when K is not 32-aligned ([vLLM #48898](https://github.com/vllm-project/vllm/issues/48898));
  - all-backend NVFP4 errors in TensorRT-LLM ([TRT-LLM #14154](https://github.com/NVIDIA/TensorRT-LLM/issues/14154)).

**Technical core.**

- **Correctness oracle.** Dequantize inputs per block scale into FP64 and compute the reference GEMM. Use error bounds proportional to K × fp32 epsilon × the input magnitudes, and require exact equality for small-K cases whose products are exactly representable.
- **Adversarial generators.** Cover scale bytes 0 and 255, ue4m3 scale extremes, K not a multiple of 32 or 64, K below 64 (the Triton small-K fallback, [triton PR #11776](https://github.com/triton-lang/triton/pull/11776)), M=1, and grouped GEMMs with empty or highly uneven experts.
- **Native-path detector.** Capture the kernels that actually launched with the PyTorch profiler or Nsight Systems, dump their SASS, and check for block-scaled MMA opcodes as opposed to plain `HMMA`. This is the same SASS-inspection method one investigator used to show what the 5090 does and does not execute ([solatticus gist](https://gist.github.com/solatticus/aab6ec3a0436748b021cbbdd12e8c739)).
- **Backends:** cuBLASLt, CUTLASS 79a/79d, Triton, FlashInfer `mm_fp4` and its grouped MoE, b12x, and llama.cpp's NVFP4/MXFP4 types.

The suite should lead to concrete upstream contributions, starting with two that fit this machine:

1. **A fused BF16→NVFP4 activation-quantize kernel for llama.cpp.** The merged Blackwell FP4 PR measured this step at about **8% of time** and explicitly put off optimizing it ([llama.cpp #22196](https://github.com/ggml-org/llama.cpp/pull/22196)). It is small and measurable, and the review path is well-trodden.
2. **Reproducing and fixing vLLM #51525** on the same laptop SKU, using an NVFP4 MoE checkpoint that fits in 24 GB.

Only after those land does it make sense to take on a harder kernel, such as FP8 or FP4 attention for sm_120 inside FlashInfer or vLLM. The research flags that as the remaining headroom, since bf16 sm_120 attention is already within ±10% of FA2 ([HF flash-attn-4-sm120](https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120)).

## Conclusion

The common thread in the evidence is that consumer Blackwell's biggest remaining problem is trust, not missing silicon features or even missing kernels. Users cannot tell whether a package will load on their GPU, whether a library will resolve its dependencies, or whether an "NVFP4" path is really using FP4 tensor cores and returning correct numbers. Maintainers with no sm_120 hardware cannot tell either. That makes a solo developer with an sm_120 machine unusually valuable. The value is not in out-writing NVIDIA-staffed kernel teams. It is in being the reliable source of ground truth that those teams, wheel builders and users currently lack.

One part of the bet is still unproven: whether maintainers adopt a CI gate they never explicitly asked for. So the first external signal to track is not stars. It is whether one wheel builder merges the check within about six weeks of v0.1. If that happens, the verifier is on the path triton-windows took. If nobody merges it, shift effort toward the conformance suite. Its outputs are bug reports and fixes that upstream already wants, and they are worth building on.
