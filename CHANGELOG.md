# Changelog

## 0.9.1 — accurate dQ in 8-bit attention, soak coverage (2026-11-24)

- **Fix: dQ of `nvfp4.attention`** was 7–24% off on a trained model's activations at 4096 tokens (dK, dV:
  2–3%), costing +0.72% validation loss on GPT-124M at that context. Two causes, both from cancellation
  (every row of dS sums to zero, so dQ is a small difference of large terms): the MXFP8 product dS K, and
  Delta = rowsum(dO O) taken from the 8-bit forward's O. dQ is now computed in BF16 against a mean-centred
  BF16 copy of K, with a correction (dQ − r B, B = P K in MXFP8, r the row sums of dS) that equals the
  consistent Delta' = rowsum(P dP): dQ error 0.5–1.5%, loss gap at 4096 tokens +0.52% (2 seeds). Forward
  unchanged (1.6–2.2× flash attention); forward + backward now 1.14–1.18× (was 1.4–1.5×). The docs now
  say where 8-bit attention pays and where it costs quality (DESIGN.md §7, RESULTS.md §7, USAGE.md §3.3).
- `tools/soak.py` covers 8-bit attention (a block with it, a GQA head-dim-128 replay check, the
  CUDA-graph block). Long-context convergence scripts `experiments/run_attn8_long*.sh`.
- The attention kernels' host launchers are plain templates (MSVC crashed on the nvcc stub of the 0.9.0
  version's generic lambdas; the Windows CI build of 0.9.0 is fixed by this).

## 0.9.0 — 8-bit flash attention (2026-11-24)

- **`nvfp4.attention`**: forward and backward attention kernels for sm_120 (DESIGN.md §7). Q K^T and
  dO V^T on INT8 tensor cores with per-token scales (4× less error than E4M3 on the score products;
  K centred by its mean first), P V, P^T dO, dS K and dS^T Q on MXFP8 (block-scaled E4M3, the only FP8
  path with FP32 accumulation at full rate on consumer Blackwell); softmax statistics and accumulation
  in FP32. Operands quantized once (`ops`-level `attn_quant`, one read) into the layouts both passes
  need; P and dS quantized in registers through a permuted sequence order. Causal / full, GQA,
  any length, head dims 64 / 128, bf16 / fp16, strided inputs, gradients in the inputs' layout, a
  **bitwise-deterministic backward** (no atomics), `torch.compile` custom ops; anything else falls back
  to SDPA.
- Speed (RTX 5090 Laptop): forward 1.6–2.2× PyTorch's flash attention from 4k tokens, forward + backward
  1.4–1.5× (head dim 128; 1.1–1.2× at 64). Llama block (d = 2048, fused NVFP4): 1.22× faster at 8192
  tokens, 1.35× at 16384 (2.27× BF16). GPT-124M: validation loss unchanged (1.4892 with BF16 linears;
  1.5095 vs 1.5073 with NVFP4 linears). `experiments/train_gpt.py --attn 8bit`,
  `bench/bench_attention.py`, `bench/bench_long_context.py`.
- GEMM: warp-tile ablation of the locked-clock gap (`NVFP4_NVCC_FLAGS=-DNVFP4_WARPS_N=2`, DESIGN.md §4.2):
  a 64 × 64 warp tile needs more than 255 registers and runs at half speed, so the layout stays.
  The cp.async loop's loaders no longer assume 256 threads.
- 682 tests (45 new); sanitizers cover the attention kernels; `tools/pod_session.sh` also runs the
  attention and long-context benchmarks. Validation on the smaller RTX 50 parts is still open: on
  2026-11-24 Runpod had no RTX 5070 Ti / 5060 Ti / 5070 capacity, and five Community RTX 5080 hosts never
  started a container (with three start commands and Runpod's own image).

## Validated on rented desktop RTX 5090s (2026-11-23)

- **2× desktop RTX 5090** (CUDA 12.8, PyTorch 2.11): 625 tests incl. the two-GPU NCCL tests (DDP,
  FSDP1, FSDP2 bit-exact across two physical GPUs), sanitizers clean, soak OK; GEMM up to 1027 TFLOPS,
  1.03–1.71× cuBLASLt; GPT-124M +32% tokens/s at the same loss. DDP on that host was bound by a
  2 GB/s all-reduce (no P2P), equally for BF16 (RESULTS.md §6). Results in
  `experiments/results/rtx5090_desktop_2gpu/`.
- `tools/allreduce_probe.py` (NCCL all-reduce bandwidth); shell scripts executable in git and
  `tools/sanitize.sh` finds python / compute-sanitizer on any machine (found on the rented machine).

## 0.8.0 — residual / epilogue fusion, persistent GEMM, cheaper stochastic rounding (2026-11-23)

- **Residual adds, biases and the q / k / v hand-off are fused** (DESIGN.md §3.6). The GEMM epilogue
  adds a bias and a residual in its store (`gemm(..., bias=, residual=)`, bit-identical to adding
  afterwards); `NVFP4Linear` uses it for its bias and for `forward(x, residual=r)`, taking the bias
  gradient from column sums of the output-gradient quantizer. The sub-blocks gain `chunks=3` (q, k, v
  as views; their gradients quantized in place by the new multi-part `quantize_dual([...])` instead of
  being stacked and copied), `return_residual=True` (a residual handle whose gradient is added inside
  the norm backward) and `residual=True` on the MLPs. A new deterministic LayerNorm / RMSNorm backward
  kernel (`csrc/norm.cu`, `ops.norm_backward`) replaces PyTorch's, producing dx (with the residual
  gradient added), dw and db from one read. Llama block: 6–8% faster than the 0.7 fused path.
- **SwiGLU gate / up biases** (added inside the SwiGLU kernels, gradients from the same read) and
  **norm-free MLPs** for post-norm blocks: `layernorm_mlp(x, None, ...)`, `gelu_mlp`, `swiglu_mlp`.
- **Persistent TMA GEMM** (DESIGN.md §4.2): one CTA per SM carries the load pipeline across output
  tiles, the last warp to release a stage issues its refill at once, and the accumulating epilogue
  batches its loads behind an L2 prefetch. 8–17% fewer cycles at locked clocks, up to 8% faster at the
  power limit (8192³: 441 TFLOPS, above cuBLASLt's 427), the accumulating wgrad GEMM 3–28% faster; it
  is now also used from K = 512 on large grids.
- **Stochastic rounding: one hash per element pair** (16-bit uniforms): SR adds 34–43% to the dual
  quantizer instead of 48–61%. The SR noise for a given seed differs from 0.7's.
- Numerics: fp32 output gradients (an fp32 residual stream) are quantized without first rounding
  them to bf16. Forward outputs of every layer and block are unchanged; gradients are not
  bit-identical to 0.7's.
- Block benchmarks time every variant in two passes in opposite orders (the power-limited GPU slows
  as it heats up). Weight-cache hits under FSDP2 when the allocator returns the same address.
- GPT block: 1.82–3.22× BF16 (3.50× with the weight cache); Llama block: 2.09–3.32× (3.64×). 637 tests.

## 0.7.0 — Llama-style blocks, weight cache, fused biases (2026-11-23)

- **Llama-style fused blocks:** `rmsnorm_linear` and `rmsnorm_swiglu_mlp`. RMSNorm, SwiGLU and both
  halves of its backward run inside the quantizers (new ops `rmsnorm_quantize_dual`,
  `swiglu_quantize_dual`, `swiglu_backward_quantize_dual`, bit-exact against the reference). The
  normalized input is quantized once for gate and up. A Llama block step is 1.88–2.96× BF16, 9–14%
  faster than unfused NVFP4, with 5–8% less memory (`bench/bench_llama_block.py`).
- **Weight-quantization cache** (`NVFP4Linear(cache_weight_quantization=True)`): micro-batches after
  the first skip the weight quantizer (2–6% per block step, bit-identical results). Invalidation
  counts optimizer steps with a global hook because fused AdamW does not bump version counters (found
  by the tests). `invalidate_weight_caches(model)` covers in-place `.data` writes. Tested under DDP,
  FSDP2, CUDA graphs, compile and checkpointing.
- **Fused GPT blocks accept biases and exact GELU:** `layernorm_mlp(..., approximate="none")`. fc1's
  bias is added inside the GELU producer, and its gradient comes from deterministic per-tile column
  sums computed in the same kernel (`gelu_backward_quantize_dual(..., bias_grad=True)`).
- Two-GPU NCCL tests (DDP, FSDP1, FSDP2, weight cache) that skip on one GPU; DDP in
  `experiments/train_gpt.py` via torchrun; `tools/guard.py` watches every GPU; `tools/pod_session.sh`
  runs the whole validation on a rented multi-GPU machine.
- 528 tests.

## 0.6.1 — Windows build with CUDA 13 (2026-11-23)

- **Fixed: the TMA GEMM did not compile with MSVC and CUDA 13.** CUDA 13 declares `CUtensorMap`
  `alignas(128)` (12.x: 64) and MSVC cannot pass a by-value kernel parameter aligned beyond 64 bytes
  (error C2719). Under MSVC the kernel now takes a 64-byte-aligned copy with the same layout (TMA
  needs 64); Linux builds are unchanged. Found by the new Windows CI job.

## 0.6.0 — platforms and the remaining integration gaps (2026-11-22)

- **Native Windows.** The extension builds with MSVC + nvcc (`tools\build.cmd`, `tools\check.cmd`;
  setup.py picks MSVC flags) and passes the test suite on the GPU under Windows with PyTorch
  2.7.1+cu128 / CUDA 12.9. CI gains a Windows job that builds against CUDA 13.0 + PyTorch 2.14.1 and
  checks the FP4 MMA is in the SASS. `tools/guard.py` runs on Windows too (Win32 memory and CPU-load
  sampling, process-tree kill).
- **FSDP1 tested**: `FullyShardedDataParallel` (flat parameters and `use_orig_params=True`, `no_sync`
  accumulation) gives gradients bit-identical to a single process, fused and unfused. Fused
  accumulation now explicitly skips non-leaf parameters (FSDP1's views), which also avoids PyTorch's
  warning about reading `.grad` of a non-leaf.
- **Reproducible attention.** Under `torch.use_deterministic_algorithms(True)` a block with NVFP4
  layers and PyTorch's default flash attention reproduces its gradients bit for bit (new test). The
  soak now runs this way instead of on the slow math backend (~1.65× more iterations per minute), and
  deterministic mode's NaN-filled allocations additionally check that no kernel reads uninitialized
  output.
- **Clear error for `CUDA_FORCE_PTX_JIT`.** With that variable set every sm_120 launch fails with "no
  kernel image"; `nvfp4.available()` is False and the ops explain why.
- Tests pass an explicit device mesh to FSDP2 (PyTorch's Windows builds have no NCCL) and account for
  PyTorch 2.7's FSDP2 summing gradients over gloo where 2.14 averages them (measured on `nn.Linear`).
  `torch.compile` tests use the `aot_eager` backend where Triton is missing. 324 tests.

## 0.5.0 — training-stack integration (2026-11-22)

- **Seeds on the GPU.** The layers draw their stochastic-rounding / RHT seeds with the CUDA generator
  into a small int64 tensor that the quantizers read directly (the low-level ops accept an int or a
  one-element int64 tensor per seed). A training step no longer synchronizes with the host, and
  activation checkpointing restores the seeds with the CUDA RNG state.
- **CUDA graphs:** whole training steps (forward, backward, optimizer) can be captured; every replay
  draws fresh seeds. `tools/soak.py` now also trains a block through a captured graph.
- **torch.compile:** the kernels are registered as `torch.library` custom ops (`nvfp4._library`), so
  `fullgraph=True` compiles models with NVFP4 layers and the fused LayerNorm / MLP blocks without graph
  breaks; with inductor's `fallback_random` the compiled step is bit-identical to eager.
- **Fused wgrad accumulation works with DDP and FSDP2.** The accumulated gradient is handed to
  autograd (adopted without a copy), so gradient hooks fire; under `torch.autograd.grad` it falls back
  to the ordinary path instead of touching `.grad`.
- **Layer outputs are not views** (like `nn.Linear`): FSDP2 warned about views returned by modules.
- **New tests** (`tests/test_integration_cuda.py`): device seeds bit-identical to int seeds, no host
  sync in a training step, compile without graph breaks, CUDA-graph training, fused accumulation
  hooks, and two-process DDP / FSDP2 (gloo) bit-exact against a single-process reference. 321 tests.
- Performance: per-kernel GPU time unchanged; a transformer-block step is within ~1% of 0.4.0
  (+0.5–1.5% at d = 768–2048, A/B in one session).
- **Quality over three seeds** (GPT-124M, 49M tokens): NVFP4 − BF16 validation loss +0.019 ± 0.002
  nats (+1.28%), constant over training for every seed, +27.7% tokens/s (RESULTS.md §4.1).
- Tooling: `tools/guard.py` also waits for the host load average to settle before starting a job;
  the sanitizer driver covers device seeds and a layer forward + backward (0 findings).

## 0.4.0 — production hardening (2026-11-22)

- **Fixed: NaN inputs were silently quantized to finite values.** `fmaxf` drops NaN, so the global
  amax ignored it and a NaN element became an ordinary 4-bit code. The amax reductions now keep NaN
  and the global scale is NaN for a NaN amax (kernels and reference), so non-finite inputs always
  produce non-finite outputs, like ±inf already did.
- **Fixed: multi-GPU processes.** The GEMM's dynamic-shared-memory attribute was set once per process
  instead of once per device; a second GPU in the same process could not launch it.
- **Fixed: arch check.** `available()` accepted any GPU with capability ≥ 12.0; the `sm_120a` code runs
  on 12.0 only. All ops now validate the device and dtype up front with clear messages.
- **New:** `tools/sanitize.sh` (compute-sanitizer memcheck / racecheck / synccheck / initcheck over
  every kernel variant: 0 findings), `tools/soak.py` (multi-minute training + GEMM soak with bit-exact
  replay checks), `tools/check.sh release`.
- **New:** `tests/test_robustness_cuda.py` (non-finite propagation, misuse errors, shapes and dtypes,
  bf16 weights, state dicts, activation checkpointing, torch.compile, CUDA graphs, concurrent streams
  with split-K, DDP). 305 tests.
- **Packaging:** version from `nvfp4.__version__`, `pyproject.toml`, wheel metadata; the public API
  exports the layers and all ops (`from nvfp4 import NVFP4Linear, layernorm_mlp, ...`).
- **Docs:** [docs/USAGE.md](docs/USAGE.md), an integration and operations guide.

## 0.3.0 — TMA GEMM, split-K, fused producers

- TMA-fed GEMM main loop (≈ cuBLASLt NVFP4 at the power limit, 7–25% faster than the cp.async loop),
  scales fetched 4 k-tiles at a time; deterministic split-K for small output grids.
- LayerNorm, GELU and GELU-backward fused into the dual quantizer (`nvfp4.fused`).
- Warp-reconvergence fix and bounded barrier waits in the TMA GEMM.
- Safer `tools/guard.py` (84 °C limit, cool-down before/after jobs); corrected energy integration.

## 0.2.0 — fused dual quantizer

- One read per operand produces both layouts; NVFP4-saved activations; hardware E2M1 convert;
  optional fused wgrad accumulation.

## 0.1.0 — first NVFP4 training stack for sm_120

- Block-scaled `mma.sync` GEMM, bit-exact quantizers with RHT and stochastic rounding, `NVFP4Linear`,
  GPT-124M BF16-vs-NVFP4 pretraining comparison.
