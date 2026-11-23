# Changelog

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
