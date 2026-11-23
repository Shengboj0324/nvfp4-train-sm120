# Changelog

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
