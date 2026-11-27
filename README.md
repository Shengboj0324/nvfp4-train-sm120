# nvfp4-train-sm120

**NVFP4 (4-bit) training kernels and PyTorch layers for consumer Blackwell GPUs (RTX 50-series, `sm_120a`),
with 8-bit flash attention and NVFP4 mixture-of-experts.**

`nvfp4` is a CUDA extension plus a thin PyTorch layer. It runs the forward, input-gradient (dgrad) and
weight-gradient (wgrad) GEMMs of linear layers on the FP4 tensor cores, using NVIDIA's NVFP4 training
recipe (E2M1 values, E4M3 scales per 16 elements, an FP32 tensor scale, 16×16 weight blocks, random
Hadamard transform and stochastic rounding on gradients). Everything a training step needs around the
GEMMs is built in: fused quantizers, a GEMM epilogue for bias and residual, fused norm / activation
sub-blocks, attention and MoE. Every quantizer output is bit-exact against a pure-PyTorch executable
spec ([`nvfp4/reference.py`](nvfp4/reference.py)).

```
pip install --no-build-isolation .      # builds nvfp4._C for sm_120a (CUDA >= 12.8)
python -c "import nvfp4; print(nvfp4.available())"
```

## Contents

- [Why a separate stack for sm_120](#why-a-separate-stack-for-sm_120)
- [What runs in 4 bits](#what-runs-in-4-bits)
- [Install](#install)
- [API](#api)
- [Performance](#performance-rtx-5090-laptop-unless-noted)
- [Training quality](#training-quality)
- [Verification and development](#verification-and-development)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)

## Why a separate stack for sm_120

The RTX 50-series and the datacenter Blackwell parts share the FP4 data formats but not the programming
model. Kernels written for B200 (CUTLASS sm100 collectives, cuDNN / Transformer Engine FP4 paths) do not
run on consumer parts.

| | B200 / GB200 (`sm_100a`) | RTX 50 (`sm_120a`) |
|---|---|---|
| FP4 MMA | `tcgen05.mma`, one thread issues for the CTA, accumulators in tensor memory | warp-level `mma.sync...kind::mxf4nvf4.block_scale`, accumulators in registers |
| Tensor memory (TMEM) | 256 KB / SM | none |
| Shared memory / SM | 228 KB | ~100 KB (99 KB per block) |
| Hardware stochastic rounding (`cvt.rs`) | yes | no (rejected by ptxas, done in software) |
| Measured dense `mma.sync` peak (5090 Laptop) | | NVFP4 914 · INT8 473 · MXFP8 438 · BF16 121 TFLOPS |

The GEMM ([`csrc/gemm_tma.cu`](csrc/gemm_tma.cu)) is therefore a persistent, TMA-fed `mma.sync` kernel
with an mbarrier pipeline in registers. The quantizers fuse every element-wise producer that feeds a GEMM,
because at 4 bits the quantize passes, not the GEMMs, set the step time
([docs/DESIGN.md](docs/DESIGN.md) §3–4).

## What runs in 4 bits

For a linear layer `Y = X Wᵀ`, `NVFP4Linear` does:

| GEMM | computation | A operand (NVFP4) | B operand (NVFP4) |
|---|---|---|---|
| fprop | `Y = X Wᵀ` | X, 1×16 blocks, round-to-nearest | W, 16×16 blocks |
| dgrad | `dX = dY W` | dY, 1×16 blocks, stochastic rounding | Wᵀ (the same 16×16 blocks) |
| wgrad | `dW = dYᵀ X` | dYᵀ, 16-point RHT along tokens, SR | Xᵀ, same RHT |

- All three run on one TN kernel (`D = A·Bᵀ`). The **dual quantizer** emits an operand and its transpose
  from one read of the BF16 / FP32 input. It can apply LayerNorm, RMSNorm, GELU, SwiGLU, or their
  backwards while loading.
- Only NVFP4 operands (about 0.56 bytes per element) are saved for backward.
- Accumulation is FP32. The bias and the residual add happen in the GEMM epilogue, and wgrad can
  accumulate into an existing FP32 `.grad` in place.
- Attention (`nvfp4.attention`) runs Q Kᵀ in INT8 and P·V in MXFP8 (e4m3, P in two terms), with an exact
  flash backward on the dequantized operands.
- Embeddings, the LM head and (by convention) the last 1–2 blocks stay in BF16.

## Install

Requirements:

- an `sm_120` GPU: RTX 5090 / 5080 / 5070 Ti / 5070 / 5060 Ti / 5060, or the laptop parts;
- CUDA ≥ 12.8 (tested with 13.0, and 12.9 on Windows);
- a CUDA build of PyTorch (tested with 2.14.1+cu130 and 2.11 on Linux, 2.11.0+cu128 and 2.7.1+cu128 on Windows);
- Python ≥ 3.10.

```bash
# Linux / WSL2
sudo tools/setup_wsl.sh                  # optional: CUDA 13.0 toolkit + venv with PyTorch cu130 in ~/nvfp4
pip install --no-build-isolation .       # or: python setup.py build_ext --inplace (development)
tools/check.sh                           # build + full test suite under the thermal guard
```

```bat
:: native Windows (Visual Studio 2022 C++ workload, CUDA >= 12.8, CUDA PyTorch >= 2.7 in the active Python)
tools\build.cmd
tools\check.cmd
```

- **Without an `sm_120` GPU:** `NVFP4_SKIP_CUDA=1 pip install -e .` installs the Python package only.
  `NVFP4Linear(..., backend="emulate")` then runs the identical recipe through the reference (on any
  device, slowly), and GPU tests skip themselves.
- **Integration guide:** [docs/USAGE.md](docs/USAGE.md) covers requirements, every option, guarantees,
  production caveats and troubleshooting.

## API

```python
import torch, nvfp4
from nvfp4 import NVFP4Linear, convert_linears, ops

# 1. Drop-in linear layer (in / out features multiples of 128; autocast and fp32 / bf16 / fp16 inputs)
lin = NVFP4Linear(4096, 4096, bias=True, fuse_wgrad_accumulation=True,
                  cache_weight_quantization=True).cuda()
y = lin(x)                       # x [..., 4096]
y = lin(x, residual=r)           # r + lin(x), the add fused into the GEMM epilogue

# 2. Convert a model in place; keep the last blocks (and anything not % 128) in BF16
convert_linears(model.blocks, skip=("10.", "11."), fuse_wgrad_accumulation=True)

# 3. Fused transformer sub-blocks (norm / activation / residual inside the quantizers)
from nvfp4 import layernorm_linear, layernorm_mlp, rmsnorm_linear, rmsnorm_swiglu_mlp
q, k, v, res = rmsnorm_linear(x, blk.norm1, blk.qkv, chunks=3, return_residual=True)
x = blk.o_proj(attn(q, k, v), residual=res)
x = rmsnorm_swiglu_mlp(x, blk.norm2, blk.gate, blk.up, blk.down, residual=True)

# 4. 8-bit flash attention: q [B, H, N, D], k / v [B, Hkv, N, D], D in {64, 128}; GQA, causal or full
from nvfp4.attention import attention
o = attention(q, k, v, causal=True)            # backward="bf16" (default, exact) | "8bit" (FP16 MMAs, deterministic)

# 5. Mixture of experts: dropless top-k; NVFP4 experts run as grouped GEMMs (one launch per projection)
from nvfp4.moe import MoE
moe = MoE(d=1024, d_ff=2816, n_experts=8, top_k=2, activation="swiglu").cuda()  # w_gate/w_up/w_down [E, ., .]
convert_linears(moe, fuse_wgrad_accumulation=True)
y = moe(x); loss = loss + 0.01 * moe.aux_loss

# 6. Recipe switches (process-wide, read at kernel launch)
ops.set_fprop_terms(2)           # two-term forward activations Q(x) + Q(x - Q(x)): half the loss gap, -3% tokens/s
ops.set_high_precision(True)     # BF16 tail: NVFP4 layers compute in BF16 from here on (flip back any time)
ops.set_ms_eden(True)            # MS-EDEN wgrad operands (64-point RHT, unbiased scale-level SR)
ops.set_ms_eden(True, dgrad=True)  # ... and dgrad operands (dY and W^T rotated along N)
ops.set_four_six(True)           # Four Over Six block scales on round-to-nearest operands
ops.set_weight_blocks("1d")      # 1x16 weight blocks + re-quantized dgrad weight (default "2d")

# 7. Low-level ops
qa = ops.quantize(a)                                   # NVFP4Tensor(data u8 [M, K/2], scales e4m3-as-u8 [M, K/16], gdec fp32)
qb, qbt = ops.quantize_dual(b, block_2d=True)          # b and bᵀ from one read
d = ops.gemm(qa, qb, torch.bfloat16, bias=bias, residual=r)  # r + a @ bᵀ + bias
y = ops.gemm_grouped(qa, qw_stacked, "m", tile_group, group_rows=N)  # per-group a_g @ W_gᵀ (MoE)
```

| Module | Public surface |
|---|---|
| [`nvfp4.linear`](nvfp4/linear.py) | `NVFP4Linear`, `convert_linears`, `WeightQuantCache`, `invalidate_weight_caches` |
| [`nvfp4.fused`](nvfp4/fused.py) | `layernorm_linear`, `layernorm_mlp`, `gelu_mlp`, `rmsnorm_linear`, `rmsnorm_swiglu_mlp`, `swiglu_mlp` |
| [`nvfp4.attention`](nvfp4/attention.py) | `attention(q, k, v, *, causal, scale, smooth_k, backward)` (falls back to SDPA outside its coverage) |
| [`nvfp4.moe`](nvfp4/moe.py) | `MoE(d, d_ff, n_experts, top_k, activation, normalize_weights, bias, capacity_factor, shared_d_ff, ep_group, ep_node_size)`, `.to_nvfp4(...)`, `.aux_loss`, `.last_counts`, `.last_dropped`, `.bf16_impl`, `.ep_dispatch` / `.ep_chunks` / `.ep_wire` / `.ep_wire_grad` / `.ep_wire_return` / `.ep_serial` / `.ep_dynamic_tiles` / `.ep_overlap`; `ignore_expert_parameters_in_ddp`, `scale_expert_grads` |
| [`nvfp4.ops`](nvfp4/ops.py) | `quantize`, `quantize_dual` (+ `layernorm_` / `rmsnorm_` / `gelu_` / `swiglu_` fused variants and backwards), `gemm`, `gemm_grouped`, `stack_transposed`, `norm_backward`, recipe switches, `available` |
| [`nvfp4.reference`](nvfp4/reference.py) | pure-PyTorch spec: `quantize`, `quantize_dual`, `dequantize`, `apply_rht`, `e2m1_round_stochastic`, ... |

**Guarantees** ([USAGE.md §4](docs/USAGE.md)):

- **Reproducibility:** for a fixed seed, results are bit-reproducible run to run, and the same in eager,
  `torch.compile(fullgraph=True)` and CUDA-graph replay.
- **No host synchronization** in the dense layers or the MoE layer (NVFP4 experts, and the BF16 experts'
  grouped Triton path). Rounding seeds and expert sizes live on the GPU, so training steps (MoE included)
  can be captured in CUDA graphs.
- **Distributed:** DDP, FSDP1 and FSDP2 produce gradients bit-identical to a single process (MoE included).

## Performance (RTX 5090 Laptop unless noted)

Raw data and methodology are in [docs/RESULTS.md](docs/RESULTS.md) and `bench/results/*.json`; every
benchmark checks results against the reference before timing.

| Component | Result |
|---|---|
| NVFP4 GEMM | 273–441 TFLOPS on the laptop: 3.1–5.0× BF16 cuBLAS, 0.91–1.98× cuBLASLt NVFP4 (ahead on 10 of 11 shapes). Desktop RTX 5090: **up to 1027 TFLOPS**, 1.03–1.71× cuBLASLt on every shape |
| Dual quantizer | operand + transpose in one pass, 1.6–2.3× faster than two quantizations |
| GPT block, fwd + bwd | 1.82× (d = 768) → 2.65× (d = 2048) → 3.22× (d = 4096) vs BF16; 3.50× with the weight cache |
| Llama block (RMSNorm, SwiGLU) | 2.09× (d = 1024) → 3.32× (d = 4096) vs BF16; up to 3.64× cached |
| 8-bit attention | forward 1.2–1.6× FlashAttention (SDPA), fwd + bwd 1.07–1.12× at BF16 quality; forward-only up to 2.1× (`backward="8bit"`) |
| MoE layer, fwd + bwd (grouped GEMMs) | 2.13× (d = 1024, 8 experts) → 2.59× (d = 2048, 8 experts); 2.21× with 32 small experts; against grouped BF16 experts in CUDA graphs 2.13–2.60× |
| End to end | GPT-124M +32% tokens/s, −22% GPU energy; GPT-507M 1.62× tokens/s, −39% energy; MoE GPT (8 experts) +23% tokens/s |

## Training quality

The test bed is GPT-2-style models pretrained on TinyStories: 3000 steps, 49M tokens, embeddings and
the last 2 of 12 blocks in BF16. Losses are final validation loss, with gaps relative to BF16.

| Setup | Gap vs BF16 | Throughput vs BF16 |
|---|---|---|
| GPT-124M, default recipe, 3 seeds | +1.32% ± 0.16 (+0.019 nats), constant over training | +32% |
| GPT-507M (d = 2048), default recipe | +1.0% | +62% |
| GPT-124M + **two-term forward activations** (`ops.set_fprop_terms(2)`), 3 seeds | **+0.70% ± 0.09** | +25% |
| GPT-124M + **BF16 tail** (last 10% of steps in BF16), 3 seeds | **+0.62%** | +27% |
| GPT-124M + two-term activations + BF16 tail, 3 seeds | **+0.47% ± 0.14** | |
| GPT-124M + MS-EDEN + Four Over Six, 3 seeds | +1.21% | +26% |
| GPT-124M, 8-bit attention at 4096 tokens, 2 seeds | same as BF16 attention | |
| MoE GPT (8 experts, 294M parameters), pooled over 4 BF16 / 6 NVFP4 runs | +1.12% | +23% |

The 2026 recipe options (Four Over Six, MS-EDEN on wgrad and dgrad, 1×16 weights) each lower the loss by only
0.03–0.13% (about 0.2% in the best combinations) at this scale. Three quarters of the gap is the forward GEMM's
activations; quantizing them as two NVFP4 terms (the second the first's rounding error, one GEMM of twice the
depth) removes half of it, and so does switching to BF16 for the last tenth of training
([DESIGN.md §2.1–2.2](docs/DESIGN.md), [RESULTS.md §4.2–4.3](docs/RESULTS.md)). Validate the recipe on your own
model before relying on it.

## Verification and development

```bash
python setup.py build_ext --inplace              # incremental build of nvfp4/_C (MAX_JOBS to limit nvcc)
pytest -q tests/                                 # GPU tests skip without sm_120
pytest -q tests/test_reference.py                # spec-only tests, run anywhere (CI runs these on CPU)
tools/check.sh release                           # tests + compute-sanitizer (memcheck / racecheck /
                                                 #   synccheck / initcheck) + 5-minute soak with bit-exact replays
tools/run.sh NAME script.py [args]               # run any job under tools/guard.py
```

- **The spec is the contract.** Change [`nvfp4/reference.py`](nvfp4/reference.py) first, then make the
  CUDA quantizer match it bit for bit; `tests/test_*_cuda.py` compare outputs with `torch.equal`, not a
  tolerance. GEMMs are checked against an fp64 matmul of the dequantized operands.
- **Thermal guard.** [`tools/guard.py`](tools/guard.py) wraps every build, test and benchmark. It samples
  GPU temperature, power, memory and host RAM, and kills the job past configurable limits. It was written
  for laptops, where a sustained 175 W load has crashed the display driver.
- **Benchmarks:** `bench/bench_gemm.py`, `bench_quantize.py`, `bench_train_step.py`,
  `bench_llama_block.py`, `bench_attention.py`, `bench_long_context.py`, `bench_moe.py`,
  `bench_sr_cost.py`.
- **Experiments:**
  - Data and single runs: `experiments/prepare_tinystories.py`, then
    `experiments/train_gpt.py --precision {bf16,nvfp4} [--fuse-wgrad --fuse-producers --attn 8bit --moe-experts 8 --bf16-tail 0.1 ...]`.
  - Three-seed recipe ablations: `experiments/run_ablation.sh NAME FLAGS...`, summarized by
    `experiments/ablation_summary.py`.
- **CI** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the spec tests on CPU and builds
  the extension for `sm_120a` on Linux and Windows. It checks that the block-scaled MMA made it into SASS
  and reports register spills. GPU tests run locally.

## Repository layout

```
csrc/gemm_tma.cu        persistent TMA + mbarrier block-scaled mma.sync TN GEMM (kind::mxf4nvf4), dense and
                        grouped (ragged-M / ragged-K work units for mixture of experts)
csrc/gemm.cu            cp.async main loop (short K / fallback), tile + split-K dispatch
csrc/gemm_common.cuh    epilogue: deterministic split-K, fused fp32 accumulate, bias / residual add
csrc/quantize_dual.cu   fused row + transposed quantizer: 2D blocks, RHT16 / RHT64, SR, Four Over Six,
                        MS-EDEN, LayerNorm / RMSNorm / GELU / SwiGLU (and backwards) applied while loading
csrc/quantize.cu        single-output quantizer; process-wide recipe switches
csrc/nvfp4_common.cuh   E2M1 / E4M3 conversion, integer-domain stochastic rounding, hashing, Hadamard
csrc/norm.cu            LayerNorm / RMSNorm backward with the residual gradient, deterministic
csrc/attention.cu       8-bit flash attention: operand quantizer, INT8 / MXFP8 forward, 8-bit backward
csrc/bindings.cpp       torch op registrations (fake tensors for torch.compile in nvfp4/_library.py)
nvfp4/reference.py      executable spec of format and recipe
nvfp4/ops.py            Python ops, NVFP4Tensor, recipe switches
nvfp4/linear.py         NVFP4Linear autograd, weight-quantization cache, convert_linears
nvfp4/fused.py          fused transformer sub-blocks (autograd functions)
nvfp4/attention.py      attention() with both backwards and the SDPA fallback
nvfp4/moe.py            MoE layer: routing, deterministic dispatch / combine, grouped NVFP4 expert MLPs
nvfp4/_bf16_grouped.py  grouped BF16 GEMMs (Triton) for the MoE layer's high-precision path
tests/                  1031 tests: bit-exact vs spec, autograd, compile / graphs, DDP / FSDP, MoE, attention
bench/, experiments/    benchmarks and pretraining runs, each with results/ (raw JSON / JSONL)
tools/                  build, check, guard, sanitize, soak, WSL setup, Runpod session script
docs/                   USAGE.md, DESIGN.md, RESULTS.md, research/ (survey of open problems, Nov 2026)
```

## Limitations

The full list is in [DESIGN.md §6](docs/DESIGN.md#6-known-limitations-and-next-steps).

- **Locked-clock GEMM efficiency.** At locked clocks the GEMM keeps the tensor pipe 71% busy, vs
  cuBLASLt's ~90%. At real clocks it is ahead of cuBLASLt on every benchmarked shape.
- **8-bit attention training speedup is small:** 1.07–1.12× at BF16 quality. The exact backward runs at
  BF16 speed, and P·V needs two e4m3 terms. Coverage is causal or full masks and head dims 64 / 128;
  everything else falls back to SDPA.
- **MoE coverage.** Top-k routing, dropless or with a capacity; SwiGLU / GELU experts with or without
  biases; a shared expert; expert parallelism (each token once per rank or, with `ep_node_size`, once per
  node; chunked overlap with serialized exchanges and run-time tile claiming; NVFP4 dispatch and MXFP8
  return messages; with DDP over the rest). On four RTX PRO 4500 whose GPU pairs are linked slowly,
  two-stage expert parallelism with chunks and both message formats is 2.9× faster per layer step than data
  parallelism (67 vs 195 ms; v0.17: 96 ms) and v0.17's setup trained a MoE GPT 1.27× faster with less
  memory; on a fast pair it ties or beats data parallelism (RESULTS.md §8.2, §8.7–8.8). Not timed on a real
  multi-node cluster. Without Triton (PyTorch's native Windows wheels) the BF16 experts run as a loop that
  reads the expert sizes on the host.
- **Stochastic rounding is software** (no `cvt.rs` on sm_120a), at about 12 extra instructions per
  element.
- **Hardware coverage.** Tested on the RTX 5090 Laptop, the desktop RTX 5090 (two over NCCL) and the RTX
  PRO 4500 Blackwell (four over NCCL; RESULTS.md §9), not yet on the smaller GeForce parts. On the RTX PRO
  4500 the 8-bit attention is no faster than SDPA.

## License

Apache-2.0
