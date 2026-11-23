# nvfp4-train-sm120

**4-bit (NVFP4) training on consumer Blackwell GPUs (RTX 50-series, `sm_120`).**

The repository contains block-scaled FP4 tensor-core GEMMs (a TMA-fed main loop plus split-K),
bit-exact fused quantizers that emit an operand and its transpose in one pass (with 16-point random
Hadamard transform and stochastic rounding, and with LayerNorm / GELU computed inside the
quantizer), and drop-in layers that run the forward, input-gradient and weight-gradient GEMMs in
NVFP4 and save only 4-bit operands for backward.

Consumer Blackwell has hardware FP4 tensor cores, but its programming model differs from datacenter
Blackwell: warp-level `mma.sync` instead of `tcgen05`, no tensor memory, and about 99 KB of shared
memory. Mainstream training stacks don't yet use it for training. These kernels are written for
that model from the ground up (see [docs/DESIGN.md](docs/DESIGN.md)).

## Results on an RTX 5090 Laptop GPU

Full details and raw data are in [docs/RESULTS.md](docs/RESULTS.md).

| | |
|---|---|
| NVFP4 GEMM throughput | **266–410 TFLOPS**: **3.2–4.7× BF16 cuBLAS**; on par with cuBLASLt NVFP4 on long K (96–121%), up to 2× faster on short K |
| Fused dual quantizer | operand + transposed operand in one pass: **1.6–2.3×** faster than two separate quantizations |
| Transformer block fwd+bwd | **1.80× (d=768) → 2.42× (d=2048) → 3.10× (d=4096)** vs BF16, with 7–14% less memory |
| GPT-507M (d=2048) pretraining | val loss **1.6954 vs 1.6786** BF16 (+1.0%), **1.62× tokens/s, −39% GPU energy** |
| GPT-124M (d=768) pretraining, 49M tokens | val loss **1.5062 vs 1.4892** BF16 (constant ~0.017 gap, no divergence), **+27% tokens/s, −22% GPU energy** |
| Same, three seeds (v0.5) | gap **+0.019 ± 0.002** nats (+1.28%), constant over training for every seed; +27.7% tokens/s |
| Framework integration (v0.5) | `torch.compile(fullgraph=True)` without graph breaks (bit-identical to eager), CUDA-graph capture of whole training steps, no host syncs, DDP and FSDP2 bit-exact vs single process |
| Correctness | quantizers and fused LayerNorm/GELU **bit-exact** vs the reference spec; GEMM = fp64 reference to ~1e-7; 321 tests; compute-sanitizer clean (memcheck, racecheck, synccheck, initcheck); 8-minute soak with bit-exact replays |

![loss curves](experiments/results/v0.3_wide/loss.png)

## Quick start (WSL2 or Linux, RTX 50-series)

```bash
sudo tools/setup_wsl.sh            # CUDA 13.0 toolkit + venv with PyTorch 2.14 cu130 (~/nvfp4)
pip install --no-build-isolation . # build the extension for sm_120a (or: pip wheel ... for a wheel)
tools/check.sh                     # run all tests; `tools/check.sh release` adds sanitizers + soak
```

**Integrating into a training stack:** read [docs/USAGE.md](docs/USAGE.md) (requirements, options,
guarantees, production caveats, troubleshooting). Changes per version: [CHANGELOG.md](CHANGELOG.md).

```python
import torch
from nvfp4 import NVFP4Linear, convert_linears

layer = NVFP4Linear(4096, 4096, bias=False).cuda()        # in/out features must be multiples of 128
y = layer(torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16))

# or convert an existing model's block linears, keeping the last blocks in BF16:
convert_linears(model.blocks, skip=("10.", "11."))

# pre-LN sub-blocks with LayerNorm / GELU fused into the NVFP4 quantizers
from nvfp4 import layernorm_linear, layernorm_mlp
qkv = layernorm_linear(x, block.ln1, block.qkv)           # block.qkv: NVFP4Linear (bias-free)
out = layernorm_mlp(x, block.ln2, block.fc1, block.fc2)   # fc2(gelu_tanh(fc1(ln2(x))))

# low-level ops
from nvfp4 import quantize, gemm
qa = quantize(a)                                          # NVFP4Tensor(data, scales, gdec)
qb = quantize(b, block_2d=True)
d = gemm(qa, qb)                                          # D = A @ B^T on FP4 tensor cores
```

Requirements: an sm_120 GPU (RTX 5090/5080/5070/5060 or their laptop parts), CUDA ≥ 12.8 (tested
with 13.0), and a CUDA build of PyTorch (tested with 2.14.1+cu130, which needs C++20). Native Windows
builds haven't been tried. On a machine without such a GPU, `NVFP4Linear(..., backend="emulate")`
runs the same recipe through the pure-PyTorch reference.

## Layout

```
csrc/quantize_dual.cu  fused BF16/FP32 -> NVFP4 rowwise + transposed (RHT, SR, 16x16 2D) in one pass,
                       optionally applying LayerNorm / GELU / GELU-backward while loading
csrc/quantize.cu       single-output quantizer (rowwise / transposed / 16x16 2D, RHT, stochastic rounding)
csrc/gemm_tma.cu       mma.sync kind::mxf4nvf4 block-scaled TN GEMM, TMA + mbarrier pipeline (K >= 1024)
csrc/gemm.cu           cp.async GEMM main loop (short K / fallback) and host dispatch
csrc/gemm_common.cuh   shared epilogue: deterministic split-K, fused fp32 accumulate (D += A @ B^T)
nvfp4/reference.py     executable spec of the format and recipe (pure PyTorch)
nvfp4/linear.py        NVFP4Linear autograd layer + convert_linears
nvfp4/fused.py         layernorm_linear / layernorm_mlp with fused producers
tests/                 bit-exact / reference tests (GPU tests auto-skip without sm_120)
bench/                 GEMM, quantizer and transformer-block benchmarks (+ results/)
experiments/           TinyStories GPT pretraining: BF16 vs NVFP4 (+ results/)
tools/guard.py         runs jobs under GPU temperature / memory / host-RAM limits, with cooldowns
tools/sanitize.sh      compute-sanitizer over every kernel variant; tools/soak.py: long-run soak test
docs/                  USAGE.md (integration guide), DESIGN.md, RESULTS.md, research/
```

## Status and limitations

Verified on one GPU model (RTX 5090 Laptop) with the test, sanitizer and soak suites above and two
end-to-end pretraining comparisons; validate the precision recipe on your own model before relying on
it. The known gaps are listed in
[docs/DESIGN.md §6](docs/DESIGN.md#6-known-limitations-and-next-steps):
- at equal clocks the GEMM keeps the tensor pipe 69% busy vs cuBLASLt's 90% (the gap disappears at
  this laptop's power limit, but would show on a GPU with more power headroom);
- the fused sub-blocks cover pre-LN blocks with bias-free linears and tanh-GELU; attention-side
  producers aren't fused;
- hardware stochastic rounding (`cvt.rs`) isn't available on sm_120a, so SR runs in software;
- multi-process tests run two ranks on one GPU (gloo); FSDP1 is untested;
- only Linux/WSL2 is supported.

## License

Apache-2.0
