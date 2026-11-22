# Results

All results were measured on one machine on 2026-11-21:

- **GPU:** NVIDIA GeForce RTX 5090 **Laptop** GPU (GB203, 82 SMs, 24 GB, ~896 GB/s, ~175 W board power)
- **Driver:** 595.79 (Windows 11), running in WSL2 Ubuntu 22.04
- **Software:** CUDA 13.0, PyTorch 2.14.1+cu130

Raw data sits next to each table: `bench/results/*.json` and `experiments/results/*.jsonl|csv`.
Every benchmark checks correctness against the fp64 reference before timing. Timings are the
median of 7 trials, each a batch of back-to-back launches (see `bench/bench_gemm.py::timeit` for why
per-launch syncing is misleading on a laptop GPU). This is one laptop with real thermal and power
behaviour; expect about 5–10% run-to-run variation.

## 1. GEMM throughput (TFLOPS, `bench/results/gemm.json`)

| M × N × K | BF16 cuBLAS | **NVFP4, this repo** | this repo incl. quantizing both operands | NVFP4 cuBLASLt (`torch._scaled_mm`) | ours / BF16 | ours / cuBLASLt |
|---|---|---|---|---|---|---|
| 2048³ | 86 | **261** | 147 | 233 | 3.05× | 1.12 |
| 4096³ | 93 | **337** | 231 | 336 | 3.60× | 1.00 |
| 8192³ | 90 | **356** | 276 | 413 | 3.95× | 0.86 |
| 8192×8192×2048 | 99 | **332** | 273 | 243 | 3.35× | 1.37 |
| 8192×2048×8192 | 93 | **350** | 209 | 343 | 3.77× | 1.02 |
| 2048×8192×8192 | 92 | **355** | 212 | 340 | 3.84× | 1.05 |
| 8192×3072×768 | 87 | **283** | 172 | 140 | 3.25× | 2.03 |
| 8192×768×3072 | 91 | **325** | 136 | 320 | 3.58× | 1.02 |
| 768×3072×8192 | 85 | **339** | 133 | 396 | 3.99× | 0.86 |

- **Against BF16:** the NVFP4 kernel runs **3.0–4.0× faster than BF16 cuBLAS**.
- **Against cuBLASLt NVFP4:** it is faster on short-K shapes (up to 2×) and reaches about 86% of
  cuBLASLt on the longest-K shapes.
- **Output agreement:** cuBLASLt and our output agree within bf16 rounding (max |Δ| / max |D| ≈ 0.3–0.6%).
  Ours matches the fp64 reference to about 1e-7.
- **Hardware ceiling:** NVIDIA's dense FP4 peak for the *desktop* RTX 5090 is 1676 TFLOPS. The
  laptop part has fewer than half its SMs and a 175 W board limit.
- **Pipeline variants:** GEMM pipeline variants (`bench/results/gemm_variants.txt`) gave no
  consistent winner, so the default stays 4 stages with 1 CTA per SM.

## 2. Quantizer bandwidth (`bench/results/quantize.json`)

Effective GB/s counts one bf16 read plus the NVFP4 write. The two-pass design actually reads the
input twice, which the L2 cache absorbs for tensors that fit in it.

| Tensor | rowwise | rowwise, 16×16 2D | transposed | transposed + RHT + SR (wgrad) |
|---|---|---|---|---|
| 8192 × 2048 bf16 | 725 GB/s (59 µs) | 470 | 503 | 254 |
| 8192 × 8192 bf16 | 488 GB/s (352 µs) | 191 | 406 | 225 |

## 3. Transformer-block training step vs width (`bench/results/train_step.json`)

This measures one pre-LN GPT block (qkv, SDPA attention, out-proj and a 4× MLP), forward and
backward, on 8192 tokens with BF16 autocast. The four linears either run on NVFP4Linear or stay
BF16. LayerNorm, attention and GELU are BF16 in both cases.

| d_model | BF16 step | NVFP4 step | **speedup** | effective linear TFLOPS, BF16 → NVFP4 |
|---|---|---|---|---|
| 768 | 6.21 ms | 4.83 ms | **1.29×** | 56 → 72 |
| 1024 | 10.17 ms | 7.08 ms | **1.44×** | 61 → 87 |
| 2048 | 33.66 ms | 19.17 ms | **1.76×** | 74 → 129 |
| 3072 | 79.33 ms | 38.92 ms | **2.04×** | 70 → 143 |
| 4096 | 134.69 ms | 63.19 ms | **2.13×** | 73 → 157 |

The benefit grows with width because GEMM time grows as d², while the quantization passes and the
BF16 parts (attention, norms) grow more slowly.

## 4. End-to-end pretraining: BF16 vs NVFP4 (`experiments/results/`)

**Setup**

- **Model:** GPT-2-small-shaped, 124M parameters (d=768, 12 layers, 12 heads, context 512, tied
  embeddings).
- **Data:** TinyStoriesV2 (first 250 MB, 63.7M GPT-2 tokens).
- **Schedule:** 3000 steps × 16,384 tokens (49M tokens), AdamW (0.9, 0.95, wd 0.1), lr 6e-4 with
  cosine decay to 10%, 200 warm-up steps, gradient clipping at 1.0.
- **Controls:** the same seed, initialization and data order in every run. Validation uses 40
  fixed batches.

**Runs**

- **NVFP4:** every block linear (qkv, out-proj, fc1, fc2) trains in NVFP4, except in the last two
  blocks (40 of 48 linears).
- **Ablations:** "all blocks" puts all 48 in NVFP4. "No RHT" disables the wgrad Hadamard transform.

![loss curves](../experiments/results/loss.png)

| Run | final val loss | Δ vs BF16 | median tokens/s | GPU energy / 1k tokens | wall time | peak GPU temp |
|---|---|---|---|---|---|---|
| BF16 (autocast) | **1.4884** | — | 63,290 | 2.85 J | 798 s | 77 °C |
| **NVFP4 recipe** (last 2 blocks BF16) | 1.5064 | +0.018 (+1.2%) | **70,086 (+10.7%)** | **2.44 J (−14.5%)** | 718 s | 76 °C |
| NVFP4, all blocks | 1.5087 | +0.020 | 71,531 (+13.0%) | 2.39 J (−16.2%) | 703 s | 78 °C |
| NVFP4, no wgrad RHT | 1.5075 | +0.019 | 70,526 | 2.43 J | 714 s | 77 °C |

**What this shows**

- **Training is stable.** NVFP4 tracks BF16 from the first steps to the end, with no instability,
  no loss spikes and no divergence.
- **The quality gap is constant.** It stays within +0.016 to +0.019 nats at every evaluation from
  step 250 to step 3000, so it doesn't grow as training continues.
- **Keeping the last two blocks in BF16** recovers about 0.002 of the gap, roughly 10%.
- **The wgrad RHT is not measurable here.** At this scale its effect (0.001) is below what one run
  per configuration can resolve. NVIDIA reports that it matters for much larger models and longer
  runs, where outlier-heavy gradients appear. We don't claim a benefit we didn't measure.
- **End-to-end speedup is modest at this width** (+10.7% tokens/s; −14.5% energy, since the GPU sat
  at its ~170 W limit either way). That's expected from section 3: at d=768, the BF16 LM head
  (50304×768), attention, LayerNorms and quantization overhead are a large share of the step.
  Section 3 suggests models of width ≥ 2048 would see 1.7–2.1× on the transformer blocks.

**Caveats**

- One seed per configuration, one small dataset, and 49M tokens.
- The gap could be narrowed with known techniques that this repo doesn't implement yet, such as
  BF16 "healing" for the last few percent of tokens.
- Results on a desktop RTX 5090, which has twice the SMs and three times the power budget, will
  differ.

## Reproduce

```bash
tools/setup_wsl.sh                       # CUDA 13.0 toolkit + PyTorch cu130 venv (WSL2/Ubuntu 22.04)
tools/check.sh bench                     # build, 64 tests, GEMM + quantizer benchmarks
python bench/bench_train_step.py
python experiments/prepare_tinystories.py && experiments/run_main.sh && python experiments/plot.py
```
