# Results

All results were measured on one machine on 2026-11-21/23:

- **GPU:** NVIDIA GeForce RTX 5090 **Laptop** GPU (GB203, 82 SMs, 24 GB, ~896 GB/s, 64 MB L2, ~175 W board power)
- **Driver:** 595.79 (Windows 11), running in WSL2 Ubuntu 22.04
- **Software:** CUDA 13.0, PyTorch 2.14.1+cu130

There are three versions of the stack ([DESIGN.md](DESIGN.md) has the details):

- **v0.1** quantized each operand with a separate kernel and saved BF16 activations for backward.
- **v0.2** added the fused dual quantizer (one read per operand), NVFP4-saved activations, the
  hardware E2M1 convert and optional fused wgrad accumulation.
- **v0.3** adds the TMA GEMM main loop, split-K for small output grids, and LayerNorm /
  GELU / GELU-backward fused into quantization.
- **v0.7** adds the weight-quantization cache and fused Llama-style blocks (§3.1–3.2).
- **v0.8** (current) makes the TMA GEMM persistent (the pipeline carries across output tiles, refills
  start the moment a stage is free), fuses the residual adds, biases and the q/k/v hand-off (GEMM
  epilogue, in-place multi-part quantization, a LayerNorm / RMSNorm backward kernel that adds the
  residual gradient) and halves the stochastic-rounding hash work. §1 and §3 are v0.8 numbers.
- **v0.4 – v0.6** did not change the kernels' speed: v0.4 hardened them (NaN propagation,
  checks, sanitizers), v0.5 moved the rounding seeds to the GPU and integrated the layers with
  `torch.compile`, CUDA graphs, DDP and FSDP2. v0.5 adds a three-seed quality study (§4.1).

Raw data sits next to each table: `bench/results/*.json` and `experiments/results/**`.
Every benchmark checks correctness against the fp64 reference before timing. Timings are the
median of 7 trials, each a batch of back-to-back launches (see `bench/bench_gemm.py::timeit` for why
per-launch syncing is misleading on a laptop GPU). This is one laptop at its power limit; expect
about 5–10% run-to-run variation, more for cuBLASLt (the same shape measured 334–418 TFLOPS on
different runs).

## 1. GEMM throughput (TFLOPS, `bench/results/gemm.json`)

| M × N × K | BF16 cuBLAS | **NVFP4, this repo (v0.8)** | v0.7 | v0.2 main loop (cp.async) | incl. quantizing both operands | NVFP4 cuBLASLt (`torch._scaled_mm`) | ours / BF16 | ours / cuBLASLt |
|---|---|---|---|---|---|---|---|---|
| 2048³ | 86 | **293** | 289 | 258 | 189 | 237 | 3.42× | 1.24 |
| 4096³ | 93 | **406** | 385 | 336 | 282 | 338 | 4.38× | 1.20 |
| 8192³ | 89 | **441** | 410 | 375 | 310 | 427 | 4.97× | 1.03 |
| 8192×8192×2048 | 99 | **392** | 364 | 321 | 321 | 215 | 3.95× | 1.82 |
| 8192×2048×8192 | 93 | **425** | 401 | 347 | 239 | 341 | 4.59× | 1.25 |
| 2048×8192×8192 | 92 | **432** | 407 | 356 | 244 | 334 | 4.69× | 1.29 |
| 8192×3072×768 | 87 | **275** | 277 | 256 | 180 | 139 | 3.14× | 1.98 |
| 8192×768×3072 | 92 | **397** | 390 | 323 | 122 | 334 | 4.33× | 1.19 |
| 768×3072×8192 | 85 | **414** | 393 | 341 | 134 | 393 | 4.86× | 1.05 |
| 768×768×8192 (split-K) | 84 | **273** | 266 | 218 | 128 | 214 | 3.26× | 1.28 |
| 2304×768×8192 | 95 | **313** | 308 | 276 | 19\* | 344 | 3.31× | 0.91 |

\* An outlier (it was in v0.7's run too): quantizing the two operands takes ~37 + 21 µs on their own,
so this column should read about 190 TFLOPS.

- **v0.8's persistent main loop** is up to 8% faster than v0.7 at the power limit (8192³: 441 vs 410, now
  above cuBLASLt's 427) and spends 8–17% fewer cycles at locked clocks; the accumulating wgrad GEMM
  (not in this table: `D += A·B^T` into an fp32 gradient) is 3–28% faster
  ([DESIGN.md §4.2](DESIGN.md)).
- **v0.3 closed the long-K gap to cuBLASLt** (v0.2 reached about 86% of it). How:
  [DESIGN.md §4.1](DESIGN.md).
- **The TMA loop is 7–25% faster than the v0.2 loop** on every shape it handles (K ≥ 1024, and from
  K = 512 when there are at least 8 output tiles per SM).
- **Split-K** lifts the 36-tile 768×768×8192 wgrad from 218 to 273 TFLOPS (+25%).
- **Against BF16:** 3.1–5.0× BF16 cuBLAS; against cuBLASLt NVFP4: 0.91–1.98×, below it only on
  2304×768×8192.
- **Locked-clock caveat:** at equal (profiler-locked) clocks cuBLASLt's kernel still takes fewer cycles
  (8192³: 3.65M vs 4.64M, tensor pipe ~90% vs 71%; v0.7: 5.10M, 64%). At the power limit, where
  training runs, that does not show ([DESIGN.md §4.2](DESIGN.md)).
- **Output agreement:** cuBLASLt and our output agree within bf16 rounding (max |Δ| / max |D| ≈ 0.4–0.7%).
  Ours matches the fp64 reference to about 1e-7.
- **Hardware ceiling:** NVIDIA's dense FP4 peak for the *desktop* RTX 5090 is 1676 TFLOPS. The
  laptop part has fewer than half its SMs and a 175 W board limit.

## 2. Quantizer (`bench/results/quantize.json`)

### 2.1 Fused dual quantizer vs the separate calls it replaces

Each case produces exactly the operand pair NVFP4Linear needs, so the outputs are bit-identical
(`tests/test_quantize_dual_cuda.py`). "Separate" is two single-output calls plus the row padding.

| Role in NVFP4Linear | Tensor | separate | **fused** | speedup |
|---|---|---|---|---|
| activation: X (RNE) + Xᵀ (RHT, padded to 128) | 8192 × 768 bf16 | 70.1 µs | **36.0 µs** | **1.95×** |
| | 8192 × 3072 bf16 | 362.4 µs | **157.3 µs** | **2.30×** |
| gradient: dY (SR) + dYᵀ (RHT + SR, padded) | 8192 × 2304 bf16 | 311.2 µs | **166.4 µs** | **1.87×** |
| | 8192 × 8192 bf16 | 1155 µs | **609 µs** | **1.90×** |
| weight: W + Wᵀ (16×16 2D blocks) | 3072 × 768 fp32 | 57.5 µs | **25.8 µs** | **2.23×** |
| | 8192 × 2048 fp32 | 290.3 µs | **182.5 µs** | **1.59×** |

v0.8's paired stochastic-rounding hash ([DESIGN.md §3.2](DESIGN.md)) made the gradient case 5–6%
faster than v0.7 (173 → 166 µs, 648 → 609 µs); measured in isolation, SR now adds 34–43% to the dual
quantizer instead of 48–61%, and 5–6% to the fused SwiGLU-backward quantizer instead of 9–11%.

### 2.2 Single-output quantizer

Effective GB/s counts one input read plus the NVFP4 output. The input is read twice (amax, then
quantize). An 8192 × 2048 bf16 tensor (32 MB) fits in the 64 MB L2, so its first row exceeds DRAM
bandwidth; the 8192² row (128 MB) is the DRAM-bound case.

| Tensor | rowwise | rowwise, 16×16 2D | transposed | transposed + RHT + SR (wgrad) | rowwise + SR |
|---|---|---|---|---|---|
| 8192 × 2048 bf16 | **1305 GB/s (33 µs)** — v0.1: 725 | 549 (v0.1: 470) | 613 (v0.1: 503) | 485 (v0.1: 254) | 709 (v0.7: 622) |
| 8192 × 8192 bf16 | 476 GB/s (361 µs) — v0.1: 488 | 198 (v0.1: 191) | 421 (v0.1: 406) | 367 (v0.1: 225) | 453 (v0.7: 408) |
| 768 × 8192 bf16 | 678 GB/s (24 µs) | 362 | 475 | 404 | 528 (v0.7: 467) |

Most of the gain over v0.1 came from the hardware `cvt.rn.satfinite.e2m1x2` convert, a
division-free stochastic-rounding path and per-CTA amax reduction (DESIGN.md §3.2–3.3).

### 2.3 LayerNorm and GELU fused into quantization (v0.3)

Device time of one d=768 / d=2048 block step (8192 tokens, profiler), same NVFP4 GEMMs, with and
without the producer fusion of [DESIGN.md §3.4](DESIGN.md):

| d_model | unfused | **LayerNorm / GELU / GELU-backward fused** | peak memory |
|---|---|---|---|
| 768 | 3.13 ms | **3.00 ms (−4%)** | 357 → 363 MiB |
| 2048 | 14.52 ms | **13.40 ms (−8%)** | 1229 → 1219 MiB |

Gone from the step: the LayerNorm forward kernels, the GELU and GELU-backward kernels, the bf16 copy
of the LayerNorm output, and the separate amax pass over each of those outputs.

## 3. Transformer-block training step vs width (`bench/results/train_step.json`)

This measures one pre-LN GPT block (qkv, SDPA attention, out-proj and a 4× MLP), forward and
backward, on 8192 tokens with BF16 autocast. The four linears either run in NVFP4 or stay BF16;
attention is BF16 SDPA in both. "NVFP4" is NVFP4Linear; "fused" adds the LayerNorm / GELU fusion and
fused wgrad accumulation (the configuration of the v0.3 training runs); "fused + residual" (v0.8) also
hands q / k / v over without a concatenation and fuses both residual adds (`chunks=3`,
`return_residual`, `residual=True`; [DESIGN.md §3.6](DESIGN.md)). Every variant is timed in two passes
in opposite orders and the best is kept (the power-limited GPU slows as it heats up). Peak memory is
the extra CUDA memory allocated during the step.

| d_model | BF16 step | NVFP4 | fused | **fused + residual** | **speedup** (fused / NVFP4) | linear TFLOPS, BF16 → NVFP4 | peak MiB, BF16 → fused + residual |
|---|---|---|---|---|---|---|---|
| 768 | 6.24 ms | 3.74 ms | 3.50 ms | **3.43 ms** | **1.82×** (1.78× / 1.67×) | 56 → 93 | 461 → 438 |
| 1024 | 10.30 ms | 5.49 ms | 5.05 ms | **4.92 ms** | **2.09×** (2.04× / 1.87×) | 60 → 113 | 624 → 588 |
| 2048 | 34.67 ms | 15.58 ms | 13.69 ms | **13.07 ms** | **2.65×** (2.53× / 2.23×) | 71 → 159 | 1441 → 1315 |
| 3072 | 79.57 ms | 29.18 ms | 27.62 ms | **25.07 ms** | **3.17×** (2.88× / 2.73×) | 70 → 191 | 2513 → 2248 |
| 4096 | 135.57 ms | 46.11 ms | 42.89 ms | **42.14 ms** | **3.22×** (3.16× / 2.94×) | 73 → 215 | 3841 → 3387 |

v0.7 measured 1.80× / 1.96× / 2.40× / 2.97× / 2.88× for "fused" at the same widths (v0.3: up to
3.10× at d=4096 in a cooler session); the v0.8 GEMM and quantizer changes lift every NVFP4 column.
The residual fusion uses 1–3% more peak memory than "fused" (the fp32 residual-stream gradient is
held while the norm backward runs).

- **Split-K's share** at d=768 (where the 768×768 projection wgrad splits): 3.64 ms with
  `NVFP4_GEMM_SPLITS=1` vs 3.60 ms auto, about 1% of the step.
- **Why the benefit grows with width:** GEMM time grows as d², while the quantization passes and the
  BF16 parts (attention, norms) grow more slowly.

### 3.1 Weight-quantization cache (v0.7, `bench_train_step.py`, `nvfp4_fused_cached`)

With `cache_weight_quantization=True` the NVFP4 copies of a weight are reused until an optimizer
step (or any other change) touches it, so under gradient accumulation every micro-batch after the
first skips the weight quantizer. The table is the steady state of such a micro-batch (same session):

| d_model | fused + residual | + weight cache | saved | speedup vs BF16 | peak MiB |
|---|---|---|---|---|---|
| 768 | 3.43 ms | 3.33 ms | 2.8% | 1.82× → 1.87× | 438 → 444 |
| 1024 | 4.92 ms | 4.77 ms | 3.0% | 2.09× → 2.16× | 588 → 594 |
| 2048 | 13.07 ms | 12.96 ms | 0.8% | 2.65× → 2.67× | 1315 → 1342 |
| 3072 | 25.07 ms | 23.82 ms | 5.0% | 3.17× → 3.34× | 2248 → 2310 |
| 4096 | 42.14 ms | 38.78 ms | 8.0% | 3.22× → **3.50×** | 3387 → 3496 |

The cached copies cost ~1.1 bytes per weight element.

### 3.2 Llama-style block (v0.7–v0.8, `bench/results/llama_block.json`)

One Llama-style block: RMSNorm, qkv, causal SDPA, o_proj, RMSNorm and a SwiGLU MLP (hidden ≈ 8/3·d,
rounded to 256), forward and backward on 8192 tokens with BF16 autocast. "NVFP4" converts the five
linears; "fused" uses `rmsnorm_linear` / `rmsnorm_swiglu_mlp` (RMSNorm and SwiGLU and its backward
inside the quantizers, one quantization of the normalized input shared by gate and up, fused wgrad
accumulation; the v0.7 API); "fused + residual" adds the v0.8 fusions (q / k / v chunks, residual
handle into the o_proj epilogue, `residual=True` MLP); "+ cache" adds the weight cache (micro-batches
after the first). Two passes in opposite orders, best kept.

| d_model (hidden) | BF16 step | NVFP4 | fused | **fused + residual** | **+ cache** | peak MiB, BF16 → NVFP4 → fused → + residual |
|---|---|---|---|---|---|---|
| 1024 (2816) | 11.05 ms | 6.48 ms (1.70×) | 5.68 ms (1.95×) | **5.29 ms (2.09×)** | **5.14 ms (2.15×)** | 688 → 648 → 601 → 618 |
| 2048 (5632) | 39.28 ms | 17.82 ms (2.20×) | 14.85 ms (2.64×) | **13.71 ms (2.87×)** | **13.12 ms (2.99×)** | 1541 → 1426 → 1330 → 1361 |
| 3072 (8192) | 80.83 ms | 32.03 ms (2.52×) | 28.75 ms (2.81×) | **26.39 ms (3.06×)** | **23.99 ms (3.37×)** | 2601 → 2374 → 2243 → 2291 |
| 4096 (11008) | 141.50 ms | 50.15 ms (2.82×) | 45.43 ms (3.11×) | **42.58 ms (3.32×)** | **38.91 ms (3.64×)** | 3965 → 3569 → 3395 → 3459 |

- v0.7 measured 1.88× / 2.60× / 2.83× / 2.96× for "fused" and up to 3.13× with the cache.
- The residual fusion saves 6–8% over "fused": at d=4096 it removes ~3.4 ms of the 0.7 step's 39.4 ms
  of GPU time (the q / k / v gradient stack and copy, both residual adds, the residual-gradient sums
  and a bf16 cast; [DESIGN.md §3.6](DESIGN.md)).
- The first v0.7 version was no faster than unfused: its RMSNorm backward was ~10 separate fp32
  PyTorch ops. v0.7 used PyTorch's fused RMSNorm-backward kernel; v0.8 replaces it with its own kernel
  that also produces dw and adds the residual gradient in the same pass.

## 4. End-to-end pretraining: GPT-124M (`experiments/results/`, `…/v0.2/`, `…/v0.3/`)

**Setup**

- **Model:** GPT-2-small-shaped, 124M parameters (d=768, 12 layers, 12 heads, context 512, tied
  embeddings).
- **Data:** TinyStoriesV2 (first 250 MB, 63.7M GPT-2 tokens).
- **Schedule:** 3000 steps × 16,384 tokens (49M tokens), AdamW (0.9, 0.95, wd 0.1), lr 6e-4 with
  cosine decay to 10%, 200 warm-up steps, gradient clipping at 1.0.
- **Controls:** the same seed, initialization and data order in every run. Validation uses 40
  fixed batches.

**Runs**

- **NVFP4 recipe:** every block linear (qkv, out-proj, fc1, fc2) trains in NVFP4, except in the last
  two blocks (40 of 48 linears). v0.2 adds fused wgrad accumulation; v0.3 also fuses LayerNorm/GELU.
- **Ablations (v0.1):** "all blocks" puts all 48 in NVFP4. "No RHT" disables the wgrad Hadamard transform.
- **Baselines:** the BF16 code path is the same in all versions; v0.3 reuses the v0.2 BF16 run.

![v0.3 loss curves](../experiments/results/v0.3/loss.png)

| Run | final val loss | Δ vs BF16 | median tokens/s | GPU energy / 1k tokens | wall time | peak memory |
|---|---|---|---|---|---|---|
| BF16 (autocast), v0.2 run | **1.4892** | — | 63,312 | 2.77 J | 797 s | 10.12 GB |
| **NVFP4 recipe, v0.3** | 1.5062 | +0.017 (+1.1%) | **80,526 (+27.2%)** | **2.17 J (−21.6%)** | **627 s** | **9.55 GB** |
| NVFP4 recipe, v0.2 | 1.5092 | +0.020 (+1.3%) | 78,727 (+24.3%) | 2.22 J (−19.8%) | 641 s | 9.55 GB |
| BF16 (autocast), v0.1 run | 1.4884 | — | 63,290 | 2.78 J | 798 s | 10.12 GB |
| NVFP4 recipe, v0.1 | 1.5064 | +0.018 (+1.2%) | 70,086 (+10.7%) | 2.49 J (−10.3%) | 718 s | 9.97 GB |
| NVFP4, all blocks, v0.1 | 1.5087 | +0.020 | 71,531 (+13.0%) | 2.44 J (−12.0%) | 703 s | 9.94 GB |
| NVFP4, no wgrad RHT, v0.1 | 1.5075 | +0.019 | 70,526 | 2.48 J | 714 s | 9.97 GB |

Peak GPU temperature was 76–78 °C in every run. GPU energy integrates the guard's power samples over
their actual timestamps; until v0.3 the integration assumed the first sampling interval for all
samples, which put the v0.1 savings at −14.5% instead of −10.3% (all rows above use the corrected
method).

**What this shows**

- **Training is stable.** NVFP4 tracks BF16 from the first steps to the end, with no instability,
  no loss spikes and no divergence.
- **The quality gap is constant.** In v0.3 it stays between +0.010 and +0.021 nats at every
  evaluation (v0.2: +0.017 to +0.020, v0.1: +0.016 to +0.019), so it doesn't grow with training.
  The fused LayerNorm/GELU path did not change it: differences between versions (0.003) are about
  what repeating the identical BF16 run gives (0.0008) plus different RHT/SR seeds.
- **v0.3 is 15% faster than v0.1 end to end** (70.1k → 80.5k tokens/s): +27% over BF16 with 22% less
  GPU energy. Most of the v0.2 → v0.3 gain is at larger widths (§5); at d=768 the BF16 LM head,
  attention and the two BF16 blocks dominate what is left.
- **At equal wall-clock time NVFP4 is ahead.** When the v0.3 NVFP4 run finished (627 s), BF16 was at
  ≈1.547 val loss (interpolated), 0.040 worse than NVFP4's 1.506.
- **Keeping the last two blocks in BF16** recovers about 0.002 of the gap, roughly 10% (v0.1 ablation).
- **The wgrad RHT is not measurable here.** At this scale its effect (0.001) is below what one run
  per configuration can resolve. NVIDIA reports that it matters for much larger models and longer
  runs, where outlier-heavy gradients appear. We don't claim a benefit we didn't measure.

### 4.1 Three seeds (v0.5, `experiments/results/v0.5_seeds/`)

The comparison above uses one seed. To check that the gap is not luck, the same setup was repeated
with seeds 0, 1 and 2 for both precisions (`experiments/run_v05_seeds.sh`; the seed changes the
initialization, the data order and the rounding seeds). NVFP4 runs use the v0.5 stack with fused
wgrad accumulation and fused producers; seed 0 of BF16 is the v0.2 BF16 run above.

![three seeds](../experiments/results/v0.5_seeds/seeds.png)

| seed | BF16 val loss | NVFP4 val loss | gap (same seed) | BF16 tokens/s | NVFP4 tokens/s |
|---|---|---|---|---|---|
| 0 | 1.4892 | 1.5068 | +0.0176 (+1.18%) | 63,295 | 80,624 |
| 1 | 1.4922 | 1.5112 | +0.0189 (+1.27%) | 63,018 | 80,575 |
| 2 | 1.4892 | 1.5100 | +0.0209 (+1.40%) | 63,045 | 80,524 |
| **mean ± std** | 1.4902 ± 0.0018 | 1.5093 ± 0.0023 | **+0.0191 ± 0.0016 (+1.28%)** | 63,119 | **80,574 (+27.7%)** |

- **The gap is reproducible:** +0.018 to +0.021 for every seed, a standard deviation of 0.0016 across
  seeds. It is about ten times the seed-to-seed spread of either precision alone, so it is a real,
  small cost of the format, not noise; and it is consistent with every earlier single-seed run
  (+0.017 to +0.020).
- **It stays constant during training** for all three seeds (right panel): after the first 1000
  steps it moves within +0.014 to +0.021 and does not grow.
- **Throughput is stable:** +27.7% tokens/s for NVFP4 on every seed (spread under 0.2%), the same as
  v0.3's single run, so v0.5's integration work cost no end-to-end speed.
- Peak GPU temperature 76–79 °C; each NVFP4 run took 630 s, each BF16 run 805 s.

## 5. End-to-end pretraining: GPT-507M, d=2048 (`experiments/results/v0.2_wide/`, `…/v0.3_wide/`)

This run tests the claim from §3 that wider models gain more.

**Setup**

- **Model:** 507M parameters (d=2048, 8 layers, 16 heads, context 512, tied embeddings).
- **Data and controls:** as in §4.
- **Schedule:** 1500 steps × 16,384 tokens (24.6M tokens), micro-batch 8 with 4-step gradient
  accumulation, lr 3e-4 with cosine decay, 150 warm-up steps.
- **NVFP4 runs:** the last block stays BF16 (28 of 32 linears in NVFP4), with fused wgrad
  accumulation; v0.3 also fuses LayerNorm/GELU.

![v0.3 wide loss curves](../experiments/results/v0.3_wide/loss.png)

| Run | final val loss | Δ vs BF16 | median tokens/s | GPU energy / 1k tokens | wall time | peak memory |
|---|---|---|---|---|---|---|
| BF16 (autocast) | **1.6786** | — | 19,983 | 8.76 J | 1246 s | 13.55 GB |
| **NVFP4 recipe, v0.3** | 1.6954 | +0.017 (+1.0%) | **32,328 (1.62×)** | **5.38 J (−38.6%)** | **770 s** | **12.63 GB** |
| NVFP4 recipe, v0.2 | 1.6937 | +0.015 (+0.9%) | 31,124 (1.56×) | 5.60 J (−36.1%) | 800 s | 12.63 GB |

- **The speedup grows with width as predicted:** 1.62× end to end at d=2048 vs 1.27× at d=768.
- **The gap does not grow:** +0.023 at step 250, then +0.007 to +0.019 to the end (v0.2: +0.025,
  then +0.014 to +0.015).
- **At equal wall-clock time NVFP4 is far ahead.** At 770 s, BF16 was at ≈1.897 val loss
  (interpolated) vs NVFP4's 1.695. BF16 needed 1246 s to reach a loss 0.017 lower.
- **The end-to-end gain is smaller than the block benchmark's 2.4×** because the micro-batch is
  4096 tokens (vs 8192 in §3) and the LM head, attention, embeddings, optimizer and last block stay BF16.
- **Stability note.** The first v0.3 attempt of this run stopped at step ~600 with a CUDA "unknown
  error", the WSL symptom of a GPU watchdog reset. The cause was a warp-convergence hazard in the TMA
  GEMM (one lane waiting on a barrier while its warp issued `.aligned` MMA instructions); it was fixed
  and every barrier wait is now bounded (DESIGN.md §4.1). The run above is the complete rerun.

**Caveats (§4 and §5)**

- One seed per configuration, one small dataset. The 507M run is far from compute-optimal (24.6M
  tokens, ~0.05 tokens per parameter) and its learning rate wasn't tuned.
- The gap could be narrowed with known techniques that this repo doesn't implement yet, such as
  BF16 "healing" for the last few percent of tokens.
- Results on a desktop RTX 5090, which has twice the SMs and three times the power budget, will
  differ.

## Reproduce

```bash
tools/setup_wsl.sh                       # CUDA 13.0 toolkit + PyTorch cu130 venv (WSL2/Ubuntu 22.04)
tools/check.sh bench                     # build, all tests, GEMM + quantizer benchmarks
python bench/bench_train_step.py
python experiments/prepare_tinystories.py
experiments/run_main.sh                  # v0.1 runs + ablations (§4)
experiments/run_v02.sh                   # v0.2 d=768 and d=2048 runs, incl. the BF16 baselines (§4, §5)
experiments/run_v03.sh                   # v0.3 NVFP4 runs (§4, §5)
python experiments/plot.py experiments/results/v0.3
```
