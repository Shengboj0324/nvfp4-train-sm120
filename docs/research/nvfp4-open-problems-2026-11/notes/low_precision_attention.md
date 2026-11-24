# Low-precision (8-bit / 4-bit) attention for training, forward and backward (state as of Nov 2026)

Scope note: "training" = the paper reports real fine-tuning or pretraining runs with a quantized backward or a quantized forward under training. "Inference-only" = the paper reports only forward-pass or post-training-quantization (PTQ) results. All numbers are as the authors report them unless stated otherwise. All speed claims are vendor or author benchmarks, not independent reproductions.

## Q1. SageAttention 1/2/2++/3 and SageBwd: how P is quantized, which backward GEMMs stay 16-bit, training accuracy, RTX 4090/5090 speed

### Takeaway
For the forward, the SageAttention series always gives P (the softmax numerator, in [0,1]) a fixed or row-max-derived scale that fills the low-precision range: 448 for E4M3 (SA2), 224 with V capped at 4.5 for FP16-accumulate FP8 (SA2++), rowmax/(448x6) two-level scaling for NVFP4 (SA3), and per-token rowmax/127 INT8 (SageBwd). Only SageBwd is a training method. It keeps dP = dO V^T in FP16, puts the other six attention GEMMs in INT8, matches BF16 on fine-tuning, and has a pretraining loss gap (0.054 at 2.1M tokens/step) that disappears only at a small tokens-per-step setting (260K) with QK-norm.

### Cited Findings
**SageAttention2 (inference-only)**
- P is quantized to FP8 E4M3 with a static scale: P is multiplied by 448 before the cast. V is quantized per-channel to E4M3, and the V scale is folded into the output — [SageAttention2, arXiv 2411.10958](https://arxiv.org/html/2411.10958)
- The FP8 mma `mma.f32.f8.f8.f32` on Ada and Hopper actually accumulates in "FP22" (1 sign, 8 exponent, 13 mantissa bits). The authors found this by probing with zero A/B inputs and varying the accumulator. Their fix is two-level accumulation: each block product P_ij V_j (about b_k = 64 tokens) goes through the FP22 mma, then is added into an FP32 running O in registers — [SageAttention2](https://arxiv.org/html/2411.10958)
- No before/after accuracy is reported for two-level accumulation alone. E4M3 P/V averages 99.44% cosine similarity vs 99.45% for FP16 PV (Table 7) — [SageAttention2](https://arxiv.org/html/2411.10958)
- Q/K quantization is INT4 or INT8 per-thread, grouped to match the mma fragment layout: 32 Q groups and 4 K groups per 128x64 block. This matches per-token accuracy (99.45% CosSim vs 98.03% per-block). K is smoothed by subtracting its token-wise mean. Q is smoothed by subtracting its block mean and adding back a rank-1 GEMV correction. Smoothing Q+K lifts average CosSim from 80.04% to 99.46% — [SageAttention2](https://arxiv.org/html/2411.10958)
- Peak 481 TOPS on RTX 4090. Speedup over FA2: 2.93x on 4090, 2.61x on H100. FA3-FP8 averages 41.53 vs 43.05 full precision on InfiniBench Llama-3-262k, while SA2 averages 43.06 — [SageAttention2](https://arxiv.org/html/2411.10958)

**SageAttention2++ (inference-only)**
- P V uses `mma.f16.f8.f8.f16` (FP8 inputs, FP16 accumulate), which the authors report as about 4x FP16-mma throughput, vs 2x for the FP32-accumulate FP8 instruction, on RTX 4090-class parts — [SageAttention2++, arXiv 2505.21136](https://arxiv.org/html/2505.21136)
- Overflow guard: 32 products accumulated per mma requires |32·p·v| ≤ 65504, i.e. Pr·Vr ≤ 2047. Two consecutive m16n8k32 results are accumulated in FP16 before being converted and added to FP32 ("delayed FP32 buffering"), which halves the bound to Pr·Vr ≤ 2047/2. The chosen values are Pr = 224 and Vr = 4.5 — [SageAttention2++](https://arxiv.org/html/2505.21136)
- Accuracy on CogVideoX layers is unchanged vs SA2 (CosSim 99.97%, L1 0.01862 for both). Llama3.1 WikiText perplexity is 6.013 (full precision) vs 6.020 (SA2++ 8+8). Speedup over FA2 on RTX 4090: about 3.9x (4+8) and about 3.0x (8+8). RTX 5090 numbers appear only in figures — [SageAttention2++](https://arxiv.org/html/2505.21136)

**SageAttention3 (FP4 forward is inference-only; SageBwd 8-bit is training)**
- Both QK^T and PV use NVFP4 (1x16 groups, E2M1 values, E4M3 scales, s = max|X|/6). Q/K are smoothed as in SA2 — [SageAttention3, arXiv 2505.11594](https://arxiv.org/html/2505.11594v3)
- Two-level P quantization: first a row scale s_P1 = rowmax(P)/(448×6), so that the per-16 E4M3 scale factors use their full range, then NVFP4 quantization with s_P2, then multiply the output by s_P1. Without it, P in [0,1] squeezes the FP8 scale factors into a narrow range. CosSim is 99.52% with two-level scaling vs 93.32% with direct quantization. NVFP4 beats MXFP4 (98.37%) — [SageAttention3](https://arxiv.org/html/2505.11594v3)
- 1038 TOPS on RTX 5090 vs FA2 at 214 TOPS (Table 18), described as about 5x. End-to-end inference speedups are about 3x on HunyuanVideo and 2.4x on CogVideoX — [SageAttention3](https://arxiv.org/html/2505.11594v3)
- SageBwd (in the SA3 paper): six of seven attention GEMMs are INT8. Forward QK^T and PV are INT8. In the backward, the recomputed S, dV, dQ and dK are INT8, and dP = dO·V^T stays FP16. Forward P is per-token INT8 with the scale taken from the online-softmax row max. V is per-block, and Q/K are per-block with K smoothing — [SageAttention3](https://arxiv.org/html/2505.11594v3)
- Why dP stays FP16: dP determines dS, and dS errors accumulate into dQ/dK along the sequence. dQ CosSim is 97.47% with dP in INT8 vs 99.77% with dP in FP16 (Table 1c) — [SageAttention3](https://arxiv.org/html/2505.11594v3)
- An INT8 backward is more accurate than an FP8 one (CogVideoX-2B). dQ: L1 0.0290 vs 0.0696, CosSim 0.9987 vs 0.9880. dK: CosSim 0.9993 vs 0.9910. dV: CosSim 0.9995 vs 0.9955 — [SageAttention3](https://arxiv.org/html/2505.11594v3)
- Fine-tuning matches BF16. Qwen2.5-3B GSM8K 0.601 (BF16) vs 0.607 (SageBwd), MMLU 0.640 vs 0.653. Llama3.2-1B GSM8K 0.259 vs 0.268. Pretraining a 400M Llama on FineWeb-Edu converges more slowly than BF16, and the authors say SageBwd is "not suitable" for pretraining — [SageAttention3](https://arxiv.org/html/2505.11594v3)
- Speed on RTX 4090: forward+backward up to 1.67x over FA2, forward about 2x, backward 1.2–1.6x. End-to-end Llama-1B training is about 1.15x faster at 8K/16K sequence length — [SageAttention3](https://arxiv.org/html/2505.11594v3)

**SageBwd follow-up paper (training; arXiv 2603.02170, March 2026)**
- Precision split: per-block INT8 for the recomputed S, dV = P^T dO, dQ = dS K and dK = dS^T Q. dP = dO V^T stays FP16/BF16. K smoothing is on by default. Q smoothing is optional; it needs a bias term dK_bias = (dS^T 1) μ_Q^T and showed no consistent benefit — [SageBwd, arXiv 2603.02170](https://arxiv.org/html/2603.02170)
- P is handled as follows. Forward: P̂ = P̃/s_P with s_P = exp(rowmax(S) − m)/127. Backward: P = exp(S − L) is recomputed from the stored lse and quantized per-block INT8. dS = P∘(dP − D) is computed in high precision, then quantized per-block INT8 — [SageBwd](https://arxiv.org/html/2603.02170)
- D = rowsum(dO∘O) uses the O from the INT8 forward. The paper does not discuss whether this is consistent with the recomputed P — [SageBwd](https://arxiv.org/html/2603.02170)
- Pretraining setup: 325M Llama, 4096 context, OpenWebText, 78B tokens, QK-norm. At 2.1M tokens/step: SageBwd 2.640 vs full-precision 2.586 (gap 0.054). At 260K tokens/step: 2.561 vs 2.563 (within noise). Without QK-norm at 2.1M tokens/step, training becomes unstable and diverges — [SageBwd](https://arxiv.org/html/2603.02170)
- Per-tensor errors (layer 11, cosine similarity / relative L2): P 0.9917/0.129, dS 0.9789/0.2045, dQ 0.9664/0.258, dK 0.9537/0.307, dV 0.9985/0.054, O 0.9969/0.079. dS is identified as the main bottleneck. RMS magnitudes: P ≈ 5e-3, dP ≈ 5e-5, dS ≈ 1e-7 — [SageBwd](https://arxiv.org/html/2603.02170)
- In a synthetic test, dQ/dK accuracy collapses as σ_Q = σ_K grows. At σ = 10, dQ CosSim is 0.78. O and dV stay accurate — [SageBwd](https://arxiv.org/html/2603.02170)
- Speed is reported for RTX 4090 only: up to 1.67x over FA2 for forward+backward. No 5090 or H100 numbers — [SageBwd](https://arxiv.org/html/2603.02170)

### Inferences
- Your measured +0.5% loss for a fully 8-bit backward at 4096 context is in line with SageBwd's large-batch result (0.054/2.586 ≈ +2.1% at 2.1M tokens/step, +0% at 260K). SageBwd itself keeps dP in 16-bit; that GEMM is the one to keep high precision if you go back to a partly 8-bit backward.
- The SA2++ FP16-accumulate trick matters on sm_89, where non-scaled FP8 with FP32 accumulate runs at half rate. On your sm_120 measurements, block-scaled E4M3 with FP32 accumulate (438 TFLOPS) is already faster than FP16-accumulate E4M3 (240). So the SA2++ trick looks unnecessary on sm_120 if you use kind::mxf8f6f4 — consistent with your choice.
- SA3's two-level P scaling (rowmax/(448·6)) exists because NVFP4 has E4M3 scales. With MXFP8 (E8M0 power-of-two scales per 32), the analogous issue is only exponent range, which P in [0,1] does not stress. The accuracy limit for MXFP8 P is the 3-bit mantissa (relative error about 2^-4), not the scale.

### Gaps
- No SageBwd numbers on RTX 5090 / sm_120, and no TOPS figures for the SageBwd backward kernel.
- The SageBwd paper is inconsistent about the K-mean definition (global vs token-wise) and about whether P V in the forward is per-token or per-block. I could not resolve this from the text.
- The SA2++ RTX 5090 speeds are given only in figures. I found no numeric values.

## Q2. FlashAttention-3 FP8, FlashAttention-4, INT-FlashAttention, Jetfire, 2025-2026 FP8 attention-training work: what fraction of the backward is low precision, and with what error

### Takeaway
FA3's FP8 is forward-only, and FA4 as published is BF16-only. The 2026 production-grade low-precision training attentions are all on datacenter Blackwell (tcgen05): Meta's LP-FA4 (MXFP8 forward and the entire backward, neutral production metric), FLAMIA (MXFP8 or NVFP4/FP8, 1B-model pretraining loss 2.397 vs 2.392 BF16), and cuDNN's per-tensor FP8 and MXFP8 SDPA backward. Independent results warn that naive FP8 backward (TE FP8 attention) can fail to train. The failures trace to the backward reading the forward's rounded O for Delta.

### Cited Findings
**FlashAttention-3 (inference forward)**
- FP8 uses "block quantization and incoherent processing". It reaches close to 1.2 PFLOPS on H100 with about 2.6x lower numerical error than a baseline FP8 attention — [FlashAttention-3, arXiv 2407.08608](https://arxiv.org/abs/2407.08608)
- FA3's FP8 attention does not support the backward pass, which limits it to inference — stated in [SageAttention3](https://arxiv.org/html/2505.11594)

**FlashAttention-4 (BF16; training-capable, not low precision)**
- arXiv 2603.05451 (Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao; March 2026; MLSys 2026). It reaches up to 1613 TFLOPS forward on B200 BF16 (about 71% utilization), 1.3x over cuDNN 9.13. The paper describes no FP8 support — [FA4, arXiv 2603.05451](https://arxiv.org/html/2603.05451); [Tri Dao blog](https://tridao.me/blog/2026/flash4/)
- Backward: a 2-CTA mode (M = 256, dS halves exchanged via DSMEM) halves dQ global atomics. A deterministic mode with a semaphore-serialised reduction reaches up to 75% of the speed of the non-deterministic mode — [FA4](https://arxiv.org/html/2603.05451)
- Targets datacenter Blackwell (B200/GB200). sm_120 is not mentioned — [FA4](https://arxiv.org/html/2603.05451)

**Meta "Low Precision FlashAttention-4" (training, production; blog 2026-09-23)**
- MXFP8 is used for both forward GEMMs and all five backward GEMMs (S, dP, dV, dK, dQ). P is converted from FP32 to MXFP8 online (32-element blocks), reusing the softmax row max for the block scale. They note that a constant scale for P "can further enhance" performance and stays numerically robust because P is in [0,1] — [PyTorch blog, LP-FA4](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/)
- Q, K and dO use square [32,32] blocks, so the E4M3 payload is transpose-invariant and one quantized copy serves both GEMM orientations. dS is quantized once per 32×32 block using `redux.sync.max.abs.f32` and feeds both dK and dQ. dQ is reduced in FP16 with a static scale of 2^9 — [LP-FA4 blog](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/)
- Speed on GB300 LLM shapes: forward 2.85 PF/s (cuDNN 9.24 MX8 2.82, FA4 BF16 2.00). Backward 1.82 PF/s with FP32 dQ and 1.98 PF/s with FP16 dQ, vs cuDNN BF16 1.50. On internal shapes: backward 1.58 PF/s vs 1.04 BF16 (1.52x) — [LP-FA4 blog](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/)
- Numerics: SQNR is 18.84 dB on the forward output and 25–28 dB on weight gradients. Production GEM training showed "neutral NE", but needed techniques inspired by SageAttention3 and by Qiu & Yao (2510.04212) — [LP-FA4 blog](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/)
- With cuDNN 9.24 / Frontend 1.26, the MXFP8 backward with non-uniform V scales did not match the reference for dQ/dK; uniform V scales removed the mismatch — [LP-FA4 blog](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/)

**FLAMIA (training; arXiv 2611.03344, Nov 2026; Velikonivtsev, Koltakov, Kuznedelev)**
- Two paths. The MXFP8 path uses E4M3 with token-block scales for all GEMMs. The mixed path uses NVFP4 for QK^T and its recomputation, and 8-bit operands for the rest. Outputs and gradients are BF16, and dQ partials accumulate in FP16. An all-4-bit path exists but is slower and reaches higher loss — [FLAMIA](https://arxiv.org/html/2611.03344)
- P̂ is E4M3 with a fixed multiplier 2^c, combined with conditional rescaling at threshold τ. They require c + τ ≤ log2 448, otherwise weights saturate "without a NaN or runtime error". dS is E4M3 with one scale per 32×32 block — [FLAMIA](https://arxiv.org/html/2611.03344)
- 1B Llama-3-style model, 100B FineWeb-Edu tokens, 2k context, 8×B300. Final loss (mean of last 1000 steps): BF16 2.392, MXFP8 2.397, NVFP4 2.406. Downstream mean: 0.5832 / 0.5737 / 0.5814 — [FLAMIA](https://arxiv.org/html/2611.03344)
- Causal leakage: a shared scale spanning future tokens can re-round earlier keys' codes. The fixes are scales spanning at most 32 positions, rounding the scale exponent up so nothing clamps, a causal dQ scale, and "exact" options that cost 8–11% in the backward — [FLAMIA](https://arxiv.org/html/2611.03344)
- Training call (forward + backward + quantization): 1.10–1.40x on B200 and 1.06–1.42x on B300 over the fastest baseline. Example: B200, 4×4096, MXFP8, 1079 TFLOPS. Only tcgen05/TMEM hardware; sm_120 is not addressed — [FLAMIA](https://arxiv.org/html/2611.03344)

**NVIDIA cuDNN SDPA FP8 backward**
- The per-tensor FP8 (E4M3) backward takes scalar descale/scale inputs and optional amax outputs for dQ/dK/dV/dP, and is marked experimental. Delta is rowsum(dO∘O) × descale_o·descale_dO, computed internally. The MXFP8 backward takes dO with both row- and column-wise scales and has no amax outputs — [cuDNN SDPA bwd sm107 docs](https://docs.nvidia.com/cudnn/operations/Attention/sdpa_bwd_sm107.md); [cuDNN frontend MXFP8 attention scaling](https://nvidia.github.io/cudnn-frontend/mxfp8-attention-scaling/)

**Independent evidence on FP8 backward failures (training)**
- In 390M runs with an emulated FP8 attention backward, reusing the forward's rounded O for D ended at loss 7.314, while a fresh rounding reached 3.256 (matching the control); all three pairs agree. Shipped Transformer Engine FP8 attention trained from scratch for 20k steps failed on all 3 seeds (validation loss 5.10–5.23 nats above BF16). Keeping the FP8 forward with a BF16 backward was 0.003–0.006 nats above BF16 — [Backward-State Policy, arXiv 2609.39813](https://arxiv.org/html/2609.39813). (Single-paper result; I did not find corroboration.)
- Qiu & Yao (ICLR 2026): BF16 flash attention training can explode because the BF16-accumulated Ō = P̄V is biased when the row max repeats (P̄ = 1 exactly) and V is mostly negative. This biases δ = rowsum(dO∘O), and the error accumulates through a low-rank structure. Their fix sets the max to β·r_m (β = 2; [2,8] works) when the max repeats, so all P̄ < 1. GPT-2 S stayed stable for 600K steps — [Qiu & Yao, arXiv 2510.04212](https://arxiv.org/html/2510.04212v4)
- Attn-QAT (4-bit QAT, BF16 STE backward): using the low-precision O in the backward causes exploding gradients (Wan-1.3B VBench 0.7185 vs 0.8252). A naive FP4 forward with a BF16 FA backward diverged — [Attn-QAT, arXiv 2603.00040](https://arxiv.org/html/2603.00040v3)

**INT-FlashAttention (inference / PTQ)**
- Per-token INT8 for Q/K/V in FlashAttention on RTX 4090 (Triton). About 72% faster than FP16 FA at 8k. Speed roughly equals FP8. Quantization error vs FP8 is 46% smaller under normally distributed activations and 82% smaller under uniform ones — [INT-FlashAttention, arXiv 2409.16997](https://arxiv.org/pdf/2409.16997)

**DeepSeek**
- DeepSeek-V3's FP8 training framework keeps the attention operators in their original (BF16/FP32) precision; FP8 is used for the linear GEMMs only — [DeepSeek-V3 Technical Report, arXiv 2412.19437](https://arxiv.org/abs/2412.19437), Section 3.3. (From the report's mixed-precision framework description; not re-fetched this session.)

### Inferences
- Fraction of the backward in low precision: SageBwd puts 4 of 5 backward GEMMs in INT8 (dP stays FP16). LP-FA4 and FLAMIA put 5 of 5 in MXFP8, but only on tcgen05 hardware, with square-block dS, BF16 outputs or FP16-reduced dQ, and careful scaling. All three report near-BF16 loss only with additional care: QK-norm or small tokens-per-step (SageBwd), consistent recomputation (FLAMIA), techniques from SA3 and Qiu & Yao (LP-FA4).
- The cleanest data point for your architecture is the Backward-State Policy result: an FP8 forward with a BF16 backward, where D comes from the original (unrounded) O, is within 0.006 nats of BF16. That is what your current design approximates.

### Gaps
- I found no evidence that Jetfire (arXiv 2403.12422) quantizes the attention core (QK^T / PV) in the backward. I did not fetch it; treat its coverage of attention as unverified.
- I found no DeepSeek 2025-2026 paper on FP8 attention training.
- FA4's software exp2 and conditional rescaling are documented, but I found no FA4 FP8 or MXFP8 path in the official repo; the LP-FA4 fork is Meta's.
- FLAMIA's remaining ~38K characters (appendix) were not read and may contain per-GEMM tables.

## Q3. GeForce-specific tricks (half-rate FP32-accumulate FP8, FP16-accumulate, split accumulation) and block-scaled MXFP8/NVFP4 mma for attention on RTX 50

### Takeaway
On Ada and RTX 4090, the documented tricks are SA2's FP22-then-FP32 two-level accumulation and SA2++'s FP16-accumulate FP8 with Pr·Vr ≤ 1023 range limits. On RTX 5090, block-scaled NVFP4 attention exists in SageAttention3 (1038 TOPS) and Attn-QAT (1.1–1.5x over SA3). No published consumer-Blackwell (sm_120, mma.sync) MXFP8 training attention was found. All MXFP8 training kernels (LP-FA4, FLAMIA, cuDNN) target tcgen05.

### Cited Findings
- FP8 `mma.f32.f8.f8.f32` on Ada/Hopper accumulates in FP22. SA2 adds the per-block (b_k ≈ 64) result into FP32 in registers — [SageAttention2](https://arxiv.org/html/2411.10958)
- `mma.f16.f8.f8.f16` is about 2x the FP32-accumulate FP8 rate on RTX 4090/5090-class parts. It is safe with Pr = 224, Vr = 4.5, and conversion to FP32 every two m16n8k32 mmas. Accuracy is unchanged (CosSim 99.97%) — [SageAttention2++](https://arxiv.org/html/2505.21136)
- SA3 runs NVFP4 microscaled QK^T and PV on RTX 5090 at 1038 TOPS, with two-level P scaling (rowmax/(448·6) then per-16 E4M3 scale). It is inference-only for FP4 — [SageAttention3](https://arxiv.org/html/2505.11594v3)
- Attn-QAT on RTX 5090 is about 1.1–1.5x the kernel throughput of SA3, attributed to dropping Smooth-QK and two-level P quantization once the model is QAT-trained. On B200, quantizing PV on the fly is slower than BF16 at long context because softmax is the bottleneck. NVFP4 QK + FP8 PV peaks at about 2026 TFLOPS on B200 (1.31x over FA4 BF16) and about 2677 TFLOPS on B300 (1.74x) — [Attn-QAT](https://arxiv.org/html/2603.00040v3)
- LP-FA4 notes that as MMA throughput rises (2x for MXFP8, 4x for MXFP4 vs BF16), the SFU-bound softmax stays fixed and becomes the exposed bottleneck — [LP-FA4 blog](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/)
- FLAMIA uses tcgen05/TMEM/2-CTA/CLC and does not address sm_120 — [FLAMIA](https://arxiv.org/html/2611.03344)

### Inferences
- Your rates (block-scaled E4M3 f32acc 438 vs INT8 473 vs NVFP4 914) mean the SA2++ FP16-accumulate path (240) is not competitive on sm_120 when kind::mxf8f6f4 is used. The GeForce half-rate penalty seems to apply only to non-block-scaled FP8 with FP32 accumulate. This is your measurement, not a published one.
- Attn-QAT's B200 observation (on-the-fly P quantization makes PV slower than BF16 at long context, because softmax binds) likely applies even more on sm_120. Your 25% forward cost for two-term P is consistent with P-quantization ALU work, not mma, being the limit.
- A FLAMIA-style 4-bit QK^T (NVFP4, 914 TFLOPS on your part) plus MXFP8 PV is the closest published recipe to yours. FLAMIA reports +0.014 loss at 1B with that mix vs +0.005 for all-MXFP8.

### Gaps
- I found no published MXFP8 or NVFP4 attention backward running on sm_120 / RTX 5090 (mma.sync `kind::mxf8f6f4` / `kind::mxf4nvf4`).
- I found no published measurement of the sm_120 FP8 FP32-accumulate penalty for block-scaled vs non-block-scaled mma, beyond your own numbers.
- The SageAttention repo's SA3 / SageBwd sm_120 kernel status (for example, whether a SageBwd 5090 kernel is released) was not checked on GitHub this session.

## Q4. Keeping Delta / logsumexp consistent between a quantized forward and the backward; and techniques for P V accuracy in one pass and for faster softmax/exp

### Takeaway
Every successful training method makes the backward's D = rowsum(dO∘O) equal, in expectation, to rowsum(P_bwd∘dP), where P_bwd is the P the backward recomputes. There are three published ways to do this:
- Attn-QAT stores a high-precision O′ = Σ P̃ V computed alongside the quantized O, at about 50% more forward FLOPs. This is the same idea as your two-term P.
- FLAMIA recomputes P bit-identically to the forward (same quantized Q/K, tiling, exp and scaling) and forms D with the same quantized dO used for dP.
- Backward-State Policy uses an unrounded O or a fresh, independent rounding of a BF16 recompute.

For exp, FA4 offloads 10–25% of exp2 to a degree-3 FMA polynomial and skips rescaling unless the max grows by more than 8 (log2 256). With fixed P scaling, FLAMIA requires c + τ ≤ log2 448.

### Cited Findings
**Delta / lse consistency**
- Attn-QAT: O′ accumulates Σ P̃·V^F with unquantized P̃ alongside the low-precision O. D = rowsum(dO∘O′) relies on P_iᵀdP_i = dO_iᵀO′_i, which "holds only when the forward and backward precisions match". The cost is about +50% forward FLOPs, about +25% attention memory, and a 1.2–2x wall-clock slowdown in training. P is not stored; it is recomputed from L and fake-quantized as in the forward. Skipping the fake-quant gave noisier gradient norms — [Attn-QAT](https://arxiv.org/html/2603.00040v3)
- FLAMIA: the backward recomputes P from the same quantized Q and K with saved row statistics, reproducing the forward's tiling, diagonal decisions, exponential evaluation and scaling. D = dO_i·O_i uses the same quantized dO as dP. Mismatches caused divergence: different dO for D and dP, or a forward-only approximate exp — [FLAMIA](https://arxiv.org/html/2611.03344)
- Backward-State Policy: D should equal rowsum(P∘dP). Reusing the forward's rounded O is biased because dO depends on that rounding. Reading the original O, or a fresh stochastic rounding of a BF16 recompute of O onto the same FP8 grid, gives 0.005–0.008 nats above BF16; reuse fails. For P, read the rounded P at dV = Pᵀ dO and the original P at the softmax backward. The paper proves no single shared substitute is unbiased at both uses — [Backward-State Policy](https://arxiv.org/html/2609.39813)
- SageBwd stores L = m + log l from the forward and computes D from the INT8-forward O without analysing consistency — [SageBwd](https://arxiv.org/html/2603.02170)
- cuDNN FP8 computes D internally as rowsum(dO∘O)·descale_o·descale_dO — [cuDNN docs](https://docs.nvidia.com/cudnn/operations/Attention/sdpa_bwd_sm107.md)
- Qiu & Yao: bias in O from BF16 rounding when P̄ = 1 exactly (repeated max) corrupts δ. The fix is a dynamic max m = β·r_m (β = 2) when the max repeats (or m = 0 if r_m < 0); this needs no backward change — [Qiu & Yao](https://arxiv.org/html/2510.04212v4)

**P V accuracy in one pass**
- Scale P to fill the range: ×448 for E4M3 (SA2), per-token rowmax/127 for INT8 (SageBwd), two-level rowmax/(448·6) for NVFP4 (SA3), constant 2^c with c + τ ≤ log2 448 (FLAMIA), or reuse the softmax row max for the 32-block MXFP8 scale (LP-FA4) — [SA2](https://arxiv.org/html/2411.10958); [SageBwd](https://arxiv.org/html/2603.02170); [SA3](https://arxiv.org/html/2505.11594v3); [FLAMIA](https://arxiv.org/html/2611.03344); [LP-FA4](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/)
- Smooth V (subtract the per-channel mean; add mean·rowsum(P) back): CogVideoX example 98.25% → 99.75% CosSim — [SageAttention2](https://arxiv.org/html/2411.10958)
- Stochastic or fresh rounding of the stored O (not of P) removes the D bias without a second P V pass — [Backward-State Policy](https://arxiv.org/html/2609.39813)

**Softmax / exp speed**
- FA4: Cody-Waite range reduction, with the integer part placed into the exponent bits and the fraction evaluated by a Horner polynomial on FMA units. 10–25% of each row's exponentials are emulated, the rest use MUFU.EX2. Degree 3 has max relative error 3.90e-3 at BF16 level (MUFU 3.89e-3) and is within 1 BF16 ULP of hardware on 99% of inputs. At FP32 level, degree 3 has 8.77e-5 error vs 1.41e-7 for MUFU and 1.44e-7 for degree 5. MUFU is 16 ops/clk/SM on B200 and 32 on B300 — [FA4](https://arxiv.org/html/2603.05451)
- FA4 conditional rescaling: rescale only if m_new − m_old > τ = log2(256) = 8. The final division by ℓ_final corrects for skipped rescales. A whole warp rescales if any thread needs it — [FA4](https://arxiv.org/html/2603.05451)
- LP-FA4: reuse the softmax row max for the MXFP8 P block scale (monotonic exp), costing 1 extra exp per 32 elements. The scale is computed from exponent/mantissa bits (no log2 or division). Packed `fmul2` and a 3-input max are used — [LP-FA4 blog](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/)
- FLAMIA's NVFP4 path quantizes P in the log domain, taking each 32-key group's max before exponentiation — [FLAMIA](https://arxiv.org/html/2611.03344)

### Inferences
- Your two-term-P forward is the Attn-QAT O′ approach implemented more cheaply: about 25% forward cost vs their 1.2–2x. Two cheaper published alternatives:
  1. Store O computed from the one-pass E4M3 P, but make D unbiased by forming it from a fresh, independent rounding of O (Backward-State Policy). That paper's result covers an FP8 forward with a BF16 backward, which is very close to your setup. Note that it still needs an O close to the unrounded-P O, so it does not remove the need for a high-precision O by itself.
  2. Make the BF16 backward recompute P exactly as the forward rounded it (E4M3-rounded P from the same lse), and compute D = rowsum(dO∘O_q) from the quantized-P O. That is FLAMIA-style consistency. With a PyTorch BF16 flash backward, P is recomputed unrounded, so this needs a custom backward that rounds P to E4M3 before both dV and the dS formula. Backward-State Policy argues that using rounded P in dS is biased, so this option conflicts with that paper. Choosing between them needs an experiment.
- If 16 ops/clk/SM MUFU also holds on sm_120 (unverified), then at your INT8/MXFP8 mma rates the exp is likely the binding unit for head dim 64/128. FA4's FMA-polynomial offload (degree 3, 10–25% of elements) plus τ = 8 conditional rescaling are the directly portable tricks, since they need only FMA/ALU and no tcgen05. With a fixed P scale and lazy rescaling, keep c + τ ≤ log2 448 ≈ 8.8 (FLAMIA). For example, with τ = 8 the P multiplier can be at most 2^0.8.
- Qiu & Yao's repeated-max fix (all P̄ < 1) is cheap and also prevents the P̄ = 1 exact-value bias in any low-precision O accumulation.

### Gaps
- The MUFU.EX2 throughput on sm_120 / RTX 5090 was not found in a primary source.
- No paper measures the speed/accuracy of a one-pass MXFP8 P V with only a corrected D (no second P V term) on GeForce.
- The full text of two unfetched polynomial and approximate-softmax papers, [Fast Polynomial Transcendentals for LLMs, arXiv 2610.00049](https://arxiv.org/pdf/2610.00049) and [Approximating Softmax in Pretrained LLMs, arXiv 2609.33586](https://arxiv.org/pdf/2609.33586), was not read. They may have more exp-speed data.
- A personal GitHub survey of low-bit FlashAttention dated 2026-11-20 exists ([yiliu30 issue #11](https://github.com/yiliu30/yiliu30/issues/11)) but was not read and is not a primary source.
