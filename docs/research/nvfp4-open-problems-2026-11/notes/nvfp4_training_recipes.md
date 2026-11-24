# Recipes that close the NVFP4 (FP4) pretraining accuracy gap vs BF16 (state as of Nov 2026)

Reference point for the reader: the library's current recipe is NVFP4 fprop/dgrad/wgrad, RHT16 on wgrad operands only, SR on gradients, RTN on W/X, **1x16 scaling for everything (including weights)**, and the last 2 blocks in BF16. On GPT-124M / TinyStories / 49M tokens it shows +1.28% val loss (0.019 nats), constant over training. That is structurally the NVIDIA-2025 recipe minus 2D weight scaling, which several of the findings below address directly.

Chronology / supersession: NVIDIA NVFP4 recipe (Sep 2025), then TetraJet-v2 (Oct 2025, ICML 2026), then Four Over Six (Dec 2025), then Quartet II / MS-EDEN (Jan 2026), then Full-Stack FP4 (Jul 2026), UFP4 (Jun 2026), CHON (Feb 2026), then "What Matters for NVFP4 Training?" / Mosaic NVFP4 (Oct 2026, NVIDIA + IST + others; the latest consolidated recipe). Quartet v1 (May 2025, MXFP4), FP4-All-the-Way (May 2025), QuEST and HALO predate native NVFP4 work and are **superseded for NVFP4 purposes**.

---

## Q1. NVIDIA "Pretraining LLMs with NVFP4" and the Transformer Engine NVFP4 recipe: which pieces, and what each one does in ablation

### Takeaway
NVIDIA's recipe has four parts: (1) about 15% of linear layers kept in BF16, mostly at the end; (2) RHT16 on wgrad inputs only; (3) 2D 16x16 weight scales, with 1x16 for activations and gradients; (4) SR on gradients and RTNE on W/X. An optional late switch to BF16 closes most of what remains. Most of the residual gap comes from **forward-pass** quantization. The ablations are reported almost entirely as plots, so exact deltas per component mostly aren't available as numbers.

### Cited Findings
- Recipe summary: keep some linear layers in high precision (<15%, mostly at the end); 16x16 RHT on Wgrad inputs only; 2D 16x16 block scaling for weights and 1x16 for activations and gradients; SR for gradients and RTNE for weights and activations (Sec. 4). — [NVIDIA, arXiv 2509.25149v2](https://arxiv.org/html/2509.25149v2)
- Scale math (App. B): global encode scale s_enc = (6·448)/amax_tensor, with decode 1/s_enc stored as FP32. Block decode s_dec,b = amax_b/6, and E4M3 block scale = e4m3_RTNE(s_dec,b · s_enc). — [NVIDIA, arXiv 2509.25149v2](https://arxiv.org/html/2509.25149v2)
- Layers kept in BF16: 12B model, first 2 plus last 8 blocks (16% of linears). Fig. 4 suggests the last 4 blocks alone are enough for stability. 8B model: last 8 blocks (about 15%). The 1.2B ablation model (20 blocks) converges with the last 4 in BF16. Keeping only the first blocks does not help unless the last blocks are also kept. Training is stable with the first 2 plus last 2 in BF16 but not with the first 4 alone. Embeddings, output head, norms, nonlinearities and all attention-core math (softmax, QK^T, PV) stay BF16/FP32. Master weights, wgrads and optimizer states are FP32. — [NVIDIA, arXiv 2509.25149v2](https://arxiv.org/html/2509.25149v2)
- 2D weight scaling: consistent fprop and dgrad weights improve 12B training loss (Fig. 4). On 1.2B, 1x16 scales along different dims in fwd and bwd ("option 2", **which is what our library currently does for W**) were worse than 16x16 ("option 3"), even though the 16x16 blocks are coarser. Activations are less sensitive to this inconsistency, which mainly shows up during late LR decay. — [NVIDIA, arXiv 2509.25149v2](https://arxiv.org/html/2509.25149v2)
- RHT: wgrad RHT improves 1.2B val loss (Fig. 11). RHT on Fprop or Dgrad degrades quality, with "no measurable benefit" at smaller scales. Hadamard size 2/4/16/128 made virtually no difference at 1.2B. At 12B, 4x4 hurt and 128x128 gave a minor benefit, so 16 was chosen. Removing the random sign vector hurts. A per-transform seed gives no gain over one fixed seed. — [NVIDIA, arXiv 2509.25149v2](https://arxiv.org/html/2509.25149v2)
- SR: at 12B, SR on gradients is "essential for convergence" and must be applied to the gradients feeding both Dgrad and Wgrad. At 1.2B/1T tokens, SR on gradients is stable, but **SR on activations or weights diverges**. SR on all backward tensors adds little over SR on gradients alone. — [NVIDIA, arXiv 2509.25149v2](https://arxiv.org/html/2509.25149v2)
- Base case: all layers NVFP4 with RTNE diverges early, and SR alone also diverges. Only the combination converges (Fig. 8, no table numbers). — [NVIDIA, arXiv 2509.25149v2](https://arxiv.org/html/2509.25149v2)
- Late switch to BF16: on 12B/10T tokens, switching at 8.2T tokens (the last ~18%) matches FP8 loss, and switching at 10T is only slightly worse. **Switching only the forward pass cuts the relative loss error from 1.5% to 0.5%**. The higher-precision phase is about 6% of total compute. — [NVIDIA, arXiv 2509.25149v2](https://arxiv.org/html/2509.25149v2)
- Headline gap: 12B/10T tokens has <1% relative loss gap vs FP8 during the stable LR phase, widening to slightly >1.5% during decay. MMLU-Pro 62.58 (NVFP4) vs 62.62 (FP8). — [NVIDIA, arXiv 2509.25149v2](https://arxiv.org/html/2509.25149v2)
- NVFP4 vs MXFP4 (8B, 1T tokens): about 1.5% gap for NVFP4 vs about 2.5% for MXFP4 (with d=32 RHT on wgrad). MXFP4 needs about 36% more tokens (1.36T) to match NVFP4 loss. — [NVIDIA, arXiv 2509.25149v2](https://arxiv.org/html/2509.25149v2)
- Transformer Engine `NVFP4BlockScaling`: 2D weight quantization and RHT are on by default. Flags are `disable_rht`, `disable_2d_quantization` (forces 1D for weights, since activations and grads are always 1D) and `disable_stochastic_rounding` (SR is on for gradients only). RHT is applied to the columnwise (wgrad-operand) quantization of inputs and gradients. — [TE 2.13 docs](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.13/user-guide/features/low_precision_training/nvfp4/nvfp4.html), [TE 2.15](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.15/user-guide/features/low_precision_training/nvfp4/nvfp4.html), [TE 2.16](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.16/user-guide/features/low_precision_training/nvfp4/nvfp4.html), [TE 2.18](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.18/user-guide/features/low_precision_training/nvfp4/nvfp4.html) (search-snippet level; flag semantics not verified against source code)
- Production use (Nemotron 3): QKV and attention projections kept in BF16. Mamba output projections kept in MXFP8 because NVFP4 had "high flushes to zero (up to 40% on Nano)". Latent projections and MTP layers in BF16. Last 15% of the network in high precision. Gap <1% for Nano (A3B) and <0.6% for the 8B-active MoE, shrinking with scale. No late precision switch in the main recipe. An ablation that re-quantized the sensitive layers to NVFP4 from a 500B-token checkpoint showed that keeping them in high precision matters. — [Nemotron 3, arXiv 2512.20856](https://arxiv.org/html/2512.20856)

### Inferences
- Our library keeps **1x16 weight scaling in both fprop and dgrad**, which NVIDIA's 1.2B ablation identifies as the worse "option 2". That leaves two candidate fixes: 2D 16x16 weight scales (NVIDIA), or 1D weights with backward re-quantization of the already-quantized weight (Q3, Quartet II / Mosaic). Both are cheap to try.
- "Most of the gap is in the forward pass" (1.5% to 0.5% from switching only fprop) matches TetraJet-v2's sensitivity analysis (Q3). Forward-side techniques (4/6, outlier handling, late fprop switch) should therefore be first in line for the 0.019-nat gap.
- A late switch of the forward pass to BF16 (the last ~10-20% of tokens) is the cheapest guaranteed reduction. It changes the deliverable from "NVFP4 throughout" to "mostly NVFP4", though.

### Gaps
- Exact numeric deltas per ablation (2D vs 1D weights, RHT placement, SR placement) are only in figures in 2509.25149. I could not extract numbers.
- I did not verify in TE source whether newer TE releases (2.16-2.18) added 4/6, MS-EDEN or dgrad-RHT options. The TE NVFP4 page at the unversioned URL returned 404.

---

## Q2. "Four Over Six" (4/6) adaptive block-scale selection

### Takeaway
For each 16-element block, quantize twice, once with the block max mapped to 6 (standard) and once mapped to 4, and keep whichever has lower MSE. The tensor scale must use 256 instead of 448 so the 1.5x larger block scale fits in E4M3. Overhead is <15% of the quantize op. On a 30B-A3B / 1T-token run it brings loss 22.3% closer to BF16 than the NVIDIA recipe. It is the single most-adopted 2026 add-on, used by Quartet II, Mosaic and Full-Stack FP4. It is NVFP4-only, because E8M0 cannot express the 1.5x ratio.

### Cited Findings
- Mechanism: E2M1 has a gap between 4 and 6. Mapping the block max to 4 gives finer relative resolution near the top. Example: block [10,20,30,40] scaled to 6 puts 30 at 4.62, which rounds to 4 (13.4% error). Scaled to 4, 30 lands at exactly 3. — [Four Over Six, arXiv 2512.02010](https://arxiv.org/abs/2512.02010)
- Selection rule: both candidates get E4M3-rounded scales. The authors compared max-error, L1 and MSE criteria, and **MSE works best** and is adopted. Scaling every block to 4 is worse than baseline, so the choice must be adaptive. — [4/6 HTML v3](https://arxiv.org/html/2512.02010v3)
- Per-tensor scale change: replace 448 with **256** in the global scale (6·256/amax instead of 6·448/amax). Otherwise a block mapped to 4 would need an E4M3 scale of 448·6/4 = 672 (overflow). With 256 the maximum needed is 384. — [4/6 HTML v3](https://arxiv.org/html/2512.02010v3)
- Training usage: applied to W, X and G for all three GEMMs. In pretraining it was combined with SR (SR is computed under both scales and the lower-error result kept), RHT on wgrad, and 2D weight blocks. Nemotron-3-Nano architecture (30B total / 3B active, 52 blocks), 1T tokens, 384 B200. **Training loss 22.3% closer to BF16 than the NVIDIA NVFP4 recipe**; absolute values only in Fig. 4. — [4/6 HTML v3](https://arxiv.org/html/2512.02010v3)
- Cost: PTX `cvt`-based CUDA kernel keeping both candidates and errors in registers, **<15% overhead on the NVFP4 quantization op** (not the GEMM). — [4/6 HTML v3](https://arxiv.org/html/2512.02010v3); code at github.com/mit-han-lab/fouroversix (per [arXiv abstract](https://arxiv.org/abs/2512.02010))
- MXFP4 incompatibility: E8M0 scales cannot represent the 1.5x ratio between candidate scales. — [4/6 HTML v3](https://arxiv.org/html/2512.02010v3)
- Independent MSE measurement on N(0,1) (x1e-3): RTN 1x16 = 9.0, RTN+4/6 1x16 = 7.6 (-16%), RTN 16x16 = 12.4, RTN+4/6 16x16 = 12.4 (**no gain with 2D blocks**), SR 1x16 = 23.5, SR+4/6 = 17.5. — [Quartet II, arXiv 2601.22813](https://arxiv.org/html/2601.22813)
- **Conflict:** Quartet II states that 4/6 combined with SR **introduces bias** (verified in their App. A), while the 4/6 paper uses SR+4/6 together without flagging bias. — [Quartet II](https://arxiv.org/html/2601.22813) vs [4/6 HTML v3](https://arxiv.org/html/2512.02010v3)
- Quartet II ablations (Llama 30M-200M on C4): 4/6 helps both square-block and native 1x16 scaling, but the gain is **roughly twice as large with native 1x16 scales**. — [Quartet II](https://arxiv.org/html/2601.22813)
- In Quartet II's Nanochat table, the 4/6 recipe alone shows a 1.51% val-BPB gap vs NVIDIA's 1.56% at 560M/11B tokens, and 1.76% vs 1.92% at 1.9B/38B. — [Quartet II](https://arxiv.org/html/2601.22813)
- Full-Stack FP4 treats "NVFP4 4/6" as its baseline: a 1B/20B-token run with all blocks quantized gives +1.40% loss gap vs BF16. — [Full-Stack FP4, arXiv 2607.04422v2](https://arxiv.org/html/2607.04422v2)

### Inferences
- For our setup (1x16 weights, RTN forward), 4/6 on forward W and X is the most direct cheap win. Quartet II shows the benefit is largest exactly in 1x16 mode, and the change is local to the quantize kernel plus the 448 to 256 tensor-scale constant.
- Applying 4/6 to gradients that are also SR-rounded should be avoided, or at least tested carefully, given Quartet II's bias result. Keep 4/6 on RTN-quantized tensors (W, X, and W^T/X^T in backward).
- If we move to 2D 16x16 weight scales, 4/6 gives no MSE gain on those weights. 1D weights + re-quant + 4/6 is likely the better combination, and this is what Mosaic recommends (Q3).

### Gaps
- No public absolute-loss table for 4/6 pretraining at <1B scale from the 4/6 authors. Small-scale evidence comes only via Quartet II and Full-Stack FP4.
- No independent replication of the 22.3% figure found.

---

## Q3. Quartet, Quartet II (MS-EDEN), QuEST, HALO, TetraJet / TetraJet-v2, FP4-All-the-Way, Mosaic NVFP4, and other 2025-2026 methods

### Takeaway
The current best published NVFP4 recipe is **Mosaic NVFP4** ("What Matters for NVFP4 Training?", Oct 2026). It uses 1D weight quantization with backward re-quantization, dim-128 tiled RHT on **both** wgrad and dgrad with resampled signs, **MS-EDEN instead of SR on gradients**, 4/6 on all other tensors, and the last 15% of layers in BF16. That gives 0.24-0.36 pp lower loss gap than NVIDIA-2025 at 8B-30B scale, with the largest single gain from MS-EDEN plus 4/6 in backward. Alternatives with large reported gains at small scale are TetraJet-v2 (OsciReset + OutControl, about 51% gap reduction at 70M-370M), weight de-oscillation (halves the gap in AMD's MXFP4 MoE work), and Full-Stack FP4's LoRA-SVD BF16 subspace (1.40% to 0.61% at 1B).

### Cited Findings

**Quartet II / MS-EDEN (Jan 2026, IST-DASLab) — current**
- MS-EDEN, the backward quantizer: split into 128-element chunks, apply RHT (a shared seed so rotations cancel in the GEMM inner dim), RTN to NVFP4 with grid scale s = (1/0.93)·6·16/17 (allows clipping), and cap FP8 scales at 256 instead of 448 to leave headroom. Then compute a per-16-element correction S = <x_RHT, x_RHT> / <x_RHT, x_RTN> and fold it into the E4M3 group scale **using SR on the scales only**. The result is unbiased in expectation. MSE is 9.4e-3 vs 23.5e-3 for element-wise SR (>2x lower), with N(0,1) inputs. — [Quartet II](https://arxiv.org/html/2601.22813)
- Forward: RTN, native 1x16 scaling, and 4/6 on W and X, with no forward Hadamard. Backward: the saved quantized W and X are dequantized, transposed and **re-quantized** with MS-EDEN, with RHT on the inner dim in groups of 128. — [Quartet II](https://arxiv.org/html/2601.22813)
- Nanochat val-BPB gap vs BF16: 560M/11B tokens gives NVIDIA 1.56%, 4/6 1.51%, TetraJet-v2 (GPU-feasible subset) 1.72%, **Quartet II 1.26%**. 1.9B/38B tokens gives NVIDIA 1.92%, 4/6 1.76%, TetraJet-v2 1.72%, **Quartet II 1.44%**. The authors claim 15-25% gap reduction vs prior NVFP4 recipes. — [Quartet II](https://arxiv.org/html/2601.22813)
- Ablations (Llama 30M-200M, 25-800 tokens/param): MS-EDEN beats SR wherever both apply. Square 16x16 weight blocks reduce forward representation quality relative to 1x16 + re-quant. — [Quartet II](https://arxiv.org/html/2601.22813)
- Kernels: RTX 5090, >4x linear-layer speedup over BF16. End-to-end 1.1B model at 52 vs 28 ktok/s (1.85x). On B200, gains start around 3B (2.5x at 22B). Code at github.com/IST-DASLab/Quartet-II. — [Quartet II](https://arxiv.org/html/2601.22813)

**Mosaic NVFP4 / "What Matters for NVFP4 Training?" (arXiv 2610.24343, Oct 2026; Agrusa, Panferov, ..., Blankevoort, Alistarh) — current, latest**
- Incremental updates over NVIDIA-25 (Table 1): (1) 1D weight quantization with backward re-quantization; (2) dim-128 tiled RHT on WGRAD **and DGRAD**, resampled sign vectors; (3) MS-EDEN instead of SR on gradients, with 4/6 on the remaining backward tensors (X^T, W^T); (4) 4/6 on forward tensors (W, X). The last 15% of layers stay BF16. — [What Matters, arXiv 2610.24343](https://arxiv.org/html/2610.24343v1)
- Results: Nemotron 3 Nano (30B-A3B, 1T tokens) reaches <0.81% relative train-loss gap, 0.36 pp better than NVIDIA-25. Nemotron-H 8B (1T tokens) reaches <1.2%, 0.24 pp better. Seed noise is about 0.03-0.06%. **The largest single gain is MS-EDEN + 4/6 on backward tensors.** — [What Matters](https://arxiv.org/html/2610.24343v1)
- With all layers quantized, the 8B model diverges past 300B tokens even with Mosaic. The authors interpret this as a representation failure that error reduction cannot fix. With the last 15% in BF16, the backward variants become indistinguishable. **Recipe rankings do not stabilize until well past 150B tokens**, and weaker recipes can look better early. — [What Matters](https://arxiv.org/html/2610.24343v1)
- The paper contains no small-scale (100M-1B) experiments and no compute-cost numbers. — [What Matters](https://arxiv.org/html/2610.24343v1)

**TetraJet-v2 (arXiv 2510.27527, ICML 2026, Tsinghua) — current**
- Recipe: NVFP4 everywhere. Double-block scaling uses 1x16 inner blocks in 1x128 outer blocks plus a global scale (1x128 outer beat per-tensor and per-row). Weights use 1x16, **not 16x16**. RTN forward and SR backward. RHT block size 32 on both backward GEMMs (dX and dW). — [TetraJet-v2](https://arxiv.org/html/2510.27527v1)
- OsciReset: oscillation risk = |movement of quantized weight| / |movement of master weight| over a window. Weights with risk >8 are reset to the center of their quantization bin. Period is 200 steps, with stats over the first 50 steps of each period. It starts at step 8k/15k/35k for 70M/150M/370M. Q-EMA, Freeze and Dampen all severely hurt LLM training. — [TetraJet-v2](https://arxiv.org/html/2510.27527v1)
- OutControl: outlier channels chosen statically at the start of training by largest norm, with 10% kept in FP8 (a 5% BF16 option is also mentioned), applied in both fwd and bwd. — [TetraJet-v2](https://arxiv.org/html/2510.27527v1)
- C4 val PPL (BF16 / NVIDIA / TJv2-base / TJv2-full): 70M 45.27 / 50.94 / 49.33 / 47.75; 150M 33.49 / 36.73 / 35.88 / 34.95; 370M 23.70 / 26.20 / 25.50 / 25.11. That is about a 51.3% average gap reduction. OLMo-2 trained on about 52B / 107B / 212B tokens. — [TetraJet-v2](https://arxiv.org/html/2510.27527v1); [PMLR](https://proceedings.mlr.press/v306/chen26cl.html)
- Ablations at 150M: weight blocks 1x16 gives 31.49 PPL vs 16x16 at 31.74. Hadamard placement: none 28.67, forward 28.89 (worse), dW only 28.47, **dX+dW 28.42 (best)**. OutControl: none 31.49, fwd 31.41, fwd+bwd 31.28, random channels 31.47. Gradient unbiasedness: NVIDIA recipe 32.42 vs TJv2-base 31.49. — [TetraJet-v2](https://arxiv.org/html/2510.27527v1)
- Sensitivity at 370M/50B tokens (PPL increase over BF16 24.06): forward activations only +0.43 (**largest single source**), forward weights +0.22, backward dX+dW +0.21, everything +0.83. Keeping mlp ffn2 (down-proj) in BF16 recovers 0.35 and ffn1 recovers 0.29. Attention modules matter less. Switching to FP6xFP4 for the last 50B tokens gives a "substantial" improvement (simulation). — [TetraJet-v2](https://arxiv.org/html/2510.27527v1)
- Caveat: the authors had no NVFP4 hardware, so all results are simulated and the 1.67x speedup claim is unverified. — [TetraJet-v2](https://arxiv.org/html/2510.27527v1)

**Full-Stack FP4 (arXiv 2607.04422v2, Aug 2026, CAS) — current**
- LoRA-SVD projections: W = W_res(NVFP4) + β·L2·L1(BF16). The top-r subspace is realigned every 2048 steps by randomized SVD. Rank is 128 for down_proj and 64 for gate_up/qkv/out_proj. Parameter cost is +4.32% at 1B. Realignment overhead is about 0.02-0.03% of step time. — [Full-Stack FP4](https://arxiv.org/html/2607.04422v2)
- 1B/20B-token ablation (BF16 2.6415): NVFP4+4/6 baseline +1.40%; +plain LoRA +1.12%; **+LoRA-SVD +0.61%**; +Q-AdamW +0.98%; **+Q-Attn (FP4 QK^T and dS-path, BF16 PV) +0.47%** (reported as the largest single reduction, oddly for a component that quantizes more). Full-stack without Root is +0.56%. 3B/64B tokens gives 2.267 vs 2.286 (0.838%). Quality runs used fake quantization. — [Full-Stack FP4](https://arxiv.org/html/2607.04422v2)

**UFP4 / shrinkage bias (arXiv 2606.20381, Jun 2026, Ant Group)**
- E2M1 RTNE has a systematic negative bias from asymmetric bins (e.g. level 2 has a conditional bias of -0.125, and level 4 the same at double scale). The authors argue it compounds across layers and that RHT worsens it under E2M1. — [UFP4](https://arxiv.org/html/2606.20381v1)
- Uniform E1M2/INT4-style grid with RHT16 on fwd_y, bwd_dx and bwd_dw, plus SR on dY. Relative loss error over the last 1000 steps, E2M1 ref vs UFP4: dense 1.5B 1.257% vs 0.967%; MoE 7.9B 2.360% vs 1.847%; MoE 124B 1.731% vs 1.386%. Not runnable on NVFP4 tensor cores (format change). The E2M1 range-restriction workaround underperformed. Fused RHT+quant costs about 1.06-1.07x standalone quant. — [UFP4](https://arxiv.org/html/2606.20381v1)

**CHON / Hot-Channel Patch (arXiv 2602.02047, Feb 2026)**
- Outliers come from softmax (SA), gating (LA) and SwiGLU (FFN), and evolve from transient spikes into persistent "hot channels". Post-QK operations are most sensitive. HCP reinjects residuals of hot channels online. On GLA-1.3B/60B tokens the gap goes from 0.94% to 0.58%. Overhead not reported in the abstract. — [CHON, arXiv 2602.02047](https://arxiv.org/abs/2602.02047)

**ALTO, MXFP4 MoE (arXiv 2610.32229v2, Oct 2026, AMD)** — lessons transferable to NVFP4
- GPT-OSS-20B on C4 (MLPerf small MoE), simulated MXFP4. BF16 3.3283. MXFP4 with 1D activations, 32x32 2D weights, RHT on wgrad and SR on grads reaches 3.3418 (+0.014). **+weight de-oscillation reaches 3.3350 (+0.007, about half the gap closed)**. De-oscillation is a post-AdamW hook: 200-step window, threshold τ=4.0, enabled at step 2000, flagged weights snapped to bin centers. Steps to target: BF16 15,360, plain MXFP4/NVFP4 18,432 (+20%), with de-oscillation 16,128 (+5%). Ablation deltas: de-osc -0.008, M+Adam -0.007, 128x128 macro-block scaling -0.004, static clipping 0.000, low-rank compensation +0.007, dynamic (QuEST) clipping +0.181, DGE +0.201. lm_head and router stay unquantized. — [ALTO](https://arxiv.org/html/2610.32229v2)
- Operator SNR is a poor proxy: MBS and static clipping raise SNR without helping loss. — [ALTO](https://arxiv.org/html/2610.32229v2)

**Older / superseded (MXFP4 era)**
- Quartet v1 (May 2025, MXFP4 g=32): forward uses Hadamard + QuEST RMSE clipping with trust masks. Backward uses an RHT with shared seed, then RTN (not SR) with 3/4 scaling and 16/9 rescale. 30M model C4 loss at 100 tokens/param: Quartet 3.29 vs LUQ-INT4 3.66. HALO-FP4 and Jetfire-FP4 were unstable. effN/effD for Quartet are 0.65/0.95. No NVFP4 results. RTX 5090 gives up to about 1.6x over FP8 for training. Footnote: an early bug had favored SR, and the authors reversed that conclusion (RTN better in backward for their setup, with the gap shrinking on longer runs). **Superseded by Quartet II for NVFP4.** — [Quartet v1, arXiv 2505.14669v4](https://arxiv.org/html/2505.14669v4)
- HALO: block-wise Hadamard + LSQ (learned scales) forward, leverage-score sampling backward. INT8 / INT4-effective, not true FP4. Unstable as FP4 in Quartet's comparison. **Superseded.** — [Quartet v1 search summary](https://arxiv.org/pdf/2505.14669)
- FP4 All the Way (Chmiel, Fishman, Banner, Soudry; NeurIPS 2025): finds NVFP4 (16-element blocks, E4M3 scale) is the best block format. Uses RTN forward and SR in backward and update. 7B on 256 Gaudi2 up to 200B tokens. Criterion: FP4 training stops being effective when the gradient std falls below **√3 × quantization noise**, and the remedy is switching to higher precision late (QAF). — [FP4 All the Way, arXiv 2505.19115](https://arxiv.org/abs/2505.19115)
- Deterministic Hadamard on the full MXFP4 pipeline (Llama-3.1-8B, C4) restores stability and cuts the token overhead to 8-9%. In the same table, randomized H16 on wgrad only does not converge (MXFP4). — [arXiv 2605.09825](https://arxiv.org/pdf/2605.09825) (search-snippet level, not fetched)

**Scale format (E8 vs E4M3, per-tensor scale)**
- NVFP4 (E4M3 scale, g=16) vs MXFP4 (E8M0, g=32): 1.5% vs 2.5% gap and 36% more tokens for MXFP4 (8B/1T). — [NVIDIA](https://arxiv.org/html/2509.25149v2). FP4-All-the-Way also picks NVFP4 as optimal. — [arXiv 2505.19115](https://arxiv.org/abs/2505.19115)
- Per-tensor scale constants: NVIDIA uses 6·448/amax. 4/6 and MS-EDEN both reduce 448 to 256 in the global scale to leave E4M3 headroom (for the 1.5x scale or the EDEN correction). TetraJet-v2 replaces per-tensor with 1x128 outer scales (150M PPL: per-tensor 31.75, per-row 31.58, 1x128 31.49). — [4/6](https://arxiv.org/html/2512.02010v3), [Quartet II](https://arxiv.org/html/2601.22813), [TetraJet-v2](https://arxiv.org/html/2510.27527v1)

### Inferences
- Several conflicts resolve consistently. NVIDIA found dgrad RHT harmful at 1.2B with RHT16 and a fixed seed. TetraJet-v2 (RHT32, dX+dW best) and Mosaic (dim-128 tiled, resampled signs) found it helpful. UFP4 found dX+dW worse than dW-only. The likely reading is that dgrad RHT helps when it is larger and re-randomized and the gradient quantizer is unbiased. This is an inference to verify empirically, not a settled fact.
- Likewise, 2D 16x16 weights (NVIDIA) vs 1x16 (TetraJet-v2: 1x16 better by 0.25 PPL; Quartet II: square blocks hurt forward quality) is resolved by **1D weights + re-quantization of the dequantized forward-quantized weight for the backward GEMM**. That keeps forward resolution while dgrad sees a weight close to the forward one. Our current code does neither (it quantizes the raw BF16 weight independently along each axis).
- MS-EDEN replaces element-wise SR with RTN after RHT128, plus a per-16 correction factor SR-rounded into the E4M3 scale. That costs one dot-product ratio per 16 elements, and the SR runs only on the 1-in-16 scale, so it is likely cheaper in RNG than element SR.

### Gaps
- QuEST (standalone paper) details were not fetched separately; only its use inside Quartet v1 is documented here.
- MS-AMP-style (FP8 optimizer/weights) methods: I found no NVFP4-specific results. Full-Stack FP4's Q-AdamW (1.40% to 0.98% at 1B) is the closest quantized-optimizer result, and it is relevant to memory, not to closing the gap.
- No source reports learned (LSQ-style) NVFP4 block scales in pretraining with measured gains. HALO's LSQ is INT-based and superseded. ALTO's QuEST-style dynamic clipping hurt (+0.181).
- A rare-fact recall penalty for NVFP4 (1.32-1.52x more exposures needed despite a ~2% loss gap, arXiv 2611.12832) was seen only via a secondary blog ([pebblous](https://blog.pebblous.ai/blog/nvfp4-pretraining-rare-fact-lag/en/)). Unverified.

---

## Q4. Which techniques give the biggest gap reductions at small scale (100M-1B) vs large scale, and which are cheap in custom CUDA kernels

### Takeaway
At 70M-2B scale, the largest measured relative reductions are: TetraJet-v2-full (~50%: OsciReset + OutControl + double-block + dX/dW RHT), Full-Stack FP4's LoRA-SVD (1.40% to 0.61% at 1B), Quartet II / MS-EDEN + 4/6 (15-25%), and late fprop switch to higher precision (NVIDIA 12B: 1.5% to 0.5%). At 8B-30B, Mosaic's stack gives only 0.24-0.36 pp, and high-precision tail layers dominate stability. The cheapest kernel-level changes are: 4/6 (<15% quant overhead), 1D-weight re-quantization, MS-EDEN, wider dgrad RHT, and optimizer-side de-oscillation (no GEMM change).

### Cited Findings
- Small-scale gaps are larger in absolute terms. NVIDIA recipe PPL gaps vs BF16: +5.67 (70M), +3.24 (150M), +2.50 (370M). TetraJet-v2-full: +2.48, +1.46, +1.41. — [TetraJet-v2](https://arxiv.org/html/2510.27527v1)
- Quartet II gains at 560M and 1.9B: 1.56% to 1.26% and 1.92% to 1.44% val-BPB gap vs NVIDIA. — [Quartet II](https://arxiv.org/html/2601.22813)
- Large-scale Mosaic gain: 0.36 pp (Nano 30B-A3B) and 0.24 pp (8B). The gap shrinks with model size (Nemotron 3: <1% Nano, <0.6% A8B). — [What Matters](https://arxiv.org/html/2610.24343v1), [Nemotron 3](https://arxiv.org/html/2512.20856)
- At small scale, NVIDIA saw no measurable benefit of fprop/dgrad RHT, and Hadamard size made no difference at 1.2B. Random-sign strategy did not matter at 1.2B. — [NVIDIA](https://arxiv.org/html/2509.25149v2)
- Recipe rankings flip before ~150B tokens at 8B scale. — [What Matters](https://arxiv.org/html/2610.24343v1)
- Costs: 4/6 adds <15% to the quantize op. — [4/6](https://arxiv.org/html/2512.02010v3). Fused RHT16+quant costs about 1.06-1.07x quant alone; unfused costs 1.41-1.62x fused. — [UFP4](https://arxiv.org/html/2606.20381v1). MS-EDEN/Quartet II kernels give 1.85x end-to-end over BF16 on RTX 5090 (1.1B). — [Quartet II](https://arxiv.org/html/2601.22813). LoRA-SVD realign is about 0.02-0.03% of step time, plus 3-4% parameters and the extra BF16 low-rank GEMMs. — [Full-Stack FP4](https://arxiv.org/html/2607.04422v2). OsciReset and de-oscillation are per-weight bookkeeping in the optimizer (cost unmeasured). — [TetraJet-v2](https://arxiv.org/html/2510.27527v1), [ALTO](https://arxiv.org/html/2610.32229v2). Late BF16 switch is about 6% of compute for an 18% tail. — [NVIDIA](https://arxiv.org/html/2509.25149v2)
- OutControl needs mixed-precision GEMMs for 10% of channels (FP8 or BF16 side GEMMs). — [TetraJet-v2](https://arxiv.org/html/2510.27527v1)

### Inferences
- For a 124M model at about 0.4 tokens/param (49M tokens), the token horizon is far shorter than every published study (the minimum is about 25 tokens/param in Quartet II ablations). Oscillation-type fixes (OsciReset, de-oscillation), which matter "most in late training", may show less benefit at this horizon. Forward-error fixes (4/6, fixing W consistency, outlier channels in ffn2) are more likely to move a constant 0.019-nat gap.
- Suggested ordering by expected gain per implementation effort for this library, inferred and not measured: (1) 4/6 on forward W/X with tensor scale 256 (kernel-local). (2) W consistency, either 16x16 2D weight scales or 1D + re-quant from the dequantized fprop weight. (3) Replace gradient SR with MS-EDEN (RHT128 + scale correction) and add dgrad RHT. (4) Keep down-proj (ffn2) in BF16 or use OutControl channels. (5) De-oscillation hook. (6) Late fprop BF16 switch.

### Gaps
- No study covers the TinyStories / <1 token-per-param regime, so the transfer of any number above to our benchmark is unverified.
- No end-to-end overhead measurement for OsciReset, OutControl or CHON's HCP on real NVFP4 hardware.

---

## Q5. Does keeping attention projections or first/last layers in higher precision matter, and what is the effect of RTN vs SR on activations?

### Takeaway
**Last layers matter most.** NVIDIA's 1.2B model needs about the last 4 of 20 blocks in BF16; first-only doesn't help. Mosaic diverges at 8B without the last 15%. At small scale, **FFN down-proj (ffn2) is the most sensitive module**, more than attention (TetraJet-v2). Production Nemotron 3 keeps QKV/attention projections in BF16 mainly because attention layers are few in a hybrid model. On activations, **RTN beats SR**: NVIDIA's 1.2B diverges with SR on activations or weights, and Quartet measures forward SR at roughly 2x the MSE of RTN.

### Cited Findings
- Last vs first blocks (NVIDIA 1.2B): last 4 in BF16 converges; first-only does not help without the last; first 2 + last 2 is stable; first 4 alone is not. 12B uses first 2 + last 8. — [NVIDIA](https://arxiv.org/html/2509.25149v2)
- All-layers-quantized Mosaic 8B diverges past 300B tokens. With the last 15% in BF16 it is stable, and the backward variants become indistinguishable. — [What Matters](https://arxiv.org/html/2610.24343v1)
- Module sensitivity at 370M: BF16 ffn2 recovers 0.35 PPL and ffn1 recovers 0.29; attention modules matter less. Forward activation quantization is the largest single error source (+0.43 PPL of +0.83 total). — [TetraJet-v2](https://arxiv.org/html/2510.27527v1)
- Nemotron 3 keeps QKV and attention projections in BF16 "to preserve the fidelity of the few attention layers". Mamba out-proj goes to MXFP8 because of up to 40% flush-to-zero in NVFP4. — [Nemotron 3](https://arxiv.org/html/2512.20856)
- CHON: SwiGLU (FFN) and softmax are outlier sources, and post-QK operations are most quantization-sensitive. — [CHON](https://arxiv.org/abs/2602.02047)
- ALTO keeps lm_head and router gates unquantized. — [ALTO](https://arxiv.org/html/2610.32229v2)
- RTN vs SR on activations and weights: SR on activations or weights diverges at 1.2B, and SR on forward tensors increases quantization error. — [NVIDIA](https://arxiv.org/html/2509.25149v2). Forward metrics (MXFP4): SR AbsMax effN 0.42 / MSE 2.77e-2 vs RTN effN 0.59 / MSE 1.37e-2. — [Quartet v1](https://arxiv.org/html/2505.14669v4). MSE N(0,1) for NVFP4 1x16: SR 23.5e-3 vs RTN 9.0e-3. — [Quartet II](https://arxiv.org/html/2601.22813). FP4-All-the-Way and TetraJet-v2 also use RTN forward and SR backward. — [arXiv 2505.19115](https://arxiv.org/abs/2505.19115), [TetraJet-v2](https://arxiv.org/html/2510.27527v1)
- SR on gradients: essential at 12B. — [NVIDIA](https://arxiv.org/html/2509.25149v2). It is now beaten by MS-EDEN (unbiased, 2.5x lower MSE). — [Quartet II](https://arxiv.org/html/2601.22813), [What Matters](https://arxiv.org/html/2610.24343v1). ALTO keeps SR despite its lower SNR because it removes gradient bias. — [ALTO](https://arxiv.org/html/2610.32229v2). For E1M2, SR on dY adds -0.0046 loss. — [UFP4](https://arxiv.org/html/2606.20381v1)

### Inferences
- With 12 blocks, our "last 2 blocks in BF16" is about 17% of blocks, already at NVIDIA's ~15% guidance. Adding the first block, or only the ffn2 down-proj everywhere, is a cheap ablation. TetraJet-v2 suggests ffn2 is a better use of the BF16 budget than attention projections at the 100M-400M scale.
- Our RTN-on-activations / SR-on-gradients choice matches every source. The open improvement on the gradient side is MS-EDEN, not changing activation rounding.

### Gaps
- No source isolates "attention projections in BF16" as an ablation with a loss number for dense GPT-style models at small scale. Nemotron 3's reason is architectural (hybrid with few attention layers).
- The effect of keeping embeddings-adjacent first layers in BF16, separate from the last layers, has only been reported qualitatively (NVIDIA 1.2B).
