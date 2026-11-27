# Changelog

## 0.19.0 — Two-term forward activations, an 8-bit attention backward on FP16 MMAs (2026-11-27)

- **Where the GPT-124M gap comes from** (`experiments/gap_ablate.py`, `gap_fwd.py`, `run_gap*.sh`; RESULTS.md
  §4.3): three quarters is the forward GEMM, and in it the activations of every layer (BF16 fprop −0.90%,
  dgrad −0.12%, wgrad −0.14%, round-to-nearest gradients −0.02%). A third is error at evaluation time.
  Random Hadamard rotations of the activations do not help (+0.15%); MXFP8 activations or two NVFP4 terms
  remove half of it (emulated).
- **Two-term forward activations** (`ops.set_fprop_terms(2)`, `train_gpt.py --fprop-terms 2`; off by
  default): the activation is quantized as [Q(x) | Q(x − deq Q(x))] with one tensor scale, and the forward
  GEMM runs once at twice the depth against [W | W] (cached, `WeightQuantCache.wq2`). In the dual
  quantizer's row phase, so every producer path has it (`quantize_dual` / `layernorm_` / `rmsnorm_` /
  `gelu_` / `swiglu_quantize_dual(row_terms=2)`, bit-exact against `reference.quantize_dual(row_terms=2)`,
  Four Over Six included); `ops.two_term_weight`. Three seeds: −0.62% ± 0.19 paired, gap +1.34% → +0.70%,
  −3.3% tokens/s; with the BF16 tail +0.47% ± 0.14. Not for the MoE experts.
- **8-bit attention backward on FP16 MMAs** (`attention(..., backward="8bit")`): dQ, dK and dV from FP16
  operands scaled per tile row with FP16 accumulation, a consistent Delta for dK / dV; smaller errors than
  the exact backward on every layer, deterministic, as fast as flash's backward at D = 128; +0.15% loss vs
  BF16 attention in training at 4096 tokens (was +0.52%). The default stays the exact backward.
- Fixed: the dual quantizer's vectorized stores assumed an even block offset (misaligned with the second
  term at C ≡ 16 mod 32).
- `bench/bench_fprop_terms.py`; `tests/test_fprop_terms_cuda.py`.

## 0.18.0 — Expert parallelism: chunks with nodes, GEMMs beside NCCL, MXFP8 return messages (2026-11-26)

- **Chunks with nodes fixed** (`moe.ep_serial`, "auto": on with `ep_node_size` > 1 and `ep_chunks` > 1):
  v0.17's regression (2.4× slower with chunks) came from the rail and node collectives of different chunks
  running concurrently. Exchanges are now issued from one side stream, each after the previous one ends; on
  four RTX PRO 4500 with nodes of two, chunks 2 are 14–16% faster than none (layer B 80.4 vs 95.4 ms; v0.17:
  230). The side stream records its buffers (the process group's references are dropped at its wait; a
  first version let the compute stream reuse a buffer NCCL was reading).
- **Dynamic tile claiming in the persistent GEMMs** (`ops.set_dynamic_tiles`, `NVFP4_GEMM_DYNAMIC=1`;
  `moe.ep_dynamic_tiles`, "auto": on around the layer's forward and backward with chunks or an overlapped
  shared expert): v0.17's "GEMMs 1.7× slower beside the transfers" was SM residency, not memory
  contention: CTAs that could not start while NCCL held SMs ran their whole fixed share as a second wave
  (1.75× for 2–16 held SMs, `bench/bench_residency.py`; 2.07–2.22× beside the four-GPU all-to-all). CTAs now
  start on one tile and claim the next from a per-stream counter: 0.98–1.11× held, 1.04–1.08× beside the
  all-to-all. Bit-identical results; graphs being captured keep the fixed assignment; off by default (costs
  0–6% alone).
- **MXFP8 return messages** (`moe.ep_wire_return = "fp8"`, `train_gpt.py --moe-ep-wire-return`): expert
  outputs and the input gradients sent back travel as e4m3 with a power-of-two scale per 32 values (1.94×
  fewer bytes than bf16; gradients stochastically rounded). Loss over two seeds +0.26% (within noise); with
  nodes, the NVFP4 dispatch and chunks, layer B on four GPUs takes 67.2 ms (v0.17: 96.1; data parallelism:
  195.1). Kernels in `csrc/moe_combine.cu` (`nvfp4._library.mxfp8_encode` / `mxfp8_decode`, 750–800 GB/s).
- **Benchmarks:** `bench/bench_overlap.py` (a GEMM beside a real all-to-all: margins, dynamic tiles,
  `NCCL_MAX_NCHANNELS`), `bench/bench_residency.py`, new variants in `bench_moe_ep.py` /
  `profile_moe_ep.py` (`--node-size`, `--only`); `tools/pod_session.sh` `overlap` phase.
- **Measured** on 1, 2 and 4 RTX PRO 4500 Blackwell (RESULTS.md §8.5–8.8); MoE tests over NCCL pass on two
  and four GPUs; sanitizers cover the new kernels (racecheck skips the claim ring, whose spin it cannot
  run); 5-minute soak passes.
- Test spawns retry when the chosen port is taken (EADDRINUSE).

## 0.17.0 — MoE: variable-size, node-aware, chunked expert parallelism with NVFP4 messages; fused combine; RTX PRO 4500 (2026-11-26)

- **Each token once per rank** (`ep_dispatch="variable"`, the default without a capacity): a token goes once
  to every rank holding one of its experts, with their ids and routing weights in the same byte message,
  and that rank returns the weighted sum of its experts' outputs (DeepEP's layout). One all-to-all of
  counts per forward (one host read) sizes every message. Dropped slots (capacity) are not sent. v0.16's
  per-slot dropless path is replaced; the fixed-size capacity path stays (CUDA-graph capturable).
- **Nodes** (`MoE(..., ep_node_size=L)`): tokens cross to another node once per node, to the rank with the
  sender's position (rail sub-groups), which forwards them within its node (node sub-groups) and sums the
  node's rows on the way back.
- **Chunks** (`moe.ep_chunks = n`): the tokens go in n chunks in a wavefront, the next chunk's exchanges
  overlapping this chunk's experts, forward and backward. `ops.set_sm_margin(k)` leaves k SMs free in the
  persistent GEMMs (bit-identical results).
- **NVFP4 messages** (`moe.ep_wire = "nvfp4"`, `moe.ep_wire_grad = "nvfp4"`): activations (and gradients,
  stochastically rounded) travel as NVFP4 with the group's tensor scale, 3.6× fewer bytes. New:
  `ops.quantize_dual(amax_floor=)` (a shared tensor scale) and `ops.dequantize` (a kernel). Re-quantizing
  the received rows reproduces the codes, so the forward and input gradient are bit-identical to bf16
  messages. Loss over two seeds: +0.05% (activations), +0.23% (+ gradients), both within run-to-run noise.
- **Expert parallelism with DDP:** `nvfp4.moe.ignore_expert_parameters_in_ddp`, `scale_expert_grads`
  (tested against one process); `train_gpt.py --moe-ep` (`--moe-ep-chunks`, `--moe-ep-node-size`,
  `--moe-ep-wire`, `--sm-margin`).
- **Fused gather / combine** (`csrc/moe_combine.cu`): the MoE combine, the expert-parallel partial sums and
  the dispatch's backward in one kernel each (fp32 accumulation, gather-based deterministic backward),
  replacing index_select + bmm + scatter-add: the single-GPU MoE layer is 8–15% faster in NVFP4 and uses
  ~2 GB less; an expert-parallel step 31% faster on two GPUs. `NVFP4_MOE_FUSED_COMBINE=0` restores the old
  path.
- **Measured on 4× RTX PRO 4500 Blackwell** (RESULTS.md §8.2, §9): layer B (32 experts, top-4) on four GPUs
  322.8 ms (fixed-size) → 196.4 (once per rank) → 96.1 (two stages), vs data parallelism 196.3; a MoE GPT
  trains at 103.9k tokens/s with two-stage expert parallelism vs 81.9k with DDP, in 10.5 vs 12.7 GB per GPU.
  Chunks with nodes were slower (233 ms); both stages over one communicator 3× slower (reverted).
  `bench/profile_moe_ep.py` shows where a step's time goes.
- **RTX PRO 4500 Blackwell validated** (82 SMs, 200 W): tests, sanitizers, soak pass; GEMM up to 493
  TFLOPS (0.98–2.0× cuBLASLt); the 8-bit attention is no faster than SDPA there.
- **Fixes for PyTorch 2.11's torch.compile:** the attention backward's flash-attention check is decided at
  import (setting a global in the backward broke `fullgraph=True`); the MoE compile test compares
  numerically before PyTorch 2.14.
- `tools/pod_session.sh`: `ep` on four GPUs (with nodes of two), `eprof`, `moegpt` phases.

## 0.16.0 — MoE: dropless expert parallelism, overlapped exchange, measured on two GPUs; native Windows rerun (2026-11-25)

- **Dropless expert parallelism** (`MoE(..., ep_group=g)` without `capacity_factor`): the row counts go
  first (an integer all-to-all) and are read on the host once per forward; the rows, sorted by expert,
  travel in variable-size all-to-alls (as Megatron-LM's dropless dispatcher). Not CUDA-graph capturable;
  the capacity variant still is.
- **Overlap:** an exchange is now a start and a wait (async all-to-all). The shared expert runs during
  the return of the expert outputs, and autograd's ordering runs its backward during the reverse
  exchange (a test checks the call order). `moe.ep_overlap = False` runs it after.
- **Measured on 2× RTX 5090 (PCIe, NCCL)**, `bench/bench_moe_ep.py`, RESULTS.md §8.1: the MoE tests pass
  across the two GPUs. The exchange runs at about 15 GiB/s per GPU and is not hidden: expert parallelism is
  2.0–3.0× the local step and slower than data parallelism with an expert-gradient all-reduce. Dropless
  beats capacity 1.25 by 15–19%; the overlap saves 8% (BF16) / 2% (NVFP4). Its gain on such machines is
  memory.
- **Native Windows** (PyTorch 2.11.0+cu128, CUDA 12.9): 919 passed, 80 skipped.
  - That wheel has no flash attention, so the 8-bit attention's default BF16 backward falls back to the
    same math in fp32, query block by block (`_math_backward`, tested against flash on Linux).
  - Its gloo crashes on all_reduce of CUDA tensors, so the multi-process DDP / FSDP tests skip there
    (probed once per session); tests that need Triton skip on Windows.
- `tools/pod_session.sh`: `BRANCH`, and an `ep` phase (MoE tests, expert-parallel and MoE benchmarks).

## 0.15.0 — MoE: capacity-limited routing, a shared expert, expert parallelism (2026-11-25)

- **Capacity** (`MoE(..., capacity_factor=1.25)`): each expert keeps the first C = ⌈cf·T k / E⌉ of its
  slots in token order (Switch Transformer).
  - Dropped slots add nothing and point at an always-zero buffer row. The buffer shrinks to
    min(T k, E C) + E·127 rows.
  - Nothing is read on the host: CUDA graphs and compile still work.
  - `moe.capacity(T)` gives C; `moe.last_dropped` counts the dropped slots.
- **Shared expert** (`shared_d_ff=...`): a dense MLP that every token goes through, added to the routed
  output (DeepSeek-MoE; n shared experts of hidden h = one of hidden n h). Parameters `shared_w_*` /
  `shared_b_*`. It runs through the same grouped NVFP4 kernels (or Triton / the loop in high precision).
- **Expert parallelism** (`ep_group=...`, needs a capacity): each of W ranks holds E/W experts.
  - Slots travel by two fixed-size all-to-alls ([W, E/W, C, d] rows plus a small count exchange), with
    a differentiable wrapper.
  - The local experts run as in any other layout.
  - Tested with two processes over gloo on one GPU: in fp32 the output equals the single-process layer
    on each rank's tokens, and the local expert gradients equal the sum over the ranks. NVFP4 is
    checked by cosine.
  - Not measured for speed: no multi-GPU machine, no overlap of exchange and compute.
- **Internal:** one GPU-side planner (`_plan`) serves routing, capacity, the shared expert and the
  expert-parallel receive side. The dropless layout is unchanged (same buffer size: v0.14 numerics).
- **Training:** `experiments/train_gpt.py --moe-capacity`, `--moe-shared`; with a capacity it logs the
  dropped fraction.
  - The capacity-1.25 MoE GPT drops 12% of slots at step 25 and almost none after balancing.
  - It ends at BF16 1.4923 and NVFP4 1.5066 (+0.96%), 62.3k vs 50.8k tokens/s (RESULTS.md §8).
- **Tests:** `tests/test_moe_features_cuda.py`; capacity / shared variants in the CUDA-graph, compile,
  checkpointing and DDP / FSDP1 / FSDP2 tests. The test gather helper handles FSDP2's uneven shards (the
  shared expert's dim 0 is 1). The sanitizer driver covers capacity and the shared expert.
- **Layer benchmark:** 32 experts, top-4, capacity 1.25 and a shared expert of 2816. NVFP4 is 2.37× the
  BF16 loop and 2.18× the grouped BF16 path, 2.26× with both in CUDA graphs.
- **Checks:** 989 tests (968 pass on the laptop, 21 skip); memcheck / racecheck / synccheck / initcheck
  clean; 5-minute soak OK.

## 0.14.0 — MoE expert biases; grouped BF16 experts without host synchronization (2026-11-25)

- **Expert biases:** `MoE(..., bias=True)` gives every expert projection a per-expert bias: `b_gate`,
  `b_up` [E, d_ff] and `b_down` [E, d], or `b_fc1`, `b_fc2` (`moe.experts_biases()`). They are
  initialized like nn.Linear's, and work on every path.
  - **NVFP4:** the grouped GEMM's store adds the bias, as the dense layers' epilogue does. It goes only on
    each tile's real rows, so an expert's padding rows stay zero: in the weight gradient they share
    16-token quantization blocks and the RHT with real tokens.
  - **Gradients:** per-expert sums of the per-tile column sums the gradient quantizers already compute.
    One fp64 one-hot product does the sum: deterministic, nothing read on the host.
  - **Frozen weights:** biases alone train too.
- **`ops.gemm_grouped(..., bias=, tile_rows=)`:** the per-group bias epilogue (mode "m"), bit-identical
  to `D_bf16.to(out_dtype) + bias[g]` on the first `tile_rows[t]` rows of each used tile.
  `quantize_dual`, `gelu_backward_quantize_dual` and `swiglu_backward_quantize_dual` take
  `bias_grad="tiles"`: the per-64-row partial column sums. The dense GEMM is unchanged (457–463 TFLOPS
  at 8192³).
- **The high-precision experts no longer synchronize with the host.** This covers BF16 experts before
  conversion, the BF16 blocks of an NVFP4 model, and a BF16 tail under `ops.set_high_precision`.
  - They run on the NVFP4 path's padded buffer and GPU-side tables, with grouped Triton GEMMs
    (`nvfp4/_bf16_grouped.py`, `torch.library` custom ops): a ragged-M kernel for the forward and input
    gradient, a ragged-K kernel for the weight gradients.
  - The kernels are deterministic and work in fp32 too (IEEE products).
  - So every MoE configuration is CUDA-graph capturable and compiles with `fullgraph=True`.
    `torch._grouped_mm` is no alternative on sm_120: PyTorch 2.14 runs it as a synchronizing loop.
  - Speed: 2–7% faster than the per-expert loop on the benchmarked layers. NVFP4 is 1.5–2.5× the
    grouped BF16 path.
  - `moe.bf16_impl`: "auto" (grouped when Triton is installed), "grouped" or "loop". Without Triton
    (native Windows) the loop remains.
- **MoE GPT gap corrected to +1.12%** (was +0.88%):
  - The v0.14 runs train the BF16 baseline and the NVFP4 model's two BF16 blocks with the grouped experts.
    Seeds 0 / 1 give BF16 1.4761 / 1.4842 and NVFP4 1.4993 / 1.5012, at 51.7k / 63.5k tokens/s (+23%).
  - BF16 seed 0 ended 0.77% below v0.12's run, with the same math: one batch's gradients from the same
    weights are equally close to fp32 on both paths. The router turns summation-order differences into
    seed-sized trajectory changes.
  - Pooled over all 4 BF16 and 6 NVFP4 runs of this configuration, the gap is +1.12%. That is close to
    the dense model's +1.32%; the reported +0.88% rested on two high BF16 baselines (RESULTS.md §8).
- **Tested:** both paths, with and without biases:
  - vs per-token and per-group fp64 references;
  - bias gradients against the routed gradient sums;
  - grouped BF16 against the loop;
  - no host sync, CUDA graphs and compile;
  - checkpointing;
  - two-process DDP / FSDP1 / FSDP2, bit-identical to a single process.
  - Compiled vs eager (with `emulate_precision_casts`): outputs, bias gradients and NVFP4 weight gradients
    are bit-identical. The input gradient is checked by a stated bound instead of an arbitrary 1e-5 norm:
    under 1% of elements differ, by at most two bf16 steps of the gradient's RMS.
- 959 tests (938 pass on the laptop, 21 skip); memcheck / racecheck / synccheck / initcheck clean
  (the driver now covers the bias epilogue and the Triton path); 5-minute soak OK.

## 0.13.0 — MoE without host synchronization: CUDA graphs, torch.compile, data parallel (2026-11-24)

- **The NVFP4 MoE layer never synchronizes with the host.**
  - The token buffer has the worst-case size for the step's tokens (T k + E·127 rows, rounded up to 256).
    The tile table, ragged-K segments and dispatch indices are computed on the GPU from device-side
    counts (an integer `scatter_add`, deterministic).
  - Training steps with MoE layers can be captured in CUDA graphs. One graph serves any routing of the
    same token count: replays equal eager bit for bit while the routing changes, including experts that
    get no tokens. Captured Adam steps learn.
  - `torch.compile(fullgraph=True)` traces the layer with no graph breaks (`gemm_grouped` is now a
    `torch.library` custom op).
  - Layer step: 32 experts at 2.39× BF16 in a CUDA graph (2.12× eager); 8 experts at 2.23× / 2.66×.
- **Grouped GEMM: work units without k-tiles.**
  - Tile group −1 marks an unused tile, which is stored as zeros without loading anything.
  - Ragged-K segments may have length 0 (zeros, or the output left unchanged with `accumulate`).
  - The load cursors skip such units (`skip_empty`, grouped instantiation only). The dense GEMM is
    unchanged (461–468 TFLOPS at 8192³).
- **Tested now** (`tests/test_moe_integration_cuda.py`, both activations):
  - no host sync (`set_sync_debug_mode("error")`);
  - CUDA-graph forward and full training step;
  - `torch.compile` vs eager;
  - activation checkpointing (bit-identical gradients);
  - two-process DDP, FSDP1 and FSDP2, fused and unfused, bit-identical to a single process;
  - deepcopy / pickle.
- **Compiled vs eager.** Inductor's fused kernels keep fp32 where eager rounds to bf16. A slightly
  different bf16 gradient moves an NVFP4 tensor scale, so compiled MoE training matches eager
  statistically, not bit for bit. With `emulate_precision_casts=True` (and `fallback_random=True`) the
  output and the expert weight gradients are bit-identical (USAGE.md §3.4).
- **Fixed:** `copy.deepcopy` / pickling of a MoE layer after a forward pass. `aux_loss`, a non-leaf
  tensor, made deepcopy raise.
- **Changed:** `MoE.last_counts` is a property that reads the device-side counts on demand (that read
  synchronizes). The MoE GPT trains as before: 1.5033 / 1.4965 for seeds 0 / 1 against v0.12's
  1.5006 / 1.4973. The wider buffer changes only the stochastic-rounding bits of the wgrad operands.
- Sanitizer driver covers the empty units. 900 tests (879 pass on the laptop, 21 skip), memcheck /
  racecheck / synccheck / initcheck clean, 5-minute soak OK.

## 0.12.0 — grouped GEMMs for mixture of experts, MS-EDEN on dgrad (2026-11-24)

- **Grouped NVFP4 GEMMs** (`ops.gemm_grouped`, DESIGN.md §4.5). The persistent TMA kernel gets a
  work-unit decoder for groups; the pipeline, MMA loop and epilogue are unchanged.
  - Ragged M (forward, dgrad): 128-row tiles, each tagged with its group, against stacked B.
  - Ragged K (wgrad): per-group token ranges `(k0, len, row0)`. Half k-tile masks cover ranges that
    aren't multiples of 256.
  - K % 128 is accepted, with zero-filled tails and 16-byte scale strides.
  - It is a separate template instantiation, so the dense GEMM compiles as before (463 vs 464 TFLOPS,
    interleaved A/B).
  - `ops.stack_transposed` rearranges a stacked weight's 16×16-block transpose into the grouped dgrad
    operand.
- **MoE runs on grouped GEMMs.** `nvfp4.moe.MoE` quantizes the expert-sorted tokens once and runs each
  projection of all experts as one GEMM, plus one ragged-K GEMM for their weight gradients (fused
  accumulation into `.grad`).
  - Kernels per step are fixed: 258 for a 32-expert layer, down from 1,903.
  - Layer step vs BF16: 2.12× / 2.57× with 8 experts (v0.11: 1.74× / 2.15×), **2.14× with 32 small
    experts (v0.11: 1.25×)**, 1.67× for GELU at d = 768.
  - MoE GPT, 2 seeds: +0.88% loss at +25% tokens/s. v0.11.0 rerun on the same seed lands 0.1% away.
  - Dispatch and combine are now gathers (bf16 first, a batched-matmul combine), with no scatter-add
    atomics and no per-expert slicing, so both precisions are deterministic. The BF16 experts get faster
    too (32 experts: 148.9 → 77.5 ms).
- **MS-EDEN on the dgrad operands** (`ops.set_ms_eden(True, dgrad=True)`, `train_gpt.py --ms-eden-dgrad`;
  DESIGN.md §2.1), Quartet II's backward.
  - dY and the GELU / SwiGLU input gradients are rotated along N by a 64-point RHT in the dual
    quantizer's row phase and quantized with EDEN scales.
  - The rotation's signs come from the row SR seed, so no new parameters thread through the producers.
  - Rotating along N would split the 16×16 weight blocks, so the dgrad weight operand is the fprop
    weight dequantized and re-quantized along N under the same rotation (`linear.dgrad_weight_operand`).
  - The other quantizer paths are unchanged (template instantiation), and the new path is bit-exact
    against the extended reference.
  - GPT-124M, 3 seeds, paired vs the baseline: −0.08% ± 0.15 alone (same as wgrad-only MS-EDEN) and
    −0.20% ± 0.17 with Four Over Six, at 14% more wall time. Nothing resolvable, so it is off by default.
- **Breaking: MoE parameters are stacked.** `w_gate` / `w_up` / `w_down` (SwiGLU) or `w_fc1` / `w_fc2`
  (GELU), each [E, ., .].
  - The `experts` ModuleList and the `bias` argument are gone.
  - `MoE.to_nvfp4(...)` (called by `convert_linears`) replaces the per-expert conversion.
  - A model-level init that matches `nn.Linear` no longer reaches the experts: initialize
    `moe.experts_weights()` explicitly (USAGE.md §3.4). Left at the default, GPT's MoE ends 1% worse.
    `train_gpt.py` does this now.
- `experiments/run_moe.sh` runs seeds 0 and 1 into `experiments/results/v0.12_moe/`. The sanitizer
  driver covers the grouped GEMMs and the MS-EDEN dgrad paths. 881 tests (860 pass on the laptop, 21
  skip). memcheck / racecheck / synccheck / initcheck are clean, and the 5-minute soak passes.

## 0.11.0 — recipe options, BF16 tail, mixture of experts, cheaper stochastic rounding (2026-11-24)

- **`ops.set_high_precision(True)`, the BF16 tail** (NVIDIA arXiv 2509.25149 §4.4): while on, `NVFP4Linear`
  and the `nvfp4.fused` sub-blocks compute exactly what the modules they replace would (BF16 GEMMs,
  ordinary autograd; tested bit-identical), without touching weights or optimizer state, so a training
  loop can switch for its last steps. GPT-124M, last 10% of steps, three seeds: gap to BF16
  **+1.32% → +0.62%** (paired −0.70% ± 0.04, every seed), 4% more wall time.
  `experiments/train_gpt.py --bf16-tail FRAC` (also logs the same weights' NVFP4 validation loss).
- **Recipe options from the 2026 literature**, each bit-exact against the extended reference
  (DESIGN.md §2.1, USAGE.md §3.5), off by default:
  - `ops.set_four_six(True)` / `with ops.four_six_scales():`. Four Over Six block scales (each RTN block
    picks max → 6 or max → 4, by squared error; tensor constant 256).
  - `ops.set_ms_eden(True)`. MS-EDEN on the wgrad operands: a 64-point RHT (H4 ⊗ H16, the cross-lane
    stage through shared memory) and the gradient quantized with RTN elements and an EDEN-corrected,
    stochastically rounded E4M3 scale. Unbiased over the rotation; the bias test showed a 16-point
    rotation is not enough.
  - `ops.set_weight_blocks("1d")`. 1×16 weight blocks, with the dgrad weight re-quantized from the
    dequantized fprop weight.
  - Measured in a three-seed ablation harness (`experiments/run_ablation.sh`,
    `experiments/ablation_summary.py`, paired by seed), RESULTS.md §4.2: −0.03% (Four Over Six), −0.08%
    (MS-EDEN), −0.12% (both), −0.12% (1×16 weights), −0.19% (all three), −0.27% (fc2 kept BF16). The loss falls, but by at
    most a fifth of the gap at this scale, for 2–14% wall time. Quantizer cost: RESULTS.md §2.4,
    `bench/bench_sr_cost.py`.
- **`nvfp4.moe.MoE`**: dropless top-k mixture-of-experts MLP (SwiGLU / GELU experts, fp32 router softmax,
  Switch load-balancing loss). Tokens are sorted per expert and each expert runs the fused NVFP4 MLP on
  its slice (one host sync per layer).
  - Layer step: 1.74–2.15× BF16 with 8 experts, 1.25× with 32 small ones (`bench/bench_moe.py`).
  - MoE GPT (8 experts, top-2, 294M parameters), one seed: +0.76% loss at +20% tokens/s
    (`experiments/run_moe.sh`; `train_gpt.py --moe-experts`).
- **Cheaper software stochastic rounding** (no `cvt.rs` on sm_120a). An integer-domain rounder with 8
  random bits per element: one hash per four elements, plus Fitzgibbon & Felix's half-unit offset, which
  removes few-bit SR's rounding bias.
  - About 12 extra SASS instructions per element instead of 28.6: the dual quantizer with SR is 10–22%
    faster.
  - Same loss over three seeds (+1.32% ± 0.16 vs v0.5's +1.28%). The SR noise differs from 0.10's for the
    same seed.
- **Research survey** of the open problems: NVFP4 recipes, 8-bit attention, sm_120 GEMMs, software SR,
  MoE. It ranks the techniques by impact and effort; probe code is included
  (docs/research/nvfp4-open-problems-2026-11/).
  - Tried from it and rejected with measurements (DESIGN.md §6): single-term INT8 P·V in attention
    (dQ errors up to 10.6% on peaked layers).
  - Not pursued: FlashAttention-4's exp2 offload (the forward is MMA-bound, not MUFU-bound).
- README rewritten for developers: what runs in 4 bits, the API surface, the verification workflow.
- Tests: 835 (814 passed, 21 skipped on this machine: environment-dependent cases). New: Four Over Six and MS-EDEN
  bit-exact against the reference (and MS-EDEN's unbiasedness), 1×16 weights, MoE (routing, gradients vs
  BF16, fused path), the BF16 tail (bit-identical to the replaced modules, switching back and forth),
  8-bit SR bias bound.

## 0.10.0 — exact backward for 8-bit attention (2026-11-24)

- **`nvfp4.attention(..., backward="bf16")`, the new default**: the 8-bit forward with PyTorch's
  flash-attention backward run on the dequantized operands the forward multiplied (new `want_deq` output
  of `attn_quant`), with the forward's log-sum-exp and output. The forward computes P V in two e4m3 terms
  (P and its rounding residual, one more MXFP8 MMA) so that Delta = rowsum(dO O) is consistent with the
  exact backward (`attn_fwd(..., p2=True)`). Gradients are the exact gradients at the quantized operands
  (straight-through). On a trained GPT-124M's activations at 4096 tokens: dV error 0.2–2.1% (was
  2.5–3.1%), dK about halved, dQ 0.4–2.0%. GPT-124M at 4096 tokens, 2 seeds: validation loss **1.5979 /
  1.5949 vs BF16 attention's 1.6027 / 1.5928** (0.9.1: 1.6088 / 1.6033, +0.52%), same tokens/s. Per
  attention call: forward + backward 1.07–1.12× flash attention at head dim 128 (0.9.1: 1.12–1.18×),
  forward 1.2–1.6× (0.9.1: 1.5–2.1×). Bitwise deterministic only under
  `torch.use_deterministic_algorithms`, like SDPA (DESIGN.md §7.1).
- `backward="8bit"` keeps 0.9.1's deterministic 8-bit backward and one-term forward (faster forward for
  inference). `experiments/train_gpt.py --attn-backward`, `experiments/run_attn8_bf16bwd.sh`,
  `bench/attention_qbwd_probe.py` (gradient errors of each variant, with emulations that isolate the
  log-sum-exp and Delta effects); `bench/bench_attention.py` times both backwards.
- `attn_quant` returns a seventh tensor (the dequantized values) and `attn_fwd` takes `p2`; callers that
  unpack six results need updating.
- Tests: the dequantized values bit-exact against the reference, the two-term forward against exact
  attention of the quantized operands, the default backward against fp32 autograd at those operands;
  gradients, layouts and compile for both backwards. Soak and sanitizer drives cover both. 720 tests (38
  new). Llama block (d = 2048, fused NVFP4) at 16384 tokens: 1.09x from 8-bit attention with the default
  backward (1.12x with `backward="8bit"`); `bench/bench_long_context.py` times both.

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
