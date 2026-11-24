# Fastest block-scaled FP4 (NVFP4/MXFP4) GEMM kernel structure on consumer Blackwell (sm_120/sm_121), state as of Nov 2026

Scope note: about 16 tool calls. Primary sources are the CUTLASS C++ sources on GitHub `main`, two Colfax Research articles (June and Aug 2026), NVIDIA forum posts, and three community repos. I found no SASS analysis of cuBLASLt's own sm_120 FP4 kernels and no OMMA *throughput* microbenchmark. Treat every statement about "why cuBLASLt reaches about 90%" as inference.

## Q1. How CUTLASS 4.x sm120 block-scaled GEMMs are structured (warp specialization, tiles, setmaxnreg, SF loading, stages, reported performance)

### Takeaway
CUTLASS's sm120 block-scaled kernels reuse the sm90 TMA warp-specialized kernels:
- 1 producer warpgroup (setmaxnreg dec to 40, or 24 when register pressure is heavy).
- Consumer warpgroups raised to 232 (or 240) registers.
- Cooperative puts 8 MMA warps in a 4x2 layout on a 128xN tile.
- Pingpong runs 2 consumer warpgroups, each 4 warps in a 2x2 layout, alternating mainloop and epilogue.

Scale factors (SF) arrive by TMA in the same pipeline stage as A/B. They move from smem to registers with a plain auto-vectorized LDS (`UniversalCopy`), not ldmatrix. The next k-block's smem-to-register copy is issued before the current k-block's MMAs. The best published sm120 NVFP4 kernel (Colfax, CuTe DSL, RTX PRO 6000) reached 1666 TFLOP/s, 83% of nominal peak. Its biggest structural gain came from moving to **12 MMA warps (3 per scheduler)** on a 192x128 tile while keeping the per-warp tile size the same.

### Cited Findings

**CUTLASS C++ collective (`sm120_blockscaled_mma_tma.hpp`, `main`)**
- **Dispatch policy and pipeline.** The dispatch policy is `MainloopSm120TmaWarpSpecializedBlockScaled<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>`. The pipeline is `PipelineTmaAsync<Stages>`, and one elected producer lane issues the TMA copies. — [CUTLASS sm120_blockscaled_mma_tma.hpp](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp)
- **SF loads share the A/B stage.** SFA/SFB are TMA-loaded into the same stage as A/B (`tma_load_sfa`/`tma_load_sfb`, built with `make_tma_copy<uint16_t>`). The SF bytes are added to the stage's transaction count. The SF smem layout appends a stage mode to `SmemLayoutAtomSFA`. — [same](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp)
- **Mainloop order.**
  - `consumer_wait`, then `copy_kblock(0)`.
  - Then, for each k-block: `copy_kblock(k_block_next)` (A, B, SFA, SFB smem to registers), then `gemm_kblock(k_block)`.
  - The MMA consumes `make_zip_tensor(tCrA, tCrSFA)`.
  - On the last k-block of a stage: `NamedBarrier::sync(thr_size(tiled_mma), Sm120MainloopBarrier)`, then `consumer_release`, then `consumer_wait` on the next stage.
  - The final k-tile is hoisted out of the loop, and the loop is `CUTLASS_PRAGMA_NO_UNROLL`.

  — [same](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp)
- **Small N tiles.** When the N tile is under 128, SFB is padded to a 128-wide SF tile and `logical_divide`d per CTA. — [same](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp)

**CUTLASS builder (`sm120_blockscaled_mma_builder.inl`)**
- **SF copy atom.** `SmemCopyAtomSF = Copy_Atom<UniversalCopy<SmemAllocTypeSF>, SmemAllocTypeSF>; // auto-vectorized LDS`. A/B use `sm120_rr_smem_copy_selector_A/B` (ldmatrix-family selectors; their bodies are not in this file). — [CUTLASS sm120_blockscaled_mma_builder.inl](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl)
- **Warp layouts.**
  - Cooperative uses `Layout<Shape<_4,_2,_1>>` (8 warps), or `_8,_1,_1` when N < 16.
  - Pingpong uses `Layout<Shape<_2,_2,_1>>` (4 warps per consumer warpgroup).
  - The permutation is M = min(tileM, 128), N from `sm120_tile_n_permute_selector`, and K = 64 (32 for mxf8f6f4).

  — [same](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl)
- **Builder constraints and stages.**
  - The cluster must be 1x1x1 ("no programmatic multicast on this arch").
  - Only TN (K-major) is supported.
  - `SchedulerPipelineStageCount = 3`.
  - Stages come from `sm100_compute_stage_count_or_override_blockscaled` against `sm120_smem_capacity_bytes`, minus scheduler, tensormap and CLC storage.
  - The SF smem basic block is `Shape<_32,_4>:Stride<_16,_4>` in MN and `Shape<SFVec, MMA_NSF>:Stride<_0,_1>` in K.

  — [same](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl)
- **Schedule names.** Non-grouped kernels use `KernelTmaWarpSpecializedCooperativeBlockScaledSm120` and `KernelTmaWarpSpecializedPingpongBlockScaledSm120`. Grouped kernels use PtrArray variants of both. — [same](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl)

**CUTLASS kernel (`sm90_gemm_tma_warpspecialized_pingpong.hpp`, `main`)**
- **sm120 support.** The kernel compiles for sm120/121 (`CUDA_ARCH_CONDITIONAL_OR_FAMILY(1200)`) and branches on `IsSm120Family`.
- **Registers.**
  - `LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24`.
  - `MmaRegisterRequirement = !HeavyRegisterPressure ? 232 : 240`.
  - `HeavyRegisterPressure` is true when accumulator registers per thread are 208 or more.
- **Threads.** 2 MMA warpgroups plus 1 load warpgroup, 384 threads.
- **Ordering.** `MathWarpGroupOrderBarrier` (an `OrderedSequenceBarrier` with 2 stages, one for mainloop and one for epilogue) alternates the two consumer warpgroups.

— [CUTLASS sm90 pingpong kernel](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp)

**CUTLASS changelog**
- 4.5.1 (2026-05-15) added 128x32xK and 128x64xK tiles to the SM120 block-scaled builders, "up to 30% performance improvement on Blackwell SM121". 4.6.1 (2026-07-13) added tileN = 8, 16. — [CUTLASS 4.5.1 changelog](https://docs.nvidia.com/cutlass/4.5.1/CHANGELOG.html); [CUTLASS 4.6.1 changelog](https://docs.nvidia.com/cutlass/4.6.1/CHANGELOG.html)
- A community forum post says K=64 CTA shapes on SM120/121 allow 7-11 pipeline stages versus 2 with K=128, given the 99 KB smem. This is a third-party claim. — [NVIDIA forum, PSA state of FP4 on DGX Spark](https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069/131)

**Colfax tutorial (June 20, 2026; RTX PRO 6000, CUTLASS DSL `dense_gemm.py` blackwell_geforce example)**
- **Instruction.** `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3` is the only NVFP4 block-scaled form. Per thread, A holds 32 elements and B holds 16.
- **Tile and warps.** The CTA tile is 128x128x128 with 8 MMA warps in a (4,2,1) layout. The permutation is M=128, K=64, N=(8,2,2):(1,16,8).
- **SF lanes.** Threads 0/1 of each quad supply SFA and thread 0 supplies SFB; CUTLASS hard-codes the `{byte-id, thread-id}` selectors.
- **SF layouts.**
  - SFA TV layout: `((2,2,8),(16,4)):((32,0,4),(0,1))`, replicated 2x.
  - SFB TV layout: `((4,8),(16,4)):((0,4),(0,1))`, replicated 4x.
  - Per-lane SF register footprint: SFA 16 B and SFB 64 B per 128x128x128 tile.
  - SF smem layout: `(((32,4),REST),((16,4),1,REST_K)):(((16,4),512*REST_K),((0,1),4,512))`.
  - SF smem-to-register copy: `CopyUniversalOp` (not ldmatrix), using `filter_zeros`.
- **Performance.** The kernel reaches about 60% of the 2000 TFLOP/s dense FP4 peak.

— [Colfax SM12x tutorial](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/)

**Colfax optimization worklog (Aug 9, 2026; RTX PRO 6000 Server, 188 SMs, peak 2015.2 TFLOP/s FP4)**

| Version | Change | Result |
|---|---|---|
| V1 | Persistent kernel, 1 TMA warp + 8 MMA warps (4x2x1), 128x128x128 tile. setmaxregister: load warp 40, MMA warps 232. | 1476 TFLOP/s at 8k (about 73% of peak), about 93% of cuBLAS 13.6 on average. Large shapes throttle to about 2.15 GHz (clock-adjusted peak about 1782). |
| V2 | Threadblock swizzle 16 | +387 TFLOP/s at 32k |
| V3/V4 | Epilogue fixes; dedicated store warp | About +1% or less |
| V5 | SFA smem layout changed from (32,4):(16,4) to (32,4):(4,128), removing the 8.39M "excessive wavefronts", all from the SFA smem-to-register copy | Negligible on its own |
| V6 | 12 MMA warps (6x2x1) on a 192x128 tile; per-warp work stays 96x16x64 per atom, repeated 2(M)x8(N)x2(K). Stages fall from 4 to 2 because of smem. | +186 TFLOP/s at 2k (wave quantization); more than +40 TFLOP/s at 16k and 32k |
| V7 | Autotune | Final peak 1666 TFLOP/s at 16k (83%) |
| Appendix | Restricting SF loads to only the lanes the MMA reads, or toggling the selectors via inline PTX | No gain |

— [Colfax SM120 optimization](https://research.colfax-intl.com/optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/)

**Other measured numbers**
- **RTX 5090** (CUDA 13.2, cuBLASLt 13.4, CUTLASS 4.2.1), 512x4096x8192:
  - CUTLASS reference: 1216.8 TFLOP/s.
  - Custom CuTe kernel (384-thread producer/consumer warp specialization, 3-stage TMA, persistent, 128x128x128): 1208.4 TFLOP/s.
  - cuBLASLt (heuristic id 70): 1145.4 TFLOP/s.
  - The SASS instruction is `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`.

  — [xinyang-zhou/sm120_nvfp4_ops](https://github.com/xinyang-zhou/sm120_nvfp4_ops)
- **DGX Spark (sm_121a)**, CUTLASS 4.4.0 C++ builder, TN, cluster 1x1x1: 356 TFLOP/s on 4096x14336x4096, which the author puts at about 71% of a 500 TFLOP/s dense FP4 peak. GB10 smem is 101,376 B. — [NVIDIA forum, SM121 CUTLASS results](https://forums.developer.nvidia.com/t/sm121-cutlass-kernel-optimization-results-nvfp4-356-tflops-moe-grouped-gemm-on-dgx-spark/359960)
- **Tiles reported as safe within 99 KB smem:** [128,128,128], [128,128,256], [256,128,128], [128,256,128]. The same source says stages are typically 2-4. — [lna-lab sm120-architecture.md](https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm/blob/main/docs/sm120-architecture.md)

### Inferences
- **The structural difference from our kernel is warp count per scheduler, not per-warp tile size.**
  - CUTLASS Cooperative and Colfax V1 use 8 MMA warps, that is 2 per SM sub-partition, the same as ours.
  - The per-warp tile is small: 128x128 / 8 warps = 2048 FP32 accumulators per warp, about 64 registers per thread.
  - Colfax's best kernel went to 3 MMA warps per scheduler while keeping that per-warp size.
  - If our per-warp tile is larger than 32x64, a smaller warp tile with more warps may hide OMMA and LDS latency better than growing the tile. A 64x64 warp tile already spilled.
- **SF loads.** Neither CUTLASS nor Colfax uses ldmatrix for SFs; both use vectorized LDS of a small per-lane fragment, issued one k-block ahead. Colfax found that fixing SFA bank conflicts and trimming SF replication gave essentially nothing. SF fragment cost is therefore unlikely to explain a 20-point utilization gap by itself.
- **The best published sm120 NVFP4 figure is 83% of nominal peak** (about 93% of clock-adjusted peak at the throttled 2.15 GHz). Reaching about 90% tensor-pipe-busy under locked clocks is consistent with cuBLAS-class kernels.

### Gaps
- I did not find an official CUTLASS performance table for example 79 (block-scaled sm120 GEMM) on RTX 5090.
- The bodies of `sm120_rr_smem_copy_selector` and `sm120_tile_n_permute_selector`, which pick the A/B ldmatrix variant and the N permutation, were not fetched.

## Q2. What cuBLASLt does on sm_120

### Takeaway
No public SASS reverse-engineering of cuBLASLt's sm_120 FP4 kernels turned up. Colfax reports that on RTX PRO 6000, cuBLAS 13.5 dispatched NVFP4 only to CUTLASS-backend kernels, while 13.6 used nvjet at 2k and 32k. On RTX 5090 with cuBLASLt 13.4, plain CUTLASS and a custom CuTe kernel beat cuBLASLt's best heuristic by about 6% at 512x4096x8192.

### Cited Findings
- "cuBLAS 13.5 exclusively dispatched to cutlass backend kernels"; 13.6 switched to nvjet at 2k and 32k and to a different CUTLASS kernel at 16k. Colfax V1 averaged about 93% of cuBLAS 13.6. — [Colfax SM120 optimization](https://research.colfax-intl.com/optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/)
- On RTX 5090 the cuBLASLt 13.4 best heuristic (id 70) reached 1145 TFLOP/s versus 1217 for the CUTLASS reference at 512x4096x8192. — [xinyang-zhou/sm120_nvfp4_ops](https://github.com/xinyang-zhou/sm120_nvfp4_ops)
- One repo states "cuBLAS has no FP4 GEMM on sm_120" (CUDA 12.8 era, RTX PRO 6000 Max-Q). This conflicts with the Colfax and xinyang-zhou results; FP4 support in cuBLASLt was probably added in later versions. — [waynehacking8/blackwell-tensorcore-kernels](https://github.com/waynehacking8/blackwell-tensorcore-kernels)
- A separate forum report describes an RTX 5090 cuBLASLt nvjet TF32 split-K kernel raising "Warp Barrier Arrival Mismatch", which shows nvjet kernels do run on sm_120. — [NVIDIA forum](https://forums.developer.nvidia.com/t/cublaslt-sm-120-blackwell-tf32-split-k-nvjet-kernel-raises-warp-barrier-arrival-mismatch-intermittent-illegal-access-gpu-hang-rtx-5090/372787)

### Inferences
- cuBLASLt's sm_120 FP4 path is at least partly CUTLASS-derived, so the CUTLASS Cooperative and Pingpong structure in Q1 is the best public proxy for what reaches about 90% tensor-pipe busy.
- Our own observation (cuBLASLt about 90% busy under locked clocks, our kernel at or above cuBLASLt at real clocks) is consistent with the 5090 data. At boost clocks these kernels become memory- or clock-bound, which narrows the gap. At low locked clocks, issue efficiency dominates.

### Gaps
- No `cuobjdump`/nvdisasm dump of a cuBLASLt sm_120 FP4 kernel was found. The kernel name, warp count, tile and per-warp schedule are unknown publicly. Dumping the kernel cuBLASLt picks for our shapes (via `nsys`/`ncu` kernel name, then `cuobjdump -sass libcublasLt.so`) would be the direct way to answer this.
- Not investigated within this budget: Triton block-scaled `tl.dot_scaled` on sm_120, ThunderKittens on consumer Blackwell, DeepGEMM forks, and the FlashInfer/vLLM sm120 FP4 GEMM internals. lna-lab only documents patches for vLLM, FlashInfer and CUTLASS.

## Q3. SASS-level facts and tricks (OMMA issue/latency, reuse, bank conflicts, LDSM interleave, 4 vs 2 MMA warps, cp.async vs TMA)

### Takeaway
The SASS opcode is `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`, with operands D, A, B, C, SFA, SFB, URZ. Its serial dependent-chain latency is about 29 cycles, lower than HMMA and QMMA (about 35). ptxas puts `.reuse` on B and SFB within OMMA chains. With about 1024 dense FP4 MACs worth of work per sub-partition per clock, one m16n8k64 OMMA occupies a sub-partition's tensor core for about 16 cycles. Only about 2 independent accumulators are needed to cover latency, which leaves about 15 issue slots per OMMA for LDSM, LDS and address math. No public source measures OMMA issue throughput directly.

### Cited Findings
- **Latencies (dependent chains; explicitly not throughput):**
  - OMMA: about 29 cycles per `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`.
  - HMMA.16816.F32: about 35 cycles.
  - QMMA.16832 E4M3: about 35 cycles.
  - LDSM.16.M88: about 33 cycles.

  Throughput for all of these is listed as unmeasured. — [sass-king FINDINGS.md](https://raw.githubusercontent.com/florianmattana/sass-king/main/knowledge/FINDINGS.md)
- **Reuse flags.** In OMMA chains `.reuse` appears on both B and SFB for non-terminal instructions, for example `R2.reuse` (B) and `R8.reuse` (SF). The scoreboard coordinates the LDSM-to-MMA handoff with zero explicit NOPs. — [sass-king FINDINGS.md](https://raw.githubusercontent.com/florianmattana/sass-king/main/knowledge/FINDINGS.md)
- **Packed vs unpacked FP4.**
  - Packed block-scaled FP4 (`kind::mxf4`, OMMA.SF) reached 988.3 TFLOP/s at 8192^3 on RTX PRO 6000 Max-Q (300 W). That is 56% of a derived 1761 TFLOP/s peak.
  - FP4 in 8-bit containers (`kind::f8f6f4`, QMMA) reached only 517.8, the same pipe and rate as FP8.
  - The kernel: 128x128 CTA, 64x64 warp tile, XOR-swizzled smem, 2-stage cp.async, with scales fixed at 1.0.
  - The measured per-SM rate is 1023.8 FLOP/SM/clk for FP16 with FP32 accumulate, and the derived peaks are FP16 about 440, FP8 about 880, MXFP4 about 1761 TFLOP/s.
  - For FP8 on the same skeleton, register-pipelined fragments gave 493 to 502 TFLOP/s; larger tiles and deeper pipelines were dead ends.

  — [waynehacking8/blackwell-tensorcore-kernels](https://github.com/waynehacking8/blackwell-tensorcore-kernels)
- **SFA bank conflicts** under the default CUTLASS SF layout produced 8,388,608 excess L1 wavefronts (= 64x64x64x8x4). They were fixed by an SFA smem stride of (32,4):(4,128), giving one wavefront per load, with negligible end-to-end effect. — [Colfax SM120 optimization](https://research.colfax-intl.com/optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/)
- **More MMA warps.** Going from 8 to 12 MMA warps (2 to 3 per scheduler) at constant per-warp work was Colfax's largest non-swizzle gain, despite stages dropping from 4 to 2. — [Colfax SM120 optimization](https://research.colfax-intl.com/optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/)
- **CUTLASS Pingpong** keeps only one consumer warpgroup (4 warps, 1 per scheduler) in the mainloop at a time, overlapping its epilogue with the other warpgroup's mainloop. Cooperative keeps all 8 consumer warps in the mainloop. — [CUTLASS pingpong kernel](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp); [builder](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl)
- **TMA vs cp.async.** All of the fastest sm120 FP4 kernels found use TMA with `PipelineTmaAsync`: CUTLASS, Colfax and xinyang-zhou. The only cp.async kernel found (waynehacking8) is at 56% of peak. — sources above
- **Compiler register pressure.** nvcc 13.2 reportedly uses far more registers on sm_120 than on sm_90 for the same kernel.
  - One kernel: 64 registers on sm_90 versus 64 registers plus 1368 B spill stores on sm_120 under `__launch_bounds__(1024,1)`.
  - Another kernel: 167 registers on sm_90 versus 255 plus heavy spills on sm_120.
  - The poster says nvcc 12.8 spills less. No NVIDIA reply was posted.

  — [NVIDIA forum, nvcc 13.2 spilling on cc 120](https://forums.developer.nvidia.com/t/nvcc-13-2-spilling-many-registers-for-cc-120/364169)

### Inferences
- **Arithmetic** (from the waynehacking8 1024 FLOP/SM/clk FP16 figure times 4 for FP4, with 4 sub-partitions):
  - Each sub-partition does about 1024 FP4 FLOP/clk.
  - An m16n8k64 OMMA is 16,384 FLOP, so about 16 cycles of tensor-pipe time per OMMA per sub-partition.
  - The 29-cycle latency is covered by 2 or more independent accumulator tiles in flight; any warp tile of 2x2 OMMAs or larger satisfies this.
  - Raw issue slots are therefore not the bottleneck. Lost utilization more likely comes from serialization points that stall all MMA warps of a scheduler at once:
    - the per-stage `NamedBarrier::sync` before `consumer_release`;
    - `consumer_wait` on mbarriers;
    - LDSM/LDS results not yet back at the first OMMA of a new stage, since 33-cycle LDSM latency is exposed if the next stage's first fragments are not prefetched across the stage boundary;
    - epilogue gaps.
  - More warps per scheduler (Colfax 12-warp) and Pingpong-style epilogue overlap both target exactly these stalls.
- **Candidate experiments, ordered by evidence:**
  1. 3 MMA warps per scheduler (12 warps, 192x128 or similar) at the same or smaller per-warp tile.
  2. Prefetch k-block 0 of stage s+1 before the last OMMAs of stage s, so LDSM latency does not show at stage boundaries.
  3. Replace the CTA-wide named barrier before releasing a stage with per-warp mbarrier arrivals (consumer arrive count = number of MMA warps).
  4. Check the SASS for `.reuse` on B/SFB and that the accumulator chain order keeps B fixed across consecutive OMMAs. Order the inner loop N-outer/M-inner or vice versa so the reused operand changes least often.
  5. Pin the SFA smem layout to the conflict-free (32,4):(4,128) form. It is cheap, though Colfax saw no gain from it alone.
- **Compiler drift.** If our kernel spills at 64x64, nvcc 13.x sm_120 register-allocation behaviour may be a factor. Comparing SASS register counts across nvcc 12.8, 12.9 and 13.x is a cheap check.

### Gaps
- No public OMMA.SF throughput or issue-interval microbenchmark (independent chains) exists; sass-king lists it as open.
- There is no public data on register bank conflicts specific to OMMA's 7-operand form (D, A, B, C, SFA, SFB, URZ), and no measured effect of interleaving LDSM with OMMA on sm_120.
- No source compares 4 versus 2 MMA warps per scheduler at fixed total CTA tile on sm_120, apart from Colfax's 8-to-12 change, which also changed the CTA tile and the stage count.

## Q4. Is setmaxnreg dropped or unsupported on sm_120? __grid_constant__ CUtensorMap interactions

### Takeaway
setmaxnreg is supported and actively used on sm_120: CUTLASS's sm120-family kernels and Colfax's CuTe DSL sm120 kernel both rely on it (producer 40 or 24, MMA 232 or 240). It needs an arch-specific target (sm_120a or sm_120f). No source reported problems specific to `__grid_constant__` CUtensorMap on sm_120.

### Cited Findings
- libcudacxx PTX docs list `setmaxnreg.inc/.dec` as available on SM_120a and SM_120f (PTX ISA 8.0+). — [CCCL setmaxnreg docs](https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/setmaxnreg.html)
- A CICC-internals reference says sm_120 inherits "Hopper cluster operations, TMA bulk copy, setmaxnreg" but uses synchronous `mma.sync` instead of async warpgroup MMA. — [nvopen-tools sm120](https://grigoryevko.github.io/nvopen-tools/cicc/targets/sm120.html)
- The CUTLASS pingpong kernel, compiled for the 1200 family, calls `warpgroup_reg_dealloc<40|24>` and `warpgroup_reg_alloc<232|240>`. — [CUTLASS pingpong kernel](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp)
- Colfax V1 on SM120: "load warp decreases to 40; MMA warps increase to 232". — [Colfax SM120 optimization](https://research.colfax-intl.com/optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/)
- **Known pitfalls (reported on sm_90a; I found nothing equivalent for sm_120):**
  - ptxas can warn "'setmaxnreg' ignored; unable to determine register count at entry". — [NVIDIA forum](https://forums.developer.nvidia.com/t/setmaxnreg-ignored-unable-to-determine-register-count-at-entry/361145)
  - It can conflict with `__launch_bounds__` forcing a max register count. — [NVIDIA forum](https://forums.developer.nvidia.com/t/hopper-launch-bounds-and-setmaxnreg-conflicts/317408/2)

### Inferences
- With 1 producer warp plus 8 MMA warps, setmaxnreg cannot push the MMA warps above 255 registers per thread, the architectural per-thread limit. It only helps if the launch otherwise caps registers below 255 (for example 3 warpgroups at 168).
- setmaxnreg operates per warpgroup. If our kernel is not organized as warpgroups of 4 warps with uniform roles, it cannot be applied cleanly.
- The fix for the 64x64 spill is a smaller per-warp tile plus more warps (the Colfax/CUTLASS pattern), not more registers.

### Gaps
- No source confirms or refutes `setmaxnreg` being silently ignored on RTX 5090 specifically. Checking `ptxas -v` warnings and SASS for `USETMAXREG` would settle it.
- No source discussed `__grid_constant__ const CUtensorMap` behaviour (or tensormap prefetch, `prefetch.tensormap`) specific to sm_120.
