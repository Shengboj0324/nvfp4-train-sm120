// TMA-fed NVFP4 GEMM main loop for sm_120a. A, B and their scales stream into shared memory through
// TMA (cp.async.bulk.tensor, issued by one thread), eight MMA warps consume them, and mbarriers hand
// buffers back and forth. Compared with the cp.async loop in gemm.cu this removes the per-thread copy
// instructions (which throttled the shared-memory instruction queue), uses 256-deep K tiles and keeps
// the tensor pipe fed by double-buffering operand fragments in registers across k-steps and stages.
//
// Shared memory (1024-byte aligned for the swizzles), 96 KB, for the default 256-deep k-tiles:
//   data stage s (x2)   A [128 rows][128 B], B [128 rows][128 B]: one k-tile, 32 KB
//                       TMA box {128 B, 128 rows}, SWIZZLE_128B (16B chunk c of row r at c ^ (r & 7))
//   scale quad q (x2)   SA [128 rows][64 B], SB [128 rows][64 B]: scales of 4 k-tiles, 16 KB
//                       TMA box {64 B, 128 rows}, SWIZZLE_64B (16B chunk c of row r at c ^ ((r >> 1) & 3))
// Scales are fetched 4 k-tiles at a time: a scale row of one k-tile is only 16 bytes, and 128 such
// rows per operand per k-tile cost 2x the L2 requests of the data itself (measured: the kernel ran
// 1.22x slower than with scale loads disabled). 64-byte rows cut that to a quarter.
// NVFP4_TMA_BK=128 builds 4 stages of 128-deep k-tiles in the same 64 KB (64B swizzle); it measured
// within 2% of the default (docs/DESIGN.md §4), so pipeline depth is not what limits this kernel.
//
// Requires K % 256 == 0 (TMA needs 16-byte global strides for the scale rows) and 16-byte aligned
// operands; the host falls back to the cp.async kernel otherwise.

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include "gemm_common.cuh"

namespace nvfp4 {
namespace gemm {
namespace tma {

#ifndef NVFP4_TMA_BK
#define NVFP4_TMA_BK 256
#endif
constexpr int BK = NVFP4_TMA_BK;        // K elements per k-tile (128 or 256)
constexpr int KS = BK / 64;             // MMA k-steps per k-tile
constexpr int BKB = BK / 2;             // data bytes per row per k-tile (64 or 128)
constexpr int SFK = BK / 16;            // scale bytes per row per k-tile (8 or 16)
constexpr int QB = 64;                  // scale bytes per row per scale load
constexpr int QUAD = QB / SFK;          // k-tiles per scale load (8 or 4)
constexpr int STAGES = 256 / BKB;       // data stages: 64 KB of A/B either way (4 or 2)
constexpr int A_BYTES = BM * BKB, B_BYTES = BN * BKB;
constexpr int STAGE_BYTES = A_BYTES + B_BYTES;           // 32 KB (16 KB at BK = 128)
constexpr int SA_BYTES = BM * QB, SB_BYTES = BN * QB;
constexpr int QUAD_BYTES = SA_BYTES + SB_BYTES;          // 16 KB
constexpr int SCALE_BASE = STAGES * STAGE_BYTES;         // quads follow the data stages
constexpr int BAR_BASE = SCALE_BASE + 2 * QUAD_BYTES;    // full[STAGES], empty[STAGES], sfull[2], flag
// 8 MMA warps and no dedicated producer: thread 0 issues the TMA copies. A 9th warp would put 3 warps
// on one scheduler and cap every thread at 168 registers (3 x 32 x R <= 16K), which made the MMA warps
// spill and lose their fragment double-buffering; setmaxnreg could rebalance that, but ptxas drops it
// in kernels that take the address of a __grid_constant__ tensor map.
constexpr int THREADS = MMA_THREADS;
constexpr int SMEM_BYTES = BAR_BASE + (2 * STAGES + 2) * 8 + 16 /*flag*/ + 1024 /*alignment slack*/;
static_assert(SMEM_BYTES <= 101376, "sm_120 allows at most 99 KB of dynamic shared memory per block");

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
// try_wait with a suspend-time hint: a waiting warp sleeps instead of spinning on issue slots.
__device__ __forceinline__ uint64_t global_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
  return t;
}
// Bounded: a wait longer than 1 s means a pipeline bug, and an unbounded spin would instead run into the
// OS display-driver watchdog (2 s on Windows/WSL), whose GPU reset is far more disruptive than a trap,
// which surfaces as an ordinary CUDA launch failure.
constexpr uint64_t kWaitLimitNs = 1000000000ull;
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t done;
  uint64_t t0 = 0;
  for (;;) {
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(done)
        : "r"(bar), "r"(parity), "r"(0x989680)
        : "memory");
    if (done) return;
    if (t0 == 0) {
      t0 = global_ns();
    } else if (global_ns() - t0 > kWaitLimitNs) {
      __trap();
    }
  }
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
          dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(bar)
      : "memory");
}

struct Maps {
  const CUtensorMap *a, *b, *sa, *sb;
};

// Barrier addresses (8 bytes each).
__device__ __forceinline__ uint32_t full_bar(uint32_t bars, int s) { return bars + s * 8; }
__device__ __forceinline__ uint32_t empty_bar(uint32_t bars, int s) { return bars + (STAGES + s) * 8; }
__device__ __forceinline__ uint32_t sfull_bar(uint32_t bars, int q) { return bars + (2 * STAGES + q) * 8; }

// Load k-tile kt of A and B into data stage s.
__device__ __forceinline__ void issue_stage(uint32_t base, uint32_t bars, int s, int kt, int m0, int n0,
                                            const Maps& mp) {
  const uint32_t stage = base + s * STAGE_BYTES, full = full_bar(bars, s);
  mbar_expect_tx(full, STAGE_BYTES);
  tma_load_2d(stage, mp.a, kt * BKB, m0, full);
  tma_load_2d(stage + A_BYTES, mp.b, kt * BKB, n0, full);
}

// Load the scales of local k-tiles 4q .. 4q+3 into quad buffer q & 1 (past the end of K the box is
// zero-filled; those bytes are never read).
__device__ __forceinline__ void issue_quad(uint32_t base, uint32_t bars, int q, int kt0, int m0, int n0,
                                           const Maps& mp) {
  const uint32_t buf = base + SCALE_BASE + (q & 1) * QUAD_BYTES, full = sfull_bar(bars, q & 1);
  const int x = (kt0 + QUAD * q) * SFK;
  mbar_expect_tx(full, QUAD_BYTES);
  tma_load_2d(buf, mp.sa, x, m0, full);
  tma_load_2d(buf + SA_BYTES, mp.sb, x, n0, full);
}

// Operand fragments of one 64-deep k-step for one warp.
struct Frag {
  uint32_t a[MI][4];
  uint32_t b[NI][2];
};

// Per-lane shared-memory addresses, precomputed once. With the 128B swizzle, chunk c of row r sits at
// c ^ (r & 7), and r & 7 == lane & 7 for every row a lane addresses, so the swizzled offset of each
// k-step is a per-lane constant; likewise (r >> 1) & 3 for the scale rows under the 64B swizzle.
// Stage s and tile i / j then only add compile-time immediates.
struct LaneAddr {
  uint32_t a[KS];       // A ldmatrix address of k-step ks in stage 0, tile i = 0
  uint32_t b[KS];       // B ldmatrix address of k-step ks in stage 0, tiles j = 0, 1
  uint32_t sa, sb;      // start of the scale row of tile i = 0 / j = 0 in quad buffer 0
  uint32_t sx;          // scale-row swizzle term (r >> 1) & 3
};

__device__ __forceinline__ void lds128(uint32_t addr, uint4& v) {
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
}

template <int S>
__device__ __forceinline__ void load_frag(Frag& f, const LaneAddr& la, int ks) {
#pragma unroll
  for (int i = 0; i < MI; ++i)
    ldmatrix_x4(la.a[ks] + S * STAGE_BYTES + i * 16 * BKB, f.a[i][0], f.a[i][1], f.a[i][2], f.a[i][3]);
#pragma unroll
  for (int j = 0; j < NI; j += 2)
    ldmatrix_x4(la.b[ks] + S * STAGE_BYTES + j * 8 * BKB, f.b[j][0], f.b[j][1], f.b[j + 1][0], f.b[j + 1][1]);
}

__device__ __forceinline__ void lds64(uint32_t addr, uint32_t& x, uint32_t& y) {
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(x), "=r"(y) : "r"(addr));
}

// Scale words (one per k-step) of the tiles this lane needs.
struct Scales {
  uint32_t a[MI][KS], b[NI][KS];
};

// Scales of local k-tile `it` (SFK bytes per row = one word per k-step), waiting for its scale load
// first if `it` starts one. The 64-byte scale rows are stored with the 64B swizzle (16B chunk c of row
// r at c ^ ((r >> 1) & 3)); a k-tile's SFK bytes lie inside one 16B chunk.
__device__ __forceinline__ void load_scales(Scales& sc, const LaneAddr& la, uint32_t bars, int it) {
  const int q = it / QUAD, t = it % QUAD;
  if (t == 0) mbar_wait(sfull_bar(bars, q & 1), (q / 2) & 1);
  const uint32_t byte = t * SFK;
  const uint32_t off = (q & 1) * QUAD_BYTES + ((((byte >> 4) ^ la.sx) << 4) | (byte & 15));
#pragma unroll
  for (int i = 0; i < MI; ++i) {
    if constexpr (KS == 4) {
      uint4 v;
      lds128(la.sa + off + i * 16 * QB, v);
      sc.a[i][0] = v.x, sc.a[i][1] = v.y, sc.a[i][2] = v.z, sc.a[i][3] = v.w;
    } else {
      lds64(la.sa + off + i * 16 * QB, sc.a[i][0], sc.a[i][1]);
    }
  }
#pragma unroll
  for (int j = 0; j < NI; ++j) {
    if constexpr (KS == 4) {
      uint4 v;
      lds128(la.sb + off + j * 8 * QB, v);
      sc.b[j][0] = v.x, sc.b[j][1] = v.y, sc.b[j][2] = v.z, sc.b[j][3] = v.w;
    } else {
      lds64(la.sb + off + j * 8 * QB, sc.b[j][0], sc.b[j][1]);
    }
  }
}

// Consume data stage S (k-tile `it`): KS k-steps, each issuing its MMAs after the next k-step's
// fragments are requested; the last k-step prefetches the first k-step (and scales) of k-tile it + 1.
// The stage is released as soon as its last fragments are requested: mbarrier.arrive has release
// semantics, so those shared-memory reads are performed before the producer can observe the arrival.
template <int S>
__device__ __forceinline__ void consume_stage(float (&acc)[MI][NI][4], Frag (&fr)[2], Scales& sc, const LaneAddr& la,
                                              uint32_t bars, int it, int nk, int lane) {
  constexpr int SN = (S + 1) % STAGES;
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    uint32_t sa[MI], sb[NI];
#pragma unroll
    for (int i = 0; i < MI; ++i) sa[i] = sc.a[i][ks];
#pragma unroll
    for (int j = 0; j < NI; ++j) sb[j] = sc.b[j][ks];
    if (ks + 1 < KS) {
      load_frag<S>(fr[(ks + 1) & 1], la, ks + 1);
      if (ks + 2 == KS) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty_bar(bars, S));
        __syncwarp();  // reconverge before the next .aligned instruction
      }
    } else if (it + 1 < nk) {
      mbar_wait(full_bar(bars, SN), ((it + 1) / STAGES) & 1);
      load_scales(sc, la, bars, it + 1);
      load_frag<SN>(fr[0], la, 0);
    }
    // (B-reuse and serpentine orders measured the same.)
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
      for (int j = 0; j < NI; ++j) mma_nvfp4(acc[i][j], fr[ks & 1].a[i], fr[ks & 1].b[j], sa[i], sb[j]);
  }
}

// Thread 0, after consuming local k-tile `it` (data stage S): once all warps released it, refill stage
// S with k-tile it + STAGES and, when `it` ends a scale load, refill that buffer with the one after next.
template <int S>
__device__ __forceinline__ void refill(uint32_t base, uint32_t bars, int it, int nk, int kt0, int m0, int n0,
                                       const Maps& mp) {
  if (it + STAGES >= nk) return;  // nothing left to load (later scale loads start even later)
  mbar_wait(empty_bar(bars, S), (it / STAGES) & 1);
  issue_stage(base, bars, S, kt0 + it + STAGES, m0, n0, mp);
  // All warps have released k-tile `it`; its scales (and those of every earlier k-tile) were read
  // before its fragments, so the scale buffer of a finished load can be refilled.
  if (it % QUAD == QUAD - 1 && QUAD * (it / QUAD + 2) < nk) issue_quad(base, bars, it / QUAD + 2, kt0, m0, n0, mp);
}

// k-tiles it + S, it + S + 1, ... it + STAGES - 1 (stage index is a compile-time constant).
template <int S>
__device__ __forceinline__ void steps(float (&acc)[MI][NI][4], Frag (&fr)[2], Scales& sc, const LaneAddr& la,
                                      uint32_t base, uint32_t bars, int it, int nk, int kt0, int m0, int n0, int lane,
                                      const Maps& mp) {
  if constexpr (S < STAGES) {
    if (it + S < nk) {
      consume_stage<S>(acc, fr, sc, la, bars, it + S, nk, lane);
      if (threadIdx.x == 0) refill<S>(base, bars, it + S, nk, kt0, m0, n0, mp);
      // Lane 0 of warp 0 may have waited on the empty barrier above; ldmatrix / mma.sync are .aligned and
      // need the whole warp converged, which independent thread scheduling does not guarantee by itself.
      __syncwarp();
      steps<S + 1>(acc, fr, sc, la, base, bars, it, nk, kt0, m0, n0, lane, mp);
    }
  }
}

template <typename OutT, bool kAccum>
__global__ void __launch_bounds__(THREADS, 1)
    nvfp4_gemm_tma_kernel(Args p, const __grid_constant__ CUtensorMap tm_a, const __grid_constant__ CUtensorMap tm_b,
                          const __grid_constant__ CUtensorMap tm_sa, const __grid_constant__ CUtensorMap tm_sb) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t base = smem_u32(smem);
  const uint32_t bars = base + BAR_BASE;
  int* flag = reinterpret_cast<int*>(smem + BAR_BASE + (2 * STAGES + 2) * 8);
  const Maps mp{&tm_a, &tm_b, &tm_sa, &tm_sb};

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int KT = p.K / BK;
  const int kt0 = static_cast<int>((static_cast<int64_t>(KT) * blockIdx.z) / p.splits);
  const int kt1 = static_cast<int>((static_cast<int64_t>(KT) * (blockIdx.z + 1)) / p.splits);
  const int nk = kt1 - kt0;

  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(full_bar(bars, s), 1);
      mbar_init(empty_bar(bars, s), WARPS_M * WARPS_N);
    }
    for (int q = 0; q < 2; ++q) mbar_init(sfull_bar(bars, q), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    for (int q = 0; q < 2 && QUAD * q < nk; ++q) issue_quad(base, bars, q, kt0, m0, n0, mp);
    for (int s = 0; s < STAGES && s < nk; ++s) issue_stage(base, bars, s, kt0 + s, m0, n0, mp);
  }
  __syncthreads();

  const int wm = warp / WARPS_N, wn = warp % WARPS_N;
  LaneAddr la;
  {
    const int x = BKB == 128 ? (lane & 7) : ((lane & 7) >> 1);  // swizzle term of every row this lane reads
    const int row_a = wm * WM + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int row_b = wn * WN + (lane & 7) + (lane >> 4) * 8;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      la.a[ks] = base + row_a * BKB + (((2 * ks + (lane >> 4)) ^ x) << 4);
      la.b[ks] = base + A_BYTES + row_b * BKB + (((2 * ks + ((lane >> 3) & 1)) ^ x) << 4);
    }
    const int g = lane >> 2;
    la.sa = base + SCALE_BASE + (wm * WM + g + 8 * (lane & 1)) * QB;
    la.sb = base + SCALE_BASE + SA_BYTES + (wn * WN + g) * QB;
    la.sx = (g >> 1) & 3;  // (row >> 1) & 3 for every scale row this lane reads
  }

  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.0f;

  Frag fr[2];
  Scales sc;
  if (nk > 0) {
    mbar_wait(full_bar(bars, 0), 0);
    load_scales(sc, la, bars, 0);
    load_frag<0>(fr[0], la, 0);
  }
  for (int it = 0; it < nk; it += STAGES) steps<0>(acc, fr, sc, la, base, bars, it, nk, kt0, m0, n0, lane, mp);

  finish_tile<OutT, kAccum>(p, acc, m0, n0, wm, wn, lane, flag);
}

using EncodeFn = PFN_cuTensorMapEncodeTiled_v12000;

EncodeFn encode_fn() {
  static EncodeFn fn = [] {
    void* f = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &f, 12000, cudaEnableDefault, &q) != cudaSuccess ||
        q != cudaDriverEntryPointSuccess)
      f = nullptr;
    return reinterpret_cast<EncodeFn>(f);
  }();
  return fn;
}

// 2D uint8 map over a row-major [rows, cols] byte matrix, box {box_cols, box_rows}.
bool make_map(CUtensorMap* map, const void* ptr, int64_t rows, int64_t cols, int box_cols, int box_rows,
              CUtensorMapSwizzle swizzle) {
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(cols), static_cast<cuuint64_t>(rows)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(cols)};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(box_cols), static_cast<cuuint32_t>(box_rows)};
  const cuuint32_t estr[2] = {1, 1};
  return encode_fn()(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                     CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <typename OutT, bool kAccum>
void launch(const Args& p, const CUtensorMap (&maps)[4], dim3 grid, cudaStream_t stream) {
  static const bool attr = (cudaFuncSetAttribute(nvfp4_gemm_tma_kernel<OutT, kAccum>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES),
                            true);
  (void)attr;
  nvfp4_gemm_tma_kernel<OutT, kAccum><<<grid, THREADS, SMEM_BYTES, stream>>>(p, maps[0], maps[1], maps[2], maps[3]);
}

}  // namespace tma

int tma_k_tile() { return tma::BK; }

bool tma_supported(const Args& p) {
  auto aligned = [](const void* q) { return (reinterpret_cast<uintptr_t>(q) & 15) == 0; };
  // K % 256: the scale rows (K/16 bytes) must have a 16-byte global stride for TMA.
  return p.K % 256 == 0 && p.K % tma::BK == 0 && aligned(p.a) && aligned(p.b) && aligned(p.sfa) && aligned(p.sfb) &&
         tma::encode_fn() != nullptr;
}

// Returns false (nothing launched) if the operands can't be described as tensor maps.
bool launch_tma(const Args& p, bool out_fp32, bool accumulate, dim3 grid, cudaStream_t stream) {
  using namespace tma;
  if (!tma_supported(p)) return false;
  CUtensorMap maps[4];
  const CUtensorMapSwizzle swz = BKB == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;
  const bool ok = make_map(&maps[0], p.a, p.M, p.K / 2, BKB, BM, swz) &&
                  make_map(&maps[1], p.b, p.N, p.K / 2, BKB, BN, swz) &&
                  make_map(&maps[2], p.sfa, p.M, p.K / 16, QB, BM, CU_TENSOR_MAP_SWIZZLE_64B) &&
                  make_map(&maps[3], p.sfb, p.N, p.K / 16, QB, BN, CU_TENSOR_MAP_SWIZZLE_64B);
  if (!ok) return false;
  if (out_fp32) {
    if (accumulate)
      launch<float, true>(p, maps, grid, stream);
    else
      launch<float, false>(p, maps, grid, stream);
  } else {
    launch<__nv_bfloat16, false>(p, maps, grid, stream);
  }
  return true;
}

}  // namespace gemm
}  // namespace nvfp4
