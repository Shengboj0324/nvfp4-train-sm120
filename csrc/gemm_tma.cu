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

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <c10/cuda/CUDAException.h>

#include "gemm_common.cuh"

namespace nvfp4 {
namespace gemm {
// SMs the persistent kernels leave free (ops.set_sm_margin): communication kernels running at the same time
// (expert-parallel all-to-alls) then find SMs instead of waiting for, or delaying, a GEMM that holds them all.
static std::atomic<int> g_sm_margin{0};
// Dynamic tile claiming in the persistent kernels (ops.set_dynamic_tiles; NVFP4_GEMM_DYNAMIC=1 starts it on).
static std::atomic<bool> g_dynamic_tiles{[] {
  const char* e = std::getenv("NVFP4_GEMM_DYNAMIC");
  return e != nullptr && std::string(e) == "1";
}()};

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
constexpr int BAR_BASE = SCALE_BASE + 2 * QUAD_BYTES;    // full[STAGES], empty[STAGES], sfull[2], flag,
                                                         // release counters, load cursors
// 8 MMA warps and no dedicated producer: thread 0 issues the TMA copies. A 9th warp would put 3 warps
// on one scheduler and cap every thread at 168 registers (3 x 32 x R <= 16K), which made the MMA warps
// spill and lose their fragment double-buffering; setmaxnreg could rebalance that, but ptxas drops it
// in kernels that take the address of a __grid_constant__ tensor map.
constexpr int THREADS = MMA_THREADS;
constexpr int CNT_OFF = BAR_BASE + (2 * STAGES + 2) * 8 + 16;  // unsigned[STAGES]: stage releases so far
// Dynamic tile claiming (Args::sched): the CTA's claimed work units, a ring of RING slots {seq tag, unit}
// after one int (the next sequence number to claim). The load cursors run ahead of the consumers by at most
// STAGES + 2 * QUAD k-tiles, and every claimed unit has at least one k-tile, so RING slots never overrun.
constexpr int RING = 32;
static_assert(RING > STAGES + 2 * QUAD + 2, "claim ring");
constexpr int SCHED_OFF = CNT_OFF + 64;
constexpr int SMEM_BYTES = SCHED_OFF + 16 + 8 * RING + 1024 /*alignment slack*/;
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

// The descriptors are passed to the kernel by value (__grid_constant__). CUDA 13 declares CUtensorMap
// alignas(128) (12.x: 64), and MSVC cannot pass a by-value parameter aligned beyond 64 bytes (C2719).
// TMA itself requires 64-byte-aligned descriptors, so under MSVC the kernel takes a 64-byte-aligned
// copy with the same layout (tested on the GPU through the CUDA 12.9 Windows build, which is 64 too).
#if defined(_MSC_VER)
struct alignas(64) TmaParam {
  cuuint64_t opaque[CU_TENSOR_MAP_NUM_QWORDS];
};
#else
using TmaParam = CUtensorMap;
#endif
static_assert(sizeof(TmaParam) == sizeof(CUtensorMap) && alignof(TmaParam) >= 64, "tensor map parameter layout");

// Barrier addresses (8 bytes each).
__device__ __forceinline__ uint32_t full_bar(uint32_t bars, int s) { return bars + s * 8; }
__device__ __forceinline__ uint32_t empty_bar(uint32_t bars, int s) { return bars + (STAGES + s) * 8; }
__device__ __forceinline__ uint32_t sfull_bar(uint32_t bars, int q) { return bars + (2 * STAGES + q) * 8; }

// --- Work units ---------------------------------------------------------------------------------------
// The kernel is persistent: CTA c processes work units c, c + gridDim.x, ... (one output tile, or one K
// split of a tile), in the order the hardware would launch non-persistent CTAs (n tiles fastest), so the
// tiles in flight at any time are the same neighbours as before. The data stages and scale quads form one
// ring across all of a CTA's units: while the MMA warps finish a tile (last k-steps, epilogue), thread 0
// is already loading the next tile's first k-tiles and scales, so the pipeline never drains between
// tiles (a non-persistent CTA paid the full load latency, plus launch and barrier setup, per tile).

// m0 / n0: the output tile; am0 / bn0: the A / B rows it reads (different only in grouped GEMMs); k0: first
// K element (a multiple of BK in grouped GEMMs: TMA needs 16-byte-aligned box starts in the scale rows);
// nk: k-tiles; half: bit 0: the last k-tile's second half, bit 1: the first k-tile's first half lies outside
// the unit's K range (ragged K with ranges on odd multiples of 128): those MMAs are skipped, the data
// loaded for them is never used.
struct Unit {
  int m0, n0, am0, bn0, k0, nk, half, z, tile;
};

template <bool G>
__device__ __forceinline__ Unit unit_of(const Args& p, int u, int tiles_n, int tiles) {
  Unit w;
  if (G && p.group_mode == 2) {  // `tiles` per group
    const int g = u / tiles, t = u % tiles;
    const int* sg = p.segs + 3 * g;
    const int len = __ldg(sg + 1);
    w.z = 0;
    w.tile = u;
    w.am0 = (t / tiles_n) * BM;
    w.n0 = w.bn0 = (t % tiles_n) * BN;
    w.m0 = __ldg(sg + 2) + w.am0;
    const int k0 = __ldg(sg), k1 = k0 + len;
    w.k0 = k0 / BK * BK;
    w.nk = (k1 - w.k0 + BK - 1) / BK;
    w.half = ((k1 % BK) != 0 ? 1 : 0) | ((k0 % BK) != 0 ? 2 : 0);
    return w;
  }
  w.z = u / tiles;
  w.tile = u % tiles;
  w.m0 = w.am0 = (w.tile / tiles_n) * BM;
  w.n0 = w.bn0 = (w.tile % tiles_n) * BN;
  // KT * splits < 2^31 (host check): 32-bit division, no 64-bit division call. A partial last k-tile (K % BK
  // != 0, grouped GEMMs only) reads zeros past the end of K (tensor-map out-of-bounds fill).
  const int KT = G ? (p.K + BK - 1) / BK : p.K / BK;
  const int kt0 = KT * w.z / p.splits;
  w.nk = KT * (w.z + 1) / p.splits - kt0;
  w.k0 = kt0 * BK;
  w.half = 0;
  if (G && p.group_mode == 1) {  // group < 0: an unused tile, stored as zeros without loading anything
    const int g = __ldg(p.tile_group + w.tile / tiles_n);
    if (g < 0)
      w.nk = 0;
    else
      w.bn0 = g * p.b_group_rows + w.n0;
  }
  return w;
}

__device__ __forceinline__ int quads_of(int nk) { return (nk + QUAD - 1) / QUAD; }

struct Grid {
  int tiles_n, tiles, total;
  // dynamic tile claiming (Args::sched): shared memory, [0] the next sequence number to claim, then per ring
  // slot s {tag, unit} at [4 + 2 s]: the CTA's n-th claimed unit sits in slot n % RING with tag n. Null: static.
  int* ring;
};

// The unit of sequence number n among this CTA's units under dynamic claiming. The first thread to need it
// claims it (n == 0: unit blockIdx.x; then the next unit of the global claim counter) and publishes it in the
// ring; the others read it there. Units without k-tiles (unused tiles of grouped GEMMs) are skipped: the
// consumers store their zero tiles after the claimed units. Returns >= total when no unit is left. (Measured
// alternatives: a non-inlined call cost the main loop register spills, 8-16%; thread 0 claiming one unit
// ahead and publishing at its next refill was slower than this, 7-14%; this costs 0-6% alone.)
template <bool G>
__device__ __forceinline__ int claimed_unit(const Args& p, const Grid& gd, int n) {
  volatile int* r = gd.ring;
  const int s = 4 + 2 * (n % RING);
  if (r[s] != n) {
    if (atomicCAS(gd.ring, n, n + 1) == n) {
      int u = n == 0 ? static_cast<int>(blockIdx.x) : static_cast<int>(gridDim.x) + atomicAdd(p.sched, 1);
      if constexpr (G) {
        while (u < gd.total && unit_of<G>(p, u, gd.tiles_n, gd.tiles).nk == 0)
          u = static_cast<int>(gridDim.x) + atomicAdd(p.sched, 1);
      }
      r[s + 1] = u;
      __threadfence_block();
      r[s] = n;
      return u;
    }
    while (r[s] != n) {
    }
  }
  __threadfence_block();
  return r[s + 1];
}

// The load cursors: the next k-tile and the next scale quad to load (unit, index within the unit, and the
// unit decoded once per unit, not per load), and their sequence numbers in the CTA's ring (sequence n goes
// to data stage n % STAGES, quad sequence q to quad buffer q & 1). The sequence of loads is fixed, so every
// warp (lane 0) keeps its own copy and advances it at the same points; whichever warp issues a load, all
// copies agree (no shared cursor state).
struct Producer {
  int du, dk, dseq, dn;  // data (dn: the unit's sequence number among the CTA's units, dynamic claiming)
  Unit dw;
  int qu, qq, qseq, qn;  // scales
  Unit qw;
};

// Grouped GEMMs have units without k-tiles (empty groups, unused tiles): they load nothing, so the load
// cursors skip them (the consumer still runs them, storing a zero tile).
template <bool G>
__device__ __forceinline__ void skip_empty(int& u, Unit& w, const Args& p, const Grid& gd) {
  if constexpr (G) {
    while (u < gd.total && w.nk == 0) {
      u += gridDim.x;
      if (u < gd.total) w = unit_of<G>(p, u, gd.tiles_n, gd.tiles);
    }
  }
}

// The next unit of a cursor: u + gridDim.x (static; skipping units without k-tiles), or the next claimed one.
template <bool G>
__device__ __forceinline__ void next_unit(int& u, int& n, Unit& w, const Args& p, const Grid& gd) {
  if (gd.ring != nullptr) {
    u = claimed_unit<G>(p, gd, ++n);
    if (u < gd.total) w = unit_of<G>(p, u, gd.tiles_n, gd.tiles);
  } else {
    u += gridDim.x;
    if (u < gd.total) w = unit_of<G>(p, u, gd.tiles_n, gd.tiles);
    skip_empty<G>(u, w, p, gd);
  }
}

template <bool G>
__device__ __forceinline__ void producer_init(Producer& pr, const Args& p, const Grid& gd) {
  pr.dk = pr.dseq = pr.qq = pr.qseq = 0;
  pr.dn = pr.qn = 0;
  pr.du = pr.qu = gd.ring != nullptr ? claimed_unit<G>(p, gd, 0) : static_cast<int>(blockIdx.x);
  if (pr.du < gd.total) pr.dw = pr.qw = unit_of<G>(p, pr.du, gd.tiles_n, gd.tiles);
  if (gd.ring == nullptr) {
    skip_empty<G>(pr.du, pr.dw, p, gd);
    skip_empty<G>(pr.qu, pr.qw, p, gd);
  }
}

// Load the next k-tile (if any; only if `issue`) into data stage dseq % STAGES and advance the cursor.
template <bool G>
__device__ __forceinline__ void issue_next_stage(Producer& pr, const Args& p, const Grid& gd, uint32_t base,
                                                 uint32_t bars, const Maps& mp, bool issue = true) {
  if (pr.du >= gd.total) return;
  const Unit& w = pr.dw;
  if (issue) {
    const int s = pr.dseq % STAGES;
    const uint32_t stage = base + s * STAGE_BYTES, full = full_bar(bars, s);
    mbar_expect_tx(full, STAGE_BYTES);
    tma_load_2d(stage, mp.a, w.k0 / 2 + pr.dk * BKB, w.am0, full);
    tma_load_2d(stage + A_BYTES, mp.b, w.k0 / 2 + pr.dk * BKB, w.bn0, full);
  }
  ++pr.dseq;
  if (++pr.dk == w.nk) {
    pr.dk = 0;
    next_unit<G>(pr.du, pr.dn, pr.dw, p, gd);
  }
}

// Load the scales of the next 4 k-tiles (one quad; past the end of K the box is zero-filled and those
// bytes are never read) into quad buffer qseq & 1 (only if `issue`) and advance the cursor.
template <bool G>
__device__ __forceinline__ void issue_next_quad(Producer& pr, const Args& p, const Grid& gd, uint32_t base,
                                                uint32_t bars, const Maps& mp, bool issue = true) {
  if (pr.qu >= gd.total) return;
  const Unit& w = pr.qw;
  if (issue) {
    const uint32_t buf = base + SCALE_BASE + (pr.qseq & 1) * QUAD_BYTES, full = sfull_bar(bars, pr.qseq & 1);
    const int x = w.k0 / 16 + QUAD * pr.qq * SFK;
    mbar_expect_tx(full, QUAD_BYTES);
    tma_load_2d(buf, mp.sa, x, w.am0, full);
    tma_load_2d(buf + SA_BYTES, mp.sb, x, w.bn0, full);
  }
  ++pr.qseq;
  if (++pr.qq == quads_of(w.nk)) {
    pr.qq = 0;
    next_unit<G>(pr.qu, pr.qn, pr.qw, p, gd);
  }
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

// Where the current work unit sits in the CTA's ring.
struct Pos {
  int g0;    // sequence number of the unit's first k-tile
  int q0;    // sequence number of the unit's first scale quad
  int nk;    // k-tiles in the unit
  int half;  // skip half of the last / first k-tile (Unit::half)
};

// Scales of local k-tile `it`, waiting for its scale quad first if `it` starts one. The 64-byte scale
// rows are stored with the 64B swizzle (16B chunk c of row r at c ^ ((r >> 1) & 3)); a k-tile's SFK bytes
// lie inside one 16B chunk.
__device__ __forceinline__ void load_scales(Scales& sc, const LaneAddr& la, uint32_t bars, int it, const Pos& ps) {
  const int q = ps.q0 + it / QUAD, t = it % QUAD;
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

template <int S, bool G>
__device__ __forceinline__ void refill(Producer& pr, const Args& p, const Grid& gd, uint32_t base, uint32_t bars,
                                       int it, const Pos& ps, const Maps& mp, bool issue);

// What consume_stage needs to refill a stage: the release counters (shared memory) and this warp's cursors.
struct Shared {
  unsigned* cnt;  // [STAGES]
  Producer* pr;
  const Args* p;
  const Grid* gd;
  const Maps* mp;
  uint32_t base;
};

// Consume data stage S (local k-tile `it`): KS k-steps, each issuing its MMAs after the next k-step's
// fragments are requested; the last k-step prefetches the first k-step (and scales) of k-tile it + 1.
// The stage is released as soon as its last fragments are requested: mbarrier.arrive has release
// semantics, so those shared-memory reads are performed before the producer can observe the arrival.
// The warp that releases it last refills it at once (an atomic count of the releases elects it): the TMA
// load of k-tile it + STAGES then starts the moment the stage is free, rather than when one fixed thread
// gets past its own MMAs (warps waited on late data ~12% of the time with thread 0 as the only producer).
template <int S, bool G>
__device__ __forceinline__ void consume_stage(float (&acc)[MI][NI][4], Frag (&fr)[2], Scales& sc, const LaneAddr& la,
                                              uint32_t bars, int it, const Pos& ps, int lane, const Shared& sh) {
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
        if (lane == 0) {
          mbar_arrive(empty_bar(bars, S));
          // the count only grows: release r of the stage's every round is warps r * 8 .. r * 8 + 7
          const bool last = (atomicAdd(sh.cnt + S, 1u) % (WARPS_M * WARPS_N)) == WARPS_M * WARPS_N - 1;
          refill<S, G>(*sh.pr, *sh.p, *sh.gd, sh.base, bars, it, ps, *sh.mp, last);
        }
        __syncwarp();  // reconverge before the next .aligned instruction
      }
    } else if (it + 1 < ps.nk) {
      mbar_wait(full_bar(bars, SN), ((ps.g0 + it + 1) / STAGES) & 1);
      load_scales(sc, la, bars, it + 1, ps);
      load_frag<SN>(fr[0], la, 0);
    }
    // (B-reuse and serpentine orders measured the same.)
    if constexpr (G) {
      if (ps.half && ((ks >= KS / 2 && (ps.half & 1) && it + 1 == ps.nk) || (ks < KS / 2 && (ps.half & 2) && it == 0)))
        continue;
    }
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
      for (int j = 0; j < NI; ++j) mma_nvfp4(acc[i][j], fr[ks & 1].a[i], fr[ks & 1].b[j], sa[i], sb[j]);
  }
}

// After data stage S (local k-tile `it`) is released: refill it with the next k-tile of the ring (this
// unit's or the next unit's), and when `it` ends a scale quad, refill that quad's buffer with the next
// quad. Its scales (and those of every earlier k-tile) were read before its fragments, so after the stage
// release the finished quad buffer is free too. Only the last warp to release (`issue`) loads, after the
// empty-barrier wait (acquire: the TMA writes are ordered after every warp's reads); every warp advances
// its cursors.
template <int S, bool G>
__device__ __forceinline__ void refill(Producer& pr, const Args& p, const Grid& gd, uint32_t base, uint32_t bars,
                                       int it, const Pos& ps, const Maps& mp, bool issue) {
  if (pr.du >= gd.total) return;  // nothing left to load (quads run ahead of the data: none left either)
  if (issue) mbar_wait(empty_bar(bars, S), ((ps.g0 + it) / STAGES) & 1);
  issue_next_stage<G>(pr, p, gd, base, bars, mp, issue);
  if (it % QUAD == QUAD - 1 || it == ps.nk - 1) issue_next_quad<G>(pr, p, gd, base, bars, mp, issue);
}

// Local k-tiles it + J, it + J + 1, ... it + STAGES - 1 of the unit, whose first k-tile sits in stage OFF
// (k-tile it + J in stage (OFF + J) % STAGES, a compile-time constant).
template <int J, int OFF, bool G>
__device__ __forceinline__ void steps(float (&acc)[MI][NI][4], Frag (&fr)[2], Scales& sc, const LaneAddr& la,
                                      uint32_t bars, int it, const Pos& ps, int lane, const Shared& sh) {
  if constexpr (J < STAGES) {
    if (it + J < ps.nk) {
      consume_stage<(OFF + J) % STAGES, G>(acc, fr, sc, la, bars, it + J, ps, lane, sh);
      steps<J + 1, OFF, G>(acc, fr, sc, la, bars, it, ps, lane, sh);
    }
  }
}

// The main loop of one work unit whose first k-tile sits in stage OFF.
template <int OFF, bool G>
__device__ __forceinline__ void unit_loop(float (&acc)[MI][NI][4], Frag (&fr)[2], Scales& sc, const LaneAddr& la,
                                          uint32_t bars, const Pos& ps, int lane, const Shared& sh) {
  mbar_wait(full_bar(bars, OFF), (ps.g0 / STAGES) & 1);
  load_scales(sc, la, bars, 0, ps);
  load_frag<OFF>(fr[0], la, 0);
  for (int it = 0; it < ps.nk; it += STAGES) steps<0, OFF, G>(acc, fr, sc, la, bars, it, ps, lane, sh);
}

template <int OFF, bool G>
__device__ __forceinline__ void unit_loop_any(int off, float (&acc)[MI][NI][4], Frag (&fr)[2], Scales& sc,
                                              const LaneAddr& la, uint32_t bars, const Pos& ps, int lane,
                                              const Shared& sh) {
  if constexpr (OFF < STAGES) {
    if (off == OFF)
      unit_loop<OFF, G>(acc, fr, sc, la, bars, ps, lane, sh);
    else
      unit_loop_any<OFF + 1, G>(off, acc, fr, sc, la, bars, ps, lane, sh);
  }
}

// G: grouped GEMMs (Args::group_mode); a separate instantiation so the dense kernel carries none of it.
template <typename OutT, bool kAccum, bool G>
__global__ void __launch_bounds__(THREADS, 1)
    nvfp4_gemm_tma_kernel(Args p, const __grid_constant__ TmaParam tm_a, const __grid_constant__ TmaParam tm_b,
                          const __grid_constant__ TmaParam tm_sa, const __grid_constant__ TmaParam tm_sb) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t base = smem_u32(smem);
  const uint32_t bars = base + BAR_BASE;
  int* flag = reinterpret_cast<int*>(smem + BAR_BASE + (2 * STAGES + 2) * 8);
  const Maps mp{reinterpret_cast<const CUtensorMap*>(&tm_a), reinterpret_cast<const CUtensorMap*>(&tm_b),
                reinterpret_cast<const CUtensorMap*>(&tm_sa), reinterpret_cast<const CUtensorMap*>(&tm_sb)};
  Grid gd;
  gd.tiles_n = (p.N + BN - 1) / BN;
  if (G && p.group_mode == 2) {
    gd.tiles = gd.tiles_n * (p.a_rows / BM);
    gd.total = gd.tiles * p.groups;
  } else {
    gd.tiles = gd.tiles_n * ((p.M + BM - 1) / BM);
    gd.total = gd.tiles * p.splits;
  }

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  unsigned* cnt = reinterpret_cast<unsigned*>(smem + CNT_OFF);
  static_assert(STAGES * sizeof(unsigned) <= 64, "shared pipeline state");
  gd.ring = p.sched != nullptr ? reinterpret_cast<int*>(smem + SCHED_OFF) : nullptr;
  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(full_bar(bars, s), 1);
      mbar_init(empty_bar(bars, s), WARPS_M * WARPS_N);
    }
    for (int q = 0; q < 2; ++q) mbar_init(sfull_bar(bars, q), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    for (int s = 0; s < STAGES; ++s) cnt[s] = 0;
    if (gd.ring != nullptr) {
      gd.ring[0] = 0;
      for (int s = 0; s < RING; ++s) gd.ring[4 + 2 * s] = -1;
    }
  }
  if (gd.ring != nullptr) __syncthreads();  // the ring is initialized before the first claim
  // every warp's lane 0 keeps the load cursors; thread 0 also issues the prologue loads
  Producer pr;
  producer_init<G>(pr, p, gd);
  if (lane == 0) {  // the warps' cursor copies (lane 0 keeps them); thread 0 issues
    const bool issue = threadIdx.x == 0;
    for (int q = 0; q < 2; ++q) issue_next_quad<G>(pr, p, gd, base, bars, mp, issue);
    for (int s = 0; s < STAGES; ++s) issue_next_stage<G>(pr, p, gd, base, bars, mp, issue);
  }
  __syncthreads();
  const Shared sh{cnt, &pr, &p, &gd, &mp, base};

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

  Pos ps{0, 0, 0, 0};
  const bool dyn = gd.ring != nullptr;
  int n = 0;
  for (int u = dyn ? claimed_unit<G>(p, gd, 0) : static_cast<int>(blockIdx.x); u < gd.total;
       u = dyn ? claimed_unit<G>(p, gd, ++n) : u + static_cast<int>(gridDim.x)) {
    const Unit w = unit_of<G>(p, u, gd.tiles_n, gd.tiles);
    ps.nk = w.nk;
    ps.half = G ? w.half : 0;
    // the epilogue reads the old output (accumulate) or the residual: start fetching it into L2 now
    if (kAccum && p.splits == 1) prefetch_tile_l2(p.d, 4, p, w.m0, w.n0);
    if (!kAccum && p.res != nullptr) prefetch_tile_l2(p.res, p.res_bf16 ? 2 : 4, p, w.m0, w.n0);
    float acc[MI][NI][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
      for (int j = 0; j < NI; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.0f;
    if (w.nk > 0) {
      Frag fr[2];
      Scales sc;
      unit_loop_any<0, G>(ps.g0 % STAGES, acc, fr, sc, la, bars, ps, lane, sh);
    }
    ps.g0 += w.nk;
    ps.q0 += quads_of(w.nk);
    finish_tile<OutT, kAccum>(p, acc, w.m0, w.n0, wm, wn, lane, flag, w.z, w.tile);
  }
  if (!dyn) return;
  if constexpr (G) {  // the units without k-tiles, which no claim returns: zero tiles, statically spread
    for (int u = blockIdx.x; u < gd.total; u += gridDim.x) {
      const Unit w = unit_of<G>(p, u, gd.tiles_n, gd.tiles);
      if (w.nk != 0) continue;
      float acc[MI][NI][4];
#pragma unroll
      for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j)
#pragma unroll
          for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.0f;
      finish_tile<OutT, kAccum>(p, acc, w.m0, w.n0, wm, wn, lane, flag, w.z, w.tile);
    }
  }
  // This CTA claims nothing more (its last claim returned >= total). The last CTA to exit resets the
  // counters for the next launch on this stream (stream order makes the stores visible to it).
  if (threadIdx.x == 0 && atomicAdd(p.sched + 1, 1) == static_cast<int>(gridDim.x) - 1) {
    p.sched[0] = 0;
    p.sched[1] = 0;
  }
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
              CUtensorMapSwizzle swizzle, int64_t ld = 0) {
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(cols), static_cast<cuuint64_t>(rows)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(ld > 0 ? ld : cols)};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(box_cols), static_cast<cuuint32_t>(box_rows)};
  const cuuint32_t estr[2] = {1, 1};
  return encode_fn()(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                     CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// The claim counters of dynamic tile claiming (Args::sched) for launches on `stream`, or null (static
// assignment: switched off, or the stream is being captured into a CUDA graph). One pair per stream: launches
// on one stream run one after another, and each resets its pair on exit; concurrent streams get their own.
int* sched_counters(int dev, cudaStream_t stream) {
  if (!g_dynamic_tiles.load(std::memory_order_relaxed)) return nullptr;
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &st) != cudaSuccess || st != cudaStreamCaptureStatusNone) {
    (void)cudaGetLastError();
    return nullptr;
  }
  constexpr int CHUNK = 256;  // pairs per allocation
  static std::mutex mu;
  static std::map<std::pair<int, cudaStream_t>, int*> slots;
  static std::vector<std::pair<int, int*>> chunks;  // device, next free pair of its newest chunk
  static std::vector<int> used;
  std::lock_guard<std::mutex> lock(mu);
  const auto key = std::make_pair(dev, stream);
  const auto it = slots.find(key);
  if (it != slots.end()) return it->second;
  int idx = -1;
  for (size_t i = 0; i < chunks.size(); ++i)
    if (chunks[i].first == dev && used[i] < CHUNK) idx = static_cast<int>(i);
  if (idx < 0) {
    int* buf = nullptr;
    // one-time setup outside any capture (checked above): zeroed, and complete before any kernel uses it
    if (cudaMalloc(&buf, CHUNK * 2 * sizeof(int)) != cudaSuccess || cudaMemset(buf, 0, CHUNK * 2 * sizeof(int)) !=
        cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
      (void)cudaGetLastError();
      return nullptr;
    }
    chunks.emplace_back(dev, buf);
    used.push_back(0);
    idx = static_cast<int>(chunks.size()) - 1;
  }
  int* c = chunks[idx].second + 2 * used[idx]++;
  slots.emplace(key, c);
  return c;
}

template <typename OutT, bool kAccum, bool G = false>
void launch(const Args& p, const CUtensorMap (&maps)[4], dim3 grid, cudaStream_t stream) {
  static std::atomic<uint64_t> configured{0};
  const cudaError_t err = ensure_smem_attr(nvfp4_gemm_tma_kernel<OutT, kAccum, G>, SMEM_BYTES, configured);
  if (err != cudaSuccess) throw std::runtime_error(std::string("nvfp4 TMA GEMM: ") + cudaGetErrorString(err));
  TmaParam prm[4];
  static_assert(sizeof(prm) == sizeof(maps), "tensor map parameter layout");
  std::memcpy(prm, maps, sizeof(prm));
  // persistent: one CTA per SM (at most one fits), each looping over work units
  const int units = static_cast<int>(grid.x * grid.y * grid.z);
  int dev = 0, sms = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  sms = std::max(1, sms - g_sm_margin.load(std::memory_order_relaxed));
  const dim3 pgrid(std::max(1, std::min(units, sms)));
  Args q = p;
  q.sched = static_cast<int>(pgrid.x) < units ? sched_counters(dev, stream) : nullptr;
  nvfp4_gemm_tma_kernel<OutT, kAccum, G><<<pgrid, THREADS, SMEM_BYTES, stream>>>(q, prm[0], prm[1], prm[2], prm[3]);
}

}  // namespace tma

int tma_k_tile() { return tma::BK; }
void set_sm_margin(int64_t n) { g_sm_margin.store(static_cast<int>(std::max<int64_t>(0, n))); }
int64_t sm_margin() { return g_sm_margin.load(); }
void set_dynamic_tiles(bool on) { g_dynamic_tiles.store(on); }
bool dynamic_tiles() { return g_dynamic_tiles.load(); }

// Grouped GEMMs (Args::group_mode 1 / 2): A has a_map_rows rows, B b_map_rows; scale rows have stride p.sf_ld
// bytes (a multiple of 16, >= K / 16). Launches `units` persistent CTAs at most. Returns false (nothing
// launched) if a descriptor cannot be made.
bool launch_tma_grouped(const Args& p, int64_t a_map_rows, int64_t b_map_rows, bool out_fp32, bool accumulate,
                        int units, cudaStream_t stream) {
  using namespace tma;
  if (encode_fn() == nullptr) return false;
  CUtensorMap maps[4];
  const CUtensorMapSwizzle swz = BKB == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;
  const bool ok = make_map(&maps[0], p.a, a_map_rows, p.K / 2, BKB, BM, swz) &&
                  make_map(&maps[1], p.b, b_map_rows, p.K / 2, BKB, BN, swz) &&
                  make_map(&maps[2], p.sfa, a_map_rows, p.K / 16, QB, BM, CU_TENSOR_MAP_SWIZZLE_64B, p.sf_ld) &&
                  make_map(&maps[3], p.sfb, b_map_rows, p.K / 16, QB, BN, CU_TENSOR_MAP_SWIZZLE_64B, p.sf_ld);
  if (!ok) return false;
  const dim3 grid(units);
  if (out_fp32) {
    if (accumulate)
      launch<float, true, true>(p, maps, grid, stream);
    else
      launch<float, false, true>(p, maps, grid, stream);
  } else {
    launch<__nv_bfloat16, false, true>(p, maps, grid, stream);
  }
  return true;
}

bool tma_supported(const Args& p) {
  auto aligned = [](const void* q) { return (reinterpret_cast<uintptr_t>(q) & 15) == 0; };
  // K % 256: the scale rows (K/16 bytes) must have a 16-byte global stride for TMA. K > 0: every work unit
  // has at least one k-tile.
  return p.K > 0 && p.K % 256 == 0 && p.K % tma::BK == 0 && aligned(p.a) && aligned(p.b) && aligned(p.sfa) &&
         aligned(p.sfb) && tma::encode_fn() != nullptr;
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
