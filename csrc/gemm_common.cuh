// Pieces shared by the two NVFP4 GEMM main loops (gemm.cu: cp.async, gemm_tma.cu: TMA + warp
// specialization): MMA / ldmatrix wrappers, kernel arguments, and the epilogue with deterministic
// split-K reduction and optional in-place fp32 accumulation.
#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <type_traits>

namespace nvfp4 {
namespace gemm {

constexpr int BM = 128, BN = 128;                    // CTA tile (both main loops)
constexpr int WARPS_M = 2, WARPS_N = 4;              // 8 MMA warps
constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;  // 64 x 32 per warp
constexpr int MI = WM / 16, NI = WN / 8;             // 4 x 4 MMA tiles per warp
constexpr int MMA_THREADS = 32 * WARPS_M * WARPS_N;  // 256
constexpr int kMaxSplits = 16;

struct Args {
  const uint8_t* a;    // [M, K/2]
  const uint8_t* b;    // [N, K/2]
  const uint8_t* sfa;  // [M, K/16]
  const uint8_t* sfb;  // [N, K/16]
  const float* gdec_a;
  const float* gdec_b;
  void* d;             // [M, N] (ldd = N)
  float* ws;           // split-K partials [splits, M, N] (splits > 1 only)
  int* counters;       // split-K arrival counters, one per output tile, zero on entry and on exit
  int M, N, K;
  int splits;          // gridDim.z; split z covers k-tiles [KT*z/splits, KT*(z+1)/splits)
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

// D += A(16x64 e2m1) * B(64x8 e2m1) with one UE4M3 scale per 16 K-elements. Fragment layouts follow
// CUTLASS cute::MMA_Traits<SM120_16x8x64_TN_VS>; verified by tests/test_gemm_cuda.py.
__device__ __forceinline__ void mma_nvfp4(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], uint32_t sfa,
                                          uint32_t sfb) {
  const uint16_t zero = 0;  // byte-id / thread-id selectors (layout chosen so 0 is correct)
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "h"(zero), "h"(zero), "r"(sfb),
        "h"(zero), "h"(zero));
}

// Barrier over the 256 MMA threads only (the TMA kernel's producer warp does not take part).
__device__ __forceinline__ void mma_threads_sync() { asm volatile("bar.sync 1, 256;\n" ::: "memory"); }

// Visits the fragment elements this thread owns: f(i, j, h, row, col) for acc[i][j][2h], acc[i][j][2h+1]
// at D[row, col], D[row, col + 1].
template <typename F>
__device__ __forceinline__ void for_each_pair(int m0, int n0, int wm, int wn, int lane, F&& f) {
  const int g = lane >> 2, tig = lane & 3;
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) f(i, j, h, m0 + wm * WM + i * 16 + g + 8 * h, n0 + wn * WN + j * 8 + 2 * tig);
}

__device__ __forceinline__ float2 load_pair(const float* base, int row, int col, int M, int N) {
  float2 v = make_float2(0.0f, 0.0f);
  if (row < M) {
    const float* src = base + static_cast<int64_t>(row) * N + col;
    if (col + 1 < N && (N % 2 == 0)) {
      v = __ldcg(reinterpret_cast<const float2*>(src));  // L2 (other CTAs wrote it)
    } else {
      if (col < N) v.x = __ldcg(src);
      if (col + 1 < N) v.y = __ldcg(src + 1);
    }
  }
  return v;
}

__device__ __forceinline__ void store_pair(float* base, int row, int col, int M, int N, float v0, float v1) {
  if (row >= M) return;
  float* dst = base + static_cast<int64_t>(row) * N + col;
  if (col + 1 < N && (N % 2 == 0)) {
    __stcg(reinterpret_cast<float2*>(dst), make_float2(v0, v1));
  } else {
    if (col < N) __stcg(dst, v0);
    if (col + 1 < N) __stcg(dst + 1, v1);
  }
}

// Split-K: every split stores its raw fp32 partial; the last CTA of a tile to arrive sums all partials
// in split order 0..S-1 (its own from registers), so the result does not depend on arrival order.
// Returns false in CTAs that are not last (they have nothing more to do). Called by the 256 MMA threads.
__device__ __forceinline__ bool splitk_reduce(const Args& p, float (&acc)[MI][NI][4], int m0, int n0, int wm,
                                              int wn, int lane, int split, int tile, int* flag) {
  const int64_t plane = static_cast<int64_t>(p.M) * p.N;
  for_each_pair(m0, n0, wm, wn, lane, [&](int i, int j, int h, int row, int col) {
    store_pair(p.ws + split * plane, row, col, p.M, p.N, acc[i][j][2 * h], acc[i][j][2 * h + 1]);
  });
  __threadfence();
  mma_threads_sync();
  const bool leader = (wm == 0 && wn == 0 && lane == 0);
  if (leader) *flag = (atomicAdd(p.counters + tile, 1) == p.splits - 1);
  mma_threads_sync();
  if (!*flag) return false;
  __threadfence();
  // All partials (this CTA's included, re-read from L2) are summed in split order. Splits outermost so
  // that all of one split's loads are in flight together.
  for (int z = 0; z < p.splits; ++z) {
    float part[MI][NI][4];
    for_each_pair(m0, n0, wm, wn, lane, [&](int i, int j, int h, int row, int col) {
      const float2 v = load_pair(p.ws + z * plane, row, col, p.M, p.N);
      part[i][j][2 * h] = v.x;
      part[i][j][2 * h + 1] = v.y;
    });
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
      for (int j = 0; j < NI; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][j][r] = (z == 0) ? part[i][j][r] : __fadd_rn(acc[i][j][r], part[i][j][r]);
  }
  if (leader) p.counters[tile] = 0;  // ready for the next launch
  return true;
}

// D = alpha * acc, or with kAccum (fp32 only) D = D_old + alpha * acc with separate rn ops (no FMA
// contraction), bit-identical to autograd's grad + (alpha * acc).
template <typename OutT, bool kAccum>
__device__ __forceinline__ void store_tile(const Args& p, float (&acc)[MI][NI][4], int m0, int n0, int wm, int wn,
                                           int lane) {
  const float alpha = (*p.gdec_a) * (*p.gdec_b);
  if constexpr (kAccum) {
    // Loads for one m16 row group are issued together (a load cannot be moved past a store to the
    // same array, so a load-add-store loop would serialize their latency).
    float* d = reinterpret_cast<float*>(p.d);
    const int g = lane >> 2, tig = lane & 3;
#pragma unroll
    for (int i = 0; i < MI; ++i) {
      float2 old[NI][2];
#pragma unroll
      for (int j = 0; j < NI; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          old[j][h] = load_pair(d, m0 + wm * WM + i * 16 + g + 8 * h, n0 + wn * WN + j * 8 + 2 * tig, p.M, p.N);
#pragma unroll
      for (int j = 0; j < NI; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = m0 + wm * WM + i * 16 + g + 8 * h, col = n0 + wn * WN + j * 8 + 2 * tig;
          store_pair(d, row, col, p.M, p.N, __fadd_rn(old[j][h].x, __fmul_rn(acc[i][j][2 * h], alpha)),
                     __fadd_rn(old[j][h].y, __fmul_rn(acc[i][j][2 * h + 1], alpha)));
        }
    }
  } else {
    for_each_pair(m0, n0, wm, wn, lane, [&](int i, int j, int h, int row, int col) {
      if (row >= p.M) return;
      const float v0 = acc[i][j][2 * h] * alpha, v1 = acc[i][j][2 * h + 1] * alpha;
      OutT* out = reinterpret_cast<OutT*>(p.d) + static_cast<int64_t>(row) * p.N + col;
      if constexpr (std::is_same_v<OutT, float>) {
        if (col + 1 < p.N && (p.N % 2 == 0)) {
          *reinterpret_cast<float2*>(out) = make_float2(v0, v1);
        } else {
          if (col < p.N) out[0] = v0;
          if (col + 1 < p.N) out[1] = v1;
        }
      } else {
        if (col + 1 < p.N && (p.N % 2 == 0)) {
          *reinterpret_cast<__nv_bfloat162*>(out) = __floats2bfloat162_rn(v0, v1);
        } else {
          if (col < p.N) out[0] = __float2bfloat16_rn(v0);
          if (col + 1 < p.N) out[1] = __float2bfloat16_rn(v1);
        }
      }
    });
  }
}

// Common tail of both kernels: split-K reduction (if any), then the output store.
template <typename OutT, bool kAccum>
__device__ __forceinline__ void finish_tile(const Args& p, float (&acc)[MI][NI][4], int m0, int n0, int wm, int wn,
                                            int lane, int* flag) {
  if (p.splits > 1 &&
      !splitk_reduce(p, acc, m0, n0, wm, wn, lane, blockIdx.z, blockIdx.y * gridDim.x + blockIdx.x, flag))
    return;
  store_tile<OutT, kAccum>(p, acc, m0, n0, wm, wn, lane);
}

}  // namespace gemm
}  // namespace nvfp4
