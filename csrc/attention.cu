// 8-bit flash attention (forward + deterministic backward) for consumer Blackwell (sm_120a).
//
// (v0.10: nvfp4.attention's default backward is PyTorch's flash backward of this forward instead, on the
// dequantized operands attn_quant_kernel also writes (QuantArgs::deq), with the forward's P V in two e4m3
// terms (attn_fwd_kernel's P2) so that Delta = rowsum(dO O) stays consistent; docs/DESIGN.md §7.1. The
// backward kernels below serve backward="8bit".)
//
// Every attention GEMM runs on an 8-bit tensor-core path with FP32 (or exact INT32) accumulation, at 3.6-3.9x
// the rate of BF16 mma with FP32 accumulation on the RTX 5090 (docs/DESIGN.md §7):
//   - the score-type products Q K^T and dO V^T (whose errors reach the softmax / dS directly) in INT8 with
//     one FP32 scale per token (per row of either operand, over the whole head dim):
//       mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
//     4x smaller errors than E4M3 at the same speed (E4M3's 3-bit mantissa vs INT8's 7 bits on data
//     scaled to its row maximum; K is centred by its mean over the sequence first);
//   - the products with probabilities / score gradients, P V, P^T dO, dS^T Q, in MXFP8 (E4M3 with one
//     power-of-two UE8M0 scale per 32 elements of the reduction dim; P and dS span many binades):
//       mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
//   - dQ = dS K, dK = dS^T Q and dV = P^T dO in FP16 (m16n8k16 with FP16 accumulation, twice the BF16 rate):
//     every row of dS sums to zero, so dQ is a small difference of large terms and the 8-bit errors of dS and K
//     survived the cancellation (12-24% relative error in dQ on trained-model activations at 4096 tokens); dK and
//     dV with MXFP8 operands (Q, dO in e4m3) were 2-10x further from the exact gradient than the exact backward
//     (v0.19, bench/attention_bwd_probe.py). The A operand (dS or P, times the per-token scale of the other
//     operand's 8-bit row layout, which runs along the reduction) is divided by its maximum over the row's keys
//     (dQ) or queries (dK, dV) of the tile, so |A| <= 1; B is the INT8 code (K, Q: exact in FP16) or the
//     unrounded scaled value (dO, |x| <= 127) of the row layout, so the FP16 accumulator of one tile stays below
//     64 * 127 and is added to the FP32 result times the row scale (docs/DESIGN.md §7.1).
//
// Operands are quantized once by attn_quant_kernel, in two layouts per tensor:
//   row   X8 [BH, Np, D] int8, scales [BH, Np] fp32          per token (QK^T, dO V^T operands)
//   col   Xt [BH, D, Np] e4m3, scales [BH, Np/64, 2D] ue8m0  transposed, blocks of 32 along the sequence,
//                                                          the tokens of each block permuted by sigma (below)
// The 2D scale bytes of a 64-token tile (channel d = 8 j + r, block kk) sit at (r * D/8 + j) * 2 + kk: the
// D/4 bytes that lane l of an MMA needs (r = l >> 2) are contiguous (one or two 16-byte loads per tile), and
// the MMA's byte selector picks the scale of n-tile j, k-step kk out of word (2 j + kk) / 4.
// Np = N rounded up to 128 (zero padding). The probabilities P and the score gradients dS are quantized in
// registers, straight from the accumulators of the GEMM that produced them: the C fragment of an m16n8 MMA
// gives thread t the columns {2t, 2t+1} of each 8-column tile, while the A fragment of an m16n8k32 FP8 MMA
// wants the k-indices {4t..4t+3, 16+4t..16+4t+3}. Any order of the keys works as long as the second
// operand follows it, so the col layout stores at k-position k the token sigma(k), with
//   sigma(k) = 16 (k >> 4) + 8 ((k & 3) >> 1) + 2 ((k & 15) >> 2) + (k & 1),
// which is exactly the token whose score thread t holds there: no shuffles, no shared-memory round trip.
//
// Forward (attn_fwd_kernel): CTAs of 128 queries (4 warps x 32 rows), two per SM, for each (batch, head),
// streaming 64-key tiles (K row layout, V col layout) through a 2-stage cp.async pipeline; online softmax in
// the log2 domain with a lazily rescaled maximum; P is quantized as e4m3(2^7 p) (p <= 2^1.5 relative to the
// running maximum: a fixed scale loses nothing that matters next to the row's maximum, as in
// FlashAttention-3's FP8 path).
// Backward: two kernels and no atomics (bitwise deterministic), FlashAttention-2's math with dQ split off
// (two GEMMs more than with atomics, the price of determinism):
//   attn_dq_kernel    per 128 queries:  S = Q K^T, dP = dO V^T, dS = P (dP - Delta), dQ += dS K (BF16),
//                                       B += P K (MXFP8), r += rowsum(dS); dQ - r B at the end (below)
//   attn_dkdv_kernel  per 128 keys:     S^T = K Q^T, dP^T = V dO^T, dV += P^T dO, dK += dS^T Q
//                                       (looping over the query heads of a GQA group as well)
// dS is quantized with a per-(row, 32-block) UE8M0 scale computed from its amax across the quad.
// Delta = rowsum(dO O) comes from the 8-bit forward's O (~2% off), so it is not quite the row sum of P dP as
// the backward computes them, and every row of dS = P (dP - Delta) then sums to r_i != 0 instead of zero. In
// dQ = dS K, a sum over all keys, that offset turned into 4-14% error at 4096 tokens on trained-model
// activations. The dQ kernel therefore also accumulates r_i and B_i = sum_j P_ij K_j and returns
// dQ_i - r_i B_i, which is exactly what the consistent Delta' = rowsum(P dP) = Delta + r_i gives. r is small,
// so B needs little precision: one MXFP8 product (P e4m3, K col layout) instead of a second pass over S and dP
// (dQ error ~1%; docs/DESIGN.md §7). dK and dV use Delta as is (2-3%).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <type_traits>

#include "gemm_common.cuh"

namespace nvfp4 {
namespace attn {

using u8 = uint8_t;
constexpr int PAD = 128;      // sequence padding of the quantized tensors
constexpr int THREADS = 256;  // 8 warps
constexpr float kInf = __builtin_huge_valf();
// Dynamic shared memory per CTA with two CTAs per SM: sm_120 has 100 KB per SM and reserves 1 KB per CTA.
constexpr int kSmem2 = (102400 - 2 * 1024) / 2;
constexpr float kPOff = 7.0f;   // forward: P is quantized as e4m3(2^kPOff p)
constexpr float kSlack = 1.5f;  // forward: the running maximum may lag the true one by this much (log2)

// --- small device helpers -------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t su32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ uint32_t lds_u32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(addr));
  return v;
}
__device__ __forceinline__ uint32_t lds_u16(uint32_t addr) {
  uint16_t v;
  asm volatile("ld.shared.u16 %0, [%1];\n" : "=h"(v) : "r"(addr));
  return v;
}
__device__ __forceinline__ uint32_t lds_u8(uint32_t addr) {
  uint16_t v;
  asm volatile("ld.shared.u8 %0, [%1];\n" : "=h"(v) : "r"(addr));
  return v;
}
__device__ __forceinline__ float lds_f32(uint32_t addr) {
  float v;
  asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(v) : "r"(addr));
  return v;
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

// D += A(16x32 e4m3) B(32x8 e4m3), A scaled per row and B per column by byte `sel` of sfa / sfb (UE8M0).
// Scale registers: lane l supplies row (l >> 2) + 8 (l & 1) of A and column l >> 2 of B (thread-id
// selector 0), as in the NVFP4 GEMM (verified for the FP8 kind by tests/test_attention_cuda.py).
__device__ __forceinline__ void mma8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa,
                                     uint32_t sfb, uint16_t sela, uint16_t selb) {
  const uint16_t tid = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(sela), "h"(tid), "r"(sfb),
        "h"(selb), "h"(tid));
}

// Four floats -> four e4m3 bytes (a in the lowest byte), round-to-nearest-even, saturating.
__device__ __forceinline__ uint32_t pack4(float a, float b, float c, float d) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(b), "f"(a));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(d), "f"(c));
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

// Two e4m3 values (low byte first) -> f16x2, exactly.
__device__ __forceinline__ __half2 e4m3x2_to_h2(uint16_t x) {
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(x));
  return *reinterpret_cast<const __half2*>(&r);
}

// The rounding residuals of a pack4 word hi of (a, b, c, d), times 16, as e4m3 (same order): x ~ hi + lo / 16
// to ~2^-8 relative. Computed in f16x2 (x <= 2^8.5 here; the subtraction is exact and the f16 rounding of x,
// 2^-11, is below what the e4m3 residual keeps).
__device__ __forceinline__ uint32_t resid4(uint32_t hi, float a, float b, float c, float d) {
  const __half2 k16 = __float2half2_rn(16.0f);
  const __half2 r0 = __hmul2(__hsub2(__floats2half2_rn(a, b), e4m3x2_to_h2(static_cast<uint16_t>(hi))), k16);
  const __half2 r1 = __hmul2(__hsub2(__floats2half2_rn(c, d), e4m3x2_to_h2(static_cast<uint16_t>(hi >> 16))), k16);
  uint16_t lo0, lo1;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(lo0) : "r"(*reinterpret_cast<const uint32_t*>(&r0)));
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(lo1) : "r"(*reinterpret_cast<const uint32_t*>(&r1)));
  return static_cast<uint32_t>(lo0) | (static_cast<uint32_t>(lo1) << 16);
}

// UE8M0 scale of a block with absolute maximum amax: the biased exponent of the smallest power of two 2^E with
// amax / 2^E <= 448 (E4M3's largest finite value), clamped to [1, 253] so that 2^-E below is a normal float.
__device__ __forceinline__ uint32_t e8_of(float amax) {
  const uint32_t bits = __float_as_uint(amax * (1.0f / 448.0f));
  uint32_t e = (bits >> 23) + ((bits & 0x7fffffu) != 0u);
  return min(max(e, 1u), 253u);
}
__device__ __forceinline__ float inv_scale(uint32_t e) { return __uint_as_float((254u - e) << 23); }  // 2^-(e-127)

// Token stored at k-position k of a 32-token block in the col layout (see the top of the file).
__host__ __device__ constexpr int sigma(int k) {
  return 16 * (k >> 4) + 8 * ((k & 3) >> 1) + 2 * ((k & 15) >> 2) + (k & 1);
}

// Byte offset of 16-byte chunk c of row r in a tile with RB-byte rows, XOR-swizzled so that the 8 rows of an
// ldmatrix read hit distinct banks.
template <int RB>
__device__ __forceinline__ uint32_t swz(int r, int c) {
  if constexpr (RB == 128) {
    return r * 128 + ((c ^ (r & 7)) << 4);
  } else if constexpr (RB == 256) {
    return r * 256 + ((c ^ (r & 7)) << 4);
  } else if constexpr (RB == 64) {
    return r * 64 + ((c ^ ((r >> 1) & 3)) << 4);
  } else {
    static_assert(RB == 32, "row bytes");
    return r * 32 + ((c ^ ((r >> 2) & 1)) << 4);
  }
}

// cp.async ROWS rows of RB bytes (global row stride `gstride` bytes) into a swizzled tile.
template <int RB, int ROWS, int NTH = THREADS>
__device__ __forceinline__ void load_tile(uint32_t dst, const u8* src, int64_t gstride) {
  constexpr int CH = RB / 16, TOTAL = ROWS * CH;
  static_assert(TOTAL % NTH == 0 || TOTAL < NTH, "tile size");
#pragma unroll
  for (int i0 = 0; i0 < TOTAL; i0 += NTH) {
    const int i = i0 + threadIdx.x;
    if (TOTAL >= NTH || i < TOTAL) {
      const int r = i / CH, c = i % CH;
      cp16(dst + swz<RB>(r, c), src + r * gstride + c * 16);
    }
  }
}
// cp.async BYTES contiguous bytes.
template <int BYTES, int NTH = THREADS>
__device__ __forceinline__ void load_raw(uint32_t dst, const u8* src) {
  static_assert(BYTES % 16 == 0, "16-byte chunks");
#pragma unroll
  for (int i0 = 0; i0 < BYTES / 16; i0 += NTH) {
    const int i = i0 + threadIdx.x;
    if (i < BYTES / 16) cp16(dst + i * 16, src + i * 16);
  }
}

__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// D (FP16) += A(16x16 f16) B(16x8 f16), FP16 accumulation; D rows g / g + 8, columns 2t, 2t + 1 as half2.
__device__ __forceinline__ void mma_f16(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  const __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
// acc[j][0..1] += float2(d0) * m0, acc[j][2..3] += float2(d1) * m1 (an FP16 tile result into FP32).
__device__ __forceinline__ void add_h2(float (&acc)[4], const uint32_t (&d)[2], float m0, float m1) {
  const float2 x = __half22float2(*reinterpret_cast<const __half2*>(&d[0]));
  const float2 y = __half22float2(*reinterpret_cast<const __half2*>(&d[1]));
  acc[0] = fmaf(x.x, m0, acc[0]), acc[1] = fmaf(x.y, m0, acc[1]);
  acc[2] = fmaf(y.x, m1, acc[2]), acc[3] = fmaf(y.y, m1, acc[3]);
}
// The f16 A fragments (16 rows x 16 k, k-step kk) of the C fragments x[NT][4] of the same rows (columns 8j + 2t,
// 8j + 2t + 1: n-tiles 2 kk, 2 kk + 1 are exactly k-step kk), times m0 (row g) / m1 (row g + 8).
template <int NT>
__device__ __forceinline__ void acc_to_h(uint32_t (&a)[4], const float (&x)[NT][4], int kk, float m0, float m1) {
  a[0] = pack_h2(x[2 * kk][0] * m0, x[2 * kk][1] * m0);
  a[1] = pack_h2(x[2 * kk][2] * m1, x[2 * kk][3] * m1);
  a[2] = pack_h2(x[2 * kk + 1][0] * m0, x[2 * kk + 1][1] * m0);
  a[3] = pack_h2(x[2 * kk + 1][2] * m1, x[2 * kk + 1][3] * m1);
}
// Row maxima of |x| (rows g, g + 8) over the quad; the scale s and 1 / s. Rows whose maximum is below 2^-125 (1 / s
// would overflow, e.g. P times a dO scale of 1e-11 on a peaked head) are dropped: s = 0, as for an all-zero row.
template <int NT>
__device__ __forceinline__ void row_scales(const float (&x)[NT][4], float (&sc)[2], float (&inv)[2]) {
  float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    a0 = fmaxf(a0, fmaxf(fabsf(x[j][0]), fabsf(x[j][1])));
    a1 = fmaxf(a1, fmaxf(fabsf(x[j][2]), fabsf(x[j][3])));
  }
  constexpr float kMin = 2.3509887e-38f;  // 2^-125
  sc[0] = quad_max(a0), sc[1] = quad_max(a1);
  sc[0] = sc[0] >= kMin ? sc[0] : 0.0f;
  sc[1] = sc[1] >= kMin ? sc[1] : 0.0f;
  inv[0] = sc[0] > 0.0f ? 1.0f / sc[0] : 0.0f;
  inv[1] = sc[1] > 0.0f ? 1.0f / sc[1] : 0.0f;
}
// acc[D/8][4] += A B for the f16 A fragments a[KK] (k-steps of 16) and a row-major f16 tile B (16 KK rows of 2D
// bytes, ldmatrix.trans), through FP16 accumulators of the tile, added times the row scales sc.
template <int D, int KK>
__device__ __forceinline__ void f16_product(float (&acc)[D / 8][4], const uint32_t (&a)[KK][4], uint32_t btile,
                                            const float (&sc)[2], int lane);
__device__ __forceinline__ void ldsm4t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
template <int D, int KK>
__device__ __forceinline__ void f16_product(float (&acc)[D / 8][4], const uint32_t (&a)[KK][4], uint32_t btile,
                                            const float (&sc)[2], int lane) {
#pragma unroll
  for (int n = 0; n < D / 8; n += 2) {
    uint32_t d0[2] = {0u, 0u}, d1[2] = {0u, 0u};
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
      uint32_t b0, b1, b2, b3;
      ldsm4t(btile + swz<2 * D>(16 * kk + (lane & 7) + ((lane >> 3) & 1) * 8, n + (lane >> 4)), b0, b1, b2, b3);
      mma_f16(d0, a[kk], b0, b1);
      mma_f16(d1, a[kk], b2, b3);
    }
    add_h2(acc[n], d0, sc[0], sc[1]);
    add_h2(acc[n + 1], d1, sc[0], sc[1]);
  }
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ float2 lds_f32x2(uint32_t addr) {
  float2 v;
  asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];\n" : "=f"(v.x), "=f"(v.y) : "r"(addr));
  return v;
}

// A fragments (16 rows x D bytes, row layout, swizzled) of the k-steps of a QK^T-type product.
template <int D>
__device__ __forceinline__ void load_a_frags(uint32_t (&a)[D / 32][4], uint32_t tile, int row0, int lane) {
#pragma unroll
  for (int ks = 0; ks < D / 32; ++ks)
    ldsm4(tile + swz<D>(row0 + (lane & 7) + ((lane >> 3) & 1) * 8, 2 * ks + (lane >> 4)), a[ks][0], a[ks][1],
          a[ks][2], a[ks][3]);
}

// out[NT][4] = (A B^T) * rb[col]: A = 16 x D int8 (fragments a), B = NT*8 rows of D int8 bytes (row layout tile)
// with fp32 per-row scales at bs. The row scales of A are left to the caller, who folds them into the FMA that
// consumes the value (e.g. the exponent of the softmax), and the int32 -> fp32 conversion uses the magic-number
// trick (__int_as_float(i + 0x4B400000) == 1.5 * 2^23 + i exactly for |i| < 2^22; |i| <= 128 * 127^2 here): an
// integer add on the full-rate pipe instead of I2F, which runs at a quarter of the FMA rate.
template <int D, int NT, typename RB>
__device__ __forceinline__ void qk_product_rb(float (&out)[NT][4], const uint32_t (&a)[D / 32][4], uint32_t btile,
                                              RB&& rbf, int lane) {
  int acc[NT][4];
#pragma unroll
  for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0;
#pragma unroll
  for (int ks = 0; ks < D / 32; ++ks)
#pragma unroll
    for (int j = 0; j < NT; j += 2) {
      uint32_t b0, b1, b2, b3;
      ldsm4(btile + swz<D>(8 * j + (lane & 7) + (lane >> 4) * 8, 2 * ks + ((lane >> 3) & 1)), b0, b1, b2, b3);
      mma_s8(acc[j], a[ks], b0, b1);
      mma_s8(acc[j + 1], a[ks], b2, b3);
    }
  constexpr float kMagic = 12582912.0f;
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    const float2 rb = rbf(8 * j + 2 * (lane & 3));  // the scales of B rows (columns of the product) col, col + 1
    const float c0 = -kMagic * rb.x, c1 = -kMagic * rb.y;
    out[j][0] = fmaf(__int_as_float(acc[j][0] + 0x4B400000), rb.x, c0);
    out[j][1] = fmaf(__int_as_float(acc[j][1] + 0x4B400000), rb.y, c1);
    out[j][2] = fmaf(__int_as_float(acc[j][2] + 0x4B400000), rb.x, c0);
    out[j][3] = fmaf(__int_as_float(acc[j][3] + 0x4B400000), rb.y, c1);
  }
}

// B row scales from shared memory (fp32 at bs).
template <int D, int NT>
__device__ __forceinline__ void qk_product(float (&out)[NT][4], const uint32_t (&a)[D / 32][4], uint32_t btile,
                                           uint32_t bs, int lane) {
  qk_product_rb<D, NT>(out, a, btile, [&](int col) { return lds_f32x2(bs + 4 * col); }, lane);
}

// Maximum / sum of the row-g (e = 0, 1) or row-(g + 8) (e = 2, 3) values of x[NT][4], as balanced trees.
template <int NT>
__device__ __forceinline__ float2 row_max(const float (&x)[NT][4]) {
  float a[NT], b[NT];
#pragma unroll
  for (int j = 0; j < NT; ++j) a[j] = fmaxf(x[j][0], x[j][1]), b[j] = fmaxf(x[j][2], x[j][3]);
#pragma unroll
  for (int w = 1; w < NT; w *= 2)
#pragma unroll
    for (int j = 0; j + w < NT; j += 2 * w) a[j] = fmaxf(a[j], a[j + w]), b[j] = fmaxf(b[j], b[j + w]);
  return make_float2(a[0], b[0]);
}
template <int NT>
__device__ __forceinline__ float2 row_sum(const float (&x)[NT][4]) {
  float a[NT], b[NT];
#pragma unroll
  for (int j = 0; j < NT; ++j) a[j] = x[j][0] + x[j][1], b[j] = x[j][2] + x[j][3];
#pragma unroll
  for (int w = 1; w < NT; w *= 2)
#pragma unroll
    for (int j = 0; j + w < NT; j += 2 * w) a[j] += a[j + w], b[j] += b[j + w];
  return make_float2(a[0], b[0]);
}

// The e4m3 A fragment of k-step kk (32 columns) from fp32 accumulators x[NT][4] of the same 16 rows (columns
// 8j + 2t, 8j + 2t + 1), each multiplied by m0 (row g) / m1 (row g + 8). The k-positions come out in sigma
// order (see the top of the file).
template <int NT>
__device__ __forceinline__ void acc_to_a(uint32_t (&a)[4], const float (&x)[NT][4], int kk, float m0, float m1) {
  const int j = 4 * kk;
  a[0] = pack4(x[j][0] * m0, x[j][1] * m0, x[j + 1][0] * m0, x[j + 1][1] * m0);
  a[1] = pack4(x[j][2] * m1, x[j][3] * m1, x[j + 1][2] * m1, x[j + 1][3] * m1);
  a[2] = pack4(x[j + 2][0] * m0, x[j + 2][1] * m0, x[j + 3][0] * m0, x[j + 3][1] * m0);
  a[3] = pack4(x[j + 2][2] * m1, x[j + 2][3] * m1, x[j + 3][2] * m1, x[j + 3][3] * m1);
}

// The e4m3 residuals (resid4) of acc_to_a's fragment a of x (multipliers 1).
template <int NT>
__device__ __forceinline__ void acc_resid(uint32_t (&lo)[4], const uint32_t (&a)[4], const float (&x)[NT][4], int kk) {
  const int j = 4 * kk;
  lo[0] = resid4(a[0], x[j][0], x[j][1], x[j + 1][0], x[j + 1][1]);
  lo[1] = resid4(a[1], x[j][2], x[j][3], x[j + 1][2], x[j + 1][3]);
  lo[2] = resid4(a[2], x[j + 2][0], x[j + 2][1], x[j + 3][0], x[j + 3][1]);
  lo[3] = resid4(a[3], x[j + 2][2], x[j + 2][3], x[j + 3][2], x[j + 3][3]);
}

// Quantize k-step kk of x (as acc_to_a) with a per-(row, 32-column) scale from the quad's amax; returns the
// matching A scale register (byte 0).
template <int NT>
__device__ __forceinline__ uint32_t acc_to_a_scaled(uint32_t (&a)[4], const float (&x)[NT][4], int kk, int lane) {
  float am0 = 0.0f, am1 = 0.0f;
#pragma unroll
  for (int j = 4 * kk; j < 4 * kk + 4; ++j) {
    am0 = fmaxf(am0, fmaxf(fabsf(x[j][0]), fabsf(x[j][1])));
    am1 = fmaxf(am1, fmaxf(fabsf(x[j][2]), fabsf(x[j][3])));
  }
  am0 = quad_max(am0);
  am1 = quad_max(am1);
  const uint32_t e0 = e8_of(am0), e1 = e8_of(am1);
  acc_to_a<NT>(a, x, kk, inv_scale(e0), inv_scale(e1));
  return (lane & 1) ? e1 : e0;
}

// This lane's col-layout scale words of a 64-token tile (see the top of the file): D/4 bytes at bs.
template <int D>
struct ColScales {
  uint32_t w[D / 16];
};
template <int D>
__device__ __forceinline__ void load_col_scales(ColScales<D>& cs, uint32_t bs, int lane) {
  const uint32_t a = bs + (lane >> 2) * (D / 4);
#pragma unroll
  for (int i = 0; i < D / 16; i += 4) {
    uint32_t x, y, z, w;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(a + 4 * i));
    cs.w[i] = x, cs.w[i + 1] = y, cs.w[i + 2] = z, cs.w[i + 3] = w;
  }
}

// acc[D/8][4] += A (16 x 32, fragment a, scale sfa byte 0) x B, B^T given as D rows of a col-layout tile
// (RB bytes per row; this k-step = bytes 32 kc .. 32 kc + 31 of each row) whose 32-token block is block kk of
// its 64-token scale tile (scale words cs).
template <int D, int RB>
__device__ __forceinline__ void pv_product(float (&acc)[D / 8][4], const uint32_t (&a)[4], uint32_t sfa,
                                           uint32_t btile, const ColScales<D>& cs, int kc, int kk, int lane) {
#pragma unroll
  for (int j = 0; j < D / 8; j += 2) {
    uint32_t b0, b1, b2, b3;
    ldsm4(btile + swz<RB>(8 * j + (lane & 7) + (lane >> 4) * 8, 2 * kc + ((lane >> 3) & 1)), b0, b1, b2, b3);
    const int i0 = 2 * j + kk, i1 = 2 * (j + 1) + kk;  // byte indices of n-tiles j, j + 1
    mma8(acc[j], a, b0, b1, sfa, cs.w[i0 >> 2], 0, i0 & 3);
    mma8(acc[j + 1], a, b2, b3, sfa, cs.w[i1 >> 2], 0, i1 & 3);
  }
}

// Store a 16 x D fp32 accumulator (rows row0, row0 + 8 of this lane; rows >= N skipped) times `mul`.
template <typename T, int D>
__device__ __forceinline__ void store_rows(T* base, int64_t sn, const float (&acc)[D / 8][4], int row0, int N,
                                           float mul0, float mul1, int lane) {
  const int t = lane & 3;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int row = row0 + 8 * h;
    if (row >= N) continue;
    const float m = h ? mul1 : mul0;
    T* out = base + row * sn;
#pragma unroll
    for (int j = 0; j < D / 8; ++j) {
      const float v0 = acc[j][2 * h] * m, v1 = acc[j][2 * h + 1] * m;
      if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        *reinterpret_cast<__nv_bfloat162*>(out + 8 * j + 2 * t) = __floats2bfloat162_rn(v0, v1);
      } else {
        *reinterpret_cast<__half2*>(out + 8 * j + 2 * t) = __floats2half2_rn(v0, v1);
      }
    }
  }
}

// --- quantization -----------------------------------------------------------------------------------------

struct QuantArgs {
  const void* x;  // [B, Hx, N, D] with element strides sb, sh, sn (stride 1 along D)
  int64_t sb, sh, sn;
  const void* o;  // optional, same shape (strides ob, oh, on): delta = rowsum(x * o)
  int64_t ob, oh, on;
  const float* mean;  // optional [B * Hx, D], subtracted before quantization
  u8 *row, *col, *col_s;
  float* row_s;  // [B * Hx, Np] per-token scales of the int8 row layout
  float* delta;  // [B * Hx, Np] if o
  __nv_bfloat16* raw;  // optional [B * Hx, Np, D]: BF16 of the (mean-subtracted) input, zero padding
  // optional [B * Hx, Np, D] in the input dtype: the quantized values (of the row layout if requested, else of
  // the col layout), i.e. the inputs the 8-bit GEMMs actually see, for an exact backward of the 8-bit forward
  void* deq;
  // optional [B * Hx, Np, D] FP16 copy of the row layout: f16_mode 1 the INT8 codes (exact), 2 the unrounded
  // x * 127 / amax (|x| <= 127); value = f16 * row_s either way
  __half* f16;
  int f16_mode;
  int Hx, N, Np;
};

template <typename T>
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return pack_bf16(lo, hi);
  } else {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
  }
}

template <typename T>
__device__ __forceinline__ void load8(const T* p, float (&v)[8]) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[i]));
      v[2 * i] = f.x, v[2 * i + 1] = f.y;
    } else {
      const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[i]));
      v[2 * i] = f.x, v[2 * i + 1] = f.y;
    }
  }
}

// One CTA per 64 tokens of one (batch, head): the tile goes through shared memory once and comes out in both
// layouts.
template <typename T, int D>
__global__ void __launch_bounds__(THREADS) attn_quant_kernel(QuantArgs a) {
  constexpr int V = D / 8;
  __shared__ float tile[64][D + 1];
  const int t0 = blockIdx.x * 64, bh = blockIdx.y, b = bh / a.Hx, h = bh % a.Hx;
  const T* x = static_cast<const T*>(a.x) + b * a.sb + h * a.sh;
  const T* o = a.o ? static_cast<const T*>(a.o) + b * a.ob + h * a.oh : nullptr;
#pragma unroll
  for (int i0 = 0; i0 < 64 * V; i0 += THREADS) {
    const int i = i0 + threadIdx.x, r = i / V, c = (i % V) * 8, n = t0 + r;
    float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float dot = 0.0f;
    if (n < a.N) {
      load8(x + n * a.sn + c, v);
      if (o != nullptr) {
        float w[8];
        load8(o + n * a.on + c, w);
#pragma unroll
        for (int k = 0; k < 8; ++k) dot = fmaf(v[k], w[k], dot);
      }
      if (a.mean != nullptr) {
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] -= a.mean[bh * D + c + k];
      }
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) tile[r][c + k] = v[k];
    if (a.raw != nullptr) {
      uint4 w;
      w.x = pack_bf16(v[0], v[1]), w.y = pack_bf16(v[2], v[3]), w.z = pack_bf16(v[4], v[5]), w.w = pack_bf16(v[6], v[7]);
      *reinterpret_cast<uint4*>(a.raw + (static_cast<int64_t>(bh) * a.Np + n) * D + c) = w;
    }
    // the V threads of a row are consecutive lanes
    if (a.delta != nullptr) {
#pragma unroll
      for (int off = V / 2; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
      if (i % V == 0) a.delta[static_cast<int64_t>(bh) * a.Np + n] = dot;  // 0 for padding rows
    }
    if (a.row != nullptr) {
      // int8 per token: q = rn(x * (127 / amax)), scale amax / 127 (NaN if the row has a NaN: it propagates)
      float am = 0.0f;
      int nan = 0;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        am = fmaxf(am, fabsf(v[k]));
        nan |= v[k] != v[k];
      }
#pragma unroll
      for (int off = V / 2; off > 0; off >>= 1) {
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, off));
        nan |= __shfl_xor_sync(0xffffffffu, nan, off);
      }
      const float inv = am > 0.0f ? 127.0f / am : 0.0f;
      const float sc = nan ? __int_as_float(0x7fffffff) : am / 127.0f;
      uint32_t w[2] = {0u, 0u};
      float dq[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int qv = min(127, max(-127, __float2int_rn(v[k] * inv)));
        w[k >> 2] |= (static_cast<uint32_t>(qv) & 0xffu) << (8 * (k & 3));
        dq[k] = static_cast<float>(qv) * sc;
      }
      const int64_t row = static_cast<int64_t>(bh) * a.Np + t0 + r;
      if (a.f16 != nullptr) {
        float h[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) h[k] = a.f16_mode == 1 ? static_cast<float>(min(127, max(-127, __float2int_rn(v[k] * inv)))) : v[k] * inv;
        *reinterpret_cast<uint4*>(a.f16 + row * D + c) =
            make_uint4(pack_h2(h[0], h[1]), pack_h2(h[2], h[3]), pack_h2(h[4], h[5]), pack_h2(h[6], h[7]));
      }
      *reinterpret_cast<uint2*>(a.row + row * D + c) = make_uint2(w[0], w[1]);
      if (i % V == 0) a.row_s[row] = sc;
      if (a.deq != nullptr)
        *reinterpret_cast<uint4*>(static_cast<T*>(a.deq) + row * D + c) =
            make_uint4(pack2<T>(dq[0], dq[1]), pack2<T>(dq[2], dq[3]), pack2<T>(dq[4], dq[5]), pack2<T>(dq[6], dq[7]));
    }
  }
  __syncthreads();
  if (a.col != nullptr) {
    for (int i = threadIdx.x; i < 2 * D; i += THREADS) {
      const int d = i % D, blk = i / D;
      float v[32], am = 0.0f;
#pragma unroll
      for (int k = 0; k < 32; ++k) {
        v[k] = tile[32 * blk + sigma(k)][d];
        am = fmaxf(am, fabsf(v[k]));
      }
      const uint32_t e = e8_of(am);
      const float inv = inv_scale(e);
      uint32_t w[8];
#pragma unroll
      for (int q = 0; q < 8; ++q) w[q] = pack4(v[4 * q] * inv, v[4 * q + 1] * inv, v[4 * q + 2] * inv, v[4 * q + 3] * inv);
      uint4* dst = reinterpret_cast<uint4*>(a.col + (static_cast<int64_t>(bh) * D + d) * a.Np + t0 + 32 * blk);
      dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
      dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
      a.col_s[(static_cast<int64_t>(bh) * (a.Np / 64) + t0 / 64) * (2 * D) + ((d & 7) * (D / 8) + (d >> 3)) * 2 + blk] =
          static_cast<u8>(e);
      if (a.deq != nullptr && a.row == nullptr) {  // consecutive lanes: consecutive channels of one token
        const float sc = __uint_as_float(e << 23);  // 2^(e - 127)
        T* dst = static_cast<T*>(a.deq) + (static_cast<int64_t>(bh) * a.Np + t0 + 32 * blk) * D + d;
#pragma unroll
        for (int q = 0; q < 8; ++q)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const float2 f = __half22float2(e4m3x2_to_h2(static_cast<uint16_t>(w[q] >> (16 * hh))));
            const int k = 4 * q + 2 * hh;
            if constexpr (std::is_same_v<T, __nv_bfloat16>) {
              dst[sigma(k) * D] = __float2bfloat16_rn(f.x * sc);
              dst[sigma(k + 1) * D] = __float2bfloat16_rn(f.y * sc);
            } else {
              dst[sigma(k) * D] = __float2half_rn(f.x * sc);
              dst[sigma(k + 1) * D] = __float2half_rn(f.y * sc);
            }
          }
      }
    }
  }
}

// --- forward ----------------------------------------------------------------------------------------------

struct FwdArgs {
  const u8 *q, *k, *vt, *vts;
  const float *qs, *ks;
  void* o;  // [B, H, N, D] with element strides o_sb, o_sh, o_sn
  int64_t o_sb, o_sh, o_sn;
  float* lse;  // [B * H, Np]: log2-domain log-sum-exp of the scaled scores, +inf for padding rows
  int H, Hk, N, Np, causal;
  float scale_log2;  // softmax scale * log2(e)
};

// --- forward ----------------------------------------------------------------------------------------------
// Each warp owns MT m16 tiles of queries (16 MT rows) and every B fragment it loads from shared memory feeds MT
// MMAs: with MT = 2 the ldmatrix traffic per MMA halves (at 16 rows per warp the operand loads used ~half of
// the shared-memory bandwidth the 8-bit MMAs leave room for). CTAs of WARPS = 4 warps, two per SM: the two
// CTAs synchronize independently, so one's softmax overlaps the other's tensor work. Key tiles of 64 go through
// a 2-stage cp.async pipeline and are processed as two 32-key halves (one P V k-step each) to bound registers.

template <int D, int MT, int WARPS>
struct FwdCfg {
  static constexpr int NTH = 32 * WARPS;
  static constexpr int BQ = 16 * MT * WARPS;  // queries per CTA
  static constexpr int K = 0, KSC = 64 * D, VT = KSC + 256, VTS = VT + D * 64, STAGE = VTS + 2 * D;
  static constexpr int STAGES = 2;
  static constexpr int BYTES = STAGES * STAGE;  // the query tile is staged in stage 1 first
  static_assert(BQ * D <= STAGE, "the query tile must fit in a stage");
  static_assert(WARPS != 4 || BYTES <= kSmem2, "two CTAs per SM");
};

// P2: P V as e4m3(P) V + e4m3(residual) V (one more MXFP8 MMA): O to ~2^-8 in P instead of 2^-4, for an exact
// backward that takes Delta = rowsum(dO O) from this O (attention.py's backward="bf16").
template <typename T, int D, int MT, int WARPS, bool P2>
__global__ void __launch_bounds__(32 * WARPS, 8 / WARPS) attn_fwd_kernel(FwdArgs p) {
  using C = FwdCfg<D, MT, WARPS>;
  constexpr int KS = D / 32, ND = D / 8, NTH = C::NTH, WR = 16 * MT;
  extern __shared__ __align__(128) u8 smem[];
  const uint32_t sbase = su32(smem);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int q0 = (gridDim.x - 1 - blockIdx.x) * C::BQ;  // longest causal rows first
  const int bh = blockIdx.y, b = bh / p.H, h = bh % p.H, bhk = b * p.Hk + h / (p.H / p.Hk);
  const u8* kg = p.k + static_cast<int64_t>(bhk) * p.Np * D;
  const float* ksg = p.ks + static_cast<int64_t>(bhk) * p.Np;
  const u8* vtg = p.vt + static_cast<int64_t>(bhk) * D * p.Np;
  const u8* vtsg = p.vts + static_cast<int64_t>(bhk) * (p.Np / 64) * (2 * D);
  const int nkeys = p.causal ? min(p.N, q0 + C::BQ) : p.N;
  const int ntiles = (nkeys + 63) / 64;
  auto stage = [&](int i) { return sbase + i * C::STAGE; };
  auto load_kv = [&](int j, int i) {
    const uint32_t st = stage(i);
    load_tile<D, 64, NTH>(st + C::K, kg + static_cast<int64_t>(j) * 64 * D, D);
    load_raw<256, NTH>(st + C::KSC, reinterpret_cast<const u8*>(ksg + j * 64));
    load_tile<64, D, NTH>(st + C::VT, vtg + j * 64, p.Np);
    load_raw<2 * D, NTH>(st + C::VTS, vtsg + static_cast<int64_t>(j) * 2 * D);
  };

  // prologue: the query tile through stage 1, key / value tile 0 into stage 0, then tile 1 into stage 1
  load_tile<D, C::BQ, NTH>(stage(1), p.q + (static_cast<int64_t>(bh) * p.Np + q0) * D, D);
  load_kv(0, 0);
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  const int wrow = q0 + warp * WR;  // first query row of this warp
  uint32_t qa[MT][KS][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) load_a_frags<D>(qa[mt], stage(1), warp * WR + 16 * mt, lane);
  __syncthreads();
  if (ntiles > 1) load_kv(1, 1);
  cp_commit();

  // scores in the log2 domain: (int dot * k scale) * qc, qc = q scale * softmax scale * log2 e (a zero q
  // scale becomes tiny: masked scores must stay -inf after the multiplication)
  const float* qsg = p.qs + static_cast<int64_t>(bh) * p.Np;
  float qc[MT][2], m[MT][2], l[MT][2];
  float o[MT][ND][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const float c = qsg[wrow + 16 * mt + g + 8 * hh] * p.scale_log2;
      qc[mt][hh] = c == 0.0f ? 1e-30f : c;
      m[mt][hh] = -kInf;
      l[mt][hh] = 0.0f;
    }
#pragma unroll
    for (int j = 0; j < ND; ++j) o[mt][j][0] = o[mt][j][1] = o[mt][j][2] = o[mt][j][3] = 0.0f;
  }

  for (int j = 0; j < ntiles; ++j) {
    if (j > 0) {  // tile j landed and every warp is done with tile j - 1: refill its stage with tile j + 1
      cp_wait<0>();
      __syncthreads();
      if (j + 1 < ntiles) load_kv(j + 1, (j + 1) & 1);
      cp_commit();
    }
    const int key0 = j * 64;
    if (p.causal && key0 > wrow + WR - 1) continue;  // every key of the tile is after every row of this warp
    const uint32_t st = stage(j & 1);
    ColScales<D> vsc;
    load_col_scales<D>(vsc, st + C::VTS, lane);
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int k0 = key0 + 32 * hf;
      if (p.causal && k0 > wrow + WR - 1) break;
      // S = Q K^T for 32 keys: every K fragment feeds MT MMAs
      int acc[MT][4][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int n = 0; n < 4; ++n) acc[mt][n][0] = acc[mt][n][1] = acc[mt][n][2] = acc[mt][n][3] = 0;
#pragma unroll
      for (int ks = 0; ks < KS; ++ks)
#pragma unroll
        for (int n = 0; n < 4; n += 2) {
          uint32_t b0, b1, b2, b3;
          ldsm4(st + C::K + 32 * hf * D + swz<D>(8 * n + (lane & 7) + (lane >> 4) * 8, 2 * ks + ((lane >> 3) & 1)),
                b0, b1, b2, b3);
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            mma_s8(acc[mt][n], qa[mt][ks], b0, b1);
            mma_s8(acc[mt][n + 1], qa[mt][ks], b2, b3);
          }
        }
      float s[MT][4][4];
      {
        constexpr float kMagic = 12582912.0f;  // see qk_product
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          const float2 rb = lds_f32x2(st + C::KSC + 4 * (32 * hf + 8 * n + 2 * t));
          const float c0 = -kMagic * rb.x, c1 = -kMagic * rb.y;
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            s[mt][n][0] = fmaf(__int_as_float(acc[mt][n][0] + 0x4B400000), rb.x, c0);
            s[mt][n][1] = fmaf(__int_as_float(acc[mt][n][1] + 0x4B400000), rb.y, c1);
            s[mt][n][2] = fmaf(__int_as_float(acc[mt][n][2] + 0x4B400000), rb.x, c0);
            s[mt][n][3] = fmaf(__int_as_float(acc[mt][n][3] + 0x4B400000), rb.y, c1);
          }
        }
      }
      if ((p.causal && k0 + 31 > wrow) || k0 + 32 > p.N) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int key = k0 + 8 * n + 2 * t + (e & 1), row = wrow + 16 * mt + g + 8 * (e >> 1);
              if (key >= p.N || (p.causal && key > row)) s[mt][n][e] = -kInf;
            }
      }
      // online softmax with a lazily updated maximum: O and l are rescaled only when some row's
      // maximum grew by more than kSlack (warp-uniform); otherwise the stale maximum m bounds
      // p = 2^(s - m + kPOff) by 2^(kPOff + kSlack) = 2^8.5 < 448 (E4M3's max). P's e4m3 values carry the
      // factor 2^kPOff (its A scale stays 1; l carries the same factor)
      float mn[MT][2];
      bool grow = false;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const float2 xm = row_max<4>(s[mt]);
        mn[mt][0] = fmaxf(m[mt][0], quad_max(xm.x) * qc[mt][0]);
        mn[mt][1] = fmaxf(m[mt][1], quad_max(xm.y) * qc[mt][1]);
        grow |= mn[mt][0] > m[mt][0] + kSlack || mn[mt][1] > m[mt][1] + kSlack;
      }
      if (__any_sync(0xffffffffu, grow)) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          float al[2];
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            al[hh] = mn[mt][hh] == -kInf ? 1.0f : ex2(m[mt][hh] - mn[mt][hh]);
            m[mt][hh] = mn[mt][hh];
            l[mt][hh] *= al[hh];
          }
#pragma unroll
          for (int n = 0; n < ND; ++n) {
            o[mt][n][0] *= al[0], o[mt][n][1] *= al[0];
            o[mt][n][2] *= al[1], o[mt][n][3] *= al[1];
          }
        }
      }
      uint32_t pa[MT][4], pl[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const float ms0 = (m[mt][0] == -kInf ? 0.0f : m[mt][0]) - kPOff;
        const float ms1 = (m[mt][1] == -kInf ? 0.0f : m[mt][1]) - kPOff;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          s[mt][n][0] = ex2(fmaf(s[mt][n][0], qc[mt][0], -ms0));
          s[mt][n][1] = ex2(fmaf(s[mt][n][1], qc[mt][0], -ms0));
          s[mt][n][2] = ex2(fmaf(s[mt][n][2], qc[mt][1], -ms1));
          s[mt][n][3] = ex2(fmaf(s[mt][n][3], qc[mt][1], -ms1));
        }
        const float2 ls = row_sum<4>(s[mt]);
        l[mt][0] += ls.x;
        l[mt][1] += ls.y;
        acc_to_a<4>(pa[mt], s[mt], 0, 1.0f, 1.0f);
        if constexpr (P2) acc_resid<4>(pl[mt], pa[mt], s[mt], 0);
      }
      // O += P V (k-step hf of the tile): every V fragment feeds MT MMAs
#pragma unroll
      for (int n = 0; n < ND; n += 2) {
        uint32_t b0, b1, b2, b3;
        ldsm4(st + C::VT + swz<64>(8 * n + (lane & 7) + (lane >> 4) * 8, 2 * hf + ((lane >> 3) & 1)), b0, b1, b2, b3);
        const int i0 = 2 * n + hf, i1 = 2 * (n + 1) + hf;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma8(o[mt][n], pa[mt], b0, b1, 0x7f7f7f7fu, vsc.w[i0 >> 2], 0, i0 & 3);
          mma8(o[mt][n + 1], pa[mt], b2, b3, 0x7f7f7f7fu, vsc.w[i1 >> 2], 0, i1 & 3);
          if constexpr (P2) {  // the residuals carry 16x: A scale 2^-4
            mma8(o[mt][n], pl[mt], b0, b1, 0x7b7b7b7bu, vsc.w[i0 >> 2], 0, i0 & 3);
            mma8(o[mt][n + 1], pl[mt], b2, b3, 0x7b7b7b7bu, vsc.w[i1 >> 2], 0, i1 & 3);
          }
        }
      }
    }
  }
  cp_wait<0>();
  T* ob = static_cast<T*>(p.o) + b * p.o_sb + h * p.o_sh;
  float* lse = p.lse + static_cast<int64_t>(bh) * p.Np;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const float l0 = quad_sum(l[mt][0]), l1 = quad_sum(l[mt][1]);
    const int row0 = wrow + 16 * mt + g, row1 = row0 + 8;
    store_rows<T, D>(ob, p.o_sn, o[mt], row0, p.N, 1.0f / l0, 1.0f / l1, lane);
    if (t == 0) {
      lse[row0] = row0 < p.N ? m[mt][0] + __log2f(l0) - kPOff : kInf;
      lse[row1] = row1 < p.N ? m[mt][1] + __log2f(l1) - kPOff : kInf;
    }
  }
}

// --- backward ---------------------------------------------------------------------------------------------

struct BwdArgs {
  const u8* q;                      // Q, int8 row layout [B * H]
  const __half* q16;                // its INT8 codes in FP16 [B * H, Np, D] (dK = dS^T Q)
  const u8* k;                      // K (mean-subtracted) [B * Hk], int8 row layout
  const __half* k16;                // its INT8 codes in FP16 [B * Hk, Np, D] (dQ = dS K)
  const u8* v;                      // V row layout [B * Hk]
  const u8* g;                      // dO, int8 row layout [B * H]
  const __half* g16;                // dO * 127 / amax (unrounded) in FP16 [B * H, Np, D] (dV = P^T dO)
  const float *qs, *ks, *vs, *gs;   // per-token scales of the row layouts
  const u8 *kt, *kts;               // K, col layout (B = P K in the dQ kernel)
  const float *lse, *delta;         // [B * H, Np]
  // [B * H, Np]: Delta' = Delta + r, the row sums of P dP as the backward computes them (written by the dQ
  // kernel, read by the dK / dV kernel, so that the rows of its dS sum to zero as well)
  float* delta2;
  void *dq, *dk, *dv;               // [B, H(k), N, D] with element strides below
  int64_t dq_sb, dq_sh, dq_sn, dk_sb, dk_sh, dk_sn;  // dv has dk's strides
  int H, Hk, N, Np, causal;
  float scale_log2, scale;
};

// CTAs of W warps (16 rows each), two per SM at W = 4: as in the forward, two independently synchronizing CTAs
// keep the tensor pipe busy during each other's element-wise work. 2-stage cp.async pipelines; the operands that
// stay in registers (Q / dO fragments here, K / V there) are staged through stage 1 when they fit.
template <int D, int W>
struct DqCfg {
  static constexpr int NTH = 32 * W, BQ = 16 * W;  // queries per CTA
  static constexpr int K = 0, KSC = K + 64 * D, V = KSC + 256, K16 = V + 64 * D, KT = K16 + 64 * D * 2,
                       KTS = KT + D * 64, VSC = KTS + 2 * D, STAGE = VSC + 256;
  static constexpr bool kStaged = 2 * BQ * D <= STAGE;
  static constexpr int STAGE0 = kStaged ? 0 : 2 * BQ * D;
  static constexpr int QOFF = kStaged ? STAGE : 0;  // query tile, then dO tile
  static constexpr int BYTES = STAGE0 + 2 * STAGE;
  static_assert(W == 8 && BYTES <= 101376, "one CTA of 8 warps per SM");
};

template <typename T, int D, int W>
__global__ void __launch_bounds__(32 * W, 8 / W) attn_dq_kernel(BwdArgs p) {
  using C = DqCfg<D, W>;
  constexpr int KS = D / 32, NT = 8, ND = D / 8, NTH = C::NTH;
  extern __shared__ __align__(128) u8 smem[];
  const uint32_t sbase = su32(smem);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int q0 = (gridDim.x - 1 - blockIdx.x) * C::BQ;
  const int bh = blockIdx.y, b = bh / p.H, h = bh % p.H, bhk = b * p.Hk + h / (p.H / p.Hk);
  const u8* kg = p.k + static_cast<int64_t>(bhk) * p.Np * D;
  const float* ksg = p.ks + static_cast<int64_t>(bhk) * p.Np;
  const u8* vg = p.v + static_cast<int64_t>(bhk) * p.Np * D;
  const float* vsg = p.vs + static_cast<int64_t>(bhk) * p.Np;
  const u8* k16g = reinterpret_cast<const u8*>(p.k16 + static_cast<int64_t>(bhk) * p.Np * D);  // FP16 codes
  const u8* ktg = p.kt + static_cast<int64_t>(bhk) * D * p.Np;
  const u8* ktsg = p.kts + static_cast<int64_t>(bhk) * (p.Np / 64) * (2 * D);
  const int nkeys = p.causal ? min(p.N, q0 + C::BQ) : p.N;
  const int ntiles = (nkeys + 63) / 64;
  auto stage = [&](int i) { return sbase + C::STAGE0 + i * C::STAGE; };
  auto load_kv = [&](int j, int i) {
    const uint32_t st = stage(i);
    load_tile<D, 64, NTH>(st + C::K, kg + static_cast<int64_t>(j) * 64 * D, D);
    load_raw<256, NTH>(st + C::KSC, reinterpret_cast<const u8*>(ksg + j * 64));
    load_tile<D, 64, NTH>(st + C::V, vg + static_cast<int64_t>(j) * 64 * D, D);
    load_tile<2 * D, 64, NTH>(st + C::K16, k16g + static_cast<int64_t>(j) * 64 * D * 2, 2 * D);
    load_tile<64, D, NTH>(st + C::KT, ktg + j * 64, p.Np);
    load_raw<2 * D, NTH>(st + C::KTS, ktsg + static_cast<int64_t>(j) * 2 * D);
    load_raw<256, NTH>(st + C::VSC, reinterpret_cast<const u8*>(vsg + j * 64));
  };
  const int64_t rbase = static_cast<int64_t>(bh) * p.Np;
  load_tile<D, C::BQ, NTH>(sbase + C::QOFF, p.q + (rbase + q0) * D, D);
  load_tile<D, C::BQ, NTH>(sbase + C::QOFF + C::BQ * D, p.g + (rbase + q0) * D, D);
  load_kv(0, 0);
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  uint32_t qa[KS][4], ga[KS][4];
  load_a_frags<D>(qa, sbase + C::QOFF, warp * 16, lane);
  load_a_frags<D>(ga, sbase + C::QOFF + C::BQ * D, warp * 16, lane);
  __syncthreads();
  if (ntiles > 1) load_kv(1, 1);
  cp_commit();

  const int wrow = q0 + warp * 16, row0 = wrow + g, row1 = row0 + 8;
  const float qc[2] = {p.qs[rbase + row0] * p.scale_log2, p.qs[rbase + row1] * p.scale_log2};
  const float gc[2] = {p.gs[rbase + row0], p.gs[rbase + row1]};
  const float nlse[2] = {-p.lse[rbase + row0], -p.lse[rbase + row1]};
  const float nd[2] = {-p.delta[rbase + row0], -p.delta[rbase + row1]};
  float rsum[2] = {0.0f, 0.0f};  // row sums of dS
  float dq[ND][4], bq[ND][4];    // dS K and P K
#pragma unroll
  for (int j = 0; j < ND; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) dq[j][e] = bq[j][e] = 0.0f;

  for (int j = 0; j < ntiles; ++j) {
    if (j > 0) {  // tile j landed and every warp is done with tile j - 1: refill its stage with tile j + 1
      cp_wait<0>();
      __syncthreads();
      if (j + 1 < ntiles) load_kv(j + 1, (j + 1) & 1);
      cp_commit();
    }
    const int key0 = j * 64;
    if (p.causal && key0 > wrow + 15) continue;
    const uint32_t st = stage(j & 1);
    float s[NT][4], dp[NT][4];
    qk_product<D, NT>(s, qa, st + C::K, st + C::KSC, lane);
    qk_product<D, NT>(dp, ga, st + C::V, st + C::VSC, lane);
    const bool mask = (p.causal && key0 + 63 > wrow) || key0 + 64 > p.N;
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int key = key0 + 8 * n + 2 * t + (e & 1), row = e < 2 ? row0 : row1, h2 = e >> 1;
        float pr = ex2(fmaf(s[n][e], qc[h2], nlse[h2]));
        if (mask && (key >= p.N || (p.causal && key > row))) pr = 0.0f;
        s[n][e] = pr * fmaf(dp[n][e], gc[h2], nd[h2]);  // dS = P (dP - Delta)
        dp[n][e] = pr * 256.0f;                         // P with its fixed e4m3 factor (its A scale is 2^-8)
      }
    {
      const float2 rs = row_sum<NT>(s);
      rsum[0] += rs.x;
      rsum[1] += rs.y;
    }
    {  // B += P K (MXFP8)
      ColScales<D> ksc;
      load_col_scales<D>(ksc, st + C::KTS, lane);
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        uint32_t a[4];
        acc_to_a<NT>(a, dp, kk, 1.0f, 1.0f);
        pv_product<D, 64>(bq, a, 0x77777777u, st + C::KT, ksc, kk, kk, lane);
      }
    }
    // dQ += dS K in FP16: A = dS_ij ks_j / (its row maximum over the tile), B = K's INT8 codes (row-major FP16
    // tile through ldmatrix.trans); the C fragments of n-tiles 2 kk, 2 kk + 1 are the A fragment of k16-step kk
    {
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const float2 kb = lds_f32x2(st + C::KSC + 4 * (8 * n + 2 * t));
        s[n][0] *= kb.x, s[n][1] *= kb.y, s[n][2] *= kb.x, s[n][3] *= kb.y;
      }
      float sc[2], inv[2];
      row_scales<NT>(s, sc, inv);
      uint32_t a[4][4];
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) acc_to_h<NT>(a[kk], s, kk, inv[0], inv[1]);
      f16_product<D, 4>(dq, a, st + C::K16, sc, lane);
    }
  }
  cp_wait<0>();
  const float r0 = quad_sum(rsum[0]), r1 = quad_sum(rsum[1]);
  if (t == 0) {
    if (row0 < p.N) p.delta2[rbase + row0] = r0 - nd[0];
    if (row1 < p.N) p.delta2[rbase + row1] = r1 - nd[1];
  }
#pragma unroll
  for (int j = 0; j < ND; ++j) {  // dQ - r B: the dQ of the consistent Delta' = Delta + r
    dq[j][0] = fmaf(-r0, bq[j][0], dq[j][0]), dq[j][1] = fmaf(-r0, bq[j][1], dq[j][1]);
    dq[j][2] = fmaf(-r1, bq[j][2], dq[j][2]), dq[j][3] = fmaf(-r1, bq[j][3], dq[j][3]);
  }
  T* out = static_cast<T*>(p.dq) + b * p.dq_sb + h * p.dq_sh;
  store_rows<T, D>(out, p.dq_sn, dq, row0, p.N, p.scale, p.scale, lane);
}

template <int D, int W>
struct DkvCfg {
  static constexpr int NTH = 32 * W, BK = 16 * W;  // keys per CTA
  // per stage: 32 queries
  static constexpr int Q = 0, QSC = Q + 32 * D, G = QSC + 128, GSC = G + 32 * D, Q16 = GSC + 128,
                       G16 = Q16 + 32 * D * 2, LSE = G16 + 32 * D * 2, DEL = LSE + 128, STAGE = DEL + 128;
  static constexpr bool kStaged = 2 * BK * D <= STAGE;
  static constexpr int STAGE0 = kStaged ? 0 : 2 * BK * D;
  static constexpr int KOFF = kStaged ? STAGE : 0;  // key tile, then value tile
  static constexpr int BYTES = STAGE0 + 2 * STAGE;
  static_assert(W == 8 && BYTES <= 101376, "one CTA of 8 warps per SM");
};

template <typename T, int D, int W>
__global__ void __launch_bounds__(32 * W, 8 / W) attn_dkdv_kernel(BwdArgs p) {
  using C = DkvCfg<D, W>;
  constexpr int KS = D / 32, NT = 4, ND = D / 8, NTH = C::NTH;
  extern __shared__ __align__(128) u8 smem[];
  const uint32_t sbase = su32(smem);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int k0 = blockIdx.x * C::BK;
  const int bhk = blockIdx.y, b = bhk / p.Hk, hk = bhk % p.Hk, G = p.H / p.Hk;
  const int qt0 = p.causal ? k0 / 32 : 0, nqt = (p.N + 31) / 32 - qt0;  // query tiles per head
  const int iters = G * max(nqt, 0);
  auto stage = [&](int i) { return sbase + C::STAGE0 + i * C::STAGE; };
  auto load_q = [&](int it, int i) {
    const int hq = b * p.H + hk * G + it / nqt, qq = (qt0 + it % nqt) * 32;
    const uint32_t st = stage(i);
    const int64_t rowoff = static_cast<int64_t>(hq) * p.Np + qq;
    load_tile<D, 32, NTH>(st + C::Q, p.q + rowoff * D, D);
    load_raw<128, NTH>(st + C::QSC, reinterpret_cast<const u8*>(p.qs + rowoff));
    load_tile<D, 32, NTH>(st + C::G, p.g + rowoff * D, D);
    load_raw<128, NTH>(st + C::GSC, reinterpret_cast<const u8*>(p.gs + rowoff));
    load_tile<2 * D, 32, NTH>(st + C::Q16, reinterpret_cast<const u8*>(p.q16 + rowoff * D), 2 * D);
    load_tile<2 * D, 32, NTH>(st + C::G16, reinterpret_cast<const u8*>(p.g16 + rowoff * D), 2 * D);
    load_raw<128, NTH>(st + C::LSE, reinterpret_cast<const u8*>(p.lse + rowoff));
    load_raw<128, NTH>(st + C::DEL, reinterpret_cast<const u8*>(p.delta2 + rowoff));
  };
  const int64_t kbase = static_cast<int64_t>(bhk) * p.Np;
  load_tile<D, C::BK, NTH>(sbase + C::KOFF, p.k + (kbase + k0) * D, D);
  load_tile<D, C::BK, NTH>(sbase + C::KOFF + C::BK * D, p.v + (kbase + k0) * D, D);
  if (iters > 0) load_q(0, 0);
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  uint32_t ka[KS][4], va[KS][4];
  load_a_frags<D>(ka, sbase + C::KOFF, warp * 16, lane);
  load_a_frags<D>(va, sbase + C::KOFF + C::BK * D, warp * 16, lane);
  __syncthreads();
  if (iters > 1) load_q(1, 1);
  cp_commit();

  const int wkey = k0 + warp * 16, key0r = wkey + g, key1r = key0r + 8;
  const float kc[2] = {p.ks[kbase + key0r] * p.scale_log2, p.ks[kbase + key1r] * p.scale_log2};
  const float vc[2] = {p.vs[kbase + key0r], p.vs[kbase + key1r]};
  float dk[ND][4], dv[ND][4];
#pragma unroll
  for (int j = 0; j < ND; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) dk[j][e] = dv[j][e] = 0.0f;

  for (int it = 0; it < iters; ++it) {
    if (it > 0) {
      cp_wait<0>();
      __syncthreads();
      if (it + 1 < iters) load_q(it + 1, (it + 1) & 1);
      cp_commit();
    }
    const int qq = (qt0 + it % nqt) * 32;
    if (p.causal && qq + 31 < wkey) continue;  // every query of the tile precedes every key of this warp
    const uint32_t st = stage(it & 1);
    float s[NT][4], dp[NT][4];
    qk_product<D, NT>(s, ka, st + C::Q, st + C::QSC, lane);   // S^T / k scale (log2 domain / kc)
    qk_product<D, NT>(dp, va, st + C::G, st + C::GSC, lane);  // dP^T / v scale
    const bool mask = p.causal && qq < wkey + 15;
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int col = 8 * n + 2 * t + e;  // query within the tile
        const float lq = -lds_f32(st + C::LSE + 4 * col), ndq = -lds_f32(st + C::DEL + 4 * col);
        const float gq = lds_f32(st + C::GSC + 4 * col), qsq = lds_f32(st + C::QSC + 4 * col);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int idx = 2 * hh + e, key = hh ? key1r : key0r;
          float pr = ex2(fmaf(s[n][idx], kc[hh], lq));
          if (mask && qq + col < key) pr = 0.0f;
          s[n][idx] = pr * gq;                                     // P dO-scale: A of dV
          dp[n][idx] = pr * fmaf(dp[n][idx], vc[hh], ndq) * qsq;  // dS^T = P (dP - Delta), times the Q scale: A of dK
        }
      }
    {  // dV += P^T dO, dK += dS^T Q in FP16 (A divided by its row maximum over the 32 queries)
      float sc[2], inv[2];
      uint32_t a[2][4];
      row_scales<NT>(s, sc, inv);
      acc_to_h<NT>(a[0], s, 0, inv[0], inv[1]);
      acc_to_h<NT>(a[1], s, 1, inv[0], inv[1]);
      f16_product<D, 2>(dv, a, st + C::G16, sc, lane);
      row_scales<NT>(dp, sc, inv);
      acc_to_h<NT>(a[0], dp, 0, inv[0], inv[1]);
      acc_to_h<NT>(a[1], dp, 1, inv[0], inv[1]);
      f16_product<D, 2>(dk, a, st + C::Q16, sc, lane);
    }
  }
  cp_wait<0>();
  T* dkb = static_cast<T*>(p.dk) + b * p.dk_sb + hk * p.dk_sh;
  T* dvb = static_cast<T*>(p.dv) + b * p.dk_sb + hk * p.dk_sh;
  store_rows<T, D>(dkb, p.dk_sn, dk, key0r, p.N, p.scale, p.scale, lane);
  store_rows<T, D>(dvb, p.dk_sn, dv, key0r, p.N, 1.0f, 1.0f, lane);
}

// --- host -------------------------------------------------------------------------------------------------

// Launchers, one function template per kernel family. (Plain templates with an explicit dispatch below: MSVC's
// host compiler crashed on the stub nvcc generates for kernels launched from generic lambdas.) Each launcher
// instantiation records per device that its kernel may use the dynamic shared memory it needs.
template <typename T, int D>
void launch_quant(const QuantArgs& a, int64_t Np, int64_t BH, cudaStream_t stream) {
  attn_quant_kernel<T, D><<<dim3(static_cast<unsigned>(Np / 64), static_cast<unsigned>(BH)), THREADS, 0, stream>>>(a);
}

template <typename T, int D, bool P2>
void launch_fwd(const FwdArgs& p, int64_t Np, int64_t BH, cudaStream_t stream) {
  using C = FwdCfg<D, 2, 4>;  // 32 query rows per warp, 4 warps per CTA, two CTAs per SM
  static std::atomic<uint64_t> done{0};
  C10_CUDA_CHECK(nvfp4::gemm::ensure_smem_attr(attn_fwd_kernel<T, D, 2, 4, P2>, C::BYTES, done));
  attn_fwd_kernel<T, D, 2, 4, P2><<<dim3(static_cast<unsigned>(Np / C::BQ), static_cast<unsigned>(BH)), C::NTH,
                                    C::BYTES, stream>>>(p);
}

template <typename T, int D>
void launch_bwd(const BwdArgs& p, int64_t Np, int64_t BH, int64_t BHk, cudaStream_t stream) {
  constexpr int W = 8;  // 8 warps per CTA (measured slightly ahead of two 4-warp CTAs per SM for both kernels)
  using CQ = DqCfg<D, W>;
  using CK = DkvCfg<D, W>;
  static std::atomic<uint64_t> done_q{0}, done_k{0};
  C10_CUDA_CHECK(nvfp4::gemm::ensure_smem_attr(attn_dq_kernel<T, D, W>, CQ::BYTES, done_q));
  attn_dq_kernel<T, D, W><<<dim3(static_cast<unsigned>(Np / CQ::BQ), static_cast<unsigned>(BH)), CQ::NTH, CQ::BYTES,
                            stream>>>(p);
  C10_CUDA_CHECK(nvfp4::gemm::ensure_smem_attr(attn_dkdv_kernel<T, D, W>, CK::BYTES, done_k));
  attn_dkdv_kernel<T, D, W><<<dim3(static_cast<unsigned>(Np / CK::BK), static_cast<unsigned>(BHk)), CK::NTH,
                              CK::BYTES, stream>>>(p);
}

// Index of the (dtype, head dim) instantiation: 0 bf16/64, 1 bf16/128, 2 fp16/64, 3 fp16/128.
int variant(c10::ScalarType dt, int64_t D) {
  TORCH_CHECK(D == 64 || D == 128, "nvfp4 attention supports head dims 64 and 128, got ", D);
  TORCH_CHECK(dt == at::kBFloat16 || dt == at::kHalf, "nvfp4 attention needs bf16 or fp16 inputs");
  return (dt == at::kHalf ? 2 : 0) + (D == 128 ? 1 : 0);
}

int64_t padded(int64_t n) { return (n + PAD - 1) / PAD * PAD; }

bool vec_ok(const torch::Tensor& x) {
  // 16-byte vector loads of 8 elements: stride-1 head dim, 8-element aligned strides and base
  return x.stride(3) == 1 && x.stride(0) % 8 == 0 && x.stride(1) % 8 == 0 && x.stride(2) % 8 == 0 &&
         reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0;
}

}  // namespace attn
}  // namespace nvfp4

using OptTensor = c10::optional<torch::Tensor>;
using namespace nvfp4::attn;

// x [B, Hx, N, D] -> {row, row_scales, col, col_scales, delta, raw (bf16), deq (x's dtype)} (unused outputs are
// empty tensors).
std::vector<torch::Tensor> nvfp4_attn_quant(torch::Tensor x, const OptTensor& mean, const OptTensor& o, bool want_row,
                                            bool want_col, bool want_bf16, bool want_deq, int64_t want_f16) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 4, "x must be a 4-D [B, H, N, D] CUDA tensor");
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t B = x.size(0), Hx = x.size(1), N = x.size(2), D = x.size(3), Np = padded(N);
  if (!vec_ok(x)) x = x.contiguous();
  torch::Tensor oo;
  if (o.has_value() && o->defined()) {
    oo = *o;
    TORCH_CHECK(oo.sizes() == x.sizes() && oo.scalar_type() == x.scalar_type() && oo.device() == x.device(),
                "o must match x");
    if (!vec_ok(oo)) oo = oo.contiguous();
  }
  torch::Tensor mn;
  if (mean.has_value() && mean->defined()) {
    mn = mean->to(at::kFloat).contiguous();
    TORCH_CHECK(mn.numel() == B * Hx * D, "mean must have B * H * D elements");
  }
  auto u8o = x.options().dtype(at::kByte);
  torch::Tensor row = want_row ? torch::empty({B, Hx, Np, D}, u8o) : torch::empty({0}, u8o);
  torch::Tensor row_s = want_row ? torch::empty({B, Hx, Np}, x.options().dtype(at::kFloat))
                                 : torch::empty({0}, x.options().dtype(at::kFloat));
  torch::Tensor col = want_col ? torch::empty({B, Hx, D, Np}, u8o) : torch::empty({0}, u8o);
  torch::Tensor col_s = want_col ? torch::empty({B, Hx, Np / 64, 2 * D}, u8o) : torch::empty({0}, u8o);
  torch::Tensor delta = oo.defined() ? torch::empty({B, Hx, Np}, x.options().dtype(at::kFloat))
                                     : torch::empty({0}, x.options().dtype(at::kFloat));
  torch::Tensor raw = want_bf16 ? torch::empty({B, Hx, Np, D}, x.options().dtype(at::kBFloat16))
                                : torch::empty({0}, x.options().dtype(at::kBFloat16));
  TORCH_CHECK(!want_deq || want_row || want_col, "want_deq needs the row or the col layout");
  torch::Tensor deq = want_deq ? torch::empty({B, Hx, Np, D}, x.options()) : torch::empty({0}, x.options());
  TORCH_CHECK(want_f16 >= 0 && want_f16 <= 2 && (want_f16 == 0 || want_row), "want_f16 (1 codes, 2 unrounded) needs the row layout");
  auto h16o = x.options().dtype(at::kHalf);
  torch::Tensor f16 = want_f16 ? torch::empty({B, Hx, Np, D}, h16o) : torch::empty({0}, h16o);
  if (B * Hx == 0) return {row, row_s, col, col_s, delta, raw, deq, f16};
  TORCH_CHECK(B * Hx < 65536, "batch * heads must be < 65536");
  QuantArgs a{};
  a.x = x.data_ptr();
  a.sb = x.stride(0), a.sh = x.stride(1), a.sn = x.stride(2);
  if (oo.defined()) {
    a.o = oo.data_ptr();
    a.ob = oo.stride(0), a.oh = oo.stride(1), a.on = oo.stride(2);
    a.delta = delta.data_ptr<float>();
  }
  a.mean = mn.defined() ? mn.data_ptr<float>() : nullptr;
  a.row = want_row ? row.data_ptr<uint8_t>() : nullptr;
  a.row_s = want_row ? row_s.data_ptr<float>() : nullptr;
  a.col = want_col ? col.data_ptr<uint8_t>() : nullptr;
  a.col_s = want_col ? col_s.data_ptr<uint8_t>() : nullptr;
  a.raw = want_bf16 ? reinterpret_cast<__nv_bfloat16*>(raw.data_ptr()) : nullptr;
  a.deq = want_deq ? deq.data_ptr() : nullptr;
  a.f16 = want_f16 ? reinterpret_cast<__half*>(f16.data_ptr()) : nullptr;
  a.f16_mode = static_cast<int>(want_f16);
  a.Hx = static_cast<int>(Hx), a.N = static_cast<int>(N), a.Np = static_cast<int>(Np);
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (variant(x.scalar_type(), D)) {
    case 0: launch_quant<__nv_bfloat16, 64>(a, Np, B * Hx, stream); break;
    case 1: launch_quant<__nv_bfloat16, 128>(a, Np, B * Hx, stream); break;
    case 2: launch_quant<__half, 64>(a, Np, B * Hx, stream); break;
    default: launch_quant<__half, 128>(a, Np, B * Hx, stream); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {row, row_s, col, col_s, delta, raw, deq, f16};
}

static void check_s32(const torch::Tensor& t, int64_t B, int64_t H, int64_t Np, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.is_contiguous() && t.dim() == 3 && t.size(0) == B &&
                  t.size(1) == H && t.size(2) == Np,
              name, ": expected a contiguous fp32 [", B, ", ", H, ", ", Np, "] tensor");
}

static void check_q8(const torch::Tensor& t, int64_t B, int64_t H, int64_t a, int64_t b, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kByte && t.is_contiguous() && t.dim() == 4 && t.size(0) == B &&
                  t.size(1) == H && t.size(2) == a && t.size(3) == b,
              name, ": expected a contiguous uint8 [", B, ", ", H, ", ", a, ", ", b, "] tensor");
}

// Returns {o [B, N, H, D] (dtype), lse [B, H, Np] (log2 domain)}.
std::vector<torch::Tensor> nvfp4_attn_fwd(torch::Tensor q, torch::Tensor qs, torch::Tensor k, torch::Tensor ks,
                                          torch::Tensor vt, torch::Tensor vts, int64_t N, bool causal, double scale,
                                          c10::ScalarType dtype, bool p2) {
  const c10::cuda::CUDAGuard guard(q.device());
  const int64_t B = q.size(0), H = q.size(1), Np = q.size(2), D = q.size(3), Hk = k.size(1);
  TORCH_CHECK(Np == padded(N) && N > 0, "N does not match the quantized tensors");
  TORCH_CHECK(Hk > 0 && H % Hk == 0, "query heads must be a multiple of key/value heads");
  check_q8(q, B, H, Np, D, "q");
  check_s32(qs, B, H, Np, "q scales");
  check_q8(k, B, Hk, Np, D, "k");
  check_s32(ks, B, Hk, Np, "k scales");
  check_q8(vt, B, Hk, D, Np, "v (col)");
  check_q8(vts, B, Hk, Np / 64, 2 * D, "v (col) scales");
  TORCH_CHECK(B * H < 65536, "batch * heads must be < 65536");
  auto o = torch::empty({B, N, H, D}, q.options().dtype(dtype));
  auto lse = torch::empty({B, H, Np}, q.options().dtype(at::kFloat));
  FwdArgs p{};
  p.q = q.data_ptr<uint8_t>(), p.qs = qs.data_ptr<float>(), p.k = k.data_ptr<uint8_t>();
  p.ks = ks.data_ptr<float>(), p.vt = vt.data_ptr<uint8_t>(), p.vts = vts.data_ptr<uint8_t>();
  p.o = o.data_ptr();
  p.o_sb = N * H * D, p.o_sh = D, p.o_sn = H * D;
  p.lse = lse.data_ptr<float>();
  p.H = static_cast<int>(H), p.Hk = static_cast<int>(Hk), p.N = static_cast<int>(N), p.Np = static_cast<int>(Np);
  p.causal = causal ? 1 : 0;
  p.scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (variant(dtype, D) + (p2 ? 4 : 0)) {
    case 0: launch_fwd<__nv_bfloat16, 64, false>(p, Np, B * H, stream); break;
    case 1: launch_fwd<__nv_bfloat16, 128, false>(p, Np, B * H, stream); break;
    case 2: launch_fwd<__half, 64, false>(p, Np, B * H, stream); break;
    case 3: launch_fwd<__half, 128, false>(p, Np, B * H, stream); break;
    case 4: launch_fwd<__nv_bfloat16, 64, true>(p, Np, B * H, stream); break;
    case 5: launch_fwd<__nv_bfloat16, 128, true>(p, Np, B * H, stream); break;
    case 6: launch_fwd<__half, 64, true>(p, Np, B * H, stream); break;
    default: launch_fwd<__half, 128, true>(p, Np, B * H, stream); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {o, lse};
}

// Returns {dq [B, N, H, D], dk [B, N, Hk, D], dv [B, N, Hk, D]} (dtype).
std::vector<torch::Tensor> nvfp4_attn_bwd(torch::Tensor q, torch::Tensor qs, torch::Tensor q16,
                                          torch::Tensor k, torch::Tensor ks, torch::Tensor kt, torch::Tensor kts,
                                          torch::Tensor k16, torch::Tensor v, torch::Tensor vs, torch::Tensor g,
                                          torch::Tensor gs, torch::Tensor g16, torch::Tensor lse,
                                          torch::Tensor delta, int64_t N, bool causal, double scale,
                                          c10::ScalarType dtype, const std::vector<int64_t>& dq_stride,
                                          const std::vector<int64_t>& dk_stride) {
  const c10::cuda::CUDAGuard guard(q.device());
  const int64_t B = q.size(0), H = q.size(1), Np = q.size(2), D = q.size(3), Hk = k.size(1);
  TORCH_CHECK(Np == padded(N) && N > 0, "N does not match the quantized tensors");
  TORCH_CHECK(Hk > 0 && H % Hk == 0, "query heads must be a multiple of key/value heads");
  check_q8(q, B, H, Np, D, "q");
  check_s32(qs, B, H, Np, "q scales");

  check_q8(k, B, Hk, Np, D, "k");
  check_s32(ks, B, Hk, Np, "k scales");
  auto check_h = [&](const torch::Tensor& t, int64_t heads, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kHalf && t.is_contiguous() && t.dim() == 4 && t.size(0) == B &&
                    t.size(1) == heads && t.size(2) == Np && t.size(3) == D,
                name, ": expected a contiguous fp16 [B, H, Np, D] tensor");
  };
  check_h(q16, H, "q (fp16 codes)");
  check_h(k16, Hk, "k (fp16 codes)");
  check_h(g16, H, "dO (fp16)");
  check_q8(v, B, Hk, Np, D, "v");
  check_s32(vs, B, Hk, Np, "v scales");
  check_q8(g, B, H, Np, D, "dO");
  check_s32(gs, B, H, Np, "dO scales");

  for (auto* t : {&lse, &delta})
    TORCH_CHECK(t->scalar_type() == at::kFloat && t->is_contiguous() && t->numel() == B * H * Np,
                "lse / delta must be contiguous fp32 [B, H, Np]");
  check_q8(kt, B, Hk, D, Np, "k (col)");
  check_q8(kts, B, Hk, Np / 64, 2 * D, "k (col) scales");
  TORCH_CHECK(B * H < 65536, "batch * heads must be < 65536");
  // dq ([B, H, N, D]) and dk, dv ([B, Hk, N, D]) with the given strides (the inputs' layouts: their gradients
  // then need no copy) if those describe a dense layout with a unit stride along D, else [B, N, H(k), D] memory
  auto opts = q.options().dtype(dtype);
  auto alloc = [&](const std::vector<int64_t>& st, int64_t heads) {
    if (st.size() == 4 && st[3] == 1) {
      auto t = torch::empty_strided({B, heads, N, D}, st, opts);
      if (t.is_non_overlapping_and_dense()) return t;
    }
    return torch::empty({B, N, heads, D}, opts).transpose(1, 2);
  };
  auto dq = alloc(dq_stride, H), dk = alloc(dk_stride, Hk), dv = alloc(dk_stride, Hk);
  for (auto* t : {&dq, &dk, &dv})
    TORCH_CHECK(t->stride(3) == 1 && t->stride(0) % 2 == 0 && t->stride(1) % 2 == 0 && t->stride(2) % 2 == 0,
                "gradient layout");
  BwdArgs p{};
  p.q = q.data_ptr<uint8_t>(), p.qs = qs.data_ptr<float>();
  p.q16 = reinterpret_cast<const __half*>(q16.data_ptr());
  p.k = k.data_ptr<uint8_t>(), p.ks = ks.data_ptr<float>();
  p.k16 = reinterpret_cast<const __half*>(k16.data_ptr());
  p.v = v.data_ptr<uint8_t>(), p.vs = vs.data_ptr<float>();
  p.g = g.data_ptr<uint8_t>(), p.gs = gs.data_ptr<float>();
  p.g16 = reinterpret_cast<const __half*>(g16.data_ptr());
  p.lse = lse.data_ptr<float>(), p.delta = delta.data_ptr<float>();
  auto delta2 = torch::zeros_like(delta);  // padding rows stay 0 (their lse is +inf: P = 0)
  p.delta2 = delta2.data_ptr<float>();
  p.kt = kt.data_ptr<uint8_t>(), p.kts = kts.data_ptr<uint8_t>();
  p.dq = dq.data_ptr(), p.dk = dk.data_ptr(), p.dv = dv.data_ptr();
  p.dq_sb = dq.stride(0), p.dq_sh = dq.stride(1), p.dq_sn = dq.stride(2);
  p.dk_sb = dk.stride(0), p.dk_sh = dk.stride(1), p.dk_sn = dk.stride(2);
  TORCH_CHECK(dv.strides() == dk.strides(), "dk / dv layouts");
  p.H = static_cast<int>(H), p.Hk = static_cast<int>(Hk), p.N = static_cast<int>(N), p.Np = static_cast<int>(Np);
  p.causal = causal ? 1 : 0;
  p.scale = static_cast<float>(scale);
  p.scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (variant(dtype, D)) {
    case 0: launch_bwd<__nv_bfloat16, 64>(p, Np, B * H, B * Hk, stream); break;
    case 1: launch_bwd<__nv_bfloat16, 128>(p, Np, B * H, B * Hk, stream); break;
    case 2: launch_bwd<__half, 64>(p, Np, B * H, B * Hk, stream); break;
    default: launch_bwd<__half, 128>(p, Np, B * H, B * Hk, stream); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dq, dk, dv};
}
