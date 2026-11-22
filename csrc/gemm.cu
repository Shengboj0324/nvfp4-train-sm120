// NVFP4 x NVFP4 -> BF16/FP32 GEMM for consumer Blackwell (sm_120a), "TN" layout:
//   D[M, N] = alpha * sum_k A[m, k] * B[n, k],   alpha = gdec_A * gdec_B
// with A, B stored as NVFP4 (E2M1 codes packed 2/byte along K, one UE4M3 scale per 16 K-elements).
//
// Hardware path: warp-level block-scaled MMA
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
// (sm_120 has no wgmma / tcgen05; this is its 5th-gen tensor-core interface). Fragment layouts follow
// CUTLASS cute::MMA_Traits<SM120_16x8x64_TN_VS> and are verified by tests/test_gemm_cuda.py.
//
// Tiling: CTA 128x128x128, 8 warps (2 along M x 4 along N, 64x32 per warp), 4-stage cp.async
// pipeline, 18 KB per stage = 72 KB shared memory (sm_120 allows ~99 KB per block).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>

namespace nvfp4 {
namespace gemm {

constexpr int BM = 128, BN = 128, BK = 128;  // BK in elements (64 bytes per row)
constexpr int BKB = BK / 2;                  // bytes per row per k-tile
constexpr int SFK = BK / 16;                 // scale bytes per row per k-tile (8)
#ifndef NVFP4_GEMM_STAGES
#define NVFP4_GEMM_STAGES 4
#endif
#ifndef NVFP4_GEMM_MIN_BLOCKS
#define NVFP4_GEMM_MIN_BLOCKS 1
#endif
constexpr int STAGES = NVFP4_GEMM_STAGES;
constexpr int WARPS_M = 2, WARPS_N = 4;
constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;  // 64 x 32
constexpr int MI = WM / 16, NI = WN / 8;             // 4 x 4 MMA tiles per warp
constexpr int THREADS = 32 * WARPS_M * WARPS_N;      // 256

constexpr int A_STAGE = BM * BKB;   // 8192
constexpr int B_STAGE = BN * BKB;   // 8192
constexpr int SFA_STAGE = BM * SFK; // 1024
constexpr int SFB_STAGE = BN * SFK; // 1024
constexpr int STAGE_BYTES = A_STAGE + B_STAGE + SFA_STAGE + SFB_STAGE;
constexpr int SMEM_BYTES = STAGES * STAGE_BYTES;  // 73728

// 64-byte rows = four 16-byte chunks; XOR swizzle makes 8-row ldmatrix reads bank-conflict free.
__device__ __forceinline__ int swz(int row, int chunk) { return row * BKB + ((chunk ^ ((row >> 1) & 3)) << 4); }

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  const int n = pred ? 16 : 0;  // src-size 0 => zero fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async8(uint32_t dst, const void* src, bool pred) {
  const int n = pred ? 8 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

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

struct Args {
  const uint8_t* a;    // [M, K/2]
  const uint8_t* b;    // [N, K/2]
  const uint8_t* sfa;  // [M, K/16]
  const uint8_t* sfb;  // [N, K/16]
  const float* gdec_a;
  const float* gdec_b;
  void* d;  // [M, N] (ldd = N)
  int M, N, K;
};

__device__ __forceinline__ void load_stage(const Args& p, uint8_t* smem, int stage, int kt, int m0, int n0) {
  uint8_t* sa = smem + stage * STAGE_BYTES;
  uint8_t* sb = sa + A_STAGE;
  uint8_t* ssfa = sb + B_STAGE;
  uint8_t* ssfb = ssfa + SFA_STAGE;
  const int tid = threadIdx.x;
  const int64_t ldab = p.K / 2;
  const int64_t ldsf = p.K / 16;
#pragma unroll
  for (int i = 0; i < 2; ++i) {  // 512 16-byte chunks per operand tile, 2 per thread
    const int c = tid + i * THREADS;
    const int row = c >> 2, chunk = c & 3;
    {
      const int gm = m0 + row;
      const bool ok = gm < p.M;
      const uint8_t* src = p.a + (ok ? gm : 0) * ldab + kt * BKB + chunk * 16;
      cp_async16(smem_u32(sa + swz(row, chunk)), src, ok);
    }
    {
      const int gn = n0 + row;
      const bool ok = gn < p.N;
      const uint8_t* src = p.b + (ok ? gn : 0) * ldab + kt * BKB + chunk * 16;
      cp_async16(smem_u32(sb + swz(row, chunk)), src, ok);
    }
  }
  if (tid < BM) {
    const int gm = m0 + tid;
    const bool ok = gm < p.M;
    cp_async8(smem_u32(ssfa + tid * SFK), p.sfa + (ok ? gm : 0) * ldsf + kt * SFK, ok);
  } else {
    const int r = tid - BM;
    const int gn = n0 + r;
    const bool ok = gn < p.N;
    cp_async8(smem_u32(ssfb + r * SFK), p.sfb + (ok ? gn : 0) * ldsf + kt * SFK, ok);
  }
}

// kAccum (fp32 output only): D += alpha * A @ B^T, e.g. accumulating a weight gradient in place.
template <typename OutT, bool kAccum>
__global__ void __launch_bounds__(THREADS, NVFP4_GEMM_MIN_BLOCKS) nvfp4_gemm_kernel(Args p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int KT = p.K / BK;

  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.0f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) load_stage(p, smem, s, s, m0, n0);
    cp_async_commit();
  }

  const int g = lane >> 2;  // groupID
  for (int kt = 0; kt < KT; ++kt) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    {
      const int nk = kt + STAGES - 1;
      if (nk < KT) load_stage(p, smem, nk % STAGES, nk, m0, n0);
      cp_async_commit();
    }
    const uint8_t* sa = smem + (kt % STAGES) * STAGE_BYTES;
    const uint8_t* sb = sa + A_STAGE;
    const uint8_t* ssfa = sb + B_STAGE;
    const uint8_t* ssfb = ssfa + SFA_STAGE;

#pragma unroll
    for (int ks = 0; ks < BK / 64; ++ks) {
      uint32_t af[MI][4], bf[NI][2], sfa[MI], sfb[NI];
#pragma unroll
      for (int i = 0; i < MI; ++i) {
        const int row = wm * WM + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int chunk = 2 * ks + (lane >> 4);
        ldmatrix_x4(smem_u32(sa + swz(row, chunk)), af[i][0], af[i][1], af[i][2], af[i][3]);
        // scale for row g + 8*(lane&1) of this m16 tile; byte j = K-block j of this 64-wide step
        sfa[i] = *reinterpret_cast<const uint32_t*>(ssfa + (wm * WM + i * 16 + g + 8 * (lane & 1)) * SFK + 4 * ks);
      }
#pragma unroll
      for (int j = 0; j < NI; j += 2) {
        const int row = wn * WN + j * 8 + (lane & 7) + (lane >> 4) * 8;
        const int chunk = 2 * ks + ((lane >> 3) & 1);
        ldmatrix_x4(smem_u32(sb + swz(row, chunk)), bf[j][0], bf[j][1], bf[j + 1][0], bf[j + 1][1]);
      }
#pragma unroll
      for (int j = 0; j < NI; ++j)
        sfb[j] = *reinterpret_cast<const uint32_t*>(ssfb + (wn * WN + j * 8 + g) * SFK + 4 * ks);
#pragma unroll
      for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j) mma_nvfp4(acc[i][j], af[i], bf[j], sfa[i], sfb[j]);
    }
  }
  cp_async_wait<0>();

  float alpha = (*p.gdec_a) * (*p.gdec_b);
  const int tig = lane & 3;
  if constexpr (kAccum) {
    // acc <- D_old + alpha * acc. Loads for one m16 row group are issued together (a load cannot be
    // moved past a store to the same array, so a load-add-store loop would serialize their latency).
#pragma unroll
    for (int i = 0; i < MI; ++i) {
      float2 old[NI][2];
#pragma unroll
      for (int j = 0; j < NI; ++j) {
        const int col = n0 + wn * WN + j * 8 + 2 * tig;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = m0 + wm * WM + i * 16 + g + 8 * h;
          const float* src = reinterpret_cast<const float*>(p.d) + static_cast<int64_t>(row) * p.N + col;
          old[j][h] = make_float2(0.0f, 0.0f);
          if (row < p.M) {
            if (col + 1 < p.N && (p.N % 2 == 0)) {
              old[j][h] = *reinterpret_cast<const float2*>(src);
            } else {
              if (col < p.N) old[j][h].x = src[0];
              if (col + 1 < p.N) old[j][h].y = src[1];
            }
          }
        }
      }
#pragma unroll
      for (int j = 0; j < NI; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          // separate rn ops (no FMA contraction): bit-identical to autograd's grad + (alpha * acc)
          acc[i][j][2 * h] = __fadd_rn(old[j][h].x, __fmul_rn(acc[i][j][2 * h], alpha));
          acc[i][j][2 * h + 1] = __fadd_rn(old[j][h].y, __fmul_rn(acc[i][j][2 * h + 1], alpha));
        }
    }
    alpha = 1.0f;
  }
#pragma unroll
  for (int i = 0; i < MI; ++i) {
#pragma unroll
    for (int j = 0; j < NI; ++j) {
      const int col = n0 + wn * WN + j * 8 + 2 * tig;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = m0 + wm * WM + i * 16 + g + 8 * h;
        if (row >= p.M) continue;
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
      }
    }
  }
}

// One static per instantiation: each kernel needs its own max-dynamic-smem attribute.
template <typename OutT, bool kAccum>
void launch_gemm(const Args& p, dim3 grid, cudaStream_t stream) {
  static const bool attr = (cudaFuncSetAttribute(nvfp4_gemm_kernel<OutT, kAccum>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES),
                            true);
  (void)attr;
  nvfp4_gemm_kernel<OutT, kAccum><<<grid, THREADS, SMEM_BYTES, stream>>>(p);
}

}  // namespace gemm
}  // namespace nvfp4

// D = A @ B^T (into a new tensor, or into `out`); with accumulate, out += A @ B^T (fp32 only).
torch::Tensor nvfp4_gemm(torch::Tensor a, torch::Tensor sfa, torch::Tensor gdec_a, torch::Tensor b, torch::Tensor sfb,
                         torch::Tensor gdec_b, c10::ScalarType out_dtype, c10::optional<torch::Tensor> out,
                         bool accumulate) {
  using namespace nvfp4::gemm;
  for (auto* t : {&a, &sfa, &b, &sfb})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kByte && t->dim() == 2 && t->is_contiguous(),
                "NVFP4 operands must be contiguous 2D uint8 CUDA tensors");
  TORCH_CHECK(gdec_a.scalar_type() == at::kFloat && gdec_b.scalar_type() == at::kFloat, "gdec must be fp32");
  const int M = static_cast<int>(a.size(0)), N = static_cast<int>(b.size(0));
  const int K = static_cast<int>(a.size(1) * 2);
  TORCH_CHECK(b.size(1) * 2 == K, "A and B must have the same K");
  TORCH_CHECK(K % BK == 0, "K (", K, ") must be a multiple of ", BK);
  TORCH_CHECK(sfa.size(0) == M && sfa.size(1) == K / 16 && sfb.size(0) == N && sfb.size(1) == K / 16,
              "scale tensors must be [rows, K/16]");
  TORCH_CHECK(out_dtype == at::kBFloat16 || out_dtype == at::kFloat, "out_dtype must be bf16 or fp32");
  TORCH_CHECK(!accumulate || (out.has_value() && out_dtype == at::kFloat), "accumulate needs an fp32 `out`");
  const c10::cuda::CUDAGuard guard(a.device());
  torch::Tensor d;
  if (out.has_value()) {
    d = *out;
    TORCH_CHECK(d.is_cuda() && d.device() == a.device() && d.scalar_type() == out_dtype && d.is_contiguous() &&
                    d.dim() == 2 && d.size(0) == M && d.size(1) == N,
                "out must be a contiguous [M, N] tensor of out_dtype on the same device");
  } else {
    d = torch::empty({M, N}, a.options().dtype(out_dtype));
  }
  if (M == 0 || N == 0) return d;
  Args p{a.data_ptr<uint8_t>(), b.data_ptr<uint8_t>(), sfa.data_ptr<uint8_t>(), sfb.data_ptr<uint8_t>(),
         gdec_a.data_ptr<float>(), gdec_b.data_ptr<float>(), d.data_ptr(), M, N, K};
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (out_dtype == at::kFloat) {
    if (accumulate)
      launch_gemm<float, true>(p, grid, stream);
    else
      launch_gemm<float, false>(p, grid, stream);
  } else {
    launch_gemm<__nv_bfloat16, false>(p, grid, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return d;
}
