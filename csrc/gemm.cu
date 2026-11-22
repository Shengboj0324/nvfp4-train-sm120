// NVFP4 x NVFP4 -> BF16/FP32 GEMM for consumer Blackwell (sm_120a), "TN" layout:
//   D[M, N] = alpha * sum_k A[m, k] * B[n, k],   alpha = gdec_A * gdec_B
// with A, B stored as NVFP4 (E2M1 codes packed 2/byte along K, one UE4M3 scale per 16 K-elements).
//
// Hardware path: warp-level block-scaled MMA
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
// (sm_120 has no wgmma / tcgen05; this is its 5th-gen tensor-core interface).
//
// Two main loops share one epilogue (gemm_common.cuh):
//   - gemm_tma.cu: TMA producer warp + 8 MMA warps, 128x128x256 tiles, 2 stages (default when K % 256 == 0)
//   - this file:   all-threads cp.async loop, 128x128x128 tiles, 4 stages, 72 KB (fallback for any K % 128 == 0)
// Small output grids are split along K (deterministic in-kernel reduction) to fill the GPU.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>

#include "gemm_common.cuh"

namespace nvfp4 {
namespace gemm {

bool launch_tma(const Args& p, bool out_fp32, bool accumulate, dim3 grid, cudaStream_t stream);  // gemm_tma.cu
bool tma_supported(const Args& p);
int tma_k_tile();

namespace legacy {

constexpr int BK = 128;       // K elements per stage (64 bytes per row)
constexpr int BKB = BK / 2;
constexpr int SFK = BK / 16;  // 8 scale bytes per row
#ifndef NVFP4_GEMM_STAGES
#define NVFP4_GEMM_STAGES 4
#endif
constexpr int STAGES = NVFP4_GEMM_STAGES;
constexpr int THREADS = MMA_THREADS;

constexpr int A_STAGE = BM * BKB;   // 8192
constexpr int B_STAGE = BN * BKB;   // 8192
constexpr int SFA_STAGE = BM * SFK; // 1024
constexpr int SFB_STAGE = BN * SFK; // 1024
constexpr int STAGE_BYTES = A_STAGE + B_STAGE + SFA_STAGE + SFB_STAGE;
constexpr int SMEM_BYTES = STAGES * STAGE_BYTES + 16;  // + split-K flag

// 64-byte rows = four 16-byte chunks; XOR swizzle makes 8-row ldmatrix reads bank-conflict free.
__device__ __forceinline__ int swz(int row, int chunk) { return row * BKB + ((chunk ^ ((row >> 1) & 3)) << 4); }

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

template <typename OutT, bool kAccum>
__global__ void __launch_bounds__(THREADS, 1) nvfp4_gemm_kernel(Args p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int KT = p.K / BK;
  const int kt0 = static_cast<int>((static_cast<int64_t>(KT) * blockIdx.z) / p.splits);
  const int kt1 = static_cast<int>((static_cast<int64_t>(KT) * (blockIdx.z + 1)) / p.splits);
  const int nk = kt1 - kt0;

  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.0f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < nk) load_stage(p, smem, s, kt0 + s, m0, n0);
    cp_async_commit();
  }

  const int g = lane >> 2;  // groupID
  for (int it = 0; it < nk; ++it) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    {
      const int nit = it + STAGES - 1;
      if (nit < nk) load_stage(p, smem, nit % STAGES, kt0 + nit, m0, n0);
      cp_async_commit();
    }
    const uint8_t* sa = smem + (it % STAGES) * STAGE_BYTES;
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
  finish_tile<OutT, kAccum>(p, acc, m0, n0, wm, wn, lane, reinterpret_cast<int*>(smem + STAGES * STAGE_BYTES));
}

// One static per instantiation: each kernel needs its own max-dynamic-smem attribute.
template <typename OutT, bool kAccum>
void launch(const Args& p, dim3 grid, cudaStream_t stream) {
  static const bool attr = (cudaFuncSetAttribute(nvfp4_gemm_kernel<OutT, kAccum>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES),
                            true);
  (void)attr;
  nvfp4_gemm_kernel<OutT, kAccum><<<grid, THREADS, SMEM_BYTES, stream>>>(p);
}

void launch_any(const Args& p, bool out_fp32, bool accumulate, dim3 grid, cudaStream_t stream) {
  if (out_fp32) {
    if (accumulate)
      launch<float, true>(p, grid, stream);
    else
      launch<float, false>(p, grid, stream);
  } else {
    launch<__nv_bfloat16, false>(p, grid, stream);
  }
}

}  // namespace legacy

// Number of K splits for a grid of `tiles` CTAs with KT k-tiles each, on `sms` SMs at 1 CTA/SM.
// Split only grids that leave more than half the SMs idle, into as many splits as still fit in one
// wave (each keeping >= 8 k-tiles). Splitting into extra waves never paid off on the power-limited
// laptop GPU: a partly idle GPU already runs its busy SMs at higher clocks (docs/RESULTS.md §1).
int choose_splits(int tiles, int KT, int sms) {
  if (2 * tiles > sms) return 1;
  return std::max(1, std::min({sms / tiles, KT / 8, kMaxSplits}));
}

}  // namespace gemm
}  // namespace nvfp4

// D = A @ B^T (into a new tensor, or into `out`); with accumulate, out += A @ B^T (fp32 only).
// impl: 0 = auto, 1 = cp.async loop, 2 = TMA loop (error if unsupported). splits: 0 = auto.
torch::Tensor nvfp4_gemm(torch::Tensor a, torch::Tensor sfa, torch::Tensor gdec_a, torch::Tensor b, torch::Tensor sfb,
                         torch::Tensor gdec_b, c10::ScalarType out_dtype, c10::optional<torch::Tensor> out,
                         bool accumulate, int64_t impl, int64_t splits) {
  using namespace nvfp4::gemm;
  for (auto* t : {&a, &sfa, &b, &sfb})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kByte && t->dim() == 2 && t->is_contiguous(),
                "NVFP4 operands must be contiguous 2D uint8 CUDA tensors");
  TORCH_CHECK(gdec_a.scalar_type() == at::kFloat && gdec_b.scalar_type() == at::kFloat, "gdec must be fp32");
  const int M = static_cast<int>(a.size(0)), N = static_cast<int>(b.size(0));
  const int K = static_cast<int>(a.size(1) * 2);
  TORCH_CHECK(b.size(1) * 2 == K, "A and B must have the same K");
  TORCH_CHECK(K % legacy::BK == 0, "K (", K, ") must be a multiple of ", legacy::BK);
  TORCH_CHECK(sfa.size(0) == M && sfa.size(1) == K / 16 && sfb.size(0) == N && sfb.size(1) == K / 16,
              "scale tensors must be [rows, K/16]");
  TORCH_CHECK(out_dtype == at::kBFloat16 || out_dtype == at::kFloat, "out_dtype must be bf16 or fp32");
  TORCH_CHECK(!accumulate || (out.has_value() && out_dtype == at::kFloat), "accumulate needs an fp32 `out`");
  TORCH_CHECK(impl >= 0 && impl <= 2, "impl must be 0 (auto), 1 (cp.async) or 2 (TMA)");
  TORCH_CHECK(splits >= 0 && splits <= kMaxSplits, "splits must be in [0, ", kMaxSplits, "]");
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
         gdec_a.data_ptr<float>(), gdec_b.data_ptr<float>(), d.data_ptr(), nullptr, nullptr, M, N, K, 1};
  // The 2-stage TMA loop needs a few k-tiles to reach steady state; below K = 1024 the 4-stage cp.async
  // loop with 128-deep tiles is as fast (8192x3072x768: 273 vs 268 TFLOPS).
  const bool use_tma = impl == 2 || (impl == 0 && K >= 1024 && tma_supported(p));
  TORCH_CHECK(impl != 2 || tma_supported(p), "TMA GEMM needs K % 256 == 0 and 16-byte aligned operands");
  const int KT = K / (use_tma ? tma_k_tile() : legacy::BK);
  const int tiles_n = (N + BN - 1) / BN, tiles_m = (M + BM - 1) / BM;
  int s = static_cast<int>(splits);
  if (s == 0) s = choose_splits(tiles_m * tiles_n, KT, at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
  s = std::max(1, std::min(s, KT));
  torch::Tensor ws;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (s > 1) {  // one allocation: fp32 partials, then the arrival counters (zeroed here, self-resetting)
    const int64_t plane = static_cast<int64_t>(s) * M * N, tiles = static_cast<int64_t>(tiles_m) * tiles_n;
    ws = torch::empty({plane + tiles}, a.options().dtype(at::kFloat));
    p.ws = ws.data_ptr<float>();
    p.counters = reinterpret_cast<int*>(p.ws + plane);
    p.splits = s;
    C10_CUDA_CHECK(cudaMemsetAsync(p.counters, 0, tiles * sizeof(int), stream));
  }
  const dim3 grid(tiles_n, tiles_m, s);
  const bool fp32 = out_dtype == at::kFloat;
  if (!(use_tma && launch_tma(p, fp32, accumulate, grid, stream))) {
    TORCH_CHECK(impl != 2, "TMA descriptor creation failed");
    legacy::launch_any(p, fp32, accumulate, grid, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return d;
}
