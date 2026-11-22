// BF16/FP32 -> NVFP4 quantization kernels (see nvfp4/reference.py for the exact recipe).
//
// Logical output tensor Y has shape [R_out, C_out] and is quantized along its last dim:
//   transpose == false:  Y = X           (X is [R, C],  R_out = R, C_out = C)
//   transpose == true :  Y = X^T         (X is [R, C],  R_out = C, C_out = R)
// Optional per-block random Hadamard transform (applied to Y's 16-element blocks before scaling),
// optional 16x16 2D block scales, optional stochastic rounding.
//
// Thread mapping: one thread owns one 16-element block of one logical row. A warp covers 32
// consecutive logical rows of the same block column, so
//   * transpose: lanes read 32 consecutive input columns -> coalesced,
//   * 2D blocks: lanes 0-15 / 16-31 form two 16x16 tiles -> reduced with 4 shuffles.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include "nvfp4_common.cuh"

namespace nvfp4 {

template <typename T>
__device__ __forceinline__ float to_f(T v);
template <>
__device__ __forceinline__ float to_f<float>(float v) { return v; }
template <>
__device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

struct QParams {
  int R, C;          // input shape
  int R_out, C_out;  // logical output shape
  bool transpose;
  bool use_rht;
  bool lanes_along_cols;  // lanes = consecutive blocks of one row (rowwise 1D); else consecutive rows
  uint64_t rht_seed;
};

// Thread -> (logical row, block column) mapping shared by both passes.
//  * lanes_along_cols: a warp covers 32 consecutive 16-element blocks of one row (1 KB of bf16,
//    fully coalesced 16-byte vector loads). Used for rowwise 1D quantization.
//  * otherwise: a warp covers 32 consecutive rows of one block column. Coalesced for the transposed
//    read, and puts each 16x16 tile in 16 adjacent lanes for the 2D-scale reduction.
__host__ __device__ __forceinline__ int64_t num_warps(const QParams& p) {
  const int nbc = p.C_out / kBlock;
  return p.lanes_along_cols ? static_cast<int64_t>(p.R_out) * ((nbc + 31) / 32)
                            : static_cast<int64_t>((p.R_out + 31) / 32) * nbc;
}
__device__ __forceinline__ bool block_coords(const QParams& p, int64_t warp, int lane, int& orow, int& bcol) {
  const int nbc = p.C_out / kBlock;
  if (p.lanes_along_cols) {
    const int col_groups = (nbc + 31) / 32;
    orow = static_cast<int>(warp / col_groups);
    bcol = static_cast<int>(warp % col_groups) * 32 + lane;
  } else {
    const int row_groups = (p.R_out + 31) / 32;
    bcol = static_cast<int>(warp / row_groups);
    orow = static_cast<int>(warp % row_groups) * 32 + lane;
  }
  return orow < p.R_out && bcol < nbc;
}

__device__ __forceinline__ void unpack8(const uint4& u, float* v, const __nv_bfloat16*) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    v[2 * i] = f.x;
    v[2 * i + 1] = f.y;
  }
}

template <typename T>
__device__ __forceinline__ void load_block(const T* __restrict__ x, const QParams& p, int orow, int bcol,
                                           float (&v)[kBlock]) {
  if (!p.transpose) {
    const T* src = x + static_cast<int64_t>(orow) * p.C + bcol * kBlock;  // 32/64-byte aligned
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      const uint4* s4 = reinterpret_cast<const uint4*>(src);
      unpack8(__ldg(s4), v, src);
      unpack8(__ldg(s4 + 1), v + 8, src);
    } else {
      const float4* s4 = reinterpret_cast<const float4*>(src);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float4 f = __ldg(s4 + i);
        v[4 * i] = f.x;
        v[4 * i + 1] = f.y;
        v[4 * i + 2] = f.z;
        v[4 * i + 3] = f.w;
      }
    }
  } else {
    // logical row orow = input column, logical cols bcol*16.. = input rows
    const T* src = x + static_cast<int64_t>(bcol * kBlock) * p.C + orow;
#pragma unroll
    for (int i = 0; i < kBlock; ++i) v[i] = to_f(src[static_cast<int64_t>(i) * p.C]);
  }
  if (p.use_rht) rht16(v, p.rht_seed);
}

// Pass 1: amax over the (optionally transformed) logical tensor. amax_bits holds a non-negative
// float's bit pattern (integer order == float order for non-negative floats).
template <typename T>
__global__ void amax_kernel(const T* __restrict__ x, QParams p, unsigned int* __restrict__ amax_bits) {
  const int64_t nthreads = num_warps(p) * 32;
  float m = 0.0f;
  for (int64_t t = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; t < nthreads;
       t += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int orow, bcol;
    if (!block_coords(p, t / 32, static_cast<int>(t % 32), orow, bcol)) continue;
    float v[kBlock];
    load_block(x, p, orow, bcol, v);
#pragma unroll
    for (int i = 0; i < kBlock; ++i) m = fmaxf(m, fabsf(v[i]));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) atomicMax(amax_bits, __float_as_uint(m));
}

template <typename T, bool kBlock2D, bool kStochastic>
__global__ void quantize_kernel(const T* __restrict__ x, QParams p, const unsigned int* __restrict__ amax_bits,
                                uint8_t* __restrict__ data, uint8_t* __restrict__ scales, float* __restrict__ gdec_out,
                                uint64_t sr_seed) {
  const int nbc = p.C_out / kBlock;
  const int64_t warp = (blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x) / 32;
  const int lane = threadIdx.x & 31;
  if (warp >= num_warps(p)) return;  // whole warp exits together
  int orow, bcol;
  const bool valid = block_coords(p, warp, lane, orow, bcol);

  // global scales: genc = 2688 / amax (1 if amax == 0), gdec = 1 / genc
  const float amax = __uint_as_float(*amax_bits);
  const float genc = amax > 0.0f ? __fdiv_rn(kE4M3Max * kE2M1Max, amax) : 1.0f;
  if (blockIdx.x == 0 && threadIdx.x == 0) *gdec_out = __fdiv_rn(1.0f, genc);

  float v[kBlock];
  float bmax = 0.0f;
  if (valid) {
    load_block(x, p, orow, bcol, v);
#pragma unroll
    for (int i = 0; i < kBlock; ++i) bmax = fmaxf(bmax, fabsf(v[i]));
  }
  if (kBlock2D) {  // max over the 16 rows of this 16x16 tile (lanes 0-15 or 16-31)
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) bmax = fmaxf(bmax, __shfl_xor_sync(0xffffffffu, bmax, o));
  }
  if (!valid) return;

  const float s_target = fminf(__fmul_rn(__fdiv_rn(bmax, kE2M1Max), genc), kE4M3Max);
  const uint8_t s = e4m3_from_float(s_target);
  const float sf = e4m3_to_float(s);
  const float enc = sf > 0.0f ? __fdiv_rn(genc, sf) : 0.0f;

  uint32_t lo_word = 0, hi_word = 0;
#pragma unroll
  for (int i = 0; i < kBlock; ++i) {
    const float y = __fmul_rn(v[i], enc);
    uint32_t code;
    if (kStochastic) {
      const uint64_t lin = static_cast<uint64_t>(orow) * p.C_out + bcol * kBlock + i;
      code = e2m1_round_stochastic(y, sr_uniform(sr_seed, lin));
    } else {
      code = e2m1_round_rne(y);
    }
    if (i < 8)
      lo_word |= code << (4 * i);
    else
      hi_word |= code << (4 * (i - 8));
  }
  uint2* dst = reinterpret_cast<uint2*>(data + static_cast<int64_t>(orow) * (p.C_out / 2) + bcol * (kBlock / 2));
  *dst = make_uint2(lo_word, hi_word);
  scales[static_cast<int64_t>(orow) * nbc + bcol] = s;
}

}  // namespace nvfp4

// Returns {data uint8 [R_out, C_out/2], scales uint8 [R_out, C_out/16], gdec fp32 [1]}.
std::vector<torch::Tensor> nvfp4_quantize(torch::Tensor x, bool transpose, bool block_2d, bool stochastic,
                                          int64_t sr_seed, bool use_rht, int64_t rht_seed) {
  using namespace nvfp4;
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x must be a contiguous 2D CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat, "x must be bf16 or fp32");
  const c10::cuda::CUDAGuard guard(x.device());
  QParams p;
  p.R = static_cast<int>(x.size(0));
  p.C = static_cast<int>(x.size(1));
  p.transpose = transpose;
  p.R_out = transpose ? p.C : p.R;
  p.C_out = transpose ? p.R : p.C;
  p.use_rht = use_rht;
  p.rht_seed = static_cast<uint64_t>(rht_seed);
  p.lanes_along_cols = !transpose && !block_2d;
  TORCH_CHECK(p.C_out % kBlock == 0, "quantized dim (", p.C_out, ") must be a multiple of 16");
  TORCH_CHECK(!block_2d || p.R_out % kBlock == 0, "2D blocks need the other dim (", p.R_out, ") % 16 == 0");

  auto opts_u8 = x.options().dtype(at::kByte);
  auto data = torch::empty({p.R_out, p.C_out / 2}, opts_u8);
  auto scales = torch::empty({p.R_out, p.C_out / kBlock}, opts_u8);
  auto gdec = torch::empty({1}, x.options().dtype(at::kFloat));
  auto amax = torch::zeros({1}, x.options().dtype(at::kInt));
  if (x.numel() == 0) return {data, scales, gdec.fill_(1.0f)};

  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t nthreads = num_warps(p) * 32;
  const int grid = static_cast<int>((nthreads + threads - 1) / threads);
  const int amax_grid = std::min(grid, 4096);
  auto* amax_ptr = reinterpret_cast<unsigned int*>(amax.data_ptr<int>());

#define NVFP4_LAUNCH(T, B2D, SR)                                                                           \
  quantize_kernel<T, B2D, SR><<<grid, threads, 0, stream>>>(xp, p, amax_ptr, data.data_ptr<uint8_t>(),   \
                                                            scales.data_ptr<uint8_t>(), gdec.data_ptr<float>(), \
                                                            static_cast<uint64_t>(sr_seed))
#define NVFP4_DISPATCH(T)                                                                    \
  {                                                                                          \
    const T* xp = reinterpret_cast<const T*>(x.data_ptr());                                  \
    amax_kernel<T><<<amax_grid, threads, 0, stream>>>(xp, p, amax_ptr);                      \
    if (block_2d && stochastic) NVFP4_LAUNCH(T, true, true);                                 \
    else if (block_2d) NVFP4_LAUNCH(T, true, false);                                         \
    else if (stochastic) NVFP4_LAUNCH(T, false, true);                                       \
    else NVFP4_LAUNCH(T, false, false);                                                      \
  }
  if (x.scalar_type() == at::kBFloat16) {
    NVFP4_DISPATCH(__nv_bfloat16);
  } else {
    NVFP4_DISPATCH(float);
  }
#undef NVFP4_DISPATCH
#undef NVFP4_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {data, scales, gdec};
}
