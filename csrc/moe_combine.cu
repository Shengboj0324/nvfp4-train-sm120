// Mixture-of-experts gathers: out[r] = sum_j w[r, j] rows[pos[r, j]] and its backward, without the [R, k, d]
// intermediate of an index_select + batched GEMM (nvfp4/moe.py _combine): one pass over the rows read.
//
//   combine_fwd(rows [M, d], pos int64 [R, k], w fp32 [R, k] or none)  -> out [R, d] (rows' dtype)
//       fp32 accumulation in j order, one rounding; pos outside [0, M) reads nothing (a zero row).
//   combine_bwd_rows(g [R, d], inv int64 [M], w fp32 [R, k] or none, k) -> drows [M, d]
//       drows[i] = w[inv[i]] g[inv[i] / k] (inv[i] < 0: zero), inv[i] = the (r, j) slot that reads row i:
//       the gradient when every row is read by at most one slot (as in the layer; others are zero-weight).
//   combine_bwd_w(g [R, d], rows [M, d], pos [R, k]) -> dw fp32 [R, k]: dw[r, j] = <g[r], rows[pos[r, j]]>
//       (a warp per (r, j), fixed reduction order).
// Every kernel is deterministic. rows / g may be row-strided (stride(1) == 1); 16-byte vectors when aligned.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <type_traits>

#include "nvfp4_common.cuh"

namespace nvfp4 {
namespace combine {

template <typename T>
__device__ __forceinline__ float to_f(T v) {
  if constexpr (std::is_same_v<T, float>) return v;
  else if constexpr (std::is_same_v<T, __nv_bfloat16>) return __bfloat162float(v);
  else return __half2float(v);
}
template <typename T>
__device__ __forceinline__ T from_f(float v) {
  if constexpr (std::is_same_v<T, float>) return v;
  else if constexpr (std::is_same_v<T, __nv_bfloat16>) return __float2bfloat16_rn(v);
  else return __float2half_rn(v);
}

template <typename T, int VEC>
__device__ __forceinline__ void load(const T* p, float (&f)[VEC]) {
  if constexpr (VEC * sizeof(T) == 16) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const T* t = reinterpret_cast<const T*>(&u);
#pragma unroll
    for (int i = 0; i < VEC; ++i) f[i] = to_f(t[i]);
  } else {
#pragma unroll
    for (int i = 0; i < VEC; ++i) f[i] = to_f(p[i]);
  }
}

template <typename T, int VEC>
__device__ __forceinline__ void store(T* p, const float (&f)[VEC]) {
  if constexpr (VEC * sizeof(T) == 16) {
    uint4 u;
    T* t = reinterpret_cast<T*>(&u);
#pragma unroll
    for (int i = 0; i < VEC; ++i) t[i] = from_f<T>(f[i]);
    *reinterpret_cast<uint4*>(p) = u;
  } else {
#pragma unroll
    for (int i = 0; i < VEC; ++i) p[i] = from_f<T>(f[i]);
  }
}

// grid (R, ceil(d / (VEC * blockDim))): rows on x (no 65535 limit)
template <typename T, int VEC>
__global__ void __launch_bounds__(128) combine_fwd_kernel(const T* __restrict__ rows, int64_t ld_rows, int64_t M,
                                                          const int64_t* __restrict__ pos, const float* __restrict__ w,
                                                          int k, T* __restrict__ out, int64_t d) {
  const int64_t r = blockIdx.x;
  const int64_t c = (static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x) * VEC;
  if (c >= d) return;
  float acc[VEC];
#pragma unroll
  for (int i = 0; i < VEC; ++i) acc[i] = 0.0f;
  for (int j = 0; j < k; ++j) {
    const int64_t p = pos[r * k + j];
    if (p < 0 || p >= M) continue;
    const float wj = w ? w[r * k + j] : 1.0f;
    float x[VEC];
    load<T, VEC>(rows + p * ld_rows + c, x);
#pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = __fadd_rn(acc[i], __fmul_rn(wj, x[i]));
  }
  store<T, VEC>(out + r * d + c, acc);
}

template <typename T, int VEC>
__global__ void __launch_bounds__(128) combine_bwd_rows_kernel(const T* __restrict__ g, int64_t ld_g,
                                                               const int64_t* __restrict__ inv,
                                                               const float* __restrict__ w, int k,
                                                               T* __restrict__ drows, int64_t d) {
  const int64_t i = blockIdx.x;
  const int64_t c = (static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x) * VEC;
  if (c >= d) return;
  const int64_t s = inv[i];
  float v[VEC];
  if (s < 0) {
#pragma unroll
    for (int e = 0; e < VEC; ++e) v[e] = 0.0f;
  } else {
    const float ws = w ? w[s] : 1.0f;
    load<T, VEC>(g + (s / k) * ld_g + c, v);
#pragma unroll
    for (int e = 0; e < VEC; ++e) v[e] = __fmul_rn(ws, v[e]);
  }
  store<T, VEC>(drows + i * d + c, v);
}

// block: k warps (one per choice) of row r
template <typename T>
__global__ void combine_bwd_w_kernel(const T* __restrict__ g, int64_t ld_g, const T* __restrict__ rows,
                                     int64_t ld_rows, int64_t M, const int64_t* __restrict__ pos, int k,
                                     float* __restrict__ dw, int64_t d) {
  const int64_t r = blockIdx.x;
  const int j = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (j >= k) return;
  const int64_t p = pos[r * k + j];
  float acc = 0.0f;
  if (p >= 0 && p < M) {
    for (int64_t c = lane; c < d; c += 32)
      acc = __fadd_rn(acc, __fmul_rn(to_f(g[r * ld_g + c]), to_f(rows[p * ld_rows + c])));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
  if (lane == 0) dw[r * k + j] = acc;
}

template <typename T>
int vec_for(const T* p, int64_t ld, int64_t d, const T* q = nullptr, int64_t ld_q = 0) {
  constexpr int V = 16 / sizeof(T);
  auto ok = [&](const T* x, int64_t l) {
    return x == nullptr || (reinterpret_cast<uintptr_t>(x) % 16 == 0 && (l * sizeof(T)) % 16 == 0);
  };
  return (d % V == 0 && ok(p, ld) && ok(q, ld_q)) ? V : 1;
}

void check_2d(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.dim() == 2 && t.stride(1) == 1, name, " must be a 2D CUDA tensor with unit column stride");
  TORCH_CHECK(t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kFloat || t.scalar_type() == at::kHalf,
              name, " must be bf16, fp16 or fp32");
}

const float* w_ptr(const c10::optional<torch::Tensor>& w, int64_t R, int64_t k, const torch::Tensor& like) {
  if (!w.has_value() || !w->defined()) return nullptr;
  TORCH_CHECK(w->is_cuda() && w->device() == like.device() && w->scalar_type() == at::kFloat && w->is_contiguous() &&
                  w->numel() == R * k, "w must be a contiguous fp32 [R, k] CUDA tensor");
  return w->data_ptr<float>();
}


// --- MXFP8 message rows (expert-parallel return direction) ------------------------------------------------------
// [R, d] -> uint8 [R, d + d / 32]: e4m3 codes, then per 32 values a biased exponent e + 127 of the scale 2^e,
// the smallest power of two with amax / 2^e <= 448. Round to nearest, or (seed) stochastically: the scaled
// value's fp32 bits get 20 random bits added below e4m3's 3 mantissa bits and are truncated (values under the
// smallest normal, 2^-6, are lifted by 2^-6 first, where the spacing equals the subnormal one).
template <typename T>
__global__ void __launch_bounds__(256) mxfp8_encode_kernel(const T* __restrict__ x, int64_t ld_x,
                                                           uint8_t* __restrict__ out, int64_t ld_out, int64_t R,
                                                           int64_t d, const int64_t* __restrict__ seed) {
  const int64_t nb = d / 32;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= R * nb) return;
  const int64_t r = i / nb, b = i - r * nb;
  const T* src = x + r * ld_x + b * 32;
  float v[32];
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    v[j] = to_f(src[j]);
    amax = fmaxf(amax, fabsf(v[j]));
  }
  int e = -127;
  if (amax > 0.0f) {
    int ea;
    const float m = frexpf(amax, &ea);  // amax = m 2^ea, m in [0.5, 1); 448 = 0.875 2^9
    e = m <= 0.875f ? ea - 9 : ea - 8;
    e = e < -127 ? -127 : (e > 127 ? 127 : e);
  }
  const float inv = ldexpf(1.0f, -e);
  uint32_t key = 0;
  if (seed) key = nvfp4::sr_key(static_cast<uint64_t>(*seed));
  uint8_t* dst = out + r * ld_out + b * 32;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    float y = v[j] * inv;
    if (seed) {
      const float a = fabsf(y);
      const float off = a < 0x1p-6f ? 0x1p-6f : 0.0f;
      const uint32_t rnd = nvfp4::hash_u32(static_cast<uint32_t>(r * d + b * 32 + j) + key) & 0xFFFFFu;
      const uint32_t bits = (__float_as_uint(__fadd_rn(a, off)) + rnd) & ~0xFFFFFu;
      const float q = __fsub_rn(__uint_as_float(bits), off);
      y = __uint_as_float(__float_as_uint(q) | (__float_as_uint(y) & 0x80000000u));
    }
    dst[j] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(y, __NV_SATFINITE, __NV_E4M3));
  }
  out[r * ld_out + d + b] = static_cast<uint8_t>(e + 127);
}

template <typename OutT>
__global__ void __launch_bounds__(256) mxfp8_decode_kernel(const uint8_t* __restrict__ buf, int64_t ld,
                                                           OutT* __restrict__ out, int64_t R, int64_t d) {
  const int64_t nb = d / 32;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= R * nb) return;
  const int64_t r = i / nb, b = i - r * nb;
  const float s = ldexpf(1.0f, static_cast<int>(buf[r * ld + d + b]) - 127);
  const uint8_t* src = buf + r * ld + b * 32;
  OutT* dst = out + r * d + b * 32;
#pragma unroll
  for (int j = 0; j < 32; ++j) dst[j] = from_f<OutT>(__fmul_rn(nvfp4::e4m3_to_float(src[j]), s));
}

#define NVFP4_COMBINE_DISPATCH(ST, ...)                                         \
  switch (ST) {                                                                \
    case at::kBFloat16: { using T = __nv_bfloat16; __VA_ARGS__; break; }       \
    case at::kHalf: { using T = __half; __VA_ARGS__; break; }                  \
    default: { using T = float; __VA_ARGS__; break; }                          \
  }

}  // namespace combine
}  // namespace nvfp4

using OptT = c10::optional<torch::Tensor>;

torch::Tensor nvfp4_combine_fwd(torch::Tensor rows, torch::Tensor pos, OptT w) {
  using namespace nvfp4::combine;
  check_2d(rows, "rows");
  TORCH_CHECK(pos.is_cuda() && pos.device() == rows.device() && pos.dim() == 2 && pos.scalar_type() == at::kLong &&
                  pos.is_contiguous(), "pos must be a contiguous int64 [R, k] CUDA tensor");
  const c10::cuda::CUDAGuard guard(rows.device());
  const int64_t R = pos.size(0), k = pos.size(1), d = rows.size(1), M = rows.size(0);
  const float* wp = w_ptr(w, R, k, rows);
  auto out = torch::empty({R, d}, rows.options());
  if (R == 0 || d == 0) return out;
  if (k == 0) return out.zero_();
  auto stream = at::cuda::getCurrentCUDAStream();
  NVFP4_COMBINE_DISPATCH(rows.scalar_type(), {
    const T* rp = reinterpret_cast<const T*>(rows.data_ptr());
    T* op = reinterpret_cast<T*>(out.data_ptr());
    const int V = vec_for<T>(rp, rows.stride(0), d);
    if (V > 1) {
      constexpr int VV = 16 / sizeof(T);
      const dim3 grid(R, (d / VV + 127) / 128);
      combine_fwd_kernel<T, VV><<<grid, 128, 0, stream>>>(rp, rows.stride(0), M, pos.data_ptr<int64_t>(), wp, k, op, d);
    } else {
      const dim3 grid(R, (d + 127) / 128);
      combine_fwd_kernel<T, 1><<<grid, 128, 0, stream>>>(rp, rows.stride(0), M, pos.data_ptr<int64_t>(), wp, k, op, d);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor nvfp4_combine_bwd_rows(torch::Tensor g, torch::Tensor inv, OptT w, int64_t k) {
  using namespace nvfp4::combine;
  check_2d(g, "g");
  TORCH_CHECK(inv.is_cuda() && inv.device() == g.device() && inv.dim() == 1 && inv.scalar_type() == at::kLong &&
                  inv.is_contiguous(), "inv must be a contiguous int64 [M] CUDA tensor");
  TORCH_CHECK(k > 0, "k must be positive");
  const c10::cuda::CUDAGuard guard(g.device());
  const int64_t M = inv.size(0), d = g.size(1);
  const float* wp = w_ptr(w, g.size(0), k, g);
  auto out = torch::empty({M, d}, g.options().memory_format(at::MemoryFormat::Contiguous));
  if (M == 0 || d == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream();
  NVFP4_COMBINE_DISPATCH(g.scalar_type(), {
    const T* gp = reinterpret_cast<const T*>(g.data_ptr());
    T* op = reinterpret_cast<T*>(out.data_ptr());
    const int V = vec_for<T>(gp, g.stride(0), d);
    if (V > 1) {
      constexpr int VV = 16 / sizeof(T);
      const dim3 grid(M, (d / VV + 127) / 128);
      combine_bwd_rows_kernel<T, VV><<<grid, 128, 0, stream>>>(gp, g.stride(0), inv.data_ptr<int64_t>(), wp,
                                                               static_cast<int>(k), op, d);
    } else {
      const dim3 grid(M, (d + 127) / 128);
      combine_bwd_rows_kernel<T, 1><<<grid, 128, 0, stream>>>(gp, g.stride(0), inv.data_ptr<int64_t>(), wp,
                                                              static_cast<int>(k), op, d);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor nvfp4_combine_bwd_w(torch::Tensor g, torch::Tensor rows, torch::Tensor pos) {
  using namespace nvfp4::combine;
  check_2d(g, "g");
  check_2d(rows, "rows");
  TORCH_CHECK(g.scalar_type() == rows.scalar_type() && g.device() == rows.device() && g.size(1) == rows.size(1),
              "g and rows must match in dtype, device and columns");
  TORCH_CHECK(pos.is_cuda() && pos.device() == g.device() && pos.dim() == 2 && pos.scalar_type() == at::kLong &&
                  pos.is_contiguous() && pos.size(0) == g.size(0), "pos must be a contiguous int64 [R, k] tensor");
  const int64_t R = pos.size(0), k = pos.size(1), d = g.size(1);
  TORCH_CHECK(k <= 32, "at most 32 choices per row");
  const c10::cuda::CUDAGuard guard(g.device());
  auto dw = torch::empty({R, k}, g.options().dtype(at::kFloat));
  if (R == 0 || k == 0) return dw;
  auto stream = at::cuda::getCurrentCUDAStream();
  NVFP4_COMBINE_DISPATCH(g.scalar_type(), {
    combine_bwd_w_kernel<T><<<R, 32 * k, 0, stream>>>(reinterpret_cast<const T*>(g.data_ptr()), g.stride(0),
                                                       reinterpret_cast<const T*>(rows.data_ptr()), rows.stride(0),
                                                       rows.size(0), pos.data_ptr<int64_t>(), static_cast<int>(k),
                                                       dw.data_ptr<float>(), d);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dw;
}


torch::Tensor nvfp4_mxfp8_encode(torch::Tensor x, OptT seed) {
  using namespace nvfp4::combine;
  check_2d(x, "x");
  const int64_t R = x.size(0), d = x.size(1);
  TORCH_CHECK(d % 32 == 0, "MXFP8 rows need a multiple of 32 columns");
  const int64_t* sp = nullptr;
  if (seed.has_value() && seed->defined()) {
    TORCH_CHECK(seed->is_cuda() && seed->scalar_type() == at::kLong && seed->numel() == 1, "seed: one int64 on the GPU");
    sp = seed->data_ptr<int64_t>();
  }
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty({R, d + d / 32}, x.options().dtype(at::kByte));
  if (R == 0 || d == 0) return out;
  const int64_t n = R * (d / 32);
  const int grid = static_cast<int>((n + 255) / 256);
  auto stream = at::cuda::getCurrentCUDAStream();
  NVFP4_COMBINE_DISPATCH(x.scalar_type(), {
    mxfp8_encode_kernel<T><<<grid, 256, 0, stream>>>(reinterpret_cast<const T*>(x.data_ptr()), x.stride(0),
                                                    out.data_ptr<uint8_t>(), out.stride(0), R, d, sp);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor nvfp4_mxfp8_decode(torch::Tensor buf, int64_t d, at::ScalarType out_dtype) {
  using namespace nvfp4::combine;
  TORCH_CHECK(buf.is_cuda() && buf.dim() == 2 && buf.scalar_type() == at::kByte && buf.stride(1) == 1 &&
                  buf.size(1) >= d + d / 32 && d % 32 == 0, "buf must hold uint8 [R, >= d + d / 32] MXFP8 rows");
  TORCH_CHECK(out_dtype == at::kBFloat16 || out_dtype == at::kFloat || out_dtype == at::kHalf, "bad out_dtype");
  const c10::cuda::CUDAGuard guard(buf.device());
  const int64_t R = buf.size(0);
  auto out = torch::empty({R, d}, buf.options().dtype(out_dtype));
  if (R == 0 || d == 0) return out;
  const int64_t n = R * (d / 32);
  const int grid = static_cast<int>((n + 255) / 256);
  auto stream = at::cuda::getCurrentCUDAStream();
  NVFP4_COMBINE_DISPATCH(out_dtype, {
    mxfp8_decode_kernel<T><<<grid, 256, 0, stream>>>(buf.data_ptr<uint8_t>(), buf.stride(0),
                                                    reinterpret_cast<T*>(out.data_ptr()), R, d);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
