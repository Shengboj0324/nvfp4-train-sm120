// Backward of LayerNorm / RMSNorm from the forward statistics of the fused quantizers (mean / rstd), with
// an optional residual gradient added to dx: when x feeds both the norm and a residual connection, the
// two gradients of x are summed here instead of in a separate elementwise pass.
//
//   LayerNorm:  n = (x - mean) * rstd,  y = n * w + b
//               dx = rstd * (g - mean(g) - n * mean(g * n)) + dres,  g = dy * w;  dw = sum_rows dy * n,  db = sum_rows dy
//   RMSNorm:    n = x * rstd,  y = n * w
//               dx = rstd * (g - n * mean(g * n)) + dres;  dw = sum_rows dy * n
//
// dy (the gradient of the normalized output) is fp32, as the dgrad GEMM writes it. Deterministic: row
// sums are fixed-order per lane followed by a fixed shuffle tree, and dw / db are summed per tile of rows
// in row order, then over tiles (no atomics).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

using OptTensor = c10::optional<torch::Tensor>;

namespace nvfp4 {
namespace norm {

constexpr int THREADS = 256;
constexpr int VEC = 8;           // elements per lane per step of the row kernel
constexpr int ROWS_PER_TILE = 64;  // dw / db partial sums of the column kernel (long rows)

__device__ __forceinline__ void load8(const float* __restrict__ p, float* v) {
  const float4 a = __ldg(reinterpret_cast<const float4*>(p)), b = __ldg(reinterpret_cast<const float4*>(p) + 1);
  v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
}

__device__ __forceinline__ void load8(const __nv_bfloat16* __restrict__ p, float* v) {
  const uint4 u = __ldg(reinterpret_cast<const uint4*>(p));
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    v[2 * i] = f.x;
    v[2 * i + 1] = f.y;
  }
}

__device__ __forceinline__ void store8(float* p, const float* v) {
  reinterpret_cast<float4*>(p)[0] = make_float4(v[0], v[1], v[2], v[3]);
  reinterpret_cast<float4*>(p)[1] = make_float4(v[4], v[5], v[6], v[7]);
}

__device__ __forceinline__ void store8(__nv_bfloat16* p, const float* v) {
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = u;
}

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }

// One CTA per group of ROWS_PER_GROUP rows, each row held in registers (C <= blockDim * VEC * NV): one read
// of dy and x gives dx and, accumulated per thread over the group's rows (in row order), the dw / db partial
// sums of the thread's columns -> part[group, C]. Block sums are warp-shuffle trees, then the warps'
// partials in warp order (deterministic).
constexpr int ROWS_PER_GROUP = 16;

template <typename T, bool kRms, int NV>
__global__ void __launch_bounds__(THREADS) norm_bwd_row_kernel(const float* __restrict__ dy, const T* __restrict__ x,
                                                              const float* __restrict__ mean,
                                                              const float* __restrict__ rstd,
                                                              const float* __restrict__ w, const T* __restrict__ dres,
                                                              T* __restrict__ dx, float* __restrict__ dw_part,
                                                              float* __restrict__ db_part, int R, int C) {
  __shared__ float red[2][THREADS / 32];
  __shared__ float tot[2];
  const int warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
  const float inv_c = 1.0f / static_cast<float>(C);
  float acc_w[NV][VEC], acc_b[NV][VEC];
#pragma unroll
  for (int j = 0; j < NV; ++j)
#pragma unroll
    for (int i = 0; i < VEC; ++i) acc_w[j][i] = acc_b[j][i] = 0.0f;
  const int r_end = min(R, (blockIdx.x + 1) * ROWS_PER_GROUP);
  for (int r = blockIdx.x * ROWS_PER_GROUP; r < r_end; ++r) {
    const int64_t base = static_cast<int64_t>(r) * C;
    const float m = kRms ? 0.0f : __ldg(mean + r), rs = __ldg(rstd + r);
    float n[NV][VEC], g[NV][VEC];
    float s1 = 0.0f, s2 = 0.0f;
#pragma unroll
    for (int j = 0; j < NV; ++j) {
      const int c = (j * blockDim.x + threadIdx.x) * VEC;
      if (c < C) {
        float d[VEC], xv[VEC];
        load8(dy + base + c, d);
        load8(x + base + c, xv);
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          n[j][i] = kRms ? xv[i] * rs : (xv[i] - m) * rs;
          g[j][i] = d[i] * __ldg(w + c + i);
          s1 += g[j][i];
          s2 += g[j][i] * n[j][i];
          acc_w[j][i] += d[i] * n[j][i];
          acc_b[j][i] += d[i];
        }
      }
    }
    if (dx == nullptr) continue;  // uniform
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      s1 += __shfl_xor_sync(0xffffffffu, s1, o);
      s2 += __shfl_xor_sync(0xffffffffu, s2, o);
    }
    if ((threadIdx.x & 31) == 0) {
      red[0][warp] = s1;
      red[1][warp] = s2;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float a = 0.0f, b = 0.0f;
      for (int i = 0; i < nwarps; ++i) {
        a += red[0][i];
        b += red[1][i];
      }
      tot[0] = a;
      tot[1] = b;
    }
    __syncthreads();  // (red is rewritten only after the next row's first barrier: all have read tot)
    const float a = kRms ? 0.0f : tot[0] * inv_c, b = tot[1] * inv_c;
#pragma unroll
    for (int j = 0; j < NV; ++j) {
      const int c = (j * blockDim.x + threadIdx.x) * VEC;
      if (c < C) {
        float out[VEC];
#pragma unroll
        for (int i = 0; i < VEC; ++i) out[i] = rs * (g[j][i] - a - n[j][i] * b);
        if (dres != nullptr) {
          float res[VEC];
          load8(dres + base + c, res);
#pragma unroll
          for (int i = 0; i < VEC; ++i) out[i] += res[i];
        }
        store8(dx + base + c, out);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < NV; ++j) {
    const int c = (j * blockDim.x + threadIdx.x) * VEC;
    if (c < C) {
      const int64_t o = static_cast<int64_t>(blockIdx.x) * C + c;
      if (dw_part != nullptr) store8(dw_part + o, acc_w[j]);
      if (db_part != nullptr) store8(db_part + o, acc_b[j]);
    }
  }
}

// dx for long rows: one warp per row, two passes over the row (the second partly hits L2).
template <typename T, bool kRms>
__global__ void __launch_bounds__(THREADS) norm_bwd_dx_kernel(const float* __restrict__ dy, const T* __restrict__ x,
                                                             const float* __restrict__ mean,
                                                             const float* __restrict__ rstd,
                                                             const float* __restrict__ w, const T* __restrict__ dres,
                                                             T* __restrict__ dx, int R, int C) {
  const int lane = threadIdx.x & 31;
  const int r = blockIdx.x * (THREADS / 32) + (threadIdx.x >> 5);
  if (r >= R) return;
  const int64_t base = static_cast<int64_t>(r) * C;
  const float m = kRms ? 0.0f : __ldg(mean + r), rs = __ldg(rstd + r);
  float s1 = 0.0f, s2 = 0.0f;
  for (int c = lane * VEC; c < C; c += 32 * VEC) {
    float d[VEC], xv[VEC];
    load8(dy + base + c, d);
    load8(x + base + c, xv);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const float n = kRms ? xv[i] * rs : (xv[i] - m) * rs;
      const float g = d[i] * __ldg(w + c + i);
      s1 += g;
      s2 += g * n;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    s1 += __shfl_xor_sync(0xffffffffu, s1, o);
    s2 += __shfl_xor_sync(0xffffffffu, s2, o);
  }
  const float inv_c = 1.0f / static_cast<float>(C);
  const float a = kRms ? 0.0f : s1 * inv_c, b = s2 * inv_c;
  for (int c = lane * VEC; c < C; c += 32 * VEC) {
    float d[VEC], xv[VEC], out[VEC];
    load8(dy + base + c, d);
    load8(x + base + c, xv);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const float n = kRms ? xv[i] * rs : (xv[i] - m) * rs;
      const float g = d[i] * __ldg(w + c + i);
      out[i] = rs * (g - a - n * b);
    }
    if (dres != nullptr) {
      float res[VEC];
      load8(dres + base + c, res);
#pragma unroll
      for (int i = 0; i < VEC; ++i) out[i] += res[i];
    }
    store8(dx + base + c, out);
  }
}

// dw (and db) partial sums: thread = column, CTA row y = tile of ROWS_PER_TILE rows; part[y, c].
template <typename T, bool kRms>
__global__ void __launch_bounds__(THREADS) norm_bwd_dw_kernel(const float* __restrict__ dy, const T* __restrict__ x,
                                                             const float* __restrict__ mean,
                                                             const float* __restrict__ rstd, int R, int C,
                                                             float* __restrict__ dw_part, float* __restrict__ db_part) {
  const int c = blockIdx.x * THREADS + threadIdx.x;
  if (c >= C) return;
  const int r0 = blockIdx.y * ROWS_PER_TILE, r1 = min(R, r0 + ROWS_PER_TILE);
  float sw = 0.0f, sb = 0.0f;
#pragma unroll 4
  for (int r = r0; r < r1; ++r) {
    const int64_t i = static_cast<int64_t>(r) * C + c;
    const float d = __ldg(dy + i), xv = to_f(x[i]);
    const float n = kRms ? xv * __ldg(rstd + r) : (xv - __ldg(mean + r)) * __ldg(rstd + r);
    sw += d * n;
    sb += d;
  }
  const int64_t o = static_cast<int64_t>(blockIdx.y) * C + c;
  if (dw_part != nullptr) dw_part[o] = sw;
  if (db_part != nullptr) db_part[o] = sb;
}

}  // namespace norm
}  // namespace nvfp4

// LayerNorm (mean given) / RMSNorm (mean empty) backward. Returns {dx, dw, db}; each is an empty tensor
// unless requested. dres (optional, x's dtype and shape) is added to dx.
std::vector<torch::Tensor> nvfp4_norm_backward(torch::Tensor dy, torch::Tensor x, const OptTensor& mean,
                                               torch::Tensor rstd, torch::Tensor w, const OptTensor& dres, bool need_dx,
                                               bool need_dw, bool need_db) {
  using namespace nvfp4::norm;
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() &&
                  (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat),
              "x must be a contiguous 2D bf16 / fp32 CUDA tensor");
  const int64_t R = x.size(0), C = x.size(1);
  TORCH_CHECK(dy.is_cuda() && dy.scalar_type() == at::kFloat && dy.is_contiguous() && dy.sizes() == x.sizes() &&
                  dy.device() == x.device(),
              "dy must be a contiguous fp32 tensor shaped like x");
  const bool rms = !(mean.has_value() && mean->defined());
  TORCH_CHECK(!(rms && need_db), "RMSNorm has no bias");
  const torch::Tensor* stats[2] = {&rstd, rms ? &rstd : &mean.value()};
  for (const torch::Tensor* t : stats)
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kFloat && t->is_contiguous() && t->numel() == R,
                "mean / rstd must be contiguous fp32 [R] tensors");
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kFloat && w.is_contiguous() && w.numel() == C,
              "w must be a contiguous fp32 [C] tensor");
  const bool has_res = dres.has_value() && dres->defined();
  if (has_res)
    TORCH_CHECK(dres->is_cuda() && dres->scalar_type() == x.scalar_type() && dres->is_contiguous() &&
                    dres->sizes() == x.sizes() && reinterpret_cast<uintptr_t>(dres->data_ptr()) % 16 == 0,
                "dres must be a contiguous, 16-byte aligned tensor of x's dtype and shape");
  TORCH_CHECK(C % VEC == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(dy.data_ptr()) % 16 == 0,
              "x / dy must be 16-byte aligned with x.shape[1] % 8 == 0");
  const c10::cuda::CUDAGuard guard(x.device());
  auto f32 = x.options().dtype(at::kFloat);
  torch::Tensor dx = torch::empty({need_dx ? R : 0, need_dx ? C : 0}, x.options());
  // rows held in registers: threads = C / VEC rounded up to a warp (<= THREADS), NV chunks of VEC each
  const int64_t vecs = C / VEC;
  const int threads = static_cast<int>(std::max<int64_t>(32, std::min<int64_t>(THREADS, (vecs + 31) / 32 * 32)));
  const int64_t nv = (vecs + threads - 1) / threads;
  const bool row_path = nv <= 4;
  const int64_t rows_per = row_path ? ROWS_PER_GROUP : ROWS_PER_TILE;
  const int tiles = static_cast<int>(std::max<int64_t>(1, (R + rows_per - 1) / rows_per));
  torch::Tensor dw_part = torch::empty({need_dw ? tiles : 0, C}, f32), db_part = torch::empty({need_db ? tiles : 0, C}, f32);
  if (R > 0 && C > 0 && (need_dx || need_dw || need_db)) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* mp = rms ? nullptr : mean->data_ptr<float>();
    auto launch = [&](auto* xp, auto rms_c) {
      using T = std::remove_cv_t<std::remove_pointer_t<decltype(xp)>>;
      constexpr bool kRms = decltype(rms_c)::value;
      const T* rp = has_res ? reinterpret_cast<const T*>(dres->data_ptr()) : nullptr;
      T* dxp = need_dx ? reinterpret_cast<T*>(dx.data_ptr()) : nullptr;
      float* dwp = need_dw ? dw_part.data_ptr<float>() : nullptr;
      float* dbp = need_db ? db_part.data_ptr<float>() : nullptr;
      auto row = [&](auto nv_c) {
        constexpr int NVc = decltype(nv_c)::value;
        norm_bwd_row_kernel<T, kRms, NVc><<<tiles, threads, 0, stream>>>(dy.data_ptr<float>(), xp, mp,
                                                                         rstd.data_ptr<float>(), w.data_ptr<float>(),
                                                                         rp, dxp, dwp, dbp, static_cast<int>(R),
                                                                         static_cast<int>(C));
      };
      if (nv == 1) {
        row(std::integral_constant<int, 1>{});
      } else if (nv == 2) {
        row(std::integral_constant<int, 2>{});
      } else if (row_path) {
        row(std::integral_constant<int, 4>{});
      } else {  // long rows: two-pass warp kernel for dx, column kernel for dw / db
        if (need_dx)
          norm_bwd_dx_kernel<T, kRms><<<static_cast<int>((R + THREADS / 32 - 1) / (THREADS / 32)), THREADS, 0,
                                        stream>>>(dy.data_ptr<float>(), xp, mp, rstd.data_ptr<float>(),
                                                  w.data_ptr<float>(), rp, dxp, static_cast<int>(R),
                                                  static_cast<int>(C));
        if (need_dw || need_db)
          norm_bwd_dw_kernel<T, kRms><<<dim3(static_cast<int>((C + THREADS - 1) / THREADS), tiles), THREADS, 0,
                                        stream>>>(dy.data_ptr<float>(), xp, mp, rstd.data_ptr<float>(),
                                                  static_cast<int>(R), static_cast<int>(C), dwp, dbp);
      }
    };
    using RmsT = std::true_type;
    using LnT = std::false_type;
    if (x.scalar_type() == at::kBFloat16) {
      const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
      if (rms) launch(xp, RmsT{}); else launch(xp, LnT{});
    } else {
      const auto* xp = x.data_ptr<float>();
      if (rms) launch(xp, RmsT{}); else launch(xp, LnT{});
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    dw_part.zero_();
    db_part.zero_();
  }
  return {dx, need_dw ? dw_part.sum(0) : dw_part, need_db ? db_part.sum(0) : db_part};
}
