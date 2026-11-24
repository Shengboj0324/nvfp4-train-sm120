// SASS instruction-count probe for software stochastic rounding to E2M1 on sm_120a.
// Each kernel: one thread quantizes 16 pre-scaled fp32 values (one NVFP4 block) to 8 bytes.
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t cvt_rn_e2m1x2(float hi, float lo) {
  uint32_t r;
  asm("{\n .reg .b8 t;\n cvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n cvt.u32.u8 %0, t;\n}" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ uint64_t pack16(const float* q) {
  uint64_t out = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) out |= (uint64_t)cvt_rn_e2m1x2(q[2 * i + 1], q[2 * i]) << (8 * i);
  return out;
}

// ---------------- RNGs ----------------
__device__ __forceinline__ uint32_t lowbias32(uint32_t x) {
  x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16; return x;
}
__device__ __forceinline__ uint4 pcg4d(uint4 v) {
  v.x = v.x * 1664525u + 1013904223u; v.y = v.y * 1664525u + 1013904223u;
  v.z = v.z * 1664525u + 1013904223u; v.w = v.w * 1664525u + 1013904223u;
  v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
  v.x ^= v.x >> 16; v.y ^= v.y >> 16; v.z ^= v.z >> 16; v.w ^= v.w >> 16;
  v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
  return v;
}
template <int R>
__device__ __forceinline__ uint4 philox4x32(uint4 c, uint2 k) {
#pragma unroll
  for (int i = 0; i < R; ++i) {
    uint32_t hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
    uint32_t hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
    c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    k.x += 0x9E3779B9u; k.y += 0xBB67AE85u;
  }
  return c;
}

// ---------------- SR value functions ----------------
// (a) project's current float-domain SR (16-bit uniform), copied from csrc/nvfp4_common.cuh
__device__ __forceinline__ float e2m1_sr_value(float y, float u) {
  const float a = fminf(fabsf(y), 6.0f);
  const uint32_t abits = __float_as_uint(a);
  const bool ge1 = a >= 1.0f;
  const float glo = ge1 ? __uint_as_float(abits & 0xFFC00000u) : (a >= 0.5f ? 0.5f : 0.0f);
  const uint32_t e = abits >> 23;
  const float inv_w = ge1 ? __uint_as_float((255u - e) << 23) : 2.0f;
  const float w = ge1 ? __uint_as_float((e - 1u) << 23) : 0.5f;
  const float p_up = __fmul_rn(__fsub_rn(a, glo), inv_w);
  const float r = (u < p_up) ? __fadd_rn(glo, w) : glo;
  return __uint_as_float(__float_as_uint(r) | (__float_as_uint(y) & 0x80000000u));
}
__device__ __forceinline__ float u16(uint32_t n16) { return __fsub_rn(__uint_as_float(0x3F800000u | (n16 << 7)), 1.0f); }

// (b) Transformer Engine's software fallback (8 random bits), copied from TE util/ptx.cuh
__device__ __forceinline__ float te_sr(const float x, const uint32_t rbits8) {
  const float u = static_cast<float>(rbits8 & 0xFFu) * (1.0f / 256.0f);
  const float a = fabsf(x);
  const float step = (a >= 4.0f) ? 2.0f : ((a >= 2.0f) ? 1.0f : 0.5f);
  const float t = fmaf(u, step, a);
  const float step_t = (t >= 4.0f) ? 2.0f : ((t >= 2.0f) ? 1.0f : 0.5f);
  const float inv_step_t = (t >= 4.0f) ? 0.5f : ((t >= 2.0f) ? 1.0f : 2.0f);
  const float q = fminf(floorf(t * inv_step_t) * step_t, 6.0f);
  return copysignf(q, (x != x) ? 1.0f : x);
}

// (c) integer-domain SR: add 8 random bits (+ half-LSB offset, "SRF") to the 22 discarded fp32 mantissa
// bits, truncate to 1 mantissa bit. |x| < 1 is lifted into [1,2) (+1), where fp32 spacing at 1 kept
// mantissa bit is 0.5 = the E2M1 spacing on [0,2). Result is on the E2M1 grid, so cvt.rn is exact.
__device__ __forceinline__ float int_sr(const float x, const uint32_t r8) {
  const float a = fabsf(x);
  const float off = a < 1.0f ? 1.0f : 0.0f;
  const float z = a + off;
  uint32_t b = __float_as_uint(z) + ((r8 & 0xFFu) << 14) + (1u << 13);
  b &= 0xFFC00000u;
  const float q = fminf(__uint_as_float(b) - off, 6.0f);
  return copysignf(q, x);
}

// ---------------- kernels ----------------
#define LOAD16                                                                   \
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;                      \
  float v[16];                                                                   \
  _Pragma("unroll") for (int i = 0; i < 4; ++i) {                                \
    float4 f = reinterpret_cast<const float4*>(x)[4 * t + i];                    \
    v[4 * i] = f.x; v[4 * i + 1] = f.y; v[4 * i + 2] = f.z; v[4 * i + 3] = f.w;  \
  }                                                                              \
  float q[16];

extern "C" __global__ void k_rn(const float* __restrict__ x, uint64_t* __restrict__ y, uint64_t seed) {
  LOAD16
#pragma unroll
  for (int i = 0; i < 16; ++i) q[i] = v[i];
  y[t] = pack16(q);
}
// current: 2x nested lowbias32 per pair, 16-bit uniforms, float-domain SR
extern "C" __global__ void k_current(const float* __restrict__ x, uint64_t* __restrict__ y, uint64_t seed) {
  LOAD16
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    uint32_t k = 8 * t + p;
    uint32_t h = lowbias32(lowbias32(k ^ (uint32_t)seed) + (uint32_t)(seed >> 32));
    q[2 * p] = e2m1_sr_value(v[2 * p], u16(h & 0xFFFFu));
    q[2 * p + 1] = e2m1_sr_value(v[2 * p + 1], u16(h >> 16));
  }
  y[t] = pack16(q);
}
// hash only (current RNG), SR replaced by int_sr: isolates the RNG cost
extern "C" __global__ void k_hash2x_per_pair_intsr(const float* __restrict__ x, uint64_t* __restrict__ y, uint64_t seed) {
  LOAD16
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    uint32_t h = lowbias32(lowbias32((8 * t + p) ^ (uint32_t)seed) + (uint32_t)(seed >> 32));
    q[2 * p] = int_sr(v[2 * p], h >> 8);
    q[2 * p + 1] = int_sr(v[2 * p + 1], h >> 24);
  }
  y[t] = pack16(q);
}
// one lowbias32 per 4 elements (8 bits each), key folded once per thread
extern "C" __global__ void k_hash1x_per_quad_intsr(const float* __restrict__ x, uint64_t* __restrict__ y, uint64_t seed) {
  LOAD16
  const uint32_t base = lowbias32(t ^ (uint32_t)seed) + (uint32_t)(seed >> 32);  // per-thread key
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    uint32_t h = lowbias32(base + 0x9E3779B9u * g);
#pragma unroll
    for (int j = 0; j < 4; ++j) q[4 * g + j] = int_sr(v[4 * g + j], h >> (8 * j));
  }
  y[t] = pack16(q);
}
extern "C" __global__ void k_hash1x_per_quad_cursr(const float* __restrict__ x, uint64_t* __restrict__ y, uint64_t seed) {
  LOAD16
  const uint32_t base = lowbias32(t ^ (uint32_t)seed) + (uint32_t)(seed >> 32);
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    uint32_t h = lowbias32(base + 0x9E3779B9u * g);
#pragma unroll
    for (int j = 0; j < 4; ++j) q[4 * g + j] = e2m1_sr_value(v[4 * g + j], u16(((h >> (8 * j)) & 0xFFu) << 8 | 0x80u));
  }
  y[t] = pack16(q);
}
extern "C" __global__ void k_pcg4d_intsr(const float* __restrict__ x, uint64_t* __restrict__ y, uint64_t seed) {
  LOAD16
  uint4 h = pcg4d(make_uint4(t, (uint32_t)seed, (uint32_t)(seed >> 32), 0x2545F491u));
  uint32_t w[4] = {h.x, h.y, h.z, h.w};
#pragma unroll
  for (int i = 0; i < 16; ++i) q[i] = int_sr(v[i], w[i >> 2] >> (8 * (i & 3)));
  y[t] = pack16(q);
}
extern "C" __global__ void k_philox10_tesr(const float* __restrict__ x, uint64_t* __restrict__ y, uint64_t seed) {
  LOAD16
  uint4 h = philox4x32<10>(make_uint4(t, 0, 0, 0), make_uint2((uint32_t)seed, (uint32_t)(seed >> 32)));
  uint32_t w[4] = {h.x, h.y, h.z, h.w};
#pragma unroll
  for (int i = 0; i < 16; ++i) q[i] = te_sr(v[i], w[i >> 2] >> (8 * (i & 3)));
  y[t] = pack16(q);
}
extern "C" __global__ void k_philox10_intsr(const float* __restrict__ x, uint64_t* __restrict__ y, uint64_t seed) {
  LOAD16
  uint4 h = philox4x32<10>(make_uint4(t, 0, 0, 0), make_uint2((uint32_t)seed, (uint32_t)(seed >> 32)));
  uint32_t w[4] = {h.x, h.y, h.z, h.w};
#pragma unroll
  for (int i = 0; i < 16; ++i) q[i] = int_sr(v[i], w[i >> 2] >> (8 * (i & 3)));
  y[t] = pack16(q);
}
extern "C" __global__ void k_philox7_intsr(const float* __restrict__ x, uint64_t* __restrict__ y, uint64_t seed) {
  LOAD16
  uint4 h = philox4x32<7>(make_uint4(t, 0, 0, 0), make_uint2((uint32_t)seed, (uint32_t)(seed >> 32)));
  uint32_t w[4] = {h.x, h.y, h.z, h.w};
#pragma unroll
  for (int i = 0; i < 16; ++i) q[i] = int_sr(v[i], w[i >> 2] >> (8 * (i & 3)));
  y[t] = pack16(q);
}
// SR math alone with a free "random" source (bits of the thread index), to isolate SR-value cost
extern "C" __global__ void k_norng_intsr(const float* __restrict__ x, uint64_t* __restrict__ y, uint64_t seed) {
  LOAD16
  uint32_t w = t ^ (uint32_t)seed;
#pragma unroll
  for (int i = 0; i < 16; ++i) q[i] = int_sr(v[i], w >> (2 * i));
  y[t] = pack16(q);
}
extern "C" __global__ void k_norng_cursr(const float* __restrict__ x, uint64_t* __restrict__ y, uint64_t seed) {
  LOAD16
  uint32_t w = t ^ (uint32_t)seed;
#pragma unroll
  for (int i = 0; i < 16; ++i) q[i] = e2m1_sr_value(v[i], u16((w >> i) & 0xFFFFu));
  y[t] = pack16(q);
}
extern "C" __global__ void k_norng_tesr(const float* __restrict__ x, uint64_t* __restrict__ y, uint64_t seed) {
  LOAD16
  uint32_t w = t ^ (uint32_t)seed;
#pragma unroll
  for (int i = 0; i < 16; ++i) q[i] = te_sr(v[i], w >> (2 * i));
  y[t] = pack16(q);
}
