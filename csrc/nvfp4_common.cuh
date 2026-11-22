// Shared device helpers. Every function here mirrors a function in nvfp4/reference.py and must stay
// bit-identical to it (tests/test_quantize_cuda.py enforces this).
#pragma once

#include <cstdint>
#include <cuda_fp8.h>

namespace nvfp4 {

constexpr int kBlock = 16;
constexpr float kE2M1Max = 6.0f;
constexpr float kE4M3Max = 448.0f;

// reference.e2m1_round_rne
__device__ __forceinline__ uint32_t e2m1_round_rne(float v) {
  float a = fabsf(v);
  uint32_t code = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) +
                  (a > 5.0f);
  uint32_t sign = (v < 0.0f && code > 0) ? 8u : 0u;
  return code | sign;
}

__device__ __forceinline__ float e2m1_grid(uint32_t c) {
  // 0, .5, 1, 1.5, 2, 3, 4, 6
  return c < 4 ? 0.5f * float(c) : (c == 4 ? 2.0f : (c == 5 ? 3.0f : (c == 6 ? 4.0f : 6.0f)));
}

// reference.e2m1_round_stochastic
__device__ __forceinline__ uint32_t e2m1_round_stochastic(float v, float u) {
  float a = fminf(fabsf(v), kE2M1Max);
  uint32_t lo = (a >= 0.5f) + (a >= 1.0f) + (a >= 1.5f) + (a >= 2.0f) + (a >= 3.0f) + (a >= 4.0f) + (a >= 6.0f);
  uint32_t hi = lo < 7 ? lo + 1 : 7;
  float glo = e2m1_grid(lo), ghi = e2m1_grid(hi);
  float width = ghi - glo;
  float p_up = width > 0.0f ? __fdiv_rn(a - glo, width) : 0.0f;
  uint32_t code = (u < p_up) ? hi : lo;
  uint32_t sign = (v < 0.0f && code > 0) ? 8u : 0u;
  return code | sign;
}

// reference.e4m3_from_float (input already clamped to [0, 448])
__device__ __forceinline__ uint8_t e4m3_from_float(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(fminf(fmaxf(x, 0.0f), kE4M3Max), __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3);
  return __half2float(__half(h));
}

// reference.hash_u32 ("lowbias32")
__device__ __forceinline__ uint32_t hash_u32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

// reference.sr_uniform
__device__ __forceinline__ float sr_uniform(uint64_t seed, uint64_t index) {
  uint32_t idx_lo = static_cast<uint32_t>(index);  // reference masks the int64 index to 32 bits
  uint32_t h = hash_u32(hash_u32(idx_lo ^ static_cast<uint32_t>(seed)) + static_cast<uint32_t>(seed >> 32));
  return float(h >> 8) * (1.0f / 16777216.0f);
}

// reference.rht_signs: bit 0 of hash(i ^ seed)
__device__ __forceinline__ float rht_sign(uint64_t seed, int i) {
  return (hash_u32(static_cast<uint32_t>(i) ^ static_cast<uint32_t>(seed)) & 1u) ? -1.0f : 1.0f;
}

// reference.apply_rht for one 16-element block, in place.
__device__ __forceinline__ void rht16(float (&v)[kBlock], uint64_t seed) {
#pragma unroll
  for (int i = 0; i < kBlock; ++i) v[i] *= rht_sign(seed, i);
#pragma unroll
  for (int h = 1; h < kBlock; h *= 2) {
#pragma unroll
    for (int i = 0; i < kBlock; ++i) {
      if ((i & h) == 0) {
        float a = v[i], b = v[i + h];
        v[i] = __fadd_rn(a, b);
        v[i + h] = __fsub_rn(a, b);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < kBlock; ++i) v[i] *= 0.25f;
}

}  // namespace nvfp4
