// Shared device helpers. Every function here mirrors a function in nvfp4/reference.py and must stay
// bit-identical to it (tests/test_quantize_cuda.py enforces this).
#pragma once

#include <cstdint>
#include <cuda_fp8.h>

namespace nvfp4 {

constexpr int kBlock = 16;
constexpr float kE2M1Max = 6.0f;
constexpr float kE4M3Max = 448.0f;

// reference.e4m3_from_float (input already clamped to [0, 448])
__device__ __forceinline__ uint8_t e4m3_from_float(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(fminf(fmaxf(x, 0.0f), kE4M3Max), __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3);
  return __half2float(__half(h));
}

// reference.hash_u32 ("lowbias32")
__host__ __device__ __forceinline__ uint32_t hash_u32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

// reference.sr_uniform: (h >> 8) * 2^-24, built without an int->float conversion. With n = h >> 8 < 2^24:
// if n >= 2^23 the result lies in [0.5, 1) and its mantissa is n - 2^23; otherwise 0x3F000000 | n is
// 0.5 + n * 2^-24 and subtracting 0.5 is exact (Sterbenz).
__device__ __forceinline__ float sr_uniform(uint64_t seed, uint64_t index) {
  const uint32_t idx_lo = static_cast<uint32_t>(index);  // reference masks the int64 index to 32 bits
  const uint32_t h = hash_u32(hash_u32(idx_lo ^ static_cast<uint32_t>(seed)) + static_cast<uint32_t>(seed >> 32));
  const float f = __uint_as_float(0x3F000000u | ((h >> 8) & 0x7FFFFFu));
  return (h & 0x80000000u) ? f : __fsub_rn(f, 0.5f);
}

// Stochastic rounding of y onto the signed E2M1 grid, returned as a float grid value (bit-identical to
// e2m1_decode(reference.e2m1_round_stochastic(y, u)) up to the sign of zero). Same arithmetic as the reference:
// glo = largest grid point <= a, p_up = (a - glo) / step, round up iff u < p_up. For a >= 1 the grid is
// "1 mantissa bit", so glo is a with its mantissa truncated to 1 bit and 1/step = 2^(1-E) comes from the
// exponent bits; below 1 the grid is {0, 0.5, 1} with step 0.5.
__device__ __forceinline__ float e2m1_sr_value(float y, float u) {
  const float a = fminf(fabsf(y), kE2M1Max);
  const uint32_t abits = __float_as_uint(a);
  const bool ge1 = a >= 1.0f;
  const float glo = ge1 ? __uint_as_float(abits & 0xFFC00000u) : (a >= 0.5f ? 0.5f : 0.0f);
  const uint32_t e = abits >> 23;                                            // biased exponent (a >= 0)
  const float inv_w = ge1 ? __uint_as_float((255u - e) << 23) : 2.0f;       // 1 / step
  const float w = ge1 ? __uint_as_float((e - 1u) << 23) : 0.5f;             // step = 2^(E-1)
  const float p_up = __fmul_rn(__fsub_rn(a, glo), inv_w);                   // 0 when a == 6
  const float r = (u < p_up) ? __fadd_rn(glo, w) : glo;
  return __uint_as_float(__float_as_uint(r) | (__float_as_uint(y) & 0x80000000u));
}

// reference.rht_signs: element i is negated iff bit 0 of hash(i ^ seed) is set. Computed once on the
// host as a 16-bit mask (bit i set = negate element i).
__host__ __device__ __forceinline__ uint32_t rht_sign_mask(uint64_t seed) {
  uint32_t m = 0;
  for (int i = 0; i < kBlock; ++i) m |= (hash_u32(static_cast<uint32_t>(i) ^ static_cast<uint32_t>(seed)) & 1u) << i;
  return m;
}

// reference.apply_rht for one 16-element block, in place. Negation by sign-bit flip is exact, so this
// equals the reference's multiplication by -1 bit for bit.
__device__ __forceinline__ void rht16(float (&v)[kBlock], uint32_t sign_mask) {
#pragma unroll
  for (int i = 0; i < kBlock; ++i) v[i] = __uint_as_float(__float_as_uint(v[i]) ^ (((sign_mask >> i) & 1u) << 31));
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

// Hardware round-to-nearest-even, saturating f32 -> E2M1 for two values (sm_100a / sm_120a):
// low nibble = lo, high nibble = hi. Same result as reference.e2m1_round_rne except that negative values which
// round to zero come out as -0 (code 8); clear_negative_zeros() maps those to +0 like the reference.
__device__ __forceinline__ uint32_t cvt_e2m1x2_rn(float lo, float hi) {
  uint32_t r;
  asm("{\n .reg .b8 t;\n cvt.rn.satfinite.e2m1x2.f32 t, %2, %1;\n cvt.u32.u8 %0, t;\n}" : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}

// For 8 packed E2M1 codes: drop the sign bit of every nibble whose magnitude bits are zero. Adding 7 to
// a 3-bit magnitude carries into bit 3 iff the magnitude is non-zero (and never into the next nibble).
__device__ __forceinline__ uint32_t clear_negative_zeros(uint32_t w) {
  const uint32_t mag = w & 0x77777777u;
  const uint32_t nonzero = (mag + 0x77777777u) & 0x88888888u;
  return mag | (w & nonzero);
}

// reference.global_scales: genc = 2688 / amax (1 if amax == 0); gdec = 1 / genc.
__device__ __forceinline__ float global_encode_scale(float amax) {
  return amax > 0.0f ? __fdiv_rn(kE4M3Max * kE2M1Max, amax) : 1.0f;
}

// reference.quantize for one 16-element block whose abs-max (over the 1x16 or 16x16 block) is bmax.
// Returns the UE4M3 scale byte and the 16 E2M1 codes packed low-nibble-first into `packed`.
// Stochastic rounding uses sr_uniform(sr_seed, lin0 + i), lin0 = linear index of element 0.
template <bool kStochastic>
__device__ __forceinline__ uint8_t quantize16(const float (&v)[kBlock], float bmax, float genc, uint64_t sr_seed,
                                              uint64_t lin0, uint2& packed) {
  const float s_target = fminf(__fmul_rn(__fdiv_rn(bmax, kE2M1Max), genc), kE4M3Max);
  const uint8_t s = e4m3_from_float(s_target);
  const float sf = e4m3_to_float(s);
  const float enc = sf > 0.0f ? __fdiv_rn(genc, sf) : 0.0f;
  uint32_t lo = 0, hi = 0;
#pragma unroll
  for (int i = 0; i < kBlock; i += 2) {
    float y0 = __fmul_rn(v[i], enc), y1 = __fmul_rn(v[i + 1], enc);
    if constexpr (kStochastic) {  // round onto the grid first; the convert below is then exact
      y0 = e2m1_sr_value(y0, sr_uniform(sr_seed, lin0 + i));
      y1 = e2m1_sr_value(y1, sr_uniform(sr_seed, lin0 + i + 1));
    }
    const uint32_t b = cvt_e2m1x2_rn(y0, y1);
    if (i < 8)
      lo |= b << (4 * i);
    else
      hi |= b << (4 * (i - 8));
  }
  packed = make_uint2(clear_negative_zeros(lo), clear_negative_zeros(hi));
  return s;
}

}  // namespace nvfp4
