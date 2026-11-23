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

// reference.sr_uniform2: the stochastic-rounding uniforms of elements 2k and 2k + 1 from one 32-bit hash of
// the pair index k (keyed by the 64-bit seed), 16 bits each: u0 = (h & 0xFFFF) * 2^-16, u1 = (h >> 16) * 2^-16.
// Built without int->float conversions: 0x3F800000 | (n << 7) is 1 + n * 2^-16, and subtracting 1 is exact.
// (One hash per two elements: sm_120a has no cvt.rs, so the hash is most of stochastic rounding's cost.)
__device__ __forceinline__ void sr_uniform2(uint64_t seed, uint64_t pair, float& u0, float& u1) {
  const uint32_t k = static_cast<uint32_t>(pair);  // reference masks the int64 pair index to 32 bits
  const uint32_t h = hash_u32(hash_u32(k ^ static_cast<uint32_t>(seed)) + static_cast<uint32_t>(seed >> 32));
  u0 = __fsub_rn(__uint_as_float(0x3F800000u | ((h & 0xFFFFu) << 7)), 1.0f);
  u1 = __fsub_rn(__uint_as_float(0x3F800000u | ((h >> 16) << 7)), 1.0f);
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

// rht_sign_mask computed by a whole warp: lane i < 16 hashes bit i, one ballot assembles the mask.
// One hash per thread instead of sixteen; must be called by all 32 lanes (convergent).
__device__ __forceinline__ uint32_t rht_sign_mask_warp(uint64_t seed) {
  const unsigned lane = threadIdx.x & 31u;
  const bool bit = lane < kBlock && (hash_u32(lane ^ static_cast<uint32_t>(seed)) & 1u);
  return __ballot_sync(0xffffffffu, bit) & ((1u << kBlock) - 1u);
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

// reference.gelu_tanh / reference.gelu_tanh_backward: tanh-approximate GELU and its gradient, one
// IEEE-rounded op at a time (no FMA contraction) so the reference reproduces them bit for bit with
// separate torch ops (tanhf is the same libdevice function torch.tanh uses on CUDA).
constexpr float kGeluBeta = 0.7978845608028654f;  // sqrt(2 / pi)
constexpr float kGeluKappa = 0.044715f;
constexpr float kGelu3Kappa = 0.134145f;          // 3 * kappa

__device__ __forceinline__ float gelu_tanh(float x) {
  const float x3 = __fmul_rn(__fmul_rn(x, x), x);
  const float inner = __fmul_rn(__fadd_rn(x, __fmul_rn(x3, kGeluKappa)), kGeluBeta);
  return __fmul_rn(__fmul_rn(x, 0.5f), __fadd_rn(tanhf(inner), 1.0f));
}

__device__ __forceinline__ float gelu_tanh_backward(float dy, float x) {
  const float x2 = __fmul_rn(x, x);
  const float inner = __fmul_rn(__fadd_rn(x, __fmul_rn(__fmul_rn(x2, x), kGeluKappa)), kGeluBeta);
  const float t = tanhf(inner);
  const float left_d = __fmul_rn(__fadd_rn(t, 1.0f), 0.5f);
  const float right_d = __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(x, 0.5f), __fsub_rn(1.0f, __fmul_rn(t, t))), kGeluBeta),
                                  __fadd_rn(__fmul_rn(x2, kGelu3Kappa), 1.0f));
  return __fmul_rn(dy, __fadd_rn(left_d, right_d));
}

// reference.gelu_erf / gelu_erf_backward: exact GELU (BERT, ViT; torch's default approximate="none"),
// x * 0.5 * (1 + erf(x / sqrt(2))), in torch's order; erff / expf are the libdevice functions torch uses.
constexpr float kSqrt1_2 = 0.7071067811865476f;        // 1 / sqrt(2)
constexpr float kInvSqrt2Pi = 0.3989422804014327f;     // 1 / sqrt(2 pi)

__device__ __forceinline__ float gelu_erf(float x) {
  return __fmul_rn(__fmul_rn(x, 0.5f), __fadd_rn(1.0f, erff(__fmul_rn(x, kSqrt1_2))));
}

__device__ __forceinline__ float gelu_erf_backward(float dy, float x) {
  const float cdf = __fmul_rn(0.5f, __fadd_rn(1.0f, erff(__fmul_rn(x, kSqrt1_2))));
  const float pdf = __fmul_rn(expf(__fmul_rn(__fmul_rn(-0.5f, x), x)), kInvSqrt2Pi);
  return __fmul_rn(dy, __fadd_rn(cdf, __fmul_rn(x, pdf)));
}

// reference.silu / swiglu / swiglu_backward_*: SiLU(g) = g / (1 + exp(-g)) (Llama's SwiGLU MLP computes
// silu(gate) * up), one IEEE-rounded op at a time; expf is the libdevice function torch.exp uses on CUDA.
__device__ __forceinline__ float silu_sigmoid(float g) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g))); }

__device__ __forceinline__ float silu(float g) { return __fdiv_rn(g, __fadd_rn(1.0f, expf(-g))); }

// silu(g) * u
__device__ __forceinline__ float swiglu(float g, float u) { return __fmul_rn(silu(g), u); }

// d/dg of silu(g) * u, times dy: (dy * u) * (s * (1 + g * (1 - s))), s = sigmoid(g)
__device__ __forceinline__ float swiglu_backward_gate(float dy, float g, float u) {
  const float s = silu_sigmoid(g);
  return __fmul_rn(__fmul_rn(dy, u), __fmul_rn(s, __fadd_rn(1.0f, __fmul_rn(g, __fsub_rn(1.0f, s)))));
}

// d/du of silu(g) * u, times dy: dy * silu(g)
__device__ __forceinline__ float swiglu_backward_up(float dy, float g) { return __fmul_rn(dy, silu(g)); }

// reference.rmsnorm_apply for one element: (x * rstd) * gamma.
__device__ __forceinline__ float rmsnorm_apply(float x, float rstd, float gamma) {
  return __fmul_rn(__fmul_rn(x, rstd), gamma);
}

// reference.layernorm_apply for one element: ((x - mean) * rstd) * gamma + beta.
__device__ __forceinline__ float layernorm_apply(float x, float mean, float rstd, float gamma, float beta) {
  return __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(x, mean), rstd), gamma), beta);
}

// Running max of |x| for the global amax that keeps NaN: fmaxf drops NaN operands, which would let a
// NaN input quantize to finite values and corrupt training silently. Once m is NaN it stays NaN, and
// as uint bits a positive NaN is above +inf, so the atomicMax over CTAs keeps it too.
__device__ __forceinline__ float nan_max(float m, float v) { return (v > m || v != v) ? v : m; }

// reference.global_scales: genc = 2688 / amax (1 if amax == 0, NaN if amax is NaN); gdec = 1 / genc.
// A NaN (or inf) anywhere in the tensor thus makes gdec non-finite and every GEMM output it feeds NaN.
__device__ __forceinline__ float global_encode_scale(float amax) {
  if (amax > 0.0f) return __fdiv_rn(kE4M3Max * kE2M1Max, amax);
  return amax == 0.0f ? 1.0f : __int_as_float(0x7fffffff);
}

// reference.quantize for one 16-element block whose abs-max (over the 1x16 or 16x16 block) is bmax.
// Returns the UE4M3 scale byte and the 16 E2M1 codes packed low-nibble-first into `packed`.
// Stochastic rounding uses sr_uniform2(sr_seed, (lin0 + i) / 2), lin0 = linear index of element 0 (even).
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
      float u0, u1;
      sr_uniform2(sr_seed, (lin0 + i) >> 1, u0, u1);
      y0 = e2m1_sr_value(y0, u0);
      y1 = e2m1_sr_value(y1, u1);
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
