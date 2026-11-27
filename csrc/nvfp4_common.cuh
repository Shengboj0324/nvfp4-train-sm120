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

// reference.sr_key / sr_bits: 8 random bits per element. Elements 4q .. 4q + 3 take bytes 0 .. 3 of
// hash(q + key), key = hash(lo ^ hash(hi)) of the 64-bit seed (computed once per block; one hash per four
// elements: sm_120a has no cvt.rs, so the hash is a large part of stochastic rounding's cost).
__device__ __forceinline__ uint32_t sr_key(uint64_t seed) {
  return hash_u32(hash_u32(static_cast<uint32_t>(seed >> 32)) ^ static_cast<uint32_t>(seed));
}

// reference.e2m1_round_stochastic, as a float grid value (the RN convert after it is exact): magnitudes below 1
// are lifted into [1, 2), where E2M1's 0.5 spacing is one mantissa bit; the random byte r8 plus half a unit of
// it (unbiased few-bit "SRF" rounding) is added below the kept mantissa bit and the rest truncated (a carry
// moves to the next binade on its own). P(up) = round(256 f) / 256 for the position f within the grid step.
__device__ __forceinline__ float e2m1_sr_value(float y, uint32_t r8) {
  const float a = fabsf(y);
  const float off = a < 1.0f ? 1.0f : 0.0f;
  const uint32_t b = (__float_as_uint(__fadd_rn(a, off)) + ((r8 & 0xFFu) << 14) + (1u << 13)) & 0xFFC00000u;
  const float r = fminf(__fsub_rn(__uint_as_float(b), off), kE2M1Max);
  return __uint_as_float(__float_as_uint(r) | (__float_as_uint(y) & 0x80000000u));
}

// reference.rht_signs: element i is negated iff bit 0 of hash(i ^ seed) is set. Computed once on the
// host as a 16-bit mask (bit i set = negate element i).
__host__ __device__ __forceinline__ uint32_t rht_sign_mask(uint64_t seed) {
  uint32_t m = 0;
  for (int i = 0; i < kBlock; ++i) m |= (hash_u32(static_cast<uint32_t>(i) ^ static_cast<uint32_t>(seed)) & 1u) << i;
  return m;
}

// reference.rht_signs(seed, 64): the 64 signs of the 64-point RHT (the first 16 are rht_sign_mask's).
__host__ __device__ __forceinline__ uint64_t rht_sign_mask64(uint64_t seed) {
  uint64_t m = 0;
  for (int i = 0; i < 64; ++i)
    m |= static_cast<uint64_t>(hash_u32(static_cast<uint32_t>(i) ^ static_cast<uint32_t>(seed)) & 1u) << i;
  return m;
}

// MS-EDEN on the dgrad operands: the row output's 64-point RHT takes its signs from the row SR seed XOR this
// salt (reference.ROW_RHT_SALT), so the dgrad weight operand can be rotated identically from the same seed.
constexpr uint64_t kRowRhtSalt = 0x1E3779B97F4A7C15ull;

// rht_sign_mask64 computed by a whole warp (two ballots; convergent).
__device__ __forceinline__ uint64_t rht_sign_mask64_warp(uint64_t seed) {
  const unsigned lane = threadIdx.x & 31u;
  const uint32_t lo = __ballot_sync(0xffffffffu, hash_u32(lane ^ static_cast<uint32_t>(seed)) & 1u);
  const uint32_t hi = __ballot_sync(0xffffffffu, hash_u32((lane + 32u) ^ static_cast<uint32_t>(seed)) & 1u);
  return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
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

// reference.global_scales: genc = 2688 / amax (1536 / amax with 4/6 block scales; 1 if amax == 0, NaN if amax
// is NaN); gdec = 1 / genc. A NaN (or inf) anywhere in the tensor thus makes gdec non-finite and every GEMM
// output it feeds NaN.
constexpr float kFourSixE4M3Max = 256.0f;
__device__ __forceinline__ float global_encode_scale(float amax, bool four_six = false) {
  if (amax > 0.0f) return __fdiv_rn((four_six ? kFourSixE4M3Max : kE4M3Max) * kE2M1Max, amax);
  return amax == 0.0f ? 1.0f : __int_as_float(0x7fffffff);
}

// Host-side recipe switch for "Four Over Six" block scales on every round-to-nearest NVFP4 output
// (nvfp4.ops.set_four_six; read when a quantizer is launched). Defined in quantize.cu.
bool four_six_enabled();
// Host-side recipe switch for MS-EDEN (nvfp4.ops.set_ms_eden): the dual quantizer's RHT outputs use a 64-point
// RHT, and its stochastically rounded RHT outputs (the wgrad gradient operand) MS-EDEN instead of element SR.
bool ms_eden_enabled();
bool ms_eden_dgrad_enabled();

// MS-EDEN (Quartet II, arXiv 2601.22813): round-to-nearest with the block max mapped to s = 6 * 16/17 / 0.93
// (clipping slightly; minimises the MSE of N(0, 1)), tensor scale genc = s * 256 / amax (headroom for the
// correction), then the per-block EDEN factor S = <y, y> / <y, Q(y)> folded into the E4M3 block scale with
// stochastic rounding. Unbiased over the RHT signs and the scale rounding; no element-wise random numbers.
constexpr float kEdenS = 6.07210636138916f;      // fp32 of (1 / 0.93) * 6 * 16 / 17
constexpr float kEdenNum = 1554.459228515625f;   // fp32 of kEdenS * 256
__device__ __forceinline__ float eden_encode_scale(float amax) {
  if (amax > 0.0f) return __fdiv_rn(kEdenNum, amax);
  return amax == 0.0f ? 1.0f : __int_as_float(0x7fffffff);
}

// reference.quantize for one 16-element block whose abs-max (over the 1x16 or 16x16 block) is bmax.
// Returns the UE4M3 scale byte and the 16 E2M1 codes packed low-nibble-first into `packed`.
// Stochastic rounding uses sr_bits(sr_seed, lin0 + i), lin0 = linear index of element 0 (a multiple of 4).
template <bool kStochastic>
__device__ __forceinline__ uint8_t quantize16(const float (&v)[kBlock], float bmax, float genc, uint64_t sr_seed,
                                              uint64_t lin0, uint2& packed) {
  const float s_target = fminf(__fmul_rn(__fdiv_rn(bmax, kE2M1Max), genc), kE4M3Max);
  const uint8_t s = e4m3_from_float(s_target);
  const float sf = e4m3_to_float(s);
  const float enc = sf > 0.0f ? __fdiv_rn(genc, sf) : 0.0f;
  uint32_t lo = 0, hi = 0;
  uint32_t key = 0, h = 0;
  if constexpr (kStochastic) key = sr_key(sr_seed);
#pragma unroll
  for (int i = 0; i < kBlock; i += 2) {
    float y0 = __fmul_rn(v[i], enc), y1 = __fmul_rn(v[i + 1], enc);
    if constexpr (kStochastic) {  // round onto the grid first; the convert below is then exact
      if ((i & 3) == 0) h = hash_u32(static_cast<uint32_t>((lin0 + i) >> 2) + key);
      const uint32_t r = h >> (8 * (i & 3));
      y0 = e2m1_sr_value(y0, r);
      y1 = e2m1_sr_value(y1, r >> 8);
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

// Magnitude of a 3-bit E2M1 code: 0, 0.5, then (1 + m / 2) * 2^(e - 1) for code = 2 e + m.
__device__ __forceinline__ float e2m1_mag(uint32_t c) {
  return c >= 2u ? __uint_as_float(((126u + (c >> 1)) << 23) | ((c & 1u) << 22)) : (c ? 0.5f : 0.0f);
}

// Residual of a quantized block in the input's units, as reference.quantize_dual(row_terms=2) computes it:
// r_i = v_i - (q_i * sf) * gdec, q_i the decoded E2M1 code (sign bit 3), sf the decoded scale byte `s`. Returns
// max |r_i|.
__device__ __forceinline__ float residual16(const float (&v)[kBlock], uint8_t s, const uint2& packed, float gdec,
                                            float (&r)[kBlock]) {
  const float sf = e4m3_to_float(s);
  float m = 0.0f;
#pragma unroll
  for (int i = 0; i < kBlock; ++i) {
    const uint32_t c = ((i < 8 ? packed.x : packed.y) >> (4 * (i & 7))) & 15u;
    const float mag = e2m1_mag(c & 7u);
    const float d = __fmul_rn(__fmul_rn((c & 8u) ? -mag : mag, sf), gdec);
    r[i] = __fsub_rn(v[i], d);
    m = fmaxf(m, fabsf(r[i]));
  }
  return m;
}

// reference._rtn_candidate: round-to-nearest with the block scale target bdiv * genc (bdiv = bmax / 6 or
// bmax / 4); returns the scale byte, the codes, and the squared error sum((|y| - q)^2) * sf^2 (summed in
// element order).
__device__ __forceinline__ uint8_t rtn16_err(const float (&v)[kBlock], float bdiv, float genc, uint2& packed,
                                            float& err) {
  const float s_target = fminf(__fmul_rn(bdiv, genc), kE4M3Max);
  const uint8_t s = e4m3_from_float(s_target);
  const float sf = e4m3_to_float(s);
  const float enc = sf > 0.0f ? __fdiv_rn(genc, sf) : 0.0f;
  uint32_t lo = 0, hi = 0;
  float e = 0.0f;
#pragma unroll
  for (int i = 0; i < kBlock; i += 2) {
    const float y0 = __fmul_rn(v[i], enc), y1 = __fmul_rn(v[i + 1], enc);
    const uint32_t b = cvt_e2m1x2_rn(y0, y1);
    const float d0 = __fsub_rn(fabsf(y0), e2m1_mag(b & 7u));
    const float d1 = __fsub_rn(fabsf(y1), e2m1_mag((b >> 4) & 7u));
    e = __fadd_rn(e, __fmul_rn(d0, d0));
    e = __fadd_rn(e, __fmul_rn(d1, d1));
    if (i < 8)
      lo |= b << (4 * i);
    else
      hi |= b << (4 * (i - 8));
  }
  packed = make_uint2(clear_negative_zeros(lo), clear_negative_zeros(hi));
  err = __fmul_rn(e, __fmul_rn(sf, sf));
  return s;
}

// reference.quantize(eden=True) for one block: RTN codes at the MS-EDEN block scale, then the scale replaced by
// SR_e4m3(S * sf) with S = sum(y^2) / sum(y q) (summed in element order; S = 1 if the codes are all zero). The
// scale's stochastic rounding uses the random byte sr_bits(seed, blk) (blk = block linear index) on the E4M3
// grid (3 mantissa bits; the SRF half-unit offset as for E2M1), and round-to-nearest below E4M3's normal range.
__device__ __forceinline__ uint8_t quantize16_eden(const float (&v)[kBlock], float bmax, float genc, uint64_t seed,
                                                  uint64_t blk, uint2& packed) {
  const float s_target = fminf(__fmul_rn(__fdiv_rn(bmax, kEdenS), genc), kE4M3Max);
  const float sf = e4m3_to_float(e4m3_from_float(s_target));
  const float enc = sf > 0.0f ? __fdiv_rn(genc, sf) : 0.0f;
  uint32_t lo = 0, hi = 0;
  float yy = 0.0f, yq = 0.0f;
#pragma unroll
  for (int i = 0; i < kBlock; i += 2) {
    const float y0 = __fmul_rn(v[i], enc), y1 = __fmul_rn(v[i + 1], enc);
    const uint32_t b = cvt_e2m1x2_rn(y0, y1);
    const float m0 = e2m1_mag(b & 7u), m1 = e2m1_mag((b >> 4) & 7u);
    const float q0 = (b & 8u) ? -m0 : m0, q1 = (b & 0x80u) ? -m1 : m1;
    yy = __fadd_rn(yy, __fmul_rn(y0, y0));
    yq = __fadd_rn(yq, __fmul_rn(y0, q0));
    yy = __fadd_rn(yy, __fmul_rn(y1, y1));
    yq = __fadd_rn(yq, __fmul_rn(y1, q1));
    if (i < 8)
      lo |= b << (4 * i);
    else
      hi |= b << (4 * (i - 8));
  }
  packed = make_uint2(clear_negative_zeros(lo), clear_negative_zeros(hi));
  const float corr = yq > 0.0f ? __fdiv_rn(yy, yq) : 1.0f;
  float t = __fmul_rn(corr, sf);
  if (t >= 0.015625f) {  // E4M3 normal range: SR onto its 3-mantissa-bit grid
    const uint32_t h = hash_u32(static_cast<uint32_t>(blk >> 2) + sr_key(seed));
    const uint32_t r8 = (h >> (8 * (blk & 3u))) & 0xFFu;
    t = __uint_as_float((__float_as_uint(t) + (r8 << 12) + (1u << 11)) & 0xFFF00000u);
  }
  return e4m3_from_float(fminf(t, kE4M3Max));
}

// reference.quantize(four_six=True) for one block (round-to-nearest): the candidate with the block max mapped
// to 4 replaces the standard one (max mapped to 6) when its squared error is strictly smaller. genc must come
// from global_encode_scale(amax, true).
__device__ __forceinline__ uint8_t quantize16_four_six(const float (&v)[kBlock], float bmax, float genc,
                                                      uint2& packed) {
  uint2 p6, p4;
  float e6, e4;
  const uint8_t s6 = rtn16_err(v, __fdiv_rn(bmax, kE2M1Max), genc, p6, e6);
  const uint8_t s4 = rtn16_err(v, __fdiv_rn(bmax, 4.0f), genc, p4, e4);
  const bool four = e4 < e6;
  packed = four ? p4 : p6;
  return four ? s4 : s6;
}

}  // namespace nvfp4
