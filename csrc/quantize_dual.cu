// Fused "dual" NVFP4 quantization: one read of X [R, C] produces both
//   row output  Q(X)    quantized along C  (fprop operand / dgrad operand)       [R, C/2]
//   col output  Q(X^T)  quantized along R  (wgrad operand), optional 16-point RHT [C, R_pad/2]
// with independent stochastic-rounding seeds, and optional 16x16 2D blocks (weights: the col output
// is then the exact transpose of the row output). R is zero-padded to R_pad for the col output, which
// is what the wgrad GEMM needs (K % 128 == 0).
//
// Each output is bit-identical to nvfp4.reference.quantize on X (row) and on pad(X)^T with the RHT
// applied (col), i.e. to two separate calls of the single-output kernel in quantize.cu.
//
// Structure: a CTA stages a 64x64 tile of X in shared memory (coalesced 16-byte loads), then
//   row phase: thread (b, r) quantizes the 16-element block (row r, block column b),
//   col phase: thread (b, c) quantizes the 16-element block (column c, block row b).
// Both phases read shared memory conflict-free (row stride 65 floats), and in both a warp's lanes
// are 32 consecutive rows / columns, so a 16x16 tile lies in 16 adjacent lanes (4-shuffle 2D max).
// Each phase stages its codes and scales in shared memory and writes whole 32-byte rows with 16-byte
// stores (direct per-thread 8-byte / 1-byte stores were the bottleneck).
// Global amax needs its own pass; it is a plain vectorized max unless the col output uses the RHT,
// in which case the same tile kernel computes both amax(X) and amax(RHT(X^T)). Both amax kernels are
// persistent with one atomic per CTA.
//
// Producer fusion: the tile loader can apply the op that produced X, so that X itself is never
// written to memory (Pro below): LayerNorm / RMSNorm (from x and per-row statistics computed by
// ln_stats_kernel), tanh-GELU (from its input h), the GELU backward (dY * gelu'(h), from dY and h),
// SwiGLU (silu(g) * u, from the gate and up projections) and its two backward halves (the gradients
// w.r.t. g and u, from dY, g and u). The GELUs (tanh or exact erf) can add a per-column bias to their
// input first (the producing linear's bias). The amax pass and the quantize pass both apply it; each
// is bit-identical to the reference op followed by reference.quantize_dual.
//
// Column sums: the quantize pass can also write per-tile column sums of op(X) (Outputs::colsum,
// [tiles along R, C], summed in a fixed order: deterministic, no atomics), from which the host gets
// a bias gradient (sum over rows of the GELU input gradient) without op(X) ever being written.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include "nvfp4_common.cuh"

using OptTensor = c10::optional<torch::Tensor>;

namespace nvfp4 {
namespace dual {

constexpr int TILE = 64;
constexpr int LD = TILE + 1;  // padded row stride (floats): conflict-free row and column access
constexpr int THREADS = 256;

struct Params {
  int R, C;        // input shape
  int R_pad;       // col output row length (R rounded up to the col padding multiple)
  int R_tiles;     // rows covered by the grid (R_pad if col output, else R)
  bool do_row, do_col, row_sr, col_sr, col_rht, block_2d;
  uint64_t row_seed, col_seed;
  uint32_t rht_mask;  // rht_sign_mask(rht_seed)
  // Optional device-resident seeds (int64 scalars). When set they replace row_seed / col_seed /
  // rht_seed, so the seeds can come from a GPU random generator: no host sync, CUDA-graph capturable
  // (every replay reads that replay's values), and traceable by torch.compile.
  const int64_t *row_seed_ptr, *col_seed_ptr, *rht_seed_ptr;
  // Multi-part input (producer op kNone only): X = [part0 | part1 | ...] along C, each part a separate
  // contiguous [R, part_cols] tensor (ProArgs::part), e.g. the q / k / v gradients of a fused qkv
  // projection, quantized without first being concatenated. 0: X is one contiguous tensor.
  int part_cols;
};

// Device seeds: thread i < 3 of a CTA loads seed i at kernel entry, parks it in shared memory after its
// tile loads are issued, and the barrier that follows the tile load publishes it; each warp then builds
// the RHT sign mask with one hash per lane (rht_sign_mask_warp). Per-kernel GPU time is the same as
// with host seeds. (Hashing all sixteen mask bits in every thread was compiled to predicated code that
// ran even without device seeds and made these issue-bound kernels ~10% slower.)
struct Seeds {
  uint64_t row, col;
  uint32_t rht_mask;
};

__device__ __forceinline__ const int64_t* seed_ptr(const Params& p, unsigned i) {
  return i == 0 ? p.row_seed_ptr : i == 1 ? p.col_seed_ptr : i == 2 ? p.rht_seed_ptr : nullptr;
}

// After the barrier that publishes s_seed (convergent: all threads).
__device__ __forceinline__ Seeds resolve_seeds(const Params& p, const int64_t (&s_seed)[3]) {
  return {p.row_seed_ptr ? static_cast<uint64_t>(s_seed[0]) : p.row_seed,
          p.col_seed_ptr ? static_cast<uint64_t>(s_seed[1]) : p.col_seed,
          p.rht_seed_ptr ? rht_sign_mask_warp(static_cast<uint64_t>(s_seed[2])) : p.rht_mask};
}

struct Outputs {
  uint8_t *row_data, *row_scales, *col_data, *col_scales;
  float *row_gdec, *col_gdec;
  float* colsum;  // optional [gridDim.y, C]: per-tile column sums of op(X)
};

// Producer op applied to every loaded element (zero padding stays zero).
enum Pro : int {
  kNone = 0,
  kLayerNorm = 1,      // X = x:  ((x - mean) * rstd) * gamma + beta
  kGelu = 2,           // X = h:  gelu_tanh(h)
  kGeluBwd = 3,        // X = dY: dY * gelu_tanh'(h),          aux = h
  kRmsNorm = 4,        // X = x:  (x * rstd) * gamma
  kSwiglu = 5,         // X = g:  silu(g) * u,                 aux = u
  kSwigluBwdGate = 6,  // X = dY: dY * u * silu'(g),           aux = g, aux2 = u
  kSwigluBwdUp = 7,    // X = dY: dY * silu(g),                aux = g
  kGeluErf = 8,        // X = h:  gelu_erf(h)
  kGeluErfBwd = 9,     // X = dY: dY * gelu_erf'(h),           aux = h
};
// The four GELU variants add the per-column bias ProArgs::beta (if set) to h first; the three SwiGLU
// variants add ProArgs::beta (if set) to g and ProArgs::gamma (if set) to u first (the gate / up biases).

template <typename T>
struct ProArgs {
  const float* mean;   // [R]   LayerNorm
  const float* rstd;   // [R]   LayerNorm / RMSNorm
  const float* gamma;  // [C]   LayerNorm / RMSNorm weight
  const float* beta;   // [C]   LayerNorm bias / bias added to the GELU input (optional)
  const T* aux;        // [R,C] second input (see Pro)
  const T* aux2;       // [R,C] third input (see Pro)
  const T* part[3];    // multi-part input (Params::part_cols)
};

// Per-column bias b (optional) added to v[0..n) in fp32: the bias of the linear that produced v.
template <int n>
__device__ __forceinline__ void add_bias(float* v, const float* b, int gc) {
  if (b == nullptr) return;
#pragma unroll
  for (int i = 0; i < n; ++i) v[i] = __fadd_rn(v[i], __ldg(b + gc + i));
}

__device__ __forceinline__ void unpack8(const uint4& u, float* v) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    v[2 * i] = f.x;
    v[2 * i + 1] = f.y;
  }
}

// One 16-byte vector of T as floats.
template <typename T>
__device__ __forceinline__ void load_vec(const T* __restrict__ src, float* v) {
  const uint4 u = __ldg(reinterpret_cast<const uint4*>(src));
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    unpack8(u, v);
  } else {
    v[0] = __uint_as_float(u.x);
    v[1] = __uint_as_float(u.y);
    v[2] = __uint_as_float(u.z);
    v[3] = __uint_as_float(u.w);
  }
}

// Apply the producer op to the VEC elements X[gr, gc .. gc + VEC).
template <typename T, int PRO>
__device__ __forceinline__ void apply_pro(float* v, const ProArgs<T>& pa, int64_t off, int gr, int gc) {
  constexpr int VEC = 16 / sizeof(T);
  if constexpr (PRO == kLayerNorm) {
    const float m = __ldg(pa.mean + gr), rs = __ldg(pa.rstd + gr);
#pragma unroll
    for (int i = 0; i < VEC; ++i)
      v[i] = layernorm_apply(v[i], m, rs, __ldg(pa.gamma + gc + i), __ldg(pa.beta + gc + i));
  } else if constexpr (PRO == kGelu || PRO == kGeluErf) {
    add_bias<VEC>(v, pa.beta, gc);
#pragma unroll
    for (int i = 0; i < VEC; ++i) v[i] = PRO == kGelu ? gelu_tanh(v[i]) : gelu_erf(v[i]);
  } else if constexpr (PRO == kGeluBwd || PRO == kGeluErfBwd) {
    float h[VEC];
    load_vec(pa.aux + off, h);
    add_bias<VEC>(h, pa.beta, gc);
#pragma unroll
    for (int i = 0; i < VEC; ++i) v[i] = PRO == kGeluBwd ? gelu_tanh_backward(v[i], h[i]) : gelu_erf_backward(v[i], h[i]);
  } else if constexpr (PRO == kRmsNorm) {
    const float rs = __ldg(pa.rstd + gr);
#pragma unroll
    for (int i = 0; i < VEC; ++i) v[i] = rmsnorm_apply(v[i], rs, __ldg(pa.gamma + gc + i));
  } else if constexpr (PRO == kSwiglu) {
    float u[VEC];
    load_vec(pa.aux + off, u);
    add_bias<VEC>(v, pa.beta, gc);
    add_bias<VEC>(u, pa.gamma, gc);
#pragma unroll
    for (int i = 0; i < VEC; ++i) v[i] = swiglu(v[i], u[i]);
  } else if constexpr (PRO == kSwigluBwdGate) {
    float g[VEC], u[VEC];
    load_vec(pa.aux + off, g);
    load_vec(pa.aux2 + off, u);
    add_bias<VEC>(g, pa.beta, gc);
    add_bias<VEC>(u, pa.gamma, gc);
#pragma unroll
    for (int i = 0; i < VEC; ++i) v[i] = swiglu_backward_gate(v[i], g[i], u[i]);
  } else if constexpr (PRO == kSwigluBwdUp) {
    float g[VEC];
    load_vec(pa.aux + off, g);
    add_bias<VEC>(g, pa.beta, gc);
#pragma unroll
    for (int i = 0; i < VEC; ++i) v[i] = swiglu_backward_up(v[i], g[i]);
  }
}

// Stage op(X)[r0:r0+64, c0:c0+64] (zeros outside X) into tile; returns this thread's max |value|.
template <typename T, int PRO>
__device__ __forceinline__ float load_tile(const T* __restrict__ x, const Params& p, const ProArgs<T>& pa, int r0,
                                           int c0, float (*tile)[LD]) {
  constexpr int VEC = 16 / sizeof(T);             // elements per 16-byte load (8 bf16 / 4 fp32)
  constexpr int PER_ROW = TILE / VEC;             // loads per tile row
  constexpr int ITERS = TILE * PER_ROW / THREADS; // 2 (bf16) / 4 (fp32)
  float m = 0.0f;
#pragma unroll
  for (int it = 0; it < ITERS; ++it) {
    const int idx = threadIdx.x + it * THREADS;
    const int row = idx / PER_ROW, col = (idx % PER_ROW) * VEC;
    const int gr = r0 + row, gc = c0 + col;
    float v[VEC];
    if (gr < p.R && gc < p.C) {  // C % 16 == 0, so a vector is either fully inside or fully outside
      const int64_t off = static_cast<int64_t>(gr) * p.C + gc;
      if (PRO == kNone && p.part_cols > 0) {  // part_cols % 16 == 0: a vector lies in one part
        const int part = gc / p.part_cols;
        load_vec(pa.part[part] + static_cast<int64_t>(gr) * p.part_cols + (gc - part * p.part_cols), v);
      } else {
        load_vec(x + off, v);
      }
      apply_pro<T, PRO>(v, pa, off, gr, gc);
    } else {
#pragma unroll
      for (int i = 0; i < VEC; ++i) v[i] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      tile[row][col + i] = v[i];
      m = nan_max(m, fabsf(v[i]));
    }
  }
  return m;
}

__device__ __forceinline__ float warp_max(float m) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = nan_max(m, __shfl_xor_sync(0xffffffffu, m, o));
  return m;
}

// Max over the 16 adjacent lanes holding one 16x16 tile.
__device__ __forceinline__ float max16(float m) {
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  return m;
}

// Max over the whole CTA, delivered to thread 0. Global amax uses one atomicMax per CTA: same-address
// atomics serialize at the L2, and one per warp over thousands of CTAs cost more than the reads.
__device__ __forceinline__ float block_max(float m) {
  __shared__ float red[THREADS / 32];
  m = warp_max(m);
  __syncthreads();  // red may still be read by a previous call
  if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = m;
  __syncthreads();
  if (threadIdx.x < 32) m = warp_max(threadIdx.x < THREADS / 32 ? red[threadIdx.x] : 0.0f);
  return m;
}

// Plain amax over a contiguous tensor (16-byte vectors, grid-stride).
template <typename T>
__global__ void __launch_bounds__(THREADS) amax_flat_kernel(const T* __restrict__ x, int64_t nvec,
                                                            unsigned int* __restrict__ out) {
  constexpr int VEC = 16 / sizeof(T);
  float m = 0.0f;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < nvec;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(x) + i);
    float v[VEC];
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      unpack8(u, v);
    } else {
      v[0] = __uint_as_float(u.x);
      v[1] = __uint_as_float(u.y);
      v[2] = __uint_as_float(u.z);
      v[3] = __uint_as_float(u.w);
    }
#pragma unroll
    for (int k = 0; k < VEC; ++k) m = nan_max(m, fabsf(v[k]));
  }
  m = block_max(m);
  if (threadIdx.x == 0) atomicMax(out, __float_as_uint(m));
}

// amax(op(X)) -> amax[0] and, if p.col_rht, amax(RHT applied to the 16-row column blocks of op(X))
// -> amax[1]. Persistent: each CTA loops over tiles (grid-stride), then contributes one atomic per output.
template <typename T, int PRO>
__global__ void __launch_bounds__(THREADS) amax_tile_kernel(const T* __restrict__ x, const Params p, ProArgs<T> pa,
                                                            unsigned int* __restrict__ amax) {
  __shared__ float tile[TILE][LD];
  __shared__ int64_t s_rht_seed;
  const bool seed_loader = threadIdx.x == 0 && p.rht_seed_ptr != nullptr;
  const int64_t rht_seed = seed_loader ? __ldg(p.rht_seed_ptr) : 0;
  uint32_t rht_mask = p.rht_mask;
  bool first = true;
  const int tiles_c = (p.C + TILE - 1) / TILE;
  const int ntiles = tiles_c * ((p.R + TILE - 1) / TILE);  // padding rows add nothing
  const int b = threadIdx.x / TILE, c = threadIdx.x % TILE;
  float m_plain = 0.0f, m_rht = 0.0f;
  for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
    const int r0 = (t / tiles_c) * TILE, c0 = (t % tiles_c) * TILE;
    __syncthreads();  // previous iteration's column reads are done
    m_plain = nan_max(m_plain, load_tile<T, PRO>(x, p, pa, r0, c0, tile));
    if (!p.col_rht) continue;
    if (first && seed_loader) s_rht_seed = rht_seed;
    __syncthreads();
    if (first) {  // uniform: every thread runs the same iterations
      if (p.rht_seed_ptr) rht_mask = rht_sign_mask_warp(static_cast<uint64_t>(s_rht_seed));
      first = false;
    }
    float v[kBlock];
#pragma unroll
    for (int i = 0; i < kBlock; ++i) v[i] = tile[b * kBlock + i][c];
    rht16(v, rht_mask);
#pragma unroll
    for (int i = 0; i < kBlock; ++i) m_rht = nan_max(m_rht, fabsf(v[i]));
  }
  m_plain = block_max(m_plain);
  m_rht = block_max(m_rht);
  if (threadIdx.x == 0) {
    atomicMax(amax, __float_as_uint(m_plain));
    atomicMax(amax + 1, __float_as_uint(m_rht));
  }
}

// Output staging: a tile produces, per output row, 4 blocks = 32 bytes of codes + 4 scale bytes.
// Writing them straight from the quantizing threads scatters 8-byte / 1-byte stores over 32 rows per
// warp instruction (four warps then complete each 32-byte sector); staging them in shared memory lets
// each sector be written whole by two 16-byte stores.
struct Stage {
  uint2 codes[TILE][4];
  uint8_t scales[TILE][4];
};

// Write the staged outputs of rows [row0, row0 + nrows) whose blocks [blk0, blk0 + nblk) are valid
// (nblk <= 4) into data [*, row_bytes] / scales [*, row_bytes / 8].
__device__ __forceinline__ void store_stage(const Stage& st, uint8_t* __restrict__ data, uint8_t* __restrict__ scales,
                                            int64_t row_bytes, int row0, int nrows, int blk0, int nblk) {
  const int t = threadIdx.x;
  const int64_t sc_row = row_bytes / 8;
  if (t < 2 * TILE) {  // codes: two threads per row, 16 bytes (2 blocks) each
    const int r = t >> 1, j = 2 * (t & 1);
    if (r < nrows && j < nblk) {
      uint8_t* dst = data + static_cast<int64_t>(row0 + r) * row_bytes + (blk0 + j) * (kBlock / 2);
      if (j + 1 < nblk && (row_bytes % 16) == 0) {
        const uint2 a = st.codes[r][j], b = st.codes[r][j + 1];
        *reinterpret_cast<uint4*>(dst) = make_uint4(a.x, a.y, b.x, b.y);
      } else {
        *reinterpret_cast<uint2*>(dst) = st.codes[r][j];
        if (j + 1 < nblk) *reinterpret_cast<uint2*>(dst + 8) = st.codes[r][j + 1];
      }
    }
  } else if (t < 3 * TILE) {  // scales: one thread per row, 4 bytes
    const int r = t - 2 * TILE;
    if (r < nrows) {
      uint8_t* dst = scales + static_cast<int64_t>(row0 + r) * sc_row + blk0;
      if (nblk == 4 && (sc_row % 4) == 0) {
        *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(st.scales[r]);
      } else {
        for (int j = 0; j < nblk; ++j) dst[j] = st.scales[r][j];
      }
    }
  }
}

template <typename T, int PRO>
__global__ void __launch_bounds__(THREADS) dual_quantize_kernel(const T* __restrict__ x, const Params p, ProArgs<T> pa,
                                                                const unsigned int* __restrict__ amax_row,
                                                                const unsigned int* __restrict__ amax_col,
                                                                Outputs o) {
  __shared__ float tile[TILE][LD];
  __shared__ __align__(16) Stage st;
  __shared__ int64_t s_seed[3];
  const int64_t* my_seed_ptr = seed_ptr(p, threadIdx.x);
  const int64_t my_seed = my_seed_ptr ? __ldg(my_seed_ptr) : 0;
  const int r0 = blockIdx.y * TILE, c0 = blockIdx.x * TILE;
  load_tile<T, PRO>(x, p, pa, r0, c0, tile);
  if (my_seed_ptr) s_seed[threadIdx.x] = my_seed;
  __syncthreads();
  const Seeds seeds = resolve_seeds(p, s_seed);
  if (o.colsum != nullptr && threadIdx.x < TILE && c0 + static_cast<int>(threadIdx.x) < p.C) {
    float cs = 0.0f;  // rows past R are zero padding
#pragma unroll 8
    for (int r = 0; r < TILE; ++r) cs = __fadd_rn(cs, tile[r][threadIdx.x]);
    o.colsum[static_cast<int64_t>(blockIdx.y) * p.C + c0 + threadIdx.x] = cs;
  }

  const int b = threadIdx.x / TILE;  // block index within the tile (0..3)
  const int k = threadIdx.x % TILE;  // row (row phase) / column (col phase) within the tile
  const bool first = blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0;
  float v[kBlock];

  if (p.do_row) {
    const float genc = global_encode_scale(__uint_as_float(*amax_row));
    if (first) *o.row_gdec = __fdiv_rn(1.0f, genc);
    const int gr = r0 + k, gb = c0 / kBlock + b;
    float bmax = 0.0f;
#pragma unroll
    for (int i = 0; i < kBlock; ++i) {
      v[i] = tile[k][b * kBlock + i];
      bmax = fmaxf(bmax, fabsf(v[i]));
    }
    if (p.block_2d) bmax = max16(bmax);
    const uint64_t lin0 = static_cast<uint64_t>(gr) * p.C + gb * kBlock;
    st.scales[k][b] = p.row_sr ? quantize16<true>(v, bmax, genc, seeds.row, lin0, st.codes[k][b])
                               : quantize16<false>(v, bmax, genc, 0, lin0, st.codes[k][b]);
    __syncthreads();
    store_stage(st, o.row_data, o.row_scales, p.C / 2, r0, min(TILE, p.R - r0), c0 / kBlock,
                min(4, (p.C - c0) / kBlock));
    if (p.do_col) __syncthreads();  // st is reused below
  }

  if (p.do_col) {
    const float genc = global_encode_scale(__uint_as_float(*amax_col));
    if (first) *o.col_gdec = __fdiv_rn(1.0f, genc);
    const int gc = c0 + k, gb = r0 / kBlock + b;  // logical row / block column of X^T
#pragma unroll
    for (int i = 0; i < kBlock; ++i) v[i] = tile[b * kBlock + i][k];
    if (p.col_rht) rht16(v, seeds.rht_mask);
    float bmax = 0.0f;
#pragma unroll
    for (int i = 0; i < kBlock; ++i) bmax = fmaxf(bmax, fabsf(v[i]));
    if (p.block_2d) bmax = max16(bmax);
    const uint64_t lin0 = static_cast<uint64_t>(gc) * p.R_pad + gb * kBlock;
    st.scales[k][b] = p.col_sr ? quantize16<true>(v, bmax, genc, seeds.col, lin0, st.codes[k][b])
                               : quantize16<false>(v, bmax, genc, 0, lin0, st.codes[k][b]);
    __syncthreads();
    // the last tile row may extend past R_pad (and the last tile column past C)
    store_stage(st, o.col_data, o.col_scales, p.R_pad / 2, c0, min(TILE, p.C - c0), r0 / kBlock,
                min(4, (p.R_pad - r0) / kBlock));
  }
}

// LayerNorm row statistics: mean and rstd = 1 / sqrt(var + eps) (biased variance), one warp per row,
// two passes over the row (the second hits L1). Sums are warp-shuffle trees, so the result is
// deterministic; it matches torch.layer_norm's statistics to fp32 rounding, not bit for bit.
// kRms: RMSNorm's rstd = 1 / sqrt(mean(x^2) + eps) (one pass, no mean written).
template <typename T, bool kRms>
__global__ void __launch_bounds__(THREADS) ln_stats_kernel(const T* __restrict__ x, int R, int C, float eps,
                                                           float* __restrict__ mean, float* __restrict__ rstd) {
  constexpr int VEC = 16 / sizeof(T);
  const int lane = threadIdx.x & 31;
  const int r = blockIdx.x * (THREADS / 32) + (threadIdx.x >> 5);
  if (r >= R) return;
  const T* row = x + static_cast<int64_t>(r) * C;
  float m = 0.0f;
  if constexpr (!kRms) {
    float s = 0.0f;
    for (int c = lane * VEC; c < C; c += 32 * VEC) {
      float v[VEC];
      load_vec(row + c, v);
#pragma unroll
      for (int i = 0; i < VEC; ++i) s = __fadd_rn(s, v[i]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
    m = __fdiv_rn(s, static_cast<float>(C));
  }
  float q = 0.0f;
  for (int c = lane * VEC; c < C; c += 32 * VEC) {
    float v[VEC];
    load_vec(row + c, v);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      const float d = __fsub_rn(v[i], m);
      q = __fadd_rn(q, __fmul_rn(d, d));
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) q = __fadd_rn(q, __shfl_xor_sync(0xffffffffu, q, o));
  if (lane == 0) {
    if constexpr (!kRms) mean[r] = m;
    rstd[r] = __frsqrt_rn(__fadd_rn(__fdiv_rn(q, static_cast<float>(C)), eps));
  }
}

}  // namespace dual
}  // namespace nvfp4

namespace {

using namespace nvfp4;
using namespace nvfp4::dual;

struct SeedTensors {
  OptTensor row, col, rht;
};

const int64_t* seed_ptr(const OptTensor& t, const torch::Tensor& x, const char* name) {
  if (!t.has_value() || !t->defined()) return nullptr;
  TORCH_CHECK(t->is_cuda() && t->device() == x.device() && t->scalar_type() == at::kLong && t->numel() == 1,
              name, " must be a one-element int64 tensor on x's device");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % alignof(int64_t) == 0, name, " must be aligned");
  return t->data_ptr<int64_t>();
}

// Shared host path of all quantize_dual entry points. `pro` selects the producer op; LayerNorm needs
// mean / rstd / gamma / beta, the GELU backward needs `aux` (h).
// `parts` (producer op kNone only): x is parts[0] and the quantized tensor is the concatenation of all parts
// along dim 1 (see Params::part_cols).
std::vector<torch::Tensor> run_dual(const torch::Tensor& x, int pro, const torch::Tensor* aux,
                                    const torch::Tensor* aux2, const float* mean,
                                    const float* rstd, const float* gamma, const float* beta, bool do_row, bool do_col,
                                    bool block_2d, bool row_sr, int64_t row_seed, bool col_sr, int64_t col_seed,
                                    bool col_rht, int64_t rht_seed, int64_t col_pad, const SeedTensors& seeds,
                                    bool col_sums = false, const std::vector<torch::Tensor>* parts = nullptr) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x must be a contiguous 2D CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat, "x must be bf16 or fp32");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x must be 16-byte aligned");
  TORCH_CHECK(do_row || do_col, "nothing to do");
  TORCH_CHECK(col_pad > 0 && col_pad % kBlock == 0, "col_pad must be a positive multiple of 16");
  const c10::cuda::CUDAGuard guard(x.device());

  Params p{};
  p.R = static_cast<int>(x.size(0));
  p.C = static_cast<int>(x.size(1));
  const int nparts = parts ? static_cast<int>(parts->size()) : 1;
  if (nparts > 1) {
    TORCH_CHECK(pro == kNone && nparts <= 3, "at most 3 parts, without a producer op");
    for (const auto& t : *parts)
      TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.scalar_type() == x.scalar_type() && t.dim() == 2 &&
                      t.is_contiguous() && t.sizes() == x.sizes() && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                  "parts must be contiguous, 16-byte aligned tensors of one shape, dtype and device");
    p.part_cols = p.C;
    p.C *= nparts;
  }
  TORCH_CHECK(p.C % kBlock == 0, "x.shape[1] (", p.C, ") must be a multiple of 16");
  TORCH_CHECK(!block_2d || p.R % kBlock == 0, "2D blocks need x.shape[0] % 16 == 0");
  TORCH_CHECK(!(block_2d && col_rht), "2D blocks and the RHT are not combined");
  p.R_pad = static_cast<int>((p.R + col_pad - 1) / col_pad * col_pad);
  p.R_tiles = do_col ? p.R_pad : p.R;
  p.do_row = do_row;
  p.do_col = do_col;
  p.row_sr = row_sr;
  p.col_sr = col_sr;
  p.col_rht = do_col && col_rht;
  p.block_2d = block_2d;
  p.row_seed = static_cast<uint64_t>(row_seed);
  p.col_seed = static_cast<uint64_t>(col_seed);
  p.rht_mask = rht_sign_mask(static_cast<uint64_t>(rht_seed));
  p.row_seed_ptr = seed_ptr(seeds.row, x, "row_seed tensor");
  p.col_seed_ptr = seed_ptr(seeds.col, x, "col_seed tensor");
  p.rht_seed_ptr = seed_ptr(seeds.rht, x, "rht_seed tensor");

  auto u8 = x.options().dtype(at::kByte);
  auto f32 = x.options().dtype(at::kFloat);
  torch::Tensor rd, rs, rg, cd, cs, cg;
  rd = torch::empty({do_row ? p.R : 0, p.C / 2}, u8);
  rs = torch::empty({do_row ? p.R : 0, p.C / kBlock}, u8);
  rg = torch::empty({do_row ? 1 : 0}, f32);
  cd = torch::empty({do_col ? p.C : 0, p.R_pad / 2}, u8);
  cs = torch::empty({do_col ? p.C : 0, p.R_pad / kBlock}, u8);
  cg = torch::empty({do_col ? 1 : 0}, f32);
  const dim3 grid((p.C + TILE - 1) / TILE, (p.R_tiles + TILE - 1) / TILE);
  torch::Tensor csum = torch::zeros({col_sums ? std::max<int>(1, grid.y) : 0, p.C}, f32);
  if (p.R == 0 || p.C == 0) {
    rg.fill_(1.0f);  // empty tensor: amax 0 -> genc 1 -> gdec 1
    cg.fill_(1.0f);
    if (do_col) {  // all-padding col output: codes and scales are zero
      cd.zero_();
      cs.zero_();
    }
    if (col_sums) return {rd, rs, rg, cd, cs, cg, csum};
    return {rd, rs, rg, cd, cs, cg};
  }

  auto amax = torch::zeros({2}, x.options().dtype(at::kInt));
  auto* amax_ptr = reinterpret_cast<unsigned int*>(amax.data_ptr<int>());
  auto stream = at::cuda::getCurrentCUDAStream();
  Outputs o{do_row ? rd.data_ptr<uint8_t>() : nullptr, do_row ? rs.data_ptr<uint8_t>() : nullptr,
            do_col ? cd.data_ptr<uint8_t>() : nullptr, do_col ? cs.data_ptr<uint8_t>() : nullptr,
            do_row ? rg.data_ptr<float>() : nullptr,   do_col ? cg.data_ptr<float>() : nullptr,
            col_sums ? csum.data_ptr<float>() : nullptr};
  const unsigned int* amax_col = p.col_rht ? amax_ptr + 1 : amax_ptr;

  auto run = [&](auto* xp, auto pro_c) {
    using T = std::remove_cv_t<std::remove_pointer_t<decltype(xp)>>;
    constexpr int PRO = decltype(pro_c)::value;
    ProArgs<T> pa{mean, rstd, gamma, beta, aux ? reinterpret_cast<const T*>(aux->data_ptr()) : nullptr,
                  aux2 ? reinterpret_cast<const T*>(aux2->data_ptr()) : nullptr, {nullptr, nullptr, nullptr}};
    if (p.part_cols > 0)
      for (int i = 0; i < nparts; ++i) pa.part[i] = reinterpret_cast<const T*>((*parts)[i].data_ptr());
    // amax passes are persistent: ~4 CTAs per SM, grid-stride
    const int max_ctas = 4 * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    if (p.col_rht || PRO != kNone) {
      const int64_t ntiles = static_cast<int64_t>((p.C + TILE - 1) / TILE) * ((p.R + TILE - 1) / TILE);
      const int agrid = static_cast<int>(std::min<int64_t>(ntiles, max_ctas));
      amax_tile_kernel<T, PRO><<<agrid, THREADS, 0, stream>>>(xp, p, pa, amax_ptr);
    } else {
      const int64_t nvec = x.numel() / (16 / static_cast<int64_t>(sizeof(T)));
      const int agrid = static_cast<int>(std::min<int64_t>((nvec + THREADS - 1) / THREADS, max_ctas));
      for (int i = 0; i < nparts; ++i)  // one pass per part: they share the atomic max
        amax_flat_kernel<T><<<agrid, THREADS, 0, stream>>>(p.part_cols > 0 ? pa.part[i] : xp, nvec, amax_ptr);
    }
    dual_quantize_kernel<T, PRO><<<grid, THREADS, 0, stream>>>(xp, p, pa, amax_ptr, amax_col, o);
  };
  const bool bf16 = x.scalar_type() == at::kBFloat16;
  const auto* xb = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const auto* xf = reinterpret_cast<const float*>(x.data_ptr());
  using CNone = std::integral_constant<int, kNone>;
  using CLn = std::integral_constant<int, kLayerNorm>;
  switch (pro) {
    case kNone:
      if (bf16) run(xb, CNone{}); else run(xf, CNone{});
      break;
    case kLayerNorm:
      if (bf16) run(xb, CLn{}); else run(xf, CLn{});
      break;
    case kGelu:
      TORCH_CHECK(bf16, "the GELU input must be bf16");
      run(xb, std::integral_constant<int, kGelu>{});
      break;
    case kGeluBwd:
      TORCH_CHECK(bf16, "the GELU gradient must be bf16");
      run(xb, std::integral_constant<int, kGeluBwd>{});
      break;
    case kRmsNorm:
      if (bf16) run(xb, std::integral_constant<int, kRmsNorm>{}); else run(xf, std::integral_constant<int, kRmsNorm>{});
      break;
    case kSwiglu:
      TORCH_CHECK(bf16, "the SwiGLU inputs must be bf16");
      run(xb, std::integral_constant<int, kSwiglu>{});
      break;
    case kSwigluBwdGate:
      TORCH_CHECK(bf16, "the SwiGLU gradient must be bf16");
      run(xb, std::integral_constant<int, kSwigluBwdGate>{});
      break;
    case kSwigluBwdUp:
      TORCH_CHECK(bf16, "the SwiGLU gradient must be bf16");
      run(xb, std::integral_constant<int, kSwigluBwdUp>{});
      break;
    case kGeluErf:
      TORCH_CHECK(bf16, "the GELU input must be bf16");
      run(xb, std::integral_constant<int, kGeluErf>{});
      break;
    case kGeluErfBwd:
      TORCH_CHECK(bf16, "the GELU gradient must be bf16");
      run(xb, std::integral_constant<int, kGeluErfBwd>{});
      break;
    default:
      TORCH_CHECK(false, "unknown producer op");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (col_sums) return {rd, rs, rg, cd, cs, cg, csum};
  return {rd, rs, rg, cd, cs, cg};
}

const float* bias_ptr(const OptTensor& bias, const torch::Tensor& x) {
  if (!bias.has_value() || !bias->defined()) return nullptr;
  TORCH_CHECK(bias->is_cuda() && bias->device() == x.device() && bias->scalar_type() == at::kFloat &&
                  bias->is_contiguous() && bias->dim() == 1 && bias->size(0) == x.size(1),
              "bias must be a contiguous fp32 CUDA tensor with x.shape[1] elements");
  return bias->data_ptr<float>();
}

}  // namespace

// Returns {row_data, row_scales, row_gdec, col_data, col_scales, col_gdec}; a disabled output is
// returned as empty tensors.
std::vector<torch::Tensor> nvfp4_quantize_dual(torch::Tensor x, bool do_row, bool do_col, bool block_2d,
                                               bool row_sr, int64_t row_seed, bool col_sr, int64_t col_seed,
                                               bool col_rht, int64_t rht_seed, int64_t col_pad,
                                               const OptTensor& row_seed_t, const OptTensor& col_seed_t,
                                               const OptTensor& rht_seed_t) {
  return run_dual(x, kNone, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, do_row, do_col, block_2d, row_sr,
                  row_seed,
                  col_sr, col_seed, col_rht, rht_seed, col_pad, {row_seed_t, col_seed_t, rht_seed_t});
}

// quantize_dual of the concatenation of 1-3 same-shaped tensors along dim 1 (e.g. the q, k, v gradients of a
// fused qkv projection), read in place. With col_sums, a 7th output holds per-tile column sums ([tiles, C]
// fp32; their sum over dim 0 is the gradient of a bias added to the quantized tensor).
std::vector<torch::Tensor> nvfp4_quantize_dual_parts(std::vector<torch::Tensor> parts, bool do_row, bool do_col,
                                                     bool row_sr, int64_t row_seed, bool col_sr, int64_t col_seed,
                                                     bool col_rht, int64_t rht_seed, int64_t col_pad,
                                                     const OptTensor& row_seed_t, const OptTensor& col_seed_t,
                                                     const OptTensor& rht_seed_t, bool col_sums) {
  TORCH_CHECK(!parts.empty(), "no input");
  return run_dual(parts[0], kNone, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, do_row, do_col, false, row_sr,
                  row_seed, col_sr, col_seed, col_rht, rht_seed, col_pad, {row_seed_t, col_seed_t, rht_seed_t},
                  col_sums, &parts);
}

// LayerNorm fused into the dual quantizer: quantizes y = LayerNorm(x; gamma, beta, eps) without
// writing y. Returns {mean, rstd, <the six quantize_dual outputs>}; mean / rstd ([R] fp32) are what
// the LayerNorm backward needs.
std::vector<torch::Tensor> nvfp4_layernorm_quantize_dual(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                                                         double eps, bool do_row, bool do_col, bool row_sr,
                                                         int64_t row_seed, bool col_sr, int64_t col_seed, bool col_rht,
                                                         int64_t rht_seed, int64_t col_pad, const OptTensor& row_seed_t,
                                                         const OptTensor& col_seed_t,
                                                         const OptTensor& rht_seed_t) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x must be a contiguous 2D CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat, "x must be bf16 or fp32");
  const int64_t R = x.size(0), C = x.size(1);
  for (auto* t : {&gamma, &beta})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kFloat && t->is_contiguous() && t->dim() == 1 &&
                    t->size(0) == C,
                "LayerNorm weight / bias must be contiguous fp32 [C] CUDA tensors");
  const c10::cuda::CUDAGuard guard(x.device());
  auto mean = torch::empty({R}, x.options().dtype(at::kFloat));
  auto rstd = torch::empty({R}, x.options().dtype(at::kFloat));
  if (R > 0 && C > 0) {
    TORCH_CHECK(C % 16 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                "x must be 16-byte aligned with x.shape[1] % 16 == 0");
    const int blocks = static_cast<int>((R + THREADS / 32 - 1) / (THREADS / 32));
    auto stream = at::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == at::kBFloat16)
      ln_stats_kernel<__nv_bfloat16, false><<<blocks, THREADS, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<int>(R), static_cast<int>(C),
          static_cast<float>(eps), mean.data_ptr<float>(), rstd.data_ptr<float>());
    else
      ln_stats_kernel<float, false><<<blocks, THREADS, 0, stream>>>(x.data_ptr<float>(), static_cast<int>(R),
                                                             static_cast<int>(C), static_cast<float>(eps),
                                                             mean.data_ptr<float>(), rstd.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  auto q = run_dual(x, kLayerNorm, nullptr, nullptr, mean.data_ptr<float>(), rstd.data_ptr<float>(),
                    gamma.data_ptr<float>(),
                    beta.data_ptr<float>(), do_row, do_col, false, row_sr, row_seed, col_sr, col_seed, col_rht,
                    rht_seed, col_pad, {row_seed_t, col_seed_t, rht_seed_t});
  q.insert(q.begin(), {mean, rstd});
  return q;
}

// GELU fused into the dual quantizer: quantizes gelu(h + bias) (bf16 h, optional fp32 per-column bias;
// tanh approximation, or the exact erf form if `exact`) without writing it.
std::vector<torch::Tensor> nvfp4_gelu_quantize_dual(torch::Tensor h, bool do_row, bool do_col, bool row_sr,
                                                    int64_t row_seed, bool col_sr, int64_t col_seed, bool col_rht,
                                                    int64_t rht_seed, int64_t col_pad, const OptTensor& row_seed_t,
                                                    const OptTensor& col_seed_t, const OptTensor& rht_seed_t,
                                                    const OptTensor& bias, bool exact) {
  return run_dual(h, exact ? kGeluErf : kGelu, nullptr, nullptr, nullptr, nullptr, nullptr, bias_ptr(bias, h), do_row,
                  do_col, false, row_sr, row_seed,
                  col_sr, col_seed, col_rht, rht_seed, col_pad, {row_seed_t, col_seed_t, rht_seed_t});
}

// GELU backward fused into the dual quantizer: quantizes dh = dy * gelu'(h + bias) (bf16 dy and h) without
// writing it. With col_sums, a 7th output holds per-tile column sums of dh ([tiles, C] fp32; their sum
// over dim 0 is the bias gradient).
std::vector<torch::Tensor> nvfp4_gelu_backward_quantize_dual(torch::Tensor dy, torch::Tensor h, bool do_row,
                                                             bool do_col, bool row_sr, int64_t row_seed, bool col_sr,
                                                             int64_t col_seed, bool col_rht, int64_t rht_seed,
                                                             int64_t col_pad, const OptTensor& row_seed_t,
                                                             const OptTensor& col_seed_t,
                                                             const OptTensor& rht_seed_t, const OptTensor& bias,
                                                             bool exact, bool col_sums) {
  TORCH_CHECK(h.is_cuda() && h.scalar_type() == at::kBFloat16 && h.is_contiguous() && h.sizes() == dy.sizes() &&
                  reinterpret_cast<uintptr_t>(h.data_ptr()) % 16 == 0,
              "h must be a contiguous, 16-byte aligned bf16 tensor shaped like dy");
  return run_dual(dy, exact ? kGeluErfBwd : kGeluBwd, &h, nullptr, nullptr, nullptr, nullptr, bias_ptr(bias, dy),
                  do_row, do_col, false, row_sr, row_seed, col_sr, col_seed, col_rht, rht_seed, col_pad,
                  {row_seed_t, col_seed_t, rht_seed_t}, col_sums);
}

namespace {

void check_like(const torch::Tensor& t, const torch::Tensor& x, const char* what) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous() && t.sizes() == x.sizes() &&
                  t.device() == x.device() && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
              what, " must be a contiguous, 16-byte aligned bf16 tensor shaped like the first input");
}

}  // namespace

// RMSNorm fused into the dual quantizer: quantizes y = (x * rstd) * gamma, rstd = 1 / sqrt(mean(x^2) + eps),
// without writing y. Returns {rstd, <the six quantize_dual outputs>}; rstd ([R] fp32) is what the
// RMSNorm backward needs.
std::vector<torch::Tensor> nvfp4_rmsnorm_quantize_dual(torch::Tensor x, torch::Tensor gamma, double eps, bool do_row,
                                                       bool do_col, bool row_sr, int64_t row_seed, bool col_sr,
                                                       int64_t col_seed, bool col_rht, int64_t rht_seed, int64_t col_pad,
                                                       const OptTensor& row_seed_t, const OptTensor& col_seed_t,
                                                       const OptTensor& rht_seed_t) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x must be a contiguous 2D CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat, "x must be bf16 or fp32");
  const int64_t R = x.size(0), C = x.size(1);
  TORCH_CHECK(gamma.is_cuda() && gamma.scalar_type() == at::kFloat && gamma.is_contiguous() && gamma.dim() == 1 &&
                  gamma.size(0) == C,
              "RMSNorm weight must be a contiguous fp32 [C] CUDA tensor");
  const c10::cuda::CUDAGuard guard(x.device());
  auto rstd = torch::empty({R}, x.options().dtype(at::kFloat));
  if (R > 0 && C > 0) {
    TORCH_CHECK(C % 16 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                "x must be 16-byte aligned with x.shape[1] % 16 == 0");
    const int blocks = static_cast<int>((R + THREADS / 32 - 1) / (THREADS / 32));
    auto stream = at::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == at::kBFloat16)
      ln_stats_kernel<__nv_bfloat16, true><<<blocks, THREADS, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<int>(R), static_cast<int>(C),
          static_cast<float>(eps), nullptr, rstd.data_ptr<float>());
    else
      ln_stats_kernel<float, true><<<blocks, THREADS, 0, stream>>>(x.data_ptr<float>(), static_cast<int>(R),
                                                                   static_cast<int>(C), static_cast<float>(eps), nullptr,
                                                                   rstd.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  auto q = run_dual(x, kRmsNorm, nullptr, nullptr, nullptr, rstd.data_ptr<float>(), gamma.data_ptr<float>(), nullptr,
                    do_row, do_col, false, row_sr, row_seed, col_sr, col_seed, col_rht, rht_seed, col_pad,
                    {row_seed_t, col_seed_t, rht_seed_t});
  q.insert(q.begin(), rstd);
  return q;
}

// SwiGLU fused into the dual quantizer: quantizes silu(g + bias_g) * (u + bias_u) (bf16 gate / up
// projections, optional fp32 per-column biases added in fp32) without writing it.
std::vector<torch::Tensor> nvfp4_swiglu_quantize_dual(torch::Tensor g, torch::Tensor u, bool do_row, bool do_col,
                                                      bool row_sr, int64_t row_seed, bool col_sr, int64_t col_seed,
                                                      bool col_rht, int64_t rht_seed, int64_t col_pad,
                                                      const OptTensor& row_seed_t, const OptTensor& col_seed_t,
                                                      const OptTensor& rht_seed_t, const OptTensor& bias_g,
                                                      const OptTensor& bias_u) {
  check_like(u, g, "u");
  return run_dual(g, kSwiglu, &u, nullptr, nullptr, nullptr, bias_ptr(bias_u, g), bias_ptr(bias_g, g), do_row, do_col,
                  false, row_sr, row_seed, col_sr, col_seed, col_rht, rht_seed, col_pad,
                  {row_seed_t, col_seed_t, rht_seed_t});
}

// SwiGLU backward fused into the dual quantizer: quantizes the gradient w.r.t. the gate projection,
// dy * u * silu'(g) (gate = true), or w.r.t. the up projection, dy * silu(g) (gate = false), from the
// bf16 gradient dy of silu(g) * u, without writing it. The biases (as in nvfp4_swiglu_quantize_dual) are
// added to g and u first; with col_sums a 7th output holds per-tile column sums of the quantized gradient
// (their sum over dim 0 is the gradient of that projection's bias).
std::vector<torch::Tensor> nvfp4_swiglu_backward_quantize_dual(torch::Tensor dy, torch::Tensor g, torch::Tensor u,
                                                               bool gate, bool do_row, bool do_col, bool row_sr,
                                                               int64_t row_seed, bool col_sr, int64_t col_seed,
                                                               bool col_rht, int64_t rht_seed, int64_t col_pad,
                                                               const OptTensor& row_seed_t, const OptTensor& col_seed_t,
                                                               const OptTensor& rht_seed_t, const OptTensor& bias_g,
                                                               const OptTensor& bias_u, bool col_sums) {
  check_like(g, dy, "g");
  if (gate) check_like(u, dy, "u");
  return run_dual(dy, gate ? kSwigluBwdGate : kSwigluBwdUp, &g, gate ? &u : nullptr, nullptr, nullptr,
                  gate ? bias_ptr(bias_u, dy) : nullptr, bias_ptr(bias_g, dy), do_row, do_col, false, row_sr, row_seed,
                  col_sr, col_seed, col_rht, rht_seed, col_pad, {row_seed_t, col_seed_t, rht_seed_t}, col_sums);
}
