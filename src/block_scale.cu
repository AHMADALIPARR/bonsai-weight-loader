// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Block-scale output stage (standalone) and fp32 tiled GEMM with the block
// scale fused into its epilogue.
//
// Both kernels load the block-scale factors that cover their output tile into
// __shared__ once, __syncthreads(), and reuse them for every element.
#include <stdexcept>

#include "bonsai/cuda_check.cuh"
#include "bonsai/kernels.cuh"

namespace bonsai {

namespace {

// 4-wide vector access: float4, 2 x half2, 2 x nv_bfloat162 (8-byte transaction).
template <typename T>
struct Vec4;

template <>
struct Vec4<float> {
  static __device__ __forceinline__ float4 load(const float* p) {
    return __ldg(reinterpret_cast<const float4*>(p));
  }
  static __device__ __forceinline__ void store(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
  static __device__ __forceinline__ float to_f(float v) { return v; }
  static __device__ __forceinline__ float from_f(float v) { return v; }
};

template <>
struct Vec4<__half> {
  static __device__ __forceinline__ float4 load(const __half* p) {
    const uint2 raw = __ldg(reinterpret_cast<const uint2*>(p));
    const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&raw.x));
    const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
    return make_float4(lo.x, lo.y, hi.x, hi.y);
  }
  static __device__ __forceinline__ void store(__half* p, float4 v) {
    const __half2 lo = __floats2half2_rn(v.x, v.y);
    const __half2 hi = __floats2half2_rn(v.z, v.w);
    uint2 raw;
    raw.x = *reinterpret_cast<const unsigned*>(&lo);
    raw.y = *reinterpret_cast<const unsigned*>(&hi);
    *reinterpret_cast<uint2*>(p) = raw;
  }
  static __device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
  static __device__ __forceinline__ __half from_f(float v) { return __float2half_rn(v); }
};

template <>
struct Vec4<__nv_bfloat16> {
  static __device__ __forceinline__ float4 load(const __nv_bfloat16* p) {
    const uint2 raw = __ldg(reinterpret_cast<const uint2*>(p));
    const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
    const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
    return make_float4(lo.x, lo.y, hi.x, hi.y);
  }
  static __device__ __forceinline__ void store(__nv_bfloat16* p, float4 v) {
    const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
    const __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
    uint2 raw;
    raw.x = *reinterpret_cast<const unsigned*>(&lo);
    raw.y = *reinterpret_cast<const unsigned*>(&hi);
    *reinterpret_cast<uint2*>(p) = raw;
  }
  static __device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ __nv_bfloat16 from_f(float v) { return __float2bfloat16_rn(v); }
};

// Loads the scale sub-matrix covering output rows [m0, m1) and cols [n0, n1)
// into smem (row-major, cover_cols wide). Must be followed by __syncthreads().
__device__ __forceinline__ void load_scale_cover(float* smem, const float* __restrict__ scales,
                                                 int m0, int m1, int n0, int n1, int br, int bc,
                                                 int scale_cols, int* cover_cols) {
  const int sr0 = m0 / br, sr1 = (m1 - 1) / br;
  const int sc0 = n0 / bc, sc1 = (n1 - 1) / bc;
  const int cc = sc1 - sc0 + 1;
  const int total = (sr1 - sr0 + 1) * cc;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int nthreads = blockDim.x * blockDim.y;
  for (int i = tid; i < total; i += nthreads) {
    const int r = i / cc, c = i - r * cc;
    smem[i] = __ldg(scales + (size_t)(sr0 + r) * scale_cols + sc0 + c);
  }
  *cover_cols = cc;
}

}  // namespace

template <typename T>
__global__ void __launch_bounds__(256, 4)
    apply_block_scale_stage(const T* __restrict__ in, T* __restrict__ out,
                            const float* __restrict__ scales, BlockScaleShape s, int vec_ok) {
  extern __shared__ float s_scale[];
  const int m0 = blockIdx.y * kBsTileM, n0 = blockIdx.x * kBsTileN;
  const int m1 = min(m0 + kBsTileM, s.M), n1 = min(n0 + kBsTileN, s.N);
  int cc;
  load_scale_cover(s_scale, scales, m0, m1, n0, n1, s.block_rows, s.block_cols, s.scale_cols(), &cc);
  __syncthreads();

  const int c = n0 + (threadIdx.x & 63) * 4;  // 4 consecutive columns per thread
  if (c >= n1) return;
  const int sc0 = n0 / s.block_cols, sr0 = m0 / s.block_rows;
  const int cs0 = c / s.block_cols - sc0;
  const int cs1 = min(c + 1, n1 - 1) / s.block_cols - sc0;
  const int cs2 = min(c + 2, n1 - 1) / s.block_cols - sc0;
  const int cs3 = min(c + 3, n1 - 1) / s.block_cols - sc0;

#pragma unroll
  for (int i = 0; i < kBsTileM / 4; ++i) {
    const int r = m0 + (threadIdx.x >> 6) + 4 * i;
    if (r >= m1) break;
    const float* srow = s_scale + (r / s.block_rows - sr0) * cc;
    const size_t base = (size_t)r * s.N + c;
    if (vec_ok) {  // N % 4 == 0 and 16/8-byte aligned rows: c + 3 < N here
      float4 v = Vec4<T>::load(in + base);
      v.x *= srow[cs0];
      v.y *= srow[cs1];
      v.z *= srow[cs2];
      v.w *= srow[cs3];
      Vec4<T>::store(out + base, v);
    } else {  // scalar tail path
      const int n_here = min(4, n1 - c);
      const int cs[4] = {cs0, cs1, cs2, cs3};
#pragma unroll
      for (int j = 0; j < 4; ++j)
        if (j < n_here) out[base + j] = Vec4<T>::from_f(Vec4<T>::to_f(in[base + j]) * srow[cs[j]]);
    }
  }
}

namespace {

inline int scale_cover_floats(int tile_m, int tile_n, int br, int bc) {
  // worst case over tile positions
  return ((tile_m - 1) / br + 2) * ((tile_n - 1) / bc + 2);
}

template <typename T>
int vec_alignment_ok(const T* in, const T* out, int N) {
  const uintptr_t need = sizeof(T) * 4;
  return (N % 4 == 0) && (reinterpret_cast<uintptr_t>(in) % need == 0) &&
         (reinterpret_cast<uintptr_t>(out) % need == 0);
}

void check_shape(const BlockScaleShape& s) {
  if (s.M <= 0 || s.N <= 0 || s.block_rows <= 0 || s.block_cols <= 0)
    throw std::invalid_argument("bad block-scale shape");
}

}  // namespace

template <typename T>
void launch_apply_block_scale(const T* in, T* out, const float* scales, BlockScaleShape s,
                              cudaStream_t stream) {
  check_shape(s);
  const dim3 grid((s.N + kBsTileN - 1) / kBsTileN, (s.M + kBsTileM - 1) / kBsTileM);
  const size_t smem = sizeof(float) * scale_cover_floats(kBsTileM, kBsTileN, s.block_rows, s.block_cols);
  apply_block_scale_stage<T><<<grid, 256, smem, stream>>>(in, out, scales, s, vec_alignment_ok(in, out, s.N));
  BONSAI_CUDA_LAUNCH();
}

template void launch_apply_block_scale<float>(const float*, float*, const float*, BlockScaleShape, cudaStream_t);
template void launch_apply_block_scale<__half>(const __half*, __half*, const float*, BlockScaleShape, cudaStream_t);
template void launch_apply_block_scale<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, const float*,
                                                      BlockScaleShape, cudaStream_t);

// ---- fp32 tiled GEMM, block scale fused into the epilogue ---------------------
// 256 threads, 64x64 output tile, each thread owns a 4x4 accumulator block.
template <bool kFused>
__global__ void __launch_bounds__(256, 4)
    gemm_f32_block_scale(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                         int M, int N, int K, const float* __restrict__ scales, int br, int bc, int vec_ok) {
  __shared__ float As[kGemmBK][kGemmBM + 4];
  __shared__ float Bs[kGemmBK][kGemmBN + 4];
  extern __shared__ float s_scale[];
  const int tid = threadIdx.x;
  const int m0 = blockIdx.y * kGemmBM, n0 = blockIdx.x * kGemmBN;
  const int tr = (tid >> 4) * 4, tc = (tid & 15) * 4;

  int cc = 0;
  if (kFused) {  // scales loaded once per tile; the __syncthreads in the K loop publishes them
    load_scale_cover(s_scale, scales, m0, min(m0 + kGemmBM, M), n0, min(n0 + kGemmBN, N), br, bc,
                     (N + bc - 1) / bc, &cc);
  }

  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += kGemmBK) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = tid + i * 256;
      const int ar = idx >> 4, ak = idx & 15;  // A tile 64 x 16
      const int gm = m0 + ar, gk = k0 + ak;
      As[ak][ar] = (gm < M && gk < K) ? __ldg(A + (size_t)gm * K + gk) : 0.0f;
      const int bk = idx >> 6, bn = idx & 63;  // B tile 16 x 64
      const int gk2 = k0 + bk, gn = n0 + bn;
      Bs[bk][bn] = (gk2 < K && gn < N) ? __ldg(B + (size_t)gk2 * N + gn) : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kGemmBK; ++k) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = As[k][tr + i];
#pragma unroll
      for (int j = 0; j < 4; ++j) b[j] = Bs[k][tc + j];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }
  if (kFused && K == 0) __syncthreads();

  const int sr0 = m0 / br, sc0 = n0 / bc;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int gm = m0 + tr + i;
    if (gm >= M) break;
    const int gn = n0 + tc;
    if (kFused) {  // epilogue: scale the accumulator registers right before the store
      const float* srow = s_scale + (gm / br - sr0) * cc;
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[i][j] *= srow[min(gn + j, N - 1) / bc - sc0];
    }
    float* crow = C + (size_t)gm * N + gn;
    if (vec_ok && gn + 3 < N) {
      *reinterpret_cast<float4*>(crow) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j)
        if (gn + j < N) crow[j] = acc[i][j];
    }
  }
}

void launch_gemm_f32(const float* A, const float* B, float* C, int M, int N, int K, const float* scales,
                     int block_rows, int block_cols, bool fused, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K < 0) throw std::invalid_argument("bad GEMM shape");
  const dim3 grid((N + kGemmBN - 1) / kGemmBN, (M + kGemmBM - 1) / kGemmBM);
  const int vec_ok = (N % 4 == 0) && (reinterpret_cast<uintptr_t>(C) % 16 == 0);
  if (fused) {
    if (!scales || block_rows <= 0 || block_cols <= 0) throw std::invalid_argument("fused GEMM needs scales");
    const size_t smem = sizeof(float) * scale_cover_floats(kGemmBM, kGemmBN, block_rows, block_cols);
    gemm_f32_block_scale<true><<<grid, 256, smem, stream>>>(A, B, C, M, N, K, scales, block_rows,
                                                            block_cols, vec_ok);
  } else {
    gemm_f32_block_scale<false><<<grid, 256, 0, stream>>>(A, B, C, M, N, K, nullptr, 1, 1, vec_ok);
  }
  BONSAI_CUDA_LAUNCH();
}

}  // namespace bonsai
