// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Ternary / INT4 packed-weight tiled GEMM: C[M][N] = scale(m,n) * A[M][K] . W[N][K]^T
//
// A is fp32 row-major. W ("B") is N packed GGUF rows of K weights, PTQ1_0
// (ternary, 128 per 28-byte block) or Q4_0 (32 per 18-byte block); see
// docs/packing.md. Block (16, 16) = 256 threads; each thread owns two output
// rows (ty, ty + 1) of one column tx, so a CTA computes a 32 x 16 tile.
#include <stdexcept>

#include "bonsai/cuda_check.cuh"
#include "bonsai/ggml_types.hpp"
#include "bonsai/kernels.cuh"
#include "bonsai/quant.hpp"

namespace bonsai {

struct Int4GemmParams {
  uint32_t format;
  const float* scales;  // [ceil(M/br)][ceil(N/bc)] or null
  int block_rows, block_cols, scale_cols;
  int row_bytes;  // bytes per packed row of K weights
};

__constant__ Int4GemmParams c_int4;

__global__ void __launch_bounds__(256, 4)
    custom_int4_tiled_gemm_kernel(const void* __restrict__ A, const void* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K) {
  __shared__ float smem_A[kI4TileM][kI4BK + 1];
  __shared__ float smem_B[kI4BK][kI4TileN + 1];
  __shared__ float s_scale[kI4TileM * kI4TileN];

  const float* __restrict__ Af = static_cast<const float*>(A);
  const uint8_t* __restrict__ Bq = static_cast<const uint8_t*>(B);
  const int tx = threadIdx.x;
  const int ty = threadIdx.y * 2;
  const int tid = threadIdx.y * 16 + threadIdx.x;
  const int m0 = blockIdx.y * kI4TileM, n0 = blockIdx.x * kI4TileN;
  const uint32_t format = c_int4.format;
  const int row_bytes = c_int4.row_bytes;

  // Output block scales for this tile: loaded once, published by the first
  // __syncthreads of the K loop, reused in the epilogue.
  const bool has_scale = c_int4.scales != nullptr;
  int sc0 = 0, sr0 = 0, cc = 1;
  if (has_scale) {
    const int br = c_int4.block_rows, bc = c_int4.block_cols;
    sr0 = m0 / br;
    sc0 = n0 / bc;
    const int sr1 = (min(m0 + kI4TileM, M) - 1) / br, sc1 = (min(n0 + kI4TileN, N) - 1) / bc;
    cc = sc1 - sc0 + 1;
    const int total = (sr1 - sr0 + 1) * cc;
    for (int i = tid; i < total; i += 256) {
      const int r = i / cc, c = i - r * cc;
      s_scale[i] = __ldg(c_int4.scales + (size_t)(sr0 + r) * c_int4.scale_cols + sc0 + c);
    }
  }

  float accum_fragment[2] = {0.0f, 0.0f};

  for (int k0 = 0; k0 < K; k0 += kI4BK) {
    // A tile: 32 x 64 fp32, 8 per thread, coalesced along K.
#pragma unroll
    for (int i = 0; i < (kI4TileM * kI4BK) / 256; ++i) {
      const int idx = tid + i * 256;
      const int r = idx >> 6, k = idx & 63;
      const int gm = m0 + r;
      smem_A[r][k] = gm < M ? __ldg(Af + (size_t)gm * K + k0 + k) : 0.0f;
    }
    // B tile: 16 packed rows x 64 weights, unpacked (times group scale d) into
    // smem_B[k][n]. Thread -> row n = tid / 16, elements kk = lane + 16 * i.
    {
      const int n = tid >> 4, lane = tid & 15;
      const int gn = n0 + n;
      if (gn < N) {
        const uint8_t* row = Bq + (size_t)gn * row_bytes;
        if (format == GGML_PTQ1_0) {
          const uint8_t* blk = row + (size_t)(k0 >> 7) * kPtq1BlockBytes;  // 64 | 128: one block
          const float d = ptq1_0_scale(blk);
          const int e0 = k0 & 127;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int kk = lane + 16 * i;
            smem_B[kk][n] = (float)ptq1_0_trit(blk, e0 + kk) * d;
          }
        } else {  // Q4_0: two 32-weight blocks per 64-wide K tile
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int kk = lane + 16 * i;
            const uint8_t* blk = row + (size_t)((k0 + kk) >> 5) * kQ40BlockBytes;
            smem_B[kk][n] = (float)q4_0_code(blk, (k0 + kk) & 31) * q4_0_scale(blk);
          }
        }
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) smem_B[lane + 16 * i][n] = 0.0f;
      }
    }
    __syncthreads();

#pragma unroll 4
    for (int k = 0; k < kI4BK; ++k) {
      float b_val = smem_B[k][tx];
      accum_fragment[0] += smem_A[ty][k] * b_val;
      accum_fragment[1] += smem_A[ty + 1][k] * b_val;
    }
    __syncthreads();
  }
  if (K == 0) __syncthreads();

  // Epilogue: block scale applied to the accumulators right before the store.
  const int gn = n0 + tx;
  if (gn >= N) return;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int gm = m0 + ty + i;
    if (gm >= M) break;
    float v = accum_fragment[i];
    if (has_scale)
      v *= s_scale[(gm / c_int4.block_rows - sr0) * cc + gn / c_int4.block_cols - sc0];
    C[(size_t)gm * N + gn] = v;
  }
}

void launch_custom_int4_tiled_gemm(const float* A, const uint8_t* B, float* C, int M, int N, int K,
                                   uint32_t format, const float* scales, int block_rows,
                                   int block_cols, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0 || K % kI4BK != 0) throw std::invalid_argument("K must be a multiple of 64");
  Int4GemmParams p{};
  p.format = format;
  if (format == GGML_PTQ1_0) {
    if (K % kPtq1BlockElems != 0) throw std::invalid_argument("PTQ1_0 needs K % 128 == 0");
    p.row_bytes = K / kPtq1BlockElems * kPtq1BlockBytes;
  } else if (format == GGML_Q4_0) {
    p.row_bytes = K / kQ40BlockElems * kQ40BlockBytes;
  } else {
    throw std::invalid_argument("format must be PTQ1_0 (143) or Q4_0 (2)");
  }
  p.scales = scales;
  p.block_rows = scales ? block_rows : 1;
  p.block_cols = scales ? block_cols : 1;
  if (p.block_rows <= 0 || p.block_cols <= 0) throw std::invalid_argument("bad block-scale size");
  // s_scale holds 32 x 16 floats: enough for any block size >= 1.
  p.scale_cols = (N + p.block_cols - 1) / p.block_cols;
  BONSAI_CUDA(cudaMemcpyToSymbolAsync(c_int4, &p, sizeof(p), 0, cudaMemcpyHostToDevice, stream));
  const dim3 grid((N + kI4TileN - 1) / kI4TileN, (M + kI4TileM - 1) / kI4TileM);
  custom_int4_tiled_gemm_kernel<<<grid, dim3(16, 16), 0, stream>>>(A, B, C, M, N, K);
  BONSAI_CUDA_LAUNCH();
}

}  // namespace bonsai
