// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Device kernels: block-scale output stage, fp32 tiled GEMM with fused
// block-scale epilogue, and the ternary/INT4 packed-weight tiled GEMM.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "bonsai/block_scale.hpp"

namespace bonsai {

// ---- block-scale output stage ------------------------------------------------
// CTA tile: kBsTileM rows x kBsTileN columns, 256 threads, 4 columns per thread.
constexpr int kBsTileM = 16;
constexpr int kBsTileN = 256;

template <typename T>
__global__ void __launch_bounds__(256, 4)
    apply_block_scale_stage(const T* __restrict__ in, T* __restrict__ out,
                            const float* __restrict__ scales, BlockScaleShape s, int vec_ok);

// T = float, __half or __nv_bfloat16. in may equal out.
template <typename T>
void launch_apply_block_scale(const T* in, T* out, const float* scales, BlockScaleShape s,
                              cudaStream_t stream);

// ---- fp32 tiled GEMM (C = A[M,K] * B[K,N], row-major) -------------------------
constexpr int kGemmBM = 64, kGemmBN = 64, kGemmBK = 16;

// fused = true: block scale applied to the accumulators before the final store.
// fused = false: plain GEMM (scales ignored); run launch_apply_block_scale after
// it for the unfused comparison.
void launch_gemm_f32(const float* A, const float* B, float* C, int M, int N, int K,
                     const float* scales, int block_rows, int block_cols, bool fused,
                     cudaStream_t stream);

// ---- ternary / INT4 packed-weight GEMM ----------------------------------------
// A: fp32 [M][K] row-major. B: N packed rows of K weights in GGUF row layout
// (PTQ1_0 or Q4_0, see docs/packing.md). C: fp32 [M][N] row-major.
// Format and the output block scale live in constant memory, set by the launcher.
constexpr int kI4TileM = 32, kI4TileN = 16, kI4BK = 64;

__global__ void __launch_bounds__(256, 4)
    custom_int4_tiled_gemm_kernel(const void* __restrict__ A, const void* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K);

// Requires K % 64 == 0, and K % 128 == 0 for PTQ1_0. scales may be null.
void launch_custom_int4_tiled_gemm(const float* A, const uint8_t* B, float* C, int M, int N, int K,
                                   uint32_t format, const float* scales, int block_rows,
                                   int block_cols, cudaStream_t stream);

}  // namespace bonsai
