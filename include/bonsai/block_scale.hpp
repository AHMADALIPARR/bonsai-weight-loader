// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Block-scale output stage and GEMM CPU references.
//
// Block scale: out[m][n] = in[m][n] * scales[m / block_rows][n / block_cols]
// with scales stored row-major as [ceil(M / block_rows)][ceil(N / block_cols)].
// Half/BF16 paths compute the product in fp32 and round once to nearest-even,
// the same as the device kernel.
#pragma once
#include <cstdint>

#include "bonsai/common.hpp"

namespace bonsai {

struct BlockScaleShape {
  int M = 0, N = 0;
  int block_rows = 1, block_cols = 1;
  BONSAI_HD int scale_rows() const { return (M + block_rows - 1) / block_rows; }
  BONSAI_HD int scale_cols() const { return (N + block_cols - 1) / block_cols; }
};

void apply_block_scale_ref_f32(const float* in, float* out, const float* scales, BlockScaleShape s);
void apply_block_scale_ref_f16(const uint16_t* in, uint16_t* out, const float* scales, BlockScaleShape s);
void apply_block_scale_ref_bf16(const uint16_t* in, uint16_t* out, const float* scales, BlockScaleShape s);

// C[M][N] = A[M][K] * B[K][N], all row-major fp32. scales may be null (no scale).
void gemm_block_scale_ref(const float* A, const float* B, float* C, int M, int N, int K,
                          const float* scales, int block_rows, int block_cols);

// Packed-weight GEMM reference: C[m][n] = scale(m, n) * sum_k A[m][k] * W[n][k]
// where A is fp32 [M][K] row-major and W is N rows of K weights packed in
// `format` (GGML_PTQ1_0 or GGML_Q4_0), one GGUF row per output column.
void packed_gemm_ref(const float* A, const uint8_t* W, uint32_t format, float* C, int M, int N, int K,
                     const float* scales, int block_rows, int block_cols);

// Bytes of one packed weight row of K elements.
uint64_t packed_row_bytes(uint32_t format, int K);

}  // namespace bonsai
