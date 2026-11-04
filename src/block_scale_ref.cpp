// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// CPU references for the block-scale stage, fp32 GEMM and packed GEMM.
#include <stdexcept>
#include <vector>

#include "bonsai/block_scale.hpp"
#include "bonsai/ggml_types.hpp"
#include "bonsai/quant.hpp"

namespace bonsai {

namespace {
inline float scale_at(const float* scales, BlockScaleShape s, int m, int n) {
  return scales ? scales[(m / s.block_rows) * s.scale_cols() + n / s.block_cols] : 1.0f;
}
}  // namespace

void apply_block_scale_ref_f32(const float* in, float* out, const float* scales, BlockScaleShape s) {
  for (int m = 0; m < s.M; ++m)
    for (int n = 0; n < s.N; ++n)
      out[(size_t)m * s.N + n] = in[(size_t)m * s.N + n] * scale_at(scales, s, m, n);
}

void apply_block_scale_ref_f16(const uint16_t* in, uint16_t* out, const float* scales, BlockScaleShape s) {
  for (int m = 0; m < s.M; ++m)
    for (int n = 0; n < s.N; ++n) {
      size_t i = (size_t)m * s.N + n;
      out[i] = float_to_half_bits(half_bits_to_float(in[i]) * scale_at(scales, s, m, n));
    }
}

void apply_block_scale_ref_bf16(const uint16_t* in, uint16_t* out, const float* scales, BlockScaleShape s) {
  for (int m = 0; m < s.M; ++m)
    for (int n = 0; n < s.N; ++n) {
      size_t i = (size_t)m * s.N + n;
      out[i] = float_to_bf16_bits(bf16_bits_to_float(in[i]) * scale_at(scales, s, m, n));
    }
}

void gemm_block_scale_ref(const float* A, const float* B, float* C, int M, int N, int K,
                          const float* scales, int block_rows, int block_cols) {
  BlockScaleShape s{M, N, block_rows, block_cols};
  std::vector<float> row(N);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) row[n] = 0.0f;
    for (int k = 0; k < K; ++k) {
      const float a = A[(size_t)m * K + k];
      const float* b = B + (size_t)k * N;
      for (int n = 0; n < N; ++n) row[n] += a * b[n];
    }
    for (int n = 0; n < N; ++n) C[(size_t)m * N + n] = row[n] * scale_at(scales, s, m, n);
  }
}

uint64_t packed_row_bytes(uint32_t format, int K) {
  if (format == GGML_PTQ1_0) return (uint64_t)K / kPtq1BlockElems * kPtq1BlockBytes;
  if (format == GGML_Q4_0) return (uint64_t)K / kQ40BlockElems * kQ40BlockBytes;
  throw std::runtime_error("packed format must be PTQ1_0 or Q4_0");
}

void packed_gemm_ref(const float* A, const uint8_t* W, uint32_t format, float* C, int M, int N, int K,
                     const float* scales, int block_rows, int block_cols) {
  const uint64_t rb = packed_row_bytes(format, K);
  BlockScaleShape s{M, N, block_rows, block_cols};
  std::vector<float> w((size_t)N * K);
  for (int n = 0; n < N; ++n) {
    if (format == GGML_PTQ1_0)
      dequantize_row_ptq1_0(W + n * rb, w.data() + (size_t)n * K, K);
    else
      dequantize_row_q4_0(W + n * rb, w.data() + (size_t)n * K, K);
  }
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) acc += A[(size_t)m * K + k] * w[(size_t)n * K + k];
      C[(size_t)m * N + n] = acc * scale_at(scales, s, m, n);
    }
}

}  // namespace bonsai
