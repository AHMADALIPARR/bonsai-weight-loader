// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Block-scale math (CPU reference) and the fused-epilogue identity
// scale(A.B) == fused GEMM.
#include <cmath>
#include <random>
#include <vector>

#include "bonsai/block_scale.hpp"
#include "bonsai/quant.hpp"
#include "check.hpp"

using namespace bonsai;

int main() {
  // Hand-checked 3 x 5 with 2 x 2 blocks -> 2 x 3 scale grid.
  {
    BlockScaleShape s{3, 5, 2, 2};
    CHECK(s.scale_rows() == 2 && s.scale_cols() == 3);
    const float sc[6] = {1, 2, 3, 4, 5, 6};
    std::vector<float> in(15, 1.0f), out(15);
    apply_block_scale_ref_f32(in.data(), out.data(), sc, s);
    const float want[15] = {1, 1, 2, 2, 3, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6};
    for (int i = 0; i < 15; ++i) CHECK(out[i] == want[i]);
  }
  std::mt19937 rng(9);
  std::uniform_real_distribution<float> uni(-2, 2);
  const int shapes[][4] = {{1, 1, 1, 1}, {17, 33, 16, 16}, {64, 260, 1, 128}, {40, 7, 3, 5}, {16, 256, 16, 256}};
  for (auto& sh : shapes) {
    BlockScaleShape s{sh[0], sh[1], sh[2], sh[3]};
    const size_t n = (size_t)s.M * s.N;
    std::vector<float> sc((size_t)s.scale_rows() * s.scale_cols());
    for (auto& v : sc) v = uni(rng);
    std::vector<float> in(n), out(n);
    for (auto& v : in) v = uni(rng);
    apply_block_scale_ref_f32(in.data(), out.data(), sc.data(), s);
    for (int m = 0; m < s.M; ++m)
      for (int c = 0; c < s.N; ++c)
        CHECK(out[(size_t)m * s.N + c] == in[(size_t)m * s.N + c] * sc[(m / s.block_rows) * s.scale_cols() + c / s.block_cols]);
    std::vector<uint16_t> h(n), ho(n), b(n), bo(n);
    for (size_t i = 0; i < n; ++i) {
      h[i] = float_to_half_bits(in[i]);
      b[i] = float_to_bf16_bits(in[i]);
    }
    apply_block_scale_ref_f16(h.data(), ho.data(), sc.data(), s);
    apply_block_scale_ref_bf16(b.data(), bo.data(), sc.data(), s);
    for (int m = 0; m < s.M; ++m)
      for (int c = 0; c < s.N; ++c) {
        const size_t i = (size_t)m * s.N + c;
        const float f = sc[(m / s.block_rows) * s.scale_cols() + c / s.block_cols];
        CHECK(ho[i] == float_to_half_bits(half_bits_to_float(h[i]) * f));
        CHECK(bo[i] == float_to_bf16_bits(bf16_bits_to_float(b[i]) * f));
      }
  }
  // Fused epilogue == GEMM followed by the standalone stage (same fp32 rounding).
  {
    const int M = 37, N = 70, K = 29;
    std::vector<float> A((size_t)M * K), B((size_t)K * N), sc(((M + 15) / 16) * ((N + 31) / 32));
    for (auto& v : A) v = uni(rng);
    for (auto& v : B) v = uni(rng);
    for (auto& v : sc) v = uni(rng);
    std::vector<float> fused((size_t)M * N), plain((size_t)M * N), staged((size_t)M * N);
    gemm_block_scale_ref(A.data(), B.data(), fused.data(), M, N, K, sc.data(), 16, 32);
    gemm_block_scale_ref(A.data(), B.data(), plain.data(), M, N, K, nullptr, 1, 1);
    apply_block_scale_ref_f32(plain.data(), staged.data(), sc.data(), BlockScaleShape{M, N, 16, 32});
    CHECK(fused == staged);
    double maxerr = 0;
    for (int m = 0; m < M; ++m)
      for (int c = 0; c < N; ++c) {
        double acc = 0;
        for (int k = 0; k < K; ++k) acc += (double)A[(size_t)m * K + k] * B[(size_t)k * N + c];
        maxerr = std::fmax(maxerr, std::fabs(acc - plain[(size_t)m * N + c]));
      }
    CHECK(maxerr < 1e-4);
  }
  return bonsai_test::finish("test_block_scale");
}
