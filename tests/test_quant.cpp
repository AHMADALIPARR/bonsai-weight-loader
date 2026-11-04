// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// PTQ1_0 / Q4_0 packing and half / bf16 conversion.
#include <cmath>
#include <random>
#include <vector>

#include "bonsai/quant.hpp"
#include "check.hpp"

using namespace bonsai;

int main() {
  CHECK(half_bits_to_float(0x3c00) == 1.0f);
  CHECK(half_bits_to_float(0xc000) == -2.0f);
  CHECK(half_bits_to_float(0x0001) == std::ldexp(1.0f, -24));
  CHECK(std::isinf(half_bits_to_float(0x7c00)));
  CHECK(float_to_half_bits(1.0f) == 0x3c00);
  CHECK(float_to_half_bits(65504.0f) == 0x7bff);
  CHECK(float_to_half_bits(1.0f + std::ldexp(1.0f, -11)) == 0x3c00);  // tie -> even
  CHECK(float_to_bf16_bits(1.0f) == 0x3f80);
  CHECK(bf16_bits_to_float(0xbf80) == -1.0f);
  for (uint32_t h = 0; h < 0x7c00; ++h) CHECK(float_to_half_bits(half_bits_to_float((uint16_t)h)) == h);

  std::mt19937 rng(5);
  // PTQ1_0: exact ternary input round-trips exactly (scale is a power of two).
  {
    const int k = 128 * 4;
    std::vector<float> x(k), y(k);
    for (auto& v : x) v = (float)((int)(rng() % 3) - 1) * 0.25f;
    std::vector<uint8_t> q(k / 128 * kPtq1BlockBytes);
    quantize_row_ptq1_0(x.data(), q.data(), k);
    dequantize_row_ptq1_0(q.data(), y.data(), k);
    CHECK(x == y);
    for (int b = 0; b < k / 128; ++b) {
      const uint8_t* blk = q.data() + b * kPtq1BlockBytes;
      CHECK(ptq1_0_scale(blk) == 0.25f);
      for (int e = 0; e < 128; ++e) CHECK(ptq1_0_trit(blk, e) == (int)(x[b * 128 + e] / 0.25f));
    }
    // every byte value of a full 5-trit byte decodes to valid trits
    uint8_t blk[28] = {};
    for (int v = 0; v < 256; ++v) {
      blk[0] = (uint8_t)v;
      for (int n = 0; n < 5; ++n) {
        int t = ptq1_0_trit(blk, n * 16);
        CHECK(t >= -1 && t <= 1);
      }
    }
  }
  // PTQ1_0 on real-valued input: nearest of {-d, 0, d} with d = amax.
  {
    std::vector<float> x(128), y(128);
    for (auto& v : x) v = std::uniform_real_distribution<float>(-1, 1)(rng);
    std::vector<uint8_t> q(kPtq1BlockBytes);
    quantize_row_ptq1_0(x.data(), q.data(), 128);
    dequantize_row_ptq1_0(q.data(), y.data(), 128);
    const float d = ptq1_0_scale(q.data());
    for (int i = 0; i < 128; ++i) {
      CHECK(y[i] == 0.0f || std::fabs(std::fabs(y[i]) - d) == 0.0f);
      CHECK(std::fabs(x[i] - y[i]) <= 0.5f * d + 1e-3f);
    }
  }
  // Q4_0: code / scale accessors agree with dequantize_row, error <= d/2.
  {
    const int k = 32 * 6;
    std::vector<float> x(k), y(k);
    for (auto& v : x) v = std::uniform_real_distribution<float>(-3, 3)(rng);
    std::vector<uint8_t> q(k / 32 * kQ40BlockBytes);
    quantize_row_q4_0(x.data(), q.data(), k);
    dequantize_row_q4_0(q.data(), y.data(), k);
    for (int b = 0; b < k / 32; ++b) {
      const uint8_t* blk = q.data() + b * kQ40BlockBytes;
      const float d = q4_0_scale(blk);
      for (int j = 0; j < 32; ++j) {
        CHECK(q4_0_code(blk, j) >= -8 && q4_0_code(blk, j) <= 7);
        CHECK(y[b * 32 + j] == (float)q4_0_code(blk, j) * d);
        CHECK(std::fabs(x[b * 32 + j] - y[b * 32 + j]) <= std::fabs(d) + 1e-3f);  // +8 clamps to 7
      }
    }
  }
  return bonsai_test::finish("test_quant");
}
