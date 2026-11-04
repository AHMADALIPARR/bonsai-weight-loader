// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Row quantizers/dequantizers, ported from ggml-quants.c (PrismML-Eng/llama.cpp,
// branch "prism") so the tests check against the same stage order.
#include "bonsai/quant.hpp"

#include <algorithm>
#include <cmath>

namespace bonsai {

namespace {
const size_t kStages[3] = {32, 16, 8};  // ptq1_0_stages
constexpr size_t kQs = 24, kQh = 2;
}  // namespace

void quantize_row_ptq1_0(const float* x, uint8_t* out, int64_t k) {
  const int64_t nb = k / kPtq1BlockElems;
  for (int64_t i = 0; i < nb; ++i) {
    uint8_t* y = out + i * kPtq1BlockBytes;
    float amax = 0.0f;
    for (int j = 0; j < kPtq1BlockElems; ++j) amax = std::max(amax, std::fabs(x[j]));
    const float d = amax, id = d ? 1.0f / d : 0.0f;
    const uint16_t dh = float_to_half_bits(d);
    y[26] = (uint8_t)(dh & 0xff);
    y[27] = (uint8_t)(dh >> 8);
    size_t j = 0;
    for (size_t s = 0; s < 3; ++s) {
      const size_t c = kStages[s];
      for (; j + c <= kQs; j += c) {
        for (size_t m = 0; m < c; ++m) {
          uint8_t q = 0;
          for (size_t n = 0; n < 5; ++n) {
            int xi = (int)std::lround(x[m + n * c] * id) + 1;
            q = (uint8_t)(q * 3 + xi);
          }
          q = (uint8_t)(((uint16_t)q * 256 + (243 - 1)) / 243);
          y[j + m] = q;
        }
        x += 5 * c;
      }
    }
    for (size_t h = 0; h < kQh; ++h) {
      uint8_t q = 0;
      for (size_t m = 0; m < 4; ++m) {
        int xi = (int)std::lround(x[h + m * kQh] * id) + 1;
        q = (uint8_t)(q * 3 + xi);
      }
      q = (uint8_t)(q * 3);
      q = (uint8_t)(((uint16_t)q * 256 + (243 - 1)) / 243);
      y[kQs + h] = q;
    }
    x += 4 * kQh;
  }
}

void dequantize_row_ptq1_0(const uint8_t* in, float* y, int64_t k) {
  const int64_t nb = k / kPtq1BlockElems;
  const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* b = in + i * kPtq1BlockBytes;
    const float d = ptq1_0_scale(b);
    size_t j = 0;
    for (size_t s = 0; s < 3; ++s) {
      const size_t c = kStages[s];
      for (; j + c <= kQs; j += c)
        for (size_t n = 0; n < 5; ++n)
          for (size_t m = 0; m < c; ++m) {
            uint8_t q = (uint8_t)(b[j + m] * pow3[n]);
            int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
            *y++ = (float)(xi - 1) * d;
          }
    }
    for (size_t n = 0; n < 4; ++n)
      for (size_t h = 0; h < kQh; ++h) {
        uint8_t q = (uint8_t)(b[kQs + h] * pow3[n]);
        int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
        *y++ = (float)(xi - 1) * d;
      }
  }
}

void quantize_row_q4_0(const float* x, uint8_t* out, int64_t k) {
  const int64_t nb = k / kQ40BlockElems;
  for (int64_t i = 0; i < nb; ++i) {
    const float* xb = x + i * kQ40BlockElems;
    uint8_t* y = out + i * kQ40BlockBytes;
    float amax = 0.0f, max = 0.0f;
    for (int j = 0; j < kQ40BlockElems; ++j)
      if (std::fabs(xb[j]) > amax) {
        amax = std::fabs(xb[j]);
        max = xb[j];
      }
    const float d = max / -8, id = d ? 1.0f / d : 0.0f;
    const uint16_t dh = float_to_half_bits(d);
    y[0] = (uint8_t)(dh & 0xff);
    y[1] = (uint8_t)(dh >> 8);
    for (int j = 0; j < kQ40BlockElems / 2; ++j) {
      const uint8_t x0 = (uint8_t)std::min(15, (int)(int8_t)(xb[j] * id + 8.5f));
      const uint8_t x1 = (uint8_t)std::min(15, (int)(int8_t)(xb[j + 16] * id + 8.5f));
      y[2 + j] = (uint8_t)(x0 | (x1 << 4));
    }
  }
}

void dequantize_row_q4_0(const uint8_t* in, float* y, int64_t k) {
  const int64_t nb = k / kQ40BlockElems;
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* b = in + i * kQ40BlockBytes;
    const float d = q4_0_scale(b);
    for (int j = 0; j < 16; ++j) {
      y[i * 32 + j] = (float)((b[2 + j] & 15) - 8) * d;
      y[i * 32 + j + 16] = (float)((b[2 + j] >> 4) - 8) * d;
    }
  }
}

}  // namespace bonsai
