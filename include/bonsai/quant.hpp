// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Packed weight formats used by the ternary/INT4 GEMM. See docs/packing.md.
//
// PTQ1_0 (ggml type 143, PrismML-Eng/llama.cpp "prism" branch, ggml-common.h):
//   struct { uint8_t qs[24]; uint8_t qh[2]; ggml_half d; }  -> 28 bytes / 128 weights
//   qs[0..15]  : 5 trits each, element e = n*16 + m   (m = byte, n = 0..4)
//   qs[16..23] : 5 trits each, element e = 80 + n*8 + m
//   qh[0..1]   : 4 trits each, element e = 120 + n*2 + h
//   trit n of byte q: t = ((uint8_t)(q * 3^n) * 3) >> 8, weight = (t - 1) * d
// Q4_0 (ggml type 2, upstream):
//   struct { ggml_half d; uint8_t qs[16]; }  -> 18 bytes / 32 weights
//   element j < 16: (qs[j] & 15) - 8, element j >= 16: (qs[j-16] >> 4) - 8, times d
#pragma once
#include <cstdint>
#include <cstring>

#include "bonsai/common.hpp"

namespace bonsai {

constexpr int kPtq1BlockElems = 128;
constexpr int kPtq1BlockBytes = 28;
constexpr int kQ40BlockElems = 32;
constexpr int kQ40BlockBytes = 18;

// IEEE half <-> float, bit exact, round-to-nearest-even.
BONSAI_HD float half_bits_to_float(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ffu, bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {  // subnormal
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        --exp;
      }
      man &= 0x3ffu;
      bits = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (man << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

inline uint16_t float_to_half_bits(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
  const uint32_t ax = x & 0x7fffffffu;
  if (ax >= 0x7f800000u) return sign | 0x7c00u | (ax > 0x7f800000u ? 0x200u : 0);  // inf/nan
  if (ax >= 0x477ff000u) return sign | 0x7c00u;                                       // overflow
  if (ax < 0x38800000u) {                                                             // subnormal
    if (ax < 0x33000000u) return sign;
    const uint32_t e = ax >> 23, m = (ax & 0x7fffffu) | 0x800000u;
    const uint32_t shift = 126 - e;  // 14 + (113 - e)
    uint32_t r = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (r & 1))) ++r;
    return sign | (uint16_t)r;
  }
  uint32_t r = ((ax >> 13) - (112u << 10));
  const uint32_t rem = ax & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
  return sign | (uint16_t)r;
}

BONSAI_HD float bf16_bits_to_float(uint16_t b) {
  uint32_t bits = (uint32_t)b << 16;
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

inline uint16_t float_to_bf16_bits(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  if ((x & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((x >> 16) | 0x40u);
  x += 0x7fffu + ((x >> 16) & 1u);
  return (uint16_t)(x >> 16);
}

// Ternary code {-1, 0, +1} of element e (0..127) of one PTQ1_0 block.
BONSAI_HD int ptq1_0_trit(const uint8_t* blk, int e) {
  int byte_idx, n;
  if (e < 80) {
    byte_idx = e & 15;
    n = e >> 4;
  } else if (e < 120) {
    const int f = e - 80;
    byte_idx = 16 + (f & 7);
    n = f >> 3;
  } else {
    const int f = e - 120;
    byte_idx = 24 + (f & 1);
    n = f >> 1;
  }
  const uint32_t pow3 = n == 0 ? 1u : n == 1 ? 3u : n == 2 ? 9u : n == 3 ? 27u : 81u;
  const uint32_t q = (uint32_t)(uint8_t)(blk[byte_idx] * pow3);
  return (int)((q * 3u) >> 8) - 1;
}

BONSAI_HD float ptq1_0_scale(const uint8_t* blk) {
  return half_bits_to_float((uint16_t)(blk[26] | (blk[27] << 8)));
}

// Signed 4-bit code (-8..7) of element j (0..31) of one Q4_0 block.
BONSAI_HD int q4_0_code(const uint8_t* blk, int j) {
  const uint8_t b = blk[2 + (j & 15)];
  return (int)(j < 16 ? (b & 15) : (b >> 4)) - 8;
}

BONSAI_HD float q4_0_scale(const uint8_t* blk) {
  return half_bits_to_float((uint16_t)(blk[0] | (blk[1] << 8)));
}

// Row helpers (k must be a multiple of the block size).
void quantize_row_ptq1_0(const float* x, uint8_t* out, int64_t k);  // port of quantize_row_ptq1_0_ref
void dequantize_row_ptq1_0(const uint8_t* in, float* y, int64_t k);  // port of dequantize_row_ptq1_0
void quantize_row_q4_0(const float* x, uint8_t* out, int64_t k);     // port of quantize_row_q4_0_ref
void dequantize_row_q4_0(const uint8_t* in, float* y, int64_t k);

}  // namespace bonsai
