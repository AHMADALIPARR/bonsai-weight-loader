// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Ternary / INT4 unpack and packed GEMM on the CPU. Includes a CPU emulation of
// custom_int4_tiled_gemm_kernel's tile decomposition (same tile sizes, same
// unpack mapping, same accumulation order) to check its index math.
#include <cmath>
#include <random>
#include <vector>

#include "bonsai/block_scale.hpp"
#include "bonsai/ggml_types.hpp"
#include "bonsai/quant.hpp"
#include "check.hpp"

using namespace bonsai;

namespace {

constexpr int TM = 32, TN = 16, BK = 64;  // = kI4TileM, kI4TileN, kI4BK

void emulate_kernel(const float* A, const uint8_t* W, uint32_t format, float* C, int M, int N, int K,
                    const float* scales, int br, int bc) {
  const uint64_t rb = packed_row_bytes(format, K);
  const int scols = (N + bc - 1) / bc;
  float smem_A[TM][BK + 1], smem_B[BK][TN + 1];
  for (int by = 0; by * TM < M; ++by)
    for (int bx = 0; bx * TN < N; ++bx) {
      const int m0 = by * TM, n0 = bx * TN;
      float acc[16][16][2] = {};
      for (int k0 = 0; k0 < K; k0 += BK) {
        for (int tid = 0; tid < 256; ++tid) {
          for (int i = 0; i < TM * BK / 256; ++i) {
            const int idx = tid + i * 256, r = idx >> 6, k = idx & 63;
            smem_A[r][k] = m0 + r < M ? A[(size_t)(m0 + r) * K + k0 + k] : 0.0f;
          }
          const int n = tid >> 4, lane = tid & 15, gn = n0 + n;
          for (int i = 0; i < 4; ++i) {
            const int kk = lane + 16 * i;
            float v = 0.0f;
            if (gn < N) {
              const uint8_t* row = W + gn * rb;
              if (format == GGML_PTQ1_0) {
                const uint8_t* blk = row + (k0 >> 7) * kPtq1BlockBytes;
                v = (float)ptq1_0_trit(blk, (k0 & 127) + kk) * ptq1_0_scale(blk);
              } else {
                const uint8_t* blk = row + ((k0 + kk) >> 5) * kQ40BlockBytes;
                v = (float)q4_0_code(blk, (k0 + kk) & 31) * q4_0_scale(blk);
              }
            }
            smem_B[kk][n] = v;
          }
        }
        for (int y = 0; y < 16; ++y)
          for (int tx = 0; tx < 16; ++tx) {
            const int ty = y * 2;
            for (int k = 0; k < BK; ++k) {
              float b_val = smem_B[k][tx];
              acc[y][tx][0] += smem_A[ty][k] * b_val;
              acc[y][tx][1] += smem_A[ty + 1][k] * b_val;
            }
          }
      }
      for (int y = 0; y < 16; ++y)
        for (int tx = 0; tx < 16; ++tx)
          for (int i = 0; i < 2; ++i) {
            const int gm = m0 + y * 2 + i, gn = n0 + tx;
            if (gm >= M || gn >= N) continue;
            float v = acc[y][tx][i];
            if (scales) v *= scales[(gm / br) * scols + gn / bc];
            C[(size_t)gm * N + gn] = v;
          }
    }
}

}  // namespace

int main() {
  std::mt19937 rng(21);
  std::uniform_real_distribution<float> uni(-1, 1);
  const uint32_t formats[] = {GGML_PTQ1_0, GGML_Q4_0};
  const int shapes[][3] = {{1, 1, 128}, {45, 37, 256}, {64, 16, 384}, {33, 50, 128}};
  for (uint32_t fmt : formats)
    for (auto& sh : shapes) {
      const int M = sh[0], N = sh[1], K = sh[2];
      const uint64_t rb = packed_row_bytes(fmt, K);
      std::vector<float> A((size_t)M * K), Wf((size_t)N * K), Wd((size_t)N * K);
      for (auto& v : A) v = uni(rng);
      for (auto& v : Wf) v = fmt == GGML_PTQ1_0 ? uni(rng) : 3.0f * uni(rng);
      std::vector<uint8_t> W(N * rb);
      for (int n = 0; n < N; ++n) {
        if (fmt == GGML_PTQ1_0) quantize_row_ptq1_0(&Wf[(size_t)n * K], &W[n * rb], K);
        else quantize_row_q4_0(&Wf[(size_t)n * K], &W[n * rb], K);
      }
      // unpack: per-element accessors == row dequantizer
      for (int n = 0; n < N; ++n) {
        if (fmt == GGML_PTQ1_0) dequantize_row_ptq1_0(&W[n * rb], &Wd[(size_t)n * K], K);
        else dequantize_row_q4_0(&W[n * rb], &Wd[(size_t)n * K], K);
        for (int k = 0; k < K; ++k) {
          float v;
          if (fmt == GGML_PTQ1_0) {
            const uint8_t* blk = &W[n * rb + (k / 128) * kPtq1BlockBytes];
            v = (float)ptq1_0_trit(blk, k % 128) * ptq1_0_scale(blk);
          } else {
            const uint8_t* blk = &W[n * rb + (k / 32) * kQ40BlockBytes];
            v = (float)q4_0_code(blk, k % 32) * q4_0_scale(blk);
          }
          CHECK(v == Wd[(size_t)n * K + k]);
        }
      }
      const int br = 16, bc = 8;
      std::vector<float> sc(((M + br - 1) / br) * ((N + bc - 1) / bc));
      for (auto& v : sc) v = 0.5f + uni(rng);
      std::vector<float> ref((size_t)M * N), emu((size_t)M * N, NAN);
      packed_gemm_ref(A.data(), W.data(), fmt, ref.data(), M, N, K, sc.data(), br, bc);
      emulate_kernel(A.data(), W.data(), fmt, emu.data(), M, N, K, sc.data(), br, bc);
      double maxerr = 0;
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
          double acc = 0;
          for (int k = 0; k < K; ++k) acc += (double)A[(size_t)m * K + k] * Wd[(size_t)n * K + k];
          acc *= sc[(m / br) * ((N + bc - 1) / bc) + n / bc];
          maxerr = std::fmax(maxerr, std::fabs(acc - ref[(size_t)m * N + n]));
          CHECK(emu[(size_t)m * N + n] == ref[(size_t)m * N + n]);  // same k order, same rounding
        }
      CHECK(maxerr < 1e-3);
    }
  return bonsai_test::finish("test_int4_gemm");
}
