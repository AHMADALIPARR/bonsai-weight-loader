// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// GPU: custom_int4_tiled_gemm_kernel (PTQ1_0 and Q4_0) against the CPU
// reference. Exits 77 without a GPU.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "bonsai/cuda_check.cuh"
#include "bonsai/ggml_types.hpp"
#include "bonsai/kernels.cuh"
#include "bonsai/quant.hpp"
#include "check.hpp"

using namespace bonsai;

int main() {
  if (!cuda_device_available()) return 77;
  try {
    std::mt19937 rng(8);
    std::uniform_real_distribution<float> uni(-1, 1);
    const int shapes[][3] = {{1, 1, 128}, {45, 37, 256}, {128, 128, 1024}, {33, 250, 512}};
    for (uint32_t fmt : {(uint32_t)GGML_PTQ1_0, (uint32_t)GGML_Q4_0})
      for (auto& sh : shapes)
        for (bool with_scale : {false, true}) {
          const int M = sh[0], N = sh[1], K = sh[2];
          const uint64_t rb = packed_row_bytes(fmt, K);
          std::vector<float> A((size_t)M * K), Wf(K);
          for (auto& v : A) v = uni(rng);
          std::vector<uint8_t> W(N * rb);
          for (int n = 0; n < N; ++n) {
            for (auto& v : Wf) v = uni(rng);
            if (fmt == GGML_PTQ1_0) quantize_row_ptq1_0(Wf.data(), &W[n * rb], K);
            else quantize_row_q4_0(Wf.data(), &W[n * rb], K);
          }
          const int br = 32, bc = 16;
          std::vector<float> sc(((M + br - 1) / br) * ((N + bc - 1) / bc));
          for (auto& v : sc) v = 0.5f + uni(rng);
          std::vector<float> ref((size_t)M * N), got((size_t)M * N);
          packed_gemm_ref(A.data(), W.data(), fmt, ref.data(), M, N, K, with_scale ? sc.data() : nullptr, br, bc);
          float *dA, *dC, *dS;
          uint8_t* dW;
          BONSAI_CUDA(cudaMalloc(&dA, A.size() * 4));
          BONSAI_CUDA(cudaMalloc(&dW, W.size()));
          BONSAI_CUDA(cudaMalloc(&dC, ref.size() * 4));
          BONSAI_CUDA(cudaMalloc(&dS, sc.size() * 4));
          BONSAI_CUDA(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
          BONSAI_CUDA(cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice));
          BONSAI_CUDA(cudaMemcpy(dS, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice));
          launch_custom_int4_tiled_gemm(dA, dW, dC, M, N, K, fmt, with_scale ? dS : nullptr, br, bc, 0);
          BONSAI_CUDA(cudaDeviceSynchronize());
          BONSAI_CUDA(cudaMemcpy(got.data(), dC, got.size() * 4, cudaMemcpyDeviceToHost));
          double maxerr = 0;
          for (size_t i = 0; i < ref.size(); ++i) maxerr = std::fmax(maxerr, std::fabs(got[i] - ref[i]));
          CHECK(maxerr < 1e-3 * std::sqrt((double)K));
          BONSAI_CUDA(cudaFree(dA));
          BONSAI_CUDA(cudaFree(dW));
          BONSAI_CUDA(cudaFree(dC));
          BONSAI_CUDA(cudaFree(dS));
        }
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
  return bonsai_test::finish("test_gpu_int4_gemm");
}
