// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// gpu_bench: device timings (cudaEvent) for the block-scale stage, fused vs
// unfused GEMM epilogue, and custom_int4_tiled_gemm_kernel. Exits 77 without a GPU.
// Loader throughput is printed by bonsai_load itself.
#include <cstdio>
#include <vector>

#include "bonsai/cuda_check.cuh"
#include "bonsai/ggml_types.hpp"
#include "bonsai/kernels.cuh"
#include "bonsai/quant.hpp"

using namespace bonsai;

template <typename F>
static float time_ms(F&& fn, int reps = 20) {
  cudaEvent_t a, b;
  BONSAI_CUDA(cudaEventCreate(&a));
  BONSAI_CUDA(cudaEventCreate(&b));
  fn();
  BONSAI_CUDA(cudaEventRecord(a));
  for (int i = 0; i < reps; ++i) fn();
  BONSAI_CUDA(cudaEventRecord(b));
  BONSAI_CUDA(cudaEventSynchronize(b));
  float ms = 0;
  BONSAI_CUDA(cudaEventElapsedTime(&ms, a, b));
  BONSAI_CUDA(cudaEventDestroy(a));
  BONSAI_CUDA(cudaEventDestroy(b));
  return ms / reps;
}

int main() {
  if (!cuda_device_available()) return 77;
  try {
    cudaDeviceProp p;
    BONSAI_CUDA(cudaGetDeviceProperties(&p, 0));
    std::printf("device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    std::printf("| kernel | shape | ms | throughput |\n|---|---|---|---|\n");

    const int M = 8192, N = 8192;
    BlockScaleShape s{M, N, 1, 128};
    const size_t n = (size_t)M * N;
    void *in, *out;
    float* sc;
    BONSAI_CUDA(cudaMalloc(&in, n * 4));
    BONSAI_CUDA(cudaMalloc(&out, n * 4));
    BONSAI_CUDA(cudaMalloc(&sc, (size_t)s.scale_rows() * s.scale_cols() * 4));
    BONSAI_CUDA(cudaMemset(in, 0, n * 4));
    BONSAI_CUDA(cudaMemset(sc, 0, (size_t)s.scale_rows() * s.scale_cols() * 4));
    float ms = time_ms([&] { launch_apply_block_scale<float>((float*)in, (float*)out, sc, s, 0); });
    std::printf("| apply_block_scale_stage<float> | 8192x8192, 1x128 | %.3f | %.1f GB/s |\n", ms, 2.0 * n * 4 / (ms * 1e6));
    ms = time_ms([&] { launch_apply_block_scale<__half>((__half*)in, (__half*)out, sc, s, 0); });
    std::printf("| apply_block_scale_stage<half> | 8192x8192, 1x128 | %.3f | %.1f GB/s |\n", ms, 2.0 * n * 2 / (ms * 1e6));
    ms = time_ms([&] { launch_apply_block_scale<__nv_bfloat16>((__nv_bfloat16*)in, (__nv_bfloat16*)out, sc, s, 0); });
    std::printf("| apply_block_scale_stage<bf16> | 8192x8192, 1x128 | %.3f | %.1f GB/s |\n", ms, 2.0 * n * 2 / (ms * 1e6));
    BONSAI_CUDA(cudaFree(in));
    BONSAI_CUDA(cudaFree(out));
    BONSAI_CUDA(cudaFree(sc));

    {
      const int G = 2048;
      float *A, *B, *C, *S;
      BONSAI_CUDA(cudaMalloc(&A, (size_t)G * G * 4));
      BONSAI_CUDA(cudaMalloc(&B, (size_t)G * G * 4));
      BONSAI_CUDA(cudaMalloc(&C, (size_t)G * G * 4));
      BONSAI_CUDA(cudaMalloc(&S, (size_t)(G / 16) * (G / 128) * 4));
      BONSAI_CUDA(cudaMemset(A, 0, (size_t)G * G * 4));
      BONSAI_CUDA(cudaMemset(B, 0, (size_t)G * G * 4));
      BONSAI_CUDA(cudaMemset(S, 0, (size_t)(G / 16) * (G / 128) * 4));
      const double flop = 2.0 * G * G * G;
      ms = time_ms([&] { launch_gemm_f32(A, B, C, G, G, G, S, 16, 128, true, 0); }, 5);
      std::printf("| fp32 GEMM, fused block-scale epilogue | 2048^3, 16x128 | %.3f | %.1f GFLOP/s |\n", ms, flop / (ms * 1e6));
      ms = time_ms([&] {
        launch_gemm_f32(A, B, C, G, G, G, nullptr, 1, 1, false, 0);
        launch_apply_block_scale<float>(C, C, S, BlockScaleShape{G, G, 16, 128}, 0);
      }, 5);
      std::printf("| fp32 GEMM + standalone stage | 2048^3, 16x128 | %.3f | %.1f GFLOP/s |\n", ms, flop / (ms * 1e6));
      for (float* q : {A, B, C, S}) BONSAI_CUDA(cudaFree(q));
    }

    for (uint32_t fmt : {(uint32_t)GGML_PTQ1_0, (uint32_t)GGML_Q4_0})
      for (int Mq : {1, 16, 128}) {
        const int Nq = 5120, Kq = 5120;
        const uint64_t rb = packed_row_bytes(fmt, Kq);
        float *A, *C;
        uint8_t* W;
        BONSAI_CUDA(cudaMalloc(&A, (size_t)Mq * Kq * 4));
        BONSAI_CUDA(cudaMalloc(&C, (size_t)Mq * Nq * 4));
        BONSAI_CUDA(cudaMalloc(&W, Nq * rb));
        BONSAI_CUDA(cudaMemset(A, 0, (size_t)Mq * Kq * 4));
        BONSAI_CUDA(cudaMemset(W, 0, Nq * rb));
        ms = time_ms([&] { launch_custom_int4_tiled_gemm(A, W, C, Mq, Nq, Kq, fmt, nullptr, 1, 1, 0); });
        std::printf("| custom_int4_tiled_gemm_kernel %s | M=%d N=%d K=%d | %.3f | %.1f GFLOP/s, %.1f GB/s weights |\n",
                    type_name(fmt).c_str(), Mq, Nq, Kq, ms, 2.0 * Mq * Nq * Kq / (ms * 1e6), Nq * rb / (ms * 1e6));
        BONSAI_CUDA(cudaFree(A));
        BONSAI_CUDA(cudaFree(C));
        BONSAI_CUDA(cudaFree(W));
      }
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
  return 0;
}
