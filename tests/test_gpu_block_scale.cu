// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// GPU: apply_block_scale_stage (float / half / bf16, vector and scalar paths)
// and the fused GEMM epilogue against the CPU references. Exits 77 without a GPU.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "bonsai/cuda_check.cuh"
#include "bonsai/kernels.cuh"
#include "bonsai/quant.hpp"
#include "check.hpp"

using namespace bonsai;

namespace {

template <typename T>
std::vector<uint8_t> run_stage(const std::vector<T>& in, const std::vector<float>& sc, BlockScaleShape s,
                               int offset_elems) {
  const size_t n = (size_t)s.M * s.N;
  T *d_in = nullptr, *d_out = nullptr;
  float* d_sc = nullptr;
  BONSAI_CUDA(cudaMalloc(&d_in, (n + 4) * sizeof(T)));
  BONSAI_CUDA(cudaMalloc(&d_out, (n + 4) * sizeof(T)));
  BONSAI_CUDA(cudaMalloc(&d_sc, sc.size() * sizeof(float)));
  BONSAI_CUDA(cudaMemcpy(d_in + offset_elems, in.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  BONSAI_CUDA(cudaMemcpy(d_sc, sc.data(), sc.size() * sizeof(float), cudaMemcpyHostToDevice));
  launch_apply_block_scale<T>(d_in + offset_elems, d_out + offset_elems, d_sc, s, 0);
  BONSAI_CUDA(cudaDeviceSynchronize());
  std::vector<uint8_t> out(n * sizeof(T));
  BONSAI_CUDA(cudaMemcpy(out.data(), d_out + offset_elems, out.size(), cudaMemcpyDeviceToHost));
  BONSAI_CUDA(cudaFree(d_in));
  BONSAI_CUDA(cudaFree(d_out));
  BONSAI_CUDA(cudaFree(d_sc));
  return out;
}

}  // namespace

int main() {
  if (!cuda_device_available()) return 77;
  try {
    std::mt19937 rng(4);
    std::uniform_real_distribution<float> uni(-2, 2);
    const int shapes[][4] = {{16, 256, 16, 32}, {37, 1000, 4, 128}, {5, 7, 2, 3}, {300, 4096, 1, 1}, {1, 3, 1, 1}};
    for (auto& sh : shapes)
      for (int off : {0, 1}) {  // off = 1 breaks alignment -> scalar path
        BlockScaleShape s{sh[0], sh[1], sh[2], sh[3]};
        const size_t n = (size_t)s.M * s.N;
        std::vector<float> sc((size_t)s.scale_rows() * s.scale_cols());
        for (auto& v : sc) v = uni(rng);
        std::vector<float> f(n), fr(n);
        std::vector<uint16_t> h(n), hr(n), b(n), bref(n);
        for (size_t i = 0; i < n; ++i) {
          f[i] = uni(rng);
          h[i] = float_to_half_bits(f[i]);
          b[i] = float_to_bf16_bits(f[i]);
        }
        apply_block_scale_ref_f32(f.data(), fr.data(), sc.data(), s);
        apply_block_scale_ref_f16(h.data(), hr.data(), sc.data(), s);
        apply_block_scale_ref_bf16(b.data(), bref.data(), sc.data(), s);
        std::vector<__half> hh(n);
        std::vector<__nv_bfloat16> bb(n);
        std::memcpy(hh.data(), h.data(), n * 2);
        std::memcpy(bb.data(), b.data(), n * 2);
        auto of = run_stage<float>(f, sc, s, off);
        auto oh = run_stage<__half>(hh, sc, s, off);
        auto ob = run_stage<__nv_bfloat16>(bb, sc, s, off);
        CHECK(std::memcmp(of.data(), fr.data(), n * 4) == 0);
        CHECK(std::memcmp(oh.data(), hr.data(), n * 2) == 0);
        CHECK(std::memcmp(ob.data(), bref.data(), n * 2) == 0);
      }

    // Fused GEMM epilogue vs CPU reference, and unfused GEMM + stage.
    const int dims[][5] = {{128, 128, 64, 16, 16}, {70, 37, 33, 8, 128}, {256, 512, 128, 32, 64}};
    for (auto& dm : dims) {
      const int M = dm[0], N = dm[1], K = dm[2], br = dm[3], bc = dm[4];
      std::vector<float> A((size_t)M * K), B((size_t)K * N), sc(((M + br - 1) / br) * ((N + bc - 1) / bc));
      for (auto& v : A) v = uni(rng);
      for (auto& v : B) v = uni(rng);
      for (auto& v : sc) v = uni(rng);
      std::vector<float> ref((size_t)M * N), fused((size_t)M * N), staged((size_t)M * N);
      gemm_block_scale_ref(A.data(), B.data(), ref.data(), M, N, K, sc.data(), br, bc);
      float *dA, *dB, *dC, *dS;
      BONSAI_CUDA(cudaMalloc(&dA, A.size() * 4));
      BONSAI_CUDA(cudaMalloc(&dB, B.size() * 4));
      BONSAI_CUDA(cudaMalloc(&dC, ref.size() * 4));
      BONSAI_CUDA(cudaMalloc(&dS, sc.size() * 4));
      BONSAI_CUDA(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
      BONSAI_CUDA(cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice));
      BONSAI_CUDA(cudaMemcpy(dS, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice));
      launch_gemm_f32(dA, dB, dC, M, N, K, dS, br, bc, true, 0);
      BONSAI_CUDA(cudaMemcpy(fused.data(), dC, fused.size() * 4, cudaMemcpyDeviceToHost));
      launch_gemm_f32(dA, dB, dC, M, N, K, nullptr, 1, 1, false, 0);
      launch_apply_block_scale<float>(dC, dC, dS, BlockScaleShape{M, N, br, bc}, 0);
      BONSAI_CUDA(cudaMemcpy(staged.data(), dC, staged.size() * 4, cudaMemcpyDeviceToHost));
      double e1 = 0, e2 = 0;
      for (size_t i = 0; i < ref.size(); ++i) {
        e1 = std::fmax(e1, std::fabs(fused[i] - ref[i]));
        e2 = std::fmax(e2, std::fabs(staged[i] - ref[i]));
      }
      CHECK(e1 < 1e-3 * std::sqrt((double)K));
      CHECK(e2 < 1e-3 * std::sqrt((double)K));
      for (float* p : {dA, dB, dC, dS}) BONSAI_CUDA(cudaFree(p));
    }
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
  return bonsai_test::finish("test_gpu_block_scale");
}
