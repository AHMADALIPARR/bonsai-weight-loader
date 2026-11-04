// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// cpu_bench [--header <gguf or gguf header prefix>]
// CPU timings: header parse, CPU retile (the --verify reference) on the largest
// tensor of each type in the header, block-scale stage, and the ternary/INT4
// packed GEMM reference. With no --header, the synthetic checkpoint is used.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "bonsai/block_scale.hpp"
#include "bonsai/checkpoint.hpp"
#include "bonsai/ggml_types.hpp"
#include "bonsai/mapped_file.hpp"
#include "bonsai/quant.hpp"
#include "bonsai/synth.hpp"
#include "bonsai/tile_layout.hpp"

using namespace bonsai;
using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t) {
  return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

int main(int argc, char** argv) {
  std::string header;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--header") && i + 1 < argc) header = argv[++i];
  if (header.empty()) {
    header = "/tmp/bonsai_cpu_bench.gguf";
    write_synth_gguf(header, make_synth_tensors(1));
  }
  MappedFile f(header);
  Checkpoint c;
  bool header_only = false;
  try {
    c = parse_gguf(f.data(), f.size(), false);
  } catch (const std::exception&) {
    c = parse_gguf(f.data(), f.size(), true);
    header_only = true;
  }
  const int iters = 20;
  auto t0 = clk::now();
  for (int i = 0; i < iters; ++i) c = parse_gguf(f.data(), f.size(), header_only);
  const double parse_ms = ms_since(t0) / iters;
  std::printf("| benchmark | size | time | throughput |\n|---|---|---|---|\n");
  std::printf("| header parse (%zu tensors, %zu kv) | %.1f MB header | %.2f ms | |\n", c.tensors.size(), c.kv.size(),
              c.data_offset / 1e6, parse_ms);

  // Largest tensor (up to 1 GiB) of each type, retiled from a synthetic buffer of the same size.
  std::map<uint32_t, const TensorInfo*> largest;
  for (const TensorInfo& t : c.tensors)
    if (t.ne.size() >= 2 && type_traits(t.type).known && t.bytes <= (1ull << 30) &&
        (!largest[t.type] || t.bytes > largest[t.type]->bytes))
      largest[t.type] = &t;
  std::mt19937 rng(1);
  for (auto& [type, tp] : largest) {
    if (!tp) continue;
    const TensorInfo& t = *tp;
    TensorLayout L = make_layout(t, TileConfig{16, 16});
    std::vector<uint8_t> src(L.src_bytes), dst(L.dst_bytes, 0);
    for (size_t i = 0; i < src.size(); i += 4096) src[i] = (uint8_t)rng();
    retile_rows_cpu(L, src.data(), 0, L.rows, dst.data());  // warm
    auto t1 = clk::now();
    const int reps = L.src_bytes > (64u << 20) ? 2 : 10;
    for (int r = 0; r < reps; ++r) retile_rows_cpu(L, src.data(), 0, L.rows, dst.data());
    const double ms = ms_since(t1) / reps;
    std::printf("| CPU retile 16x16, %s %s [%llu,%llu] | %.1f MB | %.2f ms | %.2f GB/s |\n", type_name(type).c_str(),
                t.name.c_str(), (unsigned long long)t.ne[0], (unsigned long long)(t.n_elements / t.ne[0]),
                L.src_bytes / 1e6, ms, L.src_bytes / (ms * 1e6));
  }

  // Block-scale stage, 4096 x 4096 fp32, 1 x 128 blocks.
  {
    BlockScaleShape s{4096, 4096, 1, 128};
    std::vector<float> in((size_t)s.M * s.N, 1.5f), out(in.size()), sc((size_t)s.scale_rows() * s.scale_cols(), 0.5f);
    apply_block_scale_ref_f32(in.data(), out.data(), sc.data(), s);
    auto t1 = clk::now();
    const int reps = 5;
    for (int r = 0; r < reps; ++r) apply_block_scale_ref_f32(in.data(), out.data(), sc.data(), s);
    const double ms = ms_since(t1) / reps;
    std::printf("| CPU block-scale fp32 4096x4096 (1x128 blocks) | %.1f MB r+w | %.2f ms | %.2f GB/s |\n",
                2.0 * in.size() * 4 / 1e6, ms, 2.0 * in.size() * 4 / (ms * 1e6));
  }
  // Packed GEMM reference, M=8, N=1024, K=5120 (one Bonsai hidden width).
  for (uint32_t fmt : {(uint32_t)GGML_PTQ1_0, (uint32_t)GGML_Q4_0}) {
    const int M = 8, N = 1024, K = 5120;
    const uint64_t rb = packed_row_bytes(fmt, K);
    std::vector<float> A((size_t)M * K, 0.25f), w(K), C((size_t)M * N);
    std::vector<uint8_t> W(N * rb);
    std::uniform_real_distribution<float> uni(-1, 1);
    for (int n = 0; n < N; ++n) {
      for (auto& v : w) v = uni(rng);
      if (fmt == GGML_PTQ1_0) quantize_row_ptq1_0(w.data(), &W[n * rb], K);
      else quantize_row_q4_0(w.data(), &W[n * rb], K);
    }
    auto t1 = clk::now();
    packed_gemm_ref(A.data(), W.data(), fmt, C.data(), M, N, K, nullptr, 1, 1);
    const double ms = ms_since(t1);
    std::printf("| CPU %s GEMM ref M=%d N=%d K=%d | %.1f MB weights | %.2f ms | %.2f GFLOP/s |\n",
                type_name(fmt).c_str(), M, N, K, W.size() / 1e6, ms, 2.0 * M * N * K / (ms * 1e6));
  }
  return 0;
}
