// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// bonsai_load <ckpt> [--tile R C] [--verify] [--checksum] [--raw <manifest>] [--chunk-mb N] [--device D]
// Loads every tensor into one device arena in the tiled layout and prints
// per-tensor and total throughput.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include "bonsai/cuda_check.cuh"
#include "bonsai/loader.hpp"
#include "cli_args.hpp"

using namespace bonsai;

static double gbps(uint64_t bytes, double ms) { return ms > 0 ? bytes / (ms * 1e6) : 0.0; }

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: bonsai_load <ckpt> [--tile R C] [--verify] [--checksum] [--raw <manifest>] "
                 "[--chunk-mb N] [--device D]\n");
    return 2;
  }
  std::string path = argv[1], manifest;
  LoadOptions opt;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--tile") && i + 2 < argc) {
      opt.tile.rows = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
      opt.tile.cols = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
    } else if (!std::strcmp(argv[i], "--verify")) {
      opt.verify = true;
    } else if (!std::strcmp(argv[i], "--checksum")) {
      opt.checksum = true;
    } else if (!std::strcmp(argv[i], "--raw") && i + 1 < argc) {
      manifest = argv[++i];
    } else if (!std::strcmp(argv[i], "--chunk-mb") && i + 1 < argc) {
      opt.chunk_bytes = (size_t)std::strtoull(argv[++i], nullptr, 10) << 20;
    } else if (!std::strcmp(argv[i], "--device") && i + 1 < argc) {
      opt.device = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr, "unknown argument %s\n", argv[i]);
      return 2;
    }
  }
  if (opt.tile.rows == 0 || opt.tile.cols == 0) {
    std::fprintf(stderr, "--tile needs R, C > 0\n");
    return 2;
  }
  try {
    MappedFile f(path);
    Checkpoint c = open_checkpoint(f, manifest, false);
    if (!cuda_device_available()) return 77;
    LoadReport r = load_checkpoint_to_device(f.data(), c, opt);
    std::printf("%-40s %-7s %-18s %12s %9s %8s%s\n", "tensor", "type", "shape", "bytes", "ms", "GB/s",
                opt.verify ? "  verify" : "");
    for (size_t i = 0; i < r.tensors.size(); ++i) {
      const TensorLoad& t = r.tensors[i];
      char shape[64] = "", *p = shape;
      for (size_t d = 0; d < c.tensors[i].ne.size(); ++d)
        p += std::snprintf(p, shape + sizeof shape - p, "%s%llu", d ? "x" : "",
                           (unsigned long long)c.tensors[i].ne[d]);
      std::printf("%-40s %-7s %-18s %12llu %9.3f %8.2f%s\n", t.name.c_str(), type_name(t.type).c_str(), shape,
                  (unsigned long long)t.bytes_copied, t.ms, gbps(t.bytes_copied, t.ms),
                  opt.verify ? (t.verified ? "  ok" : "  FAIL") : "");
    }
    std::printf("total: %zu tensors, %llu bytes in %.1f ms = %.2f GB/s, arena %llu bytes, tile %ux%u\n",
                r.tensors.size(), (unsigned long long)r.bytes_copied, r.total_ms, gbps(r.bytes_copied, r.total_ms),
                (unsigned long long)r.arena_bytes, opt.tile.rows, opt.tile.cols);
    if (opt.verify || opt.checksum)
      std::printf("%s%s passed for all tensors (%.1f ms)\n", opt.verify ? "verify" : "",
                  opt.checksum ? (opt.verify ? " + checksum" : "checksum") : "", r.verify_ms);
    free_device_arena(r);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
