// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// make_synth_gguf <out.gguf> [--raw <blob> <manifest>] [--seed N]
// Writes the small synthetic checkpoint used by the tests.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include "bonsai/synth.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: make_synth_gguf <out.gguf> [--raw <blob> <manifest>] [--seed N]\n");
    return 2;
  }
  std::string out = argv[1], blob, manifest;
  uint32_t seed = 1234;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--raw") && i + 2 < argc) {
      blob = argv[++i];
      manifest = argv[++i];
    } else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) {
      seed = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
    } else {
      std::fprintf(stderr, "unknown argument %s\n", argv[i]);
      return 2;
    }
  }
  try {
    auto tensors = bonsai::make_synth_tensors(seed);
    uint64_t bytes = bonsai::write_synth_gguf(out, tensors);
    std::printf("wrote %s (%llu bytes, %zu tensors)\n", out.c_str(), (unsigned long long)bytes, tensors.size());
    if (!blob.empty()) {
      bonsai::write_synth_raw(blob, manifest, tensors);
      std::printf("wrote %s + %s\n", blob.c_str(), manifest.c_str());
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
