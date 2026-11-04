// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Shared helpers for the command-line tools.
#pragma once
#include <cstdio>
#include <string>

#include "bonsai/checkpoint.hpp"
#include "bonsai/ggml_types.hpp"
#include "bonsai/mapped_file.hpp"

namespace bonsai {

inline Checkpoint open_checkpoint(const MappedFile& f, const std::string& manifest, bool header_only) {
  if (!manifest.empty()) return parse_raw_manifest(read_text_file(manifest), f.size());
  return parse_gguf(f.data(), f.size(), header_only);
}

inline void print_shape(const TensorInfo& t) {
  std::printf("[");
  for (size_t d = 0; d < t.ne.size(); ++d) std::printf("%s%llu", d ? "," : "", (unsigned long long)t.ne[d]);
  std::printf("]");
}

}  // namespace bonsai
