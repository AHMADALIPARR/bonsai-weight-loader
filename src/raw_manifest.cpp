// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Raw blob + text manifest parser.
#include <sstream>
#include <stdexcept>

#include "bonsai/checkpoint.hpp"
#include "bonsai/ggml_types.hpp"

namespace bonsai {

Checkpoint parse_raw_manifest(const std::string& text, uint64_t blob_bytes) {
  Checkpoint c;
  c.format = "raw";
  c.alignment = 1;
  c.data_offset = 0;
  c.file_bytes = blob_bytes;
  std::istringstream in(text);
  std::string line;
  int lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    size_t first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    std::istringstream ls(line);
    TensorInfo t;
    std::string dtype, shape;
    if (!(ls >> t.name >> t.offset >> t.bytes >> dtype >> shape))
      throw std::runtime_error("manifest line " + std::to_string(lineno) + ": expected 5 fields");
    if (!type_from_name(dtype, &t.type))
      throw std::runtime_error("manifest line " + std::to_string(lineno) + ": unknown dtype " + dtype);
    std::istringstream ss(shape);
    std::string dim;
    t.n_elements = 1;
    while (std::getline(ss, dim, ',')) {
      uint64_t v = std::stoull(dim);
      if (v == 0) throw std::runtime_error("manifest line " + std::to_string(lineno) + ": zero dim");
      t.ne.push_back(v);
      t.n_elements *= v;
    }
    if (t.ne.empty()) throw std::runtime_error("manifest line " + std::to_string(lineno) + ": empty shape");
    TypeTraits tt = type_traits(t.type);
    if (t.ne[0] % tt.block_elems != 0 ||
        t.n_elements / tt.block_elems * tt.block_bytes != t.bytes)
      throw std::runtime_error("manifest line " + std::to_string(lineno) + ": bytes do not match " +
                               dtype + " x shape");
    if (t.offset + t.bytes > blob_bytes || t.offset + t.bytes < t.offset)
      throw std::runtime_error("manifest line " + std::to_string(lineno) + ": past end of blob");
    t.file_offset = t.offset;
    t.size_from_type = true;
    c.tensors.push_back(std::move(t));
  }
  return c;
}

}  // namespace bonsai
