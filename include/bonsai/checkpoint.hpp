// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Tensor index of a checkpoint: parsed from a GGUF header or from a raw manifest.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bonsai {

struct TensorInfo {
  std::string name;
  std::vector<uint64_t> ne;  // GGUF order: ne[0] is the innermost (contiguous) dim
  uint32_t type = 0;
  uint64_t offset = 0;       // relative to the start of the tensor data section
  uint64_t file_offset = 0;  // absolute byte offset in the file
  uint64_t bytes = 0;        // bytes of this tensor in the file
  uint64_t n_elements = 0;
  bool size_from_type = false;  // true: bytes computed from type geometry and checked
                                // against the gap to the next tensor
};

struct KeyValue {
  std::string key;
  uint32_t type = 0;   // GGUF value type
  std::string value;   // printable form (arrays summarised)
};

struct Checkpoint {
  std::string format;  // "gguf" or "raw"
  uint32_t version = 0;
  uint64_t alignment = 1;
  uint64_t data_offset = 0;  // absolute offset of the tensor data section
  uint64_t file_bytes = 0;
  bool header_only = false;  // parsed without the tensor data present
  std::vector<KeyValue> kv;
  std::vector<TensorInfo> tensors;  // in file order

  const KeyValue* find_kv(const std::string& key) const;
  uint64_t total_tensor_bytes() const;
};

// Parse a GGUF v2/v3 header from memory. data/size cover the whole file (or just
// the header when allow_header_only is set). Throws std::runtime_error on any
// malformed or out-of-bounds field.
Checkpoint parse_gguf(const uint8_t* data, size_t size, bool allow_header_only = false);

// Raw mode: one blob plus a text manifest, one tensor per line:
//   <name> <offset> <bytes> <dtype> <ne0,ne1,...>
// offset is relative to the start of the blob, dtype is a ggml type name
// (F32, F16, BF16, Q4_0, PTQ1_0, ...), shape is innermost-first like GGUF.
// Blank lines and lines starting with '#' are ignored.
Checkpoint parse_raw_manifest(const std::string& manifest_text, uint64_t blob_bytes);

}  // namespace bonsai
