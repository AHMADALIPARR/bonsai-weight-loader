// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// GGML tensor type ids as stored in GGUF tensor infos, with block geometry.
// Ids 0..42 follow upstream ggml (ggml/include/ggml.h). Ids 142 and 143 are the
// Prism-private types from PrismML-Eng/llama.cpp (branch "prism").
#pragma once
#include <cstdint>
#include <string>

namespace bonsai {

enum GgmlType : uint32_t {
  GGML_F32 = 0,
  GGML_F16 = 1,
  GGML_Q4_0 = 2,
  GGML_Q4_1 = 3,
  GGML_Q8_0 = 8,
  GGML_I8 = 24,
  GGML_I16 = 25,
  GGML_I32 = 26,
  GGML_I64 = 27,
  GGML_F64 = 28,
  GGML_BF16 = 30,
  GGML_TQ1_0 = 34,
  GGML_TQ2_0 = 35,
  GGML_PQ2_0 = 142,   // Prism: 2-bit slots, group 128 (fp16 d + 32 B)
  GGML_PTQ1_0 = 143,  // Prism: dense trits, group 128 (24 B qs + 2 B qh + fp16 d)
};

struct TypeTraits {
  const char* name;
  uint32_t block_elems;  // elements per block (1 for plain scalar types)
  uint32_t block_bytes;  // bytes per block
  bool known;            // false: geometry unknown, size taken from the file
};

// Returns {"type_<id>", 0, 0, false} for ids this loader has no geometry for.
TypeTraits type_traits(uint32_t type);

// Name -> id for the raw manifest ("F32", "PTQ1_0", ...). Returns false if unknown.
bool type_from_name(const std::string& name, uint32_t* out);

std::string type_name(uint32_t type);

}  // namespace bonsai
