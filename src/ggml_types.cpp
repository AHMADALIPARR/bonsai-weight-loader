// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// GGML type table.
#include "bonsai/ggml_types.hpp"

namespace bonsai {

namespace {
struct Entry {
  uint32_t id;
  TypeTraits t;
};
const Entry kTypes[] = {
    {GGML_F32, {"F32", 1, 4, true}},
    {GGML_F16, {"F16", 1, 2, true}},
    {GGML_Q4_0, {"Q4_0", 32, 18, true}},
    {GGML_Q4_1, {"Q4_1", 32, 20, true}},
    {GGML_Q8_0, {"Q8_0", 32, 34, true}},
    {GGML_I8, {"I8", 1, 1, true}},
    {GGML_I16, {"I16", 1, 2, true}},
    {GGML_I32, {"I32", 1, 4, true}},
    {GGML_I64, {"I64", 1, 8, true}},
    {GGML_F64, {"F64", 1, 8, true}},
    {GGML_BF16, {"BF16", 1, 2, true}},
    {GGML_TQ1_0, {"TQ1_0", 256, 54, true}},
    {GGML_TQ2_0, {"TQ2_0", 256, 66, true}},
    {GGML_PQ2_0, {"PQ2_0", 128, 34, true}},
    {GGML_PTQ1_0, {"PTQ1_0", 128, 28, true}},
};
}  // namespace

TypeTraits type_traits(uint32_t type) {
  for (const Entry& e : kTypes)
    if (e.id == type) return e.t;
  return {"unknown", 0, 0, false};
}

std::string type_name(uint32_t type) {
  TypeTraits t = type_traits(type);
  if (t.known) return t.name;
  return "type_" + std::to_string(type);
}

bool type_from_name(const std::string& name, uint32_t* out) {
  for (const Entry& e : kTypes) {
    if (name == e.t.name) {
      *out = e.id;
      return true;
    }
  }
  return false;
}

}  // namespace bonsai
