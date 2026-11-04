// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// FNV-1a 64-bit checksum, streamable.
#pragma once
#include <cstddef>
#include <cstdint>

namespace bonsai {

constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ull;

inline uint64_t fnv1a64(const void* data, size_t n, uint64_t h = kFnvOffset) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 0x100000001b3ull;
  }
  return h;
}

}  // namespace bonsai
