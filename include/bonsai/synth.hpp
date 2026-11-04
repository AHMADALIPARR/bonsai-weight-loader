// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Small synthetic checkpoints for tests: a GGUF v3 file with the same tensor
// types as the Bonsai PTQ1_0 file (PTQ1_0, F32, BF16) plus F16, Q4_0 and one
// tensor of a type this loader has no geometry for.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace bonsai {

struct SynthTensor {
  std::string name;
  std::vector<uint64_t> ne;
  uint32_t type;
  std::vector<uint8_t> data;
};

std::vector<SynthTensor> make_synth_tensors(uint32_t seed);

// Writes GGUF v3 with general.alignment = alignment. Returns the file size.
uint64_t write_synth_gguf(const std::string& path, const std::vector<SynthTensor>& tensors,
                          uint32_t alignment = 32);

// Writes a raw blob plus manifest (tensors of unknown type are skipped).
void write_synth_raw(const std::string& blob_path, const std::string& manifest_path,
                     const std::vector<SynthTensor>& tensors);

}  // namespace bonsai
