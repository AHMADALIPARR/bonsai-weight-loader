// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Production loader: mmap'd checkpoint -> one device arena in the tiled layout,
// streamed through two pinned staging buffers on two CUDA streams.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "bonsai/checkpoint.hpp"
#include "bonsai/tile_layout.hpp"

namespace bonsai {

struct LoadOptions {
  TileConfig tile;
  size_t chunk_bytes = 64ull << 20;  // per staging buffer
  uint64_t arena_align = 256;
  bool verify = false;    // read every tensor back and compare to a CPU retile
  bool checksum = false;  // FNV-1a 64 of source bytes and of the un-tiled device copy
  int device = 0;
};

struct TensorLoad {
  std::string name;
  uint32_t type = 0;
  TensorLayout layout;
  uint64_t arena_offset = 0;
  uint64_t bytes_copied = 0;
  double ms = 0.0;  // first chunk enqueued -> last chunk retiled (device events)
  bool verified = false;
  uint64_t fnv_src = 0, fnv_dev = 0;
};

struct LoadReport {
  void* d_arena = nullptr;
  uint64_t arena_bytes = 0;
  uint64_t bytes_copied = 0;
  double total_ms = 0.0;   // wall clock, arena allocation excluded
  double verify_ms = 0.0;
  std::vector<TensorLoad> tensors;
};

// file_data must cover the checkpoint (data_offset + tensors). Throws on any
// CUDA error or byte-count mismatch.
LoadReport load_checkpoint_to_device(const uint8_t* file_data, const Checkpoint& ckpt,
                                     const LoadOptions& opt);

void free_device_arena(LoadReport& r);

}  // namespace bonsai
