// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Tiled device layout. See docs/tiled-layout.md.
//
// A tensor with ne = [ne0, ne1, ...] is viewed as a 2-D matrix of
//   rows = ne1 * ne2 * ...,  units_per_row = ne0 / block_elems
// where one "unit" is one element for scalar types (F32, F16, BF16, ...) and one
// quant block for block types (PTQ1_0 = 128 weights in 28 bytes, Q4_0 = 32 in 18).
// Tiles are tile_rows x tile_cols units, stored tile after tile in row-major
// tile order; inside a tile, units are row-major. The matrix is zero-padded up
// to a whole number of tiles in both directions. Units are never split, so a
// quant block always stays intact.
//
// 1-D tensors and tensors of unknown type are copied linearly (not tiled).
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

#include "bonsai/checkpoint.hpp"
#include "bonsai/common.hpp"

namespace bonsai {

struct TileConfig {
  uint32_t rows = 16;  // tile height in rows
  uint32_t cols = 16;  // tile width in units (elements or quant blocks)
};

struct TensorLayout {
  bool tiled = false;
  uint64_t rows = 0;
  uint64_t units_per_row = 0;
  uint64_t unit_bytes = 0;
  uint64_t row_bytes = 0;
  uint64_t tile_rows = 0;
  uint64_t tile_cols = 0;
  uint64_t tiles_r = 0;
  uint64_t tiles_c = 0;
  uint64_t src_bytes = 0;  // bytes in the checkpoint
  uint64_t dst_bytes = 0;  // bytes in the device arena (with tile padding)
};

TensorLayout make_layout(const TensorInfo& t, const TileConfig& cfg);

// Destination byte offset (inside the tensor's arena slot) of source byte
// (row r, byte b within the row). Reference mapping used by the CPU tests and
// by the device kernel.
BONSAI_HD uint64_t tiled_offset(uint64_t r, uint64_t b, uint64_t unit_bytes, uint64_t tile_rows,
                                uint64_t tile_cols, uint64_t tiles_c) {
  const uint64_t u = b / unit_bytes, ub = b - u * unit_bytes;
  const uint64_t tr = r / tile_rows, ir = r - tr * tile_rows;
  const uint64_t tc = u / tile_cols, ic = u - tc * tile_cols;
  const uint64_t tile_bytes = tile_rows * tile_cols * unit_bytes;
  return (tr * tiles_c + tc) * tile_bytes + (ir * tile_cols + ic) * unit_bytes + ub;
}

inline uint64_t tiled_offset(const TensorLayout& L, uint64_t r, uint64_t b) {
  return tiled_offset(r, b, L.unit_bytes, L.tile_rows, L.tile_cols, L.tiles_c);
}

// Retile rows [row_begin, row_begin + nrows) from src (row-major, starting at
// row_begin) into dst (the tensor's full tiled slot). Copies whole tile-row
// segments with memcpy. Padding bytes are not touched.
void retile_rows_cpu(const TensorLayout& L, const uint8_t* src, uint64_t row_begin, uint64_t nrows,
                     uint8_t* dst);

// Byte-at-a-time reference of retile_rows_cpu, used only by tests.
void retile_rows_naive(const TensorLayout& L, const uint8_t* src, uint64_t row_begin, uint64_t nrows,
                       uint8_t* dst);

// Inverse: tiled slot -> row-major bytes (src_bytes long).
void untile_cpu(const TensorLayout& L, const uint8_t* tiled, uint8_t* out);

// Round up to the arena slot alignment.
inline uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

}  // namespace bonsai

namespace bonsai {

// Device arena plan: one slot per tensor, each slot aligned to `align` bytes.
struct ArenaPlan {
  std::vector<TensorLayout> layouts;
  std::vector<uint64_t> offsets;
  uint64_t total_bytes = 0;
};

ArenaPlan plan_arena(const Checkpoint& c, const TileConfig& cfg, uint64_t align = 256);

}  // namespace bonsai
