// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Tile layout math and CPU retiling.
#include "bonsai/tile_layout.hpp"

#include <algorithm>
#include <stdexcept>

#include "bonsai/ggml_types.hpp"

namespace bonsai {

TensorLayout make_layout(const TensorInfo& t, const TileConfig& cfg) {
  if (cfg.rows == 0 || cfg.cols == 0) throw std::runtime_error("tile dims must be > 0");
  TensorLayout L;
  L.src_bytes = t.bytes;
  TypeTraits tt = type_traits(t.type);
  if (t.ne.size() < 2 || !tt.known) {
    L.tiled = false;
    L.rows = 1;
    L.units_per_row = t.bytes;
    L.unit_bytes = 1;
    L.row_bytes = t.bytes;
    L.dst_bytes = t.bytes;
    return L;
  }
  L.tiled = true;
  L.unit_bytes = tt.block_bytes;
  L.units_per_row = t.ne[0] / tt.block_elems;
  L.row_bytes = L.units_per_row * L.unit_bytes;
  L.rows = t.n_elements / t.ne[0];
  if (L.rows * L.row_bytes != t.bytes)
    throw std::runtime_error("layout: rows x row_bytes != tensor bytes for " + t.name);
  L.tile_rows = cfg.rows;
  L.tile_cols = cfg.cols;
  L.tiles_r = (L.rows + L.tile_rows - 1) / L.tile_rows;
  L.tiles_c = (L.units_per_row + L.tile_cols - 1) / L.tile_cols;
  L.dst_bytes = L.tiles_r * L.tiles_c * L.tile_rows * L.tile_cols * L.unit_bytes;
  return L;
}

void retile_rows_cpu(const TensorLayout& L, const uint8_t* src, uint64_t row_begin, uint64_t nrows,
                     uint8_t* dst) {
  if (!L.tiled) {
    std::memcpy(dst + row_begin * L.row_bytes, src, nrows * L.row_bytes);
    return;
  }
  const uint64_t seg_full = L.tile_cols * L.unit_bytes;
  for (uint64_t i = 0; i < nrows; ++i) {
    const uint64_t r = row_begin + i;
    const uint8_t* s = src + i * L.row_bytes;
    for (uint64_t tc = 0; tc < L.tiles_c; ++tc) {
      const uint64_t b = tc * seg_full;
      const uint64_t n = std::min(seg_full, L.row_bytes - b);
      std::memcpy(dst + tiled_offset(L, r, b), s + b, n);
    }
  }
}

void retile_rows_naive(const TensorLayout& L, const uint8_t* src, uint64_t row_begin, uint64_t nrows,
                       uint8_t* dst) {
  for (uint64_t i = 0; i < nrows; ++i)
    for (uint64_t b = 0; b < L.row_bytes; ++b) {
      uint64_t off = L.tiled ? tiled_offset(L, row_begin + i, b) : (row_begin + i) * L.row_bytes + b;
      dst[off] = src[i * L.row_bytes + b];
    }
}

void untile_cpu(const TensorLayout& L, const uint8_t* tiled, uint8_t* out) {
  if (!L.tiled) {
    std::memcpy(out, tiled, L.src_bytes);
    return;
  }
  const uint64_t seg_full = L.tile_cols * L.unit_bytes;
  for (uint64_t r = 0; r < L.rows; ++r)
    for (uint64_t tc = 0; tc < L.tiles_c; ++tc) {
      const uint64_t b = tc * seg_full;
      const uint64_t n = std::min(seg_full, L.row_bytes - b);
      std::memcpy(out + r * L.row_bytes + b, tiled + tiled_offset(L, r, b), n);
    }
}

}  // namespace bonsai

namespace bonsai {

ArenaPlan plan_arena(const Checkpoint& c, const TileConfig& cfg, uint64_t align) {
  ArenaPlan p;
  uint64_t off = 0;
  for (const TensorInfo& t : c.tensors) {
    p.layouts.push_back(make_layout(t, cfg));
    p.offsets.push_back(off);
    off = align_up(off + p.layouts.back().dst_bytes, align);
  }
  p.total_bytes = off;
  return p;
}

}  // namespace bonsai
