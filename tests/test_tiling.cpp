// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Tiling index math against a byte-wise CPU reference, on the synthetic checkpoint.
#include <cstdio>
#include <cstring>
#include <vector>

#include "bonsai/checkpoint.hpp"
#include "bonsai/ggml_types.hpp"
#include "bonsai/mapped_file.hpp"
#include "bonsai/synth.hpp"
#include "bonsai/tile_layout.hpp"
#include "check.hpp"

using namespace bonsai;

int main() {
  // Hand-checked example: 5 rows x 3 units of 2 bytes, 2 x 2 tiles.
  {
    TensorInfo t;
    t.name = "x";
    t.type = 1;  // F16
    t.ne = {3, 5};
    t.n_elements = 15;
    t.bytes = 30;
    TensorLayout L = make_layout(t, TileConfig{2, 2});
    CHECK(L.tiled && L.rows == 5 && L.units_per_row == 3 && L.unit_bytes == 2);
    CHECK(L.tiles_r == 3 && L.tiles_c == 2);
    CHECK(L.dst_bytes == 3u * 2u * (2u * 2u * 2u));
    CHECK(tiled_offset(L, 0, 0) == 0);
    CHECK(tiled_offset(L, 0, 2) == 2);       // unit 1 of row 0
    CHECK(tiled_offset(L, 0, 4) == 8);       // unit 2 -> tile (0,1)
    CHECK(tiled_offset(L, 1, 0) == 4);       // row 1, unit 0 -> second row of tile (0,0)
    CHECK(tiled_offset(L, 2, 0) == 16);      // tile (1,0)
    CHECK(tiled_offset(L, 4, 5) == 5 * 8 + 1);  // unit 2 -> tile (2,1) = tile #5, byte 1
  }

  const std::string path = bonsai_test::tmp_path("bonsai_test_tiling.gguf");
  auto tensors = make_synth_tensors(11);
  write_synth_gguf(path, tensors);
  MappedFile f(path);
  Checkpoint c = parse_gguf(f.data(), f.size());

  const TileConfig cfgs[] = {{1, 1}, {16, 16}, {3, 5}, {7, 2}, {4, 64}, {64, 1}};
  for (const TileConfig& cfg : cfgs) {
    for (const TensorInfo& t : c.tensors) {
      TensorLayout L = make_layout(t, cfg);
      const uint8_t* src = f.data() + t.file_offset;
      CHECK(L.src_bytes == t.bytes);
      if (!L.tiled) {
        CHECK(L.dst_bytes == t.bytes);
        CHECK(t.ne.size() == 1 || !type_traits(t.type).known);
        continue;
      }
      CHECK(L.rows * L.row_bytes == t.bytes);
      CHECK(L.dst_bytes == L.tiles_r * L.tiles_c * cfg.rows * cfg.cols * L.unit_bytes);
      // every source byte lands on a distinct destination byte
      std::vector<uint8_t> hit(L.dst_bytes, 0);
      bool distinct = true;
      for (uint64_t r = 0; r < L.rows; ++r)
        for (uint64_t b = 0; b < L.row_bytes; ++b) {
          uint64_t d = tiled_offset(L, r, b);
          if (d >= L.dst_bytes || hit[d]) distinct = false;
          else hit[d] = 1;
        }
      CHECK(distinct);
      // quant blocks stay contiguous
      for (uint64_t r = 0; r < L.rows; ++r)
        for (uint64_t u = 0; u < L.units_per_row; ++u)
          CHECK(tiled_offset(L, r, u * L.unit_bytes + L.unit_bytes - 1) ==
                tiled_offset(L, r, u * L.unit_bytes) + L.unit_bytes - 1);
      // fast path (chunks of rows, as the loader streams them) == byte-wise reference
      std::vector<uint8_t> ref(L.dst_bytes, 0), fast(L.dst_bytes, 0);
      retile_rows_naive(L, src, 0, L.rows, ref.data());
      for (uint64_t r0 = 0; r0 < L.rows; r0 += 3)
        retile_rows_cpu(L, src + r0 * L.row_bytes, r0, std::min<uint64_t>(3, L.rows - r0), fast.data());
      CHECK(ref == fast);
      // padding stays zero
      uint64_t nonzero_pad = 0;
      for (uint64_t i = 0; i < L.dst_bytes; ++i)
        if (!hit[i] && ref[i]) ++nonzero_pad;
      CHECK(nonzero_pad == 0);
      std::vector<uint8_t> back(L.src_bytes);
      untile_cpu(L, ref.data(), back.data());
      CHECK(std::memcmp(back.data(), src, L.src_bytes) == 0);
    }
    ArenaPlan p = plan_arena(c, cfg, 256);
    for (size_t i = 0; i < p.offsets.size(); ++i) {
      CHECK(p.offsets[i] % 256 == 0);
      if (i + 1 < p.offsets.size()) CHECK(p.offsets[i] + p.layouts[i].dst_bytes <= p.offsets[i + 1]);
    }
  }
  std::remove(path.c_str());
  return bonsai_test::finish("test_tiling");
}
