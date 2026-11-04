// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// GGUF header parsing on the synthetic checkpoint, plus malformed inputs.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "bonsai/checkpoint.hpp"
#include "bonsai/ggml_types.hpp"
#include "bonsai/mapped_file.hpp"
#include "bonsai/synth.hpp"
#include "check.hpp"

using namespace bonsai;

static std::vector<uint8_t> file_bytes(const std::string& p) {
  std::string s = read_text_file(p);
  return std::vector<uint8_t>(s.begin(), s.end());
}

static void put_u64(std::vector<uint8_t>& v, size_t at, uint64_t x) { std::memcpy(&v[at], &x, 8); }

int main() {
  const std::string path = bonsai_test::tmp_path("bonsai_test_parse.gguf");
  auto tensors = make_synth_tensors(7);
  write_synth_gguf(path, tensors, 32);

  MappedFile f(path);
  Checkpoint c = parse_gguf(f.data(), f.size());
  CHECK(c.format == "gguf");
  CHECK(c.version == 3);
  CHECK(c.alignment == 32);
  CHECK(c.data_offset % 32 == 0);
  CHECK(!c.header_only);
  CHECK(c.kv.size() == 7);
  CHECK(c.find_kv("general.architecture") && c.find_kv("general.architecture")->value == "bonsai-synth");
  CHECK(c.find_kv("general.alignment") && c.find_kv("general.alignment")->value == "32");
  CHECK(c.find_kv("synth.flag") && c.find_kv("synth.flag")->value == "true");
  CHECK(c.find_kv("missing.key") == nullptr);
  CHECK(c.tensors.size() == tensors.size());
  for (size_t i = 0; i < tensors.size() && i < c.tensors.size(); ++i) {
    const TensorInfo& t = c.tensors[i];
    const SynthTensor& s = tensors[i];
    CHECK(t.name == s.name);
    CHECK(t.type == s.type);
    CHECK(t.ne == s.ne);
    CHECK(t.file_offset == c.data_offset + t.offset);
    if (type_traits(s.type).known) {
      CHECK(t.size_from_type);
      CHECK(t.bytes == s.data.size());
    } else {
      // unknown geometry: sized by the gap to the next tensor (includes padding)
      CHECK(!t.size_from_type);
      CHECK(t.bytes == (s.data.size() + 31) / 32 * 32);
    }
    CHECK(t.file_offset + s.data.size() <= f.size());
    CHECK(std::memcmp(f.data() + t.file_offset, s.data.data(), s.data.size()) == 0);
  }
  // PTQ1_0 geometry: 256 x 40 -> 2 blocks x 28 bytes per row
  CHECK(c.tensors[0].bytes == 40u * 2u * 28u);

  // Header-only parse of just the header bytes.
  Checkpoint h = parse_gguf(f.data(), c.data_offset, true);
  CHECK(h.header_only);
  CHECK(h.tensors.size() == c.tensors.size());
  CHECK(h.data_offset == c.data_offset);
  CHECK_THROWS(parse_gguf(f.data(), c.data_offset, false));  // data missing

  std::vector<uint8_t> good = file_bytes(path);
  // Every truncation inside the header must throw, never read out of bounds.
  for (size_t n = 0; n < c.data_offset - 32; ++n) CHECK_THROWS(parse_gguf(good.data(), n, true));
  {
    auto b = good;
    b[0] = 'X';
    CHECK_THROWS(parse_gguf(b.data(), b.size()));
  }
  {
    auto b = good;
    b[4] = 9;  // version 9
    CHECK_THROWS(parse_gguf(b.data(), b.size()));
  }
  {
    auto b = good;
    put_u64(b, 8, 1ull << 40);  // absurd tensor count
    CHECK_THROWS(parse_gguf(b.data(), b.size()));
  }
  {  // tensor offset made misaligned: find the first tensor's offset field
    auto b = good;
    // first tensor info starts right after the kv section; locate by name
    const std::string nm = tensors[1].name;
    auto it = std::search(b.begin(), b.end(), nm.begin(), nm.end());
    CHECK(it != b.end());
    size_t at = (size_t)(it - b.begin()) + nm.size() + 4 + 8 * tensors[1].ne.size() + 4;
    uint64_t off;
    std::memcpy(&off, &b[at], 8);
    put_u64(b, at, off + 1);
    CHECK_THROWS(parse_gguf(b.data(), b.size()));
    put_u64(b, at, 0);  // overlaps tensor 0
    CHECK_THROWS(parse_gguf(b.data(), b.size()));
  }
  {  // file cut inside the data section
    CHECK_THROWS(parse_gguf(good.data(), good.size() - 1));
  }
  std::remove(path.c_str());
  return bonsai_test::finish("test_gguf_parse");
}
