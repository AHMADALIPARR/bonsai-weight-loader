// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Raw blob + manifest parsing.
#include <cstdio>
#include <cstring>

#include "bonsai/checkpoint.hpp"
#include "bonsai/ggml_types.hpp"
#include "bonsai/mapped_file.hpp"
#include "bonsai/synth.hpp"
#include "check.hpp"

using namespace bonsai;

int main() {
  const std::string blob = bonsai_test::tmp_path("bonsai_test_raw.bin");
  const std::string man = bonsai_test::tmp_path("bonsai_test_raw.txt");
  auto tensors = make_synth_tensors(3);
  write_synth_raw(blob, man, tensors);
  MappedFile f(blob);
  Checkpoint c = parse_raw_manifest(read_text_file(man), f.size());
  CHECK(c.format == "raw");
  size_t j = 0;
  for (const SynthTensor& s : tensors) {
    if (!type_traits(s.type).known) continue;
    CHECK(j < c.tensors.size());
    if (j >= c.tensors.size()) break;
    const TensorInfo& t = c.tensors[j++];
    CHECK(t.name == s.name && t.type == s.type && t.ne == s.ne && t.bytes == s.data.size());
    CHECK(std::memcmp(f.data() + t.file_offset, s.data.data(), t.bytes) == 0);
  }
  CHECK(j == c.tensors.size());

  CHECK(parse_raw_manifest("# only a comment\n\n", 0).tensors.empty());
  CHECK(parse_raw_manifest("w 0 616 PTQ1_0 256,11\n", 4096).tensors.size() == 1);
  CHECK_THROWS(parse_raw_manifest("w 0 100 F32\n", 4096));            // missing field
  CHECK_THROWS(parse_raw_manifest("w 0 100 F99 25\n", 4096));         // unknown dtype
  CHECK_THROWS(parse_raw_manifest("w 0 99 F32 25\n", 4096));          // bytes != 4 * 25
  CHECK_THROWS(parse_raw_manifest("w 4000 100 F32 25\n", 4096));      // past end of blob
  CHECK_THROWS(parse_raw_manifest("w 0 616 PTQ1_0 200,11\n", 4096)); // row not whole blocks
  CHECK_THROWS(parse_raw_manifest("w 0 0 F32 0\n", 4096));            // zero dim
  std::remove(blob.c_str());
  std::remove(man.c_str());
  return bonsai_test::finish("test_raw_manifest");
}
