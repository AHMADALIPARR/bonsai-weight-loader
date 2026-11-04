// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// bonsai_index <ckpt> [--raw <manifest>] [--header-only] [--tile R C] [--list] [--manifest]
// CPU-only: parses the tensor index and prints the device arena plan.
// --manifest prints a raw-mode manifest (absolute offsets into <ckpt>).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <string>

#include "bonsai/tile_layout.hpp"
#include "cli_args.hpp"

using namespace bonsai;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: bonsai_index <ckpt> [--raw <manifest>] [--header-only] [--tile R C] [--list] "
                 "[--manifest]\n");
    return 2;
  }
  std::string path = argv[1], manifest;
  bool header_only = false, list = false, emit_manifest = false;
  TileConfig cfg;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--raw") && i + 1 < argc) manifest = argv[++i];
    else if (!std::strcmp(argv[i], "--header-only")) header_only = true;
    else if (!std::strcmp(argv[i], "--list")) list = true;
    else if (!std::strcmp(argv[i], "--manifest")) emit_manifest = true;
    else if (!std::strcmp(argv[i], "--tile") && i + 2 < argc) {
      cfg.rows = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
      cfg.cols = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
    } else {
      std::fprintf(stderr, "unknown argument %s\n", argv[i]);
      return 2;
    }
  }
  try {
    MappedFile f(path);
    Checkpoint c = open_checkpoint(f, manifest, header_only);
    if (emit_manifest) {
      std::printf("# name offset bytes dtype shape  (offsets are absolute in %s)\n", path.c_str());
      for (const TensorInfo& t : c.tensors) {
        if (!type_traits(t.type).known) {
          std::printf("# skipped %s: type %u has no known geometry\n", t.name.c_str(), t.type);
          continue;
        }
        std::printf("%s %llu %llu %s ", t.name.c_str(), (unsigned long long)t.file_offset,
                    (unsigned long long)t.bytes, type_name(t.type).c_str());
        for (size_t d = 0; d < t.ne.size(); ++d) std::printf("%s%llu", d ? "," : "", (unsigned long long)t.ne[d]);
        std::printf("\n");
      }
      return 0;
    }
    ArenaPlan plan = plan_arena(c, cfg);
    std::map<std::string, std::pair<int, uint64_t>> by_type;
    for (const TensorInfo& t : c.tensors) {
      auto& e = by_type[type_name(t.type)];
      e.first++;
      e.second += t.bytes;
    }
    std::printf("file          %s (%llu bytes%s)\n", path.c_str(), (unsigned long long)f.size(),
                c.header_only ? ", header only" : "");
    std::printf("format        %s v%u, alignment %llu, data at %llu\n", c.format.c_str(), c.version,
                (unsigned long long)c.alignment, (unsigned long long)c.data_offset);
    if (const KeyValue* a = c.find_kv("general.architecture")) std::printf("architecture  %s\n", a->value.c_str());
    std::printf("kv pairs      %zu\n", c.kv.size());
    std::printf("tensors       %zu, %llu bytes\n", c.tensors.size(), (unsigned long long)c.total_tensor_bytes());
    for (auto& [name, e] : by_type)
      std::printf("  %-8s %5d tensors %14llu bytes\n", name.c_str(), e.first, (unsigned long long)e.second);
    std::printf("tile          %u rows x %u units\n", cfg.rows, cfg.cols);
    std::printf("device arena  %llu bytes\n", (unsigned long long)plan.total_bytes);
    if (list) {
      for (size_t i = 0; i < c.tensors.size(); ++i) {
        const TensorInfo& t = c.tensors[i];
        const TensorLayout& L = plan.layouts[i];
        std::printf("%-40s %-7s ", t.name.c_str(), type_name(t.type).c_str());
        print_shape(t);
        std::printf(" off %llu bytes %llu -> arena %llu (%s, %llu bytes)\n", (unsigned long long)t.file_offset,
                    (unsigned long long)t.bytes, (unsigned long long)plan.offsets[i],
                    L.tiled ? "tiled" : "linear", (unsigned long long)L.dst_bytes);
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
