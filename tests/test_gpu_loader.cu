// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// GPU: load the synthetic checkpoint (GGUF and raw mode) with --verify and
// --checksum for several tile shapes; check the direct reference loader.
// Exits 77 (skipped) when no CUDA device is present.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "bonsai/cuda_check.cuh"
#include "bonsai/direct_loader.hpp"
#include "bonsai/loader.hpp"
#include "bonsai/mapped_file.hpp"
#include "bonsai/synth.hpp"
#include "check.hpp"

using namespace bonsai;

int main() {
  if (!cuda_device_available()) return 77;
  try {
    const std::string gguf = bonsai_test::tmp_path("bonsai_gpu_loader.gguf");
    const std::string blob = bonsai_test::tmp_path("bonsai_gpu_loader.bin");
    const std::string man = bonsai_test::tmp_path("bonsai_gpu_loader.txt");
    auto tensors = make_synth_tensors(17);
    write_synth_gguf(gguf, tensors);
    write_synth_raw(blob, man, tensors);

    MappedFile f(gguf);
    Checkpoint c = parse_gguf(f.data(), f.size());
    const TileConfig cfgs[] = {{16, 16}, {1, 1}, {3, 5}, {64, 8}};
    for (const TileConfig& cfg : cfgs) {
      for (size_t chunk : {size_t(64) << 20, size_t(512)}) {  // 512 B forces many chunks per tensor
        LoadOptions opt;
        opt.tile = cfg;
        opt.chunk_bytes = chunk;
        opt.verify = true;
        opt.checksum = true;
        LoadReport r = load_checkpoint_to_device(f.data(), c, opt);
        CHECK(r.bytes_copied == c.total_tensor_bytes());
        for (const TensorLoad& t : r.tensors) CHECK(t.verified && t.fnv_src == t.fnv_dev);
        free_device_arena(r);
      }
    }
    MappedFile fb(blob);
    Checkpoint rc = parse_raw_manifest(read_text_file(man), fb.size());
    LoadOptions opt;
    opt.verify = true;
    LoadReport rr = load_checkpoint_to_device(fb.data(), rc, opt);
    CHECK(rr.tensors.size() == rc.tensors.size());
    free_device_arena(rr);

    // Reference path: one tensor in its own file, one blocking copy.
    const std::string one = bonsai_test::tmp_path("bonsai_gpu_direct.bin");
    {
      std::ofstream o(one, std::ios::binary);
      o.write((const char*)tensors[0].data.data(), (std::streamsize)tensors[0].data.size());
    }
    void* d = nullptr;
    BONSAI_CUDA(cudaMalloc(&d, tensors[0].data.size()));
    CHECK(load_bonsai_tensor_direct(one.c_str(), d, tensors[0].data.size()));
    BONSAI_CUDA(cudaDeviceSynchronize());
    std::vector<uint8_t> back(tensors[0].data.size());
    BONSAI_CUDA(cudaMemcpy(back.data(), d, back.size(), cudaMemcpyDeviceToHost));
    CHECK(back == tensors[0].data);
    CHECK(!load_bonsai_tensor_direct("/nonexistent/bonsai.bin", d, 16));
    BONSAI_CUDA(cudaFree(d));
    for (const std::string& p : {gguf, blob, man, one}) std::remove(p.c_str());
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
  return bonsai_test::finish("test_gpu_loader");
}
