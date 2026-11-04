// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Production loader.
//
// For each tensor, chunks of whole rows are copied from the mmap'd file into one
// of two pinned (cudaHostAlloc) staging buffers, sent to a device staging buffer
// with cudaMemcpyAsync on that slot's stream, and retiled into the arena by
// retile_rows_kernel. While slot A's copy and kernel run, the host fills slot B.
// Linear (untiled) tensors skip the device staging buffer.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "bonsai/checksum.hpp"
#include "bonsai/cuda_check.cuh"
#include "bonsai/loader.hpp"

namespace bonsai {

bool cuda_device_available() {
  int n = 0;
  cudaError_t e = cudaGetDeviceCount(&n);
  if (e != cudaSuccess || n == 0) {
    std::fprintf(stderr, "no CUDA device: %s\n", e != cudaSuccess ? cudaGetErrorString(e) : "count 0");
    cudaGetLastError();
    return false;
  }
  return true;
}

// One warp per tile-row segment (row r, tile column tc): a contiguous run of
// tile_cols units in both source row and destination tile.
__global__ void retile_rows_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                   uint64_t row_begin, uint64_t nrows, uint64_t row_bytes,
                                   uint64_t unit_bytes, uint64_t tile_rows, uint64_t tile_cols,
                                   uint64_t tiles_c, int word_path) {
  const uint64_t seg_full = tile_cols * unit_bytes;
  const uint64_t nseg = nrows * tiles_c;
  const unsigned lane = threadIdx.x & 31u;
  const uint64_t warps = (uint64_t)gridDim.x * (blockDim.x >> 5);
  for (uint64_t s = (uint64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); s < nseg; s += warps) {
    const uint64_t i = s / tiles_c, tc = s - i * tiles_c;
    const uint64_t b0 = tc * seg_full;
    const uint64_t n = min(seg_full, row_bytes - b0);
    const uint64_t d = tiled_offset(row_begin + i, b0, unit_bytes, tile_rows, tile_cols, tiles_c);
    const uint8_t* sp = src + i * row_bytes + b0;
    uint8_t* dp = dst + d;
    if (word_path) {
      const uint32_t* s4 = reinterpret_cast<const uint32_t*>(sp);
      uint32_t* d4 = reinterpret_cast<uint32_t*>(dp);
      for (uint64_t w = lane; w < n / 4; w += 32) d4[w] = s4[w];
    } else {
      for (uint64_t b = lane; b < n; b += 32) dp[b] = sp[b];
    }
  }
}

namespace {

struct Slot {
  uint8_t* h = nullptr;
  uint8_t* d = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t done = nullptr;
  bool busy = false;
};

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

LoadReport load_checkpoint_to_device(const uint8_t* file, const Checkpoint& ckpt, const LoadOptions& opt) {
  if (ckpt.header_only) throw std::runtime_error("checkpoint was parsed header-only; no data to load");
  BONSAI_CUDA(cudaSetDevice(opt.device));
  ArenaPlan plan = plan_arena(ckpt, opt.tile, opt.arena_align);

  LoadReport rep;
  rep.arena_bytes = plan.total_bytes;
  size_t free_b = 0, total_b = 0;
  BONSAI_CUDA(cudaMemGetInfo(&free_b, &total_b));
  uint64_t stage = std::max<uint64_t>(opt.chunk_bytes, 1);
  for (const TensorLayout& L : plan.layouts)
    if (L.tiled) stage = std::max(stage, L.row_bytes);
  stage = align_up(stage, 4096);
  if (plan.total_bytes + 2 * stage > free_b)
    throw std::runtime_error("arena needs " + std::to_string(plan.total_bytes + 2 * stage) +
                             " bytes, device has " + std::to_string(free_b) + " free");

  uint8_t* arena = nullptr;
  BONSAI_CUDA(cudaMalloc(&arena, plan.total_bytes ? plan.total_bytes : 1));
  rep.d_arena = arena;
  BONSAI_CUDA(cudaMemset(arena, 0, plan.total_bytes));  // tile padding reads as zero

  Slot slots[2];
  for (Slot& s : slots) {
    BONSAI_CUDA(cudaHostAlloc(&s.h, stage, cudaHostAllocWriteCombined));
    BONSAI_CUDA(cudaMalloc(&s.d, stage));
    BONSAI_CUDA(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
    BONSAI_CUDA(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming));
  }
  std::vector<cudaEvent_t> ev(2 * ckpt.tensors.size());
  for (cudaEvent_t& e : ev) BONSAI_CUDA(cudaEventCreate(&e));

  int dev_sms = 0;
  BONSAI_CUDA(cudaDeviceGetAttribute(&dev_sms, cudaDevAttrMultiProcessorCount, opt.device));

  const double t0 = now_ms();
  unsigned k = 0;
  for (size_t ti = 0; ti < ckpt.tensors.size(); ++ti) {
    const TensorInfo& t = ckpt.tensors[ti];
    const TensorLayout& L = plan.layouts[ti];
    TensorLoad tl;
    tl.name = t.name;
    tl.type = t.type;
    tl.layout = L;
    tl.arena_offset = plan.offsets[ti];
    const uint8_t* src = file + t.file_offset;
    uint8_t* dst = arena + plan.offsets[ti];
    const uint64_t chunk_rows = L.tiled ? std::max<uint64_t>(1, stage / L.row_bytes) : 0;
    const int word_path = (L.row_bytes % 4 == 0 && (L.tile_cols * L.unit_bytes) % 4 == 0) ? 1 : 0;

    uint64_t pos = 0, row = 0;
    bool first = true;
    while (pos < t.bytes) {
      const uint64_t n = L.tiled ? std::min(chunk_rows, L.rows - row) * L.row_bytes
                                 : std::min<uint64_t>(stage, t.bytes - pos);
      Slot& s = slots[k++ & 1u];
      if (s.busy) BONSAI_CUDA(cudaEventSynchronize(s.done));
      std::memcpy(s.h, src + pos, n);
      if (first) BONSAI_CUDA(cudaEventRecord(ev[2 * ti], s.stream));
      first = false;
      if (L.tiled) {
        BONSAI_CUDA(cudaMemcpyAsync(s.d, s.h, n, cudaMemcpyHostToDevice, s.stream));
        const uint64_t nr = n / L.row_bytes;
        const uint64_t nseg = nr * L.tiles_c;
        const unsigned blocks = (unsigned)std::min<uint64_t>((nseg + 7) / 8, (uint64_t)dev_sms * 32);
        retile_rows_kernel<<<blocks, 256, 0, s.stream>>>(s.d, dst, row, nr, L.row_bytes, L.unit_bytes,
                                                        L.tile_rows, L.tile_cols, L.tiles_c, word_path);
        BONSAI_CUDA_LAUNCH();
        row += nr;
      } else {
        BONSAI_CUDA(cudaMemcpyAsync(dst + pos, s.h, n, cudaMemcpyHostToDevice, s.stream));
      }
      pos += n;
      tl.bytes_copied += n;
      if (pos >= t.bytes) {
        Slot& other = slots[k & 1u];
        if (other.busy) BONSAI_CUDA(cudaStreamWaitEvent(s.stream, other.done, 0));
        BONSAI_CUDA(cudaEventRecord(ev[2 * ti + 1], s.stream));
      }
      BONSAI_CUDA(cudaEventRecord(s.done, s.stream));
      s.busy = true;
    }
    if (t.bytes == 0) {
      BONSAI_CUDA(cudaEventRecord(ev[2 * ti], slots[0].stream));
      BONSAI_CUDA(cudaEventRecord(ev[2 * ti + 1], slots[0].stream));
    }
    if (tl.bytes_copied != t.bytes || (L.tiled && row != L.rows))
      throw std::runtime_error("byte count mismatch for " + t.name);
    rep.bytes_copied += tl.bytes_copied;
    rep.tensors.push_back(std::move(tl));
  }
  for (Slot& s : slots) BONSAI_CUDA(cudaStreamSynchronize(s.stream));
  rep.total_ms = now_ms() - t0;
  for (size_t ti = 0; ti < rep.tensors.size(); ++ti) {
    float ms = 0.0f;
    BONSAI_CUDA(cudaEventElapsedTime(&ms, ev[2 * ti], ev[2 * ti + 1]));
    rep.tensors[ti].ms = ms;
  }
  if (rep.bytes_copied != ckpt.total_tensor_bytes()) throw std::runtime_error("total byte count mismatch");

  if (opt.verify || opt.checksum) {
    const double v0 = now_ms();
    std::vector<uint8_t> dev, expect, flat;
    for (size_t ti = 0; ti < rep.tensors.size(); ++ti) {
      TensorLoad& tl = rep.tensors[ti];
      const TensorInfo& t = ckpt.tensors[ti];
      const TensorLayout& L = tl.layout;
      dev.resize(L.dst_bytes);
      BONSAI_CUDA(cudaMemcpy(dev.data(), arena + tl.arena_offset, L.dst_bytes, cudaMemcpyDeviceToHost));
      const uint8_t* src = file + t.file_offset;
      if (opt.verify) {
        expect.assign(L.dst_bytes, 0);
        retile_rows_cpu(L, src, 0, L.rows, expect.data());
        tl.verified = std::memcmp(dev.data(), expect.data(), L.dst_bytes) == 0;
        if (!tl.verified) throw std::runtime_error("verify failed for " + t.name);
      }
      if (opt.checksum) {
        flat.resize(L.src_bytes);
        untile_cpu(L, dev.data(), flat.data());
        tl.fnv_src = fnv1a64(src, L.src_bytes);
        tl.fnv_dev = fnv1a64(flat.data(), flat.size());
        if (tl.fnv_src != tl.fnv_dev) throw std::runtime_error("checksum mismatch for " + t.name);
      }
    }
    rep.verify_ms = now_ms() - v0;
  }

  for (cudaEvent_t e : ev) BONSAI_CUDA(cudaEventDestroy(e));
  for (Slot& s : slots) {
    BONSAI_CUDA(cudaEventDestroy(s.done));
    BONSAI_CUDA(cudaStreamDestroy(s.stream));
    BONSAI_CUDA(cudaFree(s.d));
    BONSAI_CUDA(cudaFreeHost(s.h));
  }
  return rep;
}

void free_device_arena(LoadReport& r) {
  if (r.d_arena) BONSAI_CUDA(cudaFree(r.d_arena));
  r.d_arena = nullptr;
}

}  // namespace bonsai
