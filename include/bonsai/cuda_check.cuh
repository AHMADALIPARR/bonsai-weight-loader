// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// CUDA error checking: every runtime call goes through BONSAI_CUDA.
#pragma once
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define BONSAI_CUDA(expr)                                                                   \
  do {                                                                                      \
    cudaError_t bonsai_err_ = (expr);                                                       \
    if (bonsai_err_ != cudaSuccess)                                                         \
      throw std::runtime_error(std::string(#expr) + " failed at " + __FILE__ + ":" +        \
                               std::to_string(__LINE__) + ": " +                           \
                               cudaGetErrorString(bonsai_err_));                            \
  } while (0)

#define BONSAI_CUDA_LAUNCH() BONSAI_CUDA(cudaGetLastError())

namespace bonsai {
// Returns false (and prints why) when no CUDA device is usable.
bool cuda_device_available();
}  // namespace bonsai
