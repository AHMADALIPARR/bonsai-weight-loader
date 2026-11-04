// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Reference path, kept exactly as written. The production path is src/loader.cu.
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

// Direct binary loader for Bonsai checkpoint tensors
bool load_bonsai_tensor_direct(const char* filepath, void* d_dest, size_t expected_bytes) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    std::vector<char> buffer(expected_bytes);
    file.read(buffer.data(), expected_bytes);
    
    // Direct host-to-device copy, bypassing framework layers
    cudaMemcpy(d_dest, buffer.data(), expected_bytes, cudaMemcpyHostToDevice);
    return true;
}
