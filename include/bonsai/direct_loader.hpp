// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Reference loader: reads expected_bytes from the start of filepath and copies
// them to d_dest with one blocking cudaMemcpy.
#pragma once
#include <cstddef>

bool load_bonsai_tensor_direct(const char* filepath, void* d_dest, size_t expected_bytes);
