// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
// Shared macros for host and device code.
#pragma once

#if defined(__CUDACC__)
#define BONSAI_HD __host__ __device__ __forceinline__
#else
#define BONSAI_HD inline
#endif
