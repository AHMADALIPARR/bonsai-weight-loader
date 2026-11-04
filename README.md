# bonsai-weight-loader

![license](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20SnapKitty--Opaque--1.0-blue)
![C++17](https://img.shields.io/badge/C%2B%2B-17-informational)
![CUDA](https://img.shields.io/badge/CUDA-12-76B900)

A small C++17/CUDA reader that loads the Bonsai checkpoint
(`Ternary-Bonsai-2-27B-PTQ1_0.gguf`, about 5.95 GB) straight into GPU memory
in a tiled layout. It does not go through a model framework.

It works like this:

1. Memory-maps the file and parses the GGUF header for every tensor's name,
   type, shape and offset.
2. Allocates one device arena.
3. Streams the tensors through two pinned staging buffers with
   `cudaMemcpyAsync` on two streams.
4. Retiles each tensor on the device. Quant blocks are kept whole.
5. Checks every CUDA call and every byte count. `--verify` compares the device
   copy with a CPU reference.

A raw blob plus a text manifest works as well as GGUF.

`src/load_bonsai_tensor_direct.cpp` keeps the minimal path: one `ifstream`
read and one `cudaMemcpy`.

The repo also has two kernels that use the loaded weights:

- `apply_block_scale_stage`: a vectorized block-scale output stage, plus a
  fused tiled-GEMM epilogue version.
- `custom_int4_tiled_gemm_kernel`: a GEMM over packed PTQ1_0 ternary or Q4_0
  weights.

Docs: [format](docs/format.md), [tiled layout](docs/tiled-layout.md),
[packing](docs/packing.md), [Nsight Compute checks](docs/nsight-compute.md),
[ptxas report](docs/ptxas-report.md).

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build -L cpu      # no GPU needed
ctest --test-dir build -L gpu      # on a machine with a GPU
```

Without nvcc, only the CPU library, tools and tests are built. Set
`-DCMAKE_CUDA_ARCHITECTURES=86` (or similar) to build for one GPU.

## Run

```sh
build/bonsai_index Ternary-Bonsai-2-27B-PTQ1_0.gguf            # header summary
build/bonsai_load  Ternary-Bonsai-2-27B-PTQ1_0.gguf --tile 16 16 --verify
build/bonsai_load  blob.bin --raw blob.manifest                # raw mode
build/gpu_bench
```

`bonsai_load` prints each tensor's size, time and GB/s, then the total.

## Benchmarks

The CPU numbers are from `build/cpu_bench --header <real header>` on an
8-core Xeon box. The GPU columns are blank until they are run on a GPU.

| benchmark | size | CPU | GPU |
|---|---|---|---|
| header parse (851 tensors) | 11.1 MB | 4.89 ms | |
| retile 16x16, PTQ1_0 `output.weight` [5120, 248320] | 278.1 MB | 5.48 GB/s | |
| retile 16x16, BF16 `blk.0.ssm_alpha.weight` | 0.5 MB | 4.19 GB/s | |
| block-scale stage fp32, 1x128 blocks | 4096² | 4.81 GB/s | |
| block-scale stage fp32 / fp16 / bf16 | 8192² | | |
| fp32 GEMM, fused vs standalone block scale | 2048³ | | |
| PTQ1_0 GEMM, M=8 N=1024 K=5120 | 1.1 MB weights | 2.51 GFLOP/s | |
| Q4_0 GEMM, M=8 N=1024 K=5120 | 2.9 MB weights | 2.97 GFLOP/s | |
| `custom_int4_tiled_gemm_kernel` PTQ1_0, M=1/16/128, N=K=5120 | | | |
| full 27B checkpoint to device (`bonsai_load`) | 5.95 GB | | |

## License

`AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0`. See [LICENSE](LICENSE).
