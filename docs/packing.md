# Weight packing

Both formats are GGUF block formats. A row of `K` weights is `K / block_elems`
consecutive blocks. `custom_int4_tiled_gemm_kernel` reads `B` as `N` such
rows, one per output column (the GGUF `[K, N]` weight, `ne0 = K`). This is
the file's row layout, which is also what `--tile 1 C` produces when `C`
divides the blocks per row.

## PTQ1_0 (ternary, type 143): 128 weights in 28 bytes

Source: `block_ptq1_0` in `ggml/src/ggml-common.h` and
`dequantize_row_ptq1_0` in `ggml/src/ggml-quants.c`,
https://github.com/PrismML-Eng/llama.cpp/tree/prism

| bytes | content |
|---|---|
| 0-23 | `qs[24]`: 5 trits per byte |
| 24-25 | `qh[2]`: 4 trits per byte |
| 26-27 | `d`: fp16 scale, little-endian |

Each byte stores trits as a base-3 fraction. Trit `n` of byte `q` is

```
t = ((uint8_t)(q * 3^n) * 3) >> 8      // 0, 1, 2
w = (t - 1) * d                         // -d, 0, +d
```

Element `e` (0..127) of the block:

| e | byte | n |
|---|---|---|
| 0-79 | `qs[e & 15]` | `e >> 4` (0..4) |
| 80-119, f = e - 80 | `qs[16 + (f & 7)]` | `f >> 3` (0..4) |
| 120-127, f = e - 120 | `qh[f & 1]` | `f >> 1` (0..3) |

`ptq1_0_trit()` and `ptq1_0_scale()` in `include/bonsai/quant.hpp` are the
host/device accessors. The test checks them against a full-row dequantize.

## Q4_0 (type 2): 32 weights in 18 bytes

| bytes | content |
|---|---|
| 0-1 | `d`: fp16 scale |
| 2-17 | `qs[16]`: element `j` in the low nibble of `qs[j]`, element `j + 16` in the high nibble |

`w = (nibble - 8) * d`.

## The kernel

`custom_int4_tiled_gemm_kernel` (`src/int4_gemm.cu`):

- Block of 16 x 16 threads. Each thread owns rows `ty` and `ty + 1` of
  column `tx`, so a CTA computes 32 x 16 outputs.
- K step of 64. `A` is staged into `smem_A[32][65]` with coalesced loads.
- `B` is unpacked into `smem_B[64][17]` inside its own scope. For PTQ1_0 the
  64 weights of one step sit in a single 128-weight block, so each thread
  reads the block scale once.
- Inner loop over the K tile, `#pragma unroll 4`:
  ```
  float b_val = smem_B[k][tx];
  accum_fragment[0] += smem_A[ty][k] * b_val;
  accum_fragment[1] += smem_A[ty + 1][k] * b_val;
  ```
- Epilogue: the output block scale (`[ceil(M/br)][ceil(N/bc)]`, loaded to
  shared memory once) multiplies both accumulators before the store.
- Requirements: `K % 64 == 0`, plus `K % 128 == 0` for PTQ1_0.
