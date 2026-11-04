# Bonsai checkpoint format

The loader reads the GGUF header and takes every offset, type and shape from
it. Nothing about a particular checkpoint is hardcoded.

## Sources

- Checkpoint: https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf
  (`Ternary-Bonsai-2-27B-PTQ1_0.gguf`). The model card describes 1.75 bits per
  weight, ternary weights in groups of 128 with an FP16 scale, and a Hadamard
  rotation (block 1024) folded into the weights.
- Smaller sibling: https://huggingface.co/prism-ml/Bonsai-8B-gguf (Q1_0).
- Quant types: PrismML fork of llama.cpp, branch `prism`,
  https://github.com/PrismML-Eng/llama.cpp/tree/prism
  - `ggml/include/ggml.h`: `GGML_TYPE_PQ2_0 = 142`, `GGML_TYPE_PTQ1_0 = 143`
  - `ggml/src/ggml-common.h`: `block_ptq1_0`, `QK_PTQ1_0 = 128`, `block_q4_0`
  - `ggml/src/ggml-quants.c`: `quantize_row_ptq1_0_ref`, `dequantize_row_ptq1_0`
- GGUF container: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

## GGUF v3 container

```
"GGUF"  u32 version (3)  u64 n_tensors  u64 n_kv
n_kv    x { string key, u32 type, value }
n_tensors x { string name, u32 n_dims, u64 ne[n_dims], u32 ggml_type, u64 offset }
padding to general.alignment (default 32)
tensor data; each tensor at data_start + offset
```

Strings are `u64 length` + bytes. KV value types: 0 u8, 1 i8, 2 u16, 3 i16,
4 u32, 5 i32, 6 f32, 7 bool, 8 string, 9 array (u32 elem type, u64 count),
10 u64, 11 i64, 12 f64.

The parser (`src/gguf.cpp`) checks the magic, version, every length against
the file size, offset alignment, overlap between tensors, and that each
tensor fits in the file. Tensor sizes come from the type's block size; a type
the loader does not know is sized by the gap to the next tensor and copied
linearly.

## What the real header says

Parsed from the first 11,120,992 bytes of `Ternary-Bonsai-2-27B-PTQ1_0.gguf`
with `bonsai_index --header-only`:

| field | value |
|---|---|
| file size | 5,946,648,928 bytes |
| version | 3 |
| alignment | 32 |
| kv pairs | 49 |
| general.architecture | qwen35 |
| tensors | 851 |
| data offset | 11,120,992 |
| tensor bytes | 5,935,527,936 (data offset + tensor bytes = file size) |
| PTQ1_0 (143) | 402 tensors, 5,877,760,000 bytes |
| F32 (0) | 353 tensors, 10,582,016 bytes |
| BF16 (30) | 96 tensors, 47,185,920 bytes |

The largest tensor is `output.weight`, PTQ1_0 `[5120, 248320]`: 40 blocks per
row, 1120 bytes per row, 278.1 MB.

## PTQ1_0 block (type 143)

128 weights in 28 bytes:

```c
typedef struct {
    uint8_t   qs[24];  // 120 trits, 5 per byte
    uint8_t   qh[2];   //   8 trits, 4 per byte
    ggml_half d;       // scale, byte offset 26
} block_ptq1_0;
```

The trit layout is in [packing.md](packing.md).

## Raw mode

`bonsai_load --raw <manifest> <blob>` loads a flat blob described by a text
manifest, one tensor per line:

```
# name offset bytes dtype shape
token_embd.weight 0 2240 PTQ1_0 256,40
```

`bonsai_index <file.gguf> --manifest` writes such a manifest with absolute
offsets into the GGUF file, so the GGUF itself can be used as the blob.
