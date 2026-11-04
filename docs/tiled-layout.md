# Tiled device layout

## Search for an existing layout

The existing code in /workspace was searched read-only for a tiled weight
layout to match. None was found. The closest thing is WMMA padding of
matrix sizes up to multiples of 16 in
`sovereign-mimo-4b/k3_nano/k3_nano_harness.cu` (lines 692-730, also in
`sovereign-kimi/kernels/k3_nano_harness.cu`). That is padding, not a tile
order, so this loader uses its own documented layout. It is configurable
with `--tile R C`.

## Layout

A tensor with `ne = [ne0, ne1, ...]` is treated as a matrix:

- `rows = ne1 * ne2 * ...`
- `units_per_row = ne0 / block_elems`

A unit is one element for scalar types (F32, F16, BF16) and one whole quant
block for block types (PTQ1_0: 128 weights, 28 bytes; Q4_0: 32 weights,
18 bytes). Blocks are never split.

The matrix is cut into tiles of `R x C` units (default 16 x 16). Tiles are
stored in row-major tile order, and units inside a tile are row-major. The
matrix is zero-padded to whole tiles in both directions. Byte `b` of row `r`
goes to:

```
u  = b / unit_bytes,  ub = b % unit_bytes
tr = r / R,  ir = r % R
tc = u / C,  ic = u % C
dst = (tr * tiles_c + tc) * (R * C * unit_bytes) + (ir * C + ic) * unit_bytes + ub
```

`tiled_offset()` in `include/bonsai/tile_layout.hpp` is that formula, shared
by the CPU reference and the device kernel.

1-D tensors and tensors of unknown type are copied linearly.

Each tensor starts on a 256-byte boundary in a single device arena.

`--tile 1 C` with `C` dividing `units_per_row` gives the plain GGUF row
layout.

## Padding cost on the 27B checkpoint

Arena sizes from `bonsai_index --header-only --tile R C`. Tensor bytes in the
file: 5,935,527,936.

| tile | arena bytes |
|---|---|
| 1 x 1 | 5,935,534,080 |
| 16 x 8 | 5,943,398,400 |
| 16 x 16 (default) | 6,870,437,888 |
| 32 x 16 | 6,886,166,528 |

Most PTQ1_0 rows are 40 blocks (K = 5120), so a tile width that divides 40,
such as 8, avoids nearly all padding.
