# ptxas report

Output of `-Xptxas=-v` with nvcc 12.9 (`-O3`). Reproduce it with
`ctest -R kernel_resources -V`. That test also fails the build check if any
kernel's PTX has `ld.local`/`st.local`, or if ptxas reports a non-zero spill
or stack frame. The full log is in `build/ptxas/ptxas_report.txt`.

Every kernel on every architecture has a 0-byte stack frame, 0 spill
stores and 0 spill loads, and no `ld.local`/`st.local` in its PTX.

Registers per thread:

| kernel | sm_80 | sm_86 | sm_89 | sm_90 | smem (static) |
|---|---|---|---|---|---|
| retile_rows_kernel | 38 | 38 | 38 | 40 | 0 |
| apply_block_scale_stage<float> | 32 | 32 | 32 | 32 | dynamic |
| apply_block_scale_stage<half> | 30 | 32 | 32 | 32 | dynamic |
| apply_block_scale_stage<bf16> | 30 | 32 | 32 | 32 | dynamic |
| gemm_f32_block_scale<false> (unfused) | 59 | 60 | 60 | 62 | 8704 B + dynamic |
| gemm_f32_block_scale<true> (fused epilogue) | 64 | 64 | 64 | 64 | 8704 B + dynamic |
| custom_int4_tiled_gemm_kernel | 43 | 40 | 40 | 50 | 14720 B |

All of these are at most 64 registers per thread, so 4 CTAs of 256 threads
fit per SM.
