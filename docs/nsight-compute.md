# Nsight Compute checks

These have not been run yet. There was no GPU on the build machine. Run them
on the target GPU with `gpu_bench` from the build directory.

## Commands

```sh
# Block-scale stage (float / half / bf16) and the fused / unfused GEMM
ncu --kernel-name regex:apply_block_scale_stage \
    --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section Occupancy --section LaunchStats ./gpu_bench

ncu --kernel-name regex:gemm_f32_block_scale \
    --section SpeedOfLight --section Occupancy --section LaunchStats ./gpu_bench

# Packed ternary / INT4 GEMM
ncu --kernel-name custom_int4_tiled_gemm_kernel \
    --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section Occupancy --section LaunchStats ./gpu_bench

# Just the numbers below, for every kernel
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct,\
launch__registers_per_thread,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum ./gpu_bench
```

## What to look for

| check | metric | target |
|---|---|---|
| memory throughput | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | block-scale stage near peak (it is a streaming kernel) |
| L1/TEX hit rate | `l1tex__t_sector_hit_rate.pct` | scales come from shared memory, so global loads should be one-touch |
| registers | `launch__registers_per_thread` | at most 64, the limit set by `__launch_bounds__(256, 4)` |
| occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` | at least 50% |
| spills / local memory | `l1tex__t_bytes_pipe_lsu_mem_local_op_{ld,st}.sum` | 0 |

Theoretical occupancy at 4 CTAs of 256 threads per SM is 1024 threads. That
is 50% on sm_80 and sm_90 (2048 threads per SM) and 67% on sm_86 and sm_89
(1536 threads per SM).

## Results

| kernel | GPU | DRAM % peak | L1/TEX hit % | regs | achieved occupancy | local bytes |
|---|---|---|---|---|---|---|
| apply_block_scale_stage<float> | | | | | | |
| apply_block_scale_stage<half> | | | | | | |
| apply_block_scale_stage<bf16> | | | | | | |
| gemm_f32_block_scale (fused) | | | | | | |
| custom_int4_tiled_gemm_kernel PTQ1_0 | | | | | | |
| custom_int4_tiled_gemm_kernel Q4_0 | | | | | | |

The static register and spill counts from ptxas are in
[ptxas-report.md](ptxas-report.md).
