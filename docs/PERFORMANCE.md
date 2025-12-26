# Performance Guide

## Memory Traffic Analysis

### 1D Euler Kernel

Each RK stage requires:
- **Reads**: 5 cells × 3 vars = 15 doubles = 120 bytes per cell
- **Writes**: 1 cell × 3 vars = 3 doubles = 24 bytes per cell
- **Total**: 144 bytes/cell/stage

Per RK2 step (2 stages + halos + averaging):
- Fused stages: 2 × 144 = 288 bytes/cell
- Halo copy: ~10 bytes/cell (amortized for large domains)
- RK averaging: ~50 bytes/cell
- **Total**: ~340 bytes/cell/step

### 2D Euler Kernel (Directional Splitting)

Per sweep (x or y):
- **Reads**: 5 cells × 4 vars = 20 doubles = 160 bytes/cell
- **Writes**: 1 cell × 4 vars = 4 doubles = 32 bytes/cell
- **Total**: 192 bytes/cell/sweep

Per RK2 step:
- 2 stages × 2 sweeps × 192 = 768 bytes/cell
- Plus halos and averaging: ~100 bytes/cell
- **Total**: ~870 bytes/cell/step

### Shared Memory Benefits (2D)

Without shared memory, neighboring cells reload overlapping stencil data.
With shared memory tiling:
- Block loads tile + halos cooperatively
- Each cell accesses shared memory instead of global memory
- Reduces redundant global loads by ~2.5× for 5-point stencil

Enable with: `cmake -DNEST_CUDA_SHARED_MEM=ON`

## Optimization Strategies

### 1. Kernel Fusion

**Before** (separate kernels):
```
recon_kernel()         // Read: 5 cells, Write: 1 face states
riemann_kernel()       // Read: face states, Write: fluxes  
update_kernel()        // Read: fluxes + U, Write: U_new
```
Total traffic: ~300 bytes/cell/stage

**After** (fused kernel):
```
euler_fused_kernel()   // Read: 5 cells, Write: 1 cell
```
Total traffic: 144 bytes/cell/stage → **2× reduction**

### 2. Register Blocking

Load stencil into registers, compute locally, write once:
```cuda
double rho[5], mom[5], erg[5];  // 15 registers
// Load from global memory (coalesced)
for (int s = 0; s < 5; ++s) {
    rho[s] = U_in[...];
    mom[s] = U_in[...];
    erg[s] = U_in[...];
}
// All computation in registers
// Single coalesced write
```

### 3. Memory Access Patterns

SoA (Struct of Arrays) layout for coalescing:
```
Layout: [rho_0, rho_1, ..., rho_n-1, mom_0, ..., mom_n-1, ...]
Index:  var * n_total + i

Benefit: Adjacent threads access adjacent memory
```

### 4. Occupancy Tuning

Block size trade-offs:
- **Small blocks** (64): More blocks, hide latency
- **Large blocks** (256): Better shared memory efficiency
- **Sweet spot**: 256 for 1D, 32×8 for 2D

## Benchmark Results

Run with: `./benchmark_euler1d [n_zones] [t_final]`

Example output (release build):
```
Configuration:
  Zones:    100000
  t_final:  0.2

CPU Benchmark (OpenMP enabled)...
  Cells/sec:    1.23e+08
  Updates/sec:  2.46e+08 (x2 for RK2)

GPU Benchmark (CUDA)...
  Cells/sec:    2.5e+09
  Updates/sec:  5.0e+09

Speedup: 20x
```

## Profiling Tips

### NVIDIA Nsight Compute

```bash
ncu --set full ./benchmark_euler1d 100000 0.2
```

Key metrics to watch:
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` (compute utilization)
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` (memory bandwidth)
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` (shared mem conflicts)

### Apple Instruments (for M-series)

```bash
xcrun xctrace record --template "CPU Profiler" --launch -- ./benchmark_euler1d 100000 0.2
```

## Expected Performance

| Platform | Cells/sec (1D) | Cells/sec (2D 256×256) |
|----------|---------------|------------------------|
| M1 Pro (CPU, 8 cores) | ~1.2e8 | ~4e7 |
| RTX 3080 | ~2.5e9 | ~8e8 |
| A100 | ~5e9 | ~2e9 |

Memory bandwidth is typically the bottleneck:
- M1 Pro: ~200 GB/s
- RTX 3080: ~760 GB/s
- A100: ~2 TB/s

Achieved bandwidth fraction (target: 60-80%):
```
Achieved = (bytes/cell/step * cells/sec) / peak_bandwidth
```

