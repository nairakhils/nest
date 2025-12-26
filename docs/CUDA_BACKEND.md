# CUDA Backend for Nest

The CUDA backend provides GPU acceleration for the 1D Euler solver using NVIDIA GPUs.

## Architecture

The CUDA implementation maintains compatibility with the CPU numerics by using equivalent device functions:

```
include/nest/core.hpp          # Shared types (index_space, field, etc.)
examples/sod1d/euler1d.hpp     # CPU numerics (reference implementation)
src/backend_cuda/euler1d_cuda.cuh    # CUDA device functions (same algorithm)
src/backend_cuda/euler1d_solver.cu   # GPU solver with CUDA kernels
src/backend_cuda/euler1d_solver.hpp  # C++ interface (no CUDA needed)
```

### Data Layout

Both CPU and GPU use Structure-of-Arrays (SoA) layout for optimal memory access:

```
// Conserved variables [NCONS * n_total] where NCONS=3
// Layout: [rho_0, rho_1, ..., rho_n-1, 
//          mom_0, mom_1, ..., mom_n-1,
//          E_0, E_1, ..., E_n-1]
```

This gives coalesced memory access on GPU and vectorizable access on CPU.

### Kernel Design

The main kernel (`euler1d_rhs_kernel`) computes one timestep update:

1. **One thread per interior cell** - Simple parallel mapping
2. **Stencil fetch** - Each thread loads 5 cells (i-2 to i+2) for PLM reconstruction
3. **Local computation** - Convert to primitive, reconstruct, compute HLLE flux
4. **Direct update** - Write updated conserved variables

```cuda
__global__ void euler1d_rhs_kernel(
    const double* __restrict__ U_in,
    double* __restrict__ U_out,
    int n_interior,
    int n_halo,
    double dx, double dt, double gamma
)
```

### RK2 Time Integration

The solver uses Heun's method (RK2):
1. Stage 1: `U_tmp = U + dt * L(U)`
2. Stage 2: `U_star = U_tmp + dt * L(U_tmp)`
3. Average: `U = 0.5 * (U + U_star)`

Each stage requires halo exchange and kernel launch.

## Building

### Prerequisites

- CUDA Toolkit 11.0+ (for C++17/20 support)
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)

### Build Commands

```bash
# Configure with CUDA enabled
cmake --preset cuda-release

# Build
cmake --build build/cuda-release

# Run tests (includes CPU vs GPU comparison)
ctest --test-dir build/cuda-release
```

### Supported Architectures

The build targets multiple GPU architectures:
- sm_70: Volta (V100)
- sm_75: Turing (RTX 20xx)
- sm_80: Ampere (A100, RTX 30xx)
- sm_86: Ampere (RTX 30xx laptop)
- sm_89: Ada Lovelace (RTX 40xx)
- sm_90: Hopper (H100)

## Usage

### C++ Interface

```cpp
#include "euler1d_solver.hpp"

// Create solver
euler1d_gpu_t* solver = euler1d_gpu_create(n_zones, n_halo, dx, gamma);

// Upload initial conditions (SoA layout)
euler1d_gpu_upload(solver, h_data);

// Run simulation
int steps = euler1d_gpu_run(solver, t_final, cfl);

// Download results
euler1d_gpu_download(solver, h_data);

// Cleanup
euler1d_gpu_destroy(solver);
```

### Fine-grained Control

```cpp
// Compute stable timestep
double dt = euler1d_gpu_compute_dt(solver, cfl);

// Single RK2 step
euler1d_gpu_rk2_step(solver, dt);
```

## Performance Notes

This is a **baseline implementation** optimized for correctness and readability, not peak performance.

### Current Implementation

- Simple kernel launch (one thread per cell)
- Global memory for all data
- Block reduction for CFL computation

### Future Optimizations

1. **Shared memory** for stencil data
2. **Warp-level primitives** for reduction
3. **Persistent kernels** to reduce launch overhead
4. **Multi-GPU** with domain decomposition
5. **Mixed precision** for memory-bound problems

## Validation

The `test_euler_cuda` test runs the Sod shock tube on both CPU and GPU, comparing results within tolerance (should be identical within floating-point precision).

```bash
# Run GPU test
./build/cuda-release/tests/test_euler_cuda
```

Expected output:
```
=== GPU vs CPU Euler Solver Comparison ===
Configuration:
  Zones: 100
  t_final: 0.1
  CFL: 0.3

Running CPU solver...
CPU done.

Running GPU solver...
GPU: 39 steps
GPU done.

Max difference (rho): 1.11022e-16
Max difference (mom): 2.22045e-16
Max difference (E):   4.44089e-16
PASS: GPU and CPU results match within tolerance 1e-10
```

