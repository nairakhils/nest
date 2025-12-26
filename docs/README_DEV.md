# Developer README

This file contains build, test, and usage details for working on the nest
codebase locally.

## Building

### Prerequisites

- CMake ≥ 3.20
- Ninja build system
- C++20 compiler (GCC 11+, Clang 14+, Apple Clang 15+)
- OpenMP (optional, for parallel CPU execution)
- CUDA Toolkit (optional, for GPU execution)

### CPU Build

```bash
# Debug build
cmake --preset cpu-debug
cmake --build build/cpu-debug
ctest --test-dir build/cpu-debug

# Release build (optimized for current machine)
cmake --preset cpu-release
cmake --build build/cpu-release
ctest --test-dir build/cpu-release
```

### CUDA Build (on NVIDIA machines)

Requires CUDA Toolkit 11.0+ and compute capability 7.0+ (Volta or newer).

```bash
cmake --preset cuda-release
cmake --build build/cuda-release
ctest --test-dir build/cuda-release
```

See [docs/CUDA_BACKEND.md](CUDA_BACKEND.md) for GPU implementation details.

## Examples

### 1D Linear Advection

```bash
cd build/cpu-release
./examples/advect1d/advect1d
```

### 1D Sod Shock Tube

```bash
cd build/cpu-release
./examples/sod1d/sod1d
```

### 2D Sod Shock Tube

```bash
cd build/cpu-release
./examples/sod2d/sod2d
```

## Visualization

Python tools for plotting simulation results:

```bash
# Plot 1D Sod results (density, velocity, pressure)
python -m python plot1d sod_0000.dat sod_0008.dat

# Plot 2D Sod x-slices
python -m python plot2d sod2d_xslice_*.dat --mode slice

# Plot 2D Sod as images
python -m python plot2d sod2d_0006.dat --mode image
```

Requirements: `numpy`, `matplotlib`

Figures are saved to `python/figures/`. See [python/README.md](../python/README.md)
for details.

## Project Structure

```
nest/
├── include/nest/      # Header-only, backend-agnostic numerics
│   ├── core.hpp       # vec_t, index_space_t, SoA helpers
│   ├── patch.hpp      # Patch with halos, neighbor info
│   └── pipeline.hpp   # Pipeline stages
├── src/
│   ├── backend_cpu/   # CPU + OpenMP implementation
│   └── backend_cuda/  # CUDA implementation (1D Euler solver)
├── examples/          # Example applications
│   ├── advect1d/      # 1D linear advection
│   ├── sod1d/         # 1D Sod shock tube (gamma-law EOS, HLLE, PLM)
│   └── sod2d/         # 2D Sod shock tube (directional splitting)
├── tests/             # Unit tests
├── python/            # Python visualization tools
│   ├── plot_sod1d.py  # Plot 1D Sod results
│   ├── plot_sod2d.py  # Plot 2D Sod results
│   └── figures/       # Generated plots
└── docs/              # Documentation
```

## Documentation

- [docs/ARCHITECTURE.md](ARCHITECTURE.md) - Overall design and architecture
- [docs/CUDA_BACKEND.md](CUDA_BACKEND.md) - GPU implementation details
- [docs/CORE_TYPES.md](CORE_TYPES.md) - Core data structures
- [docs/EULER_SOLVER.md](EULER_SOLVER.md) - 1D Euler solver details

## License

See [LICENSE](../LICENSE) (placeholder until a license is selected).
