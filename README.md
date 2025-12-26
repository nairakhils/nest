# nest

A fast, readable C++20 hydro library with CPU (OpenMP) and CUDA backends.

## What It Does

- **GPU/CPU hydro kernels** with a backend-agnostic numerics layer
- **SoA memory layout** to drive vectorization and GPU coalescing
- **Patch + halo decomposition** for domain splitting and exchange
- **Pipeline execution** (Exchange → Compute → Reduce stages)
- **Example solvers** for 1D/2D Euler and linear advection
- **Python visualization tools** for plotting result files

Developer docs live in [docs/README_DEV.md](docs/README_DEV.md).

## License

See [LICENSE](LICENSE) (placeholder until a license is selected).

