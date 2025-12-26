# nest agent instructions

## Goal
Build a fast, readable C++20 hydro library with CPU (OpenMP) + CUDA backends.
Design uses proven patterns (index spaces, SoA layout, pipeline execution) with a clean independent implementation.

## Definition of Done for every phase
- Compiles on CPU (macOS) with CMake+Ninja
- Tests pass via ctest
- Formatting is consistent (clang-format)
- Clear public API docs updated (README + docs/ARCHITECTURE.md)

## Commands
CPU debug:
  cmake --preset cpu-debug
  ctest --test-dir build/cpu-debug

CPU release:
  cmake --preset cpu-release
  ctest --test-dir build/cpu-release

CUDA release (on NVIDIA machine):
  cmake --preset cuda-release
  ctest --test-dir build/cuda-release


  