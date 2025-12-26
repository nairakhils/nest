# Sod Shock Tube Test

Classic 1D Riemann problem for validating compressible Euler solvers.

## Problem Setup

Initial discontinuity at x = 0.5:

| Region | Density (ρ) | Velocity (u) | Pressure (p) |
|--------|-------------|--------------|--------------|
| Left   | 1.0         | 0.0          | 1.0          |
| Right  | 0.125       | 0.0          | 0.1          |

- Domain: [0, 1]
- Final time: t = 0.2
- Gamma-law EOS: γ = 1.4

## Features Implemented

### Gamma-Law Equation of State
```cpp
p = (γ - 1) * (E - 0.5 * ρ * u²)
E = p / (γ - 1) + 0.5 * ρ * u²
c = sqrt(γ * p / ρ)
```

### Primitive ↔ Conserved Conversion
- Conservative variables: U = [ρ, ρu, E]
- Primitive variables: W = [ρ, u, p]
- Bidirectional conversion with positivity checks

### PLM Reconstruction
Piecewise Linear Method with slope limiters:
- **minmod**: Most dissipative, TVD
- **MC** (Monotonized Central): Less dissipative, still TVD
- **none**: No limiting (for testing)

Reconstructs left/right states at cell interfaces using limited slopes.

### HLLE Riemann Solver
Harten-Lax-van Leer-Einfeldt approximate Riemann solver:
- Wave speed estimates: S_L = min(u_L - c_L, u_R - c_R), S_R = max(u_L + c_L, u_R + c_R)
- Robust for strong shocks
- Handles supersonic and subsonic cases

### RK2 Time Integrator
Heun's method (2nd-order Runge-Kutta):
```
U₁ = U₀ + Δt * RHS(U₀)
U_{n+1} = 0.5 * (U₀ + U₁ + Δt * RHS(U₁))
```

### Positivity Checks
Debug assertions ensure ρ > 0 and p > 0 at every step:
- In `cons_to_prim()` and `prim_to_cons()`
- In main loop every 10 steps
- Enabled with `-g` (debug builds)

## Building

```bash
# From nest root
clang++ -std=c++20 -O2 -I include -I examples/sod1d examples/sod1d/sod1d.cpp -o sod1d

# Or with CMake
cmake --preset cpu-release
cmake --build build/cpu-release
./build/cpu-release/examples/sod1d/sod1d
```

## Running

```bash
./sod1d
# Produces: sod_0000.dat, sod_0001.dat, ...
```

## Output Format

ASCII text files with columns:
```
# x rho u p e
0.00125  0.426033  -0.929784  0.302736  1.776478
...
```

- `x`: Cell center position
- `rho`: Density (ρ)
- `u`: Velocity
- `p`: Pressure
- `e`: Specific internal energy

## Plotting

```bash
python scripts/plot_sod.py sod_*.dat
```

Produces `sod_plot.png` with density, velocity, pressure, and internal energy profiles.

## Expected Solution

The Sod shock tube produces five regions:
1. **Expansion fan** (rarefaction wave)
2. **Contact discontinuity** (density jump, no pressure jump)
3. **Shock wave** (all variables jump)

At t = 0.2:
- Shock position: x ≈ 0.85
- Contact position: x ≈ 0.68
- Rarefaction tail: x ≈ 0.26

## Validation

### Positivity
All tests should pass with:
```
Positivity check: PASSED
```

### Convergence
Expected 1st-order convergence (upwind + PLM):
- L1 error ∝ Δx

### Structure
Should resolve:
- Sharp shock (few cells)
- Smooth rarefaction fan
- Contact discontinuity (1-2 cells)

## References

- Sod, G. A. (1978). "A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws"
- Toro, E. F. (2009). "Riemann Solvers and Numerical Methods for Fluid Dynamics"

