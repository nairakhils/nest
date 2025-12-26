# 2D Sod Shock Tube

Extension of the 1D Sod test to 2D with directional splitting.

## Features

- 2D mesh builder with periodic boundaries in both dimensions
- 2D patch structure with halo exchange in x and y directions
- Directional splitting with proper passive scalar advection
- 2D CFL condition
- Validation to verify solution uniformity in y

## Usage

```bash
cd build/cpu-release
./examples/sod2d/sod2d
```

Output files:
- `sod2d_NNNN.dat` - Full 2D solution at various times
- `sod2d_xslice_NNNN.dat` - 1D slice along x at middle y (for comparison with 1D solution)

## Architecture

### 2D Mesh Builder
```cpp
auto patches = build_periodic_mesh_2d<double>(
    domain,          // index_space_t<2>
    nx_patches,      // Number of patches in x
    ny_patches,      // Number of patches in y
    nvars,           // 4 for 2D Euler (rho, rho*u, rho*v, E)
    halo             // Halo width
);
```

Creates patches with:
- 4 neighbors per patch (left, right, bottom, top)
- Periodic boundaries in both directions
- Precomputed exchange plans for efficient halo updates

### Directional Splitting

```cpp
// X-sweep: update using fluxes in x-direction
x_sweep(U_in, U_out, interior, dt);

// Y-sweep: update using fluxes in y-direction  
y_sweep(U_in, U_out, interior, dt);
```

The solver uses Strang-style dimensional splitting:

1. Fill halos via `execute_all_exchanges()`
2. **X-sweep**: Update [rho, rho*u, E] using HLLE fluxes in x-direction; advect rho*v as passive scalar
3. Fill halos again
4. **Y-sweep**: Update [rho, rho*v, E] using HLLE fluxes in y-direction; advect rho*u as passive scalar

### Passive Scalar Advection

For the transverse momentum components, we use upwind advection:
- X-sweep: `(rho*v)_new = (rho*v)_old - dt/dx * (flux_v_R - flux_v_L)` where `flux_v = u_interface * rho*v`
- Y-sweep: `(rho*u)_new = (rho*u)_old - dt/dy * (flux_u_R - flux_u_L)` where `flux_u = v_interface * rho*u`

The interface velocity is computed from the HLLE wave speed estimates.

## Validation

The 2D Sod test uses initial conditions with a discontinuity along x and uniform in y:
- Left (x < 0.5): rho=1, u=0, v=0, p=1
- Right (x >= 0.5): rho=0.125, u=0, v=0, p=0.1

The solution should remain uniform in y throughout the simulation. This is verified by computing the standard deviation of each field along y at several x locations, which should be at machine precision (< 1e-10).

## Files

```
sod2d/
├── euler2d.hpp     # 2D Euler solver with directional splitting
├── sod2d.cpp       # 2D Sod test driver
├── CMakeLists.txt  # Build configuration
└── README.md       # This file
```

## References

- **Strang, G. (1968).** "On the construction and comparison of difference schemes." *SIAM Journal on Numerical Analysis*, 5(3), 506-517.
- **LeVeque, R. J. (2002).** *Finite Volume Methods for Hyperbolic Problems.* Cambridge University Press. (Chapter 19: Multidimensional Problems)

## Future Improvements

1. Strang splitting (alternating X-Y and Y-X sweeps each timestep)
2. Corner transport upwind (CTU) for improved accuracy
3. 2D-specific test cases (oblique shocks, vortex problems)
4. OpenMP parallelization of sweeps
