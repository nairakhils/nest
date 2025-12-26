#pragma once

/**
 * C API for CUDA 1D Euler Solver
 * 
 * Provides interface to the optimized GPU solver:
 * - Fused recon+riemann+divergence+update kernels
 * - Optional shared-memory tiling (compile with -DNEST_CUDA_SHARED_MEM)
 * - RK2 time integration
 */

#ifdef __cplusplus
extern "C" {
#endif

struct euler1d_gpu_t;

/**
 * Create a CUDA 1D Euler solver.
 * 
 * @param n_interior  Number of interior (non-halo) cells
 * @param n_halo      Number of halo cells on each side (2 for PLM)
 * @param dx          Cell width
 * @param gamma       Adiabatic index
 * @return            Opaque pointer to the solver
 */
euler1d_gpu_t* euler1d_gpu_create(int n_interior, int n_halo, double dx, double gamma);

/**
 * Destroy the solver and free GPU memory.
 */
void euler1d_gpu_destroy(euler1d_gpu_t* solver);

/**
 * Upload initial data from host to device.
 * 
 * @param solver   The solver instance
 * @param h_data   Host data in SoA layout: [var * n_total + i]
 *                 where n_total = n_interior + 2*n_halo
 */
void euler1d_gpu_upload(euler1d_gpu_t* solver, const double* h_data);

/**
 * Download solution data from device to host.
 * 
 * @param solver   The solver instance
 * @param h_data   Host buffer to receive data (SoA layout)
 */
void euler1d_gpu_download(euler1d_gpu_t* solver, double* h_data);

/**
 * Compute timestep based on CFL condition.
 * 
 * @param solver   The solver instance
 * @param cfl      CFL number (typically 0.3-0.8)
 * @return         Stable timestep
 */
double euler1d_gpu_compute_dt(euler1d_gpu_t* solver, double cfl);

/**
 * Perform a single RK2 timestep.
 * 
 * @param solver   The solver instance
 * @param dt       Timestep
 */
void euler1d_gpu_rk2_step(euler1d_gpu_t* solver, double dt);

/**
 * Run until t_final with automatic timestep selection.
 * 
 * @param solver   The solver instance
 * @param t_final  Final simulation time
 * @param cfl      CFL number
 * @return         Number of steps taken
 */
int euler1d_gpu_run(euler1d_gpu_t* solver, double t_final, double cfl);

/**
 * Run until t_final and return wall-clock time (for benchmarks).
 * 
 * @param solver   The solver instance
 * @param t_final  Final simulation time
 * @param cfl      CFL number
 * @param steps    Output: number of steps taken
 * @return         Wall-clock time in milliseconds
 */
double euler1d_gpu_run_timed(euler1d_gpu_t* solver, double t_final, double cfl, int* steps);

#ifdef __cplusplus
}
#endif
