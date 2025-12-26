#pragma once
#include <cuda_runtime.h>
#include <cmath>

/**
 * CUDA device functions and kernels for 1D Euler solver.
 * 
 * Optimized version with:
 * - Fused recon+riemann+divergence+update kernel (single global read/write per cell)
 * - Optional shared memory tiling (enable with NEST_CUDA_SHARED_MEM)
 * - Reduced register pressure via scalar expansion
 * 
 * Memory traffic analysis (baseline):
 *   Read:  5 cells * 3 vars = 15 doubles per thread = 120 bytes
 *   Write: 1 cell * 3 vars = 3 doubles per thread = 24 bytes
 *   Total: 144 bytes per cell per RK stage
 */

namespace nest::cuda::euler1d {

// =============================================================================
// Constants
// =============================================================================

constexpr int NCONS = 3;  // Conserved: rho, rho*u, E
constexpr int NPRIM = 3;  // Primitive: rho, u, p

constexpr int I_RHO = 0;
constexpr int I_MOM = 1;
constexpr int I_ERG = 2;

constexpr int I_VEL = 1;
constexpr int I_PRE = 2;

// Kernel configuration
constexpr int BLOCK_SIZE = 256;
constexpr int STENCIL_RADIUS = 2;  // For PLM: need cells i-2 to i+2

// =============================================================================
// Device functions for Euler numerics (inlined for performance)
// =============================================================================

__device__ __forceinline__ double pressure(double gamma, double rho, double mom, double erg) {
    double u = mom / rho;
    double ke = 0.5 * rho * u * u;
    double p = (gamma - 1.0) * (erg - ke);
    return fmax(p, 1e-10);
}

__device__ __forceinline__ double total_energy(double gamma, double rho, double u, double p) {
    return p / (gamma - 1.0) + 0.5 * rho * u * u;
}

__device__ __forceinline__ double sound_speed(double gamma, double rho, double p) {
    return sqrt(gamma * p / rho);
}

__device__ __forceinline__ double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (fabs(a) < fabs(b)) ? a : b;
}

__device__ __forceinline__ double plm_slope(double W_m, double W_0, double W_p) {
    return minmod(W_0 - W_m, W_p - W_0);
}

// Inline HLLE flux computation (returns flux components directly, no array overhead)
__device__ __forceinline__ void compute_hlle_flux(
    double rho_L, double u_L, double p_L,
    double rho_R, double u_R, double p_R,
    double gamma,
    double& F_rho, double& F_mom, double& F_erg
) {
    double c_L = sound_speed(gamma, rho_L, p_L);
    double c_R = sound_speed(gamma, rho_R, p_R);
    double E_L = total_energy(gamma, rho_L, u_L, p_L);
    double E_R = total_energy(gamma, rho_R, u_R, p_R);
    
    // Wave speeds (Davis estimate)
    double S_L = fmin(u_L - c_L, u_R - c_R);
    double S_R = fmax(u_L + c_L, u_R + c_R);
    
    // Fluxes
    double FL_rho = rho_L * u_L;
    double FL_mom = rho_L * u_L * u_L + p_L;
    double FL_erg = (E_L + p_L) * u_L;
    
    double FR_rho = rho_R * u_R;
    double FR_mom = rho_R * u_R * u_R + p_R;
    double FR_erg = (E_R + p_R) * u_R;
    
    // Conserved states
    double UL_mom = rho_L * u_L, UR_mom = rho_R * u_R;
    
    if (S_L >= 0.0) {
        F_rho = FL_rho; F_mom = FL_mom; F_erg = FL_erg;
    } else if (S_R <= 0.0) {
        F_rho = FR_rho; F_mom = FR_mom; F_erg = FR_erg;
    } else {
        double dS = S_R - S_L;
        double SL_SR = S_L * S_R;
        F_rho = (S_R * FL_rho - S_L * FR_rho + SL_SR * (rho_R - rho_L)) / dS;
        F_mom = (S_R * FL_mom - S_L * FR_mom + SL_SR * (UR_mom - UL_mom)) / dS;
        F_erg = (S_R * FL_erg - S_L * FR_erg + SL_SR * (E_R - E_L)) / dS;
    }
}

// =============================================================================
// Fused RK Stage Kernel (recon + riemann + divergence + update)
// 
// Memory traffic: 
//   Read:  5 cells * 3 vars = 15 doubles per thread (120 bytes)
//   Write: 1 cell * 3 vars = 3 doubles per thread (24 bytes)
// Total: 144 bytes per cell per stage
// =============================================================================

__global__ void euler1d_fused_stage_kernel(
    const double* __restrict__ U_in,
    double* __restrict__ U_out,
    int n_interior,
    int n_halo,
    double dx,
    double dt,
    double gamma
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_interior) return;
    
    int i = tid + n_halo;
    int n_total = n_interior + 2 * n_halo;
    
    // Load stencil into registers (15 global reads, coalesced)
    double rho[5], mom[5], erg[5];
    
    #pragma unroll
    for (int s = 0; s < 5; ++s) {
        int idx = i - 2 + s;
        rho[s] = U_in[I_RHO * n_total + idx];
        mom[s] = U_in[I_MOM * n_total + idx];
        erg[s] = U_in[I_ERG * n_total + idx];
    }
    
    // Convert to primitives in registers
    double u[5], p[5];
    
    #pragma unroll
    for (int s = 0; s < 5; ++s) {
        rho[s] = fmax(rho[s], 1e-10);
        u[s] = mom[s] / rho[s];
        p[s] = pressure(gamma, rho[s], mom[s], erg[s]);
    }
    
    // PLM slopes for cells i-1, i, i+1
    // slope[0] = for cell i-1, slope[1] = for cell i, slope[2] = for cell i+1
    double slope_rho[3], slope_u[3], slope_p[3];
    
    #pragma unroll
    for (int s = 0; s < 3; ++s) {
        slope_rho[s] = plm_slope(rho[s], rho[s+1], rho[s+2]);
        slope_u[s]   = plm_slope(u[s], u[s+1], u[s+2]);
        slope_p[s]   = plm_slope(p[s], p[s+1], p[s+2]);
    }
    
    // Reconstruct at i-1/2 interface
    // Left state: right face of cell i-1 (index 1 in local stencil)
    // Right state: left face of cell i (index 2 in local stencil)
    double rho_L_imh = fmax(rho[1] + 0.5 * slope_rho[0], 1e-10);
    double rho_R_imh = fmax(rho[2] - 0.5 * slope_rho[1], 1e-10);
    double u_L_imh   = u[1] + 0.5 * slope_u[0];
    double u_R_imh   = u[2] - 0.5 * slope_u[1];
    double p_L_imh   = fmax(p[1] + 0.5 * slope_p[0], 1e-10);
    double p_R_imh   = fmax(p[2] - 0.5 * slope_p[1], 1e-10);
    
    // Reconstruct at i+1/2 interface
    double rho_L_iph = fmax(rho[2] + 0.5 * slope_rho[1], 1e-10);
    double rho_R_iph = fmax(rho[3] - 0.5 * slope_rho[2], 1e-10);
    double u_L_iph   = u[2] + 0.5 * slope_u[1];
    double u_R_iph   = u[3] - 0.5 * slope_u[2];
    double p_L_iph   = fmax(p[2] + 0.5 * slope_p[1], 1e-10);
    double p_R_iph   = fmax(p[3] - 0.5 * slope_p[2], 1e-10);
    
    // HLLE fluxes at both interfaces
    double F_rho_L, F_mom_L, F_erg_L;
    double F_rho_R, F_mom_R, F_erg_R;
    
    compute_hlle_flux(rho_L_imh, u_L_imh, p_L_imh, 
                      rho_R_imh, u_R_imh, p_R_imh, 
                      gamma, F_rho_L, F_mom_L, F_erg_L);
    
    compute_hlle_flux(rho_L_iph, u_L_iph, p_L_iph,
                      rho_R_iph, u_R_iph, p_R_iph,
                      gamma, F_rho_R, F_mom_R, F_erg_R);
    
    // Update (3 global writes, coalesced)
    double dtdx = dt / dx;
    
    // Use original conserved values from rho[2], mom[2], erg[2] (cell i)
    U_out[I_RHO * n_total + i] = rho[2] - dtdx * (F_rho_R - F_rho_L);
    U_out[I_MOM * n_total + i] = mom[2] - dtdx * (F_mom_R - F_mom_L);
    U_out[I_ERG * n_total + i] = erg[2] - dtdx * (F_erg_R - F_erg_L);
}

// =============================================================================
// Shared Memory Version (for better cache utilization)
// Enable with -DNEST_CUDA_SHARED_MEM
// 
// Benefits for 2D: reduces redundant global loads when cells share neighbors
// =============================================================================

#ifdef NEST_CUDA_SHARED_MEM

__global__ void euler1d_fused_stage_smem_kernel(
    const double* __restrict__ U_in,
    double* __restrict__ U_out,
    int n_interior,
    int n_halo,
    double dx,
    double dt,
    double gamma
) {
    // Shared memory for tile + halos
    constexpr int TILE_SIZE = BLOCK_SIZE + 2 * STENCIL_RADIUS;
    
    __shared__ double s_rho[TILE_SIZE];
    __shared__ double s_mom[TILE_SIZE];
    __shared__ double s_erg[TILE_SIZE];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int n_total = n_interior + 2 * n_halo;
    
    // Global index including domain halos
    int i_global = blockIdx.x * blockDim.x + n_halo + tid;
    
    // Load main tile into shared memory (each thread loads one cell)
    int s_idx = tid + STENCIL_RADIUS;
    
    if (gid < n_interior) {
        s_rho[s_idx] = U_in[I_RHO * n_total + i_global];
        s_mom[s_idx] = U_in[I_MOM * n_total + i_global];
        s_erg[s_idx] = U_in[I_ERG * n_total + i_global];
    }
    
    // Load left halo (first STENCIL_RADIUS threads)
    if (tid < STENCIL_RADIUS) {
        int halo_g_idx = i_global - STENCIL_RADIUS;
        s_rho[tid] = U_in[I_RHO * n_total + halo_g_idx];
        s_mom[tid] = U_in[I_MOM * n_total + halo_g_idx];
        s_erg[tid] = U_in[I_ERG * n_total + halo_g_idx];
    }
    
    // Load right halo (last STENCIL_RADIUS threads)
    if (tid >= blockDim.x - STENCIL_RADIUS && gid < n_interior) {
        int r_off = tid - (blockDim.x - STENCIL_RADIUS);
        int halo_s_idx = STENCIL_RADIUS + blockDim.x + r_off;
        int halo_g_idx = i_global + STENCIL_RADIUS;
        if (halo_g_idx < n_total) {
            s_rho[halo_s_idx] = U_in[I_RHO * n_total + halo_g_idx];
            s_mom[halo_s_idx] = U_in[I_MOM * n_total + halo_g_idx];
            s_erg[halo_s_idx] = U_in[I_ERG * n_total + halo_g_idx];
        }
    }
    
    __syncthreads();
    
    if (gid >= n_interior) return;
    
    // Compute using shared memory
    int s = s_idx;
    
    // Load stencil from shared memory into registers
    double rho[5], mom[5], u[5], p[5];
    
    #pragma unroll
    for (int k = 0; k < 5; ++k) {
        rho[k] = fmax(s_rho[s - 2 + k], 1e-10);
        mom[k] = s_mom[s - 2 + k];
        u[k] = mom[k] / rho[k];
        p[k] = pressure(gamma, rho[k], mom[k], s_erg[s - 2 + k]);
    }
    
    // PLM slopes
    double slope_rho[3], slope_u[3], slope_p[3];
    
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        slope_rho[k] = plm_slope(rho[k], rho[k+1], rho[k+2]);
        slope_u[k]   = plm_slope(u[k], u[k+1], u[k+2]);
        slope_p[k]   = plm_slope(p[k], p[k+1], p[k+2]);
    }
    
    // Reconstruction at i-1/2
    double rho_L_imh = fmax(rho[1] + 0.5 * slope_rho[0], 1e-10);
    double rho_R_imh = fmax(rho[2] - 0.5 * slope_rho[1], 1e-10);
    double u_L_imh   = u[1] + 0.5 * slope_u[0];
    double u_R_imh   = u[2] - 0.5 * slope_u[1];
    double p_L_imh   = fmax(p[1] + 0.5 * slope_p[0], 1e-10);
    double p_R_imh   = fmax(p[2] - 0.5 * slope_p[1], 1e-10);
    
    // Reconstruction at i+1/2
    double rho_L_iph = fmax(rho[2] + 0.5 * slope_rho[1], 1e-10);
    double rho_R_iph = fmax(rho[3] - 0.5 * slope_rho[2], 1e-10);
    double u_L_iph   = u[2] + 0.5 * slope_u[1];
    double u_R_iph   = u[3] - 0.5 * slope_u[2];
    double p_L_iph   = fmax(p[2] + 0.5 * slope_p[1], 1e-10);
    double p_R_iph   = fmax(p[3] - 0.5 * slope_p[2], 1e-10);
    
    // HLLE fluxes
    double F_rho_L, F_mom_L, F_erg_L;
    double F_rho_R, F_mom_R, F_erg_R;
    
    compute_hlle_flux(rho_L_imh, u_L_imh, p_L_imh,
                      rho_R_imh, u_R_imh, p_R_imh,
                      gamma, F_rho_L, F_mom_L, F_erg_L);
    
    compute_hlle_flux(rho_L_iph, u_L_iph, p_L_iph,
                      rho_R_iph, u_R_iph, p_R_iph,
                      gamma, F_rho_R, F_mom_R, F_erg_R);
    
    // Update from original shared memory values
    double dtdx = dt / dx;
    double orig_rho = s_rho[s];
    double orig_mom = s_mom[s];
    double orig_erg = s_erg[s];
    
    U_out[I_RHO * n_total + i_global] = orig_rho - dtdx * (F_rho_R - F_rho_L);
    U_out[I_MOM * n_total + i_global] = orig_mom - dtdx * (F_mom_R - F_mom_L);
    U_out[I_ERG * n_total + i_global] = orig_erg - dtdx * (F_erg_R - F_erg_L);
}

#endif // NEST_CUDA_SHARED_MEM

// =============================================================================
// Periodic Halo Copy Kernel
// =============================================================================

__global__ void copy_halos_periodic_kernel(
    double* U,
    int n_interior,
    int n_halo,
    int nvars
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_total = n_interior + 2 * n_halo;
    
    int var = tid / n_halo;
    int h = tid % n_halo;
    
    if (var >= nvars) return;
    
    // Left halo: copy from right interior
    int src_left = n_halo + n_interior - n_halo + h;
    int dst_left = h;
    U[var * n_total + dst_left] = U[var * n_total + src_left];
    
    // Right halo: copy from left interior
    int src_right = n_halo + h;
    int dst_right = n_halo + n_interior + h;
    U[var * n_total + dst_right] = U[var * n_total + src_right];
}

// =============================================================================
// CFL Reduction Kernel (with warp-level optimization)
// =============================================================================

__global__ void compute_max_wavespeed_kernel(
    const double* __restrict__ U,
    double* __restrict__ block_max,
    int n_interior,
    int n_halo,
    double gamma
) {
    extern __shared__ double shared_max[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    int n_total = n_interior + 2 * n_halo;
    
    double local_max = 0.0;
    
    if (tid < n_interior) {
        int i = tid + n_halo;
        
        double rho = fmax(U[I_RHO * n_total + i], 1e-10);
        double mom = U[I_MOM * n_total + i];
        double erg = U[I_ERG * n_total + i];
        
        double vel = mom / rho;
        double p = pressure(gamma, rho, mom, erg);
        double c = sound_speed(gamma, rho, p);
        
        local_max = fabs(vel) + c;
    }
    
    shared_max[lid] = local_max;
    __syncthreads();
    
    // Warp-optimized reduction
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (lid < s) {
            shared_max[lid] = fmax(shared_max[lid], shared_max[lid + s]);
        }
        __syncthreads();
    }
    
    // Final warp reduction (no sync needed within warp for sm_70+)
    if (lid < 32) {
        volatile double* vs = shared_max;
        if (blockDim.x >= 64) vs[lid] = fmax(vs[lid], vs[lid + 32]);
        vs[lid] = fmax(vs[lid], vs[lid + 16]);
        vs[lid] = fmax(vs[lid], vs[lid + 8]);
        vs[lid] = fmax(vs[lid], vs[lid + 4]);
        vs[lid] = fmax(vs[lid], vs[lid + 2]);
        vs[lid] = fmax(vs[lid], vs[lid + 1]);
    }
    
    if (lid == 0) {
        block_max[blockIdx.x] = shared_max[0];
    }
}

// =============================================================================
// RK2 Averaging Kernel
// =============================================================================

__global__ void rk2_average_kernel(
    double* __restrict__ U,
    const double* __restrict__ U_star,
    int n_interior,
    int n_halo,
    int nvars
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_total = n_interior + 2 * n_halo;
    
    int var = tid / n_interior;
    int i = (tid % n_interior) + n_halo;
    
    if (var >= nvars) return;
    
    int idx = var * n_total + i;
    U[idx] = 0.5 * (U[idx] + U_star[idx]);
}

} // namespace nest::cuda::euler1d
