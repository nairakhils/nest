#pragma once
#include <cuda_runtime.h>
#include <cmath>

/**
 * CUDA device functions and kernels for 2D Euler solver.
 * 
 * Implements directional splitting (X-sweep then Y-sweep).
 * 
 * Memory traffic per RK2 step (per sweep):
 *   Global mem version: ~288 bytes/cell/sweep
 *   Shared mem version: ~180 bytes/cell/sweep (reduced redundant loads)
 */

namespace nest::cuda::euler2d {

// =============================================================================
// Constants
// =============================================================================

constexpr int NCONS_2D = 4;  // Conserved: rho, rho*u, rho*v, E
constexpr int NPRIM_2D = 4;  // Primitive: rho, u, v, p

constexpr int I_RHO = 0;
constexpr int I_MOU = 1;  // rho*u
constexpr int I_MOV = 2;  // rho*v
constexpr int I_ERG = 3;

// Kernel configuration
constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;
constexpr int STENCIL_RADIUS = 2;

// =============================================================================
// Device functions
// =============================================================================

__device__ __forceinline__ double pressure_2d(double gamma, double rho, double mou, double mov, double erg) {
    double u = mou / rho;
    double v = mov / rho;
    double ke = 0.5 * rho * (u * u + v * v);
    double p = (gamma - 1.0) * (erg - ke);
    return fmax(p, 1e-10);
}

__device__ __forceinline__ double total_energy_2d(double gamma, double rho, double u, double v, double p) {
    return p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
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

// 1D HLLE flux for x-direction (rho, rho*u, E) - v is passively advected
__device__ __forceinline__ void hlle_flux_x(
    double rho_L, double u_L, double p_L,
    double rho_R, double u_R, double p_R,
    double gamma,
    double& F_rho, double& F_mom, double& F_erg
) {
    double c_L = sound_speed(gamma, rho_L, p_L);
    double c_R = sound_speed(gamma, rho_R, p_R);
    double E_L = p_L / (gamma - 1.0) + 0.5 * rho_L * u_L * u_L;  // Only x-KE for 1D flux
    double E_R = p_R / (gamma - 1.0) + 0.5 * rho_R * u_R * u_R;
    
    double S_L = fmin(u_L - c_L, u_R - c_R);
    double S_R = fmax(u_L + c_L, u_R + c_R);
    
    double FL_rho = rho_L * u_L;
    double FL_mom = rho_L * u_L * u_L + p_L;
    double FL_erg = (E_L + p_L) * u_L;
    
    double FR_rho = rho_R * u_R;
    double FR_mom = rho_R * u_R * u_R + p_R;
    double FR_erg = (E_R + p_R) * u_R;
    
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
// X-Sweep Kernel (fused recon+riemann+update)
// =============================================================================

__global__ void euler2d_x_sweep_kernel(
    const double* __restrict__ U_in,
    double* __restrict__ U_out,
    int nx, int ny,         // Interior dimensions
    int halo,               // Halo width
    double dx, double dt,
    double gamma
) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tid_x >= nx || tid_y >= ny) return;
    
    int i = tid_x + halo;
    int j = tid_y + halo;
    int nx_total = nx + 2 * halo;
    int ny_total = ny + 2 * halo;
    int n_total = nx_total * ny_total;
    
    // SoA index: var * n_total + j * nx_total + i
    auto idx = [&](int var, int ii, int jj) {
        return var * n_total + jj * nx_total + ii;
    };
    
    // Load stencil along x (5 cells)
    double rho[5], mou[5], mov[5], erg[5];
    
    #pragma unroll
    for (int s = 0; s < 5; ++s) {
        int ii = i - 2 + s;
        rho[s] = U_in[idx(I_RHO, ii, j)];
        mou[s] = U_in[idx(I_MOU, ii, j)];
        mov[s] = U_in[idx(I_MOV, ii, j)];
        erg[s] = U_in[idx(I_ERG, ii, j)];
    }
    
    // Convert to primitives
    double u[5], v[5], p[5];
    
    #pragma unroll
    for (int s = 0; s < 5; ++s) {
        rho[s] = fmax(rho[s], 1e-10);
        u[s] = mou[s] / rho[s];
        v[s] = mov[s] / rho[s];
        p[s] = pressure_2d(gamma, rho[s], mou[s], mov[s], erg[s]);
    }
    
    // PLM slopes for cells i-1, i, i+1
    double slope_rho[3], slope_u[3], slope_p[3], slope_v[3];
    
    #pragma unroll
    for (int s = 0; s < 3; ++s) {
        slope_rho[s] = plm_slope(rho[s], rho[s+1], rho[s+2]);
        slope_u[s]   = plm_slope(u[s], u[s+1], u[s+2]);
        slope_v[s]   = plm_slope(v[s], v[s+1], v[s+2]);
        slope_p[s]   = plm_slope(p[s], p[s+1], p[s+2]);
    }
    
    // Reconstruct at i-1/2
    double rho_L_imh = fmax(rho[1] + 0.5 * slope_rho[0], 1e-10);
    double rho_R_imh = fmax(rho[2] - 0.5 * slope_rho[1], 1e-10);
    double u_L_imh   = u[1] + 0.5 * slope_u[0];
    double u_R_imh   = u[2] - 0.5 * slope_u[1];
    double v_L_imh   = v[1] + 0.5 * slope_v[0];
    double v_R_imh   = v[2] - 0.5 * slope_v[1];
    double p_L_imh   = fmax(p[1] + 0.5 * slope_p[0], 1e-10);
    double p_R_imh   = fmax(p[2] - 0.5 * slope_p[1], 1e-10);
    
    // Reconstruct at i+1/2
    double rho_L_iph = fmax(rho[2] + 0.5 * slope_rho[1], 1e-10);
    double rho_R_iph = fmax(rho[3] - 0.5 * slope_rho[2], 1e-10);
    double u_L_iph   = u[2] + 0.5 * slope_u[1];
    double u_R_iph   = u[3] - 0.5 * slope_u[2];
    double v_L_iph   = v[2] + 0.5 * slope_v[1];
    double v_R_iph   = v[3] - 0.5 * slope_v[2];
    double p_L_iph   = fmax(p[2] + 0.5 * slope_p[1], 1e-10);
    double p_R_iph   = fmax(p[3] - 0.5 * slope_p[2], 1e-10);
    
    // HLLE flux at interfaces (for rho, rho*u, E_1d)
    double F_rho_L, F_mou_L, F_erg_L;
    double F_rho_R, F_mou_R, F_erg_R;
    
    hlle_flux_x(rho_L_imh, u_L_imh, p_L_imh,
                rho_R_imh, u_R_imh, p_R_imh,
                gamma, F_rho_L, F_mou_L, F_erg_L);
    
    hlle_flux_x(rho_L_iph, u_L_iph, p_L_iph,
                rho_R_iph, u_R_iph, p_R_iph,
                gamma, F_rho_R, F_mou_R, F_erg_R);
    
    // Passive advection of rho*v
    double u_int_L = 0.5 * (u_L_imh + u_R_imh);  // Interface velocity at i-1/2
    double u_int_R = 0.5 * (u_L_iph + u_R_iph);  // Interface velocity at i+1/2
    double rho_v_L = (u_int_L > 0) ? rho_L_imh * v_L_imh : rho_R_imh * v_R_imh;
    double rho_v_R = (u_int_R > 0) ? rho_L_iph * v_L_iph : rho_R_iph * v_R_iph;
    double F_mov_L = u_int_L * rho_v_L;
    double F_mov_R = u_int_R * rho_v_R;
    
    // Compute full E flux including v kinetic energy
    double E_orig = erg[2];  // Original total energy
    double rho_v_orig = mov[2];  // Original rho*v
    
    // Update (compensate energy flux for v kinetic energy advection)
    double dtdx = dt / dx;
    
    U_out[idx(I_RHO, i, j)] = rho[2] - dtdx * (F_rho_R - F_rho_L);
    U_out[idx(I_MOU, i, j)] = mou[2] - dtdx * (F_mou_R - F_mou_L);
    U_out[idx(I_MOV, i, j)] = mov[2] - dtdx * (F_mov_R - F_mov_L);
    
    // Energy: add KE contribution from v advection
    double v_ke_flux_L = 0.5 * v_L_imh * v_L_imh * F_rho_L;  // 0.5*v^2 advected with mass flux
    double v_ke_flux_R = 0.5 * v_R_iph * v_R_iph * F_rho_R;
    U_out[idx(I_ERG, i, j)] = E_orig - dtdx * (F_erg_R - F_erg_L + v_ke_flux_R - v_ke_flux_L);
}

// =============================================================================
// Y-Sweep Kernel (fused recon+riemann+update)
// =============================================================================

__global__ void euler2d_y_sweep_kernel(
    const double* __restrict__ U_in,
    double* __restrict__ U_out,
    int nx, int ny,
    int halo,
    double dy, double dt,
    double gamma
) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tid_x >= nx || tid_y >= ny) return;
    
    int i = tid_x + halo;
    int j = tid_y + halo;
    int nx_total = nx + 2 * halo;
    int ny_total = ny + 2 * halo;
    int n_total = nx_total * ny_total;
    
    auto idx = [&](int var, int ii, int jj) {
        return var * n_total + jj * nx_total + ii;
    };
    
    // Load stencil along y (5 cells)
    double rho[5], mou[5], mov[5], erg[5];
    
    #pragma unroll
    for (int s = 0; s < 5; ++s) {
        int jj = j - 2 + s;
        rho[s] = U_in[idx(I_RHO, i, jj)];
        mou[s] = U_in[idx(I_MOU, i, jj)];
        mov[s] = U_in[idx(I_MOV, i, jj)];
        erg[s] = U_in[idx(I_ERG, i, jj)];
    }
    
    // Convert to primitives
    double u[5], v[5], p[5];
    
    #pragma unroll
    for (int s = 0; s < 5; ++s) {
        rho[s] = fmax(rho[s], 1e-10);
        u[s] = mou[s] / rho[s];
        v[s] = mov[s] / rho[s];
        p[s] = pressure_2d(gamma, rho[s], mou[s], mov[s], erg[s]);
    }
    
    // PLM slopes (along y)
    double slope_rho[3], slope_v[3], slope_p[3], slope_u[3];
    
    #pragma unroll
    for (int s = 0; s < 3; ++s) {
        slope_rho[s] = plm_slope(rho[s], rho[s+1], rho[s+2]);
        slope_v[s]   = plm_slope(v[s], v[s+1], v[s+2]);
        slope_u[s]   = plm_slope(u[s], u[s+1], u[s+2]);
        slope_p[s]   = plm_slope(p[s], p[s+1], p[s+2]);
    }
    
    // Reconstruct at j-1/2
    double rho_L_jmh = fmax(rho[1] + 0.5 * slope_rho[0], 1e-10);
    double rho_R_jmh = fmax(rho[2] - 0.5 * slope_rho[1], 1e-10);
    double v_L_jmh   = v[1] + 0.5 * slope_v[0];
    double v_R_jmh   = v[2] - 0.5 * slope_v[1];
    double u_L_jmh   = u[1] + 0.5 * slope_u[0];
    double u_R_jmh   = u[2] - 0.5 * slope_u[1];
    double p_L_jmh   = fmax(p[1] + 0.5 * slope_p[0], 1e-10);
    double p_R_jmh   = fmax(p[2] - 0.5 * slope_p[1], 1e-10);
    
    // Reconstruct at j+1/2
    double rho_L_jph = fmax(rho[2] + 0.5 * slope_rho[1], 1e-10);
    double rho_R_jph = fmax(rho[3] - 0.5 * slope_rho[2], 1e-10);
    double v_L_jph   = v[2] + 0.5 * slope_v[1];
    double v_R_jph   = v[3] - 0.5 * slope_v[2];
    double u_L_jph   = u[2] + 0.5 * slope_u[1];
    double u_R_jph   = u[3] - 0.5 * slope_u[2];
    double p_L_jph   = fmax(p[2] + 0.5 * slope_p[1], 1e-10);
    double p_R_jph   = fmax(p[3] - 0.5 * slope_p[2], 1e-10);
    
    // HLLE flux in y (for rho, rho*v, E_1d) - swap u<->v roles
    double F_rho_L, F_mov_L, F_erg_L;
    double F_rho_R, F_mov_R, F_erg_R;
    
    // Use hlle_flux_x but with v as the "x-velocity"
    hlle_flux_x(rho_L_jmh, v_L_jmh, p_L_jmh,
                rho_R_jmh, v_R_jmh, p_R_jmh,
                gamma, F_rho_L, F_mov_L, F_erg_L);
    
    hlle_flux_x(rho_L_jph, v_L_jph, p_L_jph,
                rho_R_jph, v_R_jph, p_R_jph,
                gamma, F_rho_R, F_mov_R, F_erg_R);
    
    // Passive advection of rho*u
    double v_int_L = 0.5 * (v_L_jmh + v_R_jmh);
    double v_int_R = 0.5 * (v_L_jph + v_R_jph);
    double rho_u_L = (v_int_L > 0) ? rho_L_jmh * u_L_jmh : rho_R_jmh * u_R_jmh;
    double rho_u_R = (v_int_R > 0) ? rho_L_jph * u_L_jph : rho_R_jph * u_R_jph;
    double F_mou_L = v_int_L * rho_u_L;
    double F_mou_R = v_int_R * rho_u_R;
    
    double E_orig = erg[2];
    double dtdy = dt / dy;
    
    U_out[idx(I_RHO, i, j)] = rho[2] - dtdy * (F_rho_R - F_rho_L);
    U_out[idx(I_MOU, i, j)] = mou[2] - dtdy * (F_mou_R - F_mou_L);
    U_out[idx(I_MOV, i, j)] = mov[2] - dtdy * (F_mov_R - F_mov_L);
    
    // Energy: add KE contribution from u advection
    double u_ke_flux_L = 0.5 * u_L_jmh * u_L_jmh * F_rho_L;
    double u_ke_flux_R = 0.5 * u_R_jph * u_R_jph * F_rho_R;
    U_out[idx(I_ERG, i, j)] = E_orig - dtdy * (F_erg_R - F_erg_L + u_ke_flux_R - u_ke_flux_L);
}

// =============================================================================
// Shared Memory X-Sweep (compile with NEST_CUDA_SHARED_MEM)
// =============================================================================

#ifdef NEST_CUDA_SHARED_MEM

__global__ void euler2d_x_sweep_smem_kernel(
    const double* __restrict__ U_in,
    double* __restrict__ U_out,
    int nx, int ny,
    int halo,
    double dx, double dt,
    double gamma
) {
    // Tile dimensions including halos
    constexpr int TILE_X = BLOCK_X + 2 * STENCIL_RADIUS;
    constexpr int TILE_Y = BLOCK_Y;
    
    __shared__ double s_rho[TILE_Y][TILE_X];
    __shared__ double s_mou[TILE_Y][TILE_X];
    __shared__ double s_mov[TILE_Y][TILE_X];
    __shared__ double s_erg[TILE_Y][TILE_X];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * BLOCK_X + tx;
    int gy = blockIdx.y * BLOCK_Y + ty;
    
    int nx_total = nx + 2 * halo;
    int ny_total = ny + 2 * halo;
    int n_total = nx_total * ny_total;
    
    int i_global = gx + halo;
    int j_global = gy + halo;
    
    auto idx_global = [&](int var, int ii, int jj) {
        return var * n_total + jj * nx_total + ii;
    };
    
    // Cooperative load: main tile
    int sx = tx + STENCIL_RADIUS;
    if (gx < nx && gy < ny) {
        s_rho[ty][sx] = U_in[idx_global(I_RHO, i_global, j_global)];
        s_mou[ty][sx] = U_in[idx_global(I_MOU, i_global, j_global)];
        s_mov[ty][sx] = U_in[idx_global(I_MOV, i_global, j_global)];
        s_erg[ty][sx] = U_in[idx_global(I_ERG, i_global, j_global)];
    }
    
    // Load left halo
    if (tx < STENCIL_RADIUS && gy < ny) {
        int ii = i_global - STENCIL_RADIUS;
        s_rho[ty][tx] = U_in[idx_global(I_RHO, ii, j_global)];
        s_mou[ty][tx] = U_in[idx_global(I_MOU, ii, j_global)];
        s_mov[ty][tx] = U_in[idx_global(I_MOV, ii, j_global)];
        s_erg[ty][tx] = U_in[idx_global(I_ERG, ii, j_global)];
    }
    
    // Load right halo
    if (tx >= BLOCK_X - STENCIL_RADIUS && gx < nx && gy < ny) {
        int halo_sx = sx + STENCIL_RADIUS;
        int ii = i_global + STENCIL_RADIUS;
        if (halo_sx < TILE_X && ii < nx_total) {
            s_rho[ty][halo_sx] = U_in[idx_global(I_RHO, ii, j_global)];
            s_mou[ty][halo_sx] = U_in[idx_global(I_MOU, ii, j_global)];
            s_mov[ty][halo_sx] = U_in[idx_global(I_MOV, ii, j_global)];
            s_erg[ty][halo_sx] = U_in[idx_global(I_ERG, ii, j_global)];
        }
    }
    
    __syncthreads();
    
    if (gx >= nx || gy >= ny) return;
    
    // Compute from shared memory (same logic as global memory version)
    double rho[5], mou[5], mov[5], erg[5];
    
    #pragma unroll
    for (int s = 0; s < 5; ++s) {
        int ssx = sx - 2 + s;
        rho[s] = s_rho[ty][ssx];
        mou[s] = s_mou[ty][ssx];
        mov[s] = s_mov[ty][ssx];
        erg[s] = s_erg[ty][ssx];
    }
    
    // Same computation as global memory kernel...
    double u[5], v[5], p[5];
    
    #pragma unroll
    for (int s = 0; s < 5; ++s) {
        rho[s] = fmax(rho[s], 1e-10);
        u[s] = mou[s] / rho[s];
        v[s] = mov[s] / rho[s];
        p[s] = pressure_2d(gamma, rho[s], mou[s], mov[s], erg[s]);
    }
    
    double slope_rho[3], slope_u[3], slope_v[3], slope_p[3];
    
    #pragma unroll
    for (int s = 0; s < 3; ++s) {
        slope_rho[s] = plm_slope(rho[s], rho[s+1], rho[s+2]);
        slope_u[s]   = plm_slope(u[s], u[s+1], u[s+2]);
        slope_v[s]   = plm_slope(v[s], v[s+1], v[s+2]);
        slope_p[s]   = plm_slope(p[s], p[s+1], p[s+2]);
    }
    
    double rho_L_imh = fmax(rho[1] + 0.5 * slope_rho[0], 1e-10);
    double rho_R_imh = fmax(rho[2] - 0.5 * slope_rho[1], 1e-10);
    double u_L_imh   = u[1] + 0.5 * slope_u[0];
    double u_R_imh   = u[2] - 0.5 * slope_u[1];
    double v_L_imh   = v[1] + 0.5 * slope_v[0];
    double v_R_imh   = v[2] - 0.5 * slope_v[1];
    double p_L_imh   = fmax(p[1] + 0.5 * slope_p[0], 1e-10);
    double p_R_imh   = fmax(p[2] - 0.5 * slope_p[1], 1e-10);
    
    double rho_L_iph = fmax(rho[2] + 0.5 * slope_rho[1], 1e-10);
    double rho_R_iph = fmax(rho[3] - 0.5 * slope_rho[2], 1e-10);
    double u_L_iph   = u[2] + 0.5 * slope_u[1];
    double u_R_iph   = u[3] - 0.5 * slope_u[2];
    double v_L_iph   = v[2] + 0.5 * slope_v[1];
    double v_R_iph   = v[3] - 0.5 * slope_v[2];
    double p_L_iph   = fmax(p[2] + 0.5 * slope_p[1], 1e-10);
    double p_R_iph   = fmax(p[3] - 0.5 * slope_p[2], 1e-10);
    
    double F_rho_L, F_mou_L, F_erg_L;
    double F_rho_R, F_mou_R, F_erg_R;
    
    hlle_flux_x(rho_L_imh, u_L_imh, p_L_imh,
                rho_R_imh, u_R_imh, p_R_imh,
                gamma, F_rho_L, F_mou_L, F_erg_L);
    
    hlle_flux_x(rho_L_iph, u_L_iph, p_L_iph,
                rho_R_iph, u_R_iph, p_R_iph,
                gamma, F_rho_R, F_mou_R, F_erg_R);
    
    double u_int_L = 0.5 * (u_L_imh + u_R_imh);
    double u_int_R = 0.5 * (u_L_iph + u_R_iph);
    double rho_v_L = (u_int_L > 0) ? rho_L_imh * v_L_imh : rho_R_imh * v_R_imh;
    double rho_v_R = (u_int_R > 0) ? rho_L_iph * v_L_iph : rho_R_iph * v_R_iph;
    double F_mov_L = u_int_L * rho_v_L;
    double F_mov_R = u_int_R * rho_v_R;
    
    double E_orig = erg[2];
    double dtdx = dt / dx;
    
    U_out[idx_global(I_RHO, i_global, j_global)] = rho[2] - dtdx * (F_rho_R - F_rho_L);
    U_out[idx_global(I_MOU, i_global, j_global)] = mou[2] - dtdx * (F_mou_R - F_mou_L);
    U_out[idx_global(I_MOV, i_global, j_global)] = mov[2] - dtdx * (F_mov_R - F_mov_L);
    
    double v_ke_flux_L = 0.5 * v_L_imh * v_L_imh * F_rho_L;
    double v_ke_flux_R = 0.5 * v_R_iph * v_R_iph * F_rho_R;
    U_out[idx_global(I_ERG, i_global, j_global)] = E_orig - dtdx * (F_erg_R - F_erg_L + v_ke_flux_R - v_ke_flux_L);
}

#endif // NEST_CUDA_SHARED_MEM

// =============================================================================
// Halo kernels for 2D
// =============================================================================

__global__ void copy_halos_2d_x_kernel(
    double* U,
    int nx, int ny, int halo, int nvars
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nx_total = nx + 2 * halo;
    int ny_total = ny + 2 * halo;
    int n_total = nx_total * ny_total;
    
    int var = tid / (ny_total * halo);
    int rem = tid % (ny_total * halo);
    int j = rem / halo;
    int h = rem % halo;
    
    if (var >= nvars) return;
    
    // Left halo: copy from right interior
    int src_left = var * n_total + j * nx_total + (halo + nx - halo + h);
    int dst_left = var * n_total + j * nx_total + h;
    U[dst_left] = U[src_left];
    
    // Right halo
    int src_right = var * n_total + j * nx_total + (halo + h);
    int dst_right = var * n_total + j * nx_total + (halo + nx + h);
    U[dst_right] = U[src_right];
}

__global__ void copy_halos_2d_y_kernel(
    double* U,
    int nx, int ny, int halo, int nvars
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nx_total = nx + 2 * halo;
    int ny_total = ny + 2 * halo;
    int n_total = nx_total * ny_total;
    
    int var = tid / (nx_total * halo);
    int rem = tid % (nx_total * halo);
    int i = rem / halo;
    int h = rem % halo;
    
    if (var >= nvars) return;
    
    // Bottom halo
    int src_bot = var * n_total + (halo + ny - halo + h) * nx_total + i;
    int dst_bot = var * n_total + h * nx_total + i;
    U[dst_bot] = U[src_bot];
    
    // Top halo
    int src_top = var * n_total + (halo + h) * nx_total + i;
    int dst_top = var * n_total + (halo + ny + h) * nx_total + i;
    U[dst_top] = U[src_top];
}

// =============================================================================
// CFL Kernel for 2D
// =============================================================================

__global__ void compute_max_wavespeed_2d_kernel(
    const double* __restrict__ U,
    double* __restrict__ block_max,
    int nx, int ny, int halo,
    double gamma
) {
    extern __shared__ double shared_max[];
    
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int lid = threadIdx.y * blockDim.x + threadIdx.x;
    
    int nx_total = nx + 2 * halo;
    int ny_total = ny + 2 * halo;
    int n_total = nx_total * ny_total;
    
    double local_max = 0.0;
    
    if (tid_x < nx && tid_y < ny) {
        int i = tid_x + halo;
        int j = tid_y + halo;
        int idx = j * nx_total + i;
        
        double rho = fmax(U[I_RHO * n_total + idx], 1e-10);
        double mou = U[I_MOU * n_total + idx];
        double mov = U[I_MOV * n_total + idx];
        double erg = U[I_ERG * n_total + idx];
        
        double u = mou / rho;
        double v = mov / rho;
        double p = pressure_2d(gamma, rho, mou, mov, erg);
        double c = sound_speed(gamma, rho, p);
        
        local_max = fmax(fabs(u), fabs(v)) + c;
    }
    
    shared_max[lid] = local_max;
    __syncthreads();
    
    int block_size = blockDim.x * blockDim.y;
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_max[lid] = fmax(shared_max[lid], shared_max[lid + s]);
        }
        __syncthreads();
    }
    
    if (lid == 0) {
        int block_id = blockIdx.y * gridDim.x + blockIdx.x;
        block_max[block_id] = shared_max[0];
    }
}

// =============================================================================
// RK2 Average Kernel for 2D
// =============================================================================

__global__ void rk2_average_2d_kernel(
    double* __restrict__ U,
    const double* __restrict__ U_star,
    int nx, int ny, int halo, int nvars
) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tid_x >= nx || tid_y >= ny) return;
    
    int i = tid_x + halo;
    int j = tid_y + halo;
    int nx_total = nx + 2 * halo;
    int n_total = nx_total * (ny + 2 * halo);
    int idx = j * nx_total + i;
    
    for (int var = 0; var < nvars; ++var) {
        int k = var * n_total + idx;
        U[k] = 0.5 * (U[k] + U_star[k]);
    }
}

} // namespace nest::cuda::euler2d

