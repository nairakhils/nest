#pragma once
#include "../sod1d/euler1d.hpp"
#include "nest/patch.hpp"

namespace nest::euler2d {

using namespace euler1d;

// 2D conserved variable indices
constexpr int I2_RHO = 0;   // Density
constexpr int I2_MU  = 1;   // x-momentum (rho*u)
constexpr int I2_MV  = 2;   // y-momentum (rho*v)
constexpr int I2_ERG = 3;   // Total energy

// =============================================================================
// 2D Euler via directional splitting with passive scalar advection
// =============================================================================

// Compute interface velocity for upwinding passive scalars
inline double compute_interface_velocity(double u_L, double u_R, double c_L, double c_R) {
    double S_L = std::min(u_L - c_L, u_R - c_R);
    double S_R = std::max(u_L + c_L, u_R + c_R);
    
    if (S_L >= 0.0) return u_L;
    if (S_R <= 0.0) return u_R;
    
    return (S_R * u_L - S_L * u_R) / (S_R - S_L);
}

// Compute passive scalar flux using upwinding
inline double passive_flux(double q_L, double q_R, double u_interface) {
    return (u_interface >= 0.0) ? u_interface * q_L : u_interface * q_R;
}

// Helper to get primitives from conserved at a single cell
inline void get_prim_2d(
    const field_t<double, 2>& U,
    ivec_t<2> idx,
    double W[NPRIM],      // Output: [rho, u_normal, p]
    double& q_transverse, // Output: rho * v_transverse
    const eos_t& eos,
    bool is_x_sweep       // true = x-sweep (u is normal), false = y-sweep (v is normal)
) {
    double rho = std::max(U(I2_RHO, idx), 1e-10);  // Safety clamp
    double rho_u = U(I2_MU, idx);
    double rho_v = U(I2_MV, idx);
    double E = U(I2_ERG, idx);
    
    double u = rho_u / rho;
    double v = rho_v / rho;
    double ke = 0.5 * rho * (u*u + v*v);
    double p = std::max((eos.gamma - 1.0) * (E - ke), 1e-10);  // Safety clamp
    
    W[I_RHO] = rho;
    W[I_PRE] = p;
    
    if (is_x_sweep) {
        W[I_VEL] = u;           // Normal velocity
        q_transverse = rho_v;   // Transverse momentum
    } else {
        W[I_VEL] = v;           // Normal velocity
        q_transverse = rho_u;   // Transverse momentum
    }
}

// Sweep in x-direction
struct x_sweep_t {
    const eos_t& eos;
    double dx;
    limiter_t limiter;
    
    x_sweep_t(const eos_t& eos_, double dx_, limiter_t lim = limiter_t::minmod)
        : eos(eos_), dx(dx_), limiter(lim) {}
    
    void operator()(
        const field_t<double, 2>& U_in,
        field_t<double, 2>& U_out,
        const index_space_t<2>& interior,
        double dt
    ) const {
        (void)space(U_in[0]); // extended space available if needed later
        
        auto i0 = start(interior)[0];
        auto i1 = upper(interior)[0];
        auto j0 = start(interior)[1];
        auto j1 = upper(interior)[1];
        
        for (int j = j0; j < j1; ++j) {
            for (int i = i0; i < i1; ++i) {
                auto idx = ivec(i, j);
                
                // Get primitives for stencil (on-the-fly, no corner access)
                double W_im2[NPRIM], W_im1[NPRIM], W_i[NPRIM], W_ip1[NPRIM], W_ip2[NPRIM];
                double rv_im2, rv_im1, rv_i, rv_ip1, rv_ip2;  // rho*v (transverse momentum)
                
                get_prim_2d(U_in, ivec(i-2, j), W_im2, rv_im2, eos, true);
                get_prim_2d(U_in, ivec(i-1, j), W_im1, rv_im1, eos, true);
                get_prim_2d(U_in, ivec(i, j),   W_i,   rv_i,   eos, true);
                get_prim_2d(U_in, ivec(i+1, j), W_ip1, rv_ip1, eos, true);
                get_prim_2d(U_in, ivec(i+2, j), W_ip2, rv_ip2, eos, true);
                
                // --- Left interface (i-1/2) ---
                auto recon_im1 = plm_reconstruct(W_im2, W_im1, W_i, limiter);
                auto recon_i = plm_reconstruct(W_im1, W_i, W_ip1, limiter);
                
                double W_L[NPRIM], W_R[NPRIM];
                for (std::size_t k = 0; k < NPRIM; ++k) {
                    W_L[k] = recon_im1.W_R[k];  // Right state of cell i-1
                    W_R[k] = recon_i.W_L[k];    // Left state of cell i
                }
                
                double flux_L[NCONS];
                hlle_flux(W_L, W_R, flux_L, eos);
                
                // Interface velocity and passive scalar flux for rho*v
                double c_L = eos.sound_speed(W_L[I_RHO], W_L[I_PRE]);
                double c_R = eos.sound_speed(W_R[I_RHO], W_R[I_PRE]);
                double u_int_L = compute_interface_velocity(W_L[I_VEL], W_R[I_VEL], c_L, c_R);
                
                double rv_L_recon = rv_im1 + 0.5 * apply_limiter(rv_im1 - rv_im2, rv_i - rv_im1, limiter);
                double rv_R_recon = rv_i - 0.5 * apply_limiter(rv_i - rv_im1, rv_ip1 - rv_i, limiter);
                double flux_rv_L = passive_flux(rv_L_recon, rv_R_recon, u_int_L);
                
                // --- Right interface (i+1/2) ---
                auto recon_ip1 = plm_reconstruct(W_i, W_ip1, W_ip2, limiter);
                
                for (std::size_t k = 0; k < NPRIM; ++k) {
                    W_L[k] = recon_i.W_R[k];     // Right state of cell i
                    W_R[k] = recon_ip1.W_L[k];  // Left state of cell i+1
                }
                
                double flux_R[NCONS];
                hlle_flux(W_L, W_R, flux_R, eos);
                
                c_L = eos.sound_speed(W_L[I_RHO], W_L[I_PRE]);
                c_R = eos.sound_speed(W_R[I_RHO], W_R[I_PRE]);
                double u_int_R = compute_interface_velocity(W_L[I_VEL], W_R[I_VEL], c_L, c_R);
                
                double rv_L_recon2 = rv_i + 0.5 * apply_limiter(rv_i - rv_im1, rv_ip1 - rv_i, limiter);
                double rv_R_recon2 = rv_ip1 - 0.5 * apply_limiter(rv_ip1 - rv_i, rv_ip2 - rv_ip1, limiter);
                double flux_rv_R = passive_flux(rv_L_recon2, rv_R_recon2, u_int_R);
                
                // --- Update conserved variables ---
                double dtdx = dt / dx;
                U_out(I2_RHO, idx) = U_in(I2_RHO, idx) - dtdx * (flux_R[0] - flux_L[0]);
                U_out(I2_MU, idx)  = U_in(I2_MU, idx)  - dtdx * (flux_R[1] - flux_L[1]);
                U_out(I2_MV, idx)  = U_in(I2_MV, idx)  - dtdx * (flux_rv_R - flux_rv_L);
                U_out(I2_ERG, idx) = U_in(I2_ERG, idx) - dtdx * (flux_R[2] - flux_L[2]);
            }
        }
    }
};

// Sweep in y-direction
struct y_sweep_t {
    const eos_t& eos;
    double dy;
    limiter_t limiter;
    
    y_sweep_t(const eos_t& eos_, double dy_, limiter_t lim = limiter_t::minmod)
        : eos(eos_), dy(dy_), limiter(lim) {}
    
    void operator()(
        const field_t<double, 2>& U_in,
        field_t<double, 2>& U_out,
        const index_space_t<2>& interior,
        double dt
    ) const {
        // U_out is fully overwritten on interior by the update below.
        
        auto i0 = start(interior)[0];
        auto i1 = upper(interior)[0];
        auto j0 = start(interior)[1];
        auto j1 = upper(interior)[1];
        
        for (int i = i0; i < i1; ++i) {
            for (int j = j0; j < j1; ++j) {
                auto idx = ivec(i, j);
                
                // Get primitives for stencil (on-the-fly)
                double W_jm2[NPRIM], W_jm1[NPRIM], W_j[NPRIM], W_jp1[NPRIM], W_jp2[NPRIM];
                double ru_jm2, ru_jm1, ru_j, ru_jp1, ru_jp2;  // rho*u (transverse momentum)
                
                get_prim_2d(U_in, ivec(i, j-2), W_jm2, ru_jm2, eos, false);
                get_prim_2d(U_in, ivec(i, j-1), W_jm1, ru_jm1, eos, false);
                get_prim_2d(U_in, ivec(i, j),   W_j,   ru_j,   eos, false);
                get_prim_2d(U_in, ivec(i, j+1), W_jp1, ru_jp1, eos, false);
                get_prim_2d(U_in, ivec(i, j+2), W_jp2, ru_jp2, eos, false);
                
                // --- Bottom interface (j-1/2) ---
                auto recon_jm1 = plm_reconstruct(W_jm2, W_jm1, W_j, limiter);
                auto recon_j = plm_reconstruct(W_jm1, W_j, W_jp1, limiter);
                
                double W_L[NPRIM], W_R[NPRIM];
                for (std::size_t k = 0; k < NPRIM; ++k) {
                    W_L[k] = recon_jm1.W_R[k];
                    W_R[k] = recon_j.W_L[k];
                }
                
                double flux_L[NCONS];
                hlle_flux(W_L, W_R, flux_L, eos);
                
                double c_L = eos.sound_speed(W_L[I_RHO], W_L[I_PRE]);
                double c_R = eos.sound_speed(W_R[I_RHO], W_R[I_PRE]);
                double v_int_L = compute_interface_velocity(W_L[I_VEL], W_R[I_VEL], c_L, c_R);
                
                double ru_L_recon = ru_jm1 + 0.5 * apply_limiter(ru_jm1 - ru_jm2, ru_j - ru_jm1, limiter);
                double ru_R_recon = ru_j - 0.5 * apply_limiter(ru_j - ru_jm1, ru_jp1 - ru_j, limiter);
                double flux_ru_L = passive_flux(ru_L_recon, ru_R_recon, v_int_L);
                
                // --- Top interface (j+1/2) ---
                auto recon_jp1 = plm_reconstruct(W_j, W_jp1, W_jp2, limiter);
                
                for (std::size_t k = 0; k < NPRIM; ++k) {
                    W_L[k] = recon_j.W_R[k];
                    W_R[k] = recon_jp1.W_L[k];
                }
                
                double flux_R[NCONS];
                hlle_flux(W_L, W_R, flux_R, eos);
                
                c_L = eos.sound_speed(W_L[I_RHO], W_L[I_PRE]);
                c_R = eos.sound_speed(W_R[I_RHO], W_R[I_PRE]);
                double v_int_R = compute_interface_velocity(W_L[I_VEL], W_R[I_VEL], c_L, c_R);
                
                double ru_L_recon2 = ru_j + 0.5 * apply_limiter(ru_j - ru_jm1, ru_jp1 - ru_j, limiter);
                double ru_R_recon2 = ru_jp1 - 0.5 * apply_limiter(ru_jp1 - ru_j, ru_jp2 - ru_jp1, limiter);
                double flux_ru_R = passive_flux(ru_L_recon2, ru_R_recon2, v_int_R);
                
                // --- Update conserved variables ---
                double dtdy = dt / dy;
                U_out(I2_RHO, idx) = U_in(I2_RHO, idx) - dtdy * (flux_R[0] - flux_L[0]);
                U_out(I2_MU, idx)  = U_in(I2_MU, idx)  - dtdy * (flux_ru_R - flux_ru_L);
                U_out(I2_MV, idx)  = U_in(I2_MV, idx)  - dtdy * (flux_R[1] - flux_L[1]);
                U_out(I2_ERG, idx) = U_in(I2_ERG, idx) - dtdy * (flux_R[2] - flux_L[2]);
            }
        }
    }
};

// 2D CFL condition
inline double compute_dt_2d(
    const std::vector<patch_t<double, 2>>& patches,
    double dx,
    double dy,
    double cfl,
    const eos_t& eos
) {
    double dt_x = 1e10;
    double dt_y = 1e10;
    
    for (const auto& patch : patches) {
        for (auto idx : patch.interior) {
            double rho = std::max(patch(I2_RHO, idx), 1e-10);
            double u = patch(I2_MU, idx) / rho;
            double v = patch(I2_MV, idx) / rho;
            
            double ke = 0.5 * rho * (u*u + v*v);
            double p = std::max((eos.gamma - 1.0) * (patch(I2_ERG, idx) - ke), 1e-10);
            double c = eos.sound_speed(rho, p);
            
            double speed_x = std::abs(u) + c;
            double speed_y = std::abs(v) + c;
            
            if (speed_x > 0.0) dt_x = std::min(dt_x, dx / speed_x);
            if (speed_y > 0.0) dt_y = std::min(dt_y, dy / speed_y);
        }
    }
    
    return cfl * std::min(dt_x, dt_y);
}

} // namespace nest::euler2d
