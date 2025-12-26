#pragma once
#include "nest/core.hpp"
#include <cmath>
#include <algorithm>
#include <cassert>

namespace nest::euler1d {

// =============================================================================
// Constants and types
// =============================================================================

constexpr std::size_t NCONS = 3;  // Conserved: rho, rho*u, E
constexpr std::size_t NPRIM = 3;  // Primitive: rho, u, p

constexpr int I_RHO = 0;   // Density
constexpr int I_MOM = 1;   // Momentum (rho*u)
constexpr int I_ERG = 2;   // Total energy (E)

constexpr int I_VEL = 1;   // Velocity (u)
constexpr int I_PRE = 2;   // Pressure (p)

// =============================================================================
// Gamma-law equation of state
// =============================================================================

struct eos_t {
    double gamma = 1.4;
    
    // Pressure from conserved variables: p = (gamma-1) * (E - 0.5*rho*u^2)
    auto pressure(double rho, double mom, double erg) const -> double {
        double u = mom / rho;
        double ke = 0.5 * rho * u * u;
        double p = (gamma - 1.0) * (erg - ke);
        
        #ifndef NDEBUG
        if (p < 0.0) {
            // Debug check: pressure should be positive
            assert(p >= 0.0 && "Negative pressure detected!");
        }
        #endif
        
        return std::max(p, 0.0);  // Safety clamp
    }
    
    // Total energy from primitives: E = p/(gamma-1) + 0.5*rho*u^2
    auto total_energy(double rho, double u, double p) const -> double {
        double ke = 0.5 * rho * u * u;
        double ie = p / (gamma - 1.0);
        return ie + ke;
    }
    
    // Sound speed: c = sqrt(gamma * p / rho)
    auto sound_speed(double rho, double p) const -> double {
        return std::sqrt(gamma * p / rho);
    }
    
    // Specific internal energy: e = p / (rho * (gamma - 1))
    auto specific_internal_energy(double rho, double p) const -> double {
        return p / (rho * (gamma - 1.0));
    }
};

// =============================================================================
// Primitive <-> Conserved conversion
// =============================================================================

// Convert conserved to primitive: U -> W
inline void cons_to_prim(const double* U, double* W, const eos_t& eos) {
    double rho = U[I_RHO];
    double mom = U[I_MOM];
    double erg = U[I_ERG];
    
    #ifndef NDEBUG
    assert(rho > 0.0 && "Non-positive density in cons_to_prim!");
    #endif
    
    double u = mom / rho;
    double p = eos.pressure(rho, mom, erg);
    
    W[I_RHO] = rho;
    W[I_VEL] = u;
    W[I_PRE] = p;
}

// Convert primitive to conserved: W -> U
inline void prim_to_cons(const double* W, double* U, const eos_t& eos) {
    double rho = W[I_RHO];
    double u = W[I_VEL];
    double p = W[I_PRE];
    
    #ifndef NDEBUG
    assert(rho > 0.0 && "Non-positive density in prim_to_cons!");
    assert(p > 0.0 && "Non-positive pressure in prim_to_cons!");
    #endif
    
    U[I_RHO] = rho;
    U[I_MOM] = rho * u;
    U[I_ERG] = eos.total_energy(rho, u, p);
}

// =============================================================================
// Slope limiters for PLM reconstruction
// =============================================================================

enum class limiter_t {
    minmod,
    mc,        // Monotonized central
    none       // No limiting (linear)
};

inline auto minmod(double a, double b) -> double {
    if (a * b <= 0.0) return 0.0;
    return (std::abs(a) < std::abs(b)) ? a : b;
}

inline auto minmod3(double a, double b, double c) -> double {
    return minmod(a, minmod(b, c));
}

inline auto mc_limiter(double a, double b) -> double {
    // Monotonized central limiter
    double c = 0.5 * (a + b);
    return minmod3(2.0 * a, c, 2.0 * b);
}

inline auto apply_limiter(double left, double right, limiter_t lim) -> double {
    switch (lim) {
        case limiter_t::minmod:
            return minmod(left, right);
        case limiter_t::mc:
            return mc_limiter(left, right);
        case limiter_t::none:
            return 0.5 * (left + right);
    }
    return 0.0;
}

// =============================================================================
// PLM reconstruction
// =============================================================================

struct plm_state_t {
    double W_L[NPRIM];  // Left state
    double W_R[NPRIM];  // Right state
};

// PLM reconstruction: compute left/right states at cell interface
inline auto plm_reconstruct(
    const double* W_m,  // Cell i-1
    const double* W_0,  // Cell i
    const double* W_p,  // Cell i+1
    limiter_t lim = limiter_t::minmod
) -> plm_state_t {
    plm_state_t result;
    
    for (int k = 0; k < NPRIM; ++k) {
        // Compute slopes
        double dW_L = W_0[k] - W_m[k];  // Backward difference
        double dW_R = W_p[k] - W_0[k];  // Forward difference
        
        // Apply limiter
        double slope = apply_limiter(dW_L, dW_R, lim);
        
        // Extrapolate to cell faces
        result.W_L[k] = W_0[k] - 0.5 * slope;  // Left face (i-1/2)
        result.W_R[k] = W_0[k] + 0.5 * slope;  // Right face (i+1/2)
    }
    
    return result;
}

// =============================================================================
// HLLE Riemann solver
// =============================================================================

// Compute HLLE flux at interface
inline void hlle_flux(
    const double* W_L,  // Left state
    const double* W_R,  // Right state
    double* flux,       // Output flux
    const eos_t& eos
) {
    // Left state
    double rho_L = W_L[I_RHO];
    double u_L = W_L[I_VEL];
    double p_L = W_L[I_PRE];
    double c_L = eos.sound_speed(rho_L, p_L);
    double E_L = eos.total_energy(rho_L, u_L, p_L);
    
    // Right state
    double rho_R = W_R[I_RHO];
    double u_R = W_R[I_VEL];
    double p_R = W_R[I_PRE];
    double c_R = eos.sound_speed(rho_R, p_R);
    double E_R = eos.total_energy(rho_R, u_R, p_R);
    
    // Wave speeds (Davis estimate)
    double S_L = std::min(u_L - c_L, u_R - c_R);
    double S_R = std::max(u_L + c_L, u_R + c_R);
    
    // Left flux
    double F_L[NCONS];
    F_L[I_RHO] = rho_L * u_L;
    F_L[I_MOM] = rho_L * u_L * u_L + p_L;
    F_L[I_ERG] = (E_L + p_L) * u_L;
    
    // Right flux
    double F_R[NCONS];
    F_R[I_RHO] = rho_R * u_R;
    F_R[I_MOM] = rho_R * u_R * u_R + p_R;
    F_R[I_ERG] = (E_R + p_R) * u_R;
    
    // Conserved states
    double U_L[NCONS];
    U_L[I_RHO] = rho_L;
    U_L[I_MOM] = rho_L * u_L;
    U_L[I_ERG] = E_L;
    
    double U_R[NCONS];
    U_R[I_RHO] = rho_R;
    U_R[I_MOM] = rho_R * u_R;
    U_R[I_ERG] = E_R;
    
    // HLLE flux
    if (S_L >= 0.0) {
        // Supersonic to the right
        for (int k = 0; k < NCONS; ++k) {
            flux[k] = F_L[k];
        }
    } else if (S_R <= 0.0) {
        // Supersonic to the left
        for (int k = 0; k < NCONS; ++k) {
            flux[k] = F_R[k];
        }
    } else {
        // Subsonic: HLLE average
        for (int k = 0; k < NCONS; ++k) {
            flux[k] = (S_R * F_L[k] - S_L * F_R[k] + S_L * S_R * (U_R[k] - U_L[k])) / (S_R - S_L);
        }
    }
}

// =============================================================================
// RK2 (Heun's method) integrator
// =============================================================================

template<typename UpdateFn>
void rk2_step(
    field_t<double, 1>& U,        // Conserved variables (inout)
    field_t<double, 1>& U_tmp,    // Temporary storage
    const index_space_t<1>& interior,
    double dt,
    UpdateFn&& compute_rhs        // Function to compute dU/dt
) {
    // Stage 1: U_tmp = U + dt * L(U)
    compute_rhs(U, U_tmp, interior, dt);
    
    // Stage 2: U = 0.5 * (U + U_tmp + dt * L(U_tmp))
    field_t<double, 1> U_star(U.nvars(), space(U[0]));
    compute_rhs(U_tmp, U_star, interior, dt);
    
    // Average: U_new = 0.5 * (U + U_tmp)
    for (std::size_t var = 0; var < U.nvars(); ++var) {
        auto u_view = U[var];
        auto u_tmp_view = U_tmp[var];
        auto u_star_view = U_star[var];
        
        for (auto idx : interior) {
            u_view(idx) = 0.5 * (u_view(idx) + u_star_view(idx));
        }
    }
}

// =============================================================================
// Compute RHS for Euler equations (semi-discrete form)
// =============================================================================

struct euler_rhs_t {
    const eos_t& eos;
    double dx;
    limiter_t limiter;

    // Workspace: primitive variables over extended space (allocated once)
    mutable field_t<double, 1> W;
    
    euler_rhs_t(const eos_t& eos_, double dx_, limiter_t lim = limiter_t::minmod)
        : eos(eos_), dx(dx_), limiter(lim) {}

    void ensure_workspace(index_space_t<1> extended) const {
        if (W._data == nullptr || start(W._space)[0] != start(extended)[0] || shape(W._space)[0] != shape(extended)[0]) {
            W = field_t<double, 1>(NPRIM, extended, field_init_t::uninitialized);
        }
    }
    
    // Compute dU/dt = -1/dx * (F_{i+1/2} - F_{i-1/2})
    void operator()(
        const field_t<double, 1>& U_in,
        field_t<double, 1>& U_out,
        const index_space_t<1>& interior,
        double dt
    ) const {
        // Get extended space (includes halos)
        auto extended = space(U_in[0]);
        
        ensure_workspace(extended);
        
        for (auto idx : extended) {
            double U[NCONS];
            for (std::size_t k = 0; k < NCONS; ++k) {
                U[k] = U_in(k, idx);
            }
            
            double W_local[NPRIM];
            cons_to_prim(U, W_local, eos);
            
            for (std::size_t k = 0; k < NPRIM; ++k) {
                W(k, idx) = W_local[k];
            }
        }
        
        // Compute fluxes at interfaces
        for (auto idx : interior) {
            int i = idx[0];
            
            // Get primitive states for stencil
            double W_im1[NPRIM], W_i[NPRIM], W_ip1[NPRIM];
            for (std::size_t k = 0; k < NPRIM; ++k) {
                W_im1[k] = W(k, ivec_t<1>{i-1});
                W_i[k] = W(k, ivec_t<1>{i});
                W_ip1[k] = W(k, ivec_t<1>{i+1});
            }
            
            // Reconstruct at interface i-1/2
            // Need W_{i-2}, W_{i-1}, W_i for PLM
            double W_im2[NPRIM];
            if (i >= start(interior)[0] + 1) {
                for (std::size_t k = 0; k < NPRIM; ++k) {
                    W_im2[k] = W(k, ivec_t<1>{i-2});
                }
            } else {
                // Use one-sided at boundary
                for (std::size_t k = 0; k < NPRIM; ++k) {
                    W_im2[k] = W_im1[k];
                }
            }
            
            auto recon_im1 = plm_reconstruct(W_im2, W_im1, W_i, limiter);
            double W_L_im1[NPRIM];  // Right state of cell i-1
            double W_R_im1[NPRIM];  // Left state of cell i
            for (std::size_t k = 0; k < NPRIM; ++k) {
                W_L_im1[k] = recon_im1.W_R[k];  // From cell i-1
            }
            
            auto recon_i = plm_reconstruct(W_im1, W_i, W_ip1, limiter);
            for (std::size_t k = 0; k < NPRIM; ++k) {
                W_R_im1[k] = recon_i.W_L[k];  // From cell i
            }
            
            // Flux at i-1/2
            double flux_L[NCONS];
            hlle_flux(W_L_im1, W_R_im1, flux_L, eos);
            
            // Reconstruct at interface i+1/2
            double W_ip2[NPRIM];
            if (i < upper(interior)[0] - 1) {
                for (std::size_t k = 0; k < NPRIM; ++k) {
                    W_ip2[k] = W(k, ivec_t<1>{i+2});
                }
            } else {
                for (std::size_t k = 0; k < NPRIM; ++k) {
                    W_ip2[k] = W_ip1[k];
                }
            }
            
            double W_L_ip1[NPRIM];  // Right state of cell i
            double W_R_ip1[NPRIM];  // Left state of cell i+1
            for (std::size_t k = 0; k < NPRIM; ++k) {
                W_L_ip1[k] = recon_i.W_R[k];  // From cell i
            }
            
            auto recon_ip1 = plm_reconstruct(W_i, W_ip1, W_ip2, limiter);
            for (std::size_t k = 0; k < NPRIM; ++k) {
                W_R_ip1[k] = recon_ip1.W_L[k];  // From cell i+1
            }
            
            // Flux at i+1/2
            double flux_R[NCONS];
            hlle_flux(W_L_ip1, W_R_ip1, flux_R, eos);
            
            // Update: U_new = U_old - dt/dx * (F_R - F_L)
            for (std::size_t k = 0; k < NCONS; ++k) {
                U_out(k, idx) = U_in(k, idx) - (dt / dx) * (flux_R[k] - flux_L[k]);
            }
        }
    }
};

} // namespace nest::euler1d

