// Sod Shock Tube Test
// Classic 1D Riemann problem for validating Euler solver

#include "euler1d.hpp"
#include "nest/patch.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace nest;
using namespace nest::euler1d;

// =============================================================================
// Configuration
// =============================================================================

struct config_t {
    // Domain
    double x_min = 0.0;
    double x_max = 1.0;
    int num_zones = 400;
    int num_patches = 4;
    
    // Sod initial conditions
    double rho_L = 1.0;
    double u_L = 0.0;
    double p_L = 1.0;
    
    double rho_R = 0.125;
    double u_R = 0.0;
    double p_R = 0.1;
    
    double x_discontinuity = 0.5;
    
    // Time integration
    double t_final = 0.2;
    double cfl = 0.4;
    int output_cadence = 50;
    
    // Numerics
    double gamma = 1.4;
    limiter_t limiter = limiter_t::minmod;
};

// =============================================================================
// Output
// =============================================================================

void write_output(
    const std::vector<patch_t<double, 1>>& patches,
    const config_t& cfg,
    double time,
    int step
) {
    std::ostringstream filename;
    filename << "sod_" << std::setfill('0') << std::setw(4) << step << ".dat";
    
    std::ofstream out(filename.str());
    out << std::scientific << std::setprecision(10);
    out << "# Sod shock tube at t = " << time << "\n";
    out << "# x rho u p e\n";
    
    auto dx = (cfg.x_max - cfg.x_min) / cfg.num_zones;
    auto eos = eos_t{cfg.gamma};
    
    for (const auto& patch : patches) {
        for (auto idx : patch.interior) {
            // Get conserved variables
            double U[NCONS];
            for (std::size_t k = 0; k < NCONS; ++k) {
                U[k] = patch(k, idx);
            }
            
            // Convert to primitive
            double W[NPRIM];
            cons_to_prim(U, W, eos);
            
            double rho = W[I_RHO];
            double u = W[I_VEL];
            double p = W[I_PRE];
            double e = eos.specific_internal_energy(rho, p);
            
            double x = cfg.x_min + (idx[0] + 0.5) * dx;
            
            out << x << " " << rho << " " << u << " " << p << " " << e << "\n";
        }
    }
    
    std::cout << "Wrote " << filename.str() << " at t = " << time << "\n";
}

// =============================================================================
// CFL condition
// =============================================================================

double compute_dt(
    const std::vector<patch_t<double, 1>>& patches,
    double dx,
    double cfl,
    const eos_t& eos
) {
    double max_speed = 0.0;
    
    for (const auto& patch : patches) {
        for (auto idx : patch.interior) {
            // Get conserved variables
            double U[NCONS];
            for (std::size_t k = 0; k < NCONS; ++k) {
                U[k] = patch(k, idx);
            }
            
            // Convert to primitive
            double W[NPRIM];
            cons_to_prim(U, W, eos);
            
            double rho = W[I_RHO];
            double u = W[I_VEL];
            double p = W[I_PRE];
            double c = eos.sound_speed(rho, p);
            
            double speed = std::abs(u) + c;
            max_speed = std::max(max_speed, speed);
        }
    }
    
    return cfl * dx / max_speed;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    auto cfg = config_t{};
    auto eos = eos_t{cfg.gamma};
    
    // Domain setup
    auto dx = (cfg.x_max - cfg.x_min) / cfg.num_zones;
    auto domain = index_space(ivec(0), uvec(static_cast<unsigned int>(cfg.num_zones)));
    
    std::cout << "=== Sod Shock Tube Test ===\n";
    std::cout << "Domain: [" << cfg.x_min << ", " << cfg.x_max << "]\n";
    std::cout << "Zones: " << cfg.num_zones << ", dx = " << dx << "\n";
    std::cout << "Patches: " << cfg.num_patches << "\n";
    std::cout << "Gamma: " << cfg.gamma << "\n";
    std::cout << "CFL: " << cfg.cfl << "\n";
    std::cout << "Limiter: ";
    switch (cfg.limiter) {
        case limiter_t::minmod: std::cout << "minmod\n"; break;
        case limiter_t::mc: std::cout << "MC\n"; break;
        case limiter_t::none: std::cout << "none\n"; break;
    }
    std::cout << "\n";
    
    std::cout << "Initial conditions:\n";
    std::cout << "  Left  (x < " << cfg.x_discontinuity << "): "
              << "rho=" << cfg.rho_L << ", u=" << cfg.u_L << ", p=" << cfg.p_L << "\n";
    std::cout << "  Right (x > " << cfg.x_discontinuity << "): "
              << "rho=" << cfg.rho_R << ", u=" << cfg.u_R << ", p=" << cfg.p_R << "\n";
    std::cout << "\n";
    
    // Build mesh
    auto patches = build_periodic_mesh_1d<double>(domain, cfg.num_patches, NCONS, 2);  // halo=2 for PLM
    
    std::cout << "Built " << patches.size() << " patches\n\n";
    
    // Set initial conditions
    for (auto& patch : patches) {
        for (auto idx : patch.interior) {
            double x = cfg.x_min + (idx[0] + 0.5) * dx;
            
            double W[NPRIM];
            if (x < cfg.x_discontinuity) {
                W[I_RHO] = cfg.rho_L;
                W[I_VEL] = cfg.u_L;
                W[I_PRE] = cfg.p_L;
            } else {
                W[I_RHO] = cfg.rho_R;
                W[I_VEL] = cfg.u_R;
                W[I_PRE] = cfg.p_R;
            }
            
            // Convert to conserved
            double U[NCONS];
            prim_to_cons(W, U, eos);
            
            for (std::size_t k = 0; k < NCONS; ++k) {
                patch(k, idx) = U[k];
            }
        }
    }
    
    // Time integration
    double time = 0.0;
    int step = 0;
    int output_count = 0;
    
    write_output(patches, cfg, time, output_count++);
    
    auto rhs = euler_rhs_t(eos, dx, cfg.limiter);

    // Per-patch scratch (allocated once)
    struct patch_workspace_t {
        field_t<double, 1> U0;  // interior snapshot (U^n)
        field_t<double, 1> U1;  // stage 1 result (on extended, interior valid)
        field_t<double, 1> U2;  // stage 2 result (on extended, interior valid)
    };
    std::vector<patch_workspace_t> work;
    work.reserve(patches.size());
    for (const auto& patch : patches) {
        work.push_back(patch_workspace_t{
            .U0 = field_t<double, 1>(NCONS, patch.interior),
            .U1 = field_t<double, 1>(NCONS, patch.interior, field_init_t::uninitialized),
            .U2 = field_t<double, 1>(NCONS, patch.interior, field_init_t::uninitialized),
        });
    }
    
    while (time < cfg.t_final) {
        // Compute dt from CFL condition
        double dt = compute_dt(patches, dx, cfg.cfl, eos);
        dt = std::min(dt, cfg.t_final - time);
        
        if (step % cfg.output_cadence == 0) {
            std::cout << "Step " << step << ", t = " << time << ", dt = " << dt << "\n";
        }
        
        // Fill ghost zones
        execute_all_exchanges(patches);
        
        // Save U^n (interior) and compute stage 1: U1 = U^n + dt L(U^n)
        for (std::size_t p = 0; p < patches.size(); ++p) {
            auto& patch = patches[p];
            auto& wk = work[p];

            // Save U^n (interior only)
            for (std::size_t k = 0; k < NCONS; ++k) {
                for (auto idx : patch.interior) {
                    wk.U0(k, idx) = patch(k, idx);
                }
            }

            // Stage 1 update into scratch
            rhs(patch.state.conserved, wk.U1, patch.interior, dt);

            // Copy interior back into patch state (halos will be exchanged)
            for (std::size_t k = 0; k < NCONS; ++k) {
                for (auto idx : patch.interior) {
                    patch(k, idx) = wk.U1(k, idx);
                }
            }
        }

        // Exchange halos for stage 2
        execute_all_exchanges(patches);

        // Stage 2: U2 = U1 + dt L(U1), then SSPRK2 combine: U^{n+1} = 0.5 U^n + 0.5 U2
        for (std::size_t p = 0; p < patches.size(); ++p) {
            auto& patch = patches[p];
            auto& wk = work[p];

            rhs(patch.state.conserved, wk.U2, patch.interior, dt);

            for (std::size_t k = 0; k < NCONS; ++k) {
                for (auto idx : patch.interior) {
                    patch(k, idx) = 0.5 * wk.U0(k, idx) + 0.5 * wk.U2(k, idx);
                }
            }
        }
        
        time += dt;
        step++;
        
        if (step % cfg.output_cadence == 0) {
            write_output(patches, cfg, time, output_count++);
        }
        
        // Check positivity every 10 steps
        if (step % 10 == 0) {
            for (const auto& patch : patches) {
                for (auto idx : patch.interior) {
                    double rho = patch(I_RHO, idx);
                    double U[NCONS];
                    for (std::size_t k = 0; k < NCONS; ++k) {
                        U[k] = patch(k, idx);
                    }
                    double p = eos.pressure(U[I_RHO], U[I_MOM], U[I_ERG]);
                    
                    #ifndef NDEBUG
                    assert(rho > 0.0 && "Negative density detected!");
                    assert(p > 0.0 && "Negative pressure detected!");
                    #endif
                    
                    if (rho <= 0.0 || p <= 0.0) {
                        std::cerr << "ERROR: Positivity violation at step " << step << "\n";
                        std::cerr << "  Cell " << idx[0] << ": rho = " << rho << ", p = " << p << "\n";
                        return 1;
                    }
                }
            }
        }
    }
    
    // Final output
    write_output(patches, cfg, time, output_count);
    
    std::cout << "\n=== Results ===\n";
    std::cout << "Final time: " << time << "\n";
    std::cout << "Steps: " << step << "\n";
    std::cout << "Outputs written: " << output_count << "\n";
    std::cout << "Positivity check: PASSED\n";
    
    return 0;
}

