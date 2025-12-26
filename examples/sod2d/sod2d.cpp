// 2D Sod Shock Tube Test
// Discontinuity along x, uniform in y - should match 1D solution

#include "euler2d.hpp"
#include "nest/patch.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cmath>

using namespace nest;
using namespace nest::euler2d;

// =============================================================================
// Configuration
// =============================================================================

struct config_t {
    // Domain
    double x_min = 0.0;
    double x_max = 1.0;
    double y_min = 0.0;
    double y_max = 0.1;
    
    int nx_zones = 200;
    int ny_zones = 20;
    int nx_patches = 2;
    int ny_patches = 2;
    
    // Sod initial conditions (same as 1D)
    double rho_L = 1.0;
    double u_L = 0.0;
    double v_L = 0.0;
    double p_L = 1.0;
    
    double rho_R = 0.125;
    double u_R = 0.0;
    double v_R = 0.0;
    double p_R = 0.1;
    
    double x_discontinuity = 0.5;
    
    // Time integration
    double t_final = 0.2;
    double cfl = 0.3;
    int output_cadence = 50;
    
    // Numerics
    double gamma = 1.4;
    limiter_t limiter = limiter_t::minmod;
};

// =============================================================================
// Output
// =============================================================================

void write_output_2d(
    const std::vector<patch_t<double, 2>>& patches,
    const config_t& cfg,
    double time,
    int step
) {
    std::ostringstream filename;
    filename << "sod2d_" << std::setfill('0') << std::setw(4) << step << ".dat";
    
    std::ofstream out(filename.str());
    out << std::scientific << std::setprecision(10);
    out << "# 2D Sod shock tube at t = " << time << "\n";
    out << "# x y rho u v p\n";
    
    auto dx = (cfg.x_max - cfg.x_min) / cfg.nx_zones;
    auto dy = (cfg.y_max - cfg.y_min) / cfg.ny_zones;
    auto eos = eos_t{cfg.gamma};
    
    for (const auto& patch : patches) {
        for (auto idx : patch.interior) {
            double U[4];  // rho, rho*u, rho*v, E
            for (std::size_t k = 0; k < 4; ++k) {
                U[k] = patch(k, idx);
            }
            
            double rho = U[0];
            double u = U[1] / rho;
            double v = U[2] / rho;
            double ke = 0.5 * rho * (u*u + v*v);
            double p = (eos.gamma - 1.0) * (U[3] - ke);
            
            double x = cfg.x_min + (idx[0] + 0.5) * dx;
            double y = cfg.y_min + (idx[1] + 0.5) * dy;
            
            out << x << " " << y << " " << rho << " " << u << " " << v << " " << p << "\n";
        }
    }
    
    std::cout << "Wrote " << filename.str() << " at t = " << time << "\n";
}

// Extract x-slice at middle y
void write_xslice(
    const std::vector<patch_t<double, 2>>& patches,
    const config_t& cfg,
    double time,
    int step
) {
    std::ostringstream filename;
    filename << "sod2d_xslice_" << std::setfill('0') << std::setw(4) << step << ".dat";
    
    std::ofstream out(filename.str());
    out << std::scientific << std::setprecision(10);
    out << "# x-slice at y_mid, t = " << time << "\n";
    out << "# x rho u p\n";
    
    auto dx = (cfg.x_max - cfg.x_min) / cfg.nx_zones;
    auto dy = (cfg.y_max - cfg.y_min) / cfg.ny_zones;
    auto eos = eos_t{cfg.gamma};
    
    // Find middle j index
    int j_mid = cfg.ny_zones / 2;
    
    // Collect data at this y-slice from all patches
    std::vector<std::tuple<double, double, double, double>> slice_data;
    
    for (const auto& patch : patches) {
        for (auto idx : patch.interior) {
            if (idx[1] == j_mid) {
                double U[4];
                for (std::size_t k = 0; k < 4; ++k) {
                    U[k] = patch(k, idx);
                }
                
                double rho = U[0];
                double u = U[1] / rho;
                double v = U[2] / rho;
                double ke = 0.5 * rho * (u*u + v*v);
                double p = (eos.gamma - 1.0) * (U[3] - ke);
                
                double x = cfg.x_min + (idx[0] + 0.5) * dx;
                
                slice_data.push_back({x, rho, u, p});
            }
        }
    }
    
    // Sort by x
    std::sort(slice_data.begin(), slice_data.end(),
              [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
    
    for (const auto& [x, rho, u, p] : slice_data) {
        out << x << " " << rho << " " << u << " " << p << "\n";
    }
}

// =============================================================================
// Validation against 1D solution
// =============================================================================

void validate_against_1d(
    const std::vector<patch_t<double, 2>>& patches_2d,
    const config_t& cfg,
    double tolerance = 1e-6
) {
    std::cout << "\n=== Validating 2D vs 1D ===\n";
    
    auto dx = (cfg.x_max - cfg.x_min) / cfg.nx_zones;
    auto dy = (cfg.y_max - cfg.y_min) / cfg.ny_zones;
    auto eos = eos_t{cfg.gamma};
    
    // Check uniformity in y at several x positions
    std::vector<double> x_checks = {0.25, 0.5, 0.75};
    
    for (double x_check : x_checks) {
        int i_check = static_cast<int>((x_check - cfg.x_min) / dx);
        
        // Collect all values at this x across different y
        std::vector<double> rho_values, u_values, p_values;
        
        for (const auto& patch : patches_2d) {
            for (auto idx : patch.interior) {
                if (idx[0] == i_check) {
                    double U[4];
                    for (std::size_t k = 0; k < 4; ++k) {
                        U[k] = patch(k, idx);
                    }
                    
                    double rho = U[0];
                    double u = U[1] / rho;
                    double v = U[2] / rho;
                    double ke = 0.5 * rho * (u*u + v*v);
                    double p = (eos.gamma - 1.0) * (U[3] - ke);
                    
                    rho_values.push_back(rho);
                    u_values.push_back(u);
                    p_values.push_back(p);
                }
            }
        }
        
        if (rho_values.size() > 1) {
            // Compute variance
            double rho_mean = 0.0, u_mean = 0.0, p_mean = 0.0;
            for (std::size_t k = 0; k < rho_values.size(); ++k) {
                rho_mean += rho_values[k];
                u_mean += u_values[k];
                p_mean += p_values[k];
            }
            rho_mean /= rho_values.size();
            u_mean /= u_values.size();
            p_mean /= p_values.size();
            
            double rho_var = 0.0, u_var = 0.0, p_var = 0.0;
            for (std::size_t k = 0; k < rho_values.size(); ++k) {
                rho_var += (rho_values[k] - rho_mean) * (rho_values[k] - rho_mean);
                u_var += (u_values[k] - u_mean) * (u_values[k] - u_mean);
                p_var += (p_values[k] - p_mean) * (p_values[k] - p_mean);
            }
            rho_var = std::sqrt(rho_var / rho_values.size());
            u_var = std::sqrt(u_var / u_values.size());
            p_var = std::sqrt(p_var / p_values.size());
            
            std::cout << "  x = " << x_check << ": ";
            std::cout << "rho std = " << rho_var << ", ";
            std::cout << "u std = " << u_var << ", ";
            std::cout << "p std = " << p_var << "\n";
            
            if (rho_var > tolerance || u_var > tolerance || p_var > tolerance) {
                std::cout << "    WARNING: Solution not uniform in y!\n";
            } else {
                std::cout << "    PASS: Solution uniform in y\n";
            }
        }
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    auto cfg = config_t{};
    auto eos = eos_t{cfg.gamma};
    
    auto dx = (cfg.x_max - cfg.x_min) / cfg.nx_zones;
    auto dy = (cfg.y_max - cfg.y_min) / cfg.ny_zones;
    auto domain = index_space(ivec(0, 0), uvec(cfg.nx_zones, cfg.ny_zones));
    
    std::cout << "=== 2D Sod Shock Tube Test ===\n";
    std::cout << "Domain: [" << cfg.x_min << ", " << cfg.x_max << "] x ["
              << cfg.y_min << ", " << cfg.y_max << "]\n";
    std::cout << "Zones: " << cfg.nx_zones << " x " << cfg.ny_zones << "\n";
    std::cout << "dx = " << dx << ", dy = " << dy << "\n";
    std::cout << "Patches: " << cfg.nx_patches << " x " << cfg.ny_patches
              << " = " << (cfg.nx_patches * cfg.ny_patches) << " total\n";
    std::cout << "Gamma: " << cfg.gamma << ", CFL: " << cfg.cfl << "\n\n";
    
    std::cout << "Initial discontinuity at x = " << cfg.x_discontinuity << " (uniform in y)\n";
    std::cout << "  Left:  rho=" << cfg.rho_L << ", u=" << cfg.u_L << ", v=" << cfg.v_L << ", p=" << cfg.p_L << "\n";
    std::cout << "  Right: rho=" << cfg.rho_R << ", u=" << cfg.u_R << ", v=" << cfg.v_R << ", p=" << cfg.p_R << "\n\n";
    
    // Build 2D mesh (4 conserved vars: rho, rho*u, rho*v, E)
    auto patches = build_periodic_mesh_2d<double>(domain, cfg.nx_patches, cfg.ny_patches, 4, 2);
    
    std::cout << "Built " << patches.size() << " patches\n\n";
    
    // Set initial conditions
    for (auto& patch : patches) {
        for (auto idx : patch.interior) {
            double x = cfg.x_min + (idx[0] + 0.5) * dx;
            
            double rho, u, v, p;
            if (x < cfg.x_discontinuity) {
                rho = cfg.rho_L;
                u = cfg.u_L;
                v = cfg.v_L;
                p = cfg.p_L;
            } else {
                rho = cfg.rho_R;
                u = cfg.u_R;
                v = cfg.v_R;
                p = cfg.p_R;
            }
            
            double E = p / (eos.gamma - 1.0) + 0.5 * rho * (u*u + v*v);
            
            patch(0, idx) = rho;
            patch(1, idx) = rho * u;
            patch(2, idx) = rho * v;
            patch(3, idx) = E;
        }
    }
    
    // Time integration with directional splitting
    double time = 0.0;
    int step = 0;
    int output_count = 0;
    
    write_output_2d(patches, cfg, time, output_count);
    write_xslice(patches, cfg, time, output_count);
    output_count++;
    
    auto x_sweep = x_sweep_t(eos, dx, cfg.limiter);
    auto y_sweep = y_sweep_t(eos, dy, cfg.limiter);

    // Per-patch scratch (allocated once)
    struct patch_workspace_t {
        field_t<double, 2> Ux; // scratch for x sweep
        field_t<double, 2> Uy; // scratch for y sweep
    };
    std::vector<patch_workspace_t> work;
    work.reserve(patches.size());
    for (const auto& patch : patches) {
        work.push_back(patch_workspace_t{
            .Ux = field_t<double, 2>(4, patch.interior, field_init_t::uninitialized),
            .Uy = field_t<double, 2>(4, patch.interior, field_init_t::uninitialized),
        });
    }
    
    while (time < cfg.t_final) {
        double dt = compute_dt_2d(patches, dx, dy, cfg.cfl, eos);
        dt = std::min(dt, cfg.t_final - time);
        
        if (step % cfg.output_cadence == 0) {
            std::cout << "Step " << step << ", t = " << time << ", dt = " << dt << "\n";
        }
        
        // Directional splitting: X-sweep, then Y-sweep
        
        // X-sweep
        execute_all_exchanges(patches);
        for (std::size_t p = 0; p < patches.size(); ++p) {
            auto& patch = patches[p];
            auto& wk = work[p];
            x_sweep(patch.state.conserved, wk.Ux, patch.interior, dt);
            
            // Copy result (only interior - halos will be filled by next exchange)
            for (std::size_t k = 0; k < 4; ++k) {
                for (auto idx : patch.interior) {
                    patch(k, idx) = wk.Ux(k, idx);
                }
            }
        }
        
        // Y-sweep
        execute_all_exchanges(patches);
        for (std::size_t p = 0; p < patches.size(); ++p) {
            auto& patch = patches[p];
            auto& wk = work[p];
            y_sweep(patch.state.conserved, wk.Uy, patch.interior, dt);
            
            // Copy result (only interior - halos will be filled by next exchange)
            for (std::size_t k = 0; k < 4; ++k) {
                for (auto idx : patch.interior) {
                    patch(k, idx) = wk.Uy(k, idx);
                }
            }
        }
        
        time += dt;
        step++;
        
        if (step % cfg.output_cadence == 0) {
            write_output_2d(patches, cfg, time, output_count);
            write_xslice(patches, cfg, time, output_count);
            output_count++;
        }
    }
    
    // Final output
    write_output_2d(patches, cfg, time, output_count);
    write_xslice(patches, cfg, time, output_count);
    
    // Validate
    validate_against_1d(patches, cfg, 1e-8);
    
    std::cout << "\n=== Results ===\n";
    std::cout << "Final time: " << time << "\n";
    std::cout << "Steps: " << step << "\n";
    std::cout << "Outputs written: " << (output_count + 1) << "\n";
    
    return 0;
}

