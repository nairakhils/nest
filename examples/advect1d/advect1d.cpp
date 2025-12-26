// 1D Linear Advection Example
// Demonstrates SoA layout, patch decomposition, and halo exchange

#include "nest/core.hpp"
#include "nest/patch.hpp"
#include "nest/pipeline.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace nest;

// =============================================================================
// Problem configuration
// =============================================================================

struct config_t {
    double domain_length = 1.0;
    int num_zones = 200;
    int num_patches = 4;
    double wavespeed = 1.0;
    double cfl = 0.4;
    double t_final = 1.0;          // One full period
    int output_cadence = 10;       // Steps between outputs
};

// =============================================================================
// Pipeline stages
// =============================================================================

// Initial condition: sine wave
struct initial_condition_t {
    double dx;
    double L;
    
    auto value(patch_t<double, 1> p) const -> patch_t<double, 1> {
        for (auto idx : p.interior) {
            double x = (idx[0] + 0.5) * dx;
            p(0, idx) = std::sin(2.0 * M_PI * x / L);
        }
        return p;
    }
};

// Upwind flux + update (fused)
struct flux_update_t {
    double wavespeed;
    double dt;
    double dx;
    
    auto value(patch_t<double, 1> p) const -> patch_t<double, 1> {
        auto v = wavespeed;
        auto dtdx = dt / dx;
        auto i0 = start(p.interior)[0];
        auto i1 = upper(p.interior)[0];
        
        // Access the conserved variable directly
        auto u = p[0];  // md_view of variable 0
        
        if (v > 0.0) {
            // Upwind from left: iterate backwards for stability
            for (int i = i1 - 1; i >= i0; --i) {
                auto idx = ivec_t<1>{i};
                auto idx_m = ivec_t<1>{i - 1};
                double flux_l = v * u(idx_m);
                double flux_r = v * u(idx);
                u(idx) = u(idx) - dtdx * (flux_r - flux_l);
            }
        } else {
            // Upwind from right: iterate forwards for stability
            for (int i = i0; i < i1; ++i) {
                auto idx = ivec_t<1>{i};
                auto idx_p = ivec_t<1>{i + 1};
                double flux_l = v * u(idx);
                double flux_r = v * u(idx_p);
                u(idx) = u(idx) - dtdx * (flux_r - flux_l);
            }
        }
        return p;
    }
};

// =============================================================================
// Output
// =============================================================================

void write_output(const std::vector<patch_t<double, 1>>& patches, double dx, double time, int step) {
    std::ostringstream filename;
    filename << "advect1d_" << std::setfill('0') << std::setw(4) << step << ".dat";
    
    std::ofstream out(filename.str());
    out << "# time = " << time << "\n";
    out << "# x u\n";
    
    for (const auto& patch : patches) {
        for (auto idx : patch.interior) {
            double x = (idx[0] + 0.5) * dx;
            out << x << " " << patch(0, idx) << "\n";
        }
    }
    std::cout << "Wrote " << filename.str() << "\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    auto cfg = config_t{};
    
    // Domain setup
    auto dx = cfg.domain_length / cfg.num_zones;
    auto domain = index_space(ivec(0), uvec(static_cast<unsigned int>(cfg.num_zones)));
    
    std::cout << "=== 1D Linear Advection ===\n";
    std::cout << "Zones: " << cfg.num_zones << ", Patches: " << cfg.num_patches << "\n";
    std::cout << "Wavespeed: " << cfg.wavespeed << ", CFL: " << cfg.cfl << "\n";
    std::cout << "Domain: [0, " << cfg.domain_length << "], dx = " << dx << "\n\n";
    
    // Build mesh with precomputed exchange plans
    constexpr std::size_t nvars = 1;  // Just density/concentration
    auto patches = build_periodic_mesh_1d<double>(domain, cfg.num_patches, nvars, 1);
    
    std::cout << "Built " << patches.size() << " patches\n";
    for (const auto& p : patches) {
        std::cout << "  Patch " << p.index << ": interior = ["
                  << start(p.interior)[0] << ", " << upper(p.interior)[0] << "), "
                  << "extended = [" << start(p.extended)[0] << ", " << upper(p.extended)[0] << "), "
                  << "exchange regions = " << p.exchange_plan.num_regions() << "\n";
    }
    std::cout << "\n";
    
    // Apply initial condition
    auto init_stage = initial_condition_t{dx, cfg.domain_length};
    for (auto& p : patches) {
        p = init_stage.value(std::move(p));
    }
    
    // Compute dt
    double dt = cfg.cfl * dx / std::abs(cfg.wavespeed);
    std::cout << "dt = " << dt << "\n\n";
    
    // Time stepping
    double time = 0.0;
    int step = 0;
    int output_count = 0;
    
    write_output(patches, dx, time, output_count++);
    
    while (time < cfg.t_final) {
        // Ensure we don't overshoot t_final
        double dt_step = std::min(dt, cfg.t_final - time);
        
        // 1. Fill ghost zones using precomputed exchange plans
        execute_all_exchanges(patches);
        
        // 2. Flux + update
        auto update_stage = flux_update_t{cfg.wavespeed, dt_step, dx};
        for (auto& p : patches) {
            p = update_stage.value(std::move(p));
        }
        
        time += dt_step;
        step++;
        
        if (step % cfg.output_cadence == 0) {
            write_output(patches, dx, time, output_count++);
        }
    }
    
    // Final output
    write_output(patches, dx, time, output_count);
    
    // Compute L2 error (should be zero for one full period with exact arithmetic)
    double l2_error = 0.0;
    for (const auto& patch : patches) {
        for (auto idx : patch.interior) {
            double x = (idx[0] + 0.5) * dx;
            double exact = std::sin(2.0 * M_PI * x / cfg.domain_length);
            double err = patch(0, idx) - exact;
            l2_error += err * err;
        }
    }
    l2_error = std::sqrt(l2_error * dx);
    
    std::cout << "\n=== Results ===\n";
    std::cout << "Final time: " << time << "\n";
    std::cout << "Steps: " << step << "\n";
    std::cout << "L2 error: " << l2_error << "\n";
    
    return 0;
}
