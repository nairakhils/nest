/**
 * GPU vs CPU comparison test for 1D Euler solver.
 * 
 * Runs the Sod shock tube problem on both CPU and GPU,
 * then compares the results within a tolerance.
 */

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>

// CUDA solver interface
#include "euler1d_solver.hpp"

// Include CPU solver
#include "../examples/sod1d/euler1d.hpp"

using namespace nest::euler1d;

// =============================================================================
// Test configuration
// =============================================================================

struct test_config_t {
    int n_zones = 100;
    int n_halo = 2;
    double x_min = 0.0;
    double x_max = 1.0;
    double gamma = 1.4;
    double t_final = 0.1;  // Shorter time for quick test
    double cfl = 0.3;
    
    // Sod initial conditions
    double rho_L = 1.0, u_L = 0.0, p_L = 1.0;
    double rho_R = 0.125, u_R = 0.0, p_R = 0.1;
    double x_disc = 0.5;
    
    // Comparison tolerance
    double tolerance = 1e-8;  // Allow small FP drift between CPU/GPU
};

// =============================================================================
// CPU solver (simplified single-patch version)
// =============================================================================

void run_cpu_sod(const test_config_t& cfg, std::vector<double>& U_cpu) {
    int n_total = cfg.n_zones + 2 * cfg.n_halo;
    double dx = (cfg.x_max - cfg.x_min) / cfg.n_zones;
    eos_t eos{cfg.gamma};
    
    // Initialize (SoA layout: [NCONS * n_total])
    U_cpu.resize(NCONS * n_total);
    std::fill(U_cpu.begin(), U_cpu.end(), 0.0);
    
    for (int i = 0; i < n_total; ++i) {
        double x = cfg.x_min + (i - cfg.n_halo + 0.5) * dx;
        
        double rho, u, p;
        if (x < cfg.x_disc) {
            rho = cfg.rho_L; u = cfg.u_L; p = cfg.p_L;
        } else {
            rho = cfg.rho_R; u = cfg.u_R; p = cfg.p_R;
        }
        
        double E = eos.total_energy(rho, u, p);
        
        U_cpu[I_RHO * n_total + i] = rho;
        U_cpu[I_MOM * n_total + i] = rho * u;
        U_cpu[I_ERG * n_total + i] = E;
    }
    
    // Time integration
    std::vector<double> U_tmp(NCONS * n_total);
    std::vector<double> U_star(NCONS * n_total);
    std::vector<double> W(NPRIM * n_total);
    
    double time = 0.0;
    
    while (time < cfg.t_final) {
        // Fill halos (periodic)
        for (int k = 0; k < NCONS; ++k) {
            for (int h = 0; h < cfg.n_halo; ++h) {
                // Left halo
                U_cpu[k * n_total + h] = U_cpu[k * n_total + (cfg.n_zones + h)];
                // Right halo
                U_cpu[k * n_total + (cfg.n_halo + cfg.n_zones + h)] = 
                    U_cpu[k * n_total + (cfg.n_halo + h)];
            }
        }
        
        // Compute dt
        double max_speed = 0.0;
        for (int i = cfg.n_halo; i < cfg.n_halo + cfg.n_zones; ++i) {
            double rho = U_cpu[I_RHO * n_total + i];
            double mom = U_cpu[I_MOM * n_total + i];
            double erg = U_cpu[I_ERG * n_total + i];
            double u = mom / rho;
            double p = eos.pressure(rho, mom, erg);
            double c = eos.sound_speed(rho, p);
            max_speed = std::max(max_speed, std::abs(u) + c);
        }
        double dt = cfg.cfl * dx / max_speed;
        dt = std::min(dt, cfg.t_final - time);
        
        // RK2 stage 1
        auto compute_rhs = [&](const std::vector<double>& U_in,
                               std::vector<double>& U_out, double dt_) {
            // Convert to primitive (reuse workspace)
            for (int i = 0; i < n_total; ++i) {
                double U[NCONS], W_local[NPRIM];
                for (int k = 0; k < NCONS; ++k) {
                    U[k] = U_in[k * n_total + i];
                }
                cons_to_prim(U, W_local, eos);
                for (int k = 0; k < NPRIM; ++k) {
                    W[k * n_total + i] = W_local[k];
                }
            }
            
            // Compute fluxes and update
            for (int i = cfg.n_halo; i < cfg.n_halo + cfg.n_zones; ++i) {
                double W_im2[NPRIM], W_im1[NPRIM], W_i[NPRIM], W_ip1[NPRIM], W_ip2[NPRIM];
                for (int k = 0; k < NPRIM; ++k) {
                    W_im2[k] = W[k * n_total + (i - 2)];
                    W_im1[k] = W[k * n_total + (i - 1)];
                    W_i[k]   = W[k * n_total + i];
                    W_ip1[k] = W[k * n_total + (i + 1)];
                    W_ip2[k] = W[k * n_total + (i + 2)];
                }
                
                // PLM reconstruction
                double W_L_imh[NPRIM], W_R_imh[NPRIM];
                double W_L_iph[NPRIM], W_R_iph[NPRIM];
                
                for (int k = 0; k < NPRIM; ++k) {
                    double slope_im1 = minmod(W_im1[k] - W_im2[k], W_i[k] - W_im1[k]);
                    double slope_i   = minmod(W_i[k] - W_im1[k], W_ip1[k] - W_i[k]);
                    double slope_ip1 = minmod(W_ip1[k] - W_i[k], W_ip2[k] - W_ip1[k]);
                    
                    W_L_imh[k] = W_im1[k] + 0.5 * slope_im1;
                    W_R_imh[k] = W_i[k]   - 0.5 * slope_i;
                    W_L_iph[k] = W_i[k]   + 0.5 * slope_i;
                    W_R_iph[k] = W_ip1[k] - 0.5 * slope_ip1;
                }
                
                double flux_L[NCONS], flux_R[NCONS];
                hlle_flux(W_L_imh, W_R_imh, flux_L, eos);
                hlle_flux(W_L_iph, W_R_iph, flux_R, eos);
                
                for (int k = 0; k < NCONS; ++k) {
                    U_out[k * n_total + i] = U_in[k * n_total + i] 
                        - (dt_ / dx) * (flux_R[k] - flux_L[k]);
                }
            }
        };
        
        // Stage 1
        compute_rhs(U_cpu, U_tmp, dt);
        
        // Fill halos on U_tmp
        for (int k = 0; k < NCONS; ++k) {
            for (int h = 0; h < cfg.n_halo; ++h) {
                U_tmp[k * n_total + h] = U_tmp[k * n_total + (cfg.n_zones + h)];
                U_tmp[k * n_total + (cfg.n_halo + cfg.n_zones + h)] = 
                    U_tmp[k * n_total + (cfg.n_halo + h)];
            }
        }
        
        // Stage 2
        compute_rhs(U_tmp, U_star, dt);
        
        // Average
        for (int k = 0; k < NCONS; ++k) {
            for (int i = cfg.n_halo; i < cfg.n_halo + cfg.n_zones; ++i) {
                U_cpu[k * n_total + i] = 0.5 * (U_cpu[k * n_total + i] + 
                                                 U_star[k * n_total + i]);
            }
        }
        
        time += dt;
    }
}

// =============================================================================
// GPU solver wrapper
// =============================================================================

void run_gpu_sod(const test_config_t& cfg, std::vector<double>& U_gpu) {
    int n_total = cfg.n_zones + 2 * cfg.n_halo;
    double dx = (cfg.x_max - cfg.x_min) / cfg.n_zones;
    
    // Initialize
    U_gpu.resize(NCONS * n_total);
    std::fill(U_gpu.begin(), U_gpu.end(), 0.0);
    
    eos_t eos{cfg.gamma};
    
    for (int i = 0; i < n_total; ++i) {
        double x = cfg.x_min + (i - cfg.n_halo + 0.5) * dx;
        
        double rho, u, p;
        if (x < cfg.x_disc) {
            rho = cfg.rho_L; u = cfg.u_L; p = cfg.p_L;
        } else {
            rho = cfg.rho_R; u = cfg.u_R; p = cfg.p_R;
        }
        
        double E = eos.total_energy(rho, u, p);
        
        U_gpu[I_RHO * n_total + i] = rho;
        U_gpu[I_MOM * n_total + i] = rho * u;
        U_gpu[I_ERG * n_total + i] = E;
    }
    
    // Create GPU solver
    euler1d_gpu_t* solver = euler1d_gpu_create(cfg.n_zones, cfg.n_halo, dx, cfg.gamma);
    
    // Upload
    euler1d_gpu_upload(solver, U_gpu.data());
    
    // Run
    int steps = euler1d_gpu_run(solver, cfg.t_final, cfg.cfl);
    std::cout << "GPU: " << steps << " steps\n";
    
    // Download
    euler1d_gpu_download(solver, U_gpu.data());
    
    // Cleanup
    euler1d_gpu_destroy(solver);
}

// =============================================================================
// Comparison
// =============================================================================

bool compare_results(const std::vector<double>& U_cpu, 
                     const std::vector<double>& U_gpu,
                     const test_config_t& cfg) {
    int n_total = cfg.n_zones + 2 * cfg.n_halo;
    
    double max_diff_rho = 0.0;
    double max_diff_mom = 0.0;
    double max_diff_erg = 0.0;
    
    for (int i = cfg.n_halo; i < cfg.n_halo + cfg.n_zones; ++i) {
        double diff_rho = std::abs(U_cpu[I_RHO * n_total + i] - U_gpu[I_RHO * n_total + i]);
        double diff_mom = std::abs(U_cpu[I_MOM * n_total + i] - U_gpu[I_MOM * n_total + i]);
        double diff_erg = std::abs(U_cpu[I_ERG * n_total + i] - U_gpu[I_ERG * n_total + i]);
        
        max_diff_rho = std::max(max_diff_rho, diff_rho);
        max_diff_mom = std::max(max_diff_mom, diff_mom);
        max_diff_erg = std::max(max_diff_erg, diff_erg);
    }
    
    std::cout << "Max difference (rho): " << max_diff_rho << "\n";
    std::cout << "Max difference (mom): " << max_diff_mom << "\n";
    std::cout << "Max difference (E):   " << max_diff_erg << "\n";
    
    // For numerical comparison, use relative tolerance
    double tol = cfg.tolerance;
    
    bool pass = (max_diff_rho < tol) && (max_diff_mom < tol) && (max_diff_erg < tol);
    
    if (pass) {
        std::cout << "PASS: GPU and CPU results match within tolerance " << tol << "\n";
    } else {
        std::cout << "FAIL: GPU and CPU results differ beyond tolerance " << tol << "\n";
    }
    
    return pass;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== GPU vs CPU Euler Solver Comparison ===\n\n";
    
    test_config_t cfg;
    cfg.n_zones = 100;
    cfg.t_final = 0.1;
    cfg.tolerance = 1e-8;
    
    std::cout << "Configuration:\n";
    std::cout << "  Zones: " << cfg.n_zones << "\n";
    std::cout << "  t_final: " << cfg.t_final << "\n";
    std::cout << "  CFL: " << cfg.cfl << "\n\n";
    
    std::vector<double> U_cpu, U_gpu;
    
    std::cout << "Running CPU solver...\n";
    run_cpu_sod(cfg, U_cpu);
    std::cout << "CPU done.\n\n";
    
    std::cout << "Running GPU solver...\n";
    run_gpu_sod(cfg, U_gpu);
    std::cout << "GPU done.\n\n";
    
    bool pass = compare_results(U_cpu, U_gpu, cfg);
    
    return pass ? 0 : 1;
}

