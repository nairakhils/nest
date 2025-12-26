/**
 * Benchmark: 1D Euler Solver Performance
 * 
 * Compares CPU (OpenMP) and GPU (CUDA) performance in cells/second.
 * 
 * Usage:
 *   ./benchmark_euler1d [n_zones] [t_final]
 *   ./benchmark_euler1d 10000 1.0
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

// CPU solver
#include "../sod1d/euler1d.hpp"
#include "nest/patch.hpp"

// GPU solver (if available)
#ifdef NEST_HAS_CUDA
#include "euler1d_solver.hpp"
#endif

using namespace nest;
using namespace nest::euler1d;

// =============================================================================
// Configuration
// =============================================================================

struct bench_config_t {
    int n_zones = 10000;
    int n_halo = 2;
    double x_min = 0.0;
    double x_max = 1.0;
    double gamma = 1.4;
    double t_final = 0.5;
    double cfl = 0.4;
    
    // Sod initial conditions
    double rho_L = 1.0, u_L = 0.0, p_L = 1.0;
    double rho_R = 0.125, u_R = 0.0, p_R = 0.1;
    double x_disc = 0.5;
};

// =============================================================================
// Initialize Sod problem
// =============================================================================

void initialize_sod(std::vector<double>& U, const bench_config_t& cfg) {
    int n_total = cfg.n_zones + 2 * cfg.n_halo;
    double dx = (cfg.x_max - cfg.x_min) / cfg.n_zones;
    eos_t eos{cfg.gamma};
    
    U.resize(NCONS * n_total);
    
    for (int i = 0; i < n_total; ++i) {
        double x = cfg.x_min + (i - cfg.n_halo + 0.5) * dx;
        
        double rho, u, p;
        if (x < cfg.x_disc) {
            rho = cfg.rho_L; u = cfg.u_L; p = cfg.p_L;
        } else {
            rho = cfg.rho_R; u = cfg.u_R; p = cfg.p_R;
        }
        
        double E = eos.total_energy(rho, u, p);
        
        // SoA layout
        U[I_RHO * n_total + i] = rho;
        U[I_MOM * n_total + i] = rho * u;
        U[I_ERG * n_total + i] = E;
    }
}

// =============================================================================
// CPU Benchmark
// =============================================================================

struct cpu_result_t {
    double time_ms;
    int steps;
    double cells_per_sec;
    double cell_updates_per_sec;  // Includes RK stages
};

cpu_result_t benchmark_cpu(const bench_config_t& cfg) {
    int n_total = cfg.n_zones + 2 * cfg.n_halo;
    double dx = (cfg.x_max - cfg.x_min) / cfg.n_zones;
    eos_t eos{cfg.gamma};
    
    std::vector<double> U, U_tmp, U_star;
    initialize_sod(U, cfg);
    U_tmp.resize(U.size());
    U_star.resize(U.size());
    std::vector<double> W(static_cast<std::size_t>(NPRIM) * static_cast<std::size_t>(n_total));
    
    // Fill halos helper
    auto fill_halos = [&](std::vector<double>& data) {
        for (int k = 0; k < NCONS; ++k) {
            for (int h = 0; h < cfg.n_halo; ++h) {
                data[k * n_total + h] = data[k * n_total + (cfg.n_zones + h)];
                data[k * n_total + (cfg.n_halo + cfg.n_zones + h)] = 
                    data[k * n_total + (cfg.n_halo + h)];
            }
        }
    };
    
    // Compute RHS helper
    auto compute_rhs = [&](const std::vector<double>& U_in, 
                           std::vector<double>& U_out, double dt) {
        // Convert to primitive
        for (int i = 0; i < n_total; ++i) {
            double U_local[NCONS], W_local[NPRIM];
            for (int k = 0; k < NCONS; ++k) {
                U_local[k] = U_in[k * n_total + i];
            }
            cons_to_prim(U_local, W_local, eos);
            for (int k = 0; k < NPRIM; ++k) {
                W[k * n_total + i] = W_local[k];
            }
        }
        
        // Update interior
        #ifdef NEST_HAS_OPENMP
        #pragma omp parallel for
        #endif
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
                    - (dt / dx) * (flux_R[k] - flux_L[k]);
            }
        }
    };
    
    // Compute dt
    auto compute_dt = [&]() {
        double max_speed = 0.0;
        for (int i = cfg.n_halo; i < cfg.n_halo + cfg.n_zones; ++i) {
            double rho = U[I_RHO * n_total + i];
            double mom = U[I_MOM * n_total + i];
            double erg = U[I_ERG * n_total + i];
            double u = mom / rho;
            double p = eos.pressure(rho, mom, erg);
            double c = eos.sound_speed(rho, p);
            max_speed = std::max(max_speed, std::abs(u) + c);
        }
        return cfg.cfl * dx / max_speed;
    };
    
    // Time the simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    double time = 0.0;
    int step = 0;
    
    while (time < cfg.t_final) {
        double dt = compute_dt();
        dt = std::min(dt, cfg.t_final - time);
        
        // RK2 stage 1
        fill_halos(U);
        compute_rhs(U, U_tmp, dt);
        
        // RK2 stage 2
        fill_halos(U_tmp);
        compute_rhs(U_tmp, U_star, dt);
        
        // Average
        for (int k = 0; k < NCONS; ++k) {
            for (int i = cfg.n_halo; i < cfg.n_halo + cfg.n_zones; ++i) {
                U[k * n_total + i] = 0.5 * (U[k * n_total + i] + U_star[k * n_total + i]);
            }
        }
        
        time += dt;
        step++;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    cpu_result_t result;
    result.time_ms = elapsed.count();
    result.steps = step;
    result.cells_per_sec = (double)cfg.n_zones * step / (result.time_ms / 1000.0);
    result.cell_updates_per_sec = result.cells_per_sec * 2.0;  // 2 RK stages
    
    return result;
}

// =============================================================================
// GPU Benchmark
// =============================================================================

#ifdef NEST_HAS_CUDA
struct gpu_result_t {
    double time_ms;
    int steps;
    double cells_per_sec;
    double cell_updates_per_sec;
};

gpu_result_t benchmark_gpu(const bench_config_t& cfg) {
    double dx = (cfg.x_max - cfg.x_min) / cfg.n_zones;
    
    std::vector<double> U;
    initialize_sod(U, cfg);
    
    euler1d_gpu_t* solver = euler1d_gpu_create(cfg.n_zones, cfg.n_halo, dx, cfg.gamma);
    euler1d_gpu_upload(solver, U.data());
    
    int steps;
    double time_ms = euler1d_gpu_run_timed(solver, cfg.t_final, cfg.cfl, &steps);
    
    euler1d_gpu_download(solver, U.data());
    euler1d_gpu_destroy(solver);
    
    gpu_result_t result;
    result.time_ms = time_ms;
    result.steps = steps;
    result.cells_per_sec = (double)cfg.n_zones * steps / (time_ms / 1000.0);
    result.cell_updates_per_sec = result.cells_per_sec * 2.0;
    
    return result;
}
#endif

// =============================================================================
// Main
// =============================================================================

void run_scaling_study(const bench_config_t& base_cfg) {
    std::cout << "\n========================================\n";
    std::cout << "  Scaling Study\n";
    std::cout << "========================================\n\n";
    
    std::vector<int> sizes = {1000, 10000, 100000, 1000000};
    
    std::cout << std::setw(12) << "Zones" 
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Steps"
              << std::setw(15) << "Cells/sec" << "\n";
    std::cout << std::string(57, '-') << "\n";
    
    for (int n : sizes) {
        bench_config_t cfg = base_cfg;
        cfg.n_zones = n;
        cfg.t_final = 0.1;  // Short run for scaling
        
        auto result = benchmark_cpu(cfg);
        
        std::cout << std::setw(12) << n
                  << std::setw(15) << std::fixed << std::setprecision(1) << result.time_ms
                  << std::setw(15) << result.steps
                  << std::setw(15) << std::scientific << std::setprecision(2) << result.cells_per_sec
                  << "\n";
    }
}

int main(int argc, char* argv[]) {
    bench_config_t cfg;
    
    bool run_scaling = false;
    
    if (argc > 1) {
        if (std::string(argv[1]) == "--scaling") {
            run_scaling = true;
        } else {
            cfg.n_zones = std::atoi(argv[1]);
        }
    }
    if (argc > 2) {
        cfg.t_final = std::atof(argv[2]);
    }
    
    std::cout << "========================================\n";
    std::cout << "  1D Euler Solver Benchmark\n";
    std::cout << "========================================\n\n";
    
    std::cout << "Configuration:\n";
    std::cout << "  Zones:    " << cfg.n_zones << "\n";
    std::cout << "  t_final:  " << cfg.t_final << "\n";
    std::cout << "  CFL:      " << cfg.cfl << "\n";
    std::cout << "  Gamma:    " << cfg.gamma << "\n\n";
    
    // CPU benchmark
    std::cout << "CPU Benchmark";
#ifdef NEST_HAS_OPENMP
    std::cout << " (OpenMP enabled)";
#endif
    std::cout << "...\n";
    
    auto cpu = benchmark_cpu(cfg);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Time:         " << cpu.time_ms << " ms\n";
    std::cout << "  Steps:        " << cpu.steps << "\n";
    std::cout << "  Cells/sec:    " << std::scientific << cpu.cells_per_sec << "\n";
    std::cout << "  Updates/sec:  " << cpu.cell_updates_per_sec << " (x2 for RK2)\n\n";
    
#ifdef NEST_HAS_CUDA
    // GPU benchmark
    std::cout << "GPU Benchmark (CUDA)...\n";
    
    try {
        auto gpu = benchmark_gpu(cfg);
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Time:         " << gpu.time_ms << " ms\n";
        std::cout << "  Steps:        " << gpu.steps << "\n";
        std::cout << "  Cells/sec:    " << std::scientific << gpu.cells_per_sec << "\n";
        std::cout << "  Updates/sec:  " << gpu.cell_updates_per_sec << " (x2 for RK2)\n\n";
        
        // Speedup
        double speedup = cpu.time_ms / gpu.time_ms;
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "Speedup (GPU/CPU): " << speedup << "x\n";
    } catch (const std::exception& e) {
        std::cout << "  Error: " << e.what() << "\n";
        std::cout << "  (No CUDA-capable GPU available)\n";
    }
#else
    std::cout << "GPU Benchmark: CUDA not enabled\n";
    std::cout << "  Build with: cmake --preset cuda-release\n";
#endif
    
    std::cout << "\n========================================\n";
    
    // Memory bandwidth analysis
    std::cout << "\nMemory Traffic Analysis (per RK step):\n";
    double bytes_per_cell = 144.0 * 2;  // 2 stages, 144 bytes each
    double total_bytes = bytes_per_cell * cfg.n_zones * cpu.steps;
    std::cout << "  Bytes/cell/step:  " << (int)bytes_per_cell << "\n";
    std::cout << "  Total traffic:    " << std::fixed << std::setprecision(1) 
              << (total_bytes / 1e9) << " GB\n";
    
    double bandwidth = total_bytes / (cpu.time_ms / 1000.0) / 1e9;
    std::cout << "  CPU bandwidth:    " << std::fixed << std::setprecision(1) 
              << bandwidth << " GB/s\n";
    
    // Optional scaling study
    if (run_scaling) {
        run_scaling_study(cfg);
    }
    
    return 0;
}

