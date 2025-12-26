/**
 * CUDA 1D Euler Solver
 * 
 * High-level interface with optimized fused kernels.
 * 
 * Memory traffic per RK2 step:
 *   Fused kernel: 2 stages * 144 bytes/cell = 288 bytes/cell
 *   Plus halos + averaging: ~50 bytes/cell
 *   Total: ~340 bytes/cell/step
 */

#include "euler1d_cuda.cuh"
#include <vector>
#include <stdexcept>
#include <cstring>
#include <chrono>

namespace nest::cuda {

// =============================================================================
// CUDA Error Checking
// =============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while (0)

// =============================================================================
// GPU Euler Solver
// =============================================================================

class euler1d_solver_t {
public:
    int n_interior;
    int n_halo;
    int n_total;
    double dx;
    double gamma;
    
    // Device memory (SoA layout)
    double* d_U;
    double* d_U_tmp;
    double* d_U_star;
    double* d_block_max;
    
    // Host buffer for reduction
    std::vector<double> h_block_max;
    int n_blocks;
    
    euler1d_solver_t(int n_interior_, int n_halo_, double dx_, double gamma_)
        : n_interior(n_interior_)
        , n_halo(n_halo_)
        , n_total(n_interior_ + 2 * n_halo_)
        , dx(dx_)
        , gamma(gamma_)
        , d_U(nullptr)
        , d_U_tmp(nullptr)
        , d_U_star(nullptr)
        , d_block_max(nullptr)
    {
        size_t total_bytes = euler1d::NCONS * n_total * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_U, total_bytes));
        CUDA_CHECK(cudaMalloc(&d_U_tmp, total_bytes));
        CUDA_CHECK(cudaMalloc(&d_U_star, total_bytes));
        
        n_blocks = (n_interior + euler1d::BLOCK_SIZE - 1) / euler1d::BLOCK_SIZE;
        CUDA_CHECK(cudaMalloc(&d_block_max, n_blocks * sizeof(double)));
        h_block_max.resize(n_blocks);
    }
    
    ~euler1d_solver_t() {
        if (d_U) cudaFree(d_U);
        if (d_U_tmp) cudaFree(d_U_tmp);
        if (d_U_star) cudaFree(d_U_star);
        if (d_block_max) cudaFree(d_block_max);
    }
    
    void upload(const double* h_data) {
        size_t bytes = euler1d::NCONS * n_total * sizeof(double);
        CUDA_CHECK(cudaMemcpy(d_U, h_data, bytes, cudaMemcpyHostToDevice));
    }
    
    void download(double* h_data) const {
        size_t bytes = euler1d::NCONS * n_total * sizeof(double);
        CUDA_CHECK(cudaMemcpy(h_data, d_U, bytes, cudaMemcpyDeviceToHost));
    }
    
    void fill_halos(double* d_data) {
        int threads = euler1d::BLOCK_SIZE;
        int blocks = (euler1d::NCONS * n_halo + threads - 1) / threads;
        euler1d::copy_halos_periodic_kernel<<<blocks, threads>>>(
            d_data, n_interior, n_halo, euler1d::NCONS
        );
        CUDA_CHECK(cudaGetLastError());
    }
    
    double compute_dt(double cfl) {
        int threads = euler1d::BLOCK_SIZE;
        int blocks = (n_interior + threads - 1) / threads;
        size_t shared_mem = threads * sizeof(double);
        
        euler1d::compute_max_wavespeed_kernel<<<blocks, threads, shared_mem>>>(
            d_U, d_block_max, n_interior, n_halo, gamma
        );
        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK(cudaMemcpy(h_block_max.data(), d_block_max, 
                              blocks * sizeof(double), cudaMemcpyDeviceToHost));
        
        double max_speed = 0.0;
        for (int b = 0; b < blocks; ++b) {
            max_speed = std::max(max_speed, h_block_max[b]);
        }
        
        return cfl * dx / max_speed;
    }
    
    // Fused Euler stage (recon + riemann + divergence + update in one kernel)
    void euler_stage(const double* d_in, double* d_out, double dt) {
        int threads = euler1d::BLOCK_SIZE;
        int blocks = (n_interior + threads - 1) / threads;
        
#ifdef NEST_CUDA_SHARED_MEM
        euler1d::euler1d_fused_stage_smem_kernel<<<blocks, threads>>>(
            d_in, d_out, n_interior, n_halo, dx, dt, gamma
        );
#else
        euler1d::euler1d_fused_stage_kernel<<<blocks, threads>>>(
            d_in, d_out, n_interior, n_halo, dx, dt, gamma
        );
#endif
        CUDA_CHECK(cudaGetLastError());
    }
    
    // RK2 step with fused stages
    void rk2_step(double dt) {
        // Stage 1: U_tmp = U + dt * L(U)
        fill_halos(d_U);
        euler_stage(d_U, d_U_tmp, dt);
        
        // Stage 2: U_star = U_tmp + dt * L(U_tmp)
        fill_halos(d_U_tmp);
        euler_stage(d_U_tmp, d_U_star, dt);
        
        // Average: U = 0.5 * (U + U_star)
        int threads = euler1d::BLOCK_SIZE;
        int total_elements = euler1d::NCONS * n_interior;
        int blocks = (total_elements + threads - 1) / threads;
        
        euler1d::rk2_average_kernel<<<blocks, threads>>>(
            d_U, d_U_star, n_interior, n_halo, euler1d::NCONS
        );
        CUDA_CHECK(cudaGetLastError());
    }
    
    int run(double t_final, double cfl) {
        double time = 0.0;
        int step = 0;
        
        while (time < t_final) {
            double dt = compute_dt(cfl);
            dt = std::min(dt, t_final - time);
            
            rk2_step(dt);
            
            time += dt;
            step++;
        }
        
        return step;
    }
    
    // Run with timing for benchmarks
    double run_timed(double t_final, double cfl, int& steps_out) {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        
        steps_out = run(t_final, cfl);
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    }
};

} // namespace nest::cuda

// =============================================================================
// C-style API
// =============================================================================

extern "C" {

struct euler1d_gpu_t;

euler1d_gpu_t* euler1d_gpu_create(int n_interior, int n_halo, double dx, double gamma) {
    return reinterpret_cast<euler1d_gpu_t*>(
        new nest::cuda::euler1d_solver_t(n_interior, n_halo, dx, gamma)
    );
}

void euler1d_gpu_destroy(euler1d_gpu_t* solver) {
    delete reinterpret_cast<nest::cuda::euler1d_solver_t*>(solver);
}

void euler1d_gpu_upload(euler1d_gpu_t* solver, const double* h_data) {
    reinterpret_cast<nest::cuda::euler1d_solver_t*>(solver)->upload(h_data);
}

void euler1d_gpu_download(euler1d_gpu_t* solver, double* h_data) {
    reinterpret_cast<nest::cuda::euler1d_solver_t*>(solver)->download(h_data);
}

int euler1d_gpu_run(euler1d_gpu_t* solver, double t_final, double cfl) {
    return reinterpret_cast<nest::cuda::euler1d_solver_t*>(solver)->run(t_final, cfl);
}

double euler1d_gpu_compute_dt(euler1d_gpu_t* solver, double cfl) {
    return reinterpret_cast<nest::cuda::euler1d_solver_t*>(solver)->compute_dt(cfl);
}

void euler1d_gpu_rk2_step(euler1d_gpu_t* solver, double dt) {
    reinterpret_cast<nest::cuda::euler1d_solver_t*>(solver)->rk2_step(dt);
}

double euler1d_gpu_run_timed(euler1d_gpu_t* solver, double t_final, double cfl, int* steps) {
    return reinterpret_cast<nest::cuda::euler1d_solver_t*>(solver)->run_timed(t_final, cfl, *steps);
}

} // extern "C"
