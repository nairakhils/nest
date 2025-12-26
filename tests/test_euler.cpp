// Unit tests for 1D Euler solver components

#include "../examples/sod1d/euler1d.hpp"
#include <iostream>
#include <sstream>
#include <cmath>

using namespace nest::euler1d;

// =============================================================================
// Test framework
// =============================================================================

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST(name) \
    void test_##name(); \
    struct test_runner_##name { \
        test_runner_##name() { \
            std::cout << "Running " #name "... "; \
            try { \
                test_##name(); \
                std::cout << "PASSED\n"; \
                ++g_tests_passed; \
            } catch (const std::exception& e) { \
                std::cout << "FAILED: " << e.what() << "\n"; \
            } \
            ++g_tests_run; \
        } \
    } test_instance_##name; \
    void test_##name()

#define ASSERT_NEAR(a, b, tol) \
    do { \
        if (std::abs((a) - (b)) > (tol)) { \
            std::ostringstream ss; \
            ss << "Expected " << (a) << " near " << (b) << " (tol=" << (tol) << ")"; \
            throw std::runtime_error(ss.str()); \
        } \
    } while (0)

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            throw std::runtime_error("Condition failed: " #cond); \
        } \
    } while (0)

// =============================================================================
// EOS tests
// =============================================================================

TEST(eos_pressure) {
    auto eos = eos_t{1.4};
    
    // Test case: rho=1, u=0, E=2.5 -> p = (gamma-1)*(E - 0.5*rho*u^2) = 0.4*2.5 = 1.0
    double rho = 1.0;
    double mom = 0.0;
    double erg = 2.5;
    
    double p = eos.pressure(rho, mom, erg);
    ASSERT_NEAR(p, 1.0, 1e-10);
}

TEST(eos_total_energy) {
    auto eos = eos_t{1.4};
    
    // Test case: rho=1, u=1, p=1 -> E = p/(gamma-1) + 0.5*rho*u^2 = 1/0.4 + 0.5 = 3.0
    double rho = 1.0;
    double u = 1.0;
    double p = 1.0;
    
    double E = eos.total_energy(rho, u, p);
    ASSERT_NEAR(E, 3.0, 1e-10);
}

TEST(eos_sound_speed) {
    auto eos = eos_t{1.4};
    
    // Test case: rho=1, p=1 -> c = sqrt(gamma*p/rho) = sqrt(1.4) ≈ 1.183
    double rho = 1.0;
    double p = 1.0;
    
    double c = eos.sound_speed(rho, p);
    ASSERT_NEAR(c, std::sqrt(1.4), 1e-10);
}

// =============================================================================
// Primitive <-> Conserved tests
// =============================================================================

TEST(cons_to_prim_roundtrip) {
    auto eos = eos_t{1.4};
    
    // Start with primitive state
    double W_in[NPRIM] = {1.0, 0.5, 1.0};  // rho, u, p
    
    // Convert to conserved
    double U[NCONS];
    prim_to_cons(W_in, U, eos);
    
    // Convert back to primitive
    double W_out[NPRIM];
    cons_to_prim(U, W_out, eos);
    
    // Should recover original state
    ASSERT_NEAR(W_out[I_RHO], W_in[I_RHO], 1e-10);
    ASSERT_NEAR(W_out[I_VEL], W_in[I_VEL], 1e-10);
    ASSERT_NEAR(W_out[I_PRE], W_in[I_PRE], 1e-10);
}

TEST(prim_to_cons_correctness) {
    auto eos = eos_t{1.4};
    
    double W[NPRIM] = {2.0, 1.0, 0.5};  // rho=2, u=1, p=0.5
    double U[NCONS];
    
    prim_to_cons(W, U, eos);
    
    // Check conserved variables
    ASSERT_NEAR(U[I_RHO], 2.0, 1e-10);        // rho
    ASSERT_NEAR(U[I_MOM], 2.0, 1e-10);        // rho*u = 2*1
    
    // E = p/(gamma-1) + 0.5*rho*u^2 = 0.5/0.4 + 0.5*2*1 = 1.25 + 1 = 2.25
    ASSERT_NEAR(U[I_ERG], 2.25, 1e-10);
}

// =============================================================================
// Limiter tests
// =============================================================================

TEST(minmod_limiter) {
    // Same sign: return smaller magnitude
    ASSERT_NEAR(minmod(1.0, 2.0), 1.0, 1e-10);
    ASSERT_NEAR(minmod(2.0, 1.0), 1.0, 1e-10);
    ASSERT_NEAR(minmod(-1.0, -2.0), -1.0, 1e-10);
    
    // Opposite signs: return 0
    ASSERT_NEAR(minmod(1.0, -1.0), 0.0, 1e-10);
    ASSERT_NEAR(minmod(-1.0, 1.0), 0.0, 1e-10);
    
    // Zero cases
    ASSERT_NEAR(minmod(0.0, 1.0), 0.0, 1e-10);
    ASSERT_NEAR(minmod(1.0, 0.0), 0.0, 1e-10);
}

TEST(plm_reconstruction_flat) {
    // Flat profile: slopes should be zero
    double W_m[NPRIM] = {1.0, 0.0, 1.0};
    double W_0[NPRIM] = {1.0, 0.0, 1.0};
    double W_p[NPRIM] = {1.0, 0.0, 1.0};
    
    auto result = plm_reconstruct(W_m, W_0, W_p, limiter_t::minmod);
    
    // Left and right should equal cell center
    for (int k = 0; k < NPRIM; ++k) {
        ASSERT_NEAR(result.W_L[k], W_0[k], 1e-10);
        ASSERT_NEAR(result.W_R[k], W_0[k], 1e-10);
    }
}

TEST(plm_reconstruction_monotone) {
    // Monotone increasing profile
    double W_m[NPRIM] = {1.0, 0.0, 1.0};
    double W_0[NPRIM] = {2.0, 0.0, 1.0};
    double W_p[NPRIM] = {3.0, 0.0, 1.0};
    
    auto result = plm_reconstruct(W_m, W_0, W_p, limiter_t::minmod);
    
    // Should reconstruct with slope=1
    ASSERT_NEAR(result.W_L[I_RHO], 1.5, 1e-10);  // 2.0 - 0.5*1
    ASSERT_NEAR(result.W_R[I_RHO], 2.5, 1e-10);  // 2.0 + 0.5*1
}

// =============================================================================
// HLLE flux tests
// =============================================================================

TEST(hlle_flux_stationary) {
    auto eos = eos_t{1.4};
    
    // Stationary contact: u=0 on both sides, same pressure
    double W_L[NPRIM] = {1.0, 0.0, 1.0};
    double W_R[NPRIM] = {1.0, 0.0, 1.0};
    
    double flux[NCONS];
    hlle_flux(W_L, W_R, flux, eos);
    
    // Flux should be zero (no flow)
    ASSERT_NEAR(flux[I_RHO], 0.0, 1e-10);
    ASSERT_NEAR(flux[I_MOM], 1.0, 1e-10);  // Pressure flux
    ASSERT_NEAR(flux[I_ERG], 0.0, 1e-10);
}

TEST(hlle_flux_supersonic) {
    auto eos = eos_t{1.4};
    
    // Supersonic flow to the right: u > c
    // rho=1, u=10, p=1 -> c ≈ 1.18, so u >> c
    double W_L[NPRIM] = {1.0, 10.0, 1.0};
    double W_R[NPRIM] = {1.0, 10.0, 1.0};
    
    double flux[NCONS];
    hlle_flux(W_L, W_R, flux, eos);
    
    // Should use left state flux
    ASSERT_NEAR(flux[I_RHO], 10.0, 1e-8);  // rho*u
    // Mom flux = rho*u^2 + p = 1*100 + 1 = 101
    ASSERT_NEAR(flux[I_MOM], 101.0, 1e-8);
}

// =============================================================================
// Positivity tests
// =============================================================================

TEST(positivity_sod_initial) {
    auto eos = eos_t{1.4};
    
    // Sod initial conditions should be positive
    double W_L[NPRIM] = {1.0, 0.0, 1.0};
    double W_R[NPRIM] = {0.125, 0.0, 0.1};
    
    double U_L[NCONS], U_R[NCONS];
    prim_to_cons(W_L, U_L, eos);
    prim_to_cons(W_R, U_R, eos);
    
    // Check positivity
    ASSERT_TRUE(U_L[I_RHO] > 0.0);
    ASSERT_TRUE(U_R[I_RHO] > 0.0);
    
    double p_L = eos.pressure(U_L[I_RHO], U_L[I_MOM], U_L[I_ERG]);
    double p_R = eos.pressure(U_R[I_RHO], U_R[I_MOM], U_R[I_ERG]);
    
    ASSERT_TRUE(p_L > 0.0);
    ASSERT_TRUE(p_R > 0.0);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n=== Euler Solver Tests ===\n\n";
    
    std::cout << "\n=== Summary ===\n";
    std::cout << g_tests_passed << "/" << g_tests_run << " tests passed\n";
    
    if (g_tests_passed != g_tests_run) {
        std::cout << "\nFAILED TESTS DETECTED\n";
        return 1;
    }
    
    std::cout << "\nAll tests passed successfully!\n";
    return 0;
}

