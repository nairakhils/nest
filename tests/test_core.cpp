// Comprehensive core library unit tests
// Focus: indexing correctness, halo exchange, SoA contiguity

#include "nest/core.hpp"
#include "nest/patch.hpp"
#include "nest/pipeline.hpp"
#include <cmath>
#include <iostream>
#include <sstream>
#include <cstring>

// =============================================================================
// Minimal test framework
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

#define ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            std::ostringstream ss; \
            ss << "Expected " << (a) << " == " << (b); \
            throw std::runtime_error(ss.str()); \
        } \
    } while (0)

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
// vec_t tests
// =============================================================================

TEST(vec_construction) {
    auto v = nest::ivec(1, 2, 3);
    ASSERT_EQ(v[0], 1);
    ASSERT_EQ(v[1], 2);
    ASSERT_EQ(v[2], 3);
    ASSERT_EQ(v.size(), 3u);
}

TEST(vec_arithmetic) {
    auto a = nest::ivec(1, 2);
    auto b = nest::ivec(3, 4);
    auto sum = a + b;
    ASSERT_EQ(sum[0], 4);
    ASSERT_EQ(sum[1], 6);
    
    auto diff = b - a;
    ASSERT_EQ(diff[0], 2);
    ASSERT_EQ(diff[1], 2);
    
    auto scaled = a * 3;
    ASSERT_EQ(scaled[0], 3);
    ASSERT_EQ(scaled[1], 6);
}

// =============================================================================
// index_space_t tests
// =============================================================================

TEST(index_space_iteration_order) {
    // Verify row-major iteration order
    auto s = nest::index_space(nest::ivec(0, 0), nest::uvec(2u, 3u));
    std::vector<nest::ivec_t<2>> indices;
    for (auto idx : s) {
        indices.push_back(idx);
    }
    // Row-major: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    ASSERT_EQ(indices.size(), 6u);
    ASSERT_TRUE(indices[0] == nest::ivec(0, 0));
    ASSERT_TRUE(indices[1] == nest::ivec(0, 1));
    ASSERT_TRUE(indices[2] == nest::ivec(0, 2));
    ASSERT_TRUE(indices[3] == nest::ivec(1, 0));
    ASSERT_TRUE(indices[4] == nest::ivec(1, 1));
    ASSERT_TRUE(indices[5] == nest::ivec(1, 2));
}

TEST(index_space_subspace_correctness) {
    // Test that subspaces partition the domain correctly
    auto s = nest::index_space(nest::ivec(0), nest::uvec(100u));
    
    auto p0 = nest::subspace(s, 4, 0, 0);
    auto p1 = nest::subspace(s, 4, 1, 0);
    auto p2 = nest::subspace(s, 4, 2, 0);
    auto p3 = nest::subspace(s, 4, 3, 0);
    
    // Check no gaps
    ASSERT_EQ(nest::upper(p0)[0], nest::start(p1)[0]);
    ASSERT_EQ(nest::upper(p1)[0], nest::start(p2)[0]);
    ASSERT_EQ(nest::upper(p2)[0], nest::start(p3)[0]);
    
    // Check covers entire domain
    ASSERT_EQ(nest::start(p0)[0], 0);
    ASSERT_EQ(nest::upper(p3)[0], 100);
    
    // Check total size
    auto total = nest::size(p0) + nest::size(p1) + nest::size(p2) + nest::size(p3);
    ASSERT_EQ(total, 100u);
}

// =============================================================================
// Linear indexing tests (CRITICAL for correctness)
// =============================================================================

TEST(linear_index_1d_correctness) {
    auto s = nest::index_space(nest::ivec(0), nest::uvec(10u));
    
    // Verify each index maps to expected offset
    for (int i = 0; i < 10; ++i) {
        ASSERT_EQ(nest::linear_index(s, nest::ivec(i)), static_cast<std::size_t>(i));
    }
}

TEST(linear_index_1d_with_offset) {
    auto s = nest::index_space(nest::ivec(5), nest::uvec(10u));
    
    // First element should map to offset 0
    ASSERT_EQ(nest::linear_index(s, nest::ivec(5)), 0u);
    ASSERT_EQ(nest::linear_index(s, nest::ivec(10)), 5u);
    ASSERT_EQ(nest::linear_index(s, nest::ivec(14)), 9u);
}

TEST(linear_index_2d_row_major) {
    // 2x3 grid: 2 rows, 3 columns
    auto s = nest::index_space(nest::ivec(0, 0), nest::uvec(2u, 3u));
    
    // Row-major layout: [i][j] -> i * ncols + j
    ASSERT_EQ(nest::linear_index(s, nest::ivec(0, 0)), 0u);  // [0][0]
    ASSERT_EQ(nest::linear_index(s, nest::ivec(0, 1)), 1u);  // [0][1]
    ASSERT_EQ(nest::linear_index(s, nest::ivec(0, 2)), 2u);  // [0][2]
    ASSERT_EQ(nest::linear_index(s, nest::ivec(1, 0)), 3u);  // [1][0]
    ASSERT_EQ(nest::linear_index(s, nest::ivec(1, 1)), 4u);  // [1][1]
    ASSERT_EQ(nest::linear_index(s, nest::ivec(1, 2)), 5u);  // [1][2]
}

TEST(soa_offset_correctness) {
    auto s = nest::index_space(nest::ivec(0), nest::uvec(100u));
    
    // Variable 0, index 0: offset should be 0
    ASSERT_EQ(nest::soa_offset(s, 0, nest::ivec(0)), 0u);
    
    // Variable 0, index 50: offset should be 50
    ASSERT_EQ(nest::soa_offset(s, 0, nest::ivec(50)), 50u);
    
    // Variable 1, index 0: offset should be 100 (stride)
    ASSERT_EQ(nest::soa_offset(s, 1, nest::ivec(0)), 100u);
    
    // Variable 1, index 50: offset should be 150
    ASSERT_EQ(nest::soa_offset(s, 1, nest::ivec(50)), 150u);
    
    // Variable 3, index 25: offset should be 3*100 + 25 = 325
    ASSERT_EQ(nest::soa_offset(s, 3, nest::ivec(25)), 325u);
}

// =============================================================================
// md_view tests
// =============================================================================

TEST(md_view_construction_and_access) {
    auto s = nest::index_space(nest::ivec(0), nest::uvec(10u));
    double data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    
    auto view = nest::md_view_t<double, 1>(data, s);
    
    for (int i = 0; i < 10; ++i) {
        ASSERT_NEAR(view(nest::ivec(i)), static_cast<double>(i), 1e-10);
    }
}

TEST(md_view_with_offset) {
    auto s = nest::index_space(nest::ivec(5), nest::uvec(5u));
    double data[5] = {10, 20, 30, 40, 50};
    
    auto view = nest::md_view_t<double, 1>(data, s);
    
    ASSERT_NEAR(view(nest::ivec(5)), 10.0, 1e-10);
    ASSERT_NEAR(view(nest::ivec(9)), 50.0, 1e-10);
}

TEST(md_view_2d_strides) {
    // 2x3 array
    auto s = nest::index_space(nest::ivec(0, 0), nest::uvec(2u, 3u));
    double data[6] = {1, 2, 3,    // row 0
                      4, 5, 6};   // row 1
    
    auto view = nest::md_view_t<double, 2>(data, s);
    
    // Verify strides are correct (row-major)
    ASSERT_EQ(view._strides[0], 3u);  // Row stride
    ASSERT_EQ(view._strides[1], 1u);  // Column stride
    
    // Verify access
    ASSERT_NEAR(view(nest::ivec(0, 0)), 1.0, 1e-10);
    ASSERT_NEAR(view(nest::ivec(0, 2)), 3.0, 1e-10);
    ASSERT_NEAR(view(nest::ivec(1, 0)), 4.0, 1e-10);
    ASSERT_NEAR(view(nest::ivec(1, 2)), 6.0, 1e-10);
}

TEST(md_view_contiguity_check) {
    auto s = nest::index_space(nest::ivec(0, 0), nest::uvec(2u, 3u));
    double data[6];
    
    auto view = nest::md_view_t<double, 2>(data, s);
    ASSERT_TRUE(nest::is_contiguous(view));
    
    // Create view with non-standard strides
    auto strides = nest::uvec(10u, 1u);  // Large row stride
    auto noncontig = nest::md_view_t<double, 2>(data, s, strides);
    ASSERT_TRUE(!nest::is_contiguous(noncontig));
}

TEST(md_view_subview_offsets_1d) {
    auto s = nest::index_space(nest::ivec(0), nest::uvec(10u));
    double data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto view = nest::md_view_t<double, 1>(data, s);
    auto sub = view.subview(nest::index_space(nest::ivec(5), nest::uvec(3u))); // [5,8)
    ASSERT_NEAR(sub(nest::ivec(5)), 5.0, 1e-10);
    ASSERT_NEAR(sub(nest::ivec(6)), 6.0, 1e-10);
    ASSERT_NEAR(sub(nest::ivec(7)), 7.0, 1e-10);
}

TEST(md_view_subview_offsets_2d) {
    // 4x5 array, row-major values = i*5 + j
    auto s = nest::index_space(nest::ivec(0, 0), nest::uvec(4u, 5u));
    double data[20];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 5; ++j) {
            data[i * 5 + j] = static_cast<double>(i * 5 + j);
        }
    }
    auto view = nest::md_view_t<double, 2>(data, s);
    auto subspace = nest::index_space(nest::ivec(1, 2), nest::uvec(2u, 2u)); // rows 1..2, cols 2..3
    auto sub = view.subview(subspace);
    ASSERT_NEAR(sub(nest::ivec(1, 2)), 7.0, 1e-10);  // 1*5+2
    ASSERT_NEAR(sub(nest::ivec(1, 3)), 8.0, 1e-10);
    ASSERT_NEAR(sub(nest::ivec(2, 2)), 12.0, 1e-10); // 2*5+2
    ASSERT_NEAR(sub(nest::ivec(2, 3)), 13.0, 1e-10);
}

// =============================================================================
// field_t SoA tests (CRITICAL)
// =============================================================================

TEST(field_soa_layout) {
    auto s = nest::index_space(nest::ivec(0), nest::uvec(10u));
    auto field = nest::field_t<double, 1>(3, s);  // 3 variables
    
    // Set values
    for (int i = 0; i < 10; ++i) {
        field(0, nest::ivec(i)) = static_cast<double>(i);
        field(1, nest::ivec(i)) = static_cast<double>(i + 100);
        field(2, nest::ivec(i)) = static_cast<double>(i + 200);
    }
    
    // Verify contiguous layout: [var0...][var1...][var2...]
    auto* raw = field.var_data(0);
    for (int i = 0; i < 10; ++i) {
        ASSERT_NEAR(raw[i], static_cast<double>(i), 1e-10);
        ASSERT_NEAR(raw[10 + i], static_cast<double>(i + 100), 1e-10);
        ASSERT_NEAR(raw[20 + i], static_cast<double>(i + 200), 1e-10);
    }
}

TEST(field_soa_contiguity_per_variable) {
    // CRITICAL: Each variable's data must be contiguous
    auto s = nest::index_space(nest::ivec(0, 0), nest::uvec(4u, 5u));
    auto field = nest::field_t<double, 2>(2, s);  // 2 variables, 4x5 grid
    
    // Fill with pattern
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 5; ++j) {
            field(0, nest::ivec(i, j)) = static_cast<double>(i * 5 + j);
            field(1, nest::ivec(i, j)) = static_cast<double>((i * 5 + j) + 1000);
        }
    }
    
    // Check that var0 occupies first 20 elements, var1 the next 20
    auto* raw = field.var_data(0);
    for (int k = 0; k < 20; ++k) {
        ASSERT_NEAR(raw[k], static_cast<double>(k), 1e-10);
    }
    for (int k = 0; k < 20; ++k) {
        ASSERT_NEAR(raw[20 + k], static_cast<double>(k + 1000), 1e-10);
    }
}

TEST(field_view_access) {
    auto s = nest::index_space(nest::ivec(0), nest::uvec(10u));
    auto field = nest::field_t<double, 1>(3, s);
    
    // Access as md_view
    auto var0 = field[0];
    auto var1 = field[1];
    
    for (int i = 0; i < 10; ++i) {
        var0(nest::ivec(i)) = static_cast<double>(i);
        var1(nest::ivec(i)) = static_cast<double>(i * 2);
    }
    
    // Verify via direct access
    for (int i = 0; i < 10; ++i) {
        ASSERT_NEAR(field(0, nest::ivec(i)), static_cast<double>(i), 1e-10);
        ASSERT_NEAR(field(1, nest::ivec(i)), static_cast<double>(i * 2), 1e-10);
    }
}

// =============================================================================
// state_t tests
// =============================================================================

TEST(state_construction) {
    auto s = nest::index_space(nest::ivec(0), nest::uvec(10u));
    auto state = nest::state_t<double, 1>(3, s);
    
    ASSERT_EQ(state.nvars(), 3u);
    
    // Initialize
    for (int i = 0; i < 10; ++i) {
        state(0, nest::ivec(i)) = 1.0;
        state(1, nest::ivec(i)) = 2.0;
        state(2, nest::ivec(i)) = 3.0;
    }
    
    // Verify
    for (int i = 0; i < 10; ++i) {
        ASSERT_NEAR(state(0, nest::ivec(i)), 1.0, 1e-10);
        ASSERT_NEAR(state(1, nest::ivec(i)), 2.0, 1e-10);
        ASSERT_NEAR(state(2, nest::ivec(i)), 3.0, 1e-10);
    }
}

// =============================================================================
// patch_t tests
// =============================================================================

TEST(patch_with_halos) {
    auto interior = nest::index_space(nest::ivec(0), nest::uvec(10u));
    auto patch = nest::patch_t<double, 1>(0, interior, 2, 1);
    
    ASSERT_EQ(patch.halo_width, 1);
    ASSERT_EQ(nest::size(patch.interior), 10u);
    ASSERT_EQ(nest::size(patch.extended), 12u);  // 10 + 2*1
    
    // Extended space should start at -1
    ASSERT_EQ(nest::start(patch.extended)[0], -1);
    ASSERT_EQ(nest::upper(patch.extended)[0], 11);
}

TEST(patch_access_interior_and_halos) {
    auto interior = nest::index_space(nest::ivec(0), nest::uvec(5u));
    auto patch = nest::patch_t<double, 1>(0, interior, 1, 1);
    
    // Write to interior
    for (int i = 0; i < 5; ++i) {
        patch(0, nest::ivec(i)) = static_cast<double>(i);
    }
    
    // Write to halos
    patch(0, nest::ivec(-1)) = -10.0;  // Left halo
    patch(0, nest::ivec(5)) = 50.0;    // Right halo
    
    // Verify
    ASSERT_NEAR(patch(0, nest::ivec(-1)), -10.0, 1e-10);
    for (int i = 0; i < 5; ++i) {
        ASSERT_NEAR(patch(0, nest::ivec(i)), static_cast<double>(i), 1e-10);
    }
    ASSERT_NEAR(patch(0, nest::ivec(5)), 50.0, 1e-10);
}

// =============================================================================
// Halo exchange tests (CRITICAL for correctness)
// =============================================================================

TEST(halo_exchange_plan_structure) {
    auto domain = nest::index_space(nest::ivec(0), nest::uvec(12u));
    auto patches = nest::build_periodic_mesh_1d<double>(domain, 3, 1, 1);
    
    // Each patch should have 2 regions (left and right)
    for (const auto& p : patches) {
        ASSERT_EQ(p.exchange_plan.num_regions(), 2u);
    }
    
    // Verify patch 0's plan
    auto& p0 = patches[0];
    ASSERT_EQ(p0.exchange_plan.regions[0].peer_index, 2);  // Left neighbor (periodic)
    ASSERT_EQ(p0.exchange_plan.regions[1].peer_index, 1);  // Right neighbor
    
    // Verify region sizes
    for (const auto& region : p0.exchange_plan.regions) {
        ASSERT_EQ(region.copy_size, 1u);  // Halo width = 1
    }
}

TEST(halo_exchange_periodic_1d_correctness) {
    // 3 patches of 4 cells each, periodic boundaries
    auto domain = nest::index_space(nest::ivec(0), nest::uvec(12u));
    auto patches = nest::build_periodic_mesh_1d<double>(domain, 3, 1, 1);
    
    // Patch 0: [0, 4), Patch 1: [4, 8), Patch 2: [8, 12)
    // Set interior values to patch index
    for (auto& p : patches) {
        for (auto idx : p.interior) {
            p(0, idx) = static_cast<double>(p.index);
        }
    }
    
    // Execute exchange
    nest::execute_all_exchanges(patches);
    
    // Verify halos
    auto& p0 = patches[0];
    auto& p1 = patches[1];
    auto& p2 = patches[2];
    
    // Patch 0: left halo should have value from patch 2, right from patch 1
    ASSERT_NEAR(p0(0, nest::ivec(-1)), 2.0, 1e-10);  // From patch 2
    ASSERT_NEAR(p0(0, nest::ivec(4)), 1.0, 1e-10);   // From patch 1
    
    // Patch 1: left from patch 0, right from patch 2
    ASSERT_NEAR(p1(0, nest::ivec(3)), 0.0, 1e-10);   // From patch 0
    ASSERT_NEAR(p1(0, nest::ivec(8)), 2.0, 1e-10);   // From patch 2
    
    // Patch 2: left from patch 1, right from patch 0 (periodic wrap)
    ASSERT_NEAR(p2(0, nest::ivec(7)), 1.0, 1e-10);   // From patch 1
    ASSERT_NEAR(p2(0, nest::ivec(12)), 0.0, 1e-10);  // From patch 0 (periodic)
}

TEST(halo_exchange_multivar_correctness) {
    // Test with multiple variables
    auto domain = nest::index_space(nest::ivec(0), nest::uvec(8u));
    auto patches = nest::build_periodic_mesh_1d<double>(domain, 2, 3, 1);  // 3 vars
    
    // Set different values for each variable
    for (auto& p : patches) {
        for (auto idx : p.interior) {
            p(0, idx) = static_cast<double>(p.index);          // var 0
            p(1, idx) = static_cast<double>(p.index + 10);     // var 1
            p(2, idx) = static_cast<double>(p.index + 100);    // var 2
        }
    }
    
    // Execute exchange
    nest::execute_all_exchanges(patches);
    
    // Verify all variables were exchanged correctly
    auto& p0 = patches[0];
    ASSERT_NEAR(p0(0, nest::ivec(-1)), 1.0, 1e-10);    // var 0 from patch 1
    ASSERT_NEAR(p0(1, nest::ivec(-1)), 11.0, 1e-10);   // var 1 from patch 1
    ASSERT_NEAR(p0(2, nest::ivec(-1)), 101.0, 1e-10);  // var 2 from patch 1
}

TEST(halo_exchange_wider_halos) {
    // Test with halo width = 2
    auto domain = nest::index_space(nest::ivec(0), nest::uvec(20u));
    auto patches = nest::build_periodic_mesh_1d<double>(domain, 2, 1, 2);
    
    // Set interior values
    for (auto& p : patches) {
        for (auto idx : p.interior) {
            p(0, idx) = static_cast<double>(idx[0]);  // Use global index
        }
    }
    
    // Execute exchange
    nest::execute_all_exchanges(patches);
    
    // Patch 0: interior is [0, 10), halos at [-2, -1] and [10, 11]
    auto& p0 = patches[0];
    ASSERT_NEAR(p0(0, nest::ivec(-2)), 18.0, 1e-10);  // From patch 1, cell 18
    ASSERT_NEAR(p0(0, nest::ivec(-1)), 19.0, 1e-10);  // From patch 1, cell 19
    ASSERT_NEAR(p0(0, nest::ivec(10)), 10.0, 1e-10);  // From patch 1, cell 10
    ASSERT_NEAR(p0(0, nest::ivec(11)), 11.0, 1e-10);  // From patch 1, cell 11
}

TEST(halo_exchange_periodic_2d_faces_correctness) {
    // 2x2 patches periodic in both directions, halo=1. Validate face halos (not corners).
    constexpr int nx = 6;
    constexpr int ny = 4;
    auto domain = nest::index_space(nest::ivec(0, 0), nest::uvec(static_cast<unsigned int>(nx), static_cast<unsigned int>(ny)));
    auto patches = nest::build_periodic_mesh_2d<double>(domain, 2, 2, 1, 1);

    // Fill interior with a value that depends only on wrapped global coords.
    for (auto& p : patches) {
        for (auto idx : p.interior) {
            p(0, idx) = static_cast<double>(idx[0] + 1000 * idx[1]);
        }
    }

    nest::execute_all_exchanges(patches);

    auto wrap = [](int a, int n) {
        int r = a % n;
        return (r < 0) ? (r + n) : r;
    };

    for (auto& p : patches) {
        const int i0 = nest::start(p.interior)[0];
        const int i1 = nest::upper(p.interior)[0];
        const int j0 = nest::start(p.interior)[1];
        const int j1 = nest::upper(p.interior)[1];

        // Left/right halos (x faces)
        for (int j = j0; j < j1; ++j) {
            const int iL = i0 - 1;
            const int iR = i1;
            ASSERT_NEAR(p(0, nest::ivec(iL, j)), static_cast<double>(wrap(iL, nx) + 1000 * wrap(j, ny)), 1e-10);
            ASSERT_NEAR(p(0, nest::ivec(iR, j)), static_cast<double>(wrap(iR, nx) + 1000 * wrap(j, ny)), 1e-10);
        }

        // Bottom/top halos (y faces)
        for (int i = i0; i < i1; ++i) {
            const int jB = j0 - 1;
            const int jT = j1;
            ASSERT_NEAR(p(0, nest::ivec(i, jB)), static_cast<double>(wrap(i, nx) + 1000 * wrap(jB, ny)), 1e-10);
            ASSERT_NEAR(p(0, nest::ivec(i, jT)), static_cast<double>(wrap(i, nx) + 1000 * wrap(jT, ny)), 1e-10);
        }
    }
}

// =============================================================================
// for_each tests
// =============================================================================

TEST(for_each_respects_index_order) {
    auto s = nest::index_space(nest::ivec(0), nest::uvec(100u));
    std::vector<int> order;
    
    nest::for_each(s, [&](nest::ivec_t<1> idx) {
        order.push_back(idx[0]);
    }, nest::exec::cpu);
    
    // Sequential execution should preserve order
    for (int i = 0; i < 100; ++i) {
        ASSERT_EQ(order[static_cast<std::size_t>(i)], i);
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n=== nest Comprehensive Core Tests ===\n\n";
    
    // Tests run automatically via static initialization
    
    std::cout << "\n=== Summary ===\n";
    std::cout << g_tests_passed << "/" << g_tests_run << " tests passed\n";
    
    if (g_tests_passed != g_tests_run) {
        std::cout << "\nFAILED TESTS DETECTED\n";
        return 1;
    }
    
    std::cout << "\nAll tests passed successfully!\n";
    return 0;
}
