#pragma once
#include "core.hpp"
#include <memory>
#include <vector>
#include <algorithm>
#include <cstring>

namespace nest {

// =============================================================================
// halo_exchange_plan_t - precomputed copy operations for halo exchange
// =============================================================================

template<std::size_t Dim>
struct halo_region_t {
    int peer_index = -1;           // Neighbor patch index (-1 if boundary)
    index_space_t<Dim> src_space;  // Source region in neighbor
    index_space_t<Dim> dst_space;  // Destination region in this patch
    std::size_t copy_size = 0;     // Number of elements to copy
};

template<std::size_t Dim>
struct halo_exchange_plan_t {
    std::vector<halo_region_t<Dim>> regions;  // All halo regions for this patch
    
    // Add a region to exchange
    void add_region(int peer, index_space_t<Dim> src, index_space_t<Dim> dst) {
        regions.push_back(halo_region_t<Dim>{
            .peer_index = peer,
            .src_space = src,
            .dst_space = dst,
            .copy_size = size(dst)
        });
    }
    
    auto num_regions() const -> std::size_t { return regions.size(); }
};

// =============================================================================
// patch_t - rectangular region with halos and state
// =============================================================================

template<typename T, std::size_t Dim>
struct patch_t {
    int index = 0;                          // Patch index in mesh
    index_space_t<Dim> interior;            // Owned cells (absolute indices)
    index_space_t<Dim> extended;            // Interior + halos
    int halo_width = 1;                     // Halo cells on each side
    
    state_t<T, Dim> state;                  // Conserved variables
    halo_exchange_plan_t<Dim> exchange_plan; // Precomputed exchange info
    
    patch_t() = default;
    
    patch_t(int idx, index_space_t<Dim> interior_space, std::size_t nvars, int halo = 1)
        : index(idx)
        , interior(interior_space)
        , extended(expand(interior_space, halo))
        , halo_width(halo)
        , state(nvars, extended)
    {}
    
    // Access conserved variables
    auto operator[](std::size_t var) -> md_view_t<T, Dim> {
        return state[var];
    }
    
    auto operator[](std::size_t var) const -> md_view_t<const T, Dim> {
        return state[var];
    }
    
    auto operator()(std::size_t var, ivec_t<Dim> idx) -> T& {
        return state(var, idx);
    }
    
    auto operator()(std::size_t var, ivec_t<Dim> idx) const -> const T& {
        return state(var, idx);
    }
    
    auto nvars() const -> std::size_t { return state.nvars(); }
};

// =============================================================================
// Mesh utilities - build patches and compute exchange plans
// =============================================================================

// Build 1D periodic mesh with n_patches patches
template<typename T, std::size_t Dim>
auto build_periodic_mesh_1d(index_space_t<Dim> domain, int n_patches, std::size_t nvars, int halo = 1) 
    -> std::vector<patch_t<T, Dim>>
{
    static_assert(Dim == 1, "build_periodic_mesh_1d only supports 1D");
    
    auto patches = std::vector<patch_t<T, Dim>>{};
    patches.reserve(static_cast<std::size_t>(n_patches));
    
    // Create patches
    for (int p = 0; p < n_patches; ++p) {
        auto sub = subspace(domain, n_patches, p, 0);
        patches.emplace_back(p, sub, nvars, halo);
    }
    
    // Build exchange plans (periodic boundaries)
    for (int p = 0; p < n_patches; ++p) {
        auto& patch = patches[static_cast<std::size_t>(p)];
        
        // Left neighbor (periodic)
        int left_peer = (p - 1 + n_patches) % n_patches;
        auto& left_nbr = patches[static_cast<std::size_t>(left_peer)];
        
        // Left halo: receive from right edge of left neighbor
        {
            auto dst_start = start(patch.interior);
            dst_start[0] -= halo;
            auto dst_shape = shape(patch.interior);
            dst_shape[0] = static_cast<unsigned int>(halo);
            auto dst_space = index_space(dst_start, dst_shape);
            
            auto src_start = upper(left_nbr.interior);
            src_start[0] -= halo;
            auto src_shape = dst_shape;
            auto src_space = index_space(src_start, src_shape);
            
            patch.exchange_plan.add_region(left_peer, src_space, dst_space);
        }
        
        // Right neighbor (periodic)
        int right_peer = (p + 1) % n_patches;
        auto& right_nbr = patches[static_cast<std::size_t>(right_peer)];
        
        // Right halo: receive from left edge of right neighbor
        {
            auto dst_start = upper(patch.interior);
            auto dst_shape = shape(patch.interior);
            dst_shape[0] = static_cast<unsigned int>(halo);
            auto dst_space = index_space(dst_start, dst_shape);
            
            auto src_start = start(right_nbr.interior);
            auto src_shape = dst_shape;
            auto src_space = index_space(src_start, src_shape);
            
            patch.exchange_plan.add_region(right_peer, src_space, dst_space);
        }
    }
    
    return patches;
}

// =============================================================================
// Halo exchange using precomputed plans
// =============================================================================

template<typename T, std::size_t Dim>
void execute_exchange(patch_t<T, Dim>& patch, const std::vector<patch_t<T, Dim>>& all_patches) {
    for (const auto& region : patch.exchange_plan.regions) {
        if (region.peer_index < 0) continue;  // Boundary
        
        const auto& peer = all_patches[static_cast<std::size_t>(region.peer_index)];
        
        // Copy each variable using the plan
        for (std::size_t var = 0; var < patch.nvars(); ++var) {
            // Contiguous copy if both regions are contiguous in memory
            if (Dim == 1 && region.copy_size > 0) {
                // Fast path: 1D contiguous memcpy
                auto* src = peer.state.conserved.var_data(var) + 
                           linear_index(peer.extended, start(region.src_space));
                auto* dst = patch.state.conserved.var_data(var) + 
                           linear_index(patch.extended, start(region.dst_space));
                std::memcpy(dst, src, region.copy_size * sizeof(T));
            } else {
                // General case: element-by-element copy
                for (auto dst_idx : region.dst_space) {
                    // Compute corresponding source index
                    auto src_idx = start(region.src_space);
                    for (std::size_t d = 0; d < Dim; ++d) {
                        src_idx[d] += dst_idx[d] - start(region.dst_space)[d];
                    }
                    patch(var, dst_idx) = peer(var, src_idx);
                }
            }
        }
    }
}

template<typename T, std::size_t Dim>
void execute_all_exchanges(std::vector<patch_t<T, Dim>>& patches) {
    for (auto& patch : patches) {
        execute_exchange(patch, patches);
    }
}

// Backward compatibility aliases
template<std::size_t Dim>
using neighbor_info_t = halo_region_t<Dim>;

template<typename T, std::size_t Dim>
void fill_halos(patch_t<T, Dim>& patch, const std::vector<patch_t<T, Dim>>& all_patches) {
    execute_exchange(patch, all_patches);
}

template<typename T, std::size_t Dim>
void fill_all_halos(std::vector<patch_t<T, Dim>>& patches) {
    execute_all_exchanges(patches);
}

// Build 2D periodic mesh with nx x ny patches
template<typename T>
auto build_periodic_mesh_2d(
    index_space_t<2> domain,
    int nx_patches,
    int ny_patches,
    std::size_t nvars,
    int halo = 1
) -> std::vector<patch_t<T, 2>> {
    auto patches = std::vector<patch_t<T, 2>>{};
    patches.reserve(static_cast<std::size_t>(nx_patches * ny_patches));
    
    // Create patches in row-major order
    for (int py = 0; py < ny_patches; ++py) {
        for (int px = 0; px < nx_patches; ++px) {
            int patch_idx = py * nx_patches + px;
            
            // Partition domain along both axes
            auto sub_x = subspace(domain, nx_patches, px, 0);
            auto sub = subspace(sub_x, ny_patches, py, 1);
            
            patches.emplace_back(patch_idx, sub, nvars, halo);
        }
    }
    
    // Build exchange plans (periodic boundaries in both directions)
    for (int py = 0; py < ny_patches; ++py) {
        for (int px = 0; px < nx_patches; ++px) {
            int patch_idx = py * nx_patches + px;
            auto& patch = patches[static_cast<std::size_t>(patch_idx)];
            
            // X-direction neighbors
            int left_px = (px - 1 + nx_patches) % nx_patches;
            int right_px = (px + 1) % nx_patches;
            int left_idx = py * nx_patches + left_px;
            int right_idx = py * nx_patches + right_px;
            
            auto& left_nbr = patches[static_cast<std::size_t>(left_idx)];
            auto& right_nbr = patches[static_cast<std::size_t>(right_idx)];
            
            // Left halo (x-direction): receive from right edge of left neighbor
            {
                auto dst_start = start(patch.interior);
                dst_start[0] -= halo;  // Move x to halo region
                auto dst_shape = shape(patch.interior);
                dst_shape[0] = static_cast<unsigned int>(halo);
                auto dst_space = index_space(dst_start, dst_shape);
                
                // Source: right edge of left neighbor's interior
                auto src_start = start(left_nbr.interior);
                src_start[0] = upper(left_nbr.interior)[0] - halo;  // x at right edge
                // y stays at start (correct y-range)
                auto src_shape = dst_shape;
                auto src_space = index_space(src_start, src_shape);
                
                patch.exchange_plan.add_region(left_idx, src_space, dst_space);
            }
            
            // Right halo (x-direction): receive from left edge of right neighbor
            {
                auto dst_start = start(patch.interior);
                dst_start[0] = upper(patch.interior)[0];  // x at right halo
                // y stays at start (correct y-range)
                auto dst_shape = shape(patch.interior);
                dst_shape[0] = static_cast<unsigned int>(halo);
                auto dst_space = index_space(dst_start, dst_shape);
                
                auto src_start = start(right_nbr.interior);  // Left edge of right neighbor
                auto src_shape = dst_shape;
                auto src_space = index_space(src_start, src_shape);
                
                patch.exchange_plan.add_region(right_idx, src_space, dst_space);
            }
            
            // Y-direction neighbors
            int bottom_py = (py - 1 + ny_patches) % ny_patches;
            int top_py = (py + 1) % ny_patches;
            int bottom_idx = bottom_py * nx_patches + px;
            int top_idx = top_py * nx_patches + px;
            
            auto& bottom_nbr = patches[static_cast<std::size_t>(bottom_idx)];
            auto& top_nbr = patches[static_cast<std::size_t>(top_idx)];
            
            // Bottom halo (y-direction): receive from top edge of bottom neighbor
            {
                auto dst_start = start(patch.interior);
                dst_start[1] -= halo;  // Move y to halo region
                auto dst_shape = shape(patch.interior);
                dst_shape[1] = static_cast<unsigned int>(halo);
                auto dst_space = index_space(dst_start, dst_shape);
                
                // Source: top edge of bottom neighbor's interior
                auto src_start = start(bottom_nbr.interior);
                src_start[1] = upper(bottom_nbr.interior)[1] - halo;  // y at top edge
                // x stays at start (correct x-range)
                auto src_shape = dst_shape;
                auto src_space = index_space(src_start, src_shape);
                
                patch.exchange_plan.add_region(bottom_idx, src_space, dst_space);
            }
            
            // Top halo (y-direction): receive from bottom edge of top neighbor
            {
                auto dst_start = start(patch.interior);
                dst_start[1] = upper(patch.interior)[1];  // y at top halo
                // x stays at start (correct x-range)
                auto dst_shape = shape(patch.interior);
                dst_shape[1] = static_cast<unsigned int>(halo);
                auto dst_space = index_space(dst_start, dst_shape);
                
                auto src_start = start(top_nbr.interior);  // Bottom edge of top neighbor
                auto src_shape = dst_shape;
                auto src_space = index_space(src_start, src_shape);
                
                patch.exchange_plan.add_region(top_idx, src_space, dst_space);
            }
        }
    }
    
    return patches;
}

// Legacy alias for old code
template<std::size_t Dim>
auto build_periodic_mesh(index_space_t<Dim> domain, int n_patches, std::size_t nvars, int halo = 1) 
    -> std::vector<patch_t<double, Dim>>
{
    return build_periodic_mesh_1d<double>(domain, n_patches, nvars, halo);
}

} // namespace nest
