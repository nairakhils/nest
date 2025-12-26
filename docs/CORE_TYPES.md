# nest Core Types

This document describes the core types and their design guarantees.

## Type Hierarchy

```
index_space_t<Dim>       # Rectangular region with start and shape
    ↓
md_view_t<T, Dim>        # Multi-dimensional view (pointer + strides)
    ↓
field_t<T, Dim>          # SoA layout for multiple variables
    ↓
state_t<T, Dim>          # Conserved variables + time
    ↓
patch_t<T, Dim>          # Patch with halos and exchange plan
```

## index_space_t<Dim>

Defines a rectangular region via `_start` and `_shape` (absolute indices).

**Invariants:**
- Iteration order is **row-major** (last dimension varies fastest)
- `size() == product(shape)`
- `upper() == start + shape` (one past the end)

**Operations:**
- `expand(space, n)` - add `n` cells on each side
- `subspace(space, n_parts, part_idx, axis)` - partition along axis

## md_view_t<T, Dim>

Multi-dimensional view with pointer and strides.

**Construction:**
```cpp
auto view = md_view_t<double, 1>(data, space);        // Contiguous row-major
auto view = md_view_t<double, 2>(data, space, strides); // Custom strides
```

**Invariants:**
- `view(idx)` accesses: `data[sum(strides[d] * (idx[d] - start[d]))]`
- `is_contiguous(view)` checks row-major layout
- Strides are in **elements**, not bytes

**Use cases:**
- Wrap raw pointers with spatial information
- Create subviews without copying
- Abstract strided access patterns

## field_t<T, Dim>

SoA (Structure of Arrays) layout for multiple variables.

**Layout guarantee:**
```
data[var * stride + linear_index(space, idx)]
where stride = size(space)
```

**Memory layout (nvars=3, space size=100):**
```
[var0: 100 elements][var1: 100 elements][var2: 100 elements]
```

**Invariants:**
- Each variable occupies a contiguous block
- Variables are laid out sequentially
- `field[var]` returns an `md_view_t` of that variable
- `field(var, idx)` directly accesses element

**Critical for performance:**
- GPU coalesced memory access
- CPU vectorization within a variable
- Cache-friendly when iterating over spatial indices

## state_t<T, Dim>

Conserved variables for a patch.

**Contents:**
```cpp
field_t<T, Dim> conserved;  // SoA field
double time;                // Current time
```

**Access:**
- `state[var]` - get `md_view_t` of variable `var`
- `state(var, idx)` - get/set element directly

## patch_t<T, Dim>

Rectangular region with halos and precomputed exchange plans.

**Structure:**
```cpp
index_space_t<Dim> interior;       // Owned cells
index_space_t<Dim> extended;       // Interior + halos
state_t<T, Dim> state;             // Conserved variables
halo_exchange_plan_t<Dim> exchange_plan;  // Precomputed copy operations
```

**Invariants:**
- `extended == expand(interior, halo_width)`
- `state.conserved` is allocated over `extended` space
- Exchange plan is computed once at mesh construction

## halo_exchange_plan_t<Dim>

Precomputed halo exchange operations.

**Structure:**
```cpp
struct halo_region_t {
    int peer_index;             // Neighbor patch (-1 if boundary)
    index_space_t<Dim> src_space;  // Source region in neighbor
    index_space_t<Dim> dst_space;  // Destination halo in this patch
    std::size_t copy_size;      // Number of elements (for optimization)
};
```

**Design rationale:**
- Computed once at mesh construction (amortized cost)
- No runtime neighbor search
- Enables optimized copy kernels (memcpy for contiguous 1D)
- Easily parallelizable (no data races between regions)

**Exchange algorithm:**
```cpp
void execute_exchange(patch_t& patch, const vector<patch_t>& all_patches) {
    for (const auto& region : patch.exchange_plan.regions) {
        for (var : 0..nvars) {
            copy(all_patches[region.peer_index], region.src_space,
                 patch, region.dst_space, var);
        }
    }
}
```

## Indexing Correctness

### Linear Indexing (Row-Major)

For `space = index_space(start, shape)`:

```
linear_index(space, idx) = sum((idx[d] - start[d]) * stride[d])
where stride[d] = product(shape[d+1:])
```

**1D example:**
- `space = [5, 10)` (start=5, shape=10)
- `linear_index(space, [5]) = 0`
- `linear_index(space, [9]) = 4`

**2D example (2x3 grid):**
- `space = [[0,0], [2,3]]`
- `linear_index(space, [0,0]) = 0`
- `linear_index(space, [0,2]) = 2`
- `linear_index(space, [1,0]) = 3` (row-major!)
- `linear_index(space, [1,2]) = 5`

### SoA Indexing

```
soa_offset(space, var, idx) = var * size(space) + linear_index(space, idx)
```

**Example (3 vars, 10 cells):**
- Variable 0, cell 5: offset = 0 * 10 + 5 = 5
- Variable 1, cell 5: offset = 1 * 10 + 5 = 15
- Variable 2, cell 5: offset = 2 * 10 + 5 = 25

## Halo Exchange Correctness (Periodic)

**Test case:** 3 patches, 4 cells each, halo width = 1

```
Domain: [0, 12)
Patch 0: interior [0, 4), extended [-1, 5)
Patch 1: interior [4, 8), extended [3, 9)
Patch 2: interior [8, 12), extended [7, 13)
```

**Exchange plan for Patch 0:**
1. Left halo (dst=[-1]): receive from Patch 2, src=[11] (periodic)
2. Right halo (dst=[4]): receive from Patch 1, src=[4]

**Verification:**
- Set interior values to patch index
- Execute exchange
- Check: `patch0[-1] == 2.0` and `patch0[4] == 1.0`

## SoA Contiguity Assumptions

**Assumption 1:** Each variable occupies a contiguous memory block.

```cpp
field_t<double, 1> f(3, space);  // 3 vars, N cells
auto* var0 = f.var_data(0);
auto* var1 = f.var_data(1);
assert(var1 == var0 + N);  // Variables are sequential
```

**Assumption 2:** Within a variable, elements follow row-major order.

```cpp
field_t<double, 2> f(1, space);  // 2D grid
auto* ptr = f.var_data(0);
// ptr[0] = element [0,0]
// ptr[1] = element [0,1]
// ptr[cols] = element [1,0]
```

**Assumption 3:** Memcpy is valid for contiguous regions.

```cpp
// Fast path for 1D exchanges
std::memcpy(dst, src, copy_size * sizeof(T));
```

## Unit Test Coverage

### Indexing Tests (8 tests)
- `linear_index_1d_correctness`
- `linear_index_1d_with_offset`
- `linear_index_2d_row_major`
- `soa_offset_correctness`
- `md_view_construction_and_access`
- `md_view_with_offset`
- `md_view_2d_strides`
- `md_view_contiguity_check`

### SoA Tests (3 tests)
- `field_soa_layout`
- `field_soa_contiguity_per_variable`
- `field_view_access`

### Halo Exchange Tests (4 tests)
- `halo_exchange_plan_structure`
- `halo_exchange_periodic_1d_correctness`
- `halo_exchange_multivar_correctness`
- `halo_exchange_wider_halos`

**Total: 23 tests, all passing**

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `linear_index()` | O(Dim) | Compile-time unrolled |
| `field[var]` | O(1) | Pointer arithmetic |
| `execute_exchange()` | O(regions × vars × copy_size) | Memcpy for 1D |
| `for_each()` | O(size) | Parallelized with OpenMP |

## Future Extensions

1. **2D/3D mesh builders** with face/edge/corner exchanges
2. **Non-periodic boundaries** (outflow, reflecting)
3. **CUDA kernels** using same SoA layout
4. **Async exchange** with compute/communication overlap
5. **MPI support** for distributed memory

