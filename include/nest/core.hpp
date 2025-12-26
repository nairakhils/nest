#pragma once
#include <array>
#include <cstddef>
#include <numeric>
#include <type_traits>

namespace nest {

// =============================================================================
// Allocation / initialization tags
// =============================================================================

enum class field_init_t {
    zeroed,
    uninitialized
};

// =============================================================================
// vec_t - statically sized array
// =============================================================================

template<typename T, std::size_t N>
struct vec_t {
    std::array<T, N> _data{};

    constexpr vec_t() = default;

    template<typename... Args>
        requires (sizeof...(Args) == N && (std::is_convertible_v<Args, T> && ...))
    constexpr vec_t(Args... args) : _data{static_cast<T>(args)...} {}

    constexpr auto operator[](std::size_t i) -> T& { return _data[i]; }
    constexpr auto operator[](std::size_t i) const -> const T& { return _data[i]; }

    constexpr auto size() const -> std::size_t { return N; }
    constexpr auto begin() { return _data.begin(); }
    constexpr auto end() { return _data.end(); }
    constexpr auto begin() const { return _data.begin(); }
    constexpr auto end() const { return _data.end(); }

    constexpr auto operator==(const vec_t& other) const -> bool = default;
};

// Aliases
template<std::size_t N> using ivec_t = vec_t<int, N>;
template<std::size_t N> using uvec_t = vec_t<unsigned int, N>;
template<std::size_t N> using dvec_t = vec_t<double, N>;

// Factory functions
template<typename... Args>
constexpr auto ivec(Args... args) -> ivec_t<sizeof...(Args)> {
    return ivec_t<sizeof...(Args)>{static_cast<int>(args)...};
}

template<typename... Args>
constexpr auto uvec(Args... args) -> uvec_t<sizeof...(Args)> {
    return uvec_t<sizeof...(Args)>{static_cast<unsigned int>(args)...};
}

// Arithmetic
template<typename T, std::size_t N>
constexpr auto operator+(const vec_t<T, N>& a, const vec_t<T, N>& b) -> vec_t<T, N> {
    auto result = vec_t<T, N>{};
    for (std::size_t i = 0; i < N; ++i) result[i] = a[i] + b[i];
    return result;
}

template<typename T, std::size_t N>
constexpr auto operator-(const vec_t<T, N>& a, const vec_t<T, N>& b) -> vec_t<T, N> {
    auto result = vec_t<T, N>{};
    for (std::size_t i = 0; i < N; ++i) result[i] = a[i] - b[i];
    return result;
}

template<typename T, std::size_t N, typename S>
constexpr auto operator*(const vec_t<T, N>& v, S scalar) -> vec_t<T, N> {
    auto result = vec_t<T, N>{};
    for (std::size_t i = 0; i < N; ++i) result[i] = v[i] * static_cast<T>(scalar);
    return result;
}

template<typename T, std::size_t N>
constexpr auto product(const vec_t<T, N>& v) -> T {
    auto result = T{1};
    for (std::size_t i = 0; i < N; ++i) result *= v[i];
    return result;
}

// =============================================================================
// index_space_t - rectangular region with start and shape
// =============================================================================

template<std::size_t Dim>
struct index_space_t {
    ivec_t<Dim> _start{};
    uvec_t<Dim> _shape{};

    constexpr index_space_t() = default;
    constexpr index_space_t(ivec_t<Dim> start, uvec_t<Dim> shape)
        : _start(start), _shape(shape) {}

    // Iterator for range-based for
    struct iterator {
        ivec_t<Dim> _current;
        ivec_t<Dim> _start;
        uvec_t<Dim> _shape;

        constexpr auto operator*() const -> ivec_t<Dim> { return _current; }
        
        constexpr auto operator++() -> iterator& {
            for (int d = static_cast<int>(Dim) - 1; d >= 0; --d) {
                _current[d]++;
                if (_current[d] < _start[d] + static_cast<int>(_shape[d])) {
                    return *this;
                }
                _current[d] = _start[d];
            }
            // Mark as end
            _current[0] = _start[0] + static_cast<int>(_shape[0]);
            return *this;
        }

        constexpr auto operator==(const iterator& other) const -> bool {
            return _current == other._current;
        }
        constexpr auto operator!=(const iterator& other) const -> bool {
            return !(*this == other);
        }
    };

    constexpr auto begin() const -> iterator {
        auto it = iterator{_start, _start, _shape};
        return it;
    }

    constexpr auto end() const -> iterator {
        auto end_idx = _start;
        end_idx[0] = _start[0] + static_cast<int>(_shape[0]);
        return iterator{end_idx, _start, _shape};
    }
};

// Free functions for index_space_t
template<std::size_t Dim>
constexpr auto start(const index_space_t<Dim>& s) -> ivec_t<Dim> { return s._start; }

template<std::size_t Dim>
constexpr auto shape(const index_space_t<Dim>& s) -> uvec_t<Dim> { return s._shape; }

template<std::size_t Dim>
constexpr auto size(const index_space_t<Dim>& s) -> std::size_t {
    return static_cast<std::size_t>(product(s._shape));
}

template<std::size_t Dim>
constexpr auto upper(const index_space_t<Dim>& s) -> ivec_t<Dim> {
    auto result = ivec_t<Dim>{};
    for (std::size_t d = 0; d < Dim; ++d) {
        result[d] = s._start[d] + static_cast<int>(s._shape[d]);
    }
    return result;
}

// Factory function
template<std::size_t Dim>
constexpr auto index_space(ivec_t<Dim> start, uvec_t<Dim> shape) -> index_space_t<Dim> {
    return index_space_t<Dim>{start, shape};
}

// Expand index space by amount on each side
template<std::size_t Dim>
constexpr auto expand(const index_space_t<Dim>& s, int amount) -> index_space_t<Dim> {
    auto new_start = s._start;
    auto new_shape = s._shape;
    for (std::size_t d = 0; d < Dim; ++d) {
        new_start[d] -= amount;
        new_shape[d] += static_cast<unsigned int>(2 * amount);
    }
    return index_space_t<Dim>{new_start, new_shape};
}

// Partition index space along axis
template<std::size_t Dim>
constexpr auto subspace(const index_space_t<Dim>& s, int n_parts, int part_idx, int axis) 
    -> index_space_t<Dim> 
{
    auto new_start = s._start;
    auto new_shape = s._shape;
    auto total = static_cast<int>(s._shape[axis]);
    auto base = total / n_parts;
    auto extra = total % n_parts;
    
    auto offset = part_idx * base + std::min(part_idx, extra);
    auto len = base + (part_idx < extra ? 1 : 0);
    
    new_start[axis] += offset;
    new_shape[axis] = static_cast<unsigned int>(len);
    return index_space_t<Dim>{new_start, new_shape};
}

// =============================================================================
// Linear indexing for SoA layout
// =============================================================================

// Convert multi-dimensional index to flat offset (row-major)
template<std::size_t Dim>
constexpr auto linear_index(const index_space_t<Dim>& space, ivec_t<Dim> idx) -> std::size_t {
    auto offset = std::size_t{0};
    auto stride = std::size_t{1};
    for (int d = static_cast<int>(Dim) - 1; d >= 0; --d) {
        offset += static_cast<std::size_t>(idx[d] - space._start[d]) * stride;
        stride *= space._shape[d];
    }
    return offset;
}

// SoA offset: data[var * stride + linear_index]
template<std::size_t Dim>
constexpr auto soa_offset(const index_space_t<Dim>& space, int var, ivec_t<Dim> idx) -> std::size_t {
    return static_cast<std::size_t>(var) * size(space) + linear_index(space, idx);
}

// =============================================================================
// md_view - multi-dimensional view with pointer and strides
// =============================================================================

template<typename T, std::size_t Dim>
struct md_view_t {
    T* _data = nullptr;
    index_space_t<Dim> _space;
    uvec_t<Dim> _strides;  // Strides in elements (not bytes)
    
    md_view_t() = default;
    
    // Construct from contiguous data with row-major layout
    md_view_t(T* data, index_space_t<Dim> space)
        : _data(data), _space(space)
    {
        // Compute row-major strides
        _strides[Dim - 1] = 1;
        for (int d = static_cast<int>(Dim) - 2; d >= 0; --d) {
            _strides[d] = _strides[d + 1] * space._shape[d + 1];
        }
    }
    
    // Construct with explicit strides (for non-contiguous views)
    md_view_t(T* data, index_space_t<Dim> space, uvec_t<Dim> strides)
        : _data(data), _space(space), _strides(strides) {}
    
    // Element access
    auto operator()(ivec_t<Dim> idx) -> T& {
        auto offset = std::size_t{0};
        for (std::size_t d = 0; d < Dim; ++d) {
            offset += static_cast<std::size_t>(idx[d] - _space._start[d]) * _strides[d];
        }
        return _data[offset];
    }
    
    auto operator()(ivec_t<Dim> idx) const -> const T& {
        auto offset = std::size_t{0};
        for (std::size_t d = 0; d < Dim; ++d) {
            offset += static_cast<std::size_t>(idx[d] - _space._start[d]) * _strides[d];
        }
        return _data[offset];
    }
    
    // Get pointer at index
    auto data_at(ivec_t<Dim> idx) -> T* {
        auto offset = std::size_t{0};
        for (std::size_t d = 0; d < Dim; ++d) {
            offset += static_cast<std::size_t>(idx[d] - _space._start[d]) * _strides[d];
        }
        return _data + offset;
    }
    
    // Subview of this view
    auto subview(index_space_t<Dim> subspace) const -> md_view_t<T, Dim> {
        // Offset pointer to the first element of the subspace in this view.
        auto off = std::size_t{0};
        const auto sub_start = start(subspace);
        for (std::size_t d = 0; d < Dim; ++d) {
            off += static_cast<std::size_t>(sub_start[d] - _space._start[d]) * _strides[d];
        }
        return md_view_t<T, Dim>(_data + off, subspace, _strides);
    }
};

// Free functions
template<typename T, std::size_t Dim>
auto space(const md_view_t<T, Dim>& view) -> index_space_t<Dim> { return view._space; }

template<typename T, std::size_t Dim>
auto strides(const md_view_t<T, Dim>& view) -> uvec_t<Dim> { return view._strides; }

// Check if view is contiguous (row-major)
template<typename T, std::size_t Dim>
auto is_contiguous(const md_view_t<T, Dim>& view) -> bool {
    auto expected_stride = std::size_t{1};
    for (int d = static_cast<int>(Dim) - 1; d >= 0; --d) {
        if (view._strides[d] != expected_stride) return false;
        expected_stride *= view._space._shape[d];
    }
    return true;
}

// =============================================================================
// field_t - SoA layout for multiple variables
// =============================================================================

template<typename T, std::size_t Dim>
struct field_t {
    T* _data = nullptr;
    index_space_t<Dim> _space;
    std::size_t _nvars = 0;
    std::size_t _stride = 0;  // Stride between variables (= size(space))
    bool _owns_data = false;
    
    field_t() = default;
    
    // Allocating constructor
    field_t(std::size_t nvars, index_space_t<Dim> space, field_init_t init = field_init_t::zeroed)
        : _space(space), _nvars(nvars), _stride(size(space)), _owns_data(true)
    {
        if (init == field_init_t::zeroed) {
            _data = new T[_nvars * _stride]();
        } else {
            _data = new T[_nvars * _stride];
        }
    }
    
    // Non-owning constructor
    field_t(T* data, std::size_t nvars, index_space_t<Dim> space)
        : _data(data), _space(space), _nvars(nvars), _stride(size(space)), _owns_data(false) {}
    
    ~field_t() {
        if (_owns_data && _data) {
            delete[] _data;
        }
    }
    
    // Move semantics
    field_t(field_t&& other) noexcept
        : _data(other._data), _space(other._space), _nvars(other._nvars),
          _stride(other._stride), _owns_data(other._owns_data)
    {
        other._data = nullptr;
        other._owns_data = false;
    }
    
    field_t& operator=(field_t&& other) noexcept {
        if (this != &other) {
            if (_owns_data && _data) delete[] _data;
            _data = other._data;
            _space = other._space;
            _nvars = other._nvars;
            _stride = other._stride;
            _owns_data = other._owns_data;
            other._data = nullptr;
            other._owns_data = false;
        }
        return *this;
    }
    
    // Delete copy
    field_t(const field_t&) = delete;
    field_t& operator=(const field_t&) = delete;
    
    // Access variable as md_view
    auto operator[](std::size_t var) -> md_view_t<T, Dim> {
        return md_view_t<T, Dim>(_data + var * _stride, _space);
    }
    
    auto operator[](std::size_t var) const -> md_view_t<const T, Dim> {
        return md_view_t<const T, Dim>(_data + var * _stride, _space);
    }
    
    // Direct SoA access
    auto operator()(std::size_t var, ivec_t<Dim> idx) -> T& {
        return _data[var * _stride + linear_index(_space, idx)];
    }
    
    auto operator()(std::size_t var, ivec_t<Dim> idx) const -> const T& {
        return _data[var * _stride + linear_index(_space, idx)];
    }
    
    // Get raw pointer to variable
    auto var_data(std::size_t var) -> T* { return _data + var * _stride; }
    auto var_data(std::size_t var) const -> const T* { return _data + var * _stride; }
    
    auto nvars() const -> std::size_t { return _nvars; }
    auto stride() const -> std::size_t { return _stride; }
};

// =============================================================================
// state_t - conserved variables for a patch
// =============================================================================

template<typename T, std::size_t Dim>
struct state_t {
    field_t<T, Dim> conserved;  // Conserved variables U[var][i][j]
    double time = 0.0;
    
    state_t() = default;
    
    state_t(std::size_t nvars, index_space_t<Dim> space)
        : conserved(nvars, space) {}
    
    // Access
    auto operator[](std::size_t var) -> md_view_t<T, Dim> {
        return conserved[var];
    }
    
    auto operator[](std::size_t var) const -> md_view_t<const T, Dim> {
        return conserved[var];
    }
    
    auto operator()(std::size_t var, ivec_t<Dim> idx) -> T& {
        return conserved(var, idx);
    }
    
    auto operator()(std::size_t var, ivec_t<Dim> idx) const -> const T& {
        return conserved(var, idx);
    }
    
    auto nvars() const -> std::size_t { return conserved.nvars(); }
};

// =============================================================================
// Execution policies
// =============================================================================

namespace exec {
    struct cpu_t {};
    struct omp_t {};
    struct cuda_t {};

    inline constexpr cpu_t cpu;
    inline constexpr omp_t omp;
    inline constexpr cuda_t cuda;
}

// =============================================================================
// for_each - traverse index space
// =============================================================================

template<std::size_t Dim, typename F>
void for_each(const index_space_t<Dim>& space, F&& func, exec::cpu_t) {
    for (auto idx : space) {
        func(idx);
    }
}

#if defined(NEST_HAS_OPENMP)
#include <omp.h>

template<std::size_t Dim, typename F>
void for_each(const index_space_t<Dim>& space, F&& func, exec::omp_t) {
    auto n = static_cast<int>(size(space));
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        // Convert flat index back to multi-dim (row-major)
        auto idx = start(space);
        auto remaining = static_cast<std::size_t>(i);
        for (int d = static_cast<int>(Dim) - 1; d >= 0; --d) {
            idx[d] = space._start[d] + static_cast<int>(remaining % space._shape[d]);
            remaining /= space._shape[d];
        }
        func(idx);
    }
}
#else
// Fallback to CPU if OpenMP not available
template<std::size_t Dim, typename F>
void for_each(const index_space_t<Dim>& space, F&& func, exec::omp_t) {
    for_each(space, std::forward<F>(func), exec::cpu);
}
#endif

// Default execution: CPU sequential
template<std::size_t Dim, typename F>
void for_each(const index_space_t<Dim>& space, F&& func) {
    for_each(space, std::forward<F>(func), exec::cpu);
}

} // namespace nest



