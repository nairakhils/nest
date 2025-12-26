#pragma once
#include "patch.hpp"
#include <limits>
#include <tuple>
#include <type_traits>
#include <functional>

namespace nest {

// =============================================================================
// Stage type traits
// =============================================================================

namespace detail {

// Check if type has ::name
template<typename T, typename = void>
struct has_name : std::false_type {};

template<typename T>
struct has_name<T, std::void_t<decltype(T::name)>> : std::true_type {};

// Check if type has value(patch_t) -> patch_t (Compute stage)
template<typename T, typename Patch, typename = void>
struct is_compute_stage : std::false_type {};

template<typename T, typename Patch>
struct is_compute_stage<T, Patch, std::void_t<
    decltype(std::declval<const T&>().value(std::declval<Patch>()))
>> : std::true_type {};

// Check if type has reduce(acc, patch) and finalize (Reduce stage)
template<typename T, typename Patch, typename = void>
struct is_reduce_stage : std::false_type {};

template<typename T, typename Patch>
struct is_reduce_stage<T, Patch, std::void_t<
    typename T::value_type,
    decltype(T::init()),
    decltype(std::declval<const T&>().reduce(std::declval<typename T::value_type>(), 
                                              std::declval<const Patch&>())),
    decltype(std::declval<const T&>().finalize(std::declval<typename T::value_type>(), 
                                                std::declval<Patch&>()))
>> : std::true_type {};

// Check if type has need() and provides() (Exchange stage)
template<typename T, typename Patch, typename = void>
struct is_exchange_stage : std::false_type {};

template<typename T, typename Patch>
struct is_exchange_stage<T, Patch, std::void_t<
    decltype(std::declval<const T&>().provides(std::declval<const Patch&>())),
    decltype(std::declval<const T&>().need(std::declval<Patch&>(), std::declval<std::function<void(int,int)>>()))
>> : std::true_type {};

} // namespace detail

// =============================================================================
// Pipeline - container of stages
// =============================================================================

template<typename... Stages>
struct pipeline_t {
    std::tuple<Stages...> stages;
    
    explicit pipeline_t(Stages... s) : stages(std::move(s)...) {}
};

template<typename... Stages>
auto pipeline(Stages... stages) -> pipeline_t<Stages...> {
    return pipeline_t<Stages...>(std::move(stages)...);
}

// =============================================================================
// Execute pipeline over patches (simple sequential implementation)
// =============================================================================

template<typename T, std::size_t Dim, typename Stage>
void execute_stage(Stage& stage, std::vector<patch_t<T, Dim>>& patches, exec::cpu_t) {
    using patch_type = patch_t<T, Dim>;
    
    if constexpr (detail::is_compute_stage<Stage, patch_type>::value) {
        // Compute stage: transform each patch
        for (auto& patch : patches) {
            patch = stage.value(std::move(patch));
        }
    } else if constexpr (detail::is_reduce_stage<Stage, patch_type>::value) {
        // Reduce stage: accumulate across patches, then distribute
        auto acc = Stage::init();
        for (const auto& patch : patches) {
            acc = stage.reduce(acc, patch);
        }
        for (auto& patch : patches) {
            stage.finalize(acc, patch);
        }
    } else {
        // Assume exchange-like behavior: fill halos
        fill_all_halos(patches);
    }
}

#if defined(NEST_HAS_OPENMP)
template<typename T, std::size_t Dim, typename Stage>
void execute_stage(Stage& stage, std::vector<patch_t<T, Dim>>& patches, exec::omp_t) {
    using patch_type = patch_t<T, Dim>;
    auto n = static_cast<int>(patches.size());
    
    if constexpr (detail::is_compute_stage<Stage, patch_type>::value) {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            patches[static_cast<std::size_t>(i)] = 
                stage.value(std::move(patches[static_cast<std::size_t>(i)]));
        }
    } else if constexpr (detail::is_reduce_stage<Stage, patch_type>::value) {
        auto acc = Stage::init();
        #pragma omp parallel for reduction(min:acc)
        for (int i = 0; i < n; ++i) {
            acc = stage.reduce(acc, patches[static_cast<std::size_t>(i)]);
        }
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            stage.finalize(acc, patches[static_cast<std::size_t>(i)]);
        }
    } else {
        // Exchange: each patch fills from neighbors
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            fill_halos(patches[static_cast<std::size_t>(i)], patches);
        }
    }
}
#else
template<typename T, std::size_t Dim, typename Stage>
void execute_stage(Stage& stage, std::vector<patch_t<T, Dim>>& patches, exec::omp_t) {
    execute_stage(stage, patches, exec::cpu);
}
#endif

// Execute full pipeline
template<typename T, std::size_t Dim, typename... Stages, typename Exec>
void execute(pipeline_t<Stages...>& pipeline, std::vector<patch_t<T, Dim>>& patches, Exec exec) {
    std::apply([&](auto&... stages) {
        (execute_stage<T, Dim>(stages, patches, exec), ...);
    }, pipeline.stages);
}

// Convenience: execute with default CPU exec
template<typename T, std::size_t Dim, typename... Stages>
void execute(pipeline_t<Stages...>& pipeline, std::vector<patch_t<T, Dim>>& patches) {
    execute(pipeline, patches, exec::cpu);
}

} // namespace nest



