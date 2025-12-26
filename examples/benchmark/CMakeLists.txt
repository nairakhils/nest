# Benchmark: 1D Euler Solver Performance

add_executable(benchmark_euler1d benchmark_euler1d.cpp)

target_link_libraries(benchmark_euler1d PRIVATE nest)

target_include_directories(benchmark_euler1d PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/examples/sod1d
)

# Enable OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(benchmark_euler1d PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(benchmark_euler1d PRIVATE NEST_HAS_OPENMP)
endif()

# Link CUDA solver if available
if(CMAKE_CUDA_COMPILER AND TARGET nest_cuda)
    target_link_libraries(benchmark_euler1d PRIVATE nest_cuda)
    target_compile_definitions(benchmark_euler1d PRIVATE NEST_HAS_CUDA)
    target_include_directories(benchmark_euler1d PRIVATE
        ${CMAKE_SOURCE_DIR}/src/backend_cuda
    )
endif()

# Enable fast math for release
target_compile_options(benchmark_euler1d PRIVATE
    $<$<CONFIG:Release>:-ffast-math>
)

