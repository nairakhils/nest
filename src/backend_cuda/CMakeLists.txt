# CUDA backend for nest

# Option for shared-memory tiling (experimental, best for 2D)
option(NEST_CUDA_SHARED_MEM "Enable shared-memory tiling for stencil operations" OFF)

# CUDA library with Euler solver
add_library(nest_cuda STATIC
    euler1d_solver.cu
)

target_include_directories(nest_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(nest_cuda PUBLIC nest)

# CUDA-specific settings
set_target_properties(nest_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86;89;90"  # Volta through Hopper
)

# Enable C++20 for CUDA
target_compile_features(nest_cuda PUBLIC cxx_std_20)

# Add CUDA runtime
target_link_libraries(nest_cuda PUBLIC CUDA::cudart)

# Optional shared-memory tiling
if(NEST_CUDA_SHARED_MEM)
    target_compile_definitions(nest_cuda PRIVATE NEST_CUDA_SHARED_MEM)
    message(STATUS "CUDA shared-memory tiling enabled")
endif()
