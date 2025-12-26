cmake_minimum_required(VERSION 3.20)
project(nest VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(NEST_ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(NEST_ENABLE_CUDA "Enable CUDA backend" OFF)
option(NEST_BUILD_EXAMPLES "Build example programs" ON)
option(NEST_BUILD_TESTS "Build tests" ON)

# Header-only core library
add_library(nest INTERFACE)
target_include_directories(nest INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# OpenMP support
if(NEST_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(nest INTERFACE OpenMP::OpenMP_CXX)
        target_compile_definitions(nest INTERFACE NEST_HAS_OPENMP=1)
    else()
        # macOS: try manual OpenMP flags for Apple Clang
        if(APPLE)
            # Check for Homebrew libomp
            find_library(LIBOMP_LIBRARY omp HINTS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
            find_path(LIBOMP_INCLUDE_DIR omp.h HINTS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
            if(LIBOMP_LIBRARY AND LIBOMP_INCLUDE_DIR)
                target_include_directories(nest INTERFACE ${LIBOMP_INCLUDE_DIR})
                target_link_libraries(nest INTERFACE ${LIBOMP_LIBRARY})
                target_compile_options(nest INTERFACE -Xclang -fopenmp)
                target_compile_definitions(nest INTERFACE NEST_HAS_OPENMP=1)
                message(STATUS "Using Homebrew libomp: ${LIBOMP_LIBRARY}")
            else()
                message(WARNING "OpenMP not found, building without parallelization")
            endif()
        endif()
    endif()
endif()

# CUDA backend
if(NEST_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(nest INTERFACE NEST_HAS_CUDA=1)
    
    add_subdirectory(src/backend_cuda)
endif()

# CPU backend library (compiled)
add_library(nest_cpu STATIC
    src/backend_cpu/cpu_exec.cpp
)
target_link_libraries(nest_cpu PUBLIC nest)

# Examples
if(NEST_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(NEST_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
    
    # Guard test: ensure no legacy prototype references reappear
    add_test(
        NAME no_legacy_refs
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/guard_independence.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(no_legacy_refs PROPERTIES
        LABELS "guard;sanity"
    )
endif()



