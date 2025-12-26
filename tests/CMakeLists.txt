# Tests
add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE nest nest_cpu)

add_executable(test_euler test_euler.cpp)
target_link_libraries(test_euler PRIVATE nest nest_cpu)
target_include_directories(test_euler PRIVATE ${CMAKE_SOURCE_DIR}/examples/sod1d)

# Register tests with CTest
add_test(NAME test_core COMMAND test_core)
add_test(NAME test_euler COMMAND test_euler)

# CUDA tests (only if CUDA is enabled)
if(NEST_ENABLE_CUDA)
    add_executable(test_euler_cuda test_euler_cuda.cpp)
    target_link_libraries(test_euler_cuda PRIVATE nest nest_cuda)
    target_include_directories(test_euler_cuda PRIVATE 
        ${CMAKE_SOURCE_DIR}/examples/sod1d
        ${CMAKE_SOURCE_DIR}/src/backend_cuda
    )
    add_test(NAME test_euler_cuda COMMAND test_euler_cuda)
endif()



