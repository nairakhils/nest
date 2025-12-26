add_executable(sod2d sod2d.cpp)
target_link_libraries(sod2d PRIVATE nest nest_cpu)
target_include_directories(sod2d PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples/sod1d
)

