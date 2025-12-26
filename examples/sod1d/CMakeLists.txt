add_executable(sod1d sod1d.cpp)
target_link_libraries(sod1d PRIVATE nest nest_cpu)
target_include_directories(sod1d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

