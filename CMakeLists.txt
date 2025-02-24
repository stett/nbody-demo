cmake_minimum_required(VERSION 3.5)
project(nbody_demo)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use GPU acceleration
#set(nbody_gpu ON)
add_compile_definitions(NBODY_GPU=1)

# Stop windows from stomping min and max functions with macros
if (WIN32)
    add_compile_definitions(NOMINMAX)
endif()

# add dependencies
add_subdirectory(external)

# add nbody library
file(GLOB_RECURSE sources "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp")
add_executable(nbody_demo ${sources})
target_include_directories(nbody_demo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/source")
target_link_libraries(nbody_demo PUBLIC nbody)
target_link_libraries(nbody_demo PUBLIC cinder)
set_target_properties(nbody_demo PROPERTIES WIN32_EXECUTABLE TRUE)
set_target_properties(nbody_demo PROPERTIES LINKER_LANGUAGE CXX)
target_compile_features(nbody_demo PRIVATE cxx_std_20)
if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE} /MTd")
endif()