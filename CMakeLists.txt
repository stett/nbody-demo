cmake_minimum_required(VERSION 3.24)

# Cinder's platform_macosx.cmake force-sets CMAKE_OSX_ARCHITECTURES to "arm64;x86_64"
# when it is empty. That happens partway through add_subdirectory(external), so nbody's
# targets are created single-arch and nbody_demo's are created universal -- the link then
# fails with undefined x86_64 symbols. Settle the architecture up front so every target
# in the tree agrees. Must precede project() to apply to compiler detection.
# CMAKE_HOST_SYSTEM_PROCESSOR is still empty this early, so query the host directly.
if (APPLE AND NOT CMAKE_OSX_ARCHITECTURES)
    cmake_host_system_information(RESULT host_arch QUERY OS_PLATFORM)
    set(CMAKE_OSX_ARCHITECTURES "${host_arch}" CACHE STRING "macOS architectures")
endif()

project(nbody_demo)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Stop windows from stomping min and max functions with macros
if (WIN32)
    add_compile_definitions(NOMINMAX)
endif()

# Use the dynamic CRT. Cinder's platform_msw.cmake defaults a static (non-shared) build to
# /MT, but only when this is undefined, so setting it here wins. Must precede
# add_subdirectory(external) to apply to cinder's own targets.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

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
    # runtime library is selected via CMAKE_MSVC_RUNTIME_LIBRARY above
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /ZI")
endif()