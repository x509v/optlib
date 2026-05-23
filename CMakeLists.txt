cmake_minimum_required(VERSION 3.25)
project(optlib VERSION 0.1.0 LANGUAGES CXX)

# ── Standard ────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)       # -std=c++23, not -std=gnu++23

# ── Dependencies via FetchContent ───────────────────────────────────────────
include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# Eigen3 — header-only linear algebra
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
set(EIGEN_BUILD_DOC       OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_BLAS      OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_LAPACK    OFF CACHE BOOL "" FORCE)

# GoogleTest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)
set(gtest_force_shared_crt ON  CACHE BOOL "" FORCE)
set(INSTALL_GTEST          OFF CACHE BOOL "" FORCE)

# Google Benchmark — must disable its internal tests before MakeAvailable
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
    GIT_SHALLOW    TRUE
)
set(BENCHMARK_ENABLE_TESTING     OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL     OFF CACHE BOOL "" FORCE)

# Fetch all at once so CMake can parallelise downloads
FetchContent_MakeAvailable(Eigen3 googletest benchmark)

# ── optlib — header-only INTERFACE target ───────────────────────────────────
add_library(optlib INTERFACE)
add_library(optlib::optlib ALIAS optlib)

target_include_directories(optlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(optlib INTERFACE Eigen3::Eigen)

target_compile_features(optlib INTERFACE cxx_std_23)

# These flags propagate to every target that links optlib::optlib.
# -march=native: enable AVX/AVX2/AVX-512 — do NOT use in distributed builds.
target_compile_options(optlib INTERFACE
    -O3
    -march=native
    -funroll-loops
)

# ── Tests & Benchmarks ───────────────────────────────────────────────────────
include(CTest)
enable_testing()

add_subdirectory(tests)
add_subdirectory(benchmarks)
