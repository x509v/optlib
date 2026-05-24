add_executable(optlib_bench
    bench_newton.cpp
    bench_lbfgs.cpp
    bench_cache.cpp
    bench_simd.cpp
)

target_link_libraries(optlib_bench PRIVATE
    optlib::optlib
    benchmark::benchmark
    benchmark::benchmark_main
)

target_compile_options(optlib_bench PRIVATE -O3 -march=native -funroll-loops)

# Separate executable: comparison table + per-method Google Benchmark timing.
# Has a custom main() that prints the markdown table before running benchmarks.
add_executable(optlib_methods bench_methods.cpp)
target_link_libraries(optlib_methods PRIVATE
    optlib::optlib
    benchmark::benchmark
)
target_compile_options(optlib_methods PRIVATE -O3 -march=native -funroll-loops)
