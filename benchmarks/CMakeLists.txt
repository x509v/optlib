add_executable(optlib_bench
    bench_newton.cpp
    bench_lbfgs.cpp
    bench_cache.cpp
)

target_link_libraries(optlib_bench PRIVATE
    optlib::optlib
    benchmark::benchmark
    benchmark::benchmark_main
)

target_compile_options(optlib_bench PRIVATE -O3 -march=native -funroll-loops)
