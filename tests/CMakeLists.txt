add_executable(optlib_tests
    test_core.cpp
    test_newton.cpp
    test_lbfgs.cpp
    test_genetic.cpp
    test_linesearch.cpp
    test_simd.cpp
)

target_link_libraries(optlib_tests PRIVATE
    optlib::optlib
    GTest::gtest
    GTest::gtest_main
)

target_compile_options(optlib_tests PRIVATE -O3 -march=native -funroll-loops)

include(GoogleTest)
gtest_discover_tests(optlib_tests)
