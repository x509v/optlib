#include <benchmark/benchmark.h>
#include <optlib/core/types.hpp>

namespace optlib::bench {

// Baseline: measure raw Eigen Cholesky solve (the inner kernel of Newton).
// Replace with actual newton() call once implemented.
static void BM_CholeskyKernel(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    // symmetric positive-definite matrix
    Matrix A = Matrix::Random(n, n);
    A = A.transpose() * A + Matrix::Identity(n, n) * static_cast<Scalar>(n);
    Vector b = Vector::Random(n);
    Vector x(n);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x = A.llt().solve(b));
    }

    state.SetComplexityN(n);
    state.SetLabel("LLT solve (Newton inner step)");
}

BENCHMARK(BM_CholeskyKernel)
    ->RangeMultiplier(2)
    ->Range(8, 512)
    ->Complexity(benchmark::oNCubed)
    ->Unit(benchmark::kMicrosecond);

} // namespace optlib::bench
