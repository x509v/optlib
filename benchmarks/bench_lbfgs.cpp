#include <benchmark/benchmark.h>
#include <optlib/core/types.hpp>

namespace optlib::bench {

// Baseline: measure the two-loop recursion kernel (core of L-BFGS).
// Approximates H^{-1} q using m (s, y) pairs.
static void BM_TwoLoopKernel(benchmark::State& state) {
    const int n = 1024;
    const int m = static_cast<int>(state.range(0));  // memory size

    std::vector<Vector> s_vecs(m, Vector::Random(n));
    std::vector<Vector> y_vecs(m, Vector::Random(n));
    Vector q = Vector::Random(n);
    Vector r(n);
    std::vector<Scalar> alpha(m), rho(m);

    for (auto _ : state) {
        // forward pass
        r = q;
        for (int i = m - 1; i >= 0; --i) {
            rho[i] = 1.0 / y_vecs[i].dot(s_vecs[i]);
            alpha[i] = rho[i] * s_vecs[i].dot(r);
            r -= alpha[i] * y_vecs[i];
        }
        // backward pass
        for (int i = 0; i < m; ++i) {
            Scalar beta = rho[i] * y_vecs[i].dot(r);
            r += (alpha[i] - beta) * s_vecs[i];
        }
        benchmark::DoNotOptimize(r.data());
    }

    state.SetComplexityN(m);
    state.SetLabel("L-BFGS two-loop recursion (n=1024)");
}

BENCHMARK(BM_TwoLoopKernel)
    ->RangeMultiplier(2)
    ->Range(1, 64)
    ->Complexity(benchmark::oN)
    ->Unit(benchmark::kMicrosecond);

} // namespace optlib::bench
