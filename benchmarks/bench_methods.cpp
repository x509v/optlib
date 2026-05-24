#include <benchmark/benchmark.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include <optlib/optlib.hpp>
#include "functions.hpp"

// ── Method comparison + Google Benchmark ─────────────────────────────────────
//
// Running:
//   ./optlib_methods                         # comparison table only
//   ./optlib_methods --benchmark_filter=.    # table + all benchmarks
//   ./optlib_methods --benchmark_filter=^$   # table only (skip benchmarks)

namespace optlib::bench {

// ── Result record ─────────────────────────────────────────────────────────────

struct RunResult {
    const char* func;
    const char* method;
    int         n;
    bool        converged;
    uint32_t    iterations;
    int         fg_calls;    // operator() calls (value + gradient)
    int         hess_calls;  // hessian() calls (Newton only; 0 otherwise)
    double      grad_norm;   // ‖∇f‖ at termination (0.0 for Genetic)
    double      f_val;       // f(x*) at termination
    double      ms;          // wall-clock milliseconds
};

// ── Timing helper ─────────────────────────────────────────────────────────────

using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ── Per-method runners ────────────────────────────────────────────────────────

static OptimizeConfig make_grad_cfg(int max_iter = 2000) {
    OptimizeConfig c;
    c.max_iter  = static_cast<uint32_t>(max_iter);
    c.tol_grad  = 1e-6;
    c.tol_step  = 1e-12;
    c.verbose   = false;
    return c;
}

template<typename F>
static RunResult run_newton(const char* fname, F& f, const Vector& x0) {
    f.reset();
    auto cfg = make_grad_cfg();
    auto t0  = Clock::now();
    auto r   = NewtonOptimizer(cfg).minimize(f, x0);
    return {fname, "Newton", f.n, r.converged, r.iterations,
            f.calls, f.hess_calls, r.grad_norm, r.f, elapsed_ms(t0)};
}

template<typename F>
static RunResult run_lbfgs(const char* fname, F& f, const Vector& x0, int m) {
    f.reset();
    auto cfg = make_grad_cfg();
    char mbuf[16]; std::snprintf(mbuf, sizeof mbuf, "LBFGS m=%-2d", m);
    static std::string name;    // persists until next call — safe for table printing
    name = mbuf;
    auto t0 = Clock::now();
    auto r  = LBFGSOptimizer(cfg, static_cast<uint32_t>(m)).minimize(f, x0);
    return {fname, name.c_str(), f.n, r.converged, r.iterations,
            f.calls, 0, r.grad_norm, r.f, elapsed_ms(t0)};
}

// Genetic wraps the FunctionWithGrad as a ScalarFunction (extracts .first).
// The call counter still fires since the lambda calls f.operator().
template<typename F>
static RunResult run_genetic(const char* fname, F& f, int n) {
    f.reset();

    GeneticConfig gc;
    gc.population_size = (n <= 2) ? 200u : (n <= 10) ? 300u : 500u;
    gc.max_generations = (n <= 2) ? 500u : (n <= 10) ? 1500u : 2000u;
    gc.elite_count     = gc.population_size / 20;
    gc.mutation_rate   = 0.02;
    gc.seed            = 42;

    Vector lo = -5.0 * Vector::Ones(n);
    Vector hi =  5.0 * Vector::Ones(n);

    auto scalar_f = [&f](const Vector& x) -> Scalar { return f(x).first; };

    auto t0 = Clock::now();
    auto r  = GeneticOptimizer(gc).minimize(scalar_f, n, lo, hi);
    // grad_norm is not computed by Genetic; report 0.0
    return {fname, "Genetic", n, r.converged, r.iterations,
            f.calls, 0, 0.0, r.f, elapsed_ms(t0)};
}

// ── Table helpers ─────────────────────────────────────────────────────────────

static void print_header() {
    std::puts("| Function   | n  | Method       | Conv | Iters |  f+g  |  H  | "
              "   ||g||   |    f*      | ms     |");
    std::puts("|:-----------|:--:|:-------------|:----:|------:|------:|----:|"
              "----------:|-----------:|-------:|");
}

static void print_row(const RunResult& r) {
    // grad_norm: "N/A" for Genetic (returns 0.0 with converged status from stagnation)
    char gnbuf[16];
    if (r.hess_calls == 0 && std::string(r.method) == "Genetic")
        std::snprintf(gnbuf, sizeof gnbuf, "       N/A");
    else
        std::snprintf(gnbuf, sizeof gnbuf, "%10.2e", r.grad_norm);

    std::printf("| %-10s | %2d | %-12s | %-4s | %5u | %5d | %3d | %s | %10.3e | %6.2f |\n",
                r.func, r.n, r.method,
                r.converged ? "yes" : "no ",
                r.iterations, r.fg_calls, r.hess_calls,
                gnbuf, r.f_val, r.ms);
}

// ── Comparison table ──────────────────────────────────────────────────────────

static void run_comparison_table() {
    std::puts("# Optimization Method Comparison\n");
    std::puts("Convergence criterion for gradient methods: ‖∇f‖ < 1e-6.");
    std::puts("Genetic: stagnation (< 1e-9 improvement for 30 generations) or max_gen reached.\n");
    std::puts("Starting points:  Sphere/Quadratic → x₀ = 3·1  (or 1·1 for Quadratic)");
    std::puts("                  Rosenbrock       → x₀ = −1\n");

    print_header();

    for (int n : {2, 10, 50}) {
        // ── Sphere ───────────────────────────────────────────────────────────
        {
            Sphere f(n); Vector x0 = 3.0 * Vector::Ones(n);
            print_row(run_newton("Sphere", f, x0));
        }
        {
            Sphere f(n); Vector x0 = 3.0 * Vector::Ones(n);
            print_row(run_lbfgs("Sphere", f, x0, 5));
        }
        {
            Sphere f(n); Vector x0 = 3.0 * Vector::Ones(n);
            print_row(run_lbfgs("Sphere", f, x0, 20));
        }
        {
            Sphere f(n);
            print_row(run_genetic("Sphere", f, n));
        }

        // ── Quadratic ────────────────────────────────────────────────────────
        {
            Quadratic f(n); Vector x0 = Vector::Ones(n);
            print_row(run_newton("Quadratic", f, x0));
        }
        {
            Quadratic f(n); Vector x0 = Vector::Ones(n);
            print_row(run_lbfgs("Quadratic", f, x0, 5));
        }
        {
            Quadratic f(n); Vector x0 = Vector::Ones(n);
            print_row(run_lbfgs("Quadratic", f, x0, 20));
        }
        {
            Quadratic f(n);
            print_row(run_genetic("Quadratic", f, n));
        }

        // ── Rosenbrock ───────────────────────────────────────────────────────
        {
            Rosenbrock f(n); Vector x0 = -Vector::Ones(n);
            print_row(run_newton("Rosenbrock", f, x0));
        }
        {
            Rosenbrock f(n); Vector x0 = -Vector::Ones(n);
            print_row(run_lbfgs("Rosenbrock", f, x0, 5));
        }
        {
            Rosenbrock f(n); Vector x0 = -Vector::Ones(n);
            print_row(run_lbfgs("Rosenbrock", f, x0, 20));
        }
        {
            Rosenbrock f(n);
            print_row(run_genetic("Rosenbrock", f, n));
        }

        if (n < 50) std::puts("|            |    |              |      |       |       |     |"
                              "           |            |        |");
    }
    std::putchar('\n');
}

// ── Google Benchmark: time per full optimization ──────────────────────────────
//
// Each benchmark::State iteration = one complete minimization from x0.
// DoNotOptimize on the result prevents the compiler from eliding the whole call.

// Newton ─────────────────────────────────────────────────────────────────────

static void BM_Newton_Sphere(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const OptimizeConfig cfg = make_grad_cfg();
    for (auto _ : state) {
        Sphere f(n);
        auto r = NewtonOptimizer(cfg).minimize(f, 3.0 * Vector::Ones(n));
        benchmark::DoNotOptimize(r.f);
    }
}

static void BM_Newton_Quadratic(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const OptimizeConfig cfg = make_grad_cfg();
    for (auto _ : state) {
        Quadratic f(n);
        auto r = NewtonOptimizer(cfg).minimize(f, Vector::Ones(n));
        benchmark::DoNotOptimize(r.f);
    }
}

static void BM_Newton_Rosenbrock(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const OptimizeConfig cfg = make_grad_cfg();
    for (auto _ : state) {
        Rosenbrock f(n);
        auto r = NewtonOptimizer(cfg).minimize(f, -Vector::Ones(n));
        benchmark::DoNotOptimize(r.f);
    }
}

// LBFGS ──────────────────────────────────────────────────────────────────────

static void BM_LBFGS5_Sphere(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const OptimizeConfig cfg = make_grad_cfg();
    for (auto _ : state) {
        Sphere f(n);
        auto r = LBFGSOptimizer(cfg, 5).minimize(f, 3.0 * Vector::Ones(n));
        benchmark::DoNotOptimize(r.f);
    }
}

static void BM_LBFGS5_Rosenbrock(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const OptimizeConfig cfg = make_grad_cfg();
    for (auto _ : state) {
        Rosenbrock f(n);
        auto r = LBFGSOptimizer(cfg, 5).minimize(f, -Vector::Ones(n));
        benchmark::DoNotOptimize(r.f);
    }
}

static void BM_LBFGS20_Rosenbrock(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const OptimizeConfig cfg = make_grad_cfg();
    for (auto _ : state) {
        Rosenbrock f(n);
        auto r = LBFGSOptimizer(cfg, 20).minimize(f, -Vector::Ones(n));
        benchmark::DoNotOptimize(r.f);
    }
}

// Genetic ────────────────────────────────────────────────────────────────────

static void BM_Genetic_Sphere(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    GeneticConfig gc;
    gc.population_size = (n <= 2) ? 200u : 300u;
    gc.max_generations = (n <= 2) ? 300u : 800u;
    gc.elite_count     = gc.population_size / 20;
    gc.seed            = 42;

    Vector lo = -5.0 * Vector::Ones(n);
    Vector hi =  5.0 * Vector::Ones(n);

    for (auto _ : state) {
        Sphere f(n);
        auto sf = [&f](const Vector& x) -> Scalar { return f(x).first; };
        auto r  = GeneticOptimizer(gc).minimize(sf, n, lo, hi);
        benchmark::DoNotOptimize(r.f);
    }
}

// Registration ────────────────────────────────────────────────────────────────
//
// args: n ∈ {2, 10, 50}

#define SIZES ->Arg(2)->Arg(10)->Arg(50)->Unit(benchmark::kMillisecond)

BENCHMARK(BM_Newton_Sphere)       SIZES;
BENCHMARK(BM_Newton_Quadratic)    SIZES;
BENCHMARK(BM_Newton_Rosenbrock)   SIZES;
BENCHMARK(BM_LBFGS5_Sphere)       SIZES;
BENCHMARK(BM_LBFGS5_Rosenbrock)   SIZES;
BENCHMARK(BM_LBFGS20_Rosenbrock)  SIZES;
BENCHMARK(BM_Genetic_Sphere)      ->Arg(2)->Arg(10)->Unit(benchmark::kMillisecond);

#undef SIZES

} // namespace optlib::bench

// ── Custom main: table first, then benchmarks ─────────────────────────────────

int main(int argc, char** argv) {
    optlib::bench::run_comparison_table();
    std::fflush(stdout);

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
