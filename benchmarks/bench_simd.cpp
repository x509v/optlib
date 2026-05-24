#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include <optlib/core/simd.hpp>
#include <optlib/core/types.hpp>

// ── SIMD dot-product benchmarks ───────────────────────────────────────────────
//
// Four variants measured across n ∈ {64, 256, 1024, 4096, 16384}:
//   scalar_dot    — naive C++ loop (auto-vectorized by the compiler at -O3)
//   eigen_dot     — Eigen expr template (AVX-512 on this machine)
//   avx_dot       — manual AVX2, 1 accumulator (latency-limited)
//   avx_dot_unrolled — manual AVX2, 4 accumulators (ILP-optimal)
//
// Counters reported per benchmark row:
//   GFLOPS  — effective giga-FLOPS/sec  = 2n / time_per_iter / 1e9
//   GB/s    — effective memory bandwidth = 2*n*8 / time_per_iter / 1e9
//
// ── Roofline (i5-1135G7 @ 4.2 GHz, from bench_cache.cpp measurements) ────────
//
//   Arithmetic intensity = 2n / (16n) = 0.125 FLOP/byte  → memory-bound at DRAM
//
//   Compute ceilings (single core):
//     AVX2 + FMA, 1 acc  :  6.7 GFLOPS   (1 FMA/cycle @ lat=5)
//     AVX2 + FMA, 4 acc  : 33.6 GFLOPS   (ILP-optimal)
//     AVX-512 + FMA peak : 67.2 GFLOPS
//
//   Memory ceilings (AI = 0.125):
//     L1  ≈ 300 GB/s  →  37.5 GFLOPS   n ≤ ~3 000
//     L2  ≈ 160 GB/s  →  20.0 GFLOPS   n ≤ ~80 000
//     DRAM ≈ 51 GB/s  →   6.4 GFLOPS   large n
//
//   Expected observations:
//     • scalar/eigen/avx_unrolled approach 37 GFLOPS for n=64 (L1, compute-lim)
//     • avx_dot (1 acc) stays near 6–7 GFLOPS regardless of n (latency-limited)
//     • all versions drop to ~20 GFLOPS at n=4096 (L2, memory-limited)
//     • at n=16384 all converge near the L2 ceiling (~20 GFLOPS)
//
// Run:
//   ./optlib_bench --benchmark_filter="BM_Dot" \
//                  --benchmark_counters_tabular=true

namespace optlib::bench {

using simd::scalar_dot;
using simd::eigen_dot;
using simd::avx_dot;
using simd::avx_dot_unrolled;

// ── fixture: pre-allocated random vectors ────────────────────────────────────

struct DotFixture {
    std::vector<double> a, b;
    Vector              ea, eb;  // Eigen views for eigen_dot

    explicit DotFixture(int n) : a(n), b(n), ea(n), eb(n) {
        std::mt19937_64 rng(0xC0FFEE);
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        for (int i = 0; i < n; ++i) {
            a[i]  = d(rng);
            b[i]  = d(rng);
            ea[i] = a[i];
            eb[i] = b[i];
        }
    }
};

// ── counter helper ────────────────────────────────────────────────────────────

static void set_counters(benchmark::State& state, int n) {
    // kIsIterationInvariantRate: displayed = value / (total_time / iterations)
    //   = value * (iterations / total_time) = value * iter_rate
    const double flops_per_iter = 2.0 * n;
    const double bytes_per_iter = 2.0 * n * sizeof(double);
    state.counters["GFLOPS"] = benchmark::Counter(
        flops_per_iter / 1e9,
        benchmark::Counter::kIsIterationInvariantRate);
    state.counters["GB/s"] = benchmark::Counter(
        bytes_per_iter / 1e9,
        benchmark::Counter::kIsIterationInvariantRate);
    state.SetBytesProcessed(
        static_cast<int64_t>(bytes_per_iter) * state.iterations());
}

// ── Benchmark 1: scalar (compiler-vectorized) ─────────────────────────────────

static void BM_Dot_Scalar(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    DotFixture fix(n);
    double result = 0;
    for (auto _ : state)
        benchmark::DoNotOptimize(result = scalar_dot(fix.a.data(), fix.b.data(), n));
    set_counters(state, n);
}

// ── Benchmark 2: Eigen ────────────────────────────────────────────────────────

static void BM_Dot_Eigen(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    DotFixture fix(n);
    double result = 0;
    for (auto _ : state)
        benchmark::DoNotOptimize(result = eigen_dot(fix.ea, fix.eb));
    set_counters(state, n);
}

// ── Benchmark 3: manual AVX2 (1 accumulator, latency-limited) ─────────────────

static void BM_Dot_AVX(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    DotFixture fix(n);
    double result = 0;
    for (auto _ : state)
        benchmark::DoNotOptimize(result = avx_dot(fix.a.data(), fix.b.data(), n));
    set_counters(state, n);
    state.SetLabel("AVX2 1-acc");
}

// ── Benchmark 4: manual AVX2 (4 accumulators, ILP-optimal) ────────────────────

static void BM_Dot_AVX_Unrolled(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    DotFixture fix(n);
    double result = 0;
    for (auto _ : state)
        benchmark::DoNotOptimize(result = avx_dot_unrolled(fix.a.data(), fix.b.data(), n));
    set_counters(state, n);
    state.SetLabel("AVX2 4-acc unrolled");
}

// ── Registration: n ∈ {64, 256, 1024, 4096, 16384} ───────────────────────────

#define DOT_SIZES \
    ->Arg(64)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384) \
    ->Unit(benchmark::kNanosecond)

BENCHMARK(BM_Dot_Scalar)      DOT_SIZES;
BENCHMARK(BM_Dot_Eigen)       DOT_SIZES;
BENCHMARK(BM_Dot_AVX)         DOT_SIZES;
BENCHMARK(BM_Dot_AVX_Unrolled)DOT_SIZES;

#undef DOT_SIZES

} // namespace optlib::bench
