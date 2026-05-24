#pragma once

// SIMD dot-product implementations: scalar, Eigen, and manual AVX2.
//
// All three share the same logical computation:
//   result = Σ a[i] * b[i]  for i in [0, n)
//
// Roofline analysis (i5-1135G7 @ 4.2 GHz, measured bandwidths):
//
//   Arithmetic intensity = 2n FLOPS / (2 × n × 8 bytes) = 0.125 FLOP/byte
//
//   Compute ceilings (single core):
//     AVX2  + FMA, 1 accumulator : ~6.7 GFLOPS   (latency-limited, lat=5 cy)
//     AVX2  + FMA, ILP-optimal   : 33.6 GFLOPS   (4 doubles × 2 FLOPS × 4.2 GHz)
//     AVX-512 + FMA, ILP-optimal : 67.2 GFLOPS
//
//   Memory ceilings (AI = 0.125):
//     L1  ~300 GB/s → 37.5 GFLOPS   (n ≤ ~3 000: all data in L1)
//     L2  ~160 GB/s → 20.0 GFLOPS   (n ≤ ~80 000)
//     DRAM ~ 51 GB/s → 6.4 GFLOPS   (n > ~80 000)
//
//   Crossover point (compute ↔ memory): intensity > 33.6/300 ≈ 0.11 for L1,
//   so dot product is just barely compute-limited in L1 with full ILP.
//   A single accumulator is always latency-limited (6.7 < 6.4 … 37.5).

#include <optlib/core/types.hpp>

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace optlib::simd {

// ── 1. Scalar ─────────────────────────────────────────────────────────────────
// Written in plain C++; with -O3 -march=native the compiler will auto-vectorize
// this to AVX2 or AVX-512, so it is NOT necessarily scalar at the instruction
// level — it is just what you get for free without any programmer effort.
[[nodiscard]] inline double scalar_dot(const double* __restrict__ a,
                                       const double* __restrict__ b,
                                       int n) noexcept {
    double acc = 0.0;
    for (int i = 0; i < n; ++i)
        acc += a[i] * b[i];
    return acc;
}

// ── 2. Eigen ──────────────────────────────────────────────────────────────────
// Delegates to Eigen's expression-template evaluator which emits the widest
// available SIMD (AVX-512 on this machine) with multiple loop unrolling.
[[nodiscard]] inline double eigen_dot(const Vector& a, const Vector& b) noexcept {
    return a.dot(b);
}

// ── 3. AVX2 ──────────────────────────────────────────────────────────────────
#if defined(__AVX2__)

// Manual 256-bit AVX2 dot product.
//
// Inner loop: 4 doubles per iteration (one 256-bit register lane)
//   load  a[i..i+3]  → __m256d
//   load  b[i..i+3]  → __m256d
//   FMA:  sum += va * vb          (requires __FMA__; falls back to mul+add)
//
// Horizontal reduction via _mm256_hadd_pd:
//   sum  = {s0, s1, s2, s3}
//   hadd(sum, sum) = {s0+s1, s0+s1, s2+s3, s2+s3}
//   extract 128-bit halves, add → scalar total
//
// Performance note: a SINGLE accumulator is latency-limited (FMA latency 5 cy
// on Tiger Lake → throughput = 4×2/5 × 4.2 GHz ≈ 6.7 GFLOPS).  To approach
// the AVX2 ILP-optimal peak (33.6 GFLOPS), unroll with ≥5 independent
// accumulator registers (see bench_simd.cpp for the 4-accumulator variant).
//
// Unaligned loads (_mm256_loadu_pd) handle arbitrary input alignment; callers
// with guaranteed 32-byte alignment may substitute _mm256_load_pd for ~0-5%
// extra throughput on older microarchitectures.

[[nodiscard]] inline double avx_dot(const double* __restrict__ a,
                                    const double* __restrict__ b,
                                    int n) noexcept {
    __m256d sum = _mm256_setzero_pd();

    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
#if defined(__FMA__)
        sum = _mm256_fmadd_pd(va, vb, sum);   // sum += va * vb, single rounding
#else
        sum = _mm256_add_pd(sum, _mm256_mul_pd(va, vb));
#endif
    }

    // ── Horizontal reduction ──────────────────────────────────────────────────
    // hadd_pd operates on two 128-bit halves independently:
    //   upper 128: {s2+s3, s2+s3}
    //   lower 128: {s0+s1, s0+s1}
    __m256d tmp  = _mm256_hadd_pd(sum, sum);
    __m128d lo   = _mm256_castpd256_pd128(tmp);     // {s0+s1, s0+s1}
    __m128d hi   = _mm256_extractf128_pd(tmp, 1);   // {s2+s3, s2+s3}
    double result = _mm_cvtsd_f64(_mm_add_pd(lo, hi)); // s0+s1+s2+s3

    // ── Tail: elements not covered by the 4-wide loop ────────────────────────
    for (; i < n; ++i)
        result += a[i] * b[i];

    return result;
}

// 4-accumulator unrolled variant: hides FMA latency by issuing 4 independent
// chains simultaneously (4 × lat / throughput = 4 × 5 / 1 = 20-cycle window).
// Approaches the ILP-optimal throughput of 33.6 GFLOPS.
[[nodiscard]] inline double avx_dot_unrolled(const double* __restrict__ a,
                                             const double* __restrict__ b,
                                             int n) noexcept {
    __m256d s0 = _mm256_setzero_pd();
    __m256d s1 = _mm256_setzero_pd();
    __m256d s2 = _mm256_setzero_pd();
    __m256d s3 = _mm256_setzero_pd();

    int i = 0;
    for (; i + 16 <= n; i += 16) {
#if defined(__FMA__)
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i   ), _mm256_loadu_pd(b+i   ), s0);
        s1 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i+ 4), _mm256_loadu_pd(b+i+ 4), s1);
        s2 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i+ 8), _mm256_loadu_pd(b+i+ 8), s2);
        s3 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i+12), _mm256_loadu_pd(b+i+12), s3);
#else
        s0 = _mm256_add_pd(s0, _mm256_mul_pd(_mm256_loadu_pd(a+i   ), _mm256_loadu_pd(b+i   )));
        s1 = _mm256_add_pd(s1, _mm256_mul_pd(_mm256_loadu_pd(a+i+ 4), _mm256_loadu_pd(b+i+ 4)));
        s2 = _mm256_add_pd(s2, _mm256_mul_pd(_mm256_loadu_pd(a+i+ 8), _mm256_loadu_pd(b+i+ 8)));
        s3 = _mm256_add_pd(s3, _mm256_mul_pd(_mm256_loadu_pd(a+i+12), _mm256_loadu_pd(b+i+12)));
#endif
    }
    // Merge 4 accumulators
    __m256d sum = _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3));

    // Horizontal reduction (same as avx_dot)
    __m256d tmp  = _mm256_hadd_pd(sum, sum);
    __m128d lo   = _mm256_castpd256_pd128(tmp);
    __m128d hi   = _mm256_extractf128_pd(tmp, 1);
    double result = _mm_cvtsd_f64(_mm_add_pd(lo, hi));

    // Tail: handle remaining elements (< 16) one by one
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}

#else // ── Non-AVX2 fallback ──────────────────────────────────────────────────

[[nodiscard]] inline double avx_dot(const double* a, const double* b,
                                    int n) noexcept {
    return scalar_dot(a, b, n);   // compiler may still auto-vectorize
}

[[nodiscard]] inline double avx_dot_unrolled(const double* a, const double* b,
                                             int n) noexcept {
    return scalar_dot(a, b, n);
}

#endif // __AVX2__

} // namespace optlib::simd
