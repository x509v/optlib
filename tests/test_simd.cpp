#include <gtest/gtest.h>
#include <optlib/core/simd.hpp>
#include <optlib/core/types.hpp>

#include <cmath>
#include <random>
#include <vector>

namespace optlib::test {

// ── helpers ──────────────────────────────────────────────────────────────────

struct DotVectors {
    std::vector<double> a, b;
    Vector ea, eb;

    DotVectors(int n, uint64_t seed = 42) : a(n), b(n), ea(n), eb(n) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        for (int i = 0; i < n; ++i) {
            a[i] = d(rng); b[i] = d(rng);
            ea[i] = a[i];  eb[i] = b[i];
        }
    }
};

// ── correctness: all three agree within 1e-10 ────────────────────────────────

class SimdDotCorrectness : public ::testing::TestWithParam<int> {};

TEST_P(SimdDotCorrectness, AllVersionsAgree) {
    const int n = GetParam();
    DotVectors v(n);

    const double ref  = simd::scalar_dot(v.a.data(), v.b.data(), n);
    const double eig  = simd::eigen_dot(v.ea, v.eb);
    const double avx  = simd::avx_dot(v.a.data(), v.b.data(), n);
    const double avxu = simd::avx_dot_unrolled(v.a.data(), v.b.data(), n);

    // Differences arise from floating-point reassociation and FMA-vs-two-rounds.
    // For random data in [-1,1] and n ≤ 16384: accumulated error ≤ n × ε ≈ 3e-12.
    EXPECT_NEAR(eig,  ref, 1e-10) << "Eigen vs scalar, n=" << n;
    EXPECT_NEAR(avx,  ref, 1e-10) << "avx_dot vs scalar, n=" << n;
    EXPECT_NEAR(avxu, ref, 1e-10) << "avx_dot_unrolled vs scalar, n=" << n;
}

INSTANTIATE_TEST_SUITE_P(
    Sizes, SimdDotCorrectness,
    ::testing::Values(1, 3, 4, 5, 7, 15, 16, 17,
                      64, 100, 255, 256, 257,
                      1024, 4096, 16384));

// ── edge cases ────────────────────────────────────────────────────────────────

TEST(SimdDot, LengthOne) {
    double a = 3.0, b = 4.0;
    EXPECT_DOUBLE_EQ(simd::scalar_dot(&a, &b, 1), 12.0);
    EXPECT_DOUBLE_EQ(simd::avx_dot(&a, &b, 1),    12.0);
    EXPECT_DOUBLE_EQ(simd::avx_dot_unrolled(&a, &b, 1), 12.0);
}

TEST(SimdDot, ZeroLength) {
    EXPECT_DOUBLE_EQ(simd::scalar_dot(nullptr, nullptr, 0), 0.0);
    EXPECT_DOUBLE_EQ(simd::avx_dot(nullptr, nullptr, 0),    0.0);
    EXPECT_DOUBLE_EQ(simd::avx_dot_unrolled(nullptr, nullptr, 0), 0.0);
}

TEST(SimdDot, OrthogonalVectors) {
    // {1,0,0} · {0,1,0} = 0 exactly
    std::vector<double> a = {1.0, 0.0, 0.0};
    std::vector<double> b = {0.0, 1.0, 0.0};
    EXPECT_DOUBLE_EQ(simd::avx_dot(a.data(), b.data(), 3), 0.0);
    EXPECT_DOUBLE_EQ(simd::avx_dot_unrolled(a.data(), b.data(), 3), 0.0);
}

TEST(SimdDot, UnitVectorNormSquared) {
    // ||e_1||^2 via dot with itself = 1.0
    const int n = 128;
    std::vector<double> a(n, 0.0);
    a[0] = 1.0;
    EXPECT_DOUBLE_EQ(simd::avx_dot(a.data(), a.data(), n), 1.0);
    EXPECT_DOUBLE_EQ(simd::avx_dot_unrolled(a.data(), a.data(), n), 1.0);
}

TEST(SimdDot, TailHandledCorrectly) {
    // Non-multiples of 4 and 16: the tail loop must handle residual elements.
    for (int n : {1, 2, 3, 5, 6, 7, 9, 13, 15, 17, 19, 31}) {
        DotVectors v(n, /*seed=*/n * 7919u);
        const double ref  = simd::scalar_dot(v.a.data(), v.b.data(), n);
        EXPECT_NEAR(simd::avx_dot(v.a.data(), v.b.data(), n),         ref, 1e-12)
            << "avx_dot tail, n=" << n;
        EXPECT_NEAR(simd::avx_dot_unrolled(v.a.data(), v.b.data(), n), ref, 1e-12)
            << "avx_dot_unrolled tail, n=" << n;
    }
}

TEST(SimdDot, MultipleSeeds) {
    // Verify across 10 random seeds that avx and scalar agree.
    for (uint64_t seed = 0; seed < 10; ++seed) {
        DotVectors v(1000, seed);
        const double ref = simd::scalar_dot(v.a.data(), v.b.data(), 1000);
        EXPECT_NEAR(simd::avx_dot(v.a.data(), v.b.data(), 1000),          ref, 1e-10);
        EXPECT_NEAR(simd::avx_dot_unrolled(v.a.data(), v.b.data(), 1000), ref, 1e-10);
    }
}

} // namespace optlib::test
