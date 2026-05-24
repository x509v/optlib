#include <cmath>
#include <gtest/gtest.h>
#include <optlib/methods/genetic.hpp>

namespace optlib::test {

// ── benchmark functions ──────────────────────────────────────────────────────

// Ackley function — highly multimodal, global minimum f(0,...,0)=0.
// Typical search domain: [-32.768, 32.768]^n; here we use [-5, 5]^n.
static Scalar ackley(const Vector& x) {
    const int n = static_cast<int>(x.size());
    constexpr Scalar a = 20.0, b = 0.2, c = 2.0 * M_PI;
    const Scalar sum_sq  = x.squaredNorm() / n;
    const Scalar sum_cos = x.unaryExpr([c](Scalar xi){ return std::cos(c * xi); }).sum() / n;
    return -a * std::exp(-b * std::sqrt(sum_sq)) - std::exp(sum_cos) + a + std::exp(1.0);
}

// ── tests ────────────────────────────────────────────────────────────────────

TEST(Genetic, Sphere10D) {
    // f(x) = ||x||², global minimum = 0 at origin.
    // Genetic algorithms handle high dimensions poorly in general, but
    // sphere is unimodal so 10D is tractable with sufficient population budget.
    auto f = [](const Vector& x) -> Scalar { return x.squaredNorm(); };

    Vector lo = -5.0 * Vector::Ones(10);
    Vector hi =  5.0 * Vector::Ones(10);

    GeneticConfig cfg;
    cfg.population_size  = 300;
    cfg.max_generations  = 800;
    cfg.elite_count      = 10;
    cfg.mutation_rate    = 0.02;

    auto r = GeneticOptimizer(cfg).minimize(f, 10, lo, hi);

    EXPECT_LT(r.f, 1e-4);
    EXPECT_LT(r.x.norm(), 0.02);
}

TEST(Genetic, Ackley2D) {
    // Ackley is a standard multimodal test function with a single global minimum
    // at the origin (f=0) and many deceptive local minima.
    // GA's global exploration should locate the basin around (0, 0).
    Vector lo = -5.0 * Vector::Ones(2);
    Vector hi =  5.0 * Vector::Ones(2);

    GeneticConfig cfg;
    cfg.population_size  = 400;
    cfg.max_generations  = 1000;
    cfg.elite_count      = 10;
    cfg.mutation_rate    = 0.05;
    cfg.mutation_sigma   = 0.15;
    cfg.crossover_rate   = 0.85;

    auto r = GeneticOptimizer(cfg).minimize(ackley, 2, lo, hi);

    // Ackley at origin = 0; accept anything < 1e-3 (essentially the global minimum)
    EXPECT_LT(r.f, 1e-3);
    EXPECT_LT(r.x.norm(), 0.05);
}

TEST(Genetic, SphereConvergesAtAll) {
    // Sanity check: default config with small dim converges quickly.
    auto f = [](const Vector& x) -> Scalar { return x.squaredNorm(); };
    Vector lo = -3.0 * Vector::Ones(3);
    Vector hi =  3.0 * Vector::Ones(3);

    auto r = GeneticOptimizer().minimize(f, 3, lo, hi);

    EXPECT_LT(r.f, 1e-5);
    EXPECT_GT(r.iterations, 0u);
}

TEST(Genetic, RespectsBounds) {
    // Every returned coordinate must lie in [lo, hi].
    // This also indirectly verifies that mutation and crossover clamp correctly.
    auto f = [](const Vector& x) -> Scalar { return (x - Vector::Ones(4) * 1.5).squaredNorm(); };
    Vector lo = -2.0 * Vector::Ones(4);
    Vector hi =  2.0 * Vector::Ones(4);

    GeneticConfig cfg;
    cfg.max_generations = 100;

    auto r = GeneticOptimizer(cfg).minimize(f, 4, lo, hi);

    for (int j = 0; j < 4; ++j) {
        EXPECT_GE(r.x[j], lo[j] - 1e-12);
        EXPECT_LE(r.x[j], hi[j] + 1e-12);
    }
    // Minimum inside bounds is at x* = 1.5 in every dimension (since 1.5 ≤ 2)
    EXPECT_LT(r.f, 1e-3);
}

TEST(Genetic, ElitismMonotonicFitness) {
    // Mathematical guarantee: with elitism, the best fitness is monotonically
    // non-increasing across generations within a single run.
    //
    // Proof sketch: at each generation the top elite_count individuals are
    // copied unchanged to the next population, so best(gen k+1) ≤ best(gen k).
    //
    // Test method: same seed → the first N gens of any two runs are IDENTICAL
    // (same RNG sequence, same operations).  So the N+M-gen run's first N
    // generations are exactly the N-gen run, and M more elite-preserving steps
    // can only keep or improve the best fitness.
    //
    // max_generations < STAGNATION_PATIENCE=30 prevents early exit, ensuring
    // both runs exhaust all generations and the RNG sequences are aligned.
    auto f = [](const Vector& x) -> Scalar { return x.squaredNorm(); };
    Vector lo = -5.0 * Vector::Ones(4);
    Vector hi =  5.0 * Vector::Ones(4);

    auto make = [&](uint32_t gens) {
        GeneticConfig c;
        c.population_size = 50;
        c.max_generations = gens;
        c.elite_count     = 5;
        c.seed            = 42;
        return GeneticOptimizer(c).minimize(f, 4, lo, hi);
    };

    // First 10 gens of both runs are identical; elitism guarantees monotonicity.
    auto r10 = make(10);
    auto r20 = make(20);
    auto r28 = make(28);  // still < STAGNATION_PATIENCE=30

    EXPECT_LE(r20.f, r10.f + 1e-12);
    EXPECT_LE(r28.f, r20.f + 1e-12);
}

TEST(Genetic, ElitismProtectsUnderHighMutation) {
    // With aggressive mutation (rate=0.5, sigma=0.5×range), solutions are
    // frequently destroyed.  Elitism protects the top individuals from being
    // mutated, which makes a measurable difference over many random seeds.
    //
    // Comparison is fair: max_generations < STAGNATION_PATIENCE=30 so neither
    // run exits early, and the only structural difference is the elite slots.
    // Different elite_count means different RNG consumption, so this is a
    // statistical (not deterministic) test.  We require elite wins ≥ 20/30.
    auto f = [](const Vector& x) -> Scalar { return x.squaredNorm(); };
    Vector lo = -5.0 * Vector::Ones(3);
    Vector hi =  5.0 * Vector::Ones(3);

    int elite_wins = 0;
    for (uint64_t seed = 0; seed < 30; ++seed) {
        auto make = [&](uint32_t elite) {
            GeneticConfig c;
            c.population_size = 40;
            c.max_generations = 20;  // < STAGNATION_PATIENCE
            c.elite_count     = elite;
            c.mutation_rate   = 0.5;
            c.mutation_sigma  = 0.5;  // actual σ = 0.5 × 10 = 5 (large)
            c.seed            = seed;
            return GeneticOptimizer(c).minimize(f, 3, lo, hi);
        };
        if (make(5).f <= make(0).f) ++elite_wins;
    }

    EXPECT_GE(elite_wins, 20);
}

} // namespace optlib::test
