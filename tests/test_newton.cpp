#include <gtest/gtest.h>
#include <optlib/methods/newton.hpp>
#include <cmath>

// All test functions must satisfy FunctionWithHessian:
//   operator()(const Vector&) -> std::pair<Scalar, Vector>
//   hessian(const Vector&)    -> Matrix

namespace optlib::test {

// ── Test functions ────────────────────────────────────────────────────────────

// f(x) = 0.5 * x^T diag(a) x,   minimum at x* = 0
// Newton direction d = −x  →  reaches x* in exactly one step.
// κ(H) = a_max / a_min — tests high condition number separately.
struct DiagonalQuadratic {
    Vector a;  // eigenvalues of A

    std::pair<Scalar, Vector> operator()(const Vector& x) const {
        // f = 0.5 * sum_i a_i * x_i^2,   g = A*x
        return {0.5 * (a.array() * x.array().square()).sum(),
                a.array() * x.array()};
    }

    Matrix hessian(const Vector& /*x*/) const {
        return a.asDiagonal();
    }
};

// f(x,y) = (1−x)² + 100(y−x²)²  — classic nonlinear test.
// Unique minimum at (1,1), Hessian PD almost everywhere on path from (0,0).
struct Rosenbrock {
    std::pair<Scalar, Vector> operator()(const Vector& x) const {
        const Scalar da = 1.0 - x[0];
        const Scalar db = x[1] - x[0] * x[0];
        Vector g(2);
        g[0] = -2.0 * da - 400.0 * x[0] * db;
        g[1] =  200.0 * db;
        return {da * da + 100.0 * db * db, g};
    }

    Matrix hessian(const Vector& x) const {
        const Scalar db = x[1] - x[0] * x[0];
        Matrix H(2, 2);
        H(0, 0) = 2.0 - 400.0 * db + 800.0 * x[0] * x[0];
        H(0, 1) = H(1, 0) = -400.0 * x[0];
        H(1, 1) = 200.0;
        return H;
    }
};

// f(x,y) = x² − y²  — indefinite quadratic with saddle at (0,0).
// H = diag(2, −2): λ_min = −2 → μ ≈ 2 → H_mod ≈ diag(4, 1e-8).
// Newton direction is strongly upward in y, escaping the saddle.
struct Saddle {
    std::pair<Scalar, Vector> operator()(const Vector& x) const {
        Vector g(2);
        g[0] =  2.0 * x[0];
        g[1] = -2.0 * x[1];
        return {x[0] * x[0] - x[1] * x[1], g};
    }

    Matrix hessian(const Vector& /*x*/) const {
        Matrix H = Matrix::Zero(2, 2);
        H(0, 0) =  2.0;
        H(1, 1) = -2.0;
        return H;
    }
};

// ── Quadratic: converges in exactly 1 iteration ───────────────────────────────

TEST(Newton, QuadraticConvergesInExactlyOneIteration) {
    // H = diag(1, 5, 10) is SPD →  μ = 0  →  pure Newton step d = −x₀
    // phi(α) = (1−α)²·f(x₀).  Armijo accepts α=1 → x₁ = 0 → ‖g₁‖ = 0.
    DiagonalQuadratic f;
    f.a.resize(3);
    f.a << 1.0, 5.0, 10.0;

    Vector x0(3);
    x0 << 3.0, -2.0, 1.5;

    OptimizeConfig cfg;
    cfg.tol_grad = 1e-12;

    NewtonOptimizer opt(cfg);
    auto r = opt.minimize(f, x0);

    EXPECT_TRUE(r.converged);
    EXPECT_EQ(r.iterations, 1u)
        << "Newton should reach the exact minimum in one step for a quadratic";
    EXPECT_LT(r.x.norm(), 1e-10) << "solution should be at origin";
    EXPECT_LT(r.f, 1e-20)       << "f(x*) should be (≈) 0";
}

TEST(Newton, QuadraticAlreadyAtMinimum) {
    // Starting at x* = 0: gradient is zero → should return at k=0.
    DiagonalQuadratic f;
    f.a.resize(2);
    f.a << 3.0, 7.0;

    Vector x0 = Vector::Zero(2);

    NewtonOptimizer opt(OptimizeConfig{});
    auto r = opt.minimize(f, x0);

    EXPECT_TRUE(r.converged);
    EXPECT_EQ(r.iterations, 0u) << "already at minimum: should converge at k=0";
    EXPECT_LT(r.grad_norm, 1e-15);
}

TEST(Newton, QuadraticHighConditionNumber) {
    // κ = 1e4:  μ should remain 0 (H still SPD), still converges in 1 step.
    DiagonalQuadratic f;
    f.a.resize(2);
    f.a << 1.0, 1e4;

    Vector x0(2);
    x0 << 5.0, 5.0;

    NewtonOptimizer opt(OptimizeConfig{.tol_grad = 1e-10});
    auto r = opt.minimize(f, x0);

    EXPECT_TRUE(r.converged);
    EXPECT_EQ(r.iterations, 1u) << "quadratic → 1 Newton step regardless of κ";
    EXPECT_LT(r.x.norm(), 1e-8);
}

// ── Rosenbrock: convergence within 20 iterations from (0,0) ─────────────────

TEST(Newton, RosenbrockConvergesWithin20Iters) {
    // The banana valley is challenging for gradient descent (κ≫1) but
    // Newton handles curvature directly: typically converges in ~10 iters.
    Rosenbrock f;
    Vector x0(2);
    x0 << 0.0, 0.0;

    OptimizeConfig cfg;
    cfg.max_iter = 100;
    cfg.tol_grad = 1e-8;

    NewtonOptimizer opt(cfg);
    auto r = opt.minimize(f, x0);

    EXPECT_TRUE(r.converged);
    EXPECT_LE(r.iterations, 20u)
        << "Newton + Armijo on Rosenbrock from (0,0) should converge fast";
    EXPECT_NEAR(r.x[0], 1.0, 1e-5);
    EXPECT_NEAR(r.x[1], 1.0, 1e-5);
}

TEST(Newton, RosenbrockSolutionIsAccurate) {
    // Tighter tolerance: verify high-accuracy convergence.
    Rosenbrock f;
    Vector x0(2);
    x0 << 0.0, 0.0;

    OptimizeConfig cfg;
    cfg.max_iter = 200;
    cfg.tol_grad = 1e-10;

    NewtonOptimizer opt(cfg);
    auto r = opt.minimize(f, x0);

    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.x[0], 1.0, 1e-8);
    EXPECT_NEAR(r.x[1], 1.0, 1e-8);
    EXPECT_LT(r.f, 1e-16);
}

// ── Saddle point: method must escape, not converge to the saddle ─────────────

TEST(Newton, SaddlePointEscapes) {
    // At x₀=(0.1,0.1):  f₀ = 0,  H has λ_min = −2  →  μ ≈ 2
    // H_mod ≈ diag(4, 1e-8).  Newton direction strongly along +y:
    //   d ≈ (−0.05, +2e7).  Armijo accepts α=1 immediately (Δf ≈ −4e14).
    // Test: f decreases (becomes negative), x moves away from x₀.
    Saddle f;
    Vector x0(2);
    x0 << 0.1, 0.1;

    const Scalar f0 = f(x0).first;  // = 0

    OptimizeConfig cfg;
    cfg.max_iter  = 5;
    cfg.tol_grad  = 1e-12;  // tight: don't stop early

    NewtonOptimizer opt(cfg);
    auto r = opt.minimize(f, x0);

    EXPECT_LT(r.f, f0)
        << "f should decrease below f₀=0 (escape saddle), got f=" << r.f;
    EXPECT_GT((r.x - x0).norm(), 1e-3)
        << "x should move significantly from the saddle neighbourhood";
}

TEST(Newton, SaddlePointMuIsPositive) {
    // Verify that the modified Hessian kicks in: μ > 0 at the saddle.
    // We can't test μ directly from the outside, but we can verify that
    // the iteration takes a meaningful step (not stuck at x₀).
    Saddle f;
    Vector x0(2);
    x0 << 0.05, 0.05;

    OptimizeConfig cfg;
    cfg.max_iter = 1;
    cfg.tol_grad = 1e-12;

    NewtonOptimizer opt(cfg);
    auto r = opt.minimize(f, x0);

    // f(x0) = 0.05² - 0.05² = 0; after one step f must be < 0
    EXPECT_LT(r.f, 0.0);
}

// ── Verbose mode ──────────────────────────────────────────────────────────────

TEST(Newton, VerboseModeProducesCorrectResult) {
    // Verbose should not affect the numeric result.
    DiagonalQuadratic f;
    f.a.resize(3);
    f.a << 1.0, 5.0, 10.0;
    Vector x0(3);
    x0 << 1.0, 1.0, 1.0;

    OptimizeConfig silent, loud;
    loud.verbose = true;

    auto r_silent = NewtonOptimizer(silent).minimize(f, x0);
    auto r_loud   = NewtonOptimizer(loud).minimize(f, x0);  // prints to stdout

    EXPECT_EQ(r_silent.iterations, r_loud.iterations);
    EXPECT_NEAR(r_silent.f, r_loud.f, 1e-15);
    EXPECT_LT((r_silent.x - r_loud.x).norm(), 1e-14);
}

} // namespace optlib::test
