#include <gtest/gtest.h>
#include <optlib/methods/lbfgs.hpp>
#include <optlib/core/finite_diff.hpp>

namespace optlib::test {

// ── helpers ─────────────────────────────────────────────────────────────────

// Quadratic f(x) = x^T A x where A = diag(1,...,1,κ).  Gradient: g = 2Ax.
static auto make_illcond_quad(int n, Scalar kappa) {
    return [n, kappa](const Vector& x) -> std::pair<Scalar, Vector> {
        Vector Ax = x;
        Ax(n - 1) *= kappa;
        return {x.dot(Ax), 2.0 * Ax};
    };
}

// Chained Rosenbrock in n dimensions with exact gradient.
static auto make_rosenbrock(int n) {
    return [n](const Vector& x) -> std::pair<Scalar, Vector> {
        Scalar val = 0.0;
        Vector g   = Vector::Zero(n);
        for (int i = 0; i < n - 1; ++i) {
            const Scalar a = x(i + 1) - x(i) * x(i);
            const Scalar b = 1.0 - x(i);
            val      += 100.0 * a * a + b * b;
            g(i)     += -400.0 * x(i) * a - 2.0 * b;
            g(i + 1) +=  200.0 * a;
        }
        return {val, g};
    };
}

// ── tests ────────────────────────────────────────────────────────────────────

TEST(LBFGS, IllConditionedQuadratic) {
    // κ=1000: GD needs O(κ ln 1/ε) ≈ 14000 iters; L-BFGS should need < 50.
    const int n = 5;
    auto f = make_illcond_quad(n, 1000.0);

    OptimizeConfig cfg;
    cfg.max_iter = 200;
    cfg.tol_grad = 1e-8;

    LBFGSOptimizer opt(cfg, 10);
    auto r = opt.minimize(f, Vector::Ones(n) * 5.0);

    EXPECT_TRUE(r.converged);
    EXPECT_LT(r.iterations, 50u);
    EXPECT_LT(r.f, 1e-12);
}

TEST(LBFGS, IllConditionedQuadFasterThanGD) {
    // Verify L-BFGS uses far fewer iterations than gradient descent on κ=1000.
    const int n = 5;
    const Scalar kappa = 1000.0;
    auto f = make_illcond_quad(n, kappa);

    // Count GD iterations with exact line search step 1/(2κ)
    Vector x_gd = Vector::Ones(n) * 5.0;
    uint32_t gd_iters = 0;
    for (; gd_iters < 5000; ++gd_iters) {
        auto [fv, g] = f(x_gd);
        if (g.norm() < 1e-8) break;
        x_gd -= (1.0 / (2.0 * kappa)) * g;
    }

    OptimizeConfig cfg;
    cfg.max_iter = 5000;
    cfg.tol_grad = 1e-8;
    auto r = LBFGSOptimizer(cfg, 10).minimize(f, Vector::Ones(n) * 5.0);

    EXPECT_TRUE(r.converged);
    EXPECT_LT(r.iterations, gd_iters);
}

TEST(LBFGS, Rosenbrock10D) {
    auto f = make_rosenbrock(10);

    OptimizeConfig cfg;
    cfg.max_iter = 500;
    cfg.tol_grad = 1e-6;

    auto r = LBFGSOptimizer(cfg, 20).minimize(f, -Vector::Ones(10));

    EXPECT_TRUE(r.converged);
    EXPECT_LT(r.iterations, 200u);
    EXPECT_LT(r.f, 1e-8);
    EXPECT_LT((r.x - Vector::Ones(10)).norm(), 1e-4);
}

TEST(LBFGS, SphereSmallDim) {
    // f(x)=||x||², L-BFGS with m=n should converge in n steps (exact BFGS on quadratic).
    auto f = [](const Vector& x) -> std::pair<Scalar, Vector> {
        return {x.squaredNorm(), 2.0 * x};
    };

    OptimizeConfig cfg;
    cfg.max_iter = 100;
    cfg.tol_grad = 1e-10;

    auto r = LBFGSOptimizer(cfg, 4).minimize(f, Vector::Ones(4) * 3.0);

    EXPECT_TRUE(r.converged);
    EXPECT_LT(r.f, 1e-15);
    EXPECT_LT(r.x.norm(), 1e-7);
}

TEST(LBFGS, MemoryNEqualsFullBFGS) {
    // Both m=n and m=3 should converge to the same solution.
    const int n = 6;
    auto f = make_illcond_quad(n, 50.0);

    OptimizeConfig cfg;
    cfg.max_iter = 300;
    cfg.tol_grad = 1e-12;

    Vector x0 = Vector::Ones(n) * 2.0;
    auto r_full  = LBFGSOptimizer(cfg, n).minimize(f, x0);
    auto r_small = LBFGSOptimizer(cfg, 3).minimize(f, x0);

    EXPECT_TRUE(r_full.converged);
    EXPECT_TRUE(r_small.converged);
    EXPECT_LT((r_full.x - r_small.x).norm(), 1e-6);
}

TEST(LBFGS, TinyMemoryStillConverges) {
    // memory=2 (very limited) still converges on Rosenbrock 4D.
    auto f = make_rosenbrock(4);

    OptimizeConfig cfg;
    cfg.max_iter = 1000;
    cfg.tol_grad = 1e-6;

    auto r = LBFGSOptimizer(cfg, 2).minimize(f, -Vector::Ones(4));

    EXPECT_TRUE(r.converged);
    EXPECT_LT(r.f, 1e-8);
}

TEST(LBFGS, WorksWithFiniteDiffWrapper) {
    // Integration test: finite_diff::Wrapper turns a ScalarFunction into FunctionWithGrad.
    auto f = finite_diff::Wrapper{[](const Vector& x) -> Scalar { return x.squaredNorm(); }};

    OptimizeConfig cfg;
    cfg.max_iter = 50;
    cfg.tol_grad = 1e-6;

    auto r = LBFGSOptimizer(cfg, 5).minimize(f, Vector::Ones(3) * 2.0);

    EXPECT_TRUE(r.converged);
    EXPECT_LT(r.f, 1e-8);
}

} // namespace optlib::test
