#include <gtest/gtest.h>
#include <optlib/optlib.hpp>

namespace optlib::test {

// ── types.hpp ───────────────────────────────────────────────────────────────

TEST(Types, VectorZero) {
    Vector v = Vector::Zero(4);
    EXPECT_EQ(v.size(), 4);
    EXPECT_DOUBLE_EQ(v.norm(), 0.0);
}

TEST(Types, MatrixIdentityColMajor) {
    Matrix A = Matrix::Identity(3, 3);
    EXPECT_DOUBLE_EQ(A.trace(), 3.0);
    EXPECT_DOUBLE_EQ((A * A - A).norm(), 0.0);
    // Verify ColMajor: element (0,1) is at memory offset 1 (after (1,0))
    EXPECT_EQ(A.IsRowMajor, false);
}

TEST(Types, OptimizeConfigDefaults) {
    OptimizeConfig cfg;
    EXPECT_EQ(cfg.max_iter, 500u);
    EXPECT_DOUBLE_EQ(cfg.tol_grad, 1e-6);
    EXPECT_DOUBLE_EQ(cfg.tol_step, 1e-10);
    EXPECT_FALSE(cfg.verbose);
}

TEST(Types, OptimizeResultDefaults) {
    OptimizeResult r;
    EXPECT_FALSE(r.converged);
    EXPECT_DOUBLE_EQ(r.f, 0.0);
    EXPECT_EQ(r.iterations, 0u);
    EXPECT_DOUBLE_EQ(r.grad_norm, 0.0);
}

// ── finite_diff.hpp ─────────────────────────────────────────────────────────

TEST(FiniteDiff, GradientSphere) {
    // f(x) = ||x||², ∇f = 2x — exact in floating point, so FD error < 1e-8
    auto f = [](const Vector& x) -> Scalar { return x.squaredNorm(); };
    Vector x = Vector::Ones(5) * 1.5;
    Vector g = finite_diff::gradient(f, x);
    Vector expected = 2.0 * x;
    EXPECT_LT((g - expected).norm(), 1e-8);
}

TEST(FiniteDiff, HessianSphere) {
    // f(x) = ||x||², H = 2I
    auto f = [](const Vector& x) -> Scalar { return x.squaredNorm(); };
    Vector x = Vector::Random(4);
    Matrix H = finite_diff::hessian(f, x);
    Matrix expected = 2.0 * Matrix::Identity(4, 4);
    // diagonal formula: cancellation error ~O(ε_machine·f/h²) ≈ 4e-6 at h=1e-5
    EXPECT_LT((H - expected).norm(), 1e-5);
}

TEST(FiniteDiff, WrapperSatisfiesFunctionWithGrad) {
    auto f_raw = [](const Vector& x) -> Scalar { return x.squaredNorm(); };
    auto wrapped = finite_diff::Wrapper{f_raw};
    Vector x = Vector::Ones(3);
    auto [val, grad] = wrapped(x);
    EXPECT_DOUBLE_EQ(val, x.squaredNorm());
    EXPECT_LT((grad - 2.0 * x).norm(), 1e-8);
}

} // namespace optlib::test
