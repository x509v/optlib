#pragma once

// Benchmark test functions for method comparison.
//
// All three structs satisfy FunctionWithHessian (and therefore FunctionWithGrad).
// Wrap in a lambda returning .first to obtain a ScalarFunction for GeneticOptimizer.
//
// Call counter: every call to operator() (value + gradient) increments `calls`.
// Hessian call counter: every call to hessian() increments `hess_calls`.
// reset() zeroes both before each benchmark run.

#include <cmath>
#include <optlib/core/types.hpp>

namespace optlib::bench {

// ── 1. Geometric Quadratic ────────────────────────────────────────────────────
//
// f(x) = x^T A x,  A = diag(1, κ, κ², …, κ^{n-1}),  κ = 10 by default.
//
// Condition number: κ^{n-1}  (1e9 for n=10, 1e49 for n=50).
// ∇f(x) = 2Ax  (diagonal → O(n) gradient)
// H(x)  = 2A   (constant, diagonal, PD → Newton converges in 1 step)
// Minimum: x* = 0, f* = 0.
struct Quadratic {
    int    n;
    Vector diag;          // precomputed: diag[i] = κ^i
    mutable int calls{0};
    mutable int hess_calls{0};

    Quadratic(int n_, double kappa = 10.0) : n(n_), diag(n_) {
        diag[0] = 1.0;
        for (int i = 1; i < n_; ++i) diag[i] = diag[i-1] * kappa;
    }

    void reset() const noexcept { calls = 0; hess_calls = 0; }

    std::pair<Scalar, Vector> operator()(const Vector& x) const {
        ++calls;
        const Vector Ax = x.cwiseProduct(diag);
        return {x.dot(Ax), 2.0 * Ax};
    }

    Matrix hessian(const Vector&) const {
        ++hess_calls;
        Matrix H = Matrix::Zero(n, n);
        H.diagonal() = 2.0 * diag;
        return H;
    }
};

// ── 2. Chained Rosenbrock ─────────────────────────────────────────────────────
//
// f(x) = Σ_{i=0}^{n-2} [100(x_{i+1} − x_i²)² + (1 − x_i)²]
//
// Classic non-convex, narrow curved valley.  Global minimum at x*=(1,…,1), f*=0.
// Starting point for comparison: x0 = −1·ones(n)  (standard hard start).
//
// Hessian is symmetric tridiagonal:
//   H[i,i]   = 1200x_i² − 400x_{i+1} + 2   (i < n−1)
//            + 200                            (i > 0, from term i−1)
//   H[n-1,n-1] = 200
//   H[i,i+1] = H[i+1,i] = −400 x_i
struct Rosenbrock {
    int n;
    mutable int calls{0};
    mutable int hess_calls{0};

    explicit Rosenbrock(int n_) : n(n_) {}

    void reset() const noexcept { calls = 0; hess_calls = 0; }

    std::pair<Scalar, Vector> operator()(const Vector& x) const {
        ++calls;
        Scalar val = 0.0;
        Vector g   = Vector::Zero(n);
        for (int i = 0; i < n - 1; ++i) {
            const Scalar a = x[i+1] - x[i] * x[i];
            const Scalar b = 1.0 - x[i];
            val     += 100.0 * a * a + b * b;
            g[i]    += -400.0 * x[i] * a - 2.0 * b;
            g[i+1]  +=  200.0 * a;
        }
        return {val, g};
    }

    Matrix hessian(const Vector& x) const {
        ++hess_calls;
        Matrix H = Matrix::Zero(n, n);
        for (int i = 0; i < n - 1; ++i) {
            // Contribution of term i to the (i,i) diagonal
            H(i, i)     += -400.0 * (x[i+1] - 3.0 * x[i] * x[i]) + 2.0;
            // Contribution of term i to the (i+1,i+1) diagonal
            H(i+1, i+1) += 200.0;
            // Off-diagonal (symmetric, only one term per pair)
            H(i, i+1)    = -400.0 * x[i];
            H(i+1, i)    = -400.0 * x[i];
        }
        return H;
    }
};

// ── 3. Sphere ─────────────────────────────────────────────────────────────────
//
// f(x) = ‖x‖²  — trivial convex quadratic, condition number = 1.
// ∇f = 2x,  H = 2I.  Minimum: x* = 0, f* = 0.
// Starting point: x0 = 3·ones(n).
struct Sphere {
    int n;
    mutable int calls{0};
    mutable int hess_calls{0};

    explicit Sphere(int n_) : n(n_) {}

    void reset() const noexcept { calls = 0; hess_calls = 0; }

    std::pair<Scalar, Vector> operator()(const Vector& x) const {
        ++calls;
        return {x.squaredNorm(), 2.0 * x};
    }

    Matrix hessian(const Vector&) const {
        ++hess_calls;
        return 2.0 * Matrix::Identity(n, n);
    }
};

} // namespace optlib::bench
