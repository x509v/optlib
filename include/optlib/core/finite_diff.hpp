#pragma once

#include <utility>
#include <optlib/core/types.hpp>
#include <optlib/core/concepts.hpp>

// Finite-difference utilities.
//
// Use when:
//   - prototyping: verify an analytic gradient before deploying it
//   - wrapping external code that only exposes f(x)
//
// Accuracy budget (centered differences, IEEE-754 double):
//   truncation error   O(h²)
//   cancellation error O(ε_machine / h)
//   optimal h ≈ ε_machine^(1/3) ≈ 1e-5   → total error ≈ 1e-10
//
// For the Hessian the optimal h ≈ ε_machine^(1/4) ≈ 3e-4 gives ~1e-8 error.
// AD (see dual_ad.py) achieves ε_machine ≈ 2e-16 with zero extra evaluations.

namespace optlib::finite_diff {

// ── Gradient ────────────────────────────────────────────────────────────────
// Centered differences: g_i = (f(x + h·eᵢ) − f(x − h·eᵢ)) / (2h)
// Cost: 2n function evaluations.  Mutates one element at a time to avoid
// allocating n temporary vectors.

template<ScalarFunction F>
Vector gradient(const F& f, const Vector& x, Scalar h = 1e-5) {
    const int n = static_cast<int>(x.size());
    Vector g(n);
    Vector xmod = x;   // single working copy

    for (int i = 0; i < n; ++i) {
        xmod[i] = x[i] + h;
        const Scalar fp = f(xmod);
        xmod[i] = x[i] - h;
        const Scalar fm = f(xmod);
        xmod[i] = x[i];           // restore
        g[i] = (fp - fm) / (2.0 * h);
    }
    return g;
}

// ── Hessian ─────────────────────────────────────────────────────────────────
// Diagonal:     Hᵢᵢ = (f(x+h·eᵢ) − 2f₀ + f(x−h·eᵢ)) / h²
// Off-diagonal: Hᵢⱼ = (f(x+h·eᵢ+h·eⱼ) − f(x+h·eᵢ−h·eⱼ)
//                      − f(x−h·eᵢ+h·eⱼ) + f(x−h·eᵢ−h·eⱼ)) / (4h²)
//
// Symmetry is enforced explicitly (Hᵢⱼ = Hⱼᵢ).
// Cost: 1 + 2n + 4·C(n,2) = 2n² + 1 evaluations.

template<ScalarFunction F>
Matrix hessian(const F& f, const Vector& x, Scalar h = 1e-5) {
    const int n = static_cast<int>(x.size());
    Matrix H(n, n);
    const Scalar f0 = f(x);
    Vector xmod = x;

    // diagonal
    for (int i = 0; i < n; ++i) {
        xmod[i] = x[i] + h;
        const Scalar fp = f(xmod);
        xmod[i] = x[i] - h;
        const Scalar fm = f(xmod);
        xmod[i] = x[i];
        H(i, i) = (fp - 2.0 * f0 + fm) / (h * h);
    }

    // off-diagonal upper triangle, mirror to lower
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            xmod[i] = x[i] + h;  xmod[j] = x[j] + h;
            const Scalar fpp = f(xmod);
            xmod[j] = x[j] - h;
            const Scalar fpm = f(xmod);
            xmod[i] = x[i] - h;  xmod[j] = x[j] + h;
            const Scalar fmp = f(xmod);
            xmod[j] = x[j] - h;
            const Scalar fmm = f(xmod);
            xmod[i] = x[i];  xmod[j] = x[j];   // restore both

            H(i, j) = H(j, i) = (fpp - fpm - fmp + fmm) / (4.0 * h * h);
        }
    }
    return H;
}

// ── Wrapper ─────────────────────────────────────────────────────────────────
// Lifts any ScalarFunction to FunctionWithGrad by computing the gradient
// via centered finite differences.  Use for rapid prototyping or when
// integrating external code with no analytic derivatives.
//
// Usage:
//   auto f_raw = [](const Vector& x) { return x.squaredNorm(); };
//   auto f     = finite_diff::Wrapper{f_raw};   // CTAD, h = 1e-5
//   auto [val, grad] = f(x);

template<ScalarFunction F>
struct Wrapper {
    F      f;
    Scalar h;

    explicit Wrapper(F f_, Scalar h_ = 1e-5) : f(std::move(f_)), h(h_) {}

    std::pair<Scalar, Vector> operator()(const Vector& x) const {
        // single extra call for value; gradient costs 2n more
        return {f(x), gradient(f, x, h)};
    }
};

// CTAD deduction guides
template<typename F> Wrapper(F)         -> Wrapper<F>;
template<typename F> Wrapper(F, Scalar) -> Wrapper<F>;

// ── Static test ─────────────────────────────────────────────────────────────

// Wrapper<ScalarFunction> must satisfy FunctionWithGrad
static_assert(FunctionWithGrad<
    Wrapper<decltype([](const Vector& x) -> Scalar { return x.squaredNorm(); })>
>);

// Wrapper must NOT satisfy FunctionWithHessian (it exposes no .hessian())
static_assert(!FunctionWithHessian<
    Wrapper<decltype([](const Vector& x) -> Scalar { return x.squaredNorm(); })>
>);

} // namespace optlib::finite_diff
