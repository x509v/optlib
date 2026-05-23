#pragma once

#include <concepts>
#include <utility>
#include <optlib/core/types.hpp>

namespace optlib {

// ── Three core concepts ─────────────────────────────────────────────────────
//
// Design: gradient and Hessian are bundled with the value rather than stored
// as separate .grad()/.hess() methods.  This lets the caller reuse shared
// computation (e.g. forward pass in ML) and avoids calling f twice.

// F(x) → Scalar
// Minimal interface: derivative-free methods (genetic, Nelder-Mead, grid).
template<typename F>
concept ScalarFunction = requires(const F& f, const Vector& x) {
    { f(x) } -> std::convertible_to<Scalar>;
};

// F(x) → {Scalar, Vector}   (value + gradient in one call)
// First-order methods: gradient descent, L-BFGS, conjugate gradient.
template<typename F>
concept FunctionWithGrad = requires(const F& f, const Vector& x) {
    { f(x) } -> std::convertible_to<std::pair<Scalar, Vector>>;
};

// F(x) → {Scalar, Vector}  AND  F.hessian(x) → Matrix
// Second-order methods: Newton, trust-region, Gauss-Newton.
// Hessian is a separate call: it's expensive and not always needed
// (e.g. quasi-Newton methods only use it at the first iteration).
template<typename F>
concept FunctionWithHessian =
    FunctionWithGrad<F> &&
    requires(const F& f, const Vector& x) {
        { f.hessian(x) } -> std::convertible_to<Matrix>;
    };

// ── Static concept tests (verified at every compilation) ────────────────────

namespace detail {

// Helper for FunctionWithHessian positive test:
// lambdas cannot carry extra methods, so we need a struct.
struct _SphereSecondOrder {
    std::pair<Scalar, Vector> operator()(const Vector& x) const {
        return {x.squaredNorm(), 2.0 * x};
    }
    Matrix hessian(const Vector& x) const {
        return 2.0 * Matrix::Identity(x.size(), x.size());
    }
};

} // namespace detail

// ScalarFunction — positive: returns Scalar ✓
static_assert(ScalarFunction<
    decltype([](const Vector& x) -> Scalar { return x.squaredNorm(); })
>);
// ScalarFunction — negative: returns Vector, not Scalar ✗
static_assert(!ScalarFunction<
    decltype([](const Vector& x) -> Vector { return x; })
>);

// FunctionWithGrad — positive: returns pair<Scalar, Vector> ✓
static_assert(FunctionWithGrad<
    decltype([](const Vector& x) -> std::pair<Scalar, Vector> {
        return {x.squaredNorm(), 2.0 * x};
    })
>);
// FunctionWithGrad — negative: returns only Scalar, no gradient ✗
static_assert(!FunctionWithGrad<
    decltype([](const Vector& x) -> Scalar { return x.squaredNorm(); })
>);

// FunctionWithHessian — positive: has call + .hessian() ✓
static_assert(FunctionWithHessian<detail::_SphereSecondOrder>);
// FunctionWithHessian — negative: satisfies FunctionWithGrad but lacks .hessian() ✗
static_assert(!FunctionWithHessian<
    decltype([](const Vector& x) -> std::pair<Scalar, Vector> {
        return {x.squaredNorm(), 2.0 * x};
    })
>);

} // namespace optlib
