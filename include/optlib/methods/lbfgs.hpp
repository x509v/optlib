#pragma once

#include <optlib/core/types.hpp>
#include <optlib/core/concepts.hpp>

namespace optlib {

struct LBFGSOptions {
    uint32_t max_iter{500};
    uint32_t memory{10};       // number of (s,y) pairs retained
    Scalar   tol_grad{1e-6};
    Scalar   wolfe_c1{1e-4};   // sufficient decrease (Armijo)
    Scalar   wolfe_c2{0.9};    // curvature condition
};

// Limited-memory BFGS.
// Convergence: superlinear (in practice behaves like O(ρᵏ) with ρ → 1).
// Cost per iteration: O(n·m) — two-loop recursion over m stored (s,y) pairs.
// Memory: O(n·m) — no n×n matrix ever stored.
//
// F must satisfy FunctionWithGrad: f(x) returns {value, gradient}.
//
// TODO: implement two-loop recursion + Wolfe line search
template<FunctionWithGrad F>
OptimizeResult lbfgs(F&& f, Vector x0, LBFGSOptions opts = {});

} // namespace optlib
