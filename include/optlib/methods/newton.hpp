#pragma once

#include <algorithm>    // std::max
#include <cstdio>
#include <Eigen/Dense>  // SelfAdjointEigenSolver, LDLT

#include <optlib/core/types.hpp>
#include <optlib/core/concepts.hpp>
#include <optlib/core/linesearch.hpp>

// Newton's method with modified Hessian and Armijo line search.
//
// Each iteration:
//   1. Evaluate (f, g) and H = f.hessian(x).
//   2. Regularise:  H_mod = H + μI,  μ = max(0, 1e-8 − λ_min(H))
//      Computed via SelfAdjointEigenSolver (eigenvalues only, O(n³)).
//      Guarantees H_mod is positive definite → direction is always descent.
//   3. Solve (H + μI)·d = −g via LDLT.
//   4. Armijo backtracking along d.
//   5. x ← x + α·d.
//
// Convergence:
//   μ = 0 near solution (H already PD)  → quadratic rate.
//   μ > 0 far from solution (H indefinite, e.g. saddle) → linear rate,
//     but the step still descends: the saddle is escaped, not approached.

namespace optlib {

class NewtonOptimizer {
public:
    explicit NewtonOptimizer(OptimizeConfig cfg = {}) : cfg_(std::move(cfg)) {}

    [[nodiscard]] const OptimizeConfig& config() const noexcept { return cfg_; }

    template<FunctionWithHessian F>
    [[nodiscard]] OptimizeResult minimize(F&& f, Vector x0) const {
        Vector x = std::move(x0);

        if (cfg_.verbose) print_header();

        for (uint32_t k = 0; k < cfg_.max_iter; ++k) {

            // ── 1. Evaluate (f_val, g) ────────────────────────────────────────
            auto fvg    = f(x);
            Scalar f_val = fvg.first;
            Vector g     = fvg.second;
            const Scalar gn = g.norm();

            // ── 2. Gradient convergence check ─────────────────────────────────
            if (gn < cfg_.tol_grad) {
                if (cfg_.verbose)
                    std::printf("  [converged]  iter %-4u  f = %.6g  ||g|| = %.3e\n",
                                k, f_val, gn);
                return make_result(x, f_val, k, true, gn);
            }

            // ── 3. Modified Hessian ───────────────────────────────────────────
            Matrix H = f.hessian(x);

            // Find λ_min without computing eigenvectors (≈ 2× faster).
            Eigen::SelfAdjointEigenSolver<Matrix> eig(H, Eigen::EigenvaluesOnly);
            const Scalar lam_min = (eig.info() == Eigen::Success)
                                   ? eig.eigenvalues().minCoeff()
                                   : Scalar{-1};  // conservative fallback

            // μ chosen so that every eigenvalue of H_mod is ≥ 1e-8.
            const Scalar mu = std::max(Scalar{0}, Scalar{1e-8} - lam_min);

            Matrix H_mod = H;
            H_mod.diagonal().array() += mu;  // H + μI  (in-place, no extra alloc)

            // ── 4. Solve (H + μI)·d = −g via LDLT ────────────────────────────
            // LDLT is numerically robust for SPD matrices and avoids the
            // explicit Hessian inverse.
            const Eigen::LDLT<Matrix> ldlt(H_mod);
            Vector d;
            if (ldlt.info() == Eigen::Success) {
                d = ldlt.solve(-g);
            } else {
                d = -g;  // fallback: steepest descent
            }

            // Guard: if d is not a descent direction (e.g. severe roundoff),
            // fall back to the negative gradient which always descends.
            Scalar dphi0 = g.dot(d);
            if (dphi0 >= 0.0) {
                d     = -g;
                dphi0 = -(gn * gn);
            }

            // ── 5. Armijo backtracking line search ────────────────────────────
            const Scalar alpha = linesearch::armijo(
                [&](Scalar a) -> Scalar { return f(x + a * d).first; },
                f_val, dphi0);

            // ── Verbose ───────────────────────────────────────────────────────
            if (cfg_.verbose) print_row(k, f_val, gn, alpha, mu);

            // ── 6. Update ─────────────────────────────────────────────────────
            const Vector step = alpha * d;
            x += step;

            if (step.norm() < cfg_.tol_step) {
                auto fvg2 = f(x);
                return make_result(x, fvg2.first, k + 1, true, fvg2.second.norm());
            }
        }

        // Max iterations exhausted
        auto fvg = f(x);
        return make_result(x, fvg.first, cfg_.max_iter, false, fvg.second.norm());
    }

private:
    OptimizeConfig cfg_;

    static OptimizeResult make_result(const Vector& x, Scalar f_val,
                                      uint32_t iters, bool conv,
                                      Scalar gn) noexcept {
        OptimizeResult r;
        r.x          = x;
        r.f          = f_val;
        r.iterations = iters;
        r.converged  = conv;
        r.grad_norm  = gn;
        return r;
    }

    static void print_header() {
        std::printf("%-5s  %-15s  %-11s  %-10s  %-10s\n",
                    "iter", "f", "||g||", "alpha", "mu");
        std::puts("----------------------------------------------------");
    }

    static void print_row(uint32_t k, Scalar f, Scalar gn,
                          Scalar alpha, Scalar mu) {
        std::printf("%-5u  %-15.7g  %-11.3e  %-10.4g  %-10.3e\n",
                    k, f, gn, alpha, mu);
    }
};

} // namespace optlib
