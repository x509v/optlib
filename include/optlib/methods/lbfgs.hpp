#pragma once

#include <cassert>
#include <cstdio>
#include <deque>
#include <vector>

#include <optlib/core/types.hpp>
#include <optlib/core/concepts.hpp>
#include <optlib/core/linesearch.hpp>

// Limited-memory BFGS (L-BFGS).
//
// Cost per iteration: O(n·m) — two-loop recursion over m (s,y) pairs.
// Memory: O(n·m) — no n×n matrix ever stored.
//
// History container: std::deque<CurvaturePair>
//   • pop_front() (evict oldest) is O(1), unlike std::vector which is O(n).
//   • capacity is bounded by `memory`, so total allocation is O(n·memory)
//     and no reallocation happens in the hot loop once the deque is full.
//
// Algorithm: Nocedal & Wright, Numerical Optimization (2nd ed.), Algorithm 7.4.

namespace optlib {

struct CurvaturePair {
    Vector s;      // s_k = x_{k+1} − x_k
    Vector y;      // y_k = g_{k+1} − g_k
    Scalar rho;    // ρ_k = 1 / (y_k · s_k)
};

class LBFGSOptimizer {
public:
    explicit LBFGSOptimizer(OptimizeConfig cfg = {}, uint32_t memory = 10)
        : cfg_(std::move(cfg)), memory_(memory) {}

    [[nodiscard]] const OptimizeConfig& config() const noexcept { return cfg_; }

    template<FunctionWithGrad F>
    [[nodiscard]] OptimizeResult minimize(F&& f, Vector x0) const {
        Vector x = std::move(x0);
        const int n = static_cast<int>(x.size());

        // Scratch arrays pre-allocated once; no alloc in hot loop.
        std::deque<CurvaturePair> hist;
        std::vector<Scalar> alpha_buf(memory_);

        // Cache last computed (f_new, g_new) from line search to avoid
        // re-evaluating f after the Wolfe call.
        Scalar f_new{};
        Vector g_new(n);

        if (cfg_.verbose) print_header();

        auto [f_val, g] = f(x);

        for (uint32_t k = 0; k < cfg_.max_iter; ++k) {
            const Scalar gn = g.norm();

            if (gn < cfg_.tol_grad) {
                if (cfg_.verbose)
                    std::printf("  [converged]  iter %-4u  f = %.6g  ||g|| = %.3e\n",
                                k, f_val, gn);
                return make_result(x, f_val, k, true, gn);
            }

            // ── Two-loop L-BFGS direction (N&W Algorithm 7.4) ─────────────────
            Vector q = g;

            // Backward pass: newest → oldest
            const int m = static_cast<int>(hist.size());
            for (int i = m - 1; i >= 0; --i) {
                const auto& p = hist[i];
                alpha_buf[i] = p.rho * p.s.dot(q);
                q.noalias() -= alpha_buf[i] * p.y;
            }

            // Initial Hessian scaling: H_0 = γ·I
            // γ = (s_{k-1}·y_{k-1}) / (y_{k-1}·y_{k-1})  (Nocedal §7.3)
            Scalar gamma = 1.0;
            if (m > 0) {
                const auto& last = hist.back();
                const Scalar yy = last.y.squaredNorm();
                if (yy > 0.0) gamma = (last.s.dot(last.y)) / yy;
            }
            Vector r = gamma * q;

            // Forward pass: oldest → newest
            for (int i = 0; i < m; ++i) {
                const auto& p = hist[i];
                const Scalar beta = p.rho * p.y.dot(r);
                r.noalias() += (alpha_buf[i] - beta) * p.s;
            }

            Vector d = -r;   // descent direction

            // Guard: if d not descent (roundoff or empty history edge case),
            // fall back to steepest descent and flush stale history.
            Scalar dphi0 = g.dot(d);
            if (dphi0 >= 0.0) {
                d      = -g;
                dphi0  = -(gn * gn);
                hist.clear();
            }

            // ── Wolfe line search ──────────────────────────────────────────────
            // Lambda captures g_new and f_new for reuse after the search.
            bool   wolfe_ok = false;
            Scalar alpha    = 1.0;

            {
                linesearch::WolfeOptions wopts;
                wopts.c1 = cfg_.tol_grad > 0 ? 1e-4 : 1e-4;   // c1 default
                wopts.c2 = 0.9;

                auto phi_dphi = [&](Scalar a) -> std::pair<Scalar, Scalar> {
                    Vector x_try = x + a * d;
                    auto [fv, gv] = f(x_try);
                    f_new = fv;
                    g_new = std::move(gv);
                    return {fv, g_new.dot(d)};
                };

                auto wr = linesearch::wolfe(phi_dphi, f_val, dphi0, wopts);
                alpha    = wr.alpha;
                wolfe_ok = wr.success;

                // If Wolfe returned an alpha that was never evaluated by phi_dphi
                // (bracket exhausted at alpha_init without any phi call → evals==0),
                // we need a fresh evaluation. In practice evals>0 always.
                if (wr.f_evals == 0) {
                    auto [fv, gv] = f(x + alpha * d);
                    f_new = fv;
                    g_new = std::move(gv);
                }

                // If Wolfe fails badly (alpha too small, no real progress),
                // clear history so the next iteration starts fresh.
                if (!wolfe_ok && alpha * d.norm() < cfg_.tol_step) {
                    hist.clear();
                }
            }

            if (cfg_.verbose)
                print_row(k, f_val, gn, alpha, static_cast<uint32_t>(hist.size()));

            // ── Update x ──────────────────────────────────────────────────────
            const Vector s = alpha * d;
            const Vector y = g_new - g;
            const Scalar sy = s.dot(y);

            x     += s;
            f_val  = f_new;
            g      = g_new;

            if (s.norm() < cfg_.tol_step) {
                return make_result(x, f_val, k + 1, true, g.norm());
            }

            // ── Update history ─────────────────────────────────────────────────
            // Skip curvature pairs where y·s is too small (near-zero curvature
            // causes ρ to overflow and corrupts the direction).
            if (sy > 1e-10) {
                if (hist.size() >= memory_) hist.pop_front();
                hist.push_back({s, y, 1.0 / sy});
            }
        }

        return make_result(x, f_val, cfg_.max_iter, false, g.norm());
    }

private:
    OptimizeConfig cfg_;
    uint32_t       memory_;

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
        std::printf("%-5s  %-15s  %-11s  %-10s  %-6s\n",
                    "iter", "f", "||g||", "alpha", "hist");
        std::puts("----------------------------------------------------");
    }

    static void print_row(uint32_t k, Scalar f, Scalar gn,
                          Scalar alpha, uint32_t hist_sz) {
        std::printf("%-5u  %-15.7g  %-11.3e  %-10.4g  %-6u\n",
                    k, f, gn, alpha, hist_sz);
    }
};

} // namespace optlib
