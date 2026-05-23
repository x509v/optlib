#pragma once

#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <utility>
#include <optlib/core/types.hpp>

// One-dimensional line search along a descent direction d.
//
// Both algorithms work on the univariate "phi" function:
//   φ(α) = f(x + α·d),   φ'(α) = ∇f(x + α·d) · d
//
// Callers create these closures from the full objective and pass them in.
// This decouples line-search math from the optimizer's data layout.

namespace optlib::linesearch {

// ── Options ──────────────────────────────────────────────────────────────────

struct ArmijoOptions {
    Scalar   c1{1e-4};        // sufficient decrease constant ∈ (0, 0.5)
    Scalar   rho{0.5};        // backtracking factor ∈ (0, 1)
    Scalar   alpha_init{1.0}; // initial trial step (1.0 = Newton scale)
    uint32_t max_iter{50};    // backtrack budget
};

struct WolfeOptions {
    Scalar   c1{1e-4};         // sufficient decrease,   0 < c1 < c2
    Scalar   c2{0.9};          // curvature constant,    c1 < c2 < 1
    Scalar   alpha_init{1.0};
    Scalar   alpha_max{1e3};   // ceiling for bracket expansion
    uint32_t bracket_iter{20}; // budget for bracket phase
    uint32_t zoom_iter{30};    // budget for zoom/bisection phase
};

struct WolfeResult {
    Scalar   alpha;
    bool     success{false};
    uint32_t f_evals{0};   // total phi_dphi calls
};

// ── Concepts ─────────────────────────────────────────────────────────────────

// α ↦ φ(α)
template<typename F>
concept PhiFunc = requires(const F& f, Scalar a) {
    { f(a) } -> std::convertible_to<Scalar>;
};

// α ↦ {φ(α), φ'(α)}  — returning both from one call avoids evaluating f twice
template<typename F>
concept PhiDPhiFunc = requires(const F& f, Scalar a) {
    { f(a) } -> std::convertible_to<std::pair<Scalar, Scalar>>;
};

// ── Armijo backtracking ───────────────────────────────────────────────────────
//
// Sufficient decrease (Armijo) condition:
//   φ(α) ≤ φ(0) + c1·α·φ'(0)
//
// Starting from α = alpha_init, multiply by rho until the condition holds.
//
// Failure policy: if Armijo is never satisfied in max_iter steps,
// returns the smallest α that was tried (= alpha_init · rho^(max_iter-1)).
// Guarantees monotone shrinkage; caller decides whether to accept.

template<PhiFunc Phi>
[[nodiscard]] Scalar armijo(const Phi& phi, Scalar phi0, Scalar dphi0,
                             ArmijoOptions opts = {}) {
    // dphi0 = ∇f(x)·d must be < 0 for d to be a descent direction
    assert(dphi0 < 0.0);

    // Pre-multiply once: threshold = phi0 + c1·α·dphi0 ↔ phi0 + c1·dphi0·α
    const Scalar slope = opts.c1 * dphi0;  // negative

    Scalar alpha = opts.alpha_init;
    for (uint32_t i = 0; i < opts.max_iter; ++i) {
        if (phi(alpha) <= phi0 + slope * alpha) {
            return alpha;              // Armijo satisfied
        }
        alpha *= opts.rho;            // shrink
    }
    return alpha;  // = alpha_init · rho^max_iter  (not verified to satisfy Armijo)
}

// ── Wolfe line search: bracket + zoom ────────────────────────────────────────
//
// Strong Wolfe conditions:
//   (W1) φ(α) ≤ φ(0) + c1·α·φ'(0)          [sufficient decrease]
//   (W2) |φ'(α)| ≤ c2·|φ'(0)|               [curvature / anti-zigzag]
//
// ─── Bracket phase (Nocedal & Wright, Algorithm 3.5) ─────────────────────
//   Expand α geometrically until a bracket [α_prev, α] containing a Wolfe
//   point is found, then delegate to zoom.  Terminates early if (W2) holds.
//
// ─── Zoom phase (Nocedal & Wright, Algorithm 3.6) ──────────────────────
//   Binary bisection inside [α_lo, α_hi] that maintains:
//     • φ(α_lo) satisfies (W1) and is the smallest φ seen so far
//     • the interval contains a Wolfe point
//   Bisection is O(log(1/ε)); cubic interpolation would converge faster in
//   practice but is not needed for correctness.
//
// Returns {alpha, success=true}  if both (W1) and (W2) hold.
// Returns {alpha_lo, success=false} if zoom budget exhausted; (W1) still holds.

template<PhiDPhiFunc F>
[[nodiscard]] WolfeResult wolfe(const F& phi_dphi, Scalar phi0, Scalar dphi0,
                                 WolfeOptions opts = {}) {
    if (dphi0 >= 0.0) {                   // not a descent direction
        return {opts.alpha_init, false, 0};
    }

    const Scalar c1_slope   =  opts.c1 * dphi0;   // negative: Armijo RHS slope
    const Scalar c2_curv    = -opts.c2 * dphi0;   // positive: curvature threshold

    uint32_t evals = 0;

    // ── Zoom: bisect [alo, ahi] to find a strong-Wolfe point ──────────────
    //
    // Entry invariants:
    //   φ(alo) satisfies (W1) and φ(alo) ≤ φ at all previous iterates
    //   The interval [alo, ahi] (or [ahi, alo]) brackets a minimum of φ.
    auto zoom = [&](Scalar alo, Scalar ahi, Scalar phi_alo) -> WolfeResult {
        for (uint32_t j = 0; j < opts.zoom_iter; ++j) {
            const Scalar alpha_j = 0.5 * (alo + ahi);   // bisection midpoint
            auto [phi_j, dphi_j] = phi_dphi(alpha_j);
            ++evals;

            if (phi_j > phi0 + c1_slope * alpha_j || phi_j >= phi_alo) {
                // (W1) violated OR no improvement over alo → squeeze from above
                ahi = alpha_j;
            } else {
                // (W1) satisfied and φ(alpha_j) < φ(alo)
                if (std::abs(dphi_j) <= c2_curv) {
                    return {alpha_j, true, evals};    // strong Wolfe ✓
                }
                // Sign of φ'(alpha_j) tells us which half contains the minimum
                if (dphi_j * (ahi - alo) >= 0.0) {
                    ahi = alo;
                }
                alo     = alpha_j;
                phi_alo = phi_j;
            }
        }
        // Budget exhausted: alo satisfies at least (W1)
        return {alo, false, evals};
    };

    // ── Bracket phase ──────────────────────────────────────────────────────
    Scalar alpha_prev = 0.0;
    Scalar phi_prev   = phi0;
    Scalar alpha      = opts.alpha_init;

    for (uint32_t i = 0; i < opts.bracket_iter; ++i) {
        auto [phi_i, dphi_i] = phi_dphi(alpha);
        ++evals;

        // (W1) violated, or φ increased relative to previous iterate →
        // a bracket [alpha_prev, alpha] is found; zoom inside it
        if (phi_i > phi0 + c1_slope * alpha || (i > 0 && phi_i >= phi_prev)) {
            return zoom(alpha_prev, alpha, phi_prev);
        }

        // (W2) already holds → both strong Wolfe conditions satisfied
        if (std::abs(dphi_i) <= c2_curv) {
            return {alpha, true, evals};
        }

        // φ'(α) ≥ 0 → minimum is between alpha_prev and alpha; zoom there
        if (dphi_i >= 0.0) {
            return zoom(alpha, alpha_prev, phi_i);
        }

        // Still descending: expand the step
        alpha_prev = alpha;
        phi_prev   = phi_i;
        alpha      = std::min(opts.alpha_max, 2.0 * alpha);
    }

    // bracket_iter exhausted without finding a bracket
    return {alpha, false, evals};
}

// ── Static concept checks ─────────────────────────────────────────────────────

// PhiFunc — positive: α → Scalar ✓
static_assert( PhiFunc<decltype([](Scalar a) -> Scalar { return a * a; })>);
// PhiFunc — negative: returns pair, not Scalar ✗
static_assert(!PhiFunc<decltype([](Scalar a) -> std::pair<Scalar, Scalar> { return {a, a}; })>);

// PhiDPhiFunc — positive: α → {Scalar, Scalar} ✓
static_assert( PhiDPhiFunc<decltype([](Scalar a) -> std::pair<Scalar, Scalar> { return {a * a, 2 * a}; })>);
// PhiDPhiFunc — negative: returns only Scalar ✗
static_assert(!PhiDPhiFunc<decltype([](Scalar a) -> Scalar { return a * a; })>);

} // namespace optlib::linesearch
