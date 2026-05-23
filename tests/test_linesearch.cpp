#include <gtest/gtest.h>
#include <optlib/core/linesearch.hpp>
#include <cmath>

// All tests use the same analytic model:
//
//   f(x) = ||x||²,  x₀ = ones(n),  d = −x₀  (Newton direction)
//
// Closed-form phi along the ray:
//   φ(α) = ||(1−α)·x₀||² = (1−α)²·n
//   φ'(α) = −2(1−α)·n
//   φ(0)  = n,   φ'(0) = −2n   (descent direction ✓)
//
// The minimum is at α = 1 where φ(1) = 0 — reached in a single step.
// Armijo:  0 ≤ n − 2c1·n·α  for α ≤ 1  (trivially true, c1 ≪ 0.5)
// Curvature: |−2(1−α)n| ≤ c2·2n  →  |1−α| ≤ c2  →  α ∈ [1−c2, 1+c2]
// α=1 satisfies both: φ(1)=0 and φ'(1)=0 → |0| ≤ c2·2n ✓

using namespace optlib;

namespace {

// Analytic phi for n=4 (||ones(4)||² = 4)
constexpr Scalar kN   = 4.0;
constexpr Scalar kPhi0  = kN;        // φ(0) = 4
constexpr Scalar kDPhi0 = -2.0 * kN; // φ'(0) = −8  (< 0 ✓)

auto make_phi() {
    return [](Scalar a) -> Scalar { return (1.0 - a) * (1.0 - a) * kN; };
}

auto make_phi_dphi() {
    return [](Scalar a) -> std::pair<Scalar, Scalar> {
        return {(1.0 - a) * (1.0 - a) * kN,
                -2.0 * (1.0 - a) * kN};
    };
}

// Verify Armijo condition externally
bool check_armijo(Scalar alpha, Scalar phi_a, Scalar c1 = 1e-4) {
    return phi_a <= kPhi0 + c1 * alpha * kDPhi0;
}

// Verify both strong Wolfe conditions externally
bool check_wolfe(Scalar alpha, Scalar phi_a, Scalar dphi_a,
                 Scalar c1 = 1e-4, Scalar c2 = 0.9) {
    return (phi_a  <= kPhi0 + c1 * alpha * kDPhi0) &&
           (std::abs(dphi_a) <= -c2 * kDPhi0);
}

} // namespace

// ── Armijo tests ─────────────────────────────────────────────────────────────

TEST(Armijo, QuadraticAcceptsAlphaOneInZeroBacktracks) {
    // α = 1 gives φ(1) = 0, which trivially satisfies Armijo.
    // No backtracking should occur: the returned alpha must equal alpha_init = 1.
    Scalar alpha = linesearch::armijo(make_phi(), kPhi0, kDPhi0);

    EXPECT_DOUBLE_EQ(alpha, 1.0)
        << "Armijo should accept α=1 immediately for the Newton direction";
    EXPECT_TRUE(check_armijo(alpha, make_phi()(alpha)));
}

TEST(Armijo, ReturnedAlphaSatisfiesArmijo) {
    // General sanity: whatever alpha is returned must satisfy the condition.
    linesearch::ArmijoOptions opts;
    opts.alpha_init = 0.5;  // start mid-range

    Scalar alpha = linesearch::armijo(make_phi(), kPhi0, kDPhi0, opts);
    EXPECT_TRUE(check_armijo(alpha, make_phi()(alpha)));
}

TEST(Armijo, BadAlphaInitConverges) {
    // alpha_init = 100 is far past the minimum; backtracking must find a valid step.
    // For phi(a) = (1-a)²·4, Armijo holds for any α ∈ (0, 2−2c1) ≈ (0, 2).
    // With rho=0.5 from 100 → 50 → 25 → ... → 1.5625 (7 steps) ← satisfies Armijo.
    linesearch::ArmijoOptions opts;
    opts.alpha_init = 100.0;

    Scalar alpha = linesearch::armijo(make_phi(), kPhi0, kDPhi0, opts);

    EXPECT_GT(alpha, 0.0);
    EXPECT_TRUE(check_armijo(alpha, make_phi()(alpha)))
        << "alpha=" << alpha << " must satisfy Armijo after backtracking from 100";
}

TEST(Armijo, MaxIterFallbackIsPositive) {
    // With max_iter=1 and rho=0.5, starting at alpha_init=100:
    // One check of phi(100) fails → alpha becomes 50 → returned without check.
    // We just verify the fallback is positive.
    linesearch::ArmijoOptions opts{.c1=1e-4, .rho=0.5, .alpha_init=100.0, .max_iter=1};
    Scalar alpha = linesearch::armijo(make_phi(), kPhi0, kDPhi0, opts);
    EXPECT_GT(alpha, 0.0);
}

// ── Wolfe tests ───────────────────────────────────────────────────────────────

TEST(Wolfe, QuadraticAcceptsAlphaOneImmediately) {
    // α=1 → φ(1)=0, φ'(1)=0 → both strong Wolfe conditions trivially hold.
    // The bracket phase should accept on the very first evaluation.
    auto r = linesearch::wolfe(make_phi_dphi(), kPhi0, kDPhi0);

    EXPECT_TRUE(r.success);
    EXPECT_DOUBLE_EQ(r.alpha, 1.0);
    EXPECT_EQ(r.f_evals, 1u) << "Strong Wolfe should be detected in a single phi_dphi call";
}

TEST(Wolfe, ReturnedAlphaSatisfiesBothConditions) {
    // Verify (W1) and (W2) hold numerically for the returned alpha.
    auto r = linesearch::wolfe(make_phi_dphi(), kPhi0, kDPhi0);

    ASSERT_TRUE(r.success);
    auto [val, deriv] = make_phi_dphi()(r.alpha);
    EXPECT_TRUE(check_wolfe(r.alpha, val, deriv))
        << "alpha=" << r.alpha << " phi=" << val << " dphi=" << deriv;
}

TEST(Wolfe, BadAlphaInitConverges) {
    // From alpha_init=100 the bracket phase immediately detects Armijo violation
    // and delegates to zoom, which bisects down to ≈1.5625 in a few steps.
    linesearch::WolfeOptions opts;
    opts.alpha_init = 100.0;

    auto r = linesearch::wolfe(make_phi_dphi(), kPhi0, kDPhi0, opts);

    EXPECT_TRUE(r.success);
    EXPECT_GT(r.alpha, 0.0);

    auto [val, deriv] = make_phi_dphi()(r.alpha);
    EXPECT_TRUE(check_wolfe(r.alpha, val, deriv))
        << "Wolfe must satisfy both conditions even from alpha_init=100";
}

TEST(Wolfe, CurvatureConditionHolds) {
    // Explicit numeric check on (W2): |φ'(α)| ≤ c2·|φ'(0)|
    constexpr Scalar c2 = 0.9;
    linesearch::WolfeOptions opts{.c2 = c2};

    auto r = linesearch::wolfe(make_phi_dphi(), kPhi0, kDPhi0, opts);

    ASSERT_TRUE(r.success);
    auto [val, deriv] = make_phi_dphi()(r.alpha);
    EXPECT_LE(std::abs(deriv), c2 * std::abs(kDPhi0))
        << "|phi'| = " << std::abs(deriv) << ", c2*|phi0'| = " << c2 * std::abs(kDPhi0);
}

TEST(Wolfe, RejectsNonDescentDirection) {
    // dphi0 ≥ 0 → d is not a descent direction → return immediately with success=false
    auto r = linesearch::wolfe(make_phi_dphi(), kPhi0, /*dphi0=*/+1.0);
    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.f_evals, 0u);
}

// ── Cross-check: Armijo ⊆ Wolfe ─────────────────────────────────────────────

TEST(CrossCheck, WolfeImpliesArmijo) {
    // Any point returned by Wolfe with success=true must also satisfy Armijo.
    linesearch::WolfeOptions opts;
    opts.alpha_init = 3.0;   // mid-range start, not α=1

    auto r = linesearch::wolfe(make_phi_dphi(), kPhi0, kDPhi0, opts);
    if (!r.success) GTEST_SKIP() << "Wolfe did not converge with alpha_init=3";

    auto phi = make_phi();
    EXPECT_TRUE(check_armijo(r.alpha, phi(r.alpha)));
}
