#pragma once

#include <cstdint>
#include <Eigen/Dense>

namespace optlib {

using Scalar = double;
using Vector = Eigen::VectorXd;

// Explicit ColMajor + Eigen's default aligned allocator (64-byte on AVX targets).
// Equivalent to MatrixXd but self-documenting about layout and alignment intent.
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

// ── Solver configuration ────────────────────────────────────────────────────

struct OptimizeConfig {
    uint32_t max_iter{500};
    Scalar   tol_grad{1e-6};   // stop when ||∇f||₂ < tol_grad
    Scalar   tol_step{1e-10};  // stop when ||xₖ₊₁ − xₖ||₂ < tol_step
    bool     verbose{false};
};

// ── Result returned by every solver ────────────────────────────────────────

struct OptimizeResult {
    Vector   x;               // solution (or best iterate on failure)
    Scalar   f{};             // f(x)
    uint32_t iterations{};
    bool     converged{false};
    Scalar   grad_norm{};     // ||∇f(x)||₂ at termination
};

} // namespace optlib
