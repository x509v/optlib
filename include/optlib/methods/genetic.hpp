#pragma once

#include <cstdint>
#include <optlib/core/types.hpp>
#include <optlib/core/concepts.hpp>

namespace optlib {

struct GeneticOptions {
    uint32_t population{200};
    uint32_t max_gen{1000};
    Scalar   mutation_rate{0.01};
    Scalar   crossover_rate{0.80};
    Scalar   tol{1e-8};        // stop when best-fitness change < tol
    uint64_t seed{42};
};

// Real-coded genetic / evolutionary algorithm.
// Derivative-free: only evaluates f(x) → Scalar.
// Convergence: stochastic — depends on population diversity and selection pressure.
// Bounds: search restricted to [lower_i, upper_i] per dimension.
//
// TODO: implement SBX crossover + polynomial mutation + tournament selection
template<ScalarFunction F>
OptimizeResult genetic(F&& f, Vector lower, Vector upper, GeneticOptions opts = {});

} // namespace optlib
