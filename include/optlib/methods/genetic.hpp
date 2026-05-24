#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include <Eigen/Core>

#include <optlib/core/types.hpp>
#include <optlib/core/concepts.hpp>

// Real-coded genetic algorithm with BLX-0.5 crossover, Gaussian mutation,
// tournament selection, and elitism.
//
// Derivative-free: only requires f(x) → Scalar.
// Convergence: stochastic — terminates when best fitness stagnates for
// `stagnation_patience` consecutive generations, or max_generations is reached.
//
// Population layout: "Structure of Arrays"
//
//   Flat layout:   genes[i*dim + j]   — gene j of individual i
//   Flat layout:   fitness[i]         — fitness of individual i
//
// Alternative "Array of Structs" design:
//   struct Individual { std::vector<Scalar> genes; Scalar fitness; };
//   std::vector<Individual> population;
//
// AoS disadvantages in this context:
//   1. Each Individual::genes is a separate heap allocation; accessing the
//      population during tournament selection (fitness only) pulls in gene
//      data that is not needed, wasting cache lines.
//   2. A separate heap pointer per individual means pointer chasing during
//      fitness evaluation.
//
// SoA keeps all genes in one contiguous block and all fitnesses in another.
// During fitness evaluation, individual i's genes — genes[i*dim .. (i+1)*dim-1] —
// fit into consecutive cache lines with no interleaved metadata.
// During tournament selection (fitness[] only), the gene block is not touched.

namespace optlib {

struct GeneticConfig {
    uint32_t population_size{200};
    uint32_t max_generations{500};
    Scalar   mutation_rate{0.01};
    Scalar   crossover_rate{0.80};
    uint32_t elite_count{5};
    uint32_t tournament_size{3};
    Scalar   mutation_sigma{0.1};  // σ = mutation_sigma × (hi − lo) per gene
    Scalar   tol{1e-9};            // stagnation threshold on best fitness
    uint64_t seed{42};
    bool     verbose{false};
};

class GeneticOptimizer {
public:
    explicit GeneticOptimizer(GeneticConfig cfg = {}) : cfg_(std::move(cfg)) {}

    [[nodiscard]] const GeneticConfig& config() const noexcept { return cfg_; }

    template<ScalarFunction F>
    [[nodiscard]] OptimizeResult minimize(F&& f, int dim,
                                          const Vector& lo, const Vector& hi) const {
        assert(lo.size() == dim && hi.size() == dim);

        const int      P  = static_cast<int>(cfg_.population_size);
        const uint32_t ec = std::min(cfg_.elite_count,
                                     static_cast<uint32_t>(P));

        // ── SoA buffers: single allocation per field, no per-individual heap ──
        std::vector<Scalar> genes(P * dim);       // genes[i*dim + j]
        std::vector<Scalar> fitness(P);
        std::vector<Scalar> genes_next(P * dim);
        std::vector<int>    order(P);

        std::mt19937_64                    rng(cfg_.seed);
        std::uniform_real_distribution<Scalar> unit(0.0, 1.0);
        std::normal_distribution<Scalar>       gauss(0.0, 1.0);
        std::uniform_int_distribution<int>     pick(0, P - 1);

        // Per-dimension range; precomputed to avoid recomputing in hot loops.
        std::vector<Scalar> range(dim);
        for (int j = 0; j < dim; ++j)
            range[j] = hi[j] - lo[j];

        // ── 1. Initialisation: uniform random in [lo, hi] ─────────────────────
        for (int i = 0; i < P; ++i)
            for (int j = 0; j < dim; ++j)
                genes[i * dim + j] = lo[j] + unit(rng) * range[j];

        // ── 2. Fitness evaluation via Eigen::Map (zero-copy) ──────────────────
        // Map<const Vector> provides a Vector view of genes[i*dim..] without
        // copying, so f receives exactly the type its concept requires.
        auto eval_all = [&] {
            for (int i = 0; i < P; ++i) {
                Eigen::Map<const Vector> xi(genes.data() + i * dim, dim);
                fitness[i] = f(xi);
            }
        };

        eval_all();

        // ── 3. Tournament selection ───────────────────────────────────────────
        // Pick tournament_size candidates at random; return the fittest (lowest f).
        auto tournament = [&]() -> int {
            int best = pick(rng);
            for (uint32_t t = 1; t < cfg_.tournament_size; ++t) {
                const int cand = pick(rng);
                if (fitness[cand] < fitness[best]) best = cand;
            }
            return best;
        };

        if (cfg_.verbose) print_header();

        Scalar   best_prev      = *std::min_element(fitness.begin(), fitness.end());
        uint32_t stagnation_cnt = 0;
        constexpr uint32_t STAGNATION_PATIENCE = 30;

        for (uint32_t gen = 0; gen < cfg_.max_generations; ++gen) {

            // ── 6. Elitism: copy top ec individuals to the front of genes_next ─
            // partial_sort: O(P log ec) — cheaper than full sort when ec ≪ P.
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + ec, order.end(),
                              [&](int a, int b) { return fitness[a] < fitness[b]; });

            for (uint32_t e = 0; e < ec; ++e) {
                const int src = order[e];
                std::copy(genes.data() + src * dim,
                          genes.data() + src * dim + dim,
                          genes_next.data() + e * dim);
            }

            const Scalar best_f = fitness[order[0]];
            if (cfg_.verbose) {
                const Scalar mean_f = std::accumulate(fitness.begin(), fitness.end(),
                                                      Scalar{0}) / P;
                print_row(gen, best_f, mean_f);
            }

            // Stagnation convergence check
            if (std::abs(best_f - best_prev) < cfg_.tol) {
                if (++stagnation_cnt >= STAGNATION_PATIENCE) {
                    return extract_best(genes, fitness, order[0], dim, gen + 1, true);
                }
            } else {
                stagnation_cnt = 0;
            }
            best_prev = best_f;

            // ── Fill remainder: selection + crossover + mutation ───────────────
            for (int i = static_cast<int>(ec); i < P; ++i) {
                const int p1 = tournament();
                const int p2 = tournament();

                const Scalar* par1  = genes.data() + p1 * dim;
                const Scalar* par2  = genes.data() + p2 * dim;
                Scalar*       child = genes_next.data() + i * dim;

                // ── 4. BLX-0.5 crossover (blend crossover, α=0.5) ─────────────
                // For genes g1, g2: sample uniformly from
                //   [min(g1,g2) − 0.5·|g1−g2|,  max(g1,g2) + 0.5·|g1−g2|]
                // then clamp to [lo, hi].
                // When g1==g2 the interval collapses to a point; mutation handles
                // diversity recovery in that case.
                if (unit(rng) < cfg_.crossover_rate) {
                    for (int j = 0; j < dim; ++j) {
                        const Scalar g1 = par1[j], g2 = par2[j];
                        const Scalar lo_b = std::min(g1, g2);
                        const Scalar hi_b = std::max(g1, g2);
                        const Scalar ext  = 0.5 * (hi_b - lo_b);
                        const Scalar v    = (lo_b - ext) + unit(rng) * (hi_b - lo_b + 2.0 * ext);
                        child[j] = std::clamp(v, lo[j], hi[j]);
                    }
                } else {
                    std::copy(par1, par1 + dim, child);
                }

                // ── 5. Gaussian mutation ───────────────────────────────────────
                // σ = mutation_sigma × range[j] so that mutation scales with the
                // search domain width (avoids tiny steps in wide domains).
                for (int j = 0; j < dim; ++j) {
                    if (unit(rng) < cfg_.mutation_rate) {
                        const Scalar sigma = cfg_.mutation_sigma * range[j];
                        child[j] = std::clamp(child[j] + gauss(rng) * sigma,
                                              lo[j], hi[j]);
                    }
                }
            }

            std::swap(genes, genes_next);
            eval_all();
        }

        // Max generations reached: find best in current population.
        std::iota(order.begin(), order.end(), 0);
        const int bi = *std::min_element(order.begin(), order.end(),
                                         [&](int a, int b) {
                                             return fitness[a] < fitness[b];
                                         });
        return extract_best(genes, fitness, bi, dim, cfg_.max_generations, false);
    }

private:
    GeneticConfig cfg_;

    static OptimizeResult extract_best(const std::vector<Scalar>& genes,
                                       const std::vector<Scalar>& fitness,
                                       int bi, int dim,
                                       uint32_t iters, bool conv) {
        OptimizeResult r;
        r.x          = Eigen::Map<const Vector>(genes.data() + bi * dim, dim);
        r.f          = fitness[bi];
        r.iterations = iters;
        r.converged  = conv;
        r.grad_norm  = 0.0;   // GA is derivative-free; gradient unavailable
        return r;
    }

    static void print_header() {
        std::printf("%-6s  %-15s  %-15s\n", "gen", "best_f", "mean_f");
        std::puts("----------------------------------------");
    }

    static void print_row(uint32_t gen, Scalar best_f, Scalar mean_f) {
        std::printf("%-6u  %-15.7g  %-15.7g\n", gen, best_f, mean_f);
    }
};

} // namespace optlib
