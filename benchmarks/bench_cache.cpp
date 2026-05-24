#include <benchmark/benchmark.h>

#include <cstring>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <optlib/core/types.hpp>

// ── Cache and layout microbenchmarks ─────────────────────────────────────────
//
// Run with:
//   ./optlib_bench --benchmark_filter="BM_Fitness|BM_Map|BM_Dot" \
//                  --benchmark_counters_tabular=true
//
// These benchmarks are deliberately simple so that memory access pattern is
// the only variable.  Fitness = squaredNorm (read-only, no writes to genes).

namespace optlib::bench {

// ── BM1: SoA vs AoS for fitness evaluation ───────────────────────────────────
//
// AoS: vector<Individual> — each Individual owns a separate heap allocation
//   for its genes.  Accessing individual i's genes requires dereferencing
//   pop[i].genes.data(), which may be anywhere in the heap.
//
// SoA: two flat vectors — genes[i*dim .. (i+1)*dim-1] and fitness[i].
//   Individual i's genes are a contiguous slice at a fixed offset, no
//   extra pointer dereference, CPU prefetcher can stride ahead.
//
// Expected: SoA faster because:
//   • No pointer chasing through each Individual::genes heap header
//   • All gene data in one allocation → sequential page accesses,
//     hardware prefetch works well
//   • fitness[] scan is a single stride-1 sweep (used by tournament selection)

static void BM_FitnessEval_AoS(benchmark::State& state) {
    const int P   = static_cast<int>(state.range(0));
    const int dim = static_cast<int>(state.range(1));

    struct Individual {
        std::vector<double> genes;
        double fitness{0};
    };

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> d(-5.0, 5.0);

    // Interleave junk allocations to scatter genes in the heap, approximating
    // a real program where the allocator is not fully fresh.
    std::vector<Individual>                     pop(P);
    std::vector<std::vector<double>>            junk;
    junk.reserve(P);
    for (int i = 0; i < P; ++i) {
        pop[i].genes.resize(dim);
        for (auto& g : pop[i].genes) g = d(rng);
        junk.emplace_back(dim);          // junk allocation between each individual
    }
    junk.clear();                        // free junk; pop[i].genes stay scattered

    for (auto _ : state) {
        for (int i = 0; i < P; ++i) {
            // pointer chase: pop[i].genes.data() is a heap pointer per individual
            Eigen::Map<const Vector> xi(pop[i].genes.data(), dim);
            pop[i].fitness = xi.squaredNorm();
        }
        benchmark::DoNotOptimize(pop.data());
    }

    // bytes read: P × dim doubles (genes) + P doubles (fitness writes)
    const int64_t bytes = static_cast<int64_t>(P) * (dim + 1) * sizeof(double);
    state.SetBytesProcessed(bytes * state.iterations());
    // kIsIterationInvariantRate = kIsRate with per-iteration divisor:
    //   displayed = bytes_per_iter / (total_time / iterations) = bytes/sec
    state.counters["bytes/s"] = benchmark::Counter(
        static_cast<double>(bytes),
        benchmark::Counter::kIsIterationInvariantRate);
}

static void BM_FitnessEval_SoA(benchmark::State& state) {
    const int P   = static_cast<int>(state.range(0));
    const int dim = static_cast<int>(state.range(1));

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> d(-5.0, 5.0);

    std::vector<double> genes(P * dim);
    std::vector<double> fitness(P);
    for (auto& g : genes) g = d(rng);

    for (auto _ : state) {
        for (int i = 0; i < P; ++i) {
            // direct offset: no extra dereference, prefetcher sees stride-dim
            Eigen::Map<const Vector> xi(genes.data() + i * dim, dim);
            fitness[i] = xi.squaredNorm();
        }
        benchmark::DoNotOptimize(fitness.data());
    }

    const int64_t bytes = static_cast<int64_t>(P) * (dim + 1) * sizeof(double);
    state.SetBytesProcessed(bytes * state.iterations());
    // kIsIterationInvariantRate = kIsRate with per-iteration divisor:
    //   displayed = bytes_per_iter / (total_time / iterations) = bytes/sec
    state.counters["bytes/s"] = benchmark::Counter(
        static_cast<double>(bytes),
        benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_FitnessEval_AoS)
    ->Args({1000, 100})
    ->Args({2000, 200})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_FitnessEval_SoA)
    ->Args({1000, 100})
    ->Args({2000, 200})
    ->Unit(benchmark::kMicrosecond);

// ── BM2: Eigen::Map vs copy ───────────────────────────────────────────────────
//
// Copy:  Vector x = <map>(data);
//        — triggers Eigen's copy-assignment: malloc(dim×8) + memcpy(dim×8)
//        — one heap allocation and one memcpy per individual per generation
//
// Map:   Eigen::Map<const Vector> xi(data, dim);
//        — stores only (pointer, dim): 16 bytes, no allocation, no copy
//        — squaredNorm() reads directly from the original flat array
//
// The compute (squaredNorm on dim=50 doubles) is identical in both cases;
// the only difference is whether we pay the allocation + copy overhead.
// For P=500, dim=50: copy incurs 500 malloc calls, each ~50–100 ns on
// a cold allocator.  Map incurs zero.

static void BM_MapVsCopy_Copy(benchmark::State& state) {
    const int P   = static_cast<int>(state.range(0));
    const int dim = static_cast<int>(state.range(1));

    std::vector<double> genes(P * dim, 1.0);
    std::vector<double> fitness(P);

    for (auto _ : state) {
        for (int i = 0; i < P; ++i) {
            // copy-assignment allocates a new VectorXd and copies dim doubles
            Vector x = Eigen::Map<const Vector>(genes.data() + i * dim, dim);
            benchmark::DoNotOptimize(x.data());
            fitness[i] = x.squaredNorm();
        }
        benchmark::DoNotOptimize(fitness.data());
    }

    const int64_t bytes = static_cast<int64_t>(P) * dim * sizeof(double);
    state.SetBytesProcessed(bytes * state.iterations());
    // kIsIterationInvariantRate = kIsRate with per-iteration divisor:
    //   displayed = bytes_per_iter / (total_time / iterations) = bytes/sec
    state.counters["bytes/s"] = benchmark::Counter(
        static_cast<double>(bytes),
        benchmark::Counter::kIsIterationInvariantRate);
    state.SetLabel("alloc+memcpy per individual");
}

static void BM_MapVsCopy_Map(benchmark::State& state) {
    const int P   = static_cast<int>(state.range(0));
    const int dim = static_cast<int>(state.range(1));

    std::vector<double> genes(P * dim, 1.0);
    std::vector<double> fitness(P);

    for (auto _ : state) {
        for (int i = 0; i < P; ++i) {
            // Map: 16-byte stack struct; squaredNorm reads the original buffer
            Eigen::Map<const Vector> xi(genes.data() + i * dim, dim);
            fitness[i] = xi.squaredNorm();
        }
        benchmark::DoNotOptimize(fitness.data());
    }

    const int64_t bytes = static_cast<int64_t>(P) * dim * sizeof(double);
    state.SetBytesProcessed(bytes * state.iterations());
    // kIsIterationInvariantRate = kIsRate with per-iteration divisor:
    //   displayed = bytes_per_iter / (total_time / iterations) = bytes/sec
    state.counters["bytes/s"] = benchmark::Counter(
        static_cast<double>(bytes),
        benchmark::Counter::kIsIterationInvariantRate);
    state.SetLabel("zero-copy Map");
}

BENCHMARK(BM_MapVsCopy_Copy)
    ->Args({500,  50})
    ->Args({500, 500})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_MapVsCopy_Map)
    ->Args({500,  50})
    ->Args({500, 500})
    ->Unit(benchmark::kMicrosecond);

// ── BM3: dot product with aligned vs unaligned memory ────────────────────────
//
// SIMD context:
//   AVX2  (256-bit): optimal with 32-byte alignment
//   AVX512 (512-bit): optimal with 64-byte alignment
//
// Eigen's default allocator aligns to max(16, EIGEN_ALIGN_BYTES) bytes.
// With -march=native on AVX2 hardware, EIGEN_ALIGN_BYTES = 32.
//
// Unaligned case: data starts 8 bytes past a 32-byte boundary.
//   The pointer IS 8-byte aligned (safe for double reads on x86) but NOT
//   32-byte aligned, so Eigen uses _mm256_loadu_pd instead of _mm256_load_pd.
//   On Intel Haswell and later, unaligned 256-bit loads have 0 extra latency
//   UNLESS they cross a 64-byte cache line boundary.  At large n, every other
//   32-byte AVX load crosses a cache line → ~1 extra cycle per 2 loads.
//
// Measurable effect grows with n; for n ≤ 256 it is often inside noise.
// bytes/s reported via SetBytesProcessed shows effective memory bandwidth.

static void BM_Dot_Aligned(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));

    Vector a = Vector::Random(n);
    Vector b = Vector::Random(n);
    Scalar result = 0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(result = a.dot(b));
    }

    // dot reads both vectors once: 2 × n × 8 bytes
    const int64_t bytes = 2LL * n * static_cast<int64_t>(sizeof(double));
    state.SetBytesProcessed(bytes * state.iterations());
    // kIsIterationInvariantRate = kIsRate with per-iteration divisor:
    //   displayed = bytes_per_iter / (total_time / iterations) = bytes/sec
    state.counters["bytes/s"] = benchmark::Counter(
        static_cast<double>(bytes),
        benchmark::Counter::kIsIterationInvariantRate);
    state.SetLabel("aligned (Eigen default)");
}

static void BM_Dot_Unaligned(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));

    // Allocate raw bytes; find the nearest 32-byte boundary, then offset by 8.
    // Result: 8-byte-aligned double pointer, NOT 32-byte aligned → Eigen uses
    // unaligned SIMD loads even when compiled with -march=native.
    const int    nbytes  = n * static_cast<int>(sizeof(double)) + 64;
    std::vector<char> buf_a(nbytes), buf_b(nbytes);

    auto misalign = [](char* p) -> double* {
        // Round up to 32-byte boundary, then add 8 bytes of misalignment.
        uintptr_t aligned = (reinterpret_cast<uintptr_t>(p) + 31) & ~uintptr_t{31};
        return reinterpret_cast<double*>(aligned + 8);
    };

    double* pa = misalign(buf_a.data());
    double* pb = misalign(buf_b.data());

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (int i = 0; i < n; ++i) { pa[i] = d(rng); pb[i] = d(rng); }

    // Eigen::Unaligned instructs Eigen to emit unaligned load intrinsics.
    using UMap = Eigen::Map<const Vector, Eigen::Unaligned>;
    UMap ua(pa, n), ub(pb, n);

    Scalar result = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(result = ua.dot(ub));
    }

    const int64_t bytes = 2LL * n * static_cast<int64_t>(sizeof(double));
    state.SetBytesProcessed(bytes * state.iterations());
    // kIsIterationInvariantRate = kIsRate with per-iteration divisor:
    //   displayed = bytes_per_iter / (total_time / iterations) = bytes/sec
    state.counters["bytes/s"] = benchmark::Counter(
        static_cast<double>(bytes),
        benchmark::Counter::kIsIterationInvariantRate);
    state.SetLabel("unaligned (8 B past 32 B boundary)");
}

BENCHMARK(BM_Dot_Aligned)
    ->RangeMultiplier(4)
    ->Range(256, 65536)
    ->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Dot_Unaligned)
    ->RangeMultiplier(4)
    ->Range(256, 65536)
    ->Unit(benchmark::kNanosecond);

} // namespace optlib::bench
