# optlib — C++ Optimization Library

A header-only C++ optimization library featuring Newton's method, L-BFGS, and a Genetic algorithm,
with AVX2 SIMD acceleration for inner products.

## Library Overview

| Component | Location | Description |
|:----------|:---------|:------------|
| Newton optimizer | `include/optlib/methods/newton.hpp` | Second-order method with Hessian, Armijo line search |
| L-BFGS optimizer | `include/optlib/methods/lbfgs.hpp` | Quasi-Newton, limited-memory BFGS, Wolfe line search |
| Genetic optimizer | `include/optlib/methods/genetic.hpp` | Derivative-free, SoA layout, BLX-0.5, tournament selection |
| SIMD dot products | `include/optlib/core/simd.hpp` | `scalar_dot`, `avx_dot`, `avx_dot_unrolled` |
| Types | `include/optlib/core/types.hpp` | `Scalar`, `Vector`, `Matrix` (Eigen wrappers) |

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Requires: Eigen3, Google Benchmark, GTest, a C++20 compiler with AVX2 support (`-march=native`).

## Method Comparison

Test functions:
- **Sphere**: f(x) = ‖x‖², convex, condition number = 1, x₀ = 3·**1**
- **Quadratic**: f(x) = xᵀ diag(1, 10, 10², …, 10ⁿ⁻¹) x, condition number = 10ⁿ⁻¹, x₀ = **1**
- **Rosenbrock**: f(x) = Σ [100(x_{i+1}−x_i²)² + (1−x_i)²], non-convex, x₀ = −**1**

Convergence: ‖∇f‖ < 1e-6 (gradient methods); stagnation < 1e-9 for 30 gens (Genetic).

### Results

| Function   | n  | Method       | Conv | Iters |  f+g  |  H  |    ‖∇f‖    |    f\*      |  ms    |
|:-----------|:--:|:-------------|:----:|------:|------:|----:|-----------:|------------:|-------:|
| Sphere     |  2 | Newton       | yes  |     1 |     3 |   1 |  0.00e+00  |  0.000e+00  |  0.000 |
| Sphere     |  2 | LBFGS m=5    | yes  |     1 |     3 |   0 |  0.00e+00  |  0.000e+00  |  0.000 |
| Sphere     |  2 | LBFGS m=20   | yes  |     1 |     3 |   0 |  0.00e+00  |  0.000e+00  |  0.000 |
| Sphere     |  2 | Genetic      | yes  |    43 |  8600 |   0 |       N/A  |  7.07e-24   |  0.637 |
| Quadratic  |  2 | Newton       | yes  |     1 |     3 |   1 |  0.00e+00  |  0.000e+00  |  0.001 |
| Quadratic  |  2 | LBFGS m=5    | yes  |     8 |    13 |   0 |  5.37e-07  |  1.32e-14   |  0.01  |
| Quadratic  |  2 | LBFGS m=20   | yes  |     9 |    14 |   0 |  1.29e-08  |  4.33e-18   |  0.00  |
| Quadratic  |  2 | Genetic      | yes  |    48 |  9600 |   0 |       N/A  |  2.54e-26   |  0.87  |
| Rosenbrock |  2 | Newton       | yes  |    20 |    48 |  20 |  3.27e-07  |  1.14e-16   |  0.010 |
| Rosenbrock |  2 | LBFGS m=5    | yes  |    25 |    41 |   0 |  2.67e-07  |  9.94e-17   |  0.007 |
| Rosenbrock |  2 | LBFGS m=20   | yes  |    25 |    39 |   0 |  9.05e-09  |  8.55e-20   |  0.009 |
| Rosenbrock |  2 | Genetic      | yes  |   143 | 28600 |   0 |       N/A  |  8.48e-05   |  2.57  |
|            |    |              |      |       |       |     |            |             |        |
| Sphere     | 10 | Newton       | yes  |     1 |     3 |   1 |  0.00e+00  |  0.000e+00  |  0.002 |
| Sphere     | 10 | LBFGS m=5    | yes  |     1 |     3 |   0 |  0.00e+00  |  0.000e+00  |  0.000 |
| Sphere     | 10 | LBFGS m=20   | yes  |     1 |     3 |   0 |  0.00e+00  |  0.000e+00  |  0.000 |
| Sphere     | 10 | Genetic      | yes  |   100 | 30000 |   0 |       N/A  |  1.83e-13   |  4.48  |
| Quadratic  | 10 | Newton       | yes  |     1 |     3 |   1 |  0.00e+00  |  0.000e+00  |  0.002 |
| Quadratic  | 10 | LBFGS m=5    | **no** |  2000 |  2563 |   0 |  7.42e+02  |  5.12e+00   |  0.97  |
| Quadratic  | 10 | LBFGS m=20   | yes  |   218 |   273 |   0 |  9.67e-07  |  7.46e-22   |  0.21  |
| Quadratic  | 10 | Genetic      | yes  |   135 | 40500 |   0 |       N/A  |  5.18e-13   |  6.79  |
| Rosenbrock | 10 | Newton       | yes  |    30 |    64 |  30 |  2.71e-12  |  1.39e-25   |  0.150 |
| Rosenbrock | 10 | LBFGS m=5    | yes  |    58 |    72 |   0 |  8.04e-07  |  9.56e-16   |  0.020 |
| Rosenbrock | 10 | LBFGS m=20   | yes  |    42 |    61 |   0 |  9.24e-07  |  3.25e-16   |  0.028 |
| Rosenbrock | 10 | Genetic      | **no** |  1500 | 450300 |   0 |       N/A  |  4.06e+00   | 74.97  |
|            |    |              |      |       |       |     |            |             |        |
| Sphere     | 50 | Newton       | yes  |     1 |     3 |   1 |  0.00e+00  |  0.000e+00  |  0.033 |
| Sphere     | 50 | LBFGS m=5    | yes  |     1 |     3 |   0 |  0.00e+00  |  0.000e+00  |  0.001 |
| Sphere     | 50 | LBFGS m=20   | yes  |     1 |     3 |   0 |  0.00e+00  |  0.000e+00  |  0.001 |
| Sphere     | 50 | Genetic      | yes  |  1550 | 775000 |   0 |       N/A  |  3.68e-06   | 386.82 |
| Quadratic  | 50 | Newton       | yes  |     1 |     3 |   1 |  0.00e+00  |  0.000e+00  |  0.034 |
| Quadratic  | 50 | LBFGS m=5    | ⚠ yes |     1 |    32 |   0 |  3.73e+89  | 3.47e+129   |  0.01  |
| Quadratic  | 50 | LBFGS m=20   | ⚠ yes |     1 |    32 |   0 |  3.73e+89  | 3.47e+129   |  0.00  |
| Quadratic  | 50 | Genetic      | **no** |  2000 | 1000500 |   0 |       N/A  |  3.27e+12   | 509.97 |
| Rosenbrock | 50 | Newton       | yes  |    89 |   195 |  89 |  2.40e-10  |  7.75e-23   |  8.01  |
| Rosenbrock | 50 | LBFGS m=5    | yes  |    86 |    97 |   0 |  8.56e-07  |  2.03e-15   |  0.043 |
| Rosenbrock | 50 | LBFGS m=20   | yes  |    68 |    81 |   0 |  8.73e-07  |  1.47e-15   |  0.064 |
| Rosenbrock | 50 | Genetic      | **no** |  2000 | 1000500 |   0 |       N/A  |  4.65e+01   | 583.75 |

**⚠ LBFGS / Quadratic n=50**: Reports `converged=yes` but ‖∇f‖ ≈ 3.7e+89 — false convergence.
The condition number is 10⁴⁹; Wolfe line search produces α ≈ 0, triggering `tol_step` termination
before any real progress. Treat results as meaningless for cond(A) >> 1/ε_mach.

### Google Benchmark Timing (wall-clock per minimization, ms)

| Benchmark                  |  n=2   |  n=10  |  n=50  |
|:---------------------------|-------:|-------:|-------:|
| Newton / Sphere            | 0.0000 | 0.002  | 0.033  |
| Newton / Quadratic         | 0.001  | 0.002  | 0.034  |
| Newton / Rosenbrock        | 0.010  | 0.150  | 8.01   |
| LBFGS m=5 / Sphere         | 0.0000 | 0.0000 | 0.001  |
| LBFGS m=5 / Rosenbrock     | 0.007  | 0.020  | 0.043  |
| LBFGS m=20 / Rosenbrock    | 0.009  | 0.028  | 0.064  |
| Genetic / Sphere           | 0.637  | 4.14   | —      |

Newton on Rosenbrock n=50 takes **8.0 ms** (89 Hessian factorisations of a 50×50 matrix).
LBFGS m=5 on the same problem takes **0.043 ms** — **186× faster**.

---

## When to Use Which Method

### Newton's Method

**Use when:**
- The function is smooth and twice differentiable, and computing the Hessian is affordable.
- Dimension n is small to medium (n ≲ 100).
- You need fast, reliable convergence to high precision.

**Evidence from benchmarks:**
- Converges in **1 iteration** on any quadratic f = xᵀAx with PD A, regardless of condition number
  (because d = −H⁻¹g = −A⁻¹(Ax) = −x, so the step lands exactly at the minimum).
- Converges in 30 iterations on Rosenbrock 10D, 89 iterations on 50D.
- At n=50, each iteration costs O(n³) = O(125 000) operations for the Cholesky solve, making
  8 ms per minimisation for Rosenbrock 50D.

**Avoid when:**
- n > a few hundred (Hessian storage O(n²), factorisation O(n³) become prohibitive).
- The Hessian is expensive to compute or unavailable.
- The function is non-smooth or multimodal.

---

### L-BFGS (m = 5)

**Use when:**
- n is large (hundreds to millions of dimensions).
- Only gradients are available.
- The problem is well-conditioned or mildly ill-conditioned.

**Evidence from benchmarks:**
- Matches Newton on Rosenbrock at all sizes but is **186× faster at n=50** (0.043 ms vs 8.01 ms).
- Converges in 1 iteration on Sphere (exactly like Newton).
- **Fails on Quadratic κ=10, n=10**: condition number 10⁹ is too high for m=5 history — does not
  converge in 2000 iterations (‖∇f‖ = 742 at termination).

**Avoid when:**
- The problem is highly ill-conditioned (cond ≫ 10⁶) and m is small. Increase m or switch to Newton.
- For cond ≈ 10⁴⁹ (Quadratic n=50) all quasi-Newton methods break down due to floating-point limits.

---

### L-BFGS (m = 20)

**Use when:**
- The function is ill-conditioned and you cannot afford Newton, but need more curvature history than m=5.

**Evidence from benchmarks:**
- Solves Quadratic κ=10 n=10 (where m=5 fails): converges in 218 iterations vs m=5's 2000 DNF.
- Converges slightly faster than m=5 on Rosenbrock 10D (42 vs 58 iterations).
- Marginally slower per iteration than m=5 due to larger two-loop recursion, but wall-clock
  time is comparable (0.064 ms vs 0.043 ms at n=50 Rosenbrock).

**Rule of thumb:** start with m=10. Increase to m=20–50 if you observe slow convergence
or non-convergence on ill-conditioned problems.

---

### Genetic Algorithm

**Use when:**
- The objective is **non-differentiable**, **stochastic**, or **multimodal**.
- A derivative-free global search is required.
- Precision requirements are modest (‖x − x\*‖ ∼ 10⁻⁶ or worse).

**Evidence from benchmarks:**
- Finds Sphere minimum to ≈ 10⁻¹³ at n=10 in 4.5 ms — competitive with gradient methods for
  small n when gradients are unavailable.
- **Fails to converge on Rosenbrock n=10**: gets stuck at f ≈ 4.06 after 1500 generations and
  450 000 function evaluations (75 s).
- Scales very poorly: Sphere n=50 takes 387 ms, vs 1 µs for LBFGS.
- Each function evaluation is O(n), and total cost is O(P × G × n). For P=500, G=2000, n=50:
  50 million dimension-50 evaluations.

**Avoid when:**
- n > ~10 for smooth problems: gradient methods are orders of magnitude faster and more reliable.
- High precision is required (best achievable ≈ 10⁻⁶ to 10⁻¹³ depending on n and generations).

---

### Summary Decision Table

| Situation | Recommended method |
|:----------|:------------------|
| Smooth, n ≤ 100, Hessian available | Newton |
| Smooth, n > 100, gradient available | L-BFGS m=10–20 |
| Smooth, ill-conditioned (cond ≈ 10⁶–10¹²) | L-BFGS m=20+ or Newton if n is small |
| Smooth, cond > 10¹² | Newton only (quasi-Newton methods break down) |
| Non-differentiable / multimodal / black-box, n ≤ 10 | Genetic |
| Non-differentiable, n > 10 | Genetic with large population, or specialised methods |

---

## SIMD Dot Product Performance

Four implementations in `include/optlib/core/simd.hpp`:

| Implementation | Strategy | Typical throughput |
|:---------------|:---------|:------------------|
| `scalar_dot` | Plain C loop | ~2–4 GFLOPS |
| `eigen_dot` | Eigen (AVX-512 + multiple acc.) | ~37 GFLOPS (L1 ceiling) |
| `avx_dot` | Manual AVX2, 1 accumulator | ~7–16 GFLOPS |
| `avx_dot_unrolled` | Manual AVX2, 4 accumulators | ~15–22 GFLOPS (L2 ceiling) |

The dot product has arithmetic intensity AI = 2n / (16n bytes) = 0.125 FLOP/byte.
Memory bandwidth ceiling: L1 ~300 GB/s → 37.5 GFLOPS, L2 ~160 GB/s → 20 GFLOPS.
Single-accumulator AVX2 is latency-limited (~6.7 GFLOPS theoretical); four accumulators
hide the FMA latency and approach the L2 bandwidth ceiling.

Run `./optlib_bench --benchmark_filter=BM_Dot` for roofline measurements on your hardware.

---

## Running the Benchmarks

```bash
# Method comparison table only
./build/benchmarks/optlib_methods --benchmark_filter='^$'

# Table + all Google Benchmark timings
./build/benchmarks/optlib_methods --benchmark_filter=.

# SIMD and cache benchmarks
./build/benchmarks/optlib_bench --benchmark_counters_tabular=true

# Run tests
cd build && ctest --output-on-failure
```
