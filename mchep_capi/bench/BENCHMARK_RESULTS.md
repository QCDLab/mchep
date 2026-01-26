# MCHEP vs CUBA Vegas Benchmark Results

This document summarizes the performance comparison between MCHEP and CUBA Vegas
for Monte Carlo integration.

## Test Configuration

- **Integrand**: Narrow Gaussian centered at (0.5, 0.5, 0.5, 0.5)
  ```
  f(x) = exp(-100 * sum((x[i] - 0.5)^2)) * 1013.2118364296088
  ```
- **Integration domain**: [0, 1]^4 (4-dimensional unit hypercube)
- **Exact integral value**: 1.0
- **Runs per measurement**: 5 (averaged)

### MCHEP Configuration
- Iterations: 100 (upper bound, exits early when accuracy reached)
- Evaluations per iteration: 100,000
- Grid bins: 50
- Alpha (grid damping): 1.5

### CUBA Configuration
- Max evaluations: 10,000,000 (upper bound)
- Start evaluations: 10,000
- Increase per iteration: 1,000

### Parallelization

| Library | Threading Model | Default Threads | Control |
|---------|-----------------|-----------------|---------|
| MCHEP | Rayon (work-stealing) | All available cores | `RAYON_NUM_THREADS` |
| CUBA | Fork (master-worker) | All available cores | `CUBACORES` |

- **System tested**: 16 CPU cores
- Both libraries support parallelization, but performance depends on integrand cost

### Parallel Scaling (Expensive Integrand)

Tested with an expensive integrand (~100 trig operations per evaluation, ~1M total
evaluations):

| Cores | MCHEP (ms) | CUBA (ms) | MCHEP Speedup |
|-------|------------|-----------|---------------|
| 1 | 1,897 | 1,845 | 0.97x (CUBA faster) |
| 2 | 878 | 960 | 1.09x |
| 4 | 463 | 643 | 1.39x |
| 8 | 322 | 497 | 1.54x |
| 16 | 250 | 376 | 1.50x |

**Parallel efficiency** (speedup from 1→16 cores):
- MCHEP: 7.6x (47% efficiency)
- CUBA: 4.9x (31% efficiency)

### Parallel Scaling (Cheap Integrand)

With a cheap integrand (simple Gaussian), parallelization overhead dominates:

| Cores | MCHEP (ms) | CUBA (ms) |
|-------|------------|-----------|
| 1 | 140 | 50 |
| 16 | 52 | 90 |

**Key insight**: CUBA's fork-based parallelization has higher overhead, so it actually
slows down with cheap integrands. MCHEP's Rayon parallelization benefits even cheap
integrands.

## Benchmark 1: Fixed Iterations

When running both integrators for approximately the same number of function evaluations
(~1,000,000), **MCHEP is faster** due to lower overhead and efficient parallelization.

### Timing Results

| Method | Average Time | Speedup vs CUBA |
|--------|--------------|-----------------|
| MCHEP Scalar | 52 ms | 2.4x faster |
| MCHEP SIMD | 44 ms | 2.9x faster |
| CUBA Vegas | 126 ms | baseline |

### Integration Results

| Method | Result | Error | Relative Error |
|--------|--------|-------|----------------|
| MCHEP Scalar | 0.999600 | 0.000628 | 0.063% |
| MCHEP SIMD | 0.999520 | 0.000644 | 0.064% |
| CUBA Vegas | 1.000020 | 0.000298 | 0.030% |
| **Expected** | **1.0** | - | - |

### Observations

- MCHEP is **2.4-2.9x faster** than CUBA for the same number of evaluations
- CUBA achieves slightly better precision (~2x smaller error) for the same evaluation
  count
- All methods converge to the correct value within error bounds

### Single-Threaded Comparison

To provide a fair comparison, here are results with both libraries restricted to 1 thread:

| Method | Time (1 thread) | Time (16 threads) | Parallel Speedup |
|--------|-----------------|-------------------|------------------|
| MCHEP Scalar | 140 ms | 52 ms | 2.7x |
| MCHEP SIMD | 180 ms | 44 ms | 4.1x |
| CUBA Vegas | 50 ms | 90 ms | 0.6x (slower!) |

**Key insight**: When single-threaded, CUBA is faster than MCHEP for cheap integrands
(50ms vs 140ms). However, CUBA's fork-based parallelization has such high overhead
that it actually slows down with more cores for cheap integrands, while MCHEP continues
to benefit.

## Benchmark 2: Target Accuracy

This benchmark measures the time to reach a specified relative error target for
a reasonably simple integrand.

### Results

| Target Accuracy | MCHEP Scalar | MCHEP SIMD | CUBA Vegas | Fastest |
|-----------------|--------------|------------|------------|---------|
| 5.0% | 11.2 ms | 10.7 ms | 10.8 ms | MCHEP SIMD |
| 2.0% | 11.8 ms | 10.0 ms | 11.6 ms | MCHEP SIMD |
| 1.0% | 10.8 ms | 9.1 ms | 12.1 ms | MCHEP SIMD |
| 0.5% | 15.4 ms | 14.5 ms | 13.8 ms | CUBA |
| 0.1% | 30.2 ms | 27.6 ms | 21.3 ms | CUBA |

### Achieved Accuracy

| Target | MCHEP Scalar | MCHEP SIMD | CUBA Vegas |
|--------|--------------|------------|------------|
| 5.0% | 0.55% | 0.56% | 1.55% |
| 2.0% | 0.55% | 0.56% | 1.55% |
| 1.0% | 0.55% | 0.56% | 0.56% |
| 0.5% | 0.31% | 0.30% | 0.30% |
| 0.1% | 0.08% | 0.09% | 0.10% |

### Integral Values

All methods converge to the correct value:

| Target | MCHEP Scalar | MCHEP SIMD | CUBA Vegas | Expected |
|--------|--------------|------------|------------|----------|
| 5.0% | 1.002998 | 1.002295 | 0.995119 | 1.0 |
| 2.0% | 1.002998 | 1.002295 | 0.995119 | 1.0 |
| 1.0% | 1.002998 | 1.002295 | 0.999613 | 1.0 |
| 0.5% | 1.000131 | 0.998474 | 0.999981 | 1.0 |
| 0.1% | 0.999711 | 0.999411 | 1.000400 | 1.0 |

## Benchmark 3: Scaling Analysis

This benchmark measures throughput (evaluations per millisecond) as various
parameters change.

### Scaling with Number of Evaluations

How throughput changes with batch size (evaluations per iteration for MCHEP,
max evaluations for CUBA).

| Total Evaluations | MCHEP (evals/ms) | CUBA (evals/ms) | MCHEP Speedup |
|-------------------|------------------|-----------------|---------------|
| 10,000 | 2,169 | 922 | 2.4x |
| 50,000 | 6,442 | 4,301 | 1.5x |
| 100,000 | 10,068 | 6,007 | 1.7x |
| 500,000 | 18,987 | 10,504 | 1.8x |
| 1,000,000 | 22,146 | 10,966 | 2.0x |
| 5,000,000 | 28,987 | 11,726 | 2.5x |
| 10,000,000 | 31,657 | 11,775 | 2.7x |

**Key insight**: MCHEP throughput continues to improve with larger batch sizes
(better parallelization efficiency), while CUBA plateaus around 11-12k evals/ms.

### Scaling with Number of Iterations

MCHEP throughput as the number of iterations increases (fixed 100k evals/iter):

| Iterations | Total Evals | Time (ms) | Throughput (evals/ms) |
|------------|-------------|-----------|----------------------|
| 1 | 100,000 | 4.9 | 20,502 |
| 2 | 200,000 | 6.9 | 28,797 |
| 5 | 500,000 | 17.0 | 29,431 |
| 10 | 1,000,000 | 41.1 | 24,358 |
| 20 | 2,000,000 | 85.2 | 23,465 |
| 50 | 5,000,000 | 213.7 | 23,398 |
| 100 | 10,000,000 | 426.3 | 23,457 |

**Key insight**: Throughput is fairly constant (~23-29k evals/ms) regardless of iteration
count, showing good scaling with no significant overhead per iteration.

### Scaling with Dimensions

How throughput changes with the number of integration dimensions (fixed 1M evaluations):

| Dimensions | MCHEP (evals/ms) | CUBA (evals/ms) | MCHEP Speedup |
|------------|------------------|-----------------|---------------|
| 2 | 35,513 | 11,902 | 3.0x |
| 3 | 27,518 | 11,437 | 2.4x |
| 4 | 23,689 | 11,090 | 2.1x |
| 5 | 19,834 | 9,964 | 2.0x |
| 6 | 18,475 | 9,669 | 1.9x |
| 8 | 13,852 | 9,110 | 1.5x |
| 10 | 10,990 | 8,319 | 1.3x |

**Key insight**: MCHEP has higher throughput at all dimensions but scales worse with
increasing dimensions (3.2x slowdown from 2D to 10D) compared to CUBA (1.4x slowdown).
At 10 dimensions, the speedup narrows to 1.3x.

### Scaling Summary

| Parameter | MCHEP Behavior | CUBA Behavior |
|-----------|----------------|---------------|
| Batch size | Throughput improves with larger batches | Plateaus at ~12k evals/ms |
| Iterations | Constant throughput (~23k evals/ms) | N/A |
| Dimensions | Significant slowdown at high dim | Modest slowdown |

## Key Findings

1. **Lower precision targets (1-5%)**: MCHEP is faster, with SIMD providing an additional
  ~10% speedup over scalar. MCHEP SIMD achieves targets in ~9-11ms vs CUBA's 11-12ms.

2. **Higher precision targets (≤0.5%)**: CUBA Vegas becomes more efficient for reasonably
  simple integrands. At 0.1% target accuracy, CUBA is ~25-30% faster than MCHEP.

3. **Crossover point**: The performance crossover occurs around 0.5% target accuracy.

4. **Early stopping behavior**: MCHEP tends to overshoot accuracy targets (achieving better
  precision than requested), while CUBA stops closer to the target. This is due to MCHEP
  checking accuracy only after completing full iterations.

5. **Numerical accuracy**: Both libraries produce results within the expected error bounds,
  with no systematic bias.

## Running the Benchmarks

```bash
# Fixed iterations benchmark
./benchmark.sh

# Accuracy-based benchmark
./benchmark_accuracy.sh

# Scaling benchmarks
./benchmark_scaling evals    # Scaling with evaluations
./benchmark_scaling iters    # Scaling with iterations
./benchmark_scaling dims     # Scaling with dimensions
./benchmark_scaling all      # Run all scaling tests

./cuba_scaling evals         # CUBA scaling with evaluations
./cuba_scaling dims          # CUBA scaling with dimensions
./cuba_scaling all           # Run all CUBA scaling tests

# Thread measurement
./measure_threads.sh
```

## Files

- `benchmark_mchep.cpp` - MCHEP scalar, fixed iterations
- `benchmark_mchep_simd.cpp` - MCHEP SIMD, fixed iterations
- `cuba_example.cpp` - CUBA Vegas, fixed iterations
- `benchmark_mchep_accuracy.cpp` - MCHEP scalar, accuracy-based
- `benchmark_mchep_simd_accuracy.cpp` - MCHEP SIMD, accuracy-based
- `cuba_accuracy.cpp` - CUBA Vegas, accuracy-based
- `benchmark_scaling.cpp` - MCHEP scaling benchmark
- `cuba_scaling.cpp` - CUBA scaling benchmark
- `mchep_parallel_test.cpp` - MCHEP parallel scaling test (expensive integrand)
- `cuba_parallel_test.cpp` - CUBA parallel scaling test (expensive integrand)
- `measure_threads.sh` - Thread usage measurement script
