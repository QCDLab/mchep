# SIMD Integration Tutorial for MCHEP C/C++ API

This tutorial explains how to use MCHEP's SIMD (Single Instruction Multiple Data)
integration capability to accelerate Monte Carlo integration in C/C++ applications.

## Overview

MCHEP's SIMD integration evaluates **4 integration points simultaneously**, which
can significantly improve performance by:
- Better CPU cache utilization
- Enabling compiler auto-vectorization
- Reducing function call overhead

## Prerequisites

- MCHEP C API installed (see main README)
- C++11 or later compiler
- `pkg-config` configured for `mchep_capi`

## Quick Start

Here's a minimal example that integrates a 2D Gaussian:

```cpp
#include <mchep.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>

// SIMD integrand: evaluates 4 points at once
std::array<double, 4> gaussian_simd(const std::vector<double>& x) {
    std::array<double, 4> results;
    const int dim = 2;
    const int simd_width = 4;

    for (int i = 0; i < simd_width; ++i) {
        double sum = 0.0;
        for (int d = 0; d < dim; ++d) {
            double val = x[d * simd_width + i];  // Note the memory layout!
            sum += val * val;
        }
        results[i] = std::exp(-sum);
    }
    return results;
}

int main() {
    // Integration boundaries: [0, 1] x [0, 1]
    std::vector<std::pair<double, double>> boundaries = {
        {0.0, 1.0},
        {0.0, 1.0}
    };

    // Create integrator: 10 iterations, 100k evals/iter, 50 bins, alpha=1.5
    mchep::Vegas vegas(10, 100000, 50, 1.5, boundaries);

    // Run SIMD integration
    VegasResult result = vegas.integrate_simd(gaussian_simd);

    std::cout << "Result: " << result.value << " +/- " << result.error << std::endl;
    return 0;
}
```

Compile with:
```bash
c++ -std=c++11 -O3 example.cpp $(pkg-config --cflags --libs mchep_capi) -o example
```

## Understanding the SIMD Memory Layout

The key difference between scalar and SIMD integration is the **memory layout** of
input points.

### Scalar Layout (Array of Structures - AoS)
```
Point 0: [x0, y0, z0]
Point 1: [x1, y1, z1]
Point 2: [x2, y2, z2]
Point 3: [x3, y3, z3]
```

### SIMD Layout (Structure of Arrays - SoA)
```
[x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3]
|--------------| |--------------| |--------------|
  dimension 0      dimension 1      dimension 2
```

The formula to access coordinate `d` of point `i` is:
```cpp
x[d * simd_width + i]  // where simd_width = 4
```

### Visual Example (3D, 4 points)

```
Input vector x (size = dim * 4 = 12):

Index:  0    1    2    3    4    5    6    7    8    9   10   11
      [x0 , x1 , x2 , x3 , y0 , y1 , y2 , y3 , z0 , z1 , z2 , z3]
       └─────────────────┘ └─────────────────┘ └─────────────────┘
            dim 0              dim 1               dim 2

To get point i's coordinates:
  Point 0: x[0], x[4], x[8]  → (x0, y0, z0)
  Point 1: x[1], x[5], x[9]  → (x1, y1, z1)
  Point 2: x[2], x[6], x[10] → (x2, y2, z2)
  Point 3: x[3], x[7], x[11] → (x3, y3, z3)
```

## Complete Example: Physics Integrand

Here's a more realistic example computing a phase space integral:

```cpp
#include <mchep.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>

// Physical constants
constexpr double PI = 3.14159265358979323846;
constexpr double MASS = 1.0;

// SIMD integrand for a 4D phase space integral
// Integrates: exp(-E) * sin(theta) over [0,inf) x [0,pi] x [0,2pi] x [0,inf)
std::array<double, 4> phase_space_simd(const std::vector<double>& x) {
    std::array<double, 4> results;
    constexpr int dim = 4;
    constexpr int simd_width = 4;

    for (int i = 0; i < simd_width; ++i) {
        // Extract coordinates for point i
        double p     = x[0 * simd_width + i];  // momentum magnitude [0, 10]
        double theta = x[1 * simd_width + i];  // polar angle [0, pi]
        double phi   = x[2 * simd_width + i];  // azimuthal angle [0, 2pi]
        double E     = x[3 * simd_width + i];  // energy [0, 10]

        // Phase space element with Boltzmann suppression
        double jacobian = p * p * std::sin(theta);
        double boltzmann = std::exp(-E);

        results[i] = jacobian * boltzmann;
    }
    return results;
}

int main() {
    // 4D integration boundaries
    std::vector<std::pair<double, double>> boundaries = {
        {0.0, 10.0},    // p: momentum
        {0.0, PI},      // theta: polar angle
        {0.0, 2*PI},    // phi: azimuthal angle
        {0.0, 10.0}     // E: energy
    };

    // Create Vegas integrator
    mchep::Vegas vegas(
        20,       // iterations
        200000,   // evaluations per iteration
        50,       // grid bins
        1.5,      // alpha (grid adaptation speed)
        boundaries
    );

    // Set seed for reproducibility
    vegas.set_seed(42);

    // Integrate with 0.5% target accuracy
    VegasResult result = vegas.integrate_simd(phase_space_simd, 0.5);

    std::cout << "Phase space integral:" << std::endl;
    std::cout << "  Value: " << result.value << std::endl;
    std::cout << "  Error: " << result.error << std::endl;
    std::cout << "  Relative error: " << (result.error/result.value)*100 << "%" << std::endl;
    std::cout << "  Chi2/dof: " << result.chi2_dof << std::endl;

    return 0;
}
```

## Converting Scalar to SIMD Integrand

If you have an existing scalar integrand:

```cpp
// Scalar version
double my_integrand(const std::vector<double>& x) {
    double result = 0.0;
    for (size_t d = 0; d < x.size(); ++d) {
        result += x[d] * x[d];
    }
    return std::exp(-result);
}
```

Convert it to SIMD by:

1. Changing the return type to `std::array<double, 4>`
2. Adding an outer loop over 4 points
3. Changing the index from `x[d]` to `x[d * 4 + i]`

```cpp
// SIMD version
std::array<double, 4> my_integrand_simd(const std::vector<double>& x) {
    std::array<double, 4> results;
    const int dim = x.size() / 4;  // Total size is dim * 4

    for (int i = 0; i < 4; ++i) {           // Loop over 4 points
        double sum = 0.0;
        for (int d = 0; d < dim; ++d) {
            double val = x[d * 4 + i];       // Changed indexing
            sum += val * val;
        }
        results[i] = std::exp(-sum);
    }
    return results;
}
```

## Using VegasPlus with SIMD

VegasPlus adds adaptive stratified sampling on top of Vegas, which can improve
convergence for integrands with localized peaks. It also supports SIMD integration.

### VegasPlus Parameters

VegasPlus has two additional parameters compared to Vegas:
- `n_strat`: Number of stratifications per dimension (total hypercubes = n_strat^dim)
- `beta`: Stratification adaptation parameter (typically 0.5-0.75)

### Example

```cpp
#include <mchep.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>

std::array<double, 4> peaked_integrand_simd(const std::vector<double>& x) {
    std::array<double, 4> results;
    const int dim = 4;
    const int simd_width = 4;

    for (int i = 0; i < simd_width; ++i) {
        double dx2 = 0.0;
        for (int d = 0; d < dim; ++d) {
            double val = x[d * simd_width + i];
            double diff = val - 0.5;
            dx2 += diff * diff;
        }
        // Narrow Gaussian peak at center
        results[i] = std::exp(-100.0 * dx2) * 1013.2118364296088;
    }
    return results;
}

int main() {
    std::vector<std::pair<double, double>> boundaries = {
        {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}
    };

    // Create VegasPlus integrator
    mchep::VegasPlus vegasplus(
        10,       // n_iter: iterations
        100000,   // n_eval: evaluations per iteration
        50,       // n_bins: grid bins
        1.5,      // alpha: grid adaptation
        2,        // n_strat: stratifications per dimension (2^4 = 16 hypercubes)
        0.75,     // beta: stratification adaptation
        boundaries
    );

    vegasplus.set_seed(1234);

    // Run SIMD integration
    VegasResult result = vegasplus.integrate_simd(peaked_integrand_simd, -1.0);

    std::cout << "VegasPlus SIMD Result: " << result.value
              << " +/- " << result.error << std::endl;

    return 0;
}
```

### When to Use VegasPlus

| Integrand Type | Recommended |
|----------------|-------------|
| Smooth, no peaks | Vegas |
| Single localized peak | VegasPlus |
| Multiple peaks | VegasPlus with higher n_strat |
| Very high dimensions (>6) | Vegas (stratification overhead grows as n_strat^dim) |

### Performance Comparison

For a narrow Gaussian in 4D (1M evaluations):

| Method | Time | Result |
|--------|------|--------|
| Vegas SIMD | 39 ms | 0.99952 ± 0.00064 |
| VegasPlus SIMD | 37 ms | 1.00017 ± 0.00062 |

VegasPlus achieves slightly better precision for peaked integrands.

## Using the C API Directly

If you prefer the C API over the C++ wrapper:

```c
#include <mchep_capi.h>
#include <stdio.h>
#include <math.h>

// C-style SIMD integrand
void my_simd_integrand(const double* x, int dim, void* user_data, double* result) {
    for (int i = 0; i < 4; ++i) {
        double sum = 0.0;
        for (int d = 0; d < dim; ++d) {
            double val = x[d * 4 + i];
            sum += val * val;
        }
        result[i] = exp(-sum);
    }
}

int main() {
    CBoundary boundaries[] = {{0.0, 1.0}, {0.0, 1.0}};

    VegasC* vegas = mchep_vegas_new(10, 100000, 50, 1.5, 2, boundaries);
    mchep_vegas_set_seed(vegas, 1234);

    VegasResult result = mchep_vegas_integrate_simd(
        vegas,
        my_simd_integrand,
        NULL,   // user_data (optional)
        -1.0    // target_accuracy (-1 = disabled)
    );

    printf("Result: %f +/- %f\n", result.value, result.error);

    mchep_vegas_free(vegas);
    return 0;
}
```

## Performance Tips

### 1. Enable Compiler Optimizations

```bash
c++ -O3 -march=native -ffast-math example.cpp ...
```

### 2. Minimize Branching Inside the Loop

Bad:
```cpp
for (int i = 0; i < 4; ++i) {
    if (x[i] > 0.5) {  // Branch inside loop - bad for SIMD
        results[i] = func1(x[i]);
    } else {
        results[i] = func2(x[i]);
    }
}
```

Better:
```cpp
for (int i = 0; i < 4; ++i) {
    // Branchless: compute both, blend result
    double v1 = func1(x[d * 4 + i]);
    double v2 = func2(x[d * 4 + i]);
    double t = (x[d * 4 + i] > 0.5) ? 1.0 : 0.0;
    results[i] = t * v1 + (1.0 - t) * v2;
}
```

### 3. Use Explicit SIMD Intrinsics (Advanced)

For maximum performance, use CPU intrinsics:

```cpp
#include <immintrin.h>  // AVX

std::array<double, 4> gaussian_avx(const std::vector<double>& x) {
    std::array<double, 4> results;
    const int dim = x.size() / 4;

    __m256d sum = _mm256_setzero_pd();

    for (int d = 0; d < dim; ++d) {
        __m256d val = _mm256_loadu_pd(&x[d * 4]);
        sum = _mm256_fmadd_pd(val, val, sum);  // sum += val * val
    }

    // exp(-sum) - need to extract and compute
    _mm256_storeu_pd(results.data(), sum);
    for (int i = 0; i < 4; ++i) {
        results[i] = std::exp(-results[i]);
    }

    return results;
}
```

For a complete heavy HEP-style example with 1000 FMA operations achieving **2-2.5x speedup**,
see [benchmark_simd_avx.cpp](../mchep_capi/bench/benchmark_simd_avx.cpp).

## Comparison: Scalar vs SIMD

| Aspect | Scalar | SIMD |
|--------|--------|------|
| Function signature | `double f(vector<double>)` | `array<double,4> f(vector<double>)` |
| Input size | `dim` | `dim * 4` |
| Memory layout | AoS | SoA |
| Points per call | 1 | 4 |
| Typical speedup | baseline | 1.1x - 2x |

## Troubleshooting

### Wrong Results

If you get incorrect results, check:
1. **Memory layout**: Make sure you're using `x[d * 4 + i]`, not `x[i * dim + d]`
2. **Vector size**: Input vector size should be `dim * 4`
3. **All 4 results**: Make sure you compute all 4 results, not just the first one

### No Performance Improvement

If SIMD isn't faster:
1. Enable `-O3` optimization
2. Check if your integrand is already memory-bound (SIMD helps compute-bound code)
3. Profile to find the actual bottleneck

## API Reference

### C++ API

```cpp
// Create Vegas integrator
mchep::Vegas vegas(n_iter, n_eval, n_bins, alpha, boundaries);

// Create VegasPlus integrator (with stratified sampling)
mchep::VegasPlus vegasplus(n_iter, n_eval, n_bins, alpha, n_strat, beta, boundaries);

// SIMD integration (same for both Vegas and VegasPlus)
VegasResult result = vegas.integrate_simd(integrand_func, target_accuracy);
VegasResult result = vegasplus.integrate_simd(integrand_func, target_accuracy);

// Result structure
struct VegasResult {
    double value;     // Estimated integral
    double error;     // Statistical error (1 sigma)
    double chi2_dof;  // Chi-squared per degree of freedom
};
```

### C API

```c
// SIMD integrand signature
typedef void (*CSimdIntegrand)(const double* x, int dim, void* user_data, double* result);

// Vegas SIMD integration
VegasC* mchep_vegas_new(n_iter, n_eval, n_bins, alpha, dim, boundaries);
VegasResult mchep_vegas_integrate_simd(vegas, integrand, user_data, target_accuracy);
void mchep_vegas_free(vegas);

// VegasPlus SIMD integration
VegasPlusC* mchep_vegas_plus_new(n_iter, n_eval, n_bins, alpha, n_strat, beta, dim, boundaries);
VegasResult mchep_vegas_plus_integrate_simd(vegasplus, integrand, user_data, target_accuracy);
void mchep_vegas_plus_free(vegasplus);
```

## Further Reading

- [MCHEP Benchmark Results](mchep_capi/bench/BENCHMARK_RESULTS.md) - Extensive Performance comparisons
- [Vegas Algorithm Paper](https://arxiv.org/pdf/2009.05112) - Theoretical background
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/) - For advanced SIMD optimization
