/**
 * SIMD AVX Benchmark for MCHEP
 *
 * This benchmark demonstrates the performance benefit of using AVX intrinsics
 * with MCHEP's SIMD integration mode for heavy HEP-style integrands.
 *
 * Compile with:
 *   c++ -std=c++14 -O3 -march=native -mavx2 -ffast-math benchmark_simd_avx.cpp \
 *       $(pkg-config --cflags --libs mchep_capi) -lm -o benchmark_simd_avx
 *
 * Or use the Makefile:
 *   make benchmark_simd_avx
 */

#include <mchep.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <chrono>
#include <immintrin.h>

// Pure AVX integrand - no scalar operations in hot path
// Simulates heavy HEP matrix element calculation with 1000 FMA operations
std::array<double, 4> hep_integrand_avx(const std::vector<double>& x) {
    std::array<double, 4> results;

    // Load all 4 points for each dimension using AVX
    // Memory layout is SoA: [x0,x1,x2,x3, y0,y1,y2,y3, z0,z1,z2,z3, w0,w1,w2,w3]
    __m256d x0 = _mm256_loadu_pd(&x[0]);   // dim 0: 4 points
    __m256d x1 = _mm256_loadu_pd(&x[4]);   // dim 1: 4 points
    __m256d x2 = _mm256_loadu_pd(&x[8]);   // dim 2: 4 points
    __m256d x3 = _mm256_loadu_pd(&x[12]);  // dim 3: 4 points

    __m256d half = _mm256_set1_pd(0.5);

    // Compute (x - 0.5)^2 for all dimensions, all 4 points simultaneously
    __m256d d0 = _mm256_sub_pd(x0, half);
    __m256d d1 = _mm256_sub_pd(x1, half);
    __m256d d2 = _mm256_sub_pd(x2, half);
    __m256d d3 = _mm256_sub_pd(x3, half);

    // Sum of squares: sum = d0^2 + d1^2 + d2^2 + d3^2
    __m256d sum = _mm256_mul_pd(d0, d0);
    sum = _mm256_fmadd_pd(d1, d1, sum);
    sum = _mm256_fmadd_pd(d2, d2, sum);
    sum = _mm256_fmadd_pd(d3, d3, sum);

    // Heavy computation: 1000 FMA operations (simulating |M|^2 calculation)
    __m256d acc = _mm256_set1_pd(1.0);
    __m256d term = sum;
    __m256d coef = _mm256_set1_pd(0.999);
    for (int i = 0; i < 1000; i++) {
        acc = _mm256_fmadd_pd(term, coef, acc);
        term = _mm256_mul_pd(term, coef);
    }

    // Polynomial approximation for exp-like decay (avoids scalar exp bottleneck)
    // Taylor expansion: e^x ≈ 1 + x + x^2/2 + x^3/6
    __m256d neg_sum = _mm256_mul_pd(sum, _mm256_set1_pd(-10.0));
    __m256d one = _mm256_set1_pd(1.0);
    __m256d result = _mm256_fmadd_pd(neg_sum, _mm256_set1_pd(1.0/6.0), one);
    result = _mm256_fmadd_pd(result, neg_sum, one);
    result = _mm256_fmadd_pd(result, neg_sum, one);
    result = _mm256_mul_pd(result, acc);
    result = _mm256_max_pd(result, _mm256_setzero_pd());  // clamp to >= 0

    _mm256_storeu_pd(results.data(), result);
    return results;
}

// Scalar version for comparison (same computation, but one point at a time)
double hep_integrand_scalar(const std::vector<double>& x) {
    double dx2 = 0.0;
    for (int d = 0; d < 4; ++d) {
        double diff = x[d] - 0.5;
        dx2 += diff * diff;
    }

    // Same 1000 iterations as AVX version
    double acc = 1.0;
    double term = dx2;
    for (int i = 0; i < 1000; i++) {
        acc += term * 0.999;
        term *= 0.999;
    }

    // Same polynomial approximation
    double neg_sum = -10.0 * dx2;
    double result = 1.0 + neg_sum * (1.0 + neg_sum * (1.0 + neg_sum / 6.0));
    return std::max(0.0, result * acc);
}

int main() {
    std::vector<std::pair<double, double>> boundaries = {
        {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}
    };

    const int n_iter = 20;
    const int n_eval = 100000;
    const int n_runs = 3;

    std::cout << "=== MCHEP SIMD+AVX Benchmark ===" << std::endl;
    std::cout << "Heavy HEP-style integrand with 1000 FMAs per evaluation" << std::endl;
    std::cout << n_iter << " iterations, " << n_eval << " evals/iter, "
              << n_iter * n_eval << " total evals" << std::endl;
    std::cout << std::endl;

    auto benchmark = [&](const char* name, auto fn) -> double {
        double total_time = 0;
        VegasResult last_result;
        for (int i = 0; i < n_runs; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            last_result = fn();
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_time = total_time / n_runs;
        std::cout << name << ": " << avg_time << " ms" << std::endl;
        return avg_time;
    };

    double vegas_scalar = benchmark("Vegas Scalar      ", [&]() {
        mchep::Vegas vegas(n_iter, n_eval, 50, 1.5, boundaries);
        vegas.set_seed(42);
        return vegas.integrate(hep_integrand_scalar, -1.0);
    });

    double vegas_avx = benchmark("Vegas SIMD+AVX    ", [&]() {
        mchep::Vegas vegas(n_iter, n_eval, 50, 1.5, boundaries);
        vegas.set_seed(42);
        return vegas.integrate_simd(hep_integrand_avx, -1.0);
    });

    double vegasplus_scalar = benchmark("VegasPlus Scalar  ", [&]() {
        mchep::VegasPlus vegasplus(n_iter, n_eval, 50, 1.5, 2, 0.75, boundaries);
        vegasplus.set_seed(42);
        return vegasplus.integrate(hep_integrand_scalar, -1.0);
    });

    double vegasplus_avx = benchmark("VegasPlus SIMD+AVX", [&]() {
        mchep::VegasPlus vegasplus(n_iter, n_eval, 50, 1.5, 2, 0.75, boundaries);
        vegasplus.set_seed(42);
        return vegasplus.integrate_simd(hep_integrand_avx, -1.0);
    });

    std::cout << std::endl;
    std::cout << "=== Speedup ===" << std::endl;
    std::cout << "Vegas SIMD+AVX:     " << vegas_scalar / vegas_avx << "x" << std::endl;
    std::cout << "VegasPlus SIMD+AVX: " << vegasplus_scalar / vegasplus_avx << "x" << std::endl;

    return 0;
}
