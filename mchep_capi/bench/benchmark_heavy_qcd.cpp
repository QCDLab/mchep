/**
 * Heavy QCD-style Integrand Benchmark
 *
 * Simulates extremely expensive integrands like those found in high-precision
 * QCD predictions (NLO/NNLO calculations, multi-loop diagrams).
 *
 * The computational cost per evaluation is configurable via FLOPS_PER_EVAL.
 * Real QCD matrix elements can take milliseconds to seconds per evaluation.
 *
 * Compile with:
 *   c++ -std=c++14 -O3 -march=native -mavx2 -ffast-math benchmark_heavy_qcd.cpp \
 *       $(pkg-config --cflags --libs mchep_capi) -lm -o benchmark_heavy_qcd
 *
 * Run with different costs:
 *   ./benchmark_heavy_qcd 10000      # 10k FLOPs per eval (light)
 *   ./benchmark_heavy_qcd 100000     # 100k FLOPs per eval (medium)
 *   ./benchmark_heavy_qcd 1000000    # 1M FLOPs per eval (heavy)
 *   ./benchmark_heavy_qcd 10000000   # 10M FLOPs per eval (very heavy, ~ms per eval)
 */

#include <mchep.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <chrono>
#include <immintrin.h>
#include <iomanip>

// Global configurable cost
static int g_flops_per_eval = 100000;

// Simulates heavy matrix element calculation using AVX
// Cost is approximately g_flops_per_eval FLOPs per 4 points
std::array<double, 4> heavy_qcd_avx(const std::vector<double>& x) {
    std::array<double, 4> results;

    __m256d x0 = _mm256_loadu_pd(&x[0]);
    __m256d x1 = _mm256_loadu_pd(&x[4]);
    __m256d x2 = _mm256_loadu_pd(&x[8]);
    __m256d x3 = _mm256_loadu_pd(&x[12]);

    __m256d half = _mm256_set1_pd(0.5);

    __m256d d0 = _mm256_sub_pd(x0, half);
    __m256d d1 = _mm256_sub_pd(x1, half);
    __m256d d2 = _mm256_sub_pd(x2, half);
    __m256d d3 = _mm256_sub_pd(x3, half);

    __m256d sum = _mm256_mul_pd(d0, d0);
    sum = _mm256_fmadd_pd(d1, d1, sum);
    sum = _mm256_fmadd_pd(d2, d2, sum);
    sum = _mm256_fmadd_pd(d3, d3, sum);

    // Heavy computation: simulate matrix element squared
    // Each FMA is ~2 FLOPs, loop does ~4 FLOPs per iteration (for 4 points)
    __m256d acc = _mm256_set1_pd(1.0);
    __m256d term = sum;
    __m256d coef = _mm256_set1_pd(0.9999);

    int n_iterations = g_flops_per_eval / 4;  // ~4 FLOPs per iteration
    for (int i = 0; i < n_iterations; i++) {
        acc = _mm256_fmadd_pd(term, coef, acc);
        term = _mm256_mul_pd(term, coef);
    }

    // Polynomial decay (avoids exp bottleneck)
    __m256d neg_sum = _mm256_mul_pd(sum, _mm256_set1_pd(-10.0));
    __m256d one = _mm256_set1_pd(1.0);
    __m256d result = _mm256_fmadd_pd(neg_sum, _mm256_set1_pd(1.0/6.0), one);
    result = _mm256_fmadd_pd(result, neg_sum, one);
    result = _mm256_fmadd_pd(result, neg_sum, one);
    result = _mm256_mul_pd(result, acc);
    result = _mm256_max_pd(result, _mm256_setzero_pd());

    _mm256_storeu_pd(results.data(), result);
    return results;
}

// Scalar version
double heavy_qcd_scalar(const std::vector<double>& x) {
    double dx2 = 0.0;
    for (int d = 0; d < 4; ++d) {
        double diff = x[d] - 0.5;
        dx2 += diff * diff;
    }

    double acc = 1.0;
    double term = dx2;
    int n_iterations = g_flops_per_eval / 4;
    for (int i = 0; i < n_iterations; i++) {
        acc += term * 0.9999;
        term *= 0.9999;
    }

    double neg_sum = -10.0 * dx2;
    double result = 1.0 + neg_sum * (1.0 + neg_sum * (1.0 + neg_sum / 6.0));
    return std::max(0.0, result * acc);
}

void print_time_estimate(double ms_per_run, int total_evals, int target_evals) {
    double evals_per_ms = total_evals / ms_per_run;
    double seconds_needed = target_evals / evals_per_ms / 1000.0;

    std::cout << "  Throughput: " << std::fixed << std::setprecision(1)
              << evals_per_ms << " evals/ms" << std::endl;

    if (seconds_needed < 60) {
        std::cout << "  Time for " << target_evals << " evals: "
                  << seconds_needed << " seconds" << std::endl;
    } else if (seconds_needed < 3600) {
        std::cout << "  Time for " << target_evals << " evals: "
                  << seconds_needed / 60 << " minutes" << std::endl;
    } else if (seconds_needed < 86400) {
        std::cout << "  Time for " << target_evals << " evals: "
                  << seconds_needed / 3600 << " hours" << std::endl;
    } else {
        std::cout << "  Time for " << target_evals << " evals: "
                  << seconds_needed / 86400 << " days" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        g_flops_per_eval = std::atoi(argv[1]);
    }

    std::vector<std::pair<double, double>> boundaries = {
        {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}
    };

    // Adjust iterations based on cost to keep benchmark reasonable
    int n_iter = 5;
    int n_eval = 10000;
    if (g_flops_per_eval >= 1000000) {
        n_eval = 1000;  // Reduce for very heavy integrands
    }
    if (g_flops_per_eval >= 10000000) {
        n_eval = 100;   // Further reduce for extremely heavy
    }

    int total_evals = n_iter * n_eval;

    std::cout << "=== Heavy QCD-style Integrand Benchmark ===" << std::endl;
    std::cout << "Cost: " << g_flops_per_eval << " FLOPs per evaluation" << std::endl;
    std::cout << "Config: " << n_iter << " iterations x " << n_eval << " evals = "
              << total_evals << " total" << std::endl;
    std::cout << std::endl;

    // Estimate single eval time
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> test_x = {0.5, 0.5, 0.5, 0.5};
        volatile double r = heavy_qcd_scalar(test_x);
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count();
        std::cout << "Single eval time (scalar): ~" << us << " microseconds" << std::endl;
        std::cout << std::endl;
    }

    // Benchmark scalar
    std::cout << "--- Vegas Scalar ---" << std::endl;
    {
        mchep::Vegas vegas(n_iter, n_eval, 50, 1.5, boundaries);
        vegas.set_seed(42);

        auto start = std::chrono::high_resolution_clock::now();
        VegasResult result = vegas.integrate(heavy_qcd_scalar, -1.0);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "  Time: " << ms << " ms" << std::endl;
        std::cout << "  Result: " << result.value << " +/- " << result.error << std::endl;
        print_time_estimate(ms, total_evals, 1000000000);  // 1 billion evals
    }
    std::cout << std::endl;

    // Benchmark SIMD+AVX
    std::cout << "--- Vegas SIMD+AVX ---" << std::endl;
    {
        mchep::Vegas vegas(n_iter, n_eval, 50, 1.5, boundaries);
        vegas.set_seed(42);

        auto start = std::chrono::high_resolution_clock::now();
        VegasResult result = vegas.integrate_simd(heavy_qcd_avx, -1.0);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "  Time: " << ms << " ms" << std::endl;
        std::cout << "  Result: " << result.value << " +/- " << result.error << std::endl;
        print_time_estimate(ms, total_evals, 1000000000);
    }
    std::cout << std::endl;

    // Benchmark VegasPlus SIMD+AVX
    std::cout << "--- VegasPlus SIMD+AVX ---" << std::endl;
    {
        mchep::VegasPlus vegasplus(n_iter, n_eval, 50, 1.5, 2, 0.75, boundaries);
        vegasplus.set_seed(42);

        auto start = std::chrono::high_resolution_clock::now();
        VegasResult result = vegasplus.integrate_simd(heavy_qcd_avx, -1.0);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "  Time: " << ms << " ms" << std::endl;
        std::cout << "  Result: " << result.value << " +/- " << result.error << std::endl;
        print_time_estimate(ms, total_evals, 1000000000);
    }
    std::cout << std::endl;

    // Summary
    std::cout << "=== Typical QCD Costs ===" << std::endl;
    std::cout << "  LO (tree-level):     ~1k-10k FLOPs" << std::endl;
    std::cout << "  NLO (1-loop):        ~100k-1M FLOPs" << std::endl;
    std::cout << "  NNLO (2-loop):       ~10M-100M FLOPs" << std::endl;
    std::cout << "  N3LO (3-loop):       ~1B+ FLOPs" << std::endl;
    std::cout << std::endl;
    std::cout << "Run with: ./benchmark_heavy_qcd <flops_per_eval>" << std::endl;

    return 0;
}
