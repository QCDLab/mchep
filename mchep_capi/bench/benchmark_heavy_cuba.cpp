/**
 * Heavy QCD-style Integrand Benchmark for CUBA
 *
 * Companion to benchmark_heavy_qcd.cpp for comparing MCHEP vs CUBA
 * on expensive integrands.
 *
 * Compile with:
 *   c++ -std=c++14 -O3 -march=native -ffast-math benchmark_heavy_cuba.cpp \
 *       -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib \
 *       -lcuba -lm -o benchmark_heavy_cuba
 */

#include <iostream>
#include <cmath>
#include <chrono>
#include <cuba.h>

// Global configurable cost
static int g_flops_per_eval = 100000;

// Heavy QCD-style integrand for CUBA
static int heavy_qcd_integrand(const int *ndim, const cubareal x[],
                                const int *ncomp, cubareal f[], void *userdata) {
    double dx2 = 0.0;
    for (int d = 0; d < 4; ++d) {
        double diff = x[d] - 0.5;
        dx2 += diff * diff;
    }

    // Heavy computation: same as MCHEP benchmark
    double acc = 1.0;
    double term = dx2;
    int n_iterations = g_flops_per_eval / 4;
    for (int i = 0; i < n_iterations; i++) {
        acc += term * 0.9999;
        term *= 0.9999;
    }

    // Polynomial decay
    double neg_sum = -10.0 * dx2;
    double result = 1.0 + neg_sum * (1.0 + neg_sum * (1.0 + neg_sum / 6.0));
    f[0] = std::max(0.0, result * acc);

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        g_flops_per_eval = std::atoi(argv[1]);
    }

    // Adjust evaluations based on cost
    int maxeval = 50000;
    if (g_flops_per_eval >= 1000000) {
        maxeval = 5000;
    }
    if (g_flops_per_eval >= 10000000) {
        maxeval = 500;
    }

    int neval, fail;
    cubareal integral[1], error[1], prob[1];

    auto start = std::chrono::high_resolution_clock::now();

    Vegas(4, 1, heavy_qcd_integrand, nullptr,
          1, 0.0, 0.0, 0, 0,
          100, maxeval, 1000, 100, 1000,
          0, nullptr, nullptr,
          &neval, &fail, integral, error, prob);

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    double throughput = neval / ms;

    // Output in parseable format
    std::cout << "CUBA_RESULT," << g_flops_per_eval << ","
              << ms << "," << neval << "," << throughput << ","
              << integral[0] << "," << error[0] << std::endl;

    return 0;
}
