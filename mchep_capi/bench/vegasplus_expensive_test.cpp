#include <mchep.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <chrono>

// Expensive SIMD integrand: 100 trig operations per point
std::array<double, 4> expensive_simd(const std::vector<double>& x) {
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

        // Add artificial work: 100 sin/cos evaluations
        double extra = 0.0;
        double x0 = x[0 * simd_width + i];
        double x1 = x[1 * simd_width + i];
        for (int j = 0; j < 100; j++) {
            extra += std::sin(x0 * j) * std::cos(x1 * j);
        }

        results[i] = std::exp(-100.0 * dx2) * 1013.2118364296088 + extra * 1e-10;
    }
    return results;
}

// Scalar version for comparison
double expensive_scalar(const std::vector<double>& x) {
    double dx2 = 0.0;
    for (int d = 0; d < 4; ++d) {
        double diff = x[d] - 0.5;
        dx2 += diff * diff;
    }

    double extra = 0.0;
    for (int j = 0; j < 100; j++) {
        extra += std::sin(x[0] * j) * std::cos(x[1] * j);
    }

    return std::exp(-100.0 * dx2) * 1013.2118364296088 + extra * 1e-10;
}

int main() {
    std::vector<std::pair<double, double>> boundaries = {
        {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}
    };

    const int n_runs = 3;

    std::cout << "=== Expensive Integrand Comparison ===" << std::endl;
    std::cout << "(~100 trig ops per evaluation, 1M total evals)" << std::endl << std::endl;

    // Vegas Scalar
    {
        double total_time = 0;
        VegasResult last_result;
        for (int r = 0; r < n_runs; r++) {
            mchep::Vegas vegas(10, 100000, 50, 1.5, boundaries);
            vegas.set_seed(1234 + r);
            auto start = std::chrono::high_resolution_clock::now();
            last_result = vegas.integrate(expensive_scalar, -1.0);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        std::cout << "Vegas Scalar:     " << (total_time/n_runs) << " ms, "
                  << last_result.value << " +/- " << last_result.error << std::endl;
    }

    // Vegas SIMD
    {
        double total_time = 0;
        VegasResult last_result;
        for (int r = 0; r < n_runs; r++) {
            mchep::Vegas vegas(10, 100000, 50, 1.5, boundaries);
            vegas.set_seed(1234 + r);
            auto start = std::chrono::high_resolution_clock::now();
            last_result = vegas.integrate_simd(expensive_simd, -1.0);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        std::cout << "Vegas SIMD:       " << (total_time/n_runs) << " ms, "
                  << last_result.value << " +/- " << last_result.error << std::endl;
    }

    // VegasPlus Scalar
    {
        double total_time = 0;
        VegasResult last_result;
        for (int r = 0; r < n_runs; r++) {
            mchep::VegasPlus vegasplus(10, 100000, 50, 1.5, 2, 0.75, boundaries);
            vegasplus.set_seed(1234 + r);
            auto start = std::chrono::high_resolution_clock::now();
            last_result = vegasplus.integrate(expensive_scalar, -1.0);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        std::cout << "VegasPlus Scalar: " << (total_time/n_runs) << " ms, "
                  << last_result.value << " +/- " << last_result.error << std::endl;
    }

    // VegasPlus SIMD
    {
        double total_time = 0;
        VegasResult last_result;
        for (int r = 0; r < n_runs; r++) {
            mchep::VegasPlus vegasplus(10, 100000, 50, 1.5, 2, 0.75, boundaries);
            vegasplus.set_seed(1234 + r);
            auto start = std::chrono::high_resolution_clock::now();
            last_result = vegasplus.integrate_simd(expensive_simd, -1.0);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        std::cout << "VegasPlus SIMD:   " << (total_time/n_runs) << " ms, "
                  << last_result.value << " +/- " << last_result.error << std::endl;
    }

    return 0;
}
