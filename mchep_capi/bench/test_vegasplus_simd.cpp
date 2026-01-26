#include <mchep.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <chrono>

// SIMD integrand: narrow Gaussian
std::array<double, 4> gaussian_simd(const std::vector<double>& x) {
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
        results[i] = std::exp(-100.0 * dx2) * 1013.2118364296088;
    }
    return results;
}

int main() {
    std::vector<std::pair<double, double>> boundaries = {
        {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}
    };

    // Test Vegas SIMD
    {
        mchep::Vegas vegas(10, 100000, 50, 1.5, boundaries);
        vegas.set_seed(1234);

        auto start = std::chrono::high_resolution_clock::now();
        VegasResult result = vegas.integrate_simd(gaussian_simd, -1.0);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "Vegas SIMD:" << std::endl;
        std::cout << "  Result: " << result.value << " +/- " << result.error << std::endl;
        std::cout << "  Time: " << elapsed << " ms" << std::endl;
    }

    // Test VegasPlus SIMD
    {
        // VegasPlus parameters: n_iter, n_eval, n_bins, alpha, n_strat, beta, boundaries
        mchep::VegasPlus vegasplus(10, 100000, 50, 1.5, 2, 0.75, boundaries);
        vegasplus.set_seed(1234);

        auto start = std::chrono::high_resolution_clock::now();
        VegasResult result = vegasplus.integrate_simd(gaussian_simd, -1.0);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "VegasPlus SIMD:" << std::endl;
        std::cout << "  Result: " << result.value << " +/- " << result.error << std::endl;
        std::cout << "  Time: " << elapsed << " ms" << std::endl;
    }

    return 0;
}
