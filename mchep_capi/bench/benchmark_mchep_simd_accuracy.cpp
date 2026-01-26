#include <mchep.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <utility>
#include <chrono>
#include <iomanip>

// SIMD Integrand function: evaluates 4 points at once
// f(x) = exp(-100 * sum((x[i] - 0.5)^2)) * 1013.2118364296088
// Layout: x[d * simd_width + i] where d=dimension, i=point index
std::array<double, 4> integrand_4d_simd(const std::vector<double>& x) {
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

int main(int argc, char* argv[]) {
    // Default target accuracy: 0.5%
    double target_accuracy = 0.5;
    if (argc > 1) {
        target_accuracy = std::atof(argv[1]);
    }

    const size_t n_iter = 100;  // Upper bound (will exit early when accuracy reached)
    const size_t n_eval = 100000;
    const size_t n_bins = 50;
    const double alpha = 1.5;

    std::vector<std::pair<double, double>> boundaries = {
        {0.0, 1.0},
        {0.0, 1.0},
        {0.0, 1.0},
        {0.0, 1.0}
    };

    try {
        mchep::Vegas vegas(n_iter, n_eval, n_bins, alpha, boundaries);
        vegas.set_seed(1234);

        auto start = std::chrono::high_resolution_clock::now();
        VegasResult result = vegas.integrate_simd(integrand_4d_simd, target_accuracy);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double actual_accuracy = (result.error / std::abs(result.value)) * 100.0;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "MCHEP_SIMD_ACCURACY," << target_accuracy << ","
                  << result.value << "," << result.error << ","
                  << actual_accuracy << "," << elapsed_ms << ","
                  << result.chi2_dof << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
