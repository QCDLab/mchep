#include <mchep.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <utility>
#include <array>

// Integrand function from cuba_example.cpp, adapted for SIMD
// f(x) = exp(-100 * sum((x[i] - 0.5)^2)) * 1013.2118364296088
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

int main() {
    const size_t n_iter = 10;
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
        VegasResult result = vegas.integrate_simd(integrand_4d_simd);

        std::cout << "MCHEP SIMD Result: " << result.value << " +/- " << result.error << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
