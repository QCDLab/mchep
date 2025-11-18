#include <mchep.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <utility>
#include <array>
#include <iomanip>

// Integrand function for a 2D Gaussian: exp(-x^2 - y^2)
// This version is for SIMD and expects SoA data.
std::array<double, 4> gaussian_cpp_simd(const std::vector<double>& x) {
    // x is SoA: [x0, x1, x2, x3, y0, y1, y2, y3]
    std::array<double, 4> results;
    for (int i = 0; i < 4; ++i) {
        double x_i = x[i];
        double y_i = x[i + 4];
        results[i] = std::exp(-(x_i * x_i + y_i * y_i));
    }
    return results;
}

int main() {
    std::cout << "Testing 2D Gaussian integral (SIMD)..." << "\n";

    const size_t n_iter = 10;
    const size_t n_eval = 50000;
    const size_t n_bins = 50;
    const double alpha = 0.5;

    std::vector<std::pair<double, double>> boundaries = {
        {-1.0, 1.0},
        {-1.0, 1.0}
    };

    try {
        mchep::Vegas vegas(n_iter, n_eval, n_bins, alpha, boundaries);
        vegas.set_seed(1234); // for reproducibility

        VegasResult result = vegas.integrate_simd(gaussian_cpp_simd, -1.0);

        const double expected = 2.230985;
        const double multiplier = 2.5;

        double diff = std::fabs(result.value - expected);
        assert(diff <= multiplier * result.error);

        std::cout << "Test passed!\n";
        std::cout << "Result: " << std::fixed << std::setprecision(6) << result.value << " +/- " << result.error << std::endl;

        std::cout << "\nTesting C++ API SIMD accuracy goal...\n";
        mchep::Vegas vegas2(20, 100000, n_bins, alpha, boundaries);
        vegas2.set_seed(1234);
        VegasResult result2 = vegas2.integrate_simd(gaussian_cpp_simd, 0.1);

        double accuracy = (result2.error / std::abs(result2.value)) * 100.0;
        std::cout << "Accuracy goal test: value=" << result2.value << ", error=" << result2.error << ", acc=" << accuracy << std::endl;
        assert(accuracy < 0.1);
        std::cout << "Accuracy goal test passed.\n";

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
