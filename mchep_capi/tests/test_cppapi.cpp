#include <mchep.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <utility>

// Integrand function for a 2D Gaussian: exp(-x^2 - y^2)
double gaussian_cpp(const std::vector<double>& x) {
    return std::exp(-(x[0] * x[0] + x[1] * x[1]));
}

int main() {
    std::cout << "Testing 2D Gaussian integral..." << "\n";

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

        VegasResult result = vegas.integrate(gaussian_cpp);

        const double expected = 2.230985;
        const double multiplier = 2.5;

        double diff = std::fabs(result.value - expected);
        assert(diff <= multiplier * result.error);

        std::cout << "Test passed!\n";

        std::cout << "\nTesting C++ API seeding...\n";

        // Test 1: Same seed should produce same result
        mchep::Vegas vegas1(n_iter, n_eval, n_bins, alpha, boundaries);
        vegas1.set_seed(1234);
        VegasResult result1 = vegas1.integrate(gaussian_cpp);

        mchep::Vegas vegas2(n_iter, n_eval, n_bins, alpha, boundaries);
        vegas2.set_seed(1234);
        VegasResult result2 = vegas2.integrate(gaussian_cpp);

        std::cout << "Result 1: " << result1.value << " +/- " << result1.error << std::endl;
        std::cout << "Result 2: " << result2.value << " +/- " << result2.error << std::endl;
        assert(std::abs(result1.value - result2.value) < 1e-9);
        assert(std::abs(result1.error - result2.error) < 1e-9);
        std::cout << "Same seed test passed.\n";

        // Test 2: Different seed should produce different result
        mchep::Vegas vegas3(n_iter, n_eval, n_bins, alpha, boundaries);
        vegas3.set_seed(5678);
        VegasResult result3 = vegas3.integrate(gaussian_cpp);

        std::cout << "Result 3: " << result3.value << " +/- " << result3.error << std::endl;
        assert(result1.value != result3.value);
        std::cout << "Different seed test passed.\n";

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
