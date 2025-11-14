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

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
