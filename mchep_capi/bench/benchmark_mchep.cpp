#include <mchep.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <utility>

// Integrand function from cuba_example.cpp
// f(x) = exp(-100 * sum((x[i] - 0.5)^2)) * 1013.2118364296088
double integrand_4d(const std::vector<double>& x) {
    double dx2 = 0.0;
    for (int d = 0; d < 4; d++) {
        double diff = x[d] - 0.5;
        dx2 += diff * diff;
    }
    return std::exp(-100.0 * dx2) * 1013.2118364296088;
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
        VegasResult result = vegas.integrate(integrand_4d, -1.0);

        std::cout << "MCHEP Result: " << result.value << " +/- " << result.error << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
