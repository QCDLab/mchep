#include <mchep.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// Expensive integrand: adds artificial computation (same as CUBA test)
double expensive_integrand(const std::vector<double>& x) {
    double dx2 = 0.0;
    for (int d = 0; d < 4; d++) {
        double diff = x[d] - 0.5;
        dx2 += diff * diff;
    }

    // Add artificial work: 100 sin/cos evaluations
    double extra = 0.0;
    for (int i = 0; i < 100; i++) {
        extra += std::sin(x[0] * i) * std::cos(x[1] * i);
    }

    return std::exp(-100.0 * dx2) * 1013.2118364296088 + extra * 1e-10;
}

int main() {
    const size_t n_iter = 10;
    const size_t n_eval = 100000;  // ~1M total evals like CUBA
    const size_t n_bins = 50;
    const double alpha = 1.5;

    std::vector<std::pair<double, double>> boundaries = {
        {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}
    };

    mchep::Vegas vegas(n_iter, n_eval, n_bins, alpha, boundaries);
    vegas.set_seed(1234);

    auto start = std::chrono::high_resolution_clock::now();
    VegasResult result = vegas.integrate(expensive_integrand, -1.0);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Result: " << result.value << " +/- " << result.error << std::endl;
    std::cout << "Time: " << elapsed << " ms" << std::endl;

    return 0;
}
