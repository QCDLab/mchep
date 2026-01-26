#include <iostream>
#include <cmath>
#include <chrono>
#include <cuba.h>

// Expensive integrand: adds artificial computation
static int expensive_integrand(const int *ndim, const cubareal x[],
                               const int *ncomp, cubareal f[], void *userdata) {
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

    f[0] = std::exp(-100.0 * dx2) * 1013.2118364296088 + extra * 1e-10;
    return 0;
}

int main() {
    int neval, fail;
    cubareal integral[1], error[1], prob[1];

    auto start = std::chrono::high_resolution_clock::now();

    Vegas(4, 1, expensive_integrand, nullptr,
          1, 0.0, 0.0, 0, 0,
          1000, 1000000, 10000, 1000, 1000,
          0, nullptr, nullptr,
          &neval, &fail, integral, error, prob);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Result: " << integral[0] << " +/- " << error[0] << std::endl;
    std::cout << "Evaluations: " << neval << std::endl;
    std::cout << "Time: " << elapsed << " ms" << std::endl;

    return 0;
}
