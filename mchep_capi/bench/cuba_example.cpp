#include <iostream>
#include <cmath>
#include <cuba.h>

// Integrand function: f(x) = exp(-100 * sum((x[i] - 0.5)^2)) * 1013.2118364296088
// This integrates over [0,1]^4 (4-dimensional hypercube)
static int integrand(const int *ndim, const cubareal x[],
                     const int *ncomp, cubareal f[], void *userdata) {
    double dx2 = 0.0;
    for (int d = 0; d < 4; d++) {
        double diff = x[d] - 0.5;
        dx2 += diff * diff;
    }

    f[0] = std::exp(-100.0 * dx2) * 1013.2118364296088;

    return 0;
}

int main() {
    int neval, fail;
    cubareal integral[1], error[1], prob[1];

    std::cout << "CUBA Library Integration Example (C++)" << std::endl;
    std::cout << "=======================================" << std::endl << std::endl;
    std::cout << "Integrating: f(x) = exp(-100 * sum((x[i]-0.5)^2)) * 1013.2118364296088" << std::endl;
    std::cout << "Over: [0,1]^4 (4-dimensional unit hypercube)" << std::endl << std::endl;

    std::cout << "--- Method 1: Vegas ---" << std::endl;
    Vegas(4,
          1,
          integrand,
          nullptr,
          1,
          0.0,
          0.0,
          0,
          0,
          1000,
          1000000,
          10000,
          1000,
          1000,
          0,
          nullptr,
          nullptr,
          &neval,
          &fail,
          integral,
          error,
          prob);

    std::cout << "Result:      " << integral[0] << std::endl;
    std::cout << "Error:       " << error[0] << std::endl;
    std::cout << "Chi-sq prob: " << prob[0] << std::endl;
    std::cout << "Evaluations: " << neval << std::endl;
    std::cout << "Status:      " << (fail == 0 ? "Success" : "Failed") << std::endl << std::endl;

    return 0;
}
