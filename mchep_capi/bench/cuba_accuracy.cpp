#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cuba.h>

// Integrand function: f(x) = exp(-100 * sum((x[i] - 0.5)^2)) * 1013.2118364296088
// Exact integral over [0,1]^4 is 1.0
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

int main(int argc, char* argv[]) {
    // Default target accuracy: 0.5% (epsrel = 0.005)
    double target_accuracy_percent = 0.5;
    if (argc > 1) {
        target_accuracy_percent = std::atof(argv[1]);
    }
    double epsrel = target_accuracy_percent / 100.0;  // Convert percentage to fraction

    int neval, fail;
    cubareal integral[1], error[1], prob[1];

    auto start = std::chrono::high_resolution_clock::now();

    Vegas(4,              // ndim
          1,              // ncomp
          integrand,      // integrand function
          nullptr,        // userdata
          1,              // nvec
          epsrel,         // epsrel (relative tolerance)
          0.0,            // epsabs (absolute tolerance)
          0,              // flags
          0,              // seed (0 = use Sobol)
          1000,           // mineval
          10000000,       // maxeval (high upper bound)
          10000,          // nstart
          1000,           // nincrease
          1000,           // nbatch
          0,              // gridno
          nullptr,        // statefile
          nullptr,        // spin
          &neval,         // neval output
          &fail,          // fail output
          integral,       // integral output
          error,          // error output
          prob);          // prob output

    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double actual_accuracy = (error[0] / std::abs(integral[0])) * 100.0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CUBA_ACCURACY," << target_accuracy_percent << ","
              << integral[0] << "," << error[0] << ","
              << actual_accuracy << "," << elapsed_ms << ","
              << neval << "," << fail << std::endl;

    return 0;
}
