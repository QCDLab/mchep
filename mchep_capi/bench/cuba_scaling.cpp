#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <string>
#include <cuba.h>

static int g_dim = 4;

// Integrand function: narrow Gaussian (uses global dim)
static int integrand(const int *ndim, const cubareal x[],
                     const int *ncomp, cubareal f[], void *userdata) {
    double dx2 = 0.0;
    for (int d = 0; d < g_dim; d++) {
        double diff = x[d] - 0.5;
        dx2 += diff * diff;
    }
    f[0] = std::exp(-100.0 * dx2) * 1013.2118364296088;
    return 0;
}

void run_benchmark(int dim, int maxeval, int nstart, int nincrease, int n_runs) {
    g_dim = dim;

    double total_time = 0.0;
    double last_value = 0.0, last_error = 0.0;
    int last_neval = 0;

    for (int run = 0; run < n_runs; run++) {
        int neval, fail;
        cubareal integral[1], error[1], prob[1];

        auto start = std::chrono::high_resolution_clock::now();

        Vegas(dim, 1, integrand, nullptr,
              1, 0.0, 0.0, 0, 0,
              1000, maxeval, nstart, nincrease, 1000,
              0, nullptr, nullptr,
              &neval, &fail, integral, error, prob);

        auto end = std::chrono::high_resolution_clock::now();

        total_time += std::chrono::duration<double, std::milli>(end - start).count();
        last_value = integral[0];
        last_error = error[0];
        last_neval = neval;
    }

    double avg_time = total_time / n_runs;
    double evals_per_ms = last_neval / avg_time;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << dim << "," << maxeval << "," << nstart << "," << last_neval << ","
              << avg_time << "," << evals_per_ms << ","
              << std::setprecision(6) << last_value << "," << last_error << std::endl;
}

int main(int argc, char* argv[]) {
    int n_runs = 3;
    std::string test_type = "all";

    if (argc > 1) test_type = argv[1];
    if (argc > 2) n_runs = std::atoi(argv[2]);

    std::cout << "dim,maxeval,nstart,actual_evals,time_ms,evals_per_ms,value,error" << std::endl;

    if (test_type == "all" || test_type == "evals") {
        // Scaling with max evaluations (4D)
        for (int maxeval : {10000, 50000, 100000, 500000, 1000000, 5000000, 10000000}) {
            run_benchmark(4, maxeval, 10000, 1000, n_runs);
        }
    }

    if (test_type == "all" || test_type == "nstart") {
        // Scaling with nstart (fixed 1M maxeval, 4D)
        for (int nstart : {1000, 5000, 10000, 50000, 100000}) {
            run_benchmark(4, 1000000, nstart, 1000, n_runs);
        }
    }

    if (test_type == "all" || test_type == "dims") {
        // Scaling with dimensions (fixed 1M evals)
        for (int dim : {2, 3, 4, 5, 6, 8, 10}) {
            run_benchmark(dim, 1000000, 10000, 1000, n_runs);
        }
    }

    return 0;
}
