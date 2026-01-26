#include <mchep.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <utility>
#include <chrono>
#include <iomanip>
#include <string>

// Integrand function: narrow Gaussian
double integrand_4d(const std::vector<double>& x) {
    double dx2 = 0.0;
    for (size_t d = 0; d < x.size(); d++) {
        double diff = x[d] - 0.5;
        dx2 += diff * diff;
    }
    return std::exp(-100.0 * dx2) * 1013.2118364296088;
}

void run_benchmark(size_t n_iter, size_t n_eval, size_t n_bins, size_t dim, int n_runs) {
    std::vector<std::pair<double, double>> boundaries(dim, {0.0, 1.0});

    double total_time = 0.0;
    double last_value = 0.0, last_error = 0.0;

    for (int run = 0; run < n_runs; run++) {
        mchep::Vegas vegas(n_iter, n_eval, n_bins, 1.5, boundaries);
        vegas.set_seed(1234 + run);

        auto start = std::chrono::high_resolution_clock::now();
        VegasResult result = vegas.integrate(integrand_4d, -1.0);
        auto end = std::chrono::high_resolution_clock::now();

        total_time += std::chrono::duration<double, std::milli>(end - start).count();
        last_value = result.value;
        last_error = result.error;
    }

    double avg_time = total_time / n_runs;
    size_t total_evals = n_iter * n_eval;
    double evals_per_ms = total_evals / avg_time;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << dim << "," << n_iter << "," << n_eval << "," << total_evals << ","
              << avg_time << "," << evals_per_ms << ","
              << std::setprecision(6) << last_value << "," << last_error << std::endl;
}

int main(int argc, char* argv[]) {
    int n_runs = 3;
    std::string test_type = "all";

    if (argc > 1) test_type = argv[1];
    if (argc > 2) n_runs = std::atoi(argv[2]);

    std::cout << "dim,n_iter,n_eval,total_evals,time_ms,evals_per_ms,value,error" << std::endl;

    if (test_type == "all" || test_type == "evals") {
        // Scaling with evaluations per iteration (fixed 10 iterations, 4D)
        for (size_t n_eval : {1000, 5000, 10000, 50000, 100000, 500000, 1000000}) {
            run_benchmark(10, n_eval, 50, 4, n_runs);
        }
    }

    if (test_type == "all" || test_type == "iters") {
        // Scaling with iterations (fixed 100k evals per iter, 4D)
        for (size_t n_iter : {1, 2, 5, 10, 20, 50, 100}) {
            run_benchmark(n_iter, 100000, 50, 4, n_runs);
        }
    }

    if (test_type == "all" || test_type == "dims") {
        // Scaling with dimensions (fixed 10 iter, 100k evals)
        for (size_t dim : {2, 3, 4, 5, 6, 8, 10}) {
            run_benchmark(10, 100000, 50, dim, n_runs);
        }
    }

    return 0;
}
