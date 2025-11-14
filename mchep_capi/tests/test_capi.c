#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "../mchep.h"

// Integrand function for a 2D Gaussian: exp(-x^2 - y^2)
double gaussian(const double* x, int dim, void* user_data) {
    (void)dim; // Unused
    (void)user_data; // Unused
    return exp(-(x[0] * x[0] + x[1] * x[1]));
}

int main() {
    printf("Testing 2D Gaussian integral...\n");

    // Integration parameters
    const uintptr_t n_iter = 10;
    const uintptr_t n_eval = 50000;
    const uintptr_t n_bins = 50;
    const double alpha = 0.5;
    const uintptr_t dim = 2;

    // Integration boundaries [-1, 1] for both dimensions
    struct CBoundary boundaries[2];
    boundaries[0].min = -1.0;
    boundaries[0].max = 1.0;
    boundaries[1].min = -1.0;
    boundaries[1].max = 1.0;

    // Create a new Vegas integrator
    VegasC* vegas = mchep_vegas_new(n_iter, n_eval, n_bins, alpha, dim, boundaries);
    if (!vegas) {
        printf("Failed to create Vegas integrator.\n");
        return 1;
    }

    // Run the integration
    struct VegasResult result = mchep_vegas_integrate(vegas, gaussian, NULL);

    // Free the integrator
    mchep_vegas_free(vegas);

    // Check the result
    const double expected = 2.230985;
    const double multiplier = 2.5;

    printf("Expected result: %f\n", expected);
    printf("Integral value: %f\n", result.value);
    printf("Error: %f\n", result.error);
    printf("Chi2/dof: %f\n", result.chi2_dof);

    double diff = fabs(result.value - expected);
    printf("Difference: %f\n", diff);
    printf("Tolerance: %f\n", multiplier * result.error);

    assert(diff <= multiplier * result.error);

    printf("Test passed!\n");

    return 0;
}
