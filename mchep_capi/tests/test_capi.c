#include <mchep_capi.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

// Integrand function for a 2D Gaussian: exp(-x^2 - y^2)
double gaussian(const double* x, int dim, void* user_data) {
    (void) dim;
    (void) user_data;
    return exp(-(x[0] * x[0] + x[1] * x[1]));
}

int main() {
    printf("Testing 2D Gaussian integral...\n");

    const uintptr_t n_iter = 10;
    const uintptr_t n_eval = 50000;
    const uintptr_t n_bins = 50;
    const double alpha = 0.5;
    const uintptr_t dim = 2;

    struct CBoundary boundaries[2];
    boundaries[0].min = -1.0;
    boundaries[0].max = 1.0;
    boundaries[1].min = -1.0;
    boundaries[1].max = 1.0;

    VegasC* vegas = mchep_vegas_new(n_iter, n_eval, n_bins, alpha, dim, boundaries);
    if (!vegas) {
        printf("Failed to create Vegas integrator.\n");
        return 1;
    }

    struct VegasResult result = mchep_vegas_integrate(vegas, gaussian, NULL, -1.0);
    mchep_vegas_free(vegas);

    const double expected = 2.230985;
    const double multiplier = 2.5;
    double diff = fabs(result.value - expected);
    assert(diff <= multiplier * result.error);

    printf("Test passed!\n");

    printf("\nTesting C API seeding...\n");

    // Test 1: Same seed should produce same result
    VegasC *vegas1 =
        mchep_vegas_new(n_iter, n_eval, n_bins, alpha, dim, boundaries);
    mchep_vegas_set_seed(vegas1, 1234);
    struct VegasResult result1 = mchep_vegas_integrate(vegas1, gaussian, NULL, -1.0);
    mchep_vegas_free(vegas1);

    VegasC *vegas2 =
        mchep_vegas_new(n_iter, n_eval, n_bins, alpha, dim, boundaries);
    mchep_vegas_set_seed(vegas2, 1234);
    struct VegasResult result2 = mchep_vegas_integrate(vegas2, gaussian, NULL, -1.0);
    mchep_vegas_free(vegas2);

    printf("Result 1: %f +/- %f\n", result1.value, result1.error);
    printf("Result 2: %f +/- %f\n", result2.value, result2.error);
    assert(fabs(result1.value - result2.value) < 1e-9);
    assert(fabs(result1.error - result2.error) < 1e-9);
    printf("Same seed test passed.\n");

    // Test 2: Different seed should produce different result
    VegasC *vegas3 =
        mchep_vegas_new(n_iter, n_eval, n_bins, alpha, dim, boundaries);
    mchep_vegas_set_seed(vegas3, 5678);
    struct VegasResult result3 = mchep_vegas_integrate(vegas3, gaussian, NULL, -1.0);
    mchep_vegas_free(vegas3);

    printf("Result 3: %f +/- %f\n", result3.value, result3.error);
    assert(result1.value != result3.value);
    printf("Different seed test passed.\n");

    printf("\nTesting C API accuracy goal...\n");
    VegasC *vegas4 =
        mchep_vegas_new(20, 100000, n_bins, alpha, dim, boundaries);
    mchep_vegas_set_seed(vegas4, 1234);
    struct VegasResult result4 = mchep_vegas_integrate(vegas4, gaussian, NULL, 0.1);
    mchep_vegas_free(vegas4);

    double accuracy = (result4.error / fabs(result4.value)) * 100.0;
    printf("Accuracy goal test: value=%f, error=%f, acc=%f\n", result4.value, result4.error, accuracy);
    assert(accuracy < 0.1);
    printf("Accuracy goal test passed.\n");

    return 0;
}
