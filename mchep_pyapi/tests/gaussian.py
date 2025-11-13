import mchep
import math


# Define an integrand function in Python
class GaussianIntegrand:
    def __init__(self):
        self.dim = 2

    def eval(self, x):
        # Integral of exp(-x^2 - y^2) in [-1, 1]^2
        return math.exp(-(x[0] ** 2) - x[1] ** 2)


ANALYTICAL_RESULT = 2.230985


def test_vegas_integration():
    integrand = mchep.vegas.Integrand(GaussianIntegrand().eval, 2)
    boundaries = [(-1.0, 1.0), (-1.0, 1.0)]

    vegas = mchep.vegas.Vegas(
        n_iter=10, n_eval=10000, n_bins=50, alpha=0.5, boundaries=boundaries
    )
    result = vegas.integrate(integrand)

    print(
        f"Vegas Result: Value={result.value:.6f}, Error={result.error:.6f}, Chi2/dof={result.chi2_dof:.3f}"
    )
    assert abs(result.value - ANALYTICAL_RESULT) < 5.0 * result.error
    assert result.chi2_dof < 5.0


def test_vegas_plus_integration():
    integrand = mchep.vegas.Integrand(GaussianIntegrand().eval, 2)
    boundaries = [(-1.0, 1.0), (-1.0, 1.0)]

    vegas_plus = mchep.vegas.VegasPlus(
        n_iter=10,
        n_eval=20000,
        n_bins=50,
        alpha=0.5,
        n_strat=4,
        beta=0.75,
        boundaries=boundaries,
    )
    result = vegas_plus.integrate(integrand)

    print(
        f"VegasPlus Result: Value={result.value:.6f}, Error={result.error:.6f}, Chi2/dof={result.chi2_dof:.3f}"
    )
    assert abs(result.value - ANALYTICAL_RESULT) < 5.0 * result.error
    assert result.chi2_dof < 5.0


# To run MPI tests, you would need to build with the 'mpi' feature
# and execute with mpirun, e.g.:
# mpirun -n 2 pytest --pyargs mchep_py.tests.test_mpi_integration
#
# @pytest.mark.skipif(not hasattr(mchep_py.PyVegasPlus, 'integrate_mpi'), reason="MPI feature not enabled")
# def test_vegas_plus_mpi_integration():
#     integrand = mchep_py.PyIntegrand(GaussianIntegrand().eval, 2)
#     boundaries = [(-1.0, 1.0), (-1.0, 1.0)]
#
#     vegas_plus = mchep_py.PyVegasPlus(
#         n_iter=10,
#         n_eval=20000,
#         n_bins=50,
#         alpha=0.5,
#         n_strat=4,
#         beta=0.75,
#         boundaries=boundaries
#     )
#     result = vegas_plus.integrate_mpi(integrand)
#
#     print(f"VegasPlus MPI Result: Value={result.value:.6f}, Error={result.error:.6f}, Chi2/dof={result.chi2_dof:.3f}")
#     assert abs(result.value - ANALYTICAL_RESULT) < 5.0 * result.error
#     assert result.chi2_dof < 5.0
