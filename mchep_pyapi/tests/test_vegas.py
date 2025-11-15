"""Simple test examples for VEGAS Python bindings."""

import math
import numpy as np

from mchep.vegas import Vegas, VegasPlus, Integrand


MULTIPLIER = 2.5


def test_basic_integration():
    """Simple 1D integral ∫x dx = 0.5 over [0,1]"""
    expected = 0.5

    def f(x):
        return x[0]

    vegas = Vegas(
        n_iter=5,
        n_eval=10_000,
        n_bins=30,
        alpha=0.5,
        boundaries=[(0.0, 1.0)],
    )

    result = vegas.integrate(f)
    assert abs(result.value - expected) <= MULTIPLIER * result.error


def test_2d_gaussian():
    """Test 2D Gaussian ∫∫exp(-x²-y²) dx over [-1,1]²"""
    expected = 2.230985

    def gaussian(x):
        return math.exp(-(x[0] ** 2 + x[1] ** 2))

    vegas = Vegas(
        n_iter=10,
        n_eval=50_000,
        n_bins=50,
        alpha=0.5,
        boundaries=[(-1.0, 1.0), (-1.0, 1.0)],
    )

    result = vegas.integrate(gaussian)
    assert abs(result.value - expected) <= MULTIPLIER * result.error


def test_lambda():
    """Test using lambda function for ∫x² dx over [0,1]"""
    expected = 1 / 3

    vegas = Vegas(
        n_iter=5,
        n_eval=10_000,
        n_bins=30,
        alpha=0.5,
        boundaries=[(0.0, 1.0)],
    )

    result = vegas.integrate(lambda x: x[0] ** 2)
    assert abs(result.value - expected) <= MULTIPLIER * result.error


def test_integrand_class():
    """Test using Integrand wrapper for ∫x dx = 0.5 over [0,1]"""
    expected = 0.5

    def f(x):
        return x[0]

    integrand = Integrand(f, dim=1)

    vegas = Vegas(
        n_iter=5,
        n_eval=10_000,
        n_bins=30,
        alpha=0.5,
        boundaries=[(0.0, 1.0)],
    )

    result = vegas.integrate_integrand(integrand)
    assert abs(result.value - expected) <= MULTIPLIER * result.error


def test_class_based_integrand():
    """Test using a class with __call__ for ∫x³ dx = 1/4 over [0,1]"""
    expected = 1 / 4

    class Polynomial:
        def __init__(self, power):
            self.power = power

        def __call__(self, x):
            return x[0] ** self.power

    integrand = Polynomial(3)

    vegas = Vegas(
        n_iter=5,
        n_eval=10_000,
        n_bins=30,
        alpha=0.5,
        boundaries=[(0.0, 1.0)],
    )

    result = vegas.integrate(integrand)
    assert abs(result.value - expected) <= MULTIPLIER * result.error


def test_3d_integral():
    """Test 3D integral ∫∫∫xyz dxdydz over [0,1]³"""
    expected = 1 / 8

    def f(x):
        return x[0] * x[1] * x[2]

    vegas = Vegas(
        n_iter=8,
        n_eval=100_000,
        n_bins=30,
        alpha=0.5,
        boundaries=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    )

    result = vegas.integrate(f)
    assert abs(result.value - expected) <= MULTIPLIER * result.error


def test_vegasplus():
    """Test VegasPlus integrator ∫x dx = 0.5 over [0,1]"""
    expected = 1 / 2

    def f(x):
        return x[0]

    vegas_plus = VegasPlus(
        n_iter=5,
        n_eval=10_000,
        n_bins=30,
        alpha=0.5,
        n_strat=2,
        beta=0.5,
        boundaries=[(0.0, 1.0)],
    )

    result = vegas_plus.integrate(f)
    assert abs(result.value - expected) <= MULTIPLIER * result.error


def test_seeding():
    """Test that setting a seed produces deterministic results."""

    def f(x):
        return x[0]

    vegas1 = Vegas(
        n_iter=5,
        n_eval=10_000,
        n_bins=30,
        alpha=0.5,
        boundaries=[(0.0, 1.0)],
    )
    vegas1.set_seed(1234)
    result1 = vegas1.integrate(f)

    vegas2 = Vegas(
        n_iter=5,
        n_eval=10_000,
        n_bins=30,
        alpha=0.5,
        boundaries=[(0.0, 1.0)],
    )
    vegas2.set_seed(1234)
    result2 = vegas2.integrate(f)

    np.testing.assert_almost_equal(result1.value, result2.value)
    np.testing.assert_almost_equal(result1.error, result2.error)

    vegas_plus1 = VegasPlus(
        n_iter=5,
        n_eval=10_000,
        n_bins=30,
        alpha=0.5,
        n_strat=2,
        beta=0.5,
        boundaries=[(0.0, 1.0)],
    )
    vegas_plus1.set_seed(1234)
    result_plus1 = vegas_plus1.integrate(f)

    vegas_plus2 = VegasPlus(
        n_iter=5,
        n_eval=10_000,
        n_bins=30,
        alpha=0.5,
        n_strat=2,
        beta=0.5,
        boundaries=[(0.0, 1.0)],
    )
    vegas_plus2.set_seed(1234)
    result_plus2 = vegas_plus2.integrate(f)

    np.testing.assert_almost_equal(result_plus1.value, result_plus2.value)
    np.testing.assert_almost_equal(result_plus1.error, result_plus2.error)
