"""
Unit tests for the HALO objective. Run:  python -m pytest tests -q

A custom LightGBM objective that returns a wrong gradient does not crash -
it silently trains a bad model. These tests catch that.
"""

import numpy as np

from heatwave_peak.losses import HALOObjective, numerical_gradient_check


def test_gradient_matches_finite_difference():
    for params in [dict(), dict(a0=0.05, a_heat=1.5, a_scarcity=1.0), dict(u_max=1.5)]:
        assert numerical_gradient_check(HALOObjective(**params)) < 1e-4


def test_reduces_to_half_mse_for_tiny_asymmetry():
    o = HALOObjective(a0=1e-4, a_heat=0, a_scarcity=0)
    e = np.linspace(-3, 3, 13)
    a = np.full_like(e, 1e-4)
    np.testing.assert_allclose(o.loss(e, np.zeros_like(e), a, np.ones_like(e)), e ** 2 / 2, rtol=1e-3)


def test_under_forecast_costs_more_than_over_forecast():
    o = HALOObjective()
    a = np.array([0.6, 0.6])
    lo = o.loss(np.array([4.0, -4.0]), np.zeros(2), a, np.ones(2))
    assert lo[0] > 4 * lo[1]          # 4 % short is >4x worse than 4 % surplus


def test_asymmetry_increases_with_heat_severity():
    o = HALOObjective(a0=0.1, a_heat=1.0, a_scarcity=0.0)
    a = o.asymmetry(np.array([0.0, 0.5, 1.0]), np.zeros(3))
    assert np.all(np.diff(a) > 0)


def test_hessian_positive_and_gradient_bounded():
    o = HALOObjective(a0=0.2, a_heat=1.5, a_scarcity=1.0, u_max=4.0)
    e = np.linspace(-50, 50, 1001)
    a = np.full_like(e, 1.0)
    g, h = o.grad_hess(e, np.zeros_like(e), a, np.ones_like(e))
    assert np.all(h > 0)
    assert np.all(np.isfinite(g)) and np.abs(g).max() < 1e4   # Huber cap works
