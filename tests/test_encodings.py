import numpy as np
import pytest

from slud import LIN, LUD, SINH, SLUD

LB = np.array([1e-8, 1e-2])
UB = np.array([1e8, 1e2])


def test_slud_end_points():
    enc = SLUD(LB, UB)
    np.testing.assert_allclose(enc.decode(np.zeros(2)), LB)
    np.testing.assert_allclose(enc.decode(np.ones(2)), UB)
    np.testing.assert_allclose(enc.decode(-np.ones(2)), -UB)


def test_slud_is_odd_and_monotone():
    enc = SLUD(LB, UB)
    u = np.linspace(0.01, 1, 200)[:, None] * np.ones(2)
    np.testing.assert_allclose(enc.decode(-u), -enc.decode(u))
    assert np.all(np.diff(enc.decode(u), axis=0) > 0)


def test_slud_uniform_u_is_log_uniform_on_both_signs():
    enc = SLUD([1e-6], [1e6])
    u = np.random.default_rng(0).uniform(-1, 1, size=(200_000, 1))
    x = enc.decode(u)[:, 0]
    assert abs(np.mean(x < 0) - 0.5) < 0.01
    # 12 decades -> each decade holds 1/12 of the samples on each sign
    counts, _ = np.histogram(np.log10(np.abs(x)), bins=12, range=(-6, 6))
    np.testing.assert_allclose(counts / x.size, 1 / 12, atol=0.005)


@pytest.mark.parametrize("encoding", [SLUD, LUD, LIN])
def test_decoded_values_stay_in_magnitude_bounds(encoding):
    enc = encoding(LB, UB)
    Z = np.random.default_rng(1).uniform(enc.xl, enc.xu, size=(1000, enc.n_var))
    X = enc.decode(Z)
    assert X.shape == (1000, 2)
    assert np.all(np.abs(X) >= LB * (1 - 1e-12))
    assert np.all(np.abs(X) <= UB * (1 + 1e-12))
    assert np.any(X < 0) and np.any(X > 0)


def test_lud_and_lin_use_two_variables_per_value():
    assert SLUD(LB, UB).n_var == 2
    assert LUD(LB, UB).n_var == 4
    assert LIN(LB, UB).n_var == 4
    np.testing.assert_allclose(LUD(LB, UB).decode(np.array([-8.0, 2.0, -0.5, 0.5])), [-1e-8, 1e2])


@pytest.mark.parametrize("lb, ub", [([0.0], [1.0]), ([-1.0], [1.0]), ([2.0], [1.0]), ([1.0, 2.0], [3.0])])
def test_bad_bounds_raise(lb, ub):
    with pytest.raises(ValueError):
        SLUD(lb, ub)


def test_sinh_end_points_zero_and_symmetry():
    enc = SINH(LB, UB)
    np.testing.assert_allclose(enc.decode(np.ones(2)), UB)
    np.testing.assert_allclose(enc.decode(-np.ones(2)), -UB)
    np.testing.assert_allclose(enc.decode(np.zeros(2)), 0.0)
    u = np.linspace(-1, 1, 401)[:, None] * np.ones(2)
    assert np.all(np.diff(enc.decode(u), axis=0) > 0)


def test_sinh_matches_slud_away_from_zero():
    # far from the core, SINH(lb=c) is SLUD with lb = c/2 (sinh z -> e^z / 2)
    u = np.linspace(0.3, 1, 50)[:, None]
    x_sinh = SINH([1e-6], [1e6]).decode(u)
    x_slud = SLUD([0.5e-6], [1e6]).decode(u)
    np.testing.assert_allclose(x_sinh, x_slud, rtol=1e-6)
