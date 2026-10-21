"""Test problems, written on PHYSICAL values.

Every function takes X with shape (..., n) and returns f with shape (...),
so a whole population is evaluated in one call. Global minimum is f = 0 for all.
"""

import numpy as np


def rosenbrock(X):
    """2-D Rosenbrock. Optimum (1, 1)."""
    x, y = X[..., 0], X[..., 1]
    return (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2


def powell_badly_scaled(X):
    """Powell badly scaled, More, Garbow & Hillstrom (1981) #3. Optimum (1.098e-5, 9.106)."""
    x, y = X[..., 0], X[..., 1]
    return (1.0e4 * x * y - 1.0) ** 2 + (np.exp(-x) + np.exp(-y) - 1.0001) ** 2


def brown_badly_scaled(X):
    """Brown badly scaled, More, Garbow & Hillstrom (1981) #4. Optimum (1e6, 2e-6)."""
    x, y = X[..., 0], X[..., 1]
    return (x - 1.0e6) ** 2 + (y - 2.0e-6) ** 2 + (x * y - 2.0) ** 2


# Sign-flipped variants (x2 -> -x2): same landscape, optimum moved to a negative x2.
FLIP_X2 = np.array([1.0, -1.0])


def powell_flipped(X):
    """Powell badly scaled with x2 -> -x2. Optimum (1.098e-5, -9.106)."""
    return powell_badly_scaled(X * FLIP_X2)


def brown_flipped(X):
    """Brown badly scaled with x2 -> -x2. Optimum (1e6, -2e-6)."""
    return brown_badly_scaled(X * FLIP_X2)


# Propane cp/R, NASA-7 low-temperature coefficients a1..a5 from GRI-Mech 3.0.
C3H8_A = np.array([0.93355381, 0.026424579, 6.1059727e-06, -2.1977499e-08, 9.5149253e-12])
T_GRID = np.linspace(200.0, 1000.0, 100)  # K
T_POWERS = T_GRID[:, None] ** np.arange(5)  # columns: 1, T, T^2, T^3, T^4
CP_REF = T_POWERS @ C3H8_A


def cp_fit_c3h8(X):
    """Mean squared relative error of cp/R(T) = a1 + a2 T + ... + a5 T^4 against propane.

    X holds the 5 coefficients. Optimum is C3H8_A (a4 is negative).
    """
    cp = X @ T_POWERS.T
    return np.mean((cp / CP_REF - 1.0) ** 2, axis=-1)


# Two-objective trade-off: a CUBIC cp/R(T) cannot match propane below and above 500 K at once.
LOW_T = T_GRID <= 500.0
CUBIC_POWERS = T_POWERS[:, :4]


def cp_tradeoff_errors(X):
    """Mean squared relative cp error below and above 500 K. X holds a1..a4; returns (..., 2)."""
    rel2 = (X @ CUBIC_POWERS.T / CP_REF - 1.0) ** 2
    return np.stack([rel2[..., LOW_T].mean(axis=-1), rel2[..., ~LOW_T].mean(axis=-1)], axis=-1)


def cp_tradeoff_c3h8(X):
    """log10 of cp_tradeoff_errors. The log spreads the front evenly and keeps the Pareto set."""
    return np.log10(cp_tradeoff_errors(X))


def cp_tradeoff_pareto_set(n_points=2000):
    """Exact Pareto-optimal coefficients of cp_tradeoff_c3h8, shape (n_points, 4).

    Both errors are convex quadratics in a1..a4, so every Pareto-optimal fit minimizes
    w*f_low + (1-w)*f_high for some w in [0, 1]: a weighted linear least-squares problem.
    """
    scale = 1000.0 ** np.arange(4)  # solve with T/1000 for conditioning
    A = CUBIC_POWERS / scale / CP_REF[:, None]
    weights = np.concatenate([[0.0], 1.0 / (1.0 + 10.0 ** np.linspace(8, -8, n_points - 2)), [1.0]])
    pareto_set = []
    for w in weights:
        row_weight = np.where(LOW_T, np.sqrt(w / LOW_T.sum()), np.sqrt((1.0 - w) / (~LOW_T).sum()))
        b = np.linalg.lstsq(A * row_weight[:, None], row_weight, rcond=None)[0]
        pareto_set.append(b / scale)
    return np.array(pareto_set)
