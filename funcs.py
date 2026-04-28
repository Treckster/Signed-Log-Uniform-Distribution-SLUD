import numpy as np


# poly7 evaluates a 5-coefficient polynomial at 100 fixed temperatures and compares it
# to a fixed C3H8 reference. Pre-compute the reference once.
_T_GRID = np.linspace(200.0, 1000.0, 100)
_T_POWERS = _T_GRID ** np.arange(5)[:, None]                       # (5, 100)
_CPREF = np.array([0.93355381, 0.026424579, 6.1059727e-06,
                   -2.1977499e-08, 9.5149253e-12])
_CP_REF = _CPREF @ _T_POWERS                                        # (100,)


def rosen(x):
    x = np.asarray(x, dtype=float)
    return (1.0 - x[..., 0])**2 + 100.0 * (x[..., 1] - x[..., 0]**2)**2


def brown(x):
    """Brown badly-scaled. Optimum at (1e6, 2e-6)."""
    x = np.asarray(x, dtype=float)
    f1 = x[..., 0] - 1.0e6
    f2 = x[..., 1] - 2.0e-6
    f3 = x[..., 0] * x[..., 1] - 2.0
    return f1**2 + f2**2 + f3**2


def powell(x):
    x = np.asarray(x, dtype=float)
    f1 = 1.0e4 * x[..., 0] * x[..., 1] - 1.0
    f2 = np.exp(-x[..., 0]) + np.exp(-x[..., 1]) - 1.0001
    return f1**2 + f2**2


def poly7(x):
    """Mean relative squared error of a 5-coeff polynomial against C3H8 cp(T) reference."""
    x = np.asarray(x, dtype=float)
    cp_pred = x @ _T_POWERS                                         # (pop, 100) or (100,)
    rel_diff = (cp_pred - _CP_REF) / (_CP_REF + 1e-10)
    return np.mean(rel_diff**2, axis=-1)
