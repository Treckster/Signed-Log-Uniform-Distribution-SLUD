import numpy as np


# Pre-computed once at module load: poly7 evaluates a 5-coeff polynomial at 100
# fixed temperatures and compares it to a fixed C3H8 reference.
_T_GRID = np.linspace(200.0, 1000.0, 100)
_T_POWERS = _T_GRID ** np.arange(5)[:, None]                       # (5, 100)
_CPREF = np.array([0.93355381, 0.026424579, 6.1059727e-06,
                   -2.1977499e-08, 9.5149253e-12])
_CP_REF = _CPREF @ _T_POWERS                                        # (100,)


def _xy_with_sign(x):
    """Extract (xis, yis) for 2-var problems. Length 4 = magnitude+sign-carrier; length 2 = direct."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    if n == 4:
        return x[..., 0] * np.sign(x[..., 2]), x[..., 1] * np.sign(x[..., 3])
    if n == 2:
        return x[..., 0], x[..., 1]
    raise ValueError("Input must have length 2 or 4.")


def powell(x):
    xis, yis = _xy_with_sign(x)
    f1 = 1.0e4 * xis * yis - 1.0
    f2 = np.exp(-xis) + np.exp(-yis) - 1.0001
    return f1**2 + f2**2


def rosen(x):
    xis, yis = _xy_with_sign(x)
    return (1.0 - xis)**2 + 100.0 * (yis - xis**2)**2


def brown(x):
    """Brown badly-scaled. Optimum near (1e6, 2e-6)."""
    xis, yis = _xy_with_sign(x)
    f1 = xis - 1.0e6
    f2 = yis - 2.0e-6
    f3 = xis * yis - 2.0
    return f1**2 + f2**2 + f3**2


def poly7(x):
    """Mean relative squared error of a 5-coeff polynomial against C3H8 cp(T) reference."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    if n == 10:
        xis = x[..., :5] * np.sign(x[..., 5:10])
    elif n == 5:
        xis = x[..., :5]
    else:
        raise ValueError("Input must have length 5 or 10.")

    cp_pred = xis @ _T_POWERS                                       # (pop, 100) or (100,)
    rel_diff = (cp_pred - _CP_REF) / (_CP_REF + 1e-10)
    return np.mean(rel_diff**2, axis=-1)
