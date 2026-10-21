"""Encodings: map the variables the optimizer sees (Z) to physical values (X).

Every encoding has the same small interface:
    n_var      number of optimizer variables
    xl, xu     optimizer bounds (give these to pymoo / CMA-ES / scipy)
    decode(Z)  physical values, Z has shape (..., n_var), X has shape (..., n)

lb and ub are the smallest and largest MAGNITUDE allowed for each physical value.
All three encodings search both signs: x in [-ub, -lb] U [lb, ub].
"""

import numpy as np


def check_bounds(lb, ub):
    """Magnitude bounds must satisfy 0 < lb < ub. Raises on anything else."""
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    if lb.shape != ub.shape or np.any(lb <= 0) or np.any(ub <= lb):
        raise ValueError(f"magnitude bounds need 0 < lb < ub, got lb={lb}, ub={ub}")
    return lb, ub


def sign_of(s):
    """+1 for s >= 0 and -1 for s < 0 (never 0, unlike np.sign)."""
    return np.where(s < 0, -1.0, 1.0)


class SLUD:
    """Signed log-uniform: ONE optimizer variable u in [-1, 1] per physical value.

        x = sign(u) * lb * (ub/lb)**|u|

    u = 0 gives +lb, u = +-1 gives +-ub. Every decade, on either sign,
    gets the same share of u, so uniform/LHS sampling of u is log-uniform in |x|.
    """

    def __init__(self, lb, ub):
        self.lb, self.ub = check_bounds(lb, ub)
        self.n_var = self.lb.size
        self.xl = -np.ones(self.n_var)
        self.xu = np.ones(self.n_var)

    def decode(self, Z):
        return sign_of(Z) * self.lb * (self.ub / self.lb) ** np.abs(Z)


class SINH:
    """SLUD with a linear core: ONE variable u in [-1, 1] per physical value.

        x = lb * sinh(u * asinh(ub/lb))

    |x| >> lb: log-uniform, like SLUD.   |x| << lb: linear, so x = 0 is reachable
    and the sign changes smoothly. u = 0 gives 0, u = +-1 gives +-ub.
    """

    def __init__(self, lb, ub):
        self.lb, self.ub = check_bounds(lb, ub)
        self.n_var = self.lb.size
        self.xl = -np.ones(self.n_var)
        self.xu = np.ones(self.n_var)

    def decode(self, Z):
        return self.lb * np.sinh(Z * np.arcsinh(self.ub / self.lb))


class LUD:
    """Log-uniform magnitude plus a separate sign variable: TWO variables per value.

    Z = [m_1..m_n, s_1..s_n] with m = log10|x| in [log10 lb, log10 ub], s in [-1, 1].
    """

    def __init__(self, lb, ub):
        self.lb, self.ub = check_bounds(lb, ub)
        n = self.lb.size
        self.n_var = 2 * n
        self.xl = np.concatenate([np.log10(self.lb), -np.ones(n)])
        self.xu = np.concatenate([np.log10(self.ub), np.ones(n)])

    def decode(self, Z):
        n = self.lb.size
        return sign_of(Z[..., n:]) * 10.0 ** Z[..., :n]


class LIN:
    """Linear-uniform magnitude plus a separate sign variable: TWO variables per value.

    Z = [m_1..m_n, s_1..s_n] with m = |x| in [lb, ub], s in [-1, 1].
    """

    def __init__(self, lb, ub):
        self.lb, self.ub = check_bounds(lb, ub)
        n = self.lb.size
        self.n_var = 2 * n
        self.xl = np.concatenate([self.lb, -np.ones(n)])
        self.xu = np.concatenate([self.ub, np.ones(n)])

    def decode(self, Z):
        n = self.lb.size
        return sign_of(Z[..., n:]) * Z[..., :n]


class PLAIN:
    """Plain linear: ONE variable per value, uniform on [-ub, ub]. lb is only checked, not used.

    This is the usual way to bound a signed variable (pymoo's own test problems do it).
    """

    def __init__(self, lb, ub):
        self.lb, self.ub = check_bounds(lb, ub)
        self.n_var = self.ub.size
        self.xl = -self.ub
        self.xu = self.ub

    def decode(self, Z):
        return Z


ENCODINGS = {"SLUD": SLUD, "SINH": SINH, "LUD": LUD, "LIN": LIN}
