import numpy as np

def powell(x):
    # x_transformed = 10**  # Transform the variables
    # Example objective function to minimize
    # This should be replaced with the actual function you want to optimize
    # x = np.asarray(x, dtype=float)
    if len(x) == 4:
        xis= x[0]*np.sign(x[2])
        yis = x[1]*np.sign(x[3])
    elif len(x) == 2:
        xis = x[0]
        yis = x[1]
    f1 = 1.0e4 * xis * yis - 1.0
    f2 = np.exp(-xis) + np.exp(-yis) - 1.0001
    return f1**2 + f2**2

def rosen(x):
    # Preserve the original “sign-coupling” logic for 4-vector calls
    if len(x) == 4:
        xis = x[0] * np.sign(x[2])
        yis = x[1] * np.sign(x[3])
    # …and the simple pass-through for 2-vector calls
    elif len(x) == 2:
        xis = x[0]
        yis = x[1]
    else:
        raise ValueError("Input must have length 2 or 4.")

    # Single-objective Rosenbrock
    return (1.0 - xis)**2 + 100.0 * (yis - xis**2)**2


def brown(x):
    """
    Brown badly-scaled test function.

    Parameters
    ----------
    x : array-like, shape (>=2,)
        Candidate point [x1, x2, …].  Only the first two entries are used.
    bounds_id : any, optional
        Ignored - kept so the signature matches your existing optimisation code.

    Returns
    -------
    float
        Objective value  (x1 - 1e6)² + (x2 - 2e-6)² + (x1·x2 - 2)²
    """
    if len(x) == 4:
        xis= x[0]*np.sign(x[2])
        yis = x[1]*np.sign(x[3])
    elif len(x) == 2:
        xis = x[0]
        yis = x[1]

    f1 = xis - 1.0e6     # residual 1
    f2 = yis - 2.0e-6    # residual 2
    f3 = xis*yis - 2.0  # residual 3

    return f1*f1 + f2*f2 + f3*f3


def poly7(x):
    xix = np.zeros_like(x)
    if len(x) == 10:
        xix[0] = x[0] * np.sign(x[5])
        xix[1] = x[1] * np.sign(x[6])
        xix[2] = x[2] * np.sign(x[7])
        xix[3] = x[3] * np.sign(x[8])
        xix[4] = x[4] * np.sign(x[9])
    elif len(x) == 5:
        xix[0] = x[0]
        xix[1] = x[1]
        xix[2] = x[2]
        xix[3] = x[3]
        xix[4] = x[4]

    n=100
    T_GRID = np.linspace(200.0, 1000.0, n)            # 100 temperature points
    CPREF= [0.93355381, 0.026424579, 6.1059727e-06, -2.1977499e-08, 9.5149253e-12] # c3h8
    CP_REF = np.polyval(                                 # reference cp(T)
        CPREF[::-1],                    # highest power first
        T_GRID
    )
    cp_pred = np.polyval(xix[::-1], T_GRID)            # flip → highest power first
    diff    = cp_pred - CP_REF
    # return float(np.dot(diff, diff))/n                 # SSE (no √)
    # Mean relative squared error
    rel_diff = (cp_pred - CP_REF) / (CP_REF + 1e-10)  # epsilon prevents division by zero
    return float(np.dot(rel_diff, rel_diff))/n