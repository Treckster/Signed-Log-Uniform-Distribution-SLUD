import os
import csv
from dataclasses import dataclass
from typing import Callable
import numpy as np
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.termination import TerminateIfAny
from pymoo.termination import get_termination
from pymoo.termination.fmin import MinimumFunctionValueTermination
from pymoo.operators.sampling.lhs import LHS
from pymoo.algorithms.soo.nonconvex.pso import PSO

import funcs


def SLUD_Variable_Definition(bounds):
    """Optimizer-axis bounds per variable type.

    type ==  0 → [-1, 1]   (both signs)
    type == +1 → [ 0, 1]   (positive only)
    type == -1 → [-1, 0]   (negative only)
    """
    T = bounds[:, 2].astype(int)
    xl = np.where(T == 0, -1.0, np.minimum(0, T * 2 + 1)).astype(float)
    xu = np.where(T == 0,  1.0, np.maximum(0, T * 2 - 1)).astype(float)
    return xl, xu


def SLUD(xis, bounds):
    """Decode optimizer-space xis to physical space.

    Default: signed log map  sign(xi) * MIN * (MAX/MIN)^|xi|
    Type-0 vars whose bounds include zero (MIN <= 0) use a linear map
    [-1, 1] → [MIN, MAX] to avoid division by zero / log of non-positive.
    At xi = 0 the log map is replaced by MIN to dodge sign(0)=0 collapsing
    the result to zero (which is unreachable in a log-uniform distribution).
    """
    xis = np.asarray(xis, dtype=float)
    MIN = bounds[:, 0].astype(float)
    MAX = bounds[:, 1].astype(float)
    TYP = bounds[:, 2].astype(int)

    use_linear = (TYP == 0) & (MIN <= 0.0)

    safe_min = np.where(use_linear, 1.0, MIN)
    safe_max = np.where(use_linear, 1.0, MAX)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_map = np.where(
            xis == 0,
            MIN,
            np.sign(xis) * safe_min * np.power(safe_max / safe_min, np.abs(xis)),
        )

    lin_map = MIN + 0.5 * (xis + 1.0) * (MAX - MIN)

    return np.where(use_linear, lin_map, log_map)


def LUD(xis, bounds):
    """Decode a doubled-vector LUD candidate to physical space.

    Input layout (last axis): [log10|x_1|, ..., log10|x_n|, s_1, ..., s_n].
    Output: 10**log10|x_i| * sign(s_i), length n.
    Broadcasts over (pop, 2n) inputs.
    """
    h = xis.shape[-1] // 2
    return 10.0 ** xis[..., :h] * np.sign(xis[..., h:])


def LIN(xis, bounds):
    """Decode a doubled-vector LIN candidate to physical space.

    Input layout (last axis): [|x_1|, ..., |x_n|, s_1, ..., s_n].
    Output: |x_i| * sign(s_i), length n.
    """
    h = xis.shape[-1] // 2
    return xis[..., :h] * np.sign(xis[..., h:])


class SLUDProblem(Problem):
    def __init__(self, n_var, xl, xu, decade_selector, bounds, evalfunc):
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu)
        self._decode = decade_selector
        self._bounds = bounds
        self._eval = evalfunc

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self._eval(self._decode(X, self._bounds)).reshape(-1, 1)


@dataclass
class ProblemSpec:
    """Single source of truth for a benchmark problem in physical (decoded) coordinates."""
    name: str
    evalfunc: Callable
    fobjmin: float
    lb_mag: np.ndarray   # (n_phys,) positive magnitudes
    ub_mag: np.ndarray   # (n_phys,) positive magnitudes
    types:  np.ndarray   # (n_phys,) values in {-1, 0, 1}: 0=both signs, ±1=fixed sign


PROBLEMS = {
    'rosen':  ProblemSpec('rosen',  funcs.rosen,  1.0e-8,
                          np.array([1e-4, 1e-4]),
                          np.array([1e2,  1e2 ]),
                          np.array([0, 0])),
    'brown':  ProblemSpec('brown',  funcs.brown,  1.0e-10,
                          np.array([1e-8, 1e-8]),
                          np.array([1e8,  1e8 ]),
                          np.array([0, 0])),
    'powell': ProblemSpec('powell', funcs.powell, 1.0e-10,
                          np.array([1e-6, 1e-6]),
                          np.array([1e2,  1e2 ]),
                          np.array([0, 0])),
    'poly7':  ProblemSpec('poly7',  funcs.poly7,  1.0e-5,
                          np.array([1e-2, 1e-5, 1e-8, 1e-11, 1e-14]),
                          np.array([1e2,  1e-1, 1e-4, 1e-7,  1e-10]),
                          np.array([0, 0, 0, 0, 0])),
}


def prepare_SLUD(spec):
    bounds = np.column_stack([spec.lb_mag, spec.ub_mag, spec.types])
    xl, xu = SLUD_Variable_Definition(bounds)
    return len(spec.lb_mag), bounds, xl, xu


def _doubled_bounds(spec):
    """Bounds array for LIN/LUD: doubled (magnitude half + sign-carrier half)."""
    n_phys = len(spec.lb_mag)
    return np.column_stack([
        np.concatenate([spec.lb_mag, np.full(n_phys, -1.0)]),
        np.concatenate([spec.ub_mag, np.full(n_phys,  1.0)]),
        np.zeros(2 * n_phys),
    ])


def prepare_LUD(spec):
    n_phys = len(spec.lb_mag)
    xl = np.concatenate([np.log10(spec.lb_mag), np.full(n_phys, -1.0)])
    xu = np.concatenate([np.log10(spec.ub_mag), np.full(n_phys,  1.0)])
    return 2 * n_phys, _doubled_bounds(spec), xl, xu


def prepare_LIN(spec):
    n_phys = len(spec.lb_mag)
    xl = np.concatenate([spec.lb_mag, np.full(n_phys, -1.0)])
    xu = np.concatenate([spec.ub_mag, np.full(n_phys,  1.0)])
    return 2 * n_phys, _doubled_bounds(spec), xl, xu


ENCODERS = {
    'LIN':  (LIN,  prepare_LIN),
    'LUD':  (LUD,  prepare_LUD),
    'SLUD': (SLUD, prepare_SLUD),
}


n_pop = 100
n_gen = 500
n_iterations = 50

ACTIVE_PROBLEMS = ['brown']
ACTIVE_ENCODERS = ['LIN', 'LUD', 'SLUD']


for prob_name in ACTIVE_PROBLEMS:
    spec = PROBLEMS[prob_name]
    n_phys = len(spec.lb_mag)

    for enc_name in ACTIVE_ENCODERS:
        encoder, prepare = ENCODERS[enc_name]
        n_vars, bounds, xl, xu = prepare(spec)

        print(f"Evaluating {prob_name} with {n_vars} variables ({enc_name})")

        func_dir = os.path.join("Stats", prob_name)
        os.makedirs(func_dir, exist_ok=True)
        csv_filename = os.path.join(func_dir, f"{enc_name}.csv")
        fieldnames = ['iteration', 'seed', 'final_objective_value', 'n_iter_opt'] + [f'x{i}' for i in range(n_phys)]

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for iteration in range(n_iterations):
                problem = SLUDProblem(n_vars, xl, xu, encoder, bounds, spec.evalfunc)
                algorithm = PSO(pop_size=n_pop, sampling=LHS())
                termination = TerminateIfAny(
                    get_termination("n_gen", n_gen),
                    MinimumFunctionValueTermination(spec.fobjmin),
                )

                res = minimize(
                    problem=problem,
                    algorithm=algorithm,
                    termination=termination,
                    seed=iteration,
                    verbose=False,
                    save_history=False,
                    display=None,
                )

                X_phys = encoder(res.X, bounds)
                writer.writerow({
                    'iteration': iteration,
                    'seed': iteration,
                    'final_objective_value': res.F[0],
                    'n_iter_opt': res.algorithm.n_gen,
                    **{f'x{i}': v for i, v in enumerate(X_phys)},
                })
