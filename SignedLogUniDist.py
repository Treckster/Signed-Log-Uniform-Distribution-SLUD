import os
import csv
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


n_pop = 100
n_gen = 500
n_iterations = 50


for functoeval in ['brown']:
    for decade_selector in [LIN, LUD, SLUD]:

        if functoeval == 'rosen':  # x unconstrained, fmin=0, xopt=(1,1)
            evalfunc = funcs.rosen
            fobjmin = 1.0e-8
            if decade_selector == LUD:
                n_vars = 4
                ub = [1E2, 1E2, 1, 1]
                lb = [1E-4, 1E-4, -1, -1]
                bounds = np.column_stack([lb, ub, np.zeros(n_vars)])
                xl = np.concatenate([np.log10(lb[0:2]), lb[2:4]])
                xu = np.concatenate([np.log10(ub[0:2]), ub[2:4]])
            elif decade_selector == SLUD:
                n_vars = 2
                ub = [1E2, 1E2]
                lb = [1E-4, 1E-4]
                sign = [0, 0]
                bounds = np.column_stack([lb, ub, sign])
                xl, xu = SLUD_Variable_Definition(bounds)
            else:
                n_vars = 4
                ub = [1E2, 1E2, 1, 1]
                lb = [1E-4, 1E-4, -1, -1]
                bounds = np.column_stack([lb, ub, np.zeros(n_vars)])
                xl = np.array(lb)
                xu = np.array(ub)

        elif functoeval == 'brown':  # badly-scaled, fmin=0, xopt~(1e6, 2e-6)
            evalfunc = funcs.brown
            fobjmin = 1.0e-10
            if decade_selector == LUD:
                n_vars = 4
                ub = [1E8, 1E8, 1, 1]
                lb = [1E-8, 1E-8, -1, -1]
                bounds = np.column_stack([lb, ub, np.zeros(n_vars)])
                xl = np.concatenate([np.log10(lb[0:2]), lb[2:4]])
                xu = np.concatenate([np.log10(ub[0:2]), ub[2:4]])
            elif decade_selector == SLUD:
                n_vars = 2
                ub = [1E8, 1E8]
                lb = [1E-8, 1E-8]
                sign = [0, 0]
                bounds = np.column_stack([lb, ub, sign])
                xl, xu = SLUD_Variable_Definition(bounds)
            else:
                n_vars = 4
                ub = [1E8, 1E8, 1, 1]
                lb = [1E-8, 1E-8, -1, -1]
                bounds = np.column_stack([lb, ub, np.zeros(n_vars)])
                xl = np.array(lb)
                xu = np.array(ub)

        elif functoeval == 'powell':  # badly-scaled, fmin=0, xopt~(1.098e-5, 9.106)
            evalfunc = funcs.powell
            fobjmin = 1.0e-10
            if decade_selector == LUD:
                n_vars = 4
                ub = [1E2, 1E2, 1, 1]
                lb = [1E-6, 1E-6, -1, -1]
                bounds = np.column_stack([lb, ub, np.zeros(n_vars)])
                xl = np.concatenate([np.log10(lb[0:2]), lb[2:4]])
                xu = np.concatenate([np.log10(ub[0:2]), ub[2:4]])
            elif decade_selector == SLUD:
                n_vars = 2
                ub = [1E2, 1E2]
                lb = [1E-6, 1E-6]
                sign = [0, 0]
                bounds = np.column_stack([lb, ub, sign])
                xl, xu = SLUD_Variable_Definition(bounds)
            else:
                n_vars = 4
                ub = [1E2, 1E2, 1, 1]
                lb = [1E-6, 1E-6, -1, -1]
                bounds = np.column_stack([lb, ub, np.zeros(n_vars)])
                xl = np.array(lb)
                xu = np.array(ub)

        elif functoeval == 'poly7':  # cp(T) polynomial fit for C3H8 over T in [200,1000]K
            evalfunc = funcs.poly7
            fobjmin = 1e-5
            if decade_selector == LUD:
                n_vars = 10
                ub = [1E2, 1E-1, 1E-4, 1E-7, 1E-10, 1, 1, 1, 1, 1]
                lb = [1E-2, 1E-5, 1E-8, 1E-11, 1E-14, -1, -1, -1, -1, -1]
                bounds = np.column_stack([lb, ub, np.zeros(n_vars)])
                xl = np.concatenate([np.log10(lb[0:5]), lb[5:10]])
                xu = np.concatenate([np.log10(ub[0:5]), ub[5:10]])
            elif decade_selector == SLUD:
                n_vars = 5
                ub = [1E2, 1E-1, 1E-4, 1E-7, 1E-10]
                lb = [1E-2, 1E-5, 1E-8, 1E-11, 1E-14]
                sign = [0, 0, 0, 0, 0]
                bounds = np.column_stack([lb, ub, sign])
                xl, xu = SLUD_Variable_Definition(bounds)
            else:
                n_vars = 10
                ub = [1E2, 1E-1, 1E-4, 1E-7, 1E-10, 1, 1, 1, 1, 1]
                lb = [1E-2, 1E-5, 1E-8, 1E-11, 1E-14, -1, -1, -1, -1, -1]
                bounds = np.column_stack([lb, ub, np.zeros(n_vars)])
                xl = np.array(lb)
                xu = np.array(ub)

        else:
            raise ValueError(f"Unknown function: {functoeval}")

        print(f"Evaluating {functoeval} with {n_vars} variables ({decade_selector.__name__})")

        func_dir = os.path.join("Stats", functoeval)
        os.makedirs(func_dir, exist_ok=True)
        csv_filename = os.path.join(func_dir, f"{decade_selector.__name__}.csv")
        # Encoders return n_phys-length physical vectors; n_vars is the optimizer-space size.
        n_phys = n_vars if decade_selector is SLUD else n_vars // 2
        fieldnames = ['iteration', 'seed', 'final_objective_value', 'n_iter_opt'] + [f'x{i}' for i in range(n_phys)]

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for iteration in range(n_iterations):
                problem = SLUDProblem(n_vars, xl, xu, decade_selector, bounds, evalfunc)
                algorithm = PSO(pop_size=n_pop, sampling=LHS())
                termination = TerminateIfAny(
                    get_termination("n_gen", n_gen),
                    MinimumFunctionValueTermination(fobjmin),
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

                X_transformed = decade_selector(res.X, bounds)
                row = {
                    'iteration': iteration,
                    'seed': iteration,
                    'final_objective_value': res.F[0],
                    'n_iter_opt': res.algorithm.n_gen,
                    **{f'x{i}': v for i, v in enumerate(X_transformed)},
                }
                writer.writerow(row)
