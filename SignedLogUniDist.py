import os
import csv
import numpy as np
import ray
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.termination import TerminateIfAny
from pymoo.termination import get_termination
from pymoo.termination.fmin import MinimumFunctionValueTermination
from pymoo.operators.sampling.lhs import LHS
from pymoo.algorithms.soo.nonconvex.pso import PSO

import funcs


def SLUD(xis, bounds):
    lbb, ubb = bounds[:, 0], bounds[:, 1]
    logub = 2 * np.log10(ubb)
    loglb = 2 * np.log10(lbb)
    halfrange = (logub - loglb) / 2
    halfmark = (logub + loglb) / 2
    dist = xis - halfmark
    sign = np.sign(dist)
    unit_x = np.abs(dist) / halfrange
    return sign * lbb * np.power(ubb / lbb, unit_x)


def LUD(xis, bounds):
    h = len(xis) // 2
    X = np.empty_like(xis)
    X[:h] = 10.0 ** xis[:h]
    X[h:] = xis[h:]
    return X


def LIN(xis, bounds):
    return xis


@ray.remote
def evaluate_batch(batch, decade_selector, bounds, evalfunc):
    return [evalfunc(decade_selector(x, bounds)) for x in batch]


class SLUDProblem(Problem):
    def __init__(self, n_var, xl, xu, decade_selector, bounds, evalfunc, batch_size=10):
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu)
        self._decode = decade_selector
        self._bounds = bounds
        self._eval = evalfunc
        self._batch_size = batch_size

    def _evaluate(self, X, out, *args, **kwargs):
        bs = self._batch_size
        batches = [X[i:i + bs] for i in range(0, len(X), bs)]
        futures = [evaluate_batch.remote(b, self._decode, self._bounds, self._eval)
                   for b in batches]
        results = [r for sub in ray.get(futures) for r in sub]
        out["F"] = np.array(results).reshape(-1, 1)


debug = False
n_threads = 16
n_pop = 100
n_gen = 500
n_iterations = 1000

ray.init(num_cpus=n_threads, local_mode=debug)


for functoeval in ['brown']:
    for decade_selector in [LUD, SLUD, LIN]:

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
                sgn = 2 if all(s == 0 for s in sign) else 1
                bounds = np.column_stack([lb, ub, sign])
                xl = np.log10(lb) * sgn
                xu = np.log10(ub) * sgn
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
                sgn = 2 if all(s == 0 for s in sign) else 1
                bounds = np.column_stack([lb, ub, sign])
                xl = np.log10(lb) * sgn
                xu = np.log10(ub) * sgn
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
                sgn = 2 if all(s == 0 for s in sign) else 1
                bounds = np.column_stack([lb, ub, sign])
                xl = np.log10(lb) * sgn
                xu = np.log10(ub) * sgn
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
                sgn = 2 if all(s == 0 for s in sign) else 1
                bounds = np.column_stack([lb, ub, sign])
                xl = np.log10(lb) * sgn
                xu = np.log10(ub) * sgn
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
        fieldnames = ['iteration', 'seed', 'final_objective_value', 'n_iter_opt'] + [f'x{i}' for i in range(n_vars)]

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


ray.shutdown()
