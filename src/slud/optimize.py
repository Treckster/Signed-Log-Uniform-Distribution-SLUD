"""Run one pymoo optimization on an encoded problem."""

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.config import Config
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.core.termination import TerminateIfAny
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.fmin import MinimumFunctionValueTermination

Config.warnings["not_compiled"] = False


class EncodedProblem(Problem):
    """pymoo works on the encoded variables Z; the objective sees physical values X."""

    def __init__(self, objective, encoding):
        super().__init__(n_var=encoding.n_var, n_obj=1, xl=encoding.xl, xu=encoding.xu)
        self.objective = objective
        self.encoding = encoding

    def _evaluate(self, Z, out, *args, **kwargs):
        out["F"] = self.objective(self.encoding.decode(Z))[:, None]


class BestSoFar(Callback):
    """Evaluation count and best objective value after every generation."""

    def __init__(self):
        super().__init__()
        self.n_eval = []
        self.f_best = []

    def notify(self, algorithm):
        self.n_eval.append(algorithm.evaluator.n_eval)
        self.f_best.append(algorithm.opt.get("F").min())


def run_pso(objective, encoding, f_target, seed, pop_size=100, max_gen=500):
    """PSO with LHS start; stops at f <= f_target or after max_gen generations."""
    termination = TerminateIfAny(
        get_termination("n_gen", max_gen),
        MinimumFunctionValueTermination(f_target),
    )
    trace = BestSoFar()
    res = minimize(
        EncodedProblem(objective, encoding),
        PSO(pop_size=pop_size, sampling=LHS()),
        termination,
        seed=seed,
        callback=trace,
    )
    return {
        "seed": seed,
        "success": bool(res.F[0] <= f_target),
        "n_eval": res.algorithm.evaluator.n_eval,
        "f_best": res.F[0],
        "x_best": encoding.decode(res.X),
        "trace_n_eval": np.array(trace.n_eval),
        "trace_f_best": np.array(trace.f_best),
    }
