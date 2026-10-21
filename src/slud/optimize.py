"""Run one pymoo optimization on an encoded problem."""

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.config import Config
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.core.termination import TerminateIfAny
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.fmin import MinimumFunctionValueTermination
from pymoo.util.ref_dirs import get_reference_directions

Config.warnings["not_compiled"] = False


class EncodedProblem(Problem):
    """pymoo works on the encoded variables Z; the objective sees physical values X."""

    def __init__(self, objective, encoding, n_obj=1):
        super().__init__(n_var=encoding.n_var, n_obj=n_obj, xl=encoding.xl, xu=encoding.xu)
        self.objective = objective
        self.encoding = encoding

    def _evaluate(self, Z, out, *args, **kwargs):
        out["F"] = self.objective(self.encoding.decode(Z)).reshape(len(Z), self.n_obj)


class EncodedPymooProblem(Problem):
    """Any pymoo problem (objectives and constraints) searched through an encoding."""

    def __init__(self, problem, encoding):
        super().__init__(
            n_var=encoding.n_var,
            n_obj=problem.n_obj,
            n_ieq_constr=problem.n_ieq_constr,
            xl=encoding.xl,
            xu=encoding.xu,
        )
        self.problem = problem
        self.encoding = encoding

    def _evaluate(self, Z, out, *args, **kwargs):
        out.update(self.problem.evaluate(self.encoding.decode(Z), return_as_dictionary=True))


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


class FrontSoFar(Callback):
    """Evaluation count and feasible non-dominated objective vectors after every generation."""

    def __init__(self):
        super().__init__()
        self.n_eval = []
        self.F = []

    def notify(self, algorithm):
        self.n_eval.append(algorithm.evaluator.n_eval)
        self.F.append(algorithm.opt.get("F")[algorithm.opt.get("feas")])


def run_nsga3(objectives, encoding, n_obj, seed, n_partitions=99, max_gen=300):
    """NSGA-III with Das-Dennis reference directions; population size = number of directions.

    objectives is either a function of physical X or a pymoo Problem (constraints included).
    """
    problem = (
        EncodedPymooProblem(objectives, encoding)
        if isinstance(objectives, Problem)
        else EncodedProblem(objectives, encoding, n_obj)
    )
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
    trace = FrontSoFar()
    minimize(
        problem,
        NSGA3(ref_dirs=ref_dirs, pop_size=len(ref_dirs)),
        get_termination("n_gen", max_gen),
        seed=seed,
        callback=trace,
    )
    return {"seed": seed, "trace_n_eval": np.array(trace.n_eval), "trace_F": trace.F}
