"""NSGA-III on pymoo's own signed test problems: SLUD vs pymoo's published linear bounds.

    uv run benchmarks/nsga3_pymoo_classics.py --runs 11

These problems are O(1) and their Pareto sets cross zero, so they are a CONTROL:
they show what SLUD costs (or gains) where no many-decade scaling exists.

HV ratio = hypervolume of the feasible non-dominated set / hypervolume of pymoo's
Pareto front, on objectives normalized by the front's ideal and nadir points.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.problems import get_problem

from slud.encodings import LUD, PLAIN, SLUD
from slud.optimize import run_nsga3

# name:      magnitude lb (SLUD/LUD only)   ub = pymoo's own bound
PROBLEMS = {
    "kursawe": (1e-4, 5.0),
    "srn":     (1e-4, 20.0),
}
ENCODINGS = {"PLAIN (pymoo bounds)": PLAIN, "SLUD": SLUD, "LUD": LUD}
COLORS = {"PLAIN (pymoo bounds)": "C4", "SLUD": "C3", "LUD": "C0"}
MAX_GEN = 200
BUDGET_GENS = [19, 49, MAX_GEN - 1]  # 2k, 5k, 20k evaluations


def hv_ratio_function(problem):
    front = problem.pareto_front()
    ideal, nadir = front.min(axis=0), front.max(axis=0)
    hv = HV(ref_point=np.full(problem.n_obj, 1.1))
    hv_front = hv((front - ideal) / (nadir - ideal))
    return lambda F: hv((F - ideal) / (nadir - ideal)) / hv_front


def run_job(job):
    problem_name, encoding_name, seed = job
    problem = get_problem(problem_name)
    lb, ub = PROBLEMS[problem_name]
    encoding = ENCODINGS[encoding_name]([lb] * problem.n_var, [ub] * problem.n_var)
    r = run_nsga3(problem, encoding, problem.n_obj, seed=seed, max_gen=MAX_GEN)
    hv_ratio = hv_ratio_function(problem)
    return problem_name, encoding_name, seed, [hv_ratio(r["trace_F"][g]) for g in BUDGET_GENS], r["trace_F"][-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=11)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", default="results/nsga3_pymoo_classics")
    args = parser.parse_args()

    jobs = [(p, e, s) for p in PROBLEMS for e in ENCODINGS for s in range(args.runs)]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run_job, jobs))

    print(f"HV ratio vs pymoo Pareto front, median [min - max] over {args.runs} seeds")
    print(f"{'problem':8} {'encoding':21} " + " ".join(f"{(g + 1) * 100:>22d}" for g in BUDGET_GENS) + "  evals")
    for p in PROBLEMS:
        for e in ENCODINGS:
            hv = np.array([r[3] for r in results if r[0] == p and r[1] == e])
            print(f"{p:8} {e:21} " + " ".join(f"{np.median(c):.3f} [{c.min():.3f}-{c.max():.3f}]".rjust(22) for c in hv.T))

    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(12, 5))
    for ax, p in zip(axes, PROBLEMS):
        ax.plot(*get_problem(p).pareto_front().T, "k.", ms=2, label="pymoo Pareto front")
        for e in ENCODINGS:
            F = next(r[4] for r in results if r[0] == p and r[1] == e and r[2] == 0)
            ax.plot(*F.T, "o", ms=4, mfc="none", color=COLORS[e], label=e)
        ax.set_title(f"{p}: NSGA-III, {MAX_GEN * 100} evaluations, seed 0")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fronts.png"), dpi=150)


if __name__ == "__main__":
    main()
