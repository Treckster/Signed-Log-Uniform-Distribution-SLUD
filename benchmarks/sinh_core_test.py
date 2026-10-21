"""NSGA-III test of the sinh core: PLAIN, LUD, SLUD, SINH, and SLUD/SINH with lb 1000x smaller.

    uv run benchmarks/sinh_core_test.py --runs 11

cp_tradeoff: exact front known; a1 and a4 change sign along it (zero crossings).
kursawe:     pymoo classic, O(1) and its Pareto set passes through zero.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pymoo.problems import get_problem

sys.path.insert(0, os.path.dirname(__file__))
import nsga3_cp_tradeoff as cp
import nsga3_pymoo_classics as classics

from slud import problems
from slud.encodings import LUD, PLAIN, SINH, SLUD
from slud.optimize import run_nsga3

KURSAWE = get_problem("kursawe")
CASES = {
    #               objectives                  lb                    ub                 generations  report at generation
    "cp_tradeoff": (problems.cp_tradeoff_c3h8, np.array(cp.LB),      np.array(cp.UB),   1500,        [99, 299, 1499]),
    "kursawe":     (KURSAWE,                   np.full(3, 1e-4),     np.full(3, 5.0),   200,         [19, 49, 199]),
}
HV_RATIO = {
    "cp_tradeoff": lambda F: cp.HV_INDICATOR(cp.normalize(F)) / cp.HV_FRONT,
    "kursawe": classics.hv_ratio_function(KURSAWE),
}
#          name               encoding  lb factor
VARIANTS = {
    "PLAIN":             (PLAIN, 1.0),
    "LUD":               (LUD,   1.0),
    "SLUD":              (SLUD,  1.0),
    "SINH":              (SINH,  1.0),
    "SLUD lb/1000":      (SLUD,  1e-3),
    "SINH lb/1000":      (SINH,  1e-3),
}


def run_job(job):
    case, variant, seed = job
    objectives, lb, ub, max_gen, report = CASES[case]
    encoding, lb_factor = VARIANTS[variant]
    r = run_nsga3(objectives, encoding(lb * lb_factor, ub), n_obj=2, seed=seed, max_gen=max_gen)
    return case, variant, [HV_RATIO[case](r["trace_F"][g]) for g in report]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=11)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    jobs = [(c, v, s) for c in CASES for v in VARIANTS for s in range(args.runs)]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run_job, jobs))

    print(f"HV ratio vs exact/pymoo front, median [min - max] over {args.runs} seeds")
    for case, (_, _, _, _, report) in CASES.items():
        print(f"\n{case:12} {'':14} " + " ".join(f"{(g + 1) * 100:>22d}" for g in report) + "  evals")
        for v in VARIANTS:
            hv = np.array([r[2] for r in results if r[0] == case and r[1] == v])
            print(f"{'':12} {v:14} " + " ".join(f"{np.median(c):.3f} [{c.min():.3f}-{c.max():.3f}]".rjust(22) for c in hv.T))


if __name__ == "__main__":
    main()
