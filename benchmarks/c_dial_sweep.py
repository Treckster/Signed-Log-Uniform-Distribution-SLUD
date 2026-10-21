"""The c dial of SINH: from SLUD-like (c small) to PLAIN-like (c near ub).

    uv run benchmarks/c_dial_sweep.py

SINH uses x = c * sinh(u * asinh(ub / c)). c is set on a log scale between the case's lb and ub:
    log10(c) = log10(lb) + t * (log10(ub) - log10(lb)),   t in T_LEVELS
t = 0 is the lb used so far (log-uniform almost everywhere), t near 1 is nearly linear.

Prediction: the best t is problem dependent. Optima deep inside the range (Brown x2 = 2e-6)
want small t; O(1) problems whose optimum sits near ub (Kursawe) want large t.

Writes results/c_dial_sweep/runs.csv: case, t, c_over_ub (first variable), seed, quality, success.
quality = HV ratio (NSGA-III cases) or -log10(f_best) (PSO case); success = reached the target.
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pymoo.problems import get_problem

sys.path.insert(0, os.path.dirname(__file__))
import nsga3_cp_tradeoff as cp
import nsga3_pymoo_classics as classics

from slud import problems
from slud.encodings import SINH
from slud.optimize import run_nsga3, run_pso

T_LEVELS = [0.0, 0.25, 0.5, 0.75, 0.95]
KURSAWE = get_problem("kursawe")
HV_RATIO = {
    "kursawe": classics.hv_ratio_function(KURSAWE),
    "cp_tradeoff": lambda F: cp.HV_INDICATOR(cp.normalize(F)) / cp.HV_FRONT,
}
#               algorithm  lb                    ub                 runs  budget (generations)
CASES = {
    "kursawe":     ("nsga3", np.full(3, 1e-4),   np.full(3, 5.0),   21,   200),
    "cp_tradeoff": ("nsga3", np.array(cp.LB),    np.array(cp.UB),   41,   300),
    "brown":       ("pso",   np.full(2, 1e-8),   np.full(2, 1e8),   50,   500),
}


def run_job(job):
    case, t, seed = job
    algorithm, lb, ub, _, max_gen = CASES[case]
    c = 10 ** (np.log10(lb) + t * (np.log10(ub) - np.log10(lb)))
    encoding = SINH(c, ub)
    if algorithm == "nsga3":
        objectives = KURSAWE if case == "kursawe" else problems.cp_tradeoff_c3h8
        r = run_nsga3(objectives, encoding, n_obj=2, seed=seed, max_gen=max_gen)
        quality = HV_RATIO[case](r["trace_F"][-1])
        success = quality >= 0.99
    else:
        r = run_pso(problems.brown_badly_scaled, encoding, 1e-10, seed, 100, max_gen)
        quality = -np.log10(r["f_best"])
        success = r["success"]
    return case, t, c[0] / ub[0], seed, quality, int(success)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", default="results/c_dial_sweep")
    args = parser.parse_args()

    jobs = [(case, t, seed) for case, spec in CASES.items() for t in T_LEVELS for seed in range(spec[3])]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run_job, jobs))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "runs.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["case", "t", "c_over_ub", "seed", "quality", "success"])
        writer.writerows(results)

    print(f"{'case':12} {'t':>5} {'c/ub':>9} {'median quality':>15} {'success':>9}")
    for case in CASES:
        for t in T_LEVELS:
            rows = [r for r in results if r[0] == case and r[1] == t]
            quality = np.array([r[4] for r in rows])
            print(f"{case:12} {t:5.2f} {rows[0][2]:9.1e} {np.median(quality):15.3f} {sum(r[5] for r in rows):4d}/{len(rows)}")


if __name__ == "__main__":
    main()
