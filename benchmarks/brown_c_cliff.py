"""Investigate the Brown cliff: SINH success collapses near c = 1e4 (t = 0.75) but not around it.

    uv run benchmarks/brown_c_cliff.py

Fine sweep of t (c = 10^(-8 + 16 t)) with PSO and IPOP-CMA-ES; writes results/brown_c_cliff/runs.csv
with the final point of every run, so the stuck locations can be inspected.
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from slud import problems
from slud.encodings import SINH
from slud.optimize import run_cmaes, run_pso

LB, UB = np.full(2, 1e-8), np.full(2, 1e8)
T_LEVELS = np.round(np.arange(0.50, 0.951, 0.05), 2)
RUNS = 30


def run_job(job):
    algorithm, t, seed = job
    c = 10 ** (np.log10(LB) + t * (np.log10(UB) - np.log10(LB)))
    encoding = SINH(c, UB)
    if algorithm == "pso":
        r = run_pso(problems.brown_badly_scaled, encoding, 1e-10, seed)
    else:
        r = run_cmaes(problems.brown_badly_scaled, encoding, 1e-10, seed)
    return algorithm, t, c[0], seed, int(r["success"]), r["n_eval"], r["f_best"], *r["x_best"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", default="results/brown_c_cliff")
    args = parser.parse_args()

    jobs = [(a, t, s) for a in ["pso", "cmaes"] for t in T_LEVELS for s in range(RUNS)]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run_job, jobs))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "runs.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["algorithm", "t", "c", "seed", "success", "n_eval", "f_best", "x0", "x1"])
        writer.writerows(results)

    print(f"{'t':>5} {'c':>8} | {'PSO success':>11} {'median f':>9} | {'CMA-ES success':>14} {'median evals':>12}")
    for t in T_LEVELS:
        p = [r for r in results if r[0] == "pso" and r[1] == t]
        m = [r for r in results if r[0] == "cmaes" and r[1] == t]
        print(f"{t:5.2f} {p[0][2]:8.0e} | {sum(r[4] for r in p):6d}/{len(p)} {np.median([r[6] for r in p]):9.1e} | "
              f"{sum(r[4] for r in m):9d}/{len(m)} {np.median([r[5] for r in m]):12.0f}")


if __name__ == "__main__":
    main()
