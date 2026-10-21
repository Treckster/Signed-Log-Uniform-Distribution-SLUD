"""NSGA-III on the propane cp trade-off (exact Pareto front known): SLUD vs LUD vs LIN.

    uv run benchmarks/nsga3_cp_tradeoff.py --runs 31

Quality of the non-dominated set after every generation, against the exact front:
    HV ratio  hypervolume / hypervolume of the exact front (1 = perfect)
    IGD+      distance from the exact front to the set (0 = perfect)
Both are computed on objectives normalized by the exact front's ideal and nadir points.

Writes results/nsga3_cp_tradeoff/traces.npz and prints medians at several budgets.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

from slud import problems
from slud.encodings import ENCODINGS
from slud.optimize import run_nsga3

# Same magnitude bounds as the first four coefficients in the COBEM 2025 cp study.
LB = [1e-2, 1e-5, 1e-8, 1e-11]
UB = [1e2, 1e-1, 1e-4, 1e-7]
MAX_GEN = 300
HV_TARGET = 0.99  # "front found" once the HV ratio reaches this

FRONT = problems.cp_tradeoff_c3h8(problems.cp_tradeoff_pareto_set())
IDEAL, NADIR = FRONT.min(axis=0), FRONT.max(axis=0)


def normalize(F):
    return (F - IDEAL) / (NADIR - IDEAL)


HV_INDICATOR = HV(ref_point=np.array([1.1, 1.1]))
HV_FRONT = HV_INDICATOR(normalize(FRONT))
IGD_PLUS = IGDPlus(normalize(FRONT))


def run_job(job):
    encoding_name, seed = job
    encoding = ENCODINGS[encoding_name](LB, UB)
    r = run_nsga3(problems.cp_tradeoff_c3h8, encoding, n_obj=2, seed=seed, max_gen=MAX_GEN)
    hv_ratio = np.array([HV_INDICATOR(normalize(F)) / HV_FRONT for F in r["trace_F"]])
    igd_plus = np.array([IGD_PLUS(normalize(F)) for F in r["trace_F"]])
    return encoding_name, r["trace_n_eval"], hv_ratio, igd_plus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=31, help="seeds 0..runs-1 per encoding")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", default="results/nsga3_cp_tradeoff")
    args = parser.parse_args()

    jobs = [(e, seed) for e in ENCODINGS for seed in range(args.runs)]
    print(f"{len(jobs)} runs on {args.workers} processes")
    start = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run_job, jobs))
    print(f"done in {time.time() - start:.0f} s")

    os.makedirs(args.out, exist_ok=True)
    traces = {}
    n_eval = results[0][1]  # same for every run: fixed population and generations
    budgets = [len(n_eval) // 10, len(n_eval) // 3, len(n_eval) - 1]
    print(f"\n{'':5} {'median HV ratio at evals':>36} | {'median IGD+':>10} | runs reaching HV ratio {HV_TARGET}")
    print(f"{'enc':5} " + " ".join(f"{n_eval[b]:>11d}" for b in budgets) + f" | {n_eval[-1]:>10d} | {'runs':>5} {'median evals':>13}")
    for encoding_name in ENCODINGS:
        hv = np.array([r[2] for r in results if r[0] == encoding_name])
        igd = np.array([r[3] for r in results if r[0] == encoding_name])
        traces[f"{encoding_name}/hv_ratio"] = hv
        traces[f"{encoding_name}/igd_plus"] = igd
        reached = hv.max(axis=1) >= HV_TARGET
        first = np.argmax(hv >= HV_TARGET, axis=1)
        evals_to_target = np.median(n_eval[first[reached]]) if reached.any() else np.nan
        print(
            f"{encoding_name:5} "
            + " ".join(f"{np.median(hv[:, b]):11.4f}" for b in budgets)
            + f" | {np.median(igd[:, -1]):10.2e} | {reached.sum():2d}/{len(hv):2d} {evals_to_target:13.0f}"
        )
    traces["n_eval"] = n_eval
    np.savez_compressed(os.path.join(args.out, "traces.npz"), **traces)


if __name__ == "__main__":
    main()
