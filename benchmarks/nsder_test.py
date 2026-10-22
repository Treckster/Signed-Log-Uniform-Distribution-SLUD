"""NSGA-III vs NSDE-R: same reference-direction selection, different variation operator.

    uv run benchmarks/nsder_test.py --runs 21

NSGA-III varies one variable at a time (SBX + polynomial mutation): fast on separable problems,
slow in diagonal (correlated) valleys. NSDE-R uses DE steps a + F (b - c), which follow the
population; with CR = 1.0 the variation is rotation invariant.

cp_tradeoff: correlated monomial coefficients (diagonal valley)   -> prediction: NSDE-R wins
kursawe:     weakly coupled O(1) problem                            -> prediction: NSGA-III equal or better
Each with PLAIN and SINH (c = lb). Screening budget: 30k evaluations (cp), 20k (Kursawe).

Writes results/nsder_test/runs.csv (HV ratio at each report budget).
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
from slud.encodings import PLAIN, SINH
from slud.optimize import run_nsga3

KURSAWE = get_problem("kursawe")
CASES = {
    #               objectives                  lb                    ub                 generations  report at generation
    "cp_tradeoff": (problems.cp_tradeoff_c3h8, np.array(cp.LB),      np.array(cp.UB),   300,         [99, 299]),
    "kursawe":     (KURSAWE,                   np.full(3, 1e-4),     np.full(3, 5.0),   200,         [19, 49, 199]),
}
HV_RATIO = {
    "cp_tradeoff": lambda F: cp.HV_INDICATOR(cp.normalize(F)) / cp.HV_FRONT,
    "kursawe": classics.hv_ratio_function(KURSAWE),
}
ALGORITHMS = {"NSGA-III": None, "NSDE-R CR=0.7": 0.7, "NSDE-R CR=1.0": 1.0}
ENCODINGS = {"PLAIN": PLAIN, "SINH": SINH}


def run_job(job):
    case, algorithm, encoding_name, seed = job
    objectives, lb, ub, max_gen, report = CASES[case]
    r = run_nsga3(objectives, ENCODINGS[encoding_name](lb, ub), n_obj=2, seed=seed, max_gen=max_gen, de_cr=ALGORITHMS[algorithm])
    return case, algorithm, encoding_name, seed, [((g + 1) * 100, HV_RATIO[case](r["trace_F"][g])) for g in report]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=21)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", default="results/nsder_test")
    args = parser.parse_args()

    jobs = [(c, a, e, s) for c in CASES for a in ALGORITHMS for e in ENCODINGS for s in range(args.runs)]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run_job, jobs))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "runs.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["case", "algorithm", "encoding", "seed", "n_eval", "hv_ratio"])
        for case, algorithm, encoding_name, seed, trace in results:
            for n_eval, hv in trace:
                writer.writerow([case, algorithm, encoding_name, seed, n_eval, hv])

    print(f"HV ratio, median [min - max] over {args.runs} seeds")
    for case, (_, _, _, _, report) in CASES.items():
        print(f"\n{case:12} {'':24} " + " ".join(f"{(g + 1) * 100:>22d}" for g in report) + "  evals")
        for a in ALGORITHMS:
            for e in ENCODINGS:
                hv = np.array([[v for _, v in r[4]] for r in results if r[:3] == (case, a, e)])
                print(f"{'':12} {a + ', ' + e:24} " + " ".join(f"{np.median(c):.3f} [{c.min():.3f}-{c.max():.3f}]".rjust(22) for c in hv.T))


if __name__ == "__main__":
    main()
