"""Re-run the COBEM 2025 study (PSO; SLUD vs LUD vs LIN) with the slud package.

    uv run benchmarks/cobem2025.py --runs 200

Writes, in --out (default results/cobem2025):
    <case>.csv   one row per run: encoding, seed, success, n_eval, f_best, x_best
    traces.npz   per run and generation: evaluation count and best-so-far f
"""

import argparse
import csv
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from slud import problems
from slud.encodings import ENCODINGS
from slud.optimize import run_pso

POP_SIZE = 100
MAX_GEN = 500

# Same bounds and targets as the COBEM 2025 paper.
# name:      objective                        magnitude lb                        magnitude ub                    f_target
CASES = {
    "rosen":   (problems.rosenbrock,          [1e-4, 1e-4],                       [1e2, 1e2],                     1e-8),
    "powell":  (problems.powell_badly_scaled, [1e-6, 1e-6],                       [1e2, 1e2],                     1e-10),
    "powell-": (problems.powell_flipped,      [1e-6, 1e-6],                       [1e2, 1e2],                     1e-10),
    "brown":   (problems.brown_badly_scaled,  [1e-8, 1e-8],                       [1e8, 1e8],                     1e-10),
    "brown-":  (problems.brown_flipped,       [1e-8, 1e-8],                       [1e8, 1e8],                     1e-10),
    "cp_c3h8": (problems.cp_fit_c3h8,         [1e-2, 1e-5, 1e-8, 1e-11, 1e-14],   [1e2, 1e-1, 1e-4, 1e-7, 1e-10], 1e-5),
}


def run_job(job):
    case, encoding_name, seed = job
    objective, lb, ub, f_target = CASES[case]
    encoding = ENCODINGS[encoding_name](lb, ub)
    return case, encoding_name, run_pso(objective, encoding, f_target, seed, POP_SIZE, MAX_GEN)


def pad(arrays):
    """Stack runs of different length, repeating each run's last value."""
    length = max(len(a) for a in arrays)
    return np.array([np.pad(a, (0, length - len(a)), mode="edge") for a in arrays])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=200, help="number of seeds per case and encoding")
    parser.add_argument("--first-seed", type=int, default=0, help="seeds first-seed .. first-seed+runs-1")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--encodings", nargs="+", default=list(ENCODINGS), choices=list(ENCODINGS))
    parser.add_argument("--out", default="results/cobem2025")
    args = parser.parse_args()

    jobs = list(itertools.product(args.cases, args.encodings, range(args.first_seed, args.first_seed + args.runs)))
    print(f"{len(jobs)} runs on {args.workers} processes")
    start = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run_job, jobs, chunksize=8))
    print(f"done in {time.time() - start:.0f} s")

    os.makedirs(args.out, exist_ok=True)
    traces = {}
    for case in args.cases:
        n = len(CASES[case][1])
        with open(os.path.join(args.out, f"{case}.csv"), "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["encoding", "seed", "success", "n_eval", "f_best"] + [f"x{i}" for i in range(n)])
            for c, encoding_name, r in results:
                if c == case:
                    writer.writerow([encoding_name, r["seed"], int(r["success"]), r["n_eval"], r["f_best"], *r["x_best"]])
        for encoding_name in args.encodings:
            runs = [r for c, e, r in results if c == case and e == encoding_name]
            traces[f"{case}/{encoding_name}/n_eval"] = pad([r["trace_n_eval"] for r in runs])
            traces[f"{case}/{encoding_name}/f_best"] = pad([r["trace_f_best"] for r in runs])
    np.savez_compressed(os.path.join(args.out, "traces.npz"), **traces)


if __name__ == "__main__":
    main()
