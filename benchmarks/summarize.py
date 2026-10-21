"""Summarize a cobem2025.py run and put it next to the original COBEM 2025 numbers.

    uv run benchmarks/summarize.py

ERT (expected running time) = all evaluations spent, failed runs included,
divided by the number of successful runs (the COCO definition).
"""

import argparse
import csv
import os

import numpy as np

OLD_DIR = "cobem2025/Stats"
OLD_NAME = {"cp_c3h8": "poly7"}  # case renamed since COBEM
OLD_FAILED_GEN = 501  # the old scripts logged n_gen = 501 for runs that hit the 500-generation cap


def read_csv(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/cobem2025")
    args = parser.parse_args()

    print(f"{'case':9} {'enc':5} {'runs':>5} {'success%':>9} {'ERT evals':>10} {'median evals':>13} | {'COBEM success%':>15}")
    for file_name in sorted(os.listdir(args.out)):
        if not file_name.endswith(".csv"):
            continue
        case = file_name[:-4]
        rows = read_csv(os.path.join(args.out, file_name))
        for encoding in ["SLUD", "LUD", "LIN"]:
            runs = [r for r in rows if r["encoding"] == encoding]
            success = np.array([int(r["success"]) for r in runs], dtype=bool)
            n_eval = np.array([int(r["n_eval"]) for r in runs])
            ert = n_eval.sum() / success.sum() if success.any() else np.inf
            median = np.median(n_eval[success]) if success.any() else np.nan

            old = read_csv(os.path.join(OLD_DIR, OLD_NAME.get(case, case), f"{encoding}.csv"))
            old_success = np.mean([int(r["n_iter_opt"]) < OLD_FAILED_GEN for r in old]) * 100

            print(f"{case:9} {encoding:5} {len(runs):5d} {success.mean() * 100:9.1f} {ert:10.0f} {median:13.0f} | {old_success:15.1f}")


if __name__ == "__main__":
    main()
