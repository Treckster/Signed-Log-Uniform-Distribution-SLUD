"""Does writing the cp coefficients in T/1000 units (dimensionless) beat the encodings?

    uv run benchmarks/nsga3_dimensionless_check.py --runs 5

Raw:            optimizer sees a_k (spans ~10 decades), bounds as in COBEM 2025.
Dimensionless:  optimizer sees b_k = a_k * 1000**k (all O(1-50)), magnitudes in [1e-2, 1e2].
                PLAIN is one plain linear variable per coefficient on [-1e2, 1e2].

Prints HV ratio over seeds and saves two plots of the seed-0 fronts in --out.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import nsga3_cp_tradeoff as base

from slud import problems
from slud.encodings import LIN, LUD, PLAIN, SLUD
from slud.optimize import run_nsga3

SCALE = 1000.0 ** np.arange(4)  # b_k = a_k * SCALE_k
DIMLESS_LB = [1e-2] * 4
DIMLESS_UB = [1e2] * 4
MAX_GEN = 1500
BUDGET_GENS = [99, 299, MAX_GEN - 1]  # 10k, 30k, 150k evaluations


class Dimensionless:
    """Optimizer works on b_k = a_k * 1000**k; the objective still receives a_k."""

    def __init__(self, encoding):
        self.encoding = encoding
        self.n_var, self.xl, self.xu = encoding.n_var, encoding.xl, encoding.xu

    def decode(self, Z):
        return self.encoding.decode(Z) / SCALE


VARIANTS = {
    "SLUD raw": lambda: SLUD(base.LB, base.UB),
    "LUD raw": lambda: LUD(base.LB, base.UB),
    "LIN raw": lambda: LIN(base.LB, base.UB),
    "SLUD dimensionless": lambda: Dimensionless(SLUD(DIMLESS_LB, DIMLESS_UB)),
    "LUD dimensionless": lambda: Dimensionless(LUD(DIMLESS_LB, DIMLESS_UB)),
    "LIN dimensionless": lambda: Dimensionless(LIN(DIMLESS_LB, DIMLESS_UB)),
    "PLAIN dimensionless": lambda: Dimensionless(PLAIN(DIMLESS_LB, DIMLESS_UB)),
}
COLORS = {"SLUD": "C3", "LUD": "C0", "LIN": "C2", "PLAIN": "C4"}


def run_job(job):
    name, seed = job
    r = run_nsga3(problems.cp_tradeoff_c3h8, VARIANTS[name](), n_obj=2, seed=seed, max_gen=MAX_GEN)
    hv = [base.HV_INDICATOR(base.normalize(r["trace_F"][g])) / base.HV_FRONT for g in BUDGET_GENS]
    return name, seed, hv, r["trace_F"][-1]


def plot(results, names, title, path):
    plt.figure(figsize=(7, 6))
    plt.plot(*base.FRONT.T, "k-", lw=3, label="exact front")
    for name in names:
        F = next(r[3] for r in results if r[0] == name and r[1] == 0)
        hollow = "dimensionless" in name
        plt.plot(*F.T, "s" if hollow else "o", ms=4, color=COLORS[name.split()[0]],
                 mfc="none" if hollow else COLORS[name.split()[0]], label=name)
    plt.xlabel("log10 mean sq. relative cp error, 200-500 K")
    plt.ylabel("log10 mean sq. relative cp error, 500-1000 K")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", default="results/nsga3_dimensionless_check")
    args = parser.parse_args()

    jobs = [(name, seed) for name in VARIANTS for seed in range(args.runs)]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run_job, jobs))

    print(f"HV ratio vs exact front, median [min - max] over {args.runs} seeds")
    print(f"{'variant':20} " + " ".join(f"{(g + 1) * 100:>22d}" for g in BUDGET_GENS) + "  evals")
    for name in VARIANTS:
        hv = np.array([r[2] for r in results if r[0] == name])
        cells = [f"{np.median(c):.3f} [{c.min():.3f}-{c.max():.3f}]" for c in hv.T]
        print(f"{name:20} " + " ".join(f"{c:>22}" for c in cells))

    os.makedirs(args.out, exist_ok=True)
    plot(results, list(VARIANTS), "NSGA-III, 150k evaluations, seed 0: raw and dimensionless",
         os.path.join(args.out, "fronts_raw_and_dimensionless.png"))
    plot(results, [n for n in VARIANTS if "dimensionless" in n], "NSGA-III, 150k evaluations, seed 0: dimensionless only",
         os.path.join(args.out, "fronts_dimensionless_only.png"))


if __name__ == "__main__":
    main()
