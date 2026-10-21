"""Two-way ANOVA: does the encoding (SLUD vs SINH) and the lb choice (lb vs lb/1000) change NSGA-III quality?

    uv run benchmarks/sinh_core_test.py --runs 21     # produces results/sinh_core_test/runs.csv
    uv run benchmarks/anova_slud_vs_sinh.py

Design per problem: 2 x 2 factorial, factors encoding and lb_factor, one run per seed and cell.
Response: HV ratio at the final budget.
The question "is SINH less sensitive to lb than SLUD?" is the encoding x lb_factor INTERACTION.

Prints, per problem:
    cell means          mean and standard deviation of each of the 4 cells
    classic ANOVA       F tests and partial eta^2 (share of variation explained by each effect)
    assumption checks   Shapiro-Wilk on residuals (normality), Levene across cells (equal variance)
    ART ANOVA           aligned rank transform (Wobbrock et al. 2011): same F tests on ranks,
                        valid when the residuals are not normal
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

MODEL = "hv_ratio ~ C(encoding) * C(lb_factor)"
EFFECTS = ["C(encoding)", "C(lb_factor)", "C(encoding):C(lb_factor)"]


def anova(data, response="hv_ratio"):
    model = smf.ols(MODEL.replace("hv_ratio", response), data=data).fit()
    table = sm.stats.anova_lm(model, typ=2)
    table["partial_eta2"] = (table["sum_sq"] / (table["sum_sq"] + table.loc["Residual", "sum_sq"])).drop("Residual")
    return model, table


def aligned_rank_anova(data):
    """ART: for each effect, strip the other effects out, rank, run the same ANOVA, keep that effect's row."""
    grand = data.hv_ratio.mean()
    mean_a = data.groupby("encoding").hv_ratio.transform("mean")
    mean_b = data.groupby("lb_factor").hv_ratio.transform("mean")
    cell = data.groupby(["encoding", "lb_factor"]).hv_ratio.transform("mean")
    residual = data.hv_ratio - cell
    estimated = {
        "C(encoding)": mean_a - grand,
        "C(lb_factor)": mean_b - grand,
        "C(encoding):C(lb_factor)": cell - mean_a - mean_b + grand,
    }
    rows = []
    for effect, estimate in estimated.items():
        ranked = data.assign(aligned=(residual + estimate).rank())
        _, table = anova(ranked, response="aligned")
        rows.append(table.loc[effect])
    return pd.DataFrame(rows, index=EFFECTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-csv", default="results/sinh_core_test/runs.csv")
    parser.add_argument("--out", default="results/sinh_core_test")
    args = parser.parse_args()

    runs = pd.read_csv(args.runs_csv)
    runs = runs[runs.encoding.isin(["SLUD", "SINH"]) & (runs.is_final == 1)]
    pd.set_option("display.float_format", "{:.4g}".format)

    fig, axes = plt.subplots(1, runs.case.nunique(), figsize=(12, 5))
    for ax, (case, data) in zip(axes, runs.groupby("case")):
        print(f"\n================ {case}: {len(data)} runs ================")
        print("\nCell means (std) of HV ratio:")
        print(data.pivot_table(index="encoding", columns="lb_factor", values="hv_ratio", aggfunc=["mean", "std"]))

        model, table = anova(data)
        print("\nClassic two-way ANOVA:")
        print(table[["sum_sq", "df", "F", "PR(>F)", "partial_eta2"]])

        cells = [g.hv_ratio.values for _, g in data.groupby(["encoding", "lb_factor"])]
        print(f"\nAssumptions: Shapiro-Wilk on residuals p = {stats.shapiro(model.resid).pvalue:.3g}  (small p: not normal)")
        print(f"             Levene across the 4 cells   p = {stats.levene(*cells).pvalue:.3g}  (small p: unequal variances)")

        print("\nAligned rank transform ANOVA (robust to non-normal residuals):")
        print(aligned_rank_anova(data)[["F", "PR(>F)"]])

        # interaction plot: one line per encoding; non-parallel lines = interaction
        for encoding, color in [("SLUD", "C3"), ("SINH", "C1")]:
            sub = data[data.encoding == encoding]
            means = sub.groupby("lb_factor").hv_ratio.mean().sort_index(ascending=False)
            ci = 1.96 * sub.groupby("lb_factor").hv_ratio.sem().sort_index(ascending=False)
            x = np.arange(len(means))
            ax.errorbar(x, means.values, yerr=ci.values, color=color, marker="o", capsize=4, lw=2, label=f"{encoding} mean, 95% CI")
            jitter = np.random.default_rng(0).uniform(-0.06, 0.06, len(sub))
            ax.plot((sub.lb_factor < 1).astype(int) + jitter + (0.12 if encoding == "SINH" else -0.12), sub.hv_ratio,
                    ".", color=color, alpha=0.4)
        ax.set_xticks([0, 1], ["lb", "lb / 1000"])
        ax.set_title(f"{case}: interaction plot (parallel lines = no interaction)")
        ax.set_ylabel("HV ratio at final budget")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "interaction.png"), dpi=150)


if __name__ == "__main__":
    main()
