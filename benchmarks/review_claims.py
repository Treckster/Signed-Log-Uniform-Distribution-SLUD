"""Review every claim so far with tests, effect sizes, Holm correction and power.

    uv run benchmarks/review_claims.py

A. PSO, COBEM cases, 200 seeds (results/cobem2025 + results/cobem2025_sinh)
   success rate: difference with 95% CI, Fisher exact test
   speed:        ERT ratio (other / SLUD) with bootstrap 95% CI
   power:        runs per arm needed to detect the observed success difference (80 %, alpha 0.05)
B. NSGA-III, cp_tradeoff and kursawe, 21 seeds (results/sinh_core_test/runs.csv)
   Mann-Whitney U test, A12 = P(SLUD run beats the other run), runs needed (t-test approximation)
C. Kursawe as a yes/no outcome (found the whole front: HV ratio >= 0.99)
   success counts, logistic regression success ~ encoding * lb, Fisher exact tests

All p-values within A and within B are Holm-corrected (p_holm).
"""

import glob
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_effectsize

RNG = np.random.default_rng(0)
POWER, ALPHA = 0.8, 0.05
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.3g}".format)


def runs_needed_proportions(p1, p2):
    if p1 == p2:
        return np.inf
    return NormalIndPower().solve_power(abs(proportion_effectsize(p1, p2)), alpha=ALPHA, power=POWER)


def runs_needed_means(a, b):
    d = abs(a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if d == 0:
        return np.inf
    return TTestIndPower().solve_power(d, alpha=ALPHA, power=POWER)


def ert(success, n_eval):
    return n_eval.sum() / success.sum() if success.any() else np.inf


def bootstrap_ert_ratio(other, ref, n_boot=2000):
    ratios = []
    for _ in range(n_boot):
        o = other.sample(len(other), replace=True, random_state=RNG)
        r = ref.sample(len(ref), replace=True, random_state=RNG)
        ratios.append(ert(o.success.values, o.n_eval.values) / ert(r.success.values, r.n_eval.values))
    return np.percentile(ratios, [2.5, 97.5])


def section_a():
    files = glob.glob("results/cobem2025/*.csv") + glob.glob("results/cobem2025_sinh/*.csv")
    runs = pd.concat([pd.read_csv(f).assign(case=os.path.basename(f)[:-4]) for f in files])
    pooled = runs[runs.case.isin(["powell", "powell-"])].assign(case="powell pooled")
    runs = pd.concat([runs, pooled])

    rows = []
    for case, data in runs.groupby("case"):
        ref = data[data.encoding == "SLUD"]
        for other_name in ["LUD", "LIN", "SINH"]:
            other = data[data.encoding == other_name]
            k1, n1, k2, n2 = ref.success.sum(), len(ref), other.success.sum(), len(other)
            low, high = confint_proportions_2indep(k1, n1, k2, n2, method="newcomb")
            ert_ref, ert_other = ert(ref.success.values, ref.n_eval.values), ert(other.success.values, other.n_eval.values)
            ratio_low, ratio_high = bootstrap_ert_ratio(other, ref)
            rows.append({
                "case": case, "vs": other_name,
                "SLUD %": 100 * k1 / n1, "other %": 100 * k2 / n2,
                "diff % [95% CI]": f"{100 * (k1 / n1 - k2 / n2):+.1f} [{100 * low:+.1f}, {100 * high:+.1f}]",
                "p_fisher": stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]]).pvalue,
                "ERT other/SLUD [95% CI]": f"{ert_other / ert_ref:.2f} [{ratio_low:.2f}, {ratio_high:.2f}]",
                "runs/arm needed": runs_needed_proportions(k1 / n1, k2 / n2),
            })
    table = pd.DataFrame(rows)
    table.insert(6, "p_holm", multipletests(table.p_fisher, method="holm")[1])
    print("\n=== A. PSO, COBEM cases: SLUD vs each other encoding (success rate and ERT) ===")
    print(table.to_string(index=False))


def a12(a, b):
    """Vargha-Delaney A12: probability that a random run of a beats a random run of b (0.5 = no difference)."""
    return (np.sum(a[:, None] > b[None, :]) + 0.5 * np.sum(a[:, None] == b[None, :])) / (len(a) * len(b))


def section_b(runs):
    rows = []
    base = runs[runs.lb_factor == 1.0]
    for (case, n_eval), data in base.groupby(["case", "n_eval"]):
        ref = data[data.encoding == "SLUD"].hv_ratio.values
        for other_name in ["PLAIN", "LUD", "SINH"]:
            other = data[data.encoding == other_name].hv_ratio.values
            rows.append({
                "case": case, "n_eval": n_eval, "vs": other_name,
                "SLUD median": np.median(ref), "other median": np.median(other),
                "A12": a12(ref, other),
                "p_mw": stats.mannwhitneyu(ref, other).pvalue,
                "runs/arm needed": runs_needed_means(ref, other),
            })
    table = pd.DataFrame(rows)
    table.insert(6, "p_holm", multipletests(table.p_mw, method="holm")[1])
    print("\n=== B. NSGA-III, HV ratio: SLUD vs each other encoding (lb as given) ===")
    print(table.to_string(index=False))


def section_c(runs):
    final = runs[(runs.case == "kursawe") & (runs.is_final == 1)].assign(success=lambda d: (d.hv_ratio >= 0.99).astype(int))
    print("\n=== C. Kursawe, found the whole front (HV ratio >= 0.99) at 20k evaluations ===")
    print(final.groupby("variant").success.agg(found="sum", runs="count").to_string())

    two_by_two = final[final.encoding.isin(["SLUD", "SINH"])]
    model = smf.logit("success ~ C(encoding) * C(lb_factor)", data=two_by_two).fit(disp=False)
    print("\nLogistic regression, success ~ encoding * lb (SLUD and SINH):")
    print(model.summary2().tables[1][["Coef.", "Std.Err.", "P>|z|"]])

    for label, a, b in [
        ("SLUD vs SINH (both lb values)", two_by_two.encoding == "SLUD", two_by_two.encoding == "SINH"),
        ("lb vs lb/1000 (both encodings)", two_by_two.lb_factor == 1.0, two_by_two.lb_factor < 1.0),
    ]:
        ka, na, kb, nb = two_by_two[a].success.sum(), a.sum(), two_by_two[b].success.sum(), b.sum()
        p = stats.fisher_exact([[ka, na - ka], [kb, nb - kb]]).pvalue
        print(f"Fisher, {label:32} {ka}/{na} vs {kb}/{nb}, p = {p:.3g}, runs/arm needed = {runs_needed_proportions(ka / na, kb / nb):.0f}")


def main():
    section_a()
    runs = pd.read_csv("results/sinh_core_test/runs.csv")
    section_b(runs)
    section_c(runs)


if __name__ == "__main__":
    main()
