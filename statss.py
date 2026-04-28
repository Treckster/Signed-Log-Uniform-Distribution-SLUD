import csv
import os
import statistics


# Step 4 will replace this with `from SignedLogUniDist import PROBLEMS`.
FOBJMIN = {
    'rosen':  1e-8,
    'brown':  1e-10,
    'powell': 1e-10,
    'poly7':  1e-5,
}


def _stats_summary(values):
    """mean/median/std/min/max/q25/q75 for a list of floats; safe on empty/short input."""
    if not values:
        return {'count': 0}
    return {
        'count':  len(values),
        'mean':   statistics.mean(values),
        'median': statistics.median(values),
        'std':    statistics.stdev(values) if len(values) > 1 else 0.0,
        'min':    min(values),
        'max':    max(values),
        'q25':    statistics.quantiles(values, n=4)[0] if len(values) >= 4 else min(values),
        'q75':    statistics.quantiles(values, n=4)[2] if len(values) >= 4 else max(values),
    }


def score_runs(csv_file, fobjmin):
    """Score a Stats CSV using the fobjmin success criterion.

    A row counts as a success iff `final_objective_value <= fobjmin`.
    Returns success rate, n_iter_opt distribution over successful runs,
    and final_objective_value distribution over failed runs.
    """
    successes_iter = []
    failures_obj = []

    with open(csv_file, newline='') as f:
        for row in csv.DictReader(f):
            f_val = float(row['final_objective_value'])
            n_iter = int(float(row['n_iter_opt']))
            if f_val <= fobjmin:
                successes_iter.append(n_iter)
            else:
                failures_obj.append(f_val)

    n_total = len(successes_iter) + len(failures_obj)
    return {
        'n_total':                  n_total,
        'n_success':                len(successes_iter),
        'success_rate':             len(successes_iter) / n_total if n_total else 0.0,
        'fobjmin':                  fobjmin,
        'success_n_iter_opt':       _stats_summary(successes_iter),
        'failure_final_objective':  _stats_summary(failures_obj),
    }


def print_summary(name, result):
    print(f"\n=== {name} ===")
    print(f"  total={result['n_total']}  success={result['n_success']}  "
          f"rate={result['success_rate']*100:.1f}%  (fobjmin={result['fobjmin']:.0e})")
    s = result['success_n_iter_opt']
    if s['count']:
        print(f"  successes: n_iter_opt  mean={s['mean']:.1f}  median={s['median']:.1f}  "
              f"std={s['std']:.1f}  min={s['min']}  max={s['max']}  "
              f"q25={s['q25']:.1f}  q75={s['q75']:.1f}")
    f = result['failure_final_objective']
    if f['count']:
        print(f"  failures:  final_F  mean={f['mean']:.3e}  median={f['median']:.3e}  "
              f"min={f['min']:.3e}  max={f['max']:.3e}")


if __name__ == "__main__":
    stats_dir = "Stats"
    if not os.path.isdir(stats_dir):
        print(f"No {stats_dir}/ directory found — run SignedLogUniDist.py first.")
        raise SystemExit(0)

    for problem in sorted(os.listdir(stats_dir)):
        if problem not in FOBJMIN:
            continue
        for fname in sorted(os.listdir(os.path.join(stats_dir, problem))):
            if not fname.endswith('.csv'):
                continue
            csv_path = os.path.join(stats_dir, problem, fname)
            encoder = fname[:-4]
            print_summary(f"{problem}/{encoder}", score_runs(csv_path, FOBJMIN[problem]))
