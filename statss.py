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


def _build_result(successes_iter, failures_obj, fobjmin):
    n_total = len(successes_iter) + len(failures_obj)
    return {
        'n_total':                  n_total,
        'n_success':                len(successes_iter),
        'success_rate':             len(successes_iter) / n_total if n_total else 0.0,
        'fobjmin':                  fobjmin,
        'success_n_iter_opt':       _stats_summary(successes_iter),
        'failure_final_objective':  _stats_summary(failures_obj),
    }


def score_runs(csv_file, fobjmin, group_by='algorithm'):
    """Score a Stats CSV using the fobjmin success criterion.

    A row counts as a success iff `final_objective_value <= fobjmin`.
    Returns `{group_key: stats_dict}`. If `group_by` is None or the
    column isn't present, all rows fall under the key 'all'.
    """
    groups = {}                     # key -> (success_iters, failure_objs)
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        use_group = group_by if (group_by and group_by in (reader.fieldnames or [])) else None
        for row in reader:
            key = row[use_group] if use_group else 'all'
            succ, fail = groups.setdefault(key, ([], []))
            f_val = float(row['final_objective_value'])
            if f_val <= fobjmin:
                succ.append(int(float(row['n_iter_opt'])))
            else:
                fail.append(f_val)

    return {key: _build_result(succ, fail, fobjmin) for key, (succ, fail) in groups.items()}


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
            for group_key, result in score_runs(csv_path, FOBJMIN[problem]).items():
                label = f"{problem}/{encoder}" if group_key == 'all' else f"{problem}/{encoder}/{group_key}"
                print_summary(label, result)
