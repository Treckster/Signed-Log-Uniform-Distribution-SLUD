"""Matrix plot: encoder × algorithm. Each cell shows the distribution of
final objective F for one (encoder, algorithm) pair. Top-left cell is
annotated to explain the layout."""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

import statss  # for FOBJMIN dict

PROBLEM = 'brown'
ENCODERS = ['LIN', 'LUD', 'SLUD']
ALGOS = ['PSO', 'DE', 'GA', 'ES']
FOBJMIN = statss.FOBJMIN[PROBLEM]


def load(problem):
    """Returns dict (encoder, algorithm) -> list of final F values."""
    data = {}
    for enc in ENCODERS:
        path = f"Stats/{problem}/{enc}.csv"
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                algo = row['algorithm']
                data.setdefault((enc, algo), []).append(
                    float(row['final_objective_value'])
                )
    return data


def main():
    data = load(PROBLEM)
    all_F = [v for vals in data.values() for v in vals]
    if not all_F:
        raise SystemExit(f"No data found in Stats/{PROBLEM}/*.csv")

    F_min = max(min(all_F), 1e-30)
    log_min = float(np.floor(np.log10(F_min)))
    log_max = float(np.ceil(np.log10(max(all_F))))
    bins = np.logspace(log_min, log_max, 32)

    fig, axes = plt.subplots(
        len(ENCODERS), len(ALGOS),
        figsize=(4 * len(ALGOS), 3 * len(ENCODERS)),
        sharex=True, sharey=False,
    )
    fig.suptitle(
        f"{PROBLEM}: final objective F distribution per (encoder × algorithm)\n"
        f"fobjmin = {FOBJMIN:.0e} (red dashed line)\n"
        f"top-left cell annotated as a legend for the rest",
        fontsize=11,
    )

    for i, enc in enumerate(ENCODERS):
        for j, algo in enumerate(ALGOS):
            ax = axes[i, j]
            F_vals = data.get((enc, algo), [])
            if not F_vals:
                ax.text(0.5, 0.5, "no data", ha='center', va='center',
                        transform=ax.transAxes, color='gray')
                ax.set_xscale('log')
                continue
            ax.hist(F_vals, bins=bins, color='steelblue',
                    edgecolor='black', alpha=0.75)
            ax.set_xscale('log')
            ax.axvline(FOBJMIN, color='red', linestyle='--', linewidth=1.5)

            n_success = sum(1 for v in F_vals if v <= FOBJMIN)
            rate = n_success / len(F_vals) * 100
            ax.text(
                0.97, 0.95,
                f"N={len(F_vals)}\n✓ {rate:.0f}%",
                ha='right', va='top', transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='lightgray', alpha=0.9),
            )

            if i == 0:
                ax.set_title(algo, fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(enc, fontsize=12, fontweight='bold')
            if i == len(ENCODERS) - 1:
                ax.set_xlabel('final F (log)')

    legend_ax = axes[0, 0]
    legend_ax.annotate(
        'fobjmin\n(success threshold)',
        xy=(FOBJMIN, legend_ax.get_ylim()[1] * 0.6),
        xytext=(FOBJMIN * 1e-6, legend_ax.get_ylim()[1] * 0.85),
        arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
        color='red', fontsize=8, ha='left',
    )
    legend_ax.annotate(
        'N = total runs\n✓ = % under fobjmin',
        xy=(0.85, 0.78),
        xytext=(0.05, 0.55),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.0),
        color='dimgray', fontsize=8,
    )
    legend_ax.annotate(
        'histogram of\nfinal F values',
        xy=(0.50, 0.30),
        xytext=(0.02, 0.05),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.0),
        color='steelblue', fontsize=8,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f"plots/matrix_{PROBLEM}.png"
    os.makedirs("plots", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"saved {out}")


if __name__ == "__main__":
    main()
