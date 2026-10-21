"""Figure: how SLUD, SINH and PLAIN map u to x, and the density each one puts on x.

    uv run benchmarks/plot_encodings.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slud.encodings import PLAIN, SINH, SLUD

LB, UB = 1e-3, 1e3
COLORS = {"SLUD": "C3", "SINH": "C1", "PLAIN": "C4"}


def main():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # (a) sinh is a straight line near zero and an exponential far from it
    z = np.linspace(-4, 4, 401)
    ax1.plot(z, np.sinh(z), color=COLORS["SINH"], lw=3, label="sinh(z) = (e^z - e^-z)/2")
    ax1.plot(z, z, "k--", label="z  (small z)")
    ax1.plot(z[z > 0], np.exp(z[z > 0]) / 2, ":", color="C3", lw=2, label="+e^z/2  (large z)")
    ax1.plot(z[z < 0], -np.exp(-z[z < 0]) / 2, ":", color="C3", lw=2, label="-e^-z/2  (large -z)")
    ax1.set_ylim(-20, 20)
    ax1.set_title("(a) sinh = line near 0, exponential outside")
    ax1.set_xlabel("z")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    # (b) the encodings: optimizer variable (normalized to [-1, 1] over its bounds) -> physical value x
    u = np.linspace(-1, 1, 2001)[:, None]
    for name, enc in [("SLUD", SLUD([LB], [UB])), ("SINH", SINH([LB], [UB])), ("PLAIN", PLAIN([LB], [UB]))]:
        Z = enc.xl + (u + 1) / 2 * (enc.xu - enc.xl)
        ax2.plot(u[:, 0], enc.decode(Z)[:, 0], color=COLORS[name], lw=2, label=name)
    ax2.set_yscale("symlog", linthresh=LB)
    ax2.axhspan(-LB, LB, color="0.9")
    ax2.set_title(f"(b) x(u), lb = {LB:g}, ub = {UB:g}  (grey: |x| < lb)")
    ax2.set_xlabel("optimizer variable, normalized to [-1, 1]")
    ax2.set_ylabel("physical value x (symlog axis)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    # (c) density of x for uniform u, on log-log axes (positive side)
    x = np.logspace(-6, 3, 500)
    K = np.arcsinh(UB / LB)
    p_slud = np.where(x >= LB, 1 / (2 * x * np.log(UB / LB)), 0.0)
    p_sinh = 1 / (2 * K * np.sqrt(x**2 + LB**2))
    p_plain = np.full_like(x, 1 / (2 * UB))
    ax3.loglog(x, p_slud, color=COLORS["SLUD"], lw=2, label="SLUD: 1 / (2 |x| ln(ub/lb))")
    ax3.loglog(x, p_sinh, color=COLORS["SINH"], lw=2, label="SINH: 1 / (2 K sqrt(x^2 + lb^2))")
    ax3.loglog(x, p_plain, color=COLORS["PLAIN"], lw=2, label="PLAIN: 1 / (2 ub)")
    ax3.axvline(LB, color="0.6", ls="--")
    ax3.set_title("(c) density p(x): slope -1 = log-uniform, flat = linear")
    ax3.set_xlabel("|x|")
    ax3.set_ylabel("p(x)")
    ax3.grid(alpha=0.3, which="both")
    ax3.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/encodings.png", dpi=150)


if __name__ == "__main__":
    main()
