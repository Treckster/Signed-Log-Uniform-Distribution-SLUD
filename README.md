# SLUD: Signed Log-Uniform Distribution

An encoding of decision variables for black-box optimizers. It is meant for
parameters that span many decades and can take either sign, such as NASA/JANAF
polynomial coefficients and Arrhenius parameters in virtual chemistry.

One optimizer variable `u` in [-1, 1] maps to the physical value

    x = sign(u) * lb * (ub/lb)**|u|

so `u = 0` gives `lb`, `u = ±1` gives `±ub`, and every decade on either sign
gets the same share of the search space.

## Quick start

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync                                         # creates .venv with numpy, pymoo, pytest
uv run pytest                                   # checks the encodings and test problems
uv run benchmarks/cobem2025.py --runs 200       # re-runs the COBEM 2025 study
uv run benchmarks/summarize.py                  # success rate and ERT, next to the COBEM numbers
```

## Use in your own optimization

```python
from slud import SLUD
from slud.optimize import EncodedProblem   # pymoo Problem

enc = SLUD(lb=[1e-14, 1e-2], ub=[1e-10, 1e2])  # magnitude bounds, 0 < lb < ub
enc.xl, enc.xu                                  # bounds to give the optimizer
X = enc.decode(Z)                               # optimizer values -> physical values
```

`LUD` (log magnitude + sign variable) and `LIN` (linear magnitude + sign
variable) have the same interface and are the baselines.

## Layout

| Path | Content |
|---|---|
| `src/slud/encodings.py` | SLUD, LUD, LIN |
| `src/slud/problems.py` | Test problems on physical values (Rosenbrock, Powell/Brown badly scaled, propane cp fit) |
| `src/slud/optimize.py` | pymoo glue: encoded problem, PSO run, best-so-far trace |
| `benchmarks/` | Experiment scripts |
| `tests/` | pytest checks |
| `cobem2025/` | Original COBEM 2025 scripts, results and paper, kept as published |
| `reports/`, `research_notes/` | Literature and tooling review for the follow-up paper |
