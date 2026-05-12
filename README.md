# SLUD — Signed Log-Uniform Distribution

> **Para o orientador (pitch em pt-BR):**
> Muitos problemas de engenharia exigem otimizar parâmetros que variam em
> **várias ordens de grandeza** e que podem ter **sinal positivo ou negativo**
> (ex.: coeficientes de polinômios termodinâmicos, constantes cinéticas). As
> codificações usuais — linear (LIN) ou log-uniforme (LUD) — tratam mal pelo
> menos um desses dois lados. Este repositório propõe e avalia uma codificação
> alternativa, **SLUD** (Signed Log-Uniform Distribution), que mapeia um eixo
> unitário `[-1, 1]` para o espaço físico via
> `x = sign(ξ) · MIN · (MAX/MIN)^|ξ|`, cobrindo simultaneamente magnitude
> logarítmica e sinal. O código aqui faz o **benchmark sistemático** dessa
> ideia em quatro algoritmos populacionais (PSO, DE, GA, ES) e quatro funções
> de teste. Os resultados preliminares no problema `brown` mostram que SLUD
> deixa o **PSO chegar a 100% de sucesso** onde LIN fica em 66% e LUD em 88%,
> e reduz pela metade o número de gerações até convergência para DE e ES.

---

## What this repo is

A benchmark harness for comparing three ways of encoding the search variables
seen by a black-box optimizer when the **physical** variables span many
decades and can be of either sign:

| Encoder | Optimizer-space axis | Decoded as |
|---------|----------------------|------------|
| **LIN** | `[lb, ub]` magnitude + `[-1, 1]` sign carrier (doubled vector) | `\|xᵢ\| · sign(sᵢ)` |
| **LUD** | `[log10 lb, log10 ub]` + `[-1, 1]` sign carrier (doubled vector) | `10^xᵢ · sign(sᵢ)` |
| **SLUD** | unit axis `[-1, 1]` per variable | `sign(ξᵢ) · MIN · (MAX/MIN)^\|ξᵢ\|` |

LIN and LUD are the encodings the original COB-2025 paper compared. SLUD is
the candidate this fork proposes: it carries sign and log-scaled magnitude on
a **single unit-bounded coordinate**, which (a) halves the dimensionality the
optimizer sees, and (b) removes the artificial discontinuity at `s = 0` that
the doubled-vector encoders introduce.

## What it does today

For each `(problem, encoder, algorithm)` cell:

1. Run `n_iterations` independent optimizations with different seeds, each
   capped at `n_gen` generations or the first generation where the objective
   drops below a per-problem threshold `fobjmin`.
2. Persist every run to `Stats/<problem>/<encoder>.csv` with columns:
   `iteration, seed, algorithm, final_objective_value, n_iter_opt, x0, x1, …`
3. Compute success rate and `n_iter_opt` statistics (`statss.py`).
4. Render a 3×4 (encoder × algorithm) matrix of histograms with the success
   threshold drawn as a red dashed line (`plot_matrix.py`).

Algorithms: **PSO, DE, GA, ES** (pymoo 0.6).
Test problems: **rosen, brown, powell, poly7** (all in `funcs.py`).
Runs in parallel across all CPU cores via `ProcessPoolExecutor`.

## Quickstart

The repo uses [uv](https://github.com/astral-sh/uv) for dependency
management. Python 3.13 is pinned in `.python-version`.

```bash
# 1. install deps (creates .venv/ from pyproject.toml + uv.lock)
uv sync

# 2. run the benchmark — populates Stats/brown/{LIN,LUD,SLUD}.csv
uv run python SignedLogUniDist.py

# 3. score every CSV against its fobjmin threshold
uv run python statss.py

# 4. render the matrix plot (saved to plots/matrix_brown.png)
uv run python plot_matrix.py
```

To extend to all four test problems, edit `ACTIVE_PROBLEMS` at the top of
`SignedLogUniDist.py`. The `n_pop / n_gen / n_iterations` knobs are right
above it.

## Preliminary results

Paper-quality run on `brown` (Brown badly-scaled, optimum `(10⁶, 2·10⁻⁶)`),
`n_pop=100, n_gen=500, n_iterations=300`, success threshold `fobjmin = 1e-10`:

| encoder \ algorithm | PSO | DE | GA | ES |
|---|---|---|---|---|
| LIN  | 66.0% | 100% | 0.3% | 100% |
| LUD  | 88.3% | 100% | 0.0% | 100% |
| **SLUD** | **100%** | **100%** | 0.7% | 100% |

Median generations to convergence (successes only):

| encoder \ algorithm | PSO | DE | GA | ES |
|---|---|---|---|---|
| LIN  | 132 | 113 | — | 159 |
| LUD  | 110 | 113 | — | 110 |
| **SLUD** | 111 | **69** | — | **78** |

Takeaways:
- **SLUD is the only encoder that makes PSO reliable on this problem.**
- DE and ES are encoder-robust on `brown` (100% across the board), but SLUD
  cuts their convergence cost roughly in half.
- GA fails everywhere — expected, it is not built for badly-scaled problems
  without a more elaborate operator setup.
- The full 3×4 matrix of histograms is saved at `plots/matrix_brown.png`.

> ⚠ This is *one* problem so far. The other three (`rosen`, `powell`, `poly7`)
> have working scaffolding but no paper-quality data yet — that is the next
> step.

## Repository layout

```
SignedLogUniDist.py        driver: encoders, problem registry, algorithm
                           factories, parallel runner
funcs.py                   the four objective functions (pure n_phys math,
                           vectorized over populations)
statss.py                  CSV → success rate / n_iter_opt statistics
plot_matrix.py             3×4 (encoder × algorithm) histogram matrix
plot_dists_example.py      illustrative LIN/LUD/SLUD sample distributions
Stats/<problem>/*.csv      per-run results, one CSV per (problem, encoder)
plots/                     saved figures
ExtraContext_MarkDown/     internal reference doc for AI/code-review sessions
pyproject.toml, uv.lock    dependency manifest (uv)
```

## Status, scope, lineage

- Working branch: `EstevanSLUD`. Current tag: `v0.1.13-beta`.
- **`main` is frozen** — it is the archival snapshot of the original
  COB-2025 paper code (Treckster et al.) and must not be modified. All
  development happens on `EstevanSLUD`.
- Upstream: <https://github.com/Treckster/Signed-Log-Uniform-Distribution-SLUD>
- "Beta" tags reflect the fact that the scientific claims are still being
  built up: the harness is solid, the SLUD result on `brown` is real, but the
  comparison surface (more problems, statistical significance tests,
  resumable runs) is not yet complete.

## Proposed next steps (for discussion)

1. Extend the paper-quality run to `rosen`, `powell`, `poly7`.
2. Parametrize `plot_matrix.py` to render all four problem matrices.
3. Add a non-parametric significance test (Wilcoxon signed-rank or bootstrap
   CI) on `n_iter_opt` between encoders, so the claim "SLUD halves DE's
   convergence cost" comes with a p-value rather than just point estimates.
4. Crash-resumable CSV writes (currently each cell opens in `'w'` mode).
5. Add at least one higher-dimensional or noisy problem to stress-test the
   encoder advantage beyond the n=2 / n=5 toy regime.
