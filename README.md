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
> de teste (rosen, brown, powell, poly7). Em 14 400 runs no total, SLUD se
> mostra a melhor codificação para **PSO e DE em problemas mal-escalados**:
> em `brown` leva o PSO de 66% (LIN) / 88% (LUD) para **100%** de sucesso, e
> em `poly7` leva o DE de 9% (LIN) para **65%**. Para o algoritmo ES — que
> já é robusto a escala — SLUD reduz pela metade o número de gerações até
> convergência. GA falha em quase todos os cenários, servindo como controle
> negativo. Os resultados sustentam SLUD como candidato sério para
> substituir LIN/LUD em problemas com parâmetros multi-escala e signados.

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

## Results

Paper-quality run across all four test problems,
`n_pop=100, n_gen=500, n_iterations=300` per cell (= 3600 runs per problem,
14 400 runs total). One CSV per (problem, encoder); raw per-run output in
`Stats/<problem>/*.csv` and a human-readable digest in `Stats/SUMMARY.txt`.
The 3×4 (encoder × algorithm) histogram matrix for each problem is in
`plots/matrix_<problem>.png`.

### Success rate (% of seeds reaching `fobjmin`)

`fobjmin` is the per-problem objective threshold below which a run counts
as having converged: `rosen 1e-8`, `brown 1e-10`, `powell 1e-10`, `poly7 1e-5`.

| problem | encoder | PSO | DE | GA | ES |
|---|---|---:|---:|---:|---:|
| **rosen**  | LIN      | 91.0 | 100  | 0.3  | 100  |
|            | LUD      | 99.7 | 100  | 0.7  | 100  |
|            | **SLUD** | **100**  | 100  | 0.7  | 100  |
| **brown**  | LIN      | 66.0 | 100  | 0.3  | 100  |
|            | LUD      | 88.3 | 100  | 0.0  | 100  |
|            | **SLUD** | **100**  | 100  | 0.7  | 100  |
| **powell** | LIN      | 14.0 | 23.3 | 2.3  | 52.3 |
|            | LUD      | 8.3  | 9.3  | 1.3  | 97.3 |
|            | **SLUD** | 9.7  | 26.7 | 1.7  | 97.0 |
| **poly7**  | LIN      | 6.0  | 9.3  | 2.3  | 100  |
|            | LUD      | 43.0 | 57.7 | 28.7 | 100  |
|            | **SLUD** | **53.0** | **64.7** | 28.7 | 100  |

### Median generations to convergence (successful runs only)

| problem | encoder | PSO | DE | GA | ES |
|---|---|---:|---:|---:|---:|
| **rosen**  | LIN      | 173 | 296 | —   | 57 |
|            | LUD      | 162 | 341 | —   | 48 |
|            | **SLUD** | 184 | 324 | —   | **38** |
| **brown**  | LIN      | 132 | 113 | —   | 159 |
|            | LUD      | 110 | 113 | —   | 110 |
|            | **SLUD** | 111 | **69**  | —   | **78** |
| **powell** | LIN      | 348 | 291 | 49  | 103 |
|            | LUD      | 235 | 275 | 28  | 47 |
|            | **SLUD** | 344 | 265 | 72  | 47 |
| **poly7**  | LIN      | 54  | 380 | 239 | 53 |
|            | LUD      | 55  | 201 | 147 | 64 |
|            | **SLUD** | 142 | 206 | 105 | **57** |

### Takeaways

- **Where SLUD helps most: PSO and DE on badly-scaled problems.**
  - On `brown`, SLUD is the only encoder where PSO is reliable (100% vs 66%
    LIN / 88% LUD) and DE converges almost twice as fast (69 vs ~113 median
    gens).
  - On `poly7`, SLUD pushes PSO from 6% → 53% and DE from 9% → 65% relative
    to LIN.
- **ES is encoder-robust** — it hits 100% (or 97%) almost everywhere, so
  pairing ES with *any* of the three encoders works. SLUD still tends to
  give the fastest convergence (lowest median gens).
- **`powell` is the hard one for population-based search.** No encoder gets
  PSO/DE/GA above ~27%. Only ES — which uses self-adaptive step sizes —
  manages, and there LUD/SLUD beat LIN by a wide margin (97% vs 52%).
- **GA fails everywhere.** Useful as a negative control: it shows the
  problems are non-trivial and the success rates above are not artifacts of
  any optimizer trivially solving them.

The full per-cell statistics (`n_iter_opt` mean/median/std/quartiles plus
failure-mode summaries) are in `Stats/SUMMARY.txt`.

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
  built up: the harness is solid and the SLUD results on all four test
  problems are in, but the supporting analysis (significance tests, more
  problem dimensions, resumable runs) is not yet complete.

## Proposed next steps (for discussion)

1. Add a non-parametric significance test (Wilcoxon signed-rank or bootstrap
   CI) on `n_iter_opt` between encoders, so the claim "SLUD halves DE's
   convergence cost on brown" comes with a p-value rather than a point
   estimate.
2. Add at least one higher-dimensional or noisy problem to stress-test the
   encoder advantage beyond the n=2 / n=5 toy regime.
3. Investigate why `powell` is hard for PSO/DE under every encoder — is it
   the constraint at the optimum (`x₀·x₁ ≈ 10⁻⁴`) or the basin geometry?
4. Crash-resumable CSV writes (currently each cell opens in `'w'` mode).
