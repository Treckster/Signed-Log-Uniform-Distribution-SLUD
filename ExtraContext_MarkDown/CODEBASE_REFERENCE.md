# SLUD Codebase Reference

> Living context document for the **EstevanSLUD** branch (v0.1-beta).
> Purpose: give a future Claude session enough grounding to skip re-reading every file. Update this file whenever the architecture, encodings, or problem set changes.

---

## 1. What this project is

The repo introduces and benchmarks the **Signed Log-Uniform Distribution (SLUD)** — a parameter-encoding / sampling scheme for population-based black-box optimizers (PSO, etc.) on problems whose decision variables:

1. span many orders of magnitude (e.g. `1e-14` to `1e2`), **and**
2. can take either sign.

It is benchmarked against two baselines:

| Encoder | Decision-variable space | What the optimizer sees |
|---|---|---|
| **LIN**  | linear in physical units | the raw value, sign and magnitude both linear |
| **LUD**  | log-uniform magnitude **+** separate sign axis | `n_vars` doubles: first half = log10|x|, second half ∈ [-1, 1] (sign extracted via `np.sign`) |
| **SLUD** | signed log-uniform on a single axis | `n_vars` stays the same; the axis is widened to `[2·log10(lb), 2·log10(ub)]` and the midpoint splits sign |

SLUD's selling point: same dimensionality as LIN, log-spaced resolution like LUD, sign handled implicitly without an extra coordinate.

---

## 2. File map

```
repo/
├── pyproject.toml               # uv-managed project metadata + direct deps
├── uv.lock                      # uv lockfile (full transitive pin); commit this
├── .python-version              # uv: pinned interpreter (3.13)
├── funcs.py                     # 4 test objectives (rosen, brown, powell, poly7)
├── SignedLogUniDist.py          # SOLE driver: nested loops over (function, encoder, seed), append to Stats/{func}/{encoder}.csv
├── plot_dists_example.py        # Generates dists_example.png and dists_example_semilogy.png (linear vs SLUD curve illustration)
├── statss.py                    # Score a Stats CSV by the fobjmin success criterion; per-cell summary print
├── plots/                       # Generated figures (convergence + distribution illustrations)
├── Stats/{func}/{encoder}.csv   # Persisted multi-run results (200 rows each at v0.1-beta)
├── ExtraContext_MarkDown/       # This folder — context docs for AI collaboration
└── .vscode/, .venv/, __pycache__/   # IDE / venv / runtime cache (gitignored)
```

Removed on `EstevanSLUD` (preserved on `main`):
- `bkup/`, `LATEX/`, `LogSigned.code-workspace`, `COB-2025-2080.pdf` — cleanup at v0.1-beta (commit `846f4db`)
- `SignedUniLog.py` (single-run driver), `UniLog(DEPRECATED).py` (early prototype) — consolidated post-v0.1-beta; only the multi-run harness is kept (see §11 for ideas worth preserving from the deprecated file)

`main` is frozen — do not modify.

---

## 3. The three encoders, precisely

All three live as plain functions at the top of `SignedLogUniDist.py`. Signature: `encoder(xis, bounds) -> X` where `xis` is the optimizer's decision vector and `X` is the physical-space vector handed to the objective.

### 3.1 LIN — linear magnitude + sign carrier

```python
def LIN(xis, bounds):
    h = xis.shape[-1] // 2
    return xis[..., :h] * np.sign(xis[..., h:])
```

Optimizer-space layout: doubled vector `[|x_1|, …, |x_n|, s_1, …, s_n]`. Optimizer bounds: `xl = [lb_mag, -1]`, `xu = [ub_mag, +1]`. **Output is `n_phys`-length physical vector.** Sign extraction is done by the encoder; objectives in `funcs.py` are pure n_phys-variable math.

### 3.2 LUD — log magnitude + sign carrier

```python
def LUD(xis, bounds):
    h = xis.shape[-1] // 2
    return 10.0 ** xis[..., :h] * np.sign(xis[..., h:])
```

Optimizer-space layout: same doubled vector as LIN, but the magnitude half holds `log10|x|`. Optimizer bounds: `xl = [log10(lb_mag), -1]`, `xu = [log10(ub_mag), +1]`. Output is `n_phys`-length physical vector.

### 3.3 SLUD — signed log uniform, single unit axis

The optimizer searches a unit axis `[-1, 1]` (or `[0, 1]` / `[-1, 0]` for fixed-sign vars) regardless of the physical magnitude bounds. The encoder is the symmetric signed log map:

```
X = sign(xi) * MIN * (MAX / MIN)^|xi|
```

```python
def SLUD_Variable_Definition(bounds):
    """xl,xu per type: T=0 → [-1,1]; T=+1 → [0,1]; T=-1 → [-1,0]."""
    T = bounds[:, 2].astype(int)
    xl = np.where(T == 0, -1.0, np.minimum(0, T*2 + 1)).astype(float)
    xu = np.where(T == 0,  1.0, np.maximum(0, T*2 - 1)).astype(float)
    return xl, xu

def SLUD(xis, bounds):
    MIN, MAX = bounds[:,0].astype(float), bounds[:,1].astype(float)
    TYP      = bounds[:,2].astype(int)
    use_linear = (TYP == 0) & (MIN <= 0.0)

    # safe substitutes prevent MAX/MIN blow-up where lin_map will be selected anyway
    safe_min = np.where(use_linear, 1.0, MIN)
    safe_max = np.where(use_linear, 1.0, MAX)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_map = np.where(xis == 0, MIN,
                           np.sign(xis) * safe_min * np.power(safe_max/safe_min, np.abs(xis)))
    lin_map = MIN + 0.5 * (xis + 1.0) * (MAX - MIN)
    return np.where(use_linear, lin_map, log_map)
```

**Geometry:**

- xi=0 → MIN (smallest magnitude). xi=±1 → ±MAX. Log-spaced in between, sign from `np.sign(xi)`.
- Optimizer axis is symmetric `[-1, 1]` for both-sign vars (no axis-doubling tricks).
- Special case 1: type-0 with `MIN ≤ 0` (bounds straddle/include zero) — falls back to a plain linear map `[-1, 1] → [MIN, MAX]`. Handles e.g. `[0, 2]` or `[-1, 1]` without diverging at the log.
- Special case 2: at exactly `xi = 0`, `np.sign(xi)=0` would zero out the result; we substitute `MIN` so the smallest-magnitude position remains representable. (Floating-point `xi==0` is rare in practice but pymoo's LHS can hit it.)
- The `bounds` array's third column (type) is honored by `SLUD_Variable_Definition`; values currently used in this repo are all 0 (both signs).

**Note on history.** This formulation supersedes a "doubled-axis with halfmark midpoint" version used in the original COB-2025 paper code, where the optimizer searched `[2·log10(lb), 2·log10(ub)]` and sign was inferred from which half of the axis the candidate landed in. The newer form is mathematically equivalent for type-0 in the limit, simpler, symmetric around zero, and natively supports type-±1 without separate logic. See §11 for the older formulation.

### 3.4 No hidden coupling (since v0.1.7-beta)

All three encoders return `n_phys`-length physical vectors; `funcs.py` is pure n_phys-variable math (no `len(x)` branching). Adding a new encoder requires no edits to `funcs.py`. Sign extraction lives in the encoder body for LIN and LUD; SLUD's signed-log map handles it natively.

---

## 4. Test-problem catalogue

| Function   | Optimum                                   | f* (target `fobjmin` in MANYRUNS) | LIN / LUD `n_vars` | SLUD `n_vars` | Magnitude bounds |
|------------|-------------------------------------------|-----------------------------------|--------------------|---------------|------------------|
| `rosen`    | x = (1, 1) (sign-coupled)                 | 1e-8                              | 4                  | 2             | `[1e-4, 1e2]`    |
| `brown`    | x ≈ (1e6, 2e-6)                           | 1e-10                             | 4                  | 2             | `[1e-8, 1e8]`    |
| `powell`   | x ≈ (1.098e-5, 9.106)                     | 1e-10                             | 4                  | 2             | `[1e-6, 1e2]`    |
| `poly7`    | cp(T) polynomial fit for C₃H₈, T∈[200,1000]K | 1e-5                           | 10                 | 5             | per-coefficient (see code) |

`poly7` minimizes mean relative squared error between a 5th-degree polynomial `cp_pred(T)` and a fixed reference `CPREF` over 100 temperature points. Bounds vary per coefficient (each spans ~6 decades).

---

## 5. Optimization loop

Inner body of `SignedLogUniDist.py`, run for each `(functoeval, decade_selector, iteration)`:

1. Pick `decade_selector ∈ {LIN, LUD, SLUD}` and `functoeval ∈ {rosen, brown, powell, poly7}`.
2. Build `bounds` (Nx3 array `[lb, ub, type]`) and the optimizer's `xl, xu` (unit axis for SLUD via `SLUD_Variable_Definition`; log10 for LUD; linear for LIN).
3. Wrap into a `SLUDProblem`. `_evaluate` is two lines: `out["F"] = evalfunc(decade_selector(X, bounds)).reshape(-1, 1)` — encoders and objectives both broadcast over the whole `(pop, n_var)` matrix in one numpy call. No process pool; no Ray.
4. Run **PSO** with `pop_size=n_pop` and **LHS** initial sampling.
5. Termination: `n_gen` reached **OR** `f < fobjmin` (`TerminateIfAny`).
6. Append a row to `Stats/{functoeval}/{encoder}.csv`.

Defaults: `n_pop=100`, `n_gen=500`, `n_iterations=1000`, `seed=iteration`. Outer loops currently set to `['brown']` only and `[LUD, SLUD, LIN]`.

### Multi-run output schema

`Stats/{functoeval}/{encoder_name}.csv`:

```
iteration, seed, final_objective_value, n_iter_opt, x0, x1, [x2, x3, ...]
```

`n_iter_opt` is the number of generations actually used (terminated early when `f < fobjmin`). The `x*` columns are **physical-space** values (post `decade_selector`), not raw optimizer coordinates.

### Stats post-processing (`statss.py`)

`score_runs(csv_file, fobjmin)` reads a Stats CSV and classifies each row by `final_objective_value <= fobjmin`. Returns:

- `n_total`, `n_success`, `success_rate` (fraction in [0, 1])
- `success_n_iter_opt`: count/mean/median/std/min/max/q25/q75 of `n_iter_opt` over runs that converged
- `failure_final_objective`: the same summary applied to `final_objective_value` over runs that didn't, so the failure mode is visible (stuck at saddle vs blew up)

`print_summary(name, result)` formats the dict for terminal viewing. Running `python3 statss.py` walks `Stats/{problem}/{encoder}.csv` and prints a per-cell block. The `FOBJMIN` dict at the top of `statss.py` is the per-problem threshold — kept in sync with the driver manually until Step 4 introduces a registry that statss can import from.

---

## 6. CSV results inventory at v0.1-beta

200 rows per file (1 header + 200 data). Files:

```
Stats/brown/{LIN,LUD,SLUD}.csv
Stats/brown-/{LIN,LUD,SLUD}.csv      # purpose of "-" suffix unclear; ask user
Stats/poly7/{LIN,LUD,SLUD}.csv
Stats/powell/{LIN,LUD,SLUD}.csv
Stats/powell-/{LIN,LUD,SLUD}.csv
Stats/rosen/{LIN,LUD,SLUD}.csv
```

Note the MANYRUNS driver currently loops 1000 iterations but existing CSVs only contain 200. They were generated under an older config.

---

## 7. Known smells / refactor + perf candidates

These are surfaced for future-session orientation, not action items — confirm scope with user before touching.

### Code structure
- **Massive duplication in the config block.** The if/elif tree for `(functoeval, decade_selector)` is hand-unrolled into 4 problems × 3 encoders ≈ 12 nearly-identical blocks setting `n_vars, ub, lb, bounds, xl, xu`. A `@dataclass` `ProblemSpec` per function + per-encoder `prepare(spec)` would compress this to ~80 lines.

### Performance
- **LHS sampling cost** is small but is computed every run; benign.
- **No JIT (numba/jax).** Could matter for poly7 if the population grows substantially, but at current scale the matrix multiply (`xis @ T_powers`) is already at numpy's BLAS path. Skip unless profiling proves otherwise.
- **If a future objective is genuinely expensive** (CFD/chem-kinetics, > ~10 ms/eval), reintroducing parallelism is reasonable — but go straight to a process pool (`concurrent.futures.ProcessPoolExecutor`) before reaching for Ray. Ray earns its keep only at distributed-cluster scale.

### Statistical methodology (for the "more comparisons" goal)
- Currently only PSO is used. UNSGA3 is imported but never instantiated.
- Only LHS sampling — could compare uniform-random, Sobol, Halton.
- Only one `n_pop`/`n_gen` pair. A budget-vs-success-rate sweep would strengthen the comparison.
- No statistical significance test between LIN/LUD/SLUD success rates (Wilcoxon / bootstrap CI) — straightforward to add in `statss.py`.

---

## 8. Glossary / shorthand

- **decade selector**: code-internal name for the encoder choice (LIN/LUD/SLUD). Misleading because LIN doesn't deal in decades.
- **`xis`**: the optimizer's raw decision vector (in `xl, xu` bounds).
- **`X` (capital)**: the physical-space vector after the encoder, fed to the objective.
- **`bounds`**: `Nx3` array `[lb_magnitude, ub_magnitude, type]`. `type`: 0 = both signs, +1 = positive only, -1 = negative only. Used by `SLUD_Variable_Definition` and `SLUD`.
- **`fobjmin`**: early-termination objective threshold — proxy for "converged".

---

## 9. Branch / version state

- Branch: `EstevanSLUD` (tracks `origin/EstevanSLUD`).
- Tag: `v0.1-beta` at commit `846f4db` (post-cleanup baseline).
- `main` is **frozen** (archival snapshot of the COB-2025 paper). Never modify.

---

## 10. Roadmap pointers (fill in as work proceeds)

Done in v0.1.2-beta (the simplify pass):
- [x] Vectorize LUD / SLUD encoders
- [x] Hoist class definitions out of the per-iteration loop; remove trailing dead code
- [x] Open CSV once per (function, encoder) cell; switch to `csv.DictWriter`

Done in v0.1.3-beta:
- [x] Port the unit-axis SLUD formulation from `MFChemVirt_Experimental`

Done in v0.1.4-beta:
- [x] Vectorize `funcs.py` to evaluate whole `(pop, n_var)` populations at once
- [x] Remove Ray; `_evaluate` is now a 2-line numpy chain (no IPC overhead)

Done in v0.1.5-beta:
- [x] Add `pyproject.toml` + `uv.lock` (uv-managed dependencies; Python 3.13; numpy / pymoo / matplotlib pinned via the lock file). `uv sync` reproduces the env from a fresh clone.

Done in v0.1.6-beta:
- [x] Rewrite `statss.py` to score on objective threshold (`final_objective_value <= fobjmin`), drop the magic-number `>= 501` heuristic.

Done in v0.1.7-beta:
- [x] Decouple objectives from encoder arity (Path A): sign extraction lives in the encoders; `funcs.py` is pure n_phys math. **Bug fix**: LUD's vectorized form (since v0.1.2-beta) was splitting on the population axis instead of the variable axis; pre-fix LUD numbers should not be trusted.

Open:
- [ ] Collapse the per-(problem, encoder) configuration into a registry
- [ ] Add DE / GA / ES baselines for cross-algorithm comparison

---

## 11. Former design notes

### From the deleted `UniLog(DEPRECATED).py`
Recoverable from `main` or pre-cleanup history if revisited:

- **Custom `LogUniformSampling` (pymoo `Sampling` subclass).** Initialized the population by drawing `10**uniform(log10_lo, log10_hi)` for magnitudes and random `±1` for signs, in the LUD-style 2N layout. The current code achieves a similar effect by using `LHS()` over log10-bounds, but an explicit sampler could be useful as another comparison axis (LHS vs uniform-log vs Sobol).
- **PSO knobs `adaptive=True, pertube_best=True, output=SingleObjectiveOutput()`** were used in the prototype but are not passed by the current driver. Cheap to re-enable for an ablation if PSO behavior becomes a question.
- The deprecated file used **per-individual** Ray dispatch (`func.remote(x) for x in X`) rather than the current batched-by-10 dispatch. Worth re-profiling both once the objectives are vectorized.

### Former SLUD formulation (paper version)
The original COB-2025 SLUD widened the optimizer axis to `[2·log10(lb), 2·log10(ub)]` and inferred sign from which half of the axis a candidate landed in:

```
halfrange = (2·log10(ub) - 2·log10(lb)) / 2
halfmark  = (2·log10(ub) + 2·log10(lb)) / 2
dist      = xi - halfmark
sign      = np.sign(dist)
unit_x    = |dist| / halfrange         # ∈ [0, 1]
X         = sign * lb * (ub/lb)^unit_x
```

Replaced (post-v0.1.2-beta) with the unit-axis form documented in §3.3, ported from the `MFChemVirt_Experimental` project. The new form is symmetric around zero, drops the `sgn = 2 if both signs else 1` axis-doubling bookkeeping, and natively supports type-±1 (single-sign) vars and type-0 vars whose bounds include zero. Existing CSV results in `Stats/` were generated under the *old* formulation and are NOT directly comparable to fresh runs under the new one — re-run any benchmarks before quoting numbers.
