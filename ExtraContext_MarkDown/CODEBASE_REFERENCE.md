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
├── funcs.py                     # 4 test objectives (rosen, brown, powell, poly7)
├── SignedUniLogMANYRUNS.py      # SOLE driver: nested loops over (function, encoder, seed), append to Stats/{func}/{encoder}.csv
├── plot_dists_example.py        # Generates dists_example.png and dists_example_semilogy.png (linear vs SLUD curve illustration)
├── statss.py                    # Read a Stats CSV and compute count/mean/median/std/quartiles/failure-rate for one column
├── plots/                       # Generated figures (convergence + distribution illustrations)
├── Stats/{func}/{encoder}.csv   # Persisted multi-run results (200 rows each at v0.1-beta)
├── ExtraContext_MarkDown/       # This folder — context docs for AI collaboration
└── .vscode/, __pycache__/       # IDE / runtime cache
```

Removed on `EstevanSLUD` (preserved on `main`):
- `bkup/`, `LATEX/`, `LogSigned.code-workspace`, `COB-2025-2080.pdf` — cleanup at v0.1-beta (commit `846f4db`)
- `SignedUniLog.py` (single-run driver), `UniLog(DEPRECATED).py` (early prototype) — consolidated post-v0.1-beta; only the multi-run harness is kept (see §11 for ideas worth preserving from the deprecated file)

`main` is frozen — do not modify.

---

## 3. The three encoders, precisely

All three live as plain functions at the top of `SignedUniLogMANYRUNS.py`. Signature: `encoder(xis, bounds) -> X` where `xis` is the optimizer's decision vector and `X` is the physical-space vector handed to the objective.

### 3.1 LIN — pass-through

```python
def LIN(xis, bounds):
    return xis
```

The objective receives `xis` directly. For functions that need a sign, the convention is **doubled dimensionality**: first half is magnitude (linear), second half ∈ [-1, 1] used via `np.sign(...)` inside the objective. So a 2-D problem under LIN runs with `n_vars=4`.

### 3.2 LUD — log magnitude + linear sign axis

```python
def LUD(xis, bounds):
    X = np.zeros_like(xis)
    for i in range(len(xis)//2):           # first half: 10^xi
        X[i] = 10**xis[i]
    for i in range(len(xis)//2, len(xis)): # second half: pass-through, used as sign carrier
        X[i] = xis[i]
    return X
```

Optimizer's bounds: `xl = [log10(lb_mag), -1]`, `xu = [log10(ub_mag), +1]`. `n_vars` is also doubled. Sign is recovered downstream via `np.sign(x[i+n])` in `funcs.py`.

### 3.3 SLUD — signed log uniform, single axis

```python
def SLUD(xis, bounds):
    X = np.zeros_like(xis)
    lbb, ubb, sbb = bounds[:,0], bounds[:,1], bounds[:,2]
    for i in range(len(xis)):
        if sbb[i] in (1, -1):           # single-sign branch (currently DEPRECATED stubs)
            sign = sbb[i]
        else:                           # both signs — the interesting case
            xix       = xis[i]
            logub     = np.log10(ubb[i]) * 2     # doubled bounds
            loglb     = np.log10(lbb[i]) * 2
            halfrange = (logub - loglb) / 2
            halfmark  = (logub + loglb) / 2      # axis midpoint
            dist      = xix - halfmark
            sign      = np.sign(dist)
            unit_x    = np.abs(dist) / halfrange # |x| normalized to [0, 1]
        X[i] = sign * lbb[i] * np.power(ubb[i]/lbb[i], np.abs(unit_x))
    return X
```

**Conceptual picture** (the geometry that makes SLUD work):

- The optimizer's axis is widened to `[2·log10(lb), 2·log10(ub)]`. With both bounds positive and a positive lower (e.g. `lb=1e-4, ub=1e2`), this yields `[-8, 4]`. Spans 12 units instead of 6.
- The midpoint `halfmark = (logub + loglb)` (note: not divided — the doubling already placed it correctly) splits the axis into two halves.
- `dist > 0` → positive sign in physical space. `dist < 0` → negative.
- `unit_x = |dist|/halfrange` ∈ [0, 1] is the log-distance from midpoint, normalized.
- Final mapping `lb * (ub/lb)^unit_x` walks log-uniformly from `lb` (at midpoint) to `ub` (at axis edge), then sign is applied.

So a single normalized axis encodes both magnitude (log-spaced) and sign (which half).

The `sgn` multiplier (`sgn=2` when both signs, `sgn=1` when single sign) appears in the driver scripts when constructing `xl, xu`. With both signs allowed, `xl = log10(lb)*2`, `xu = log10(ub)*2`. With single sign, the doubling collapses. The single-sign branch (`sbb[i] != 0`) is currently labelled `DEPRECATED` in the code and is a candidate for removal or rework.

### 3.4 Hidden coupling worth flagging

`funcs.py` decides what to do with the input by `len(x)` (2/4 or 5/10). LUD and LIN feed 4 (or 10) elements with sign embedded; SLUD feeds 2 (or 5). **The objective is encoder-aware.** If we add a new encoder, every test function must learn its arity. This is a major refactor target — see §7.

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

Inner body of `SignedUniLogMANYRUNS.py`, run for each `(functoeval, decade_selector, iteration)`:

1. Pick `decade_selector ∈ {LIN, LUD, SLUD}` and `functoeval ∈ {rosen, brown, powell, poly7}`.
2. Build `bounds` (Nx3 array `[lb, ub, sign_constraint]`) and the optimizer's `xl, xu` (in log10 for LUD/SLUD, linear for LIN).
3. Wrap into a pymoo `Problem` whose `_evaluate` ships population batches of size 10 to Ray actors that call `decade_selector → evalfunc`.
4. Run **PSO** with `pop_size=n_pop` and **LHS** initial sampling (rationale: equal-prob-per-decade strata when bounds are log).
5. Termination: `n_gen` reached **OR** `f < fobjmin` (`TerminateIfAny`).
6. Track `algorithm.pop.get("F").min()` per generation in a callback.
7. Append a row to `Stats/{functoeval}/{encoder}.csv`.

Defaults at v0.1-beta: `n_pop=100`, `n_gen=500`, `n_threads=16`, `seed=iteration`. Outer loops currently set to `['brown']` only and `[LUD, SLUD, LIN]`; `range(1000)` for iterations.

### Multi-run output schema

`Stats/{functoeval}/{encoder_name}.csv`:

```
iteration, seed, final_objective_value, n_iter_opt, x0, x1, [x2, x3, ...]
```

`n_iter_opt` is the number of generations actually used (terminated early when `f < fobjmin`). The `x*` columns are **physical-space** values (post `decade_selector`), not raw optimizer coordinates.

### Stats post-processing (`statss.py`)

`calculate_statistics(csv, columns)` returns count/mean/median/std/min/max/q25/q75/failures/success%.

**Failure heuristic:** a row is a failure iff the analyzed column ≥ 501. This is meant for `n_iter_opt` against `n_gen=500` (i.e. "ran out of generations"). Magic number — couples this util to the driver's `n_gen` setting. Refactor candidate.

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
- **Encoder-aware objectives.** `funcs.py` switches behavior on `len(x)` — adding a new encoder means editing every test function. The encoder, not the objective, should own the `(decision-vec) → (physical-vec-with-sign)` mapping.
- **Class definitions inside the per-iteration loop** (`evaluate_batch`, `func`, `MyProblem`, `MyCallback`). Hoist them out; nothing in their definition depends on the iteration. Re-applying `@ray.remote` every iteration is the most wasteful instance of this.
- **Trailing dead code** at the bottom of `SignedUniLogMANYRUNS.py` (lines 432–449): a `print` of `X_opt` and a `convergence_plot.png` save that runs once after all loops and reflects only the last iteration of the last (encoder, function). Leftover from the single-run script — safe to remove.
- **`SLUD` and `LUD` use Python `for` loops** over the variable index. Trivially vectorizable with numpy ops on the whole bounds array.
- **Filename `SignedUniLogMANYRUNS.py`** — the "MANYRUNS" suffix made sense when there were two drivers; now that it's the only one, a rename (e.g. `slud_bench.py` or `run.py`) would clarify intent.

### Performance
- **Ray for analytical objectives is almost certainly net-negative.** The four test functions are microsecond-scale; Ray's IPC overhead per `evaluate_batch.remote(...)` will dominate. Profile a single-process baseline first; only keep Ray if the per-eval cost grows (e.g. when `poly7` is replaced by an actual CFD/chem-kinetics evaluation).
- **`batch_size = 10`** is hard-coded; with `pop_size=100` that's 10 Ray tasks per generation. Tune empirically.
- **LHS sampling cost** is small but is computed every run; benign.
- **No JIT (numba/jax) and no vectorized objective.** `poly7` calls `np.polyval` on a 100-point grid per individual; a vectorized batch eval (whole population at once) would be 10–100× faster than the per-individual loop.

### Statistical methodology (for the "more comparisons" goal)
- Currently only PSO is used. UNSGA3 is imported but never instantiated.
- Only LHS sampling — could compare uniform-random, Sobol, Halton.
- Only one `n_pop`/`n_gen` pair. A budget-vs-success-rate sweep would strengthen the comparison.
- Failure detection is heuristic (`n_iter_opt ≥ n_gen+1`). A direct success criterion (`final_objective_value ≤ fobjmin`) is already in the data and more reliable.
- No statistical significance test between LIN/LUD/SLUD success rates (Wilcoxon / bootstrap CI) — straightforward to add in `statss.py`.

---

## 8. Glossary / shorthand

- **decade selector**: code-internal name for the encoder choice (LIN/LUD/SLUD). Misleading because LIN doesn't deal in decades.
- **`xis`**: the optimizer's raw decision vector (in `xl, xu` bounds).
- **`X` (capital)**: the physical-space vector after the encoder, fed to the objective.
- **`bounds`**: `Nx3` array `[lb_magnitude, ub_magnitude, sign_constraint]`. `sign_constraint`: 0 = both signs, +1 = positive only, -1 = negative only.
- **`sgn` (in driver code)**: doubling factor for the optimizer axis (2 if both signs, 1 if single sign).
- **`fobjmin`**: early-termination objective threshold — proxy for "converged".

---

## 9. Branch / version state

- Branch: `EstevanSLUD` (tracks `origin/EstevanSLUD`).
- Tag: `v0.1-beta` at commit `846f4db` (post-cleanup baseline).
- `main` is **frozen** (archival snapshot of the COB-2025 paper). Never modify.

---

## 10. Roadmap pointers (fill in as work proceeds)

- [ ] Decouple objectives from encoder arity (encoder owns sign handling)
- [ ] Collapse the per-(problem, encoder) configuration into a registry
- [ ] Vectorize LUD / SLUD encoders
- [ ] Profile single-process vs Ray on each objective; remove Ray where it loses
- [ ] Add UNSGA3 / DE / CMA-ES baselines for cross-algorithm comparison
- [ ] Rewrite `statss.py` to score on objective threshold, not generation count
- [ ] Hoist class definitions out of the per-iteration loop; remove trailing dead code
- [ ] Rename `SignedUniLogMANYRUNS.py` to a less awkward name
- [ ] Add a `requirements.txt` / `pyproject.toml`

---

## 11. Former design notes (extracted from removed files)

Ideas from the deleted `UniLog(DEPRECATED).py` worth keeping in mind — recoverable from `main` or pre-cleanup history if revisited:

- **Custom `LogUniformSampling` (pymoo `Sampling` subclass).** Initialized the population by drawing `10**uniform(log10_lo, log10_hi)` for magnitudes and random `±1` for signs, in the LUD-style 2N layout. The current code achieves a similar effect by using `LHS()` over log10-bounds, but an explicit sampler could be useful as another comparison axis (LHS vs uniform-log vs Sobol).
- **PSO knobs `adaptive=True, pertube_best=True, output=SingleObjectiveOutput()`** were used in the prototype but are not passed by the current driver. Cheap to re-enable for an ablation if PSO behavior becomes a question.
- The deprecated file used **per-individual** Ray dispatch (`func.remote(x) for x in X`) rather than the current batched-by-10 dispatch. Worth re-profiling both once the objectives are vectorized.
