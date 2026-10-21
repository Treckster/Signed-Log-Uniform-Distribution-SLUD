# Recast SLUD as a benchmarked, packaged encoding

SLUD can become a strong journal paper and a reusable package, but only if the renewal changes what is claimed and how it is measured. The mathematical object, a log-uniform distribution mirrored onto both signs, is not new. It is a symlog with a zero-width linear core and a special case of the GAMBIT/ScannerBit `double_log_flat_join` prior. What is defensible is narrower and still valuable. SLUD is a single-gene, monotone, two-sided log encoding of decision variables for population-based optimizers. It halves the dimension compared with sign-plus-magnitude encodings, and none of the ten mainstream Python HPO/BBO libraries inspected offers it: every one rejects non-positive log bounds. The black-box optimization community will judge that claim with tools the current repo does not use. Those tools are COCO/BBOB or IOHprofiler problem suites, runtime counted in function evaluations, ERT and runtime ECDFs that keep failed (capped) runs, effect sizes with Holm-corrected rank tests, several optimizer families beyond PSO, and archived code and per-run data. On the engineering side, the cheap analytic tier needs vectorized NumPy and run-level process parallelism, not per-generation Ray tasks. The future Cantera flame tier needs a process pool with one Cantera object per worker. The recommended stack is uv + hatchling + NumPy + pymoo/pycma/scipy + ioh/cocoex + scipy.stats/statsmodels/baycomp, with pixi only for the custom Cantera C++ build. Several claims below rest on secondary sources because documentation sites were blocked during research. They are flagged inline and collected in a dedicated section.

## What the current repo actually tests, and where reviewers will push

Reading the code before the literature shows three structural weaknesses that matter more than any tooling choice. First, **the comparison is SLUD with n genes against LUD and LIN with 2n genes**. In `SignedUniLogMANYRUNS.py`, both LUD and LIN append a continuous sign gene in [−1, 1] per variable. The objectives in `funcs.py` then branch on `len(x)` to apply `np.sign` to those genes (local files `/home/user/Signed-Log-Uniform-Distribution-SLUD/SignedUniLogMANYRUNS.py`, `/home/user/Signed-Log-Uniform-Distribution-SLUD/funcs.py`). "LIN" is therefore "linear magnitude plus a sign gene", not the natural baseline of a plain uniform variable on [−ub, ub] with n genes. For Brown, LIN draws magnitudes uniformly from [1e-8, 1e8], so it almost never samples 2e-6. That makes it a straw man unless a plain-LIN baseline with sensible bounds sits beside it.

Second, **the analytic problems barely exercise the sign**. Rosenbrock's optimum is (1, 1), Brown's is (1e6, 2e-6) and Powell's is (1.098e-5, 9.106), all positive. Only the propane Cp polynomial has a negative coefficient (a₄ ≈ −2.2e-8 among values from 0.93 down to 9.5e-12) (local `funcs.py`). The paper's sign-flipped variants (x₂ → −x₂ in Brown and Powell, results in `Stats/brown-/` and `Stats/powell-/`) do move one optimum coordinate negative, but the code that produced them is not in the repo, and they flip only one variable at one fixed magnitude. A paper whose selling point is "signed, many-decade" variables needs reproducible instances whose optima have mixed signs at large and small magnitudes.

Third, **run accounting does not match the paper**. The script loops `range(1000)` over Brown only, the paper reports 200 runs, seeds are `seed = iteration`, and runs that hit the 500-generation cap are logged as `n_iter_opt` with no success flag (local `SignedUniLogMANYRUNS.py`). These are fixable, but they decide whether a reviewer trusts the numbers.

## The community benchmarks through COCO and IOHprofiler, not ad-hoc scripts

**COCO (`cocoex` + `cocopp`) is the reference platform reviewers in GECCO, TELO and Evolutionary Computation expect for continuous black-box benchmarking.** It is installed as `coco-experiment` (2.8.3, released 2026-10-21) but imported as `cocoex` ([coco-experiment README](https://github.com/numbbo/coco-experiment/tree/main/build/python)). The canonical loop iterates `for problem in cocoex.Suite("bbob", "", "")`, attaches an `Observer`, calls any solver as `fmin(problem, x0)`, and post-processes with `cocopp.main(folder + ' bfgs!')` against archived algorithm data ([example_experiment_simple.py](https://raw.githubusercontent.com/numbbo/coco-experiment/main/build/python/example/example_experiment_simple.py)). The problem is a plain callable, so a parametrization study wraps it as `lambda u: problem(T(u))`. The observer then logs in the original x-space, which is exactly what comparing SLUD, LUD and LIN on identical f-targets requires.

There is a catch that shapes the whole experimental design. **BBOB optima are drawn uniformly in [−4, 4] on a 1e-4 grid inside a [−5, 5]ⁿ domain** ([suite_bbob_legacy_code.c](https://raw.githubusercontent.com/numbbo/coco-experiment/main/src/suite_bbob_legacy_code.c)). Every variable therefore lives at the same O(1) scale, and BBOB cannot test SLUD's central premise. Its ill-conditioned functions, such as the ellipsoid with conditioning 1e6 in f2 (separable) and f10 (rotated) ([f_ellipsoid.c](https://raw.githubusercontent.com/numbbo/coco-experiment/main/src/f_ellipsoid.c)), have badly conditioned Hessians, not badly scaled variables. The f2/f10 pair is still useful. Axis-aligned ill-conditioning is equivalent to a diagonal rescaling that a per-coordinate transform might help with, while rotation cannot be undone coordinate by coordinate. In the paper, BBOB should serve as a **"no-harm" control**: SLUD should not hurt much when scale is already known.

**IOHexperimenter (`ioh` 0.3.22) is the better harness for the problems that matter to SLUD.** It re-hosts the 24 BBOB functions, SBOX-COST and CEC 2013/2022 in a C++ core. It has a "flexible interface for adding new suites and problems", and its logs feed IOHanalyzer ([IOHexperimenter README](https://github.com/IOHprofiler/IOHexperimenter)). Registering Powell/Brown badly scaled, the Cp fit and later a Cantera case with `ioh.wrap_problem` would give them the same ECDF/ERT machinery as BBOB. *The `wrap_problem` and logger method names come from prior knowledge; the docs site was unreachable, so check them against 0.3.22.*

The two badly scaled problems already have an authoritative pedigree. **CUTEst's `POWELLBS` and `BROWNBS` cite Problems 3 and 4 of Moré, Garbow & Hillstrom (ACM TOMS 1981)** ([POWELLBS.SIF](https://bitbucket.org/optrove/sif/raw/master/POWELLBS.SIF), [BROWNBS.SIF](https://bitbucket.org/optrove/sif/raw/master/BROWNBS.SIF)), and PyCUTEst 1.8.2 exposes them ([pycutest](https://github.com/jfowkes/pycutest/)). CUTEst's POWELLBS is a chained n-dimensional generalization, so the paper must say which version it uses. Validating `funcs.py` against PyCUTEst once in a test is a cheap credibility win.

Nevergrad gives useful positioning more than a harness. **`ng.p.Log` is effectively the community's standard positive-only log-uniform parameter**, and it has no signed variant ([parametrization.rst](https://github.com/facebookresearch/nevergrad/blob/main/docs/parametrization.rst)). Its benchmark plans `illcondi` and `far_optimum_es` show that scale and offset robustness is already a recognized benchmark axis ([experiments.py](https://github.com/facebookresearch/nevergrad/blob/main/nevergrad/benchmark/experiments.py)). Its last release was 1.0.12 in April 2025, so pin it ([PyPI](https://pypi.org/project/nevergrad/)). **pymoo (0.6.2, June 2026) is an algorithm provider, not a benchmark platform** ([PyPI](https://pypi.org/project/pymoo/)). It can stay as the PSO/DE layer, but a pymoo-only problem set invites a "why not BBOB?" review. *One research note found only pymoo 0.6.1.6 (Nov 2025) through search mirrors. The PyPI JSON API and a local install both showed 0.6.2, which is taken as authoritative.*

The resulting design has three tiers, each with a clear job.

| Tier | Problems | Purpose | Tooling |
|---|---|---|---|
| 1. Control | COCO `bbob` (24 functions, dims 2–40, instances), optionally `bbob-boxed` | Show SLUD costs little when scale is known | `cocoex` + `cocopp`, comparison against archive |
| 2. Badly scaled | MGH Powell/Brown (validated against PyCUTEst), plus a scaled, sign-randomized BBOB: yᵢ = sᵢ·xᵢ·10^{kᵢ}, kᵢ ∈ [−6, 6] | Test the premise: optima spanning decades, with mixed signs | `ioh.wrap_problem` + IOHanalyzer/iohinspector |
| 3. Real-world | NASA-7 Cp fit (fixed species set, T-grid, objective); later a small Cantera case | Show it matters in combustion | Same logger, Parquet traces |

The scaled-BBOB tier is the researchers' own construction, not an official suite. Its advantage is that COCO targets are defined on f − f_opt, so the rescaling leaves target definitions unchanged while the optimizer faces a badly scaled space.

**Reviewers will also expect optimizer generality.** The minimum credible set is CMA-ES via pycma 4.5.0, with IPOP/BIPOP restarts ([pycma](https://github.com/CMA-ES/pycma)), DE, Nelder–Mead or Powell, a model-based local DFO method with multistart, NGOpt, and **uniform random search under each encoding**. The random-search control isolates the prior effect (how much probability mass SLUD puts per decade) from optimizer dynamics. CMA-ES matters for a specific reason: it adapts a full covariance and can partly learn bad scaling that PSO cannot. If SLUD helps PSO but not CMA-ES, the paper should show that contrast rather than let a reviewer discover it. *COBYQA's availability in SciPy is from memory, not verified.*

Finally, there is a real gap the paper can fill. **No standardized combustion-mechanism optimization benchmark suite exists in Python.** The closest benchmark-grade analog is from systems biology. Villaverde et al. (2019) tested six kinetic models with 36–383 parameters, and every method ran "in both linear and logarithmic parameter space" ([Freiburg benchmarking wiki](https://benchmarking.uni-freiburg.de/index.php/Benchmarking_optimization_methods_for_parameter_estimation_in_large_kinetic_models)). On the combustion side, Ashraf & Pfaendtner compared 10 stochastic optimizers on propane-mechanism rate parameters ([NSF PAR](https://par.nsf.gov/servlets/purl/10276863)). A documented, reproducible NASA-Cp fitting benchmark would itself be a modest contribution. *Villaverde's quantitative log-versus-linear results were not read, so do not cite numbers from it yet.*

## Capped runs demand ERT, ECDFs and effect sizes, not mean generations

The current headline statistic, generations-to-target with capped runs either dropped or recorded at the cap, is biased in a predictable direction. Dropping failures makes the method that fails more often look faster. **COCO defines the expected runtime (aRT/ERT) as all evaluations from all runs, failures included, divided by the number of successful runs** ([Hansen et al., arXiv:1605.03560](https://arxiv.org/pdf/1605.03560); [arXiv:1806.04563](https://arxiv.org/pdf/1806.04563)). A failed run's runtime "remains undefined" but is bounded from below by its evaluations ([COCO platform paper, arXiv:1603.08785](https://arxiv.org/pdf/1603.08785)). The journal treatment of anytime assessment through runtime ECDFs is Hansen et al. in IEEE TEVC 26(6), 2022 ([HAL](https://hal.ip-paris.fr/hal-03814997v2)). The fix is cheap: log best-so-far f per evaluation, and record the first hitting evaluation for a ladder of targets. cocopp's settings use 51 log-spaced targets from 1e2 to 1e-8 ([cocopp](https://pypi.org/project/cocopp/)). Runtime ECDFs, fixed-budget plots and ERT then all come from one stored trace.

**Count objective evaluations, not generations.** COCO, IOHprofiler and the Moré–Wild data profiles all use evaluations as the cost unit. Moré & Wild normalize by n+1, the cost of one simplex gradient ([Argonne P1471](https://www.mcs.anl.gov/uploads/cels/papers/P1471.pdf)). That normalization matters here because LUD and LIN carry twice as many genes as SLUD. Report evaluations and evaluations per dimension, take counts from `res.algorithm.evaluator.n_eval`, and state the dimension of every encoding. pymoo checks the f-target only once per generation, so first-hitting times are known only to within 100 evaluations unless evaluation is logged per call.

For hypothesis tests, current practice in EC journals has converged on one pattern. Each comparison gets a **two-sample rank test per function**, failed runs ranked worst (+∞ or PAR-10), a **Vargha–Delaney Â12 or Cliff's δ effect size**, and **Holm correction** across every reported test. Across functions, pairwise **Wilcoxon signed-rank** tests are preferred over Friedman+Nemenyi, because mean-rank post-hoc results "depend on which other algorithms were in the experiment" ([Benavoli et al., JMLR 2016](https://www.jmlr.org/papers/v17/benavoli16a.html)). The Bayesian alternative is the Bayesian signed-rank test with a region of practical equivalence (ROPE) ([Benavoli et al., JMLR 2017](https://jmlr.org/papers/v18/16-305.html)), applied to black-box benchmarking by Calvo et al. at GECCO 2019 ([HAL](https://hal.sorbonne-universite.fr/hal-02179609)). With only three encodings and a handful of functions, a Friedman critical-difference analysis has little power. Per-function effect sizes plus an across-function signed-rank or Bayesian test say more. With N = 200, trivial differences become "significant", which is why effect sizes should lead and p-values follow. Use the Â12 thresholds 0.56/0.64/0.71 (small/medium/large) and say which convention you use ([metricgate A12](https://metricgate.com/docs/vargha-delaney-a-effect-size/)).

One subtlety concerns pairing. pymoo 0.6.2 creates `np.random.default_rng(seed)` per algorithm instance, so SLUD, LUD and LIN runs with the same seed share an initial random stream. The streams decouple almost immediately, though, because the encodings and even the dimensions differ. **The defensible within-function test is therefore unpaired Mann–Whitney U** (or Brunner–Munzel if spreads differ), not signed-rank. That conclusion comes from the statistics notes' reading of pymoo's `core/algorithm.py`.

Every item of the analysis has a verified Python call in current releases (scipy 1.18.1, statsmodels 0.15.0, scikit-posthocs 0.17.1, baycomp 1.0.3, iohinspector 0.0.8 were installed and exercised on synthetic data).

| Analysis item | Python call |
|---|---|
| ERT with failures, success rate | `cocopp.toolsstats.sp`, or `iohinspector.metrics.aggregate_running_time`, or a 3-line NumPy function |
| Wilson CI on success rate | `scipy.stats.binomtest(k, n).proportion_ci(method='wilson')` |
| Censored runtime ECDF / median | `scipy.stats.ecdf(scipy.stats.CensoredData(uncensored=..., right=...))`; `scipy.stats.logrank` |
| Bootstrap CI on ERT | `scipy.stats.bootstrap((evals, success), ert, paired=True, vectorized=False)` |
| Per-function test + effect size | `scipy.stats.mannwhitneyu`; Â12 = `U/(n_a·n_b)` |
| Multiplicity | `statsmodels.stats.multitest.multipletests(p, method='holm')` |
| Across-function display | `scikit_posthocs.critical_difference_diagram` with a Wilcoxon-Holm matrix, or `autorank` |
| Bayesian complement | `baycomp.two_on_multiple(x, y, rope=...)` or `baycomp.HierarchicalTest` |
| ECDF / fixed-budget plots | `iohinspector` (0.0.x, API still changing) or IOHanalyzer, or `cocopp` for COCO data |

Sources: [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html), [statsmodels multipletests](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html), [scikit-posthocs](https://pypi.org/project/scikit-posthocs/), [autorank](https://pypi.org/project/autorank/), [baycomp](https://pypi.org/project/baycomp/), [iohinspector](https://pypi.org/project/iohinspector/).

Reproducibility is now measured, not just encouraged. An audit of ten years of GECCO ECOM-track papers, nominated for best paper at GECCO 2026, found an **average checklist completeness of 0.62, with only 36.9% of papers providing any material beyond the manuscript** ([arXiv:2602.07059](https://arxiv.org/html/2602.07059v1)). López-Ibáñez, Branke & Paquete set out the EC reproducibility framework that refines the ACM badges TELO uses ([ACM TELO 2021](https://dl.acm.org/doi/10.1145/3466624)). In practice the bar is five things:

- a **published root `SeedSequence`** whose per-run seeds come from `spawn_key=(function, encoding, run)`, so they do not depend on worker scheduling ([NumPy parallel RNG](https://numpy.org/doc/stable/reference/random/parallel.html));
- a committed lockfile;
- **raw per-run traces in Parquet** (or IOH format) with a Zenodo DOI;
- the exact N per cell matching the archived code;
- one command that regenerates every table and figure.

Hydra and MLflow are overkill for roughly ten experimental cells. A TOML config, deterministic seeds and the git hash in each output file's metadata suffice.

## The mathematics is old; the encoding claim is new

Signed-log transforms have a long history outside optimization:

- John & Draper's (1980) log-modulus sign(y)·log(|y|+1);
- Burbidge, Magee & Robb's (1988) inverse hyperbolic sine;
- Yeo–Johnson (2000);
- Webber's bi-symmetric log transform for wide-range data ([Kent repository](https://kar.kent.ac.uk/32810));
- matplotlib's symlog, which needs a linear band (−linthresh, linthresh) because "values close to zero tend toward infinity" ([matplotlib SymLogNorm](https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.colors.SymLogNorm.html)).

The closest prior art is a sampling prior from high-energy physics. **GAMBIT's ScannerBit `double_log_flat_join` prior "patches together sections uniform in log(−x), uniform in x, and uniform in log x"**. It was used for the either-sign trilinear coupling A₀ ([arXiv:1705.07959](https://arxiv.org/abs/1705.07959)). SLUD is that prior with the flat segment collapsed to zero width and (−lb, lb) removed. *The ScannerBit description and the statistics papers were read through search summaries and secondary pages; arXiv, HAL and the journals were not directly reachable.*

On the software side, the evidence is strong because it comes from source code. In ten libraries, any request for log scale with non-positive bounds is rejected, or the output is positive by construction:

| Library (version) | Behaviour with log scale and lb ≤ 0 |
|---|---|
| Optuna 5.0.0 | `ValueError("low > 0 must hold for log=True")` |
| Nevergrad 1.0.12 | `ng.p.Log` lower must be > 0; Exponent layer rejects base ≤ 0 |
| Ax 1.3.1 | "cannot take log when min <= 0" |
| BoTorch 0.18.1 | No signed input transform; `Bilog` exists only for outcomes |
| Hyperopt 0.3.0 | `loguniform` returns `exp(draw)`, always positive |
| Ray Tune 2.59.0 | "LogUniform requires a lower bound greater than 0" |
| scikit-optimize 0.10.2 | Raises when `low*high <= 0` |
| ConfigSpace 1.2.2 (SMAC) | "Negative lower bound … for log-scale is not possible" |
| scipy 1.18.1 | `loguniform` requires b > a > 0 |
| pycma 4.5.0 | Only per-coordinate scaling; custom genotype maps deprecated |

Sources: [Optuna distributions](https://optuna.readthedocs.io/en/stable/_modules/optuna/distributions.html), [nevergrad](https://github.com/facebookresearch/nevergrad), [Ax](https://github.com/facebook/Ax), [BoTorch transforms](https://botorch.readthedocs.io/en/stable/_modules/botorch/models/transforms/input.html), [hyperopt](https://github.com/hyperopt/hyperopt), [Ray](https://github.com/ray-project/ray), [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize), [ConfigSpace](https://github.com/automl/ConfigSpace), [scipy](https://github.com/scipy/scipy), [pycma](https://github.com/CMA-ES/pycma).

Ax and BoTorch already apply signed-log transforms to outputs, citing Eriksson & Poloczek's SCBO ([PMLR v130](https://proceedings.mlr.press/v130/eriksson21a/eriksson21a.pdf)). In Bayesian optimization, input warping (Snoek et al. 2014) learns Beta-CDF warpings on the unit cube and calls a fixed log transform "a crude approximation" ([arXiv:1402.0929](https://arxiv.org/pdf/1402.0929)). Neither addresses sign. **pymoo and DEAP were not inspected.** Because SLUD runs on pymoo, check pymoo before writing "no library supports this".

The EC tradition has carried the sign separately for decades. IEEE-754-style chromosomes use one sign bit, 11 exponent bits and 52 mantissa bits ([CiteSeerX](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1052.2503&rep=rep1&type=pdf)), and Deb-style encodings use separate mantissa and exponent genes ([IJEAT](https://www.ijeat.org/wp-content/uploads/papers/v1i6/F0670081612.pdf)). Both are structurally the LUD encoding. That gives the paper a clean framing: **LUD and LIN belong to the sign-gene family, and SLUD folds the sign bit into the exponent gene.**

Combustion practice explains why nobody needed SLUD before. The Turányi group optimizes transformed Arrhenius parameters (ln A, n, E/R) ([BME abstract](https://old.bme.hu/node/5314)), and MUM-PCE/FFCM normalizes x = ln(A/A₀)/ln f around a nominal mechanism ([FFCM-1 method](https://web.stanford.edu/group/haiwanglab/FFCM1/pages/method.html)). These are bounded perturbations of positive quantities. **Virtual chemistry is the exception.** Cailler et al. (Combust. Flame 2020) optimize both rate parameters and thermodynamic properties of fictitious species with a genetic algorithm ([HAL hal-02494438](https://hal-centralesupelec.archives-ouvertes.fr/hal-02494438)). There is no nominal value to perturb, so coefficients can carry either sign across many decades. That is the motivating regime, and the COBEM draft already says so (local `COB-2025-2080.pdf`). *Cailler's actual bounding and scaling of parameters could not be read, so do not claim the original method used linear bounds.* Expect a combustion reviewer to suggest optimizing physical anchors instead, such as Cp at a few temperatures, ΔHf and S°, then refitting coefficients. That makes variables single-signed and well scaled, and Préteseille et al.'s 2026 "standardised thermodynamics" extension may do so ([virtual-chemistry publications](https://virtual-chemistry.em2c.centralesupelec.fr/publications/)). The paper should acknowledge this and argue for SLUD's generality and its compatibility with coefficient-level codes.

The novelty statement that survives review is therefore:

> We found no mainstream Python optimization library that supports a signed log-uniform decision variable (cited validation lines). We propose SLUD, a single-gene, monotone, two-sided log encoding for population-based optimizers that halves the dimension relative to sign-plus-magnitude encodings. We evaluate it against those encodings on standard, badly scaled and combustion-derived problems.

Claiming that a symmetric log distribution is new, or that log-scaling kinetic parameters is new, would be refuted by the prior art above.

The same literature predicts the reviewer objections, and each becomes an ablation:

- **Zero is unreachable and x jumps by 2·lb at the midpoint.** The draft already concedes this (local `COB-2025-2080.pdf`). The answer is to choose lb below the physically relevant scale and to test a symlog-style "SLUD with linear core" variant.
- **Genotype topology.** +ub and −ub sit at opposite ends of z, so with PSO velocity clamping a large-magnitude sign change takes several iterations, whereas a sign gene flips in one move. Test optima at +ub/2 against −ub/2.
- **The dimension halving is both the selling point and a confound.** Control evaluation budgets, report evaluations per dimension, and include a plain n-gene LIN on [−ub, ub] plus LUD with a categorical sign.
- **Sensitivity to lb, ub and asymmetric ranges** (separate lb⁻, ub⁻, lb⁺, ub⁺).
- **Gradient and GP-surrogate optimizers** will see the jump as non-stationarity.
- **LHS:** an n-point LHS in z guarantees balanced sign coverage per variable in n dimensions. LUD stratifies in 2n dimensions and fills space worse for the same sample size. This is analytical, not yet cited, and is a testable advantage claim.

A small implementation note from reading the code: the current transform uses `np.sign(dist)`, which returns 0 at the exact midpoint and so maps it to x = 0. `np.copysign` removes that edge case (local `SignedUniLog.py`).

## Vectorize first, parallelize runs, and keep processes for Cantera

For the cheap analytic tier, the bottleneck is per-individual Python work, not core count. **A local measurement found a SLUD-like transform plus Rosenbrock plus the 5-coefficient Cp residual over 100 temperatures cost about 7.0 ms per generation as per-individual loops, against about 64 µs vectorized over the 100 × n population: a 110× speedup, with results equal to rtol 1e-9** (Python 3.13, NumPy 2.5.3, 4 vCPUs; the transform was illustrative, not the repo's exact code). Scaled to the repo's design of 4 functions × 3 encodings × 200 runs × 500 generations, that is roughly 2.3 hours of single-core evaluation time element-wise against roughly 1.3 minutes vectorized. pymoo's own per-generation overhead was not measured and probably dominates after vectorization. That estimate is derived, not measured end to end.

At that granularity, **the current Ray design works against itself**. Each generation spawns ten remote tasks of ten individuals, about 0.7 ms of work each. Ray's own documentation lists too-fine-grained tasks as an anti-pattern: in its example, 10,000 one-item tasks took about 1.6 s against 0.17 s for a serial loop ([Ray anti-pattern](https://docs.ray.io/en/latest/ray-core/patterns/too-fine-grained-tasks.html)). Ray on Windows is still Beta: multi-node use is untested and file I/O is slower ([Ray installation](https://docs.ray.io/en/latest/ray-overview/installation.html)). The simpler, faster structure has three parts:

- a vectorized `pymoo.core.problem.Problem` whose `_evaluate` calls `objective(slud(X))` once per generation;
- each (function, encoding, seed) run as one serial task;
- `concurrent.futures.ProcessPoolExecutor` or `joblib.Parallel` over runs, with `OMP_NUM_THREADS=1`.

That gives near-linear scaling on top of the vectorization gain and works the same on Windows.

The faster options are real but not needed yet.

- **Numba** is mature: 0.63 added Python 3.14 ([notes](https://numba.readthedocs.io/en/stable/release/0.63.0-notes.html)) and 0.67 shipped in August 2026 ([notes](https://numba.readthedocs.io/en/stable/release/0.67.0-notes.html)). It is the natural next step if a kernel resists vectorization.
- **JAX-based evosax** can `vmap` hundreds of seeds into one compiled program ([arXiv:2212.04180](https://arxiv.org/abs/2212.04180)). **EvoX** reports GPU speedups of 10× or more only "as the scale increases", and CPU wins at small populations ([arXiv:2301.12457](https://arxiv.org/pdf/2301.12457)). JAX defaults to float32, consumer GPUs run FP64 at 1/32 to 1/64 of FP32 throughput ([heterodyne ADR-006](https://heterodyne.readthedocs.io/en/latest/developer/adrs/adr_006_no_consumer_gpu.html)), and Cantera cannot be traced by JAX. A JAX rewrite would duplicate the optimizer code for little gain until runs reach about 10⁵.
- **Python 3.14** makes free-threading officially supported but optional ([PEP 779](https://peps.python.org/pep-0779/)). Its JIT is roughly break-even ([DevClass](https://devclass.com/2025/07/09/despite-30-months-work-core-developer-says-pythons-jit-compiler-is-often-slower-than-the-interpreter/)), with gains of about 4–12% expected only in 3.15 ([PEP 836](https://peps.python.org/pep-0836/)). Cantera 3.2.0's free-threaded wheel is classed "1 - Unstable" ([PyPI](https://pypi.org/project/cantera/)). Neither feature changes the advice.
- **Compiled kernels** (Cython, pybind11/nanobind, f2py, which now must use Meson on Python ≥ 3.12 ([NumPy f2py](https://numpy.org/doc/2.4/f2py/buildtools/distutils-to-meson.html))) buy little for analytic functions. They earn their place only where code already exists in C++ or Fortran.

The expensive Cantera tier inverts the trade-offs. When one evaluation takes seconds to minutes, dispatch overhead stops mattering, and portability, robustness and heterogeneous runtimes dominate. **Cantera objects cannot be pickled or shared between processes. The official 3.1/3.2 pattern is a pool whose initializer builds one Cantera object per worker in a module-level global** ([Cantera 3.2 example](https://cantera.org/3.2/examples/python/transport/multiprocessing_viscosity.html)). Applied to a customized flame:

- build the `Solution` and flame once per worker;
- in each task, set parameters, restore a reference solution, solve, and return only numbers;
- use one process per physical core, since Windows PyPI wheels use single-threaded LAPACK ([Cantera pip docs](https://cantera.org/3.1/install/pip.html));
- set a per-evaluation timeout;
- catch solver failures inside the worker, because a crashed worker breaks a `ProcessPoolExecutor`.

Synchronous generational PSO idles workers "until the longest evaluation has completed" ([arXiv:2308.04102](https://arxiv.org/html/2308.04102v3)). If flame solve times vary by more than 2–3×, an asynchronous ask/tell loop around pycma or Nevergrad recovers that time. It carries a known bias toward individuals that evaluate quickly ([arXiv:2303.15543](https://arxiv.org/pdf/2303.15543)), and the reported 2×–14× speedups come from neural-architecture search, not combustion ([Cognizant](https://www.cognizant.com/us/en/ai-lab/publications/asynchronous-evolution-neural-architecture-search)). On a cluster, Dask (which pymoo supports) or `mpi4py.futures` fits SLURM better than Ray.

## Simple, loud code and a pure-Python package fit this team

The team's constraints (readable code, few branches, physical checks that fail loudly) line up with what packaging best practice already recommends. The core should be **one pure function per encoding, vectorized, with no per-variable `if`**. Sign handling belongs in `np.copysign`, and variable kinds belong in per-variable arrays, not branches:

```python
def slud(t, lb, ub):
    """t in [-1, 1] -> x in [-ub, -lb] U [lb, ub], log-uniform in |x|; works on (pop, n) arrays."""
    return np.copysign(lb * (ub / lb) ** np.abs(t), t)

def check_bounds(lb, ub):
    if not (np.all(np.isfinite(lb)) and np.all(lb > 0) and np.all(ub > lb)):
        raise ValueError(f"SLUD needs 0 < lb < ub, finite; got lb={lb}, ub={ub}")
```

Objectives should take physical x only. The `len(x) == 4` branches in `funcs.py` disappear once each encoding owns its own decode step and the objective never sees genes.

Checks should split into two kinds. **Configuration and invariant errors raise immediately**: non-positive bounds, NaN in a decoded population, Cp(T) ≤ 0 on the reference grid, a mechanism file that does not match the expected species. **Expected per-candidate solver failures** (a flame that does not converge) return a documented penalty, increment a failure counter stored with the results, and raise once the failure rate crosses a set threshold. This split is a design recommendation, not a sourced standard. It keeps an optimizer running while making silent corruption impossible.

Tooling should stay minimal:

- **Environment and build.** Use **uv** with a committed `uv.lock`, a PEP 621 `pyproject.toml`, src layout, and **hatchling + hatch-vcs**. The scientific-python cookie template calls Hatch "probably" the best choice for pure-Python projects ([cookie README](https://raw.githubusercontent.com/scientific-python/cookie/main/README.md)). Git-tag versions map one-to-one onto Zenodo release DOIs. `uv_build` "currently only supports pure Python" and needs a static version ([uv build backend](https://raw.githubusercontent.com/astral-sh/uv/main/docs/concepts/build-backend.md)).
- **Dependencies.** Follow pymoo's pattern: the core depends only on NumPy, and pymoo, cma, Ray and Cantera are optional extras. pymoo itself puts Ray in its `parallelization` extra ([pymoo pyproject](https://raw.githubusercontent.com/anyoptimization/pymoo/main/pyproject.toml)).
- **Cantera.** It ships PyPI wheels for Python 3.10–3.14 ([Cantera pip docs](https://raw.githubusercontent.com/Cantera/cantera/main/doc/sphinx/install/pip.md)). The *customized* C++ solver build needs conda-forge compilers and libraries, so it belongs in a **pixi** environment (pre-1.0, 0.72.x in July 2026 ([newreleases](https://newreleases.io/project/github/prefix-dev/pixi/release/v0.72.2))). Keep that in a separate package or workspace. If the custom solver already builds through Cantera's Cython layer, keep that route. A new standalone binding fits scikit-build-core, which handles CMake, Cython, nanobind or f2py ([scikit-build-core](https://scikit-build-core.readthedocs.io/en/stable/guide/getting_started.html)).
- **Quality checks.** ruff (lint and format), pytest with hypothesis property tests, pre-commit, and a GitHub Actions matrix over Linux, Windows and macOS. Good properties for SLUD:
  - round-trip `inverse(slud(t)) == t`;
  - monotonicity in t;
  - |x| always within [lb, ub];
  - symmetry under sign flip;
  - a slow KS test that log|x| is uniform.
- **Notebooks and data.** marimo notebooks are plain `.py` files, git-diffable and testable ([marimo](https://github.com/marimo-team/marimo)). Raw traces go to Zenodo, not Git LFS.
- **Repository hygiene.** The repo currently commits LaTeX build products, `__pycache__`, editor files and a `bkup/` copy of `Stats/`. Move the COBEM LaTeX to its own repo or Overleaf.

**JOSS is not available yet.** The current rules require a repository that has been public for more than six months with sustained development. The software must also reflect substantial scientific effort, and the JOSS paper must not center on new research results ([JOSS submitting.md](https://raw.githubusercontent.com/openjournals/joss/main/docs/submitting.md)). Citability can start today: add `CITATION.cff`, enable the Zenodo–GitHub integration, and tag v0.1.0. pyOpenSci review is an optional route that fast-tracks JOSS. *Its current terms were not confirmed.*

## Recommended stack and phased plan

| Layer | Recommendation | Notes / pins |
|---|---|---|
| Python | 3.12 or 3.13, standard GIL build | SPEC 0 supports roughly the last 3 years of Python ([SPEC 0](https://raw.githubusercontent.com/scientific-python/specs/main/spec-0000/index.md)); scipy 1.18 needs ≥3.12 ([PyPI](https://pypi.org/project/scipy/)) |
| Env / lock | uv + `uv.lock`; pixi only for the custom Cantera C++ build | PEP 723 + `exclude-newer` for one-off scripts |
| Build | hatchling + hatch-vcs, pure Python | scikit-build-core only for a separate compiled package |
| Core `slud` | NumPy-only `slud`, `lud`, `lin` encode/decode + bounds checks | Adapters: pymoo `Problem`, pycma, scipy, nevergrad |
| Optimizers | pymoo PSO/DE 0.6.2; pycma 4.5.0 (IPOP/BIPOP); scipy DE + Nelder–Mead/Powell multistart; nevergrad NGOpt 1.0.12; random search | Equal budgets, report populations |
| Problems | cocoex `bbob` (control); MGH Powell/Brown validated against PyCUTEst; scaled, sign-randomized BBOB; NASA-7 Cp; later Cantera | Mixed-sign optima required |
| Logging | Per-evaluation best-so-far; `ioh.wrap_problem` + Analyzer logger, or a long Parquet table | Seed, encoding, git hash in metadata |
| Analysis | ERT + Wilson CI; censored ECDF; Mann–Whitney + Â12 + Holm; Wilcoxon across functions; baycomp; iohinspector/IOHanalyzer, cocopp | Effect sizes first |
| Parallel (cheap) | Vectorized `Problem` + ProcessPoolExecutor/joblib over runs | Drop per-generation Ray |
| Parallel (Cantera) | Process pool with per-worker Cantera initializer, timeout, failure counter; Dask/mpi4py on HPC; async ask/tell if runtimes vary >2–3× | No threads, no free-threaded build |
| Quality | ruff, pytest + hypothesis, pre-commit, GitHub Actions on 3 OSes | Light mypy |
| Publication | CITATION.cff, Zenodo DOI per release, data DOI, JOSS after ≥6 months public | Paper LaTeX in a separate repo |

The phases below overlap. Durations are estimates, not sourced.

| Phase | Weeks | Deliverables | Exit criterion |
|---|---|---|---|
| 0. Clean and reproduce | 1–2 | `.gitignore`; LaTeX, `bkup/` and `__pycache__` out; one config file; SeedSequence seeds; N = 200 exactly; success flag and n_eval per run | COBEM tables regenerate from one command |
| 1. Package core v0.1 | 2–6 | `src/slud` encode/decode, bounds checks, pymoo adapter, hypothesis tests, CI, CITATION.cff, Zenodo DOI | `uv sync && uv run pytest` green on Linux, Windows, macOS |
| 2. Harness and statistics | 4–10 | Vectorized problems, run-level pool, per-evaluation traces in Parquet/IOH format, analysis script producing ERT, ECDF, Â12/Holm, baycomp | Figures regenerate from archived traces |
| 3. Broaden experiments | 8–16 | Three tiers; CMA-ES, DE, local DFO, NGOpt, random search; plain n-gene LIN; categorical-sign LUD; SLUD with linear core; lb/ub sweeps; ±ub/2 optima; dims 2–20 | Pre-registered config committed before final runs |
| 4. Journal paper | 14–24 | Related work framed as symlog/ScannerBit prior → decision-variable encoding; three-tier results; reproducibility statement with DOIs | Submission to ACM TELO, Evolutionary Computation or an engineering-optimization journal (check each venue's artifact policy) |
| 5. Cantera tier | 16–36 | Small virtual-chemistry or H₂/O₂ fitting case on the process-pool runner with physical checks; possibly async ask/tell | Becomes the application paper or a revision addendum |
| 6. JOSS / pyOpenSci | ≥6 months after public v0.1 | Docs site, adapters, examples, CONTRIBUTING, AI-use disclosure | Meets the sustained-development rule |

## Claims not verified from primary sources

Documentation hosts (numbbo.github.io, iohprofiler.github.io, readthedocs, docs.astral.sh, cantera.org, arxiv.org, HAL, journal sites) were frequently blocked. Facts read from source code, PyPI JSON or raw GitHub files are treated as verified. The items below rest on search summaries, secondary pages or prior knowledge and should be checked before they appear in a manuscript.

| Claim | Status |
|---|---|
| GAMBIT ScannerBit `double_log_flat_join` details; Webber 2013, John & Draper 1980, Burbidge 1988, Yeo–Johnson | Search summaries and secondary pages only; read §3.1 of arXiv:1705.07959 |
| "No library supports signed log-uniform" | Verified for 10 libraries; **pymoo, DEAP, SMAC3, PlatEMO not inspected** |
| No prior EC/BO paper uses a single signed-log gene | Absence from about four searches; run a Scholar search before claiming |
| Cailler 2020 parametrization of virtual-chemistry variables | Full text not read |
| Villaverde 2019 log-versus-linear numbers | Not read |
| IOH `wrap_problem`/logger API names, `logger.watch` | Prior knowledge; check against ioh 0.3.22 |
| MGH optima values; f11–f14 conditioning constants; BBOB 2026 workshop | Prior knowledge / not checked |
| COBYQA in SciPy; pymoo parallelization class names | Prior knowledge |
| pymoo latest version | Conflict: PyPI JSON gives 0.6.2, search mirrors gave 0.6.1.6; PyPI taken as correct |
| Derrac 2011, LaTorre 2021, Bartz-Beielstein 2020 step-by-step prescriptions; Gould & Scott profile caveat; Moré–Wild convergence test | Full texts not read |
| ACM "reproduced" vs "replicated" badge naming; TELO and other journals' 2026 code/data mandates | Ambiguous or unchecked |
| JOSS 2026 checklist details; pyOpenSci fast-track terms | Partly from search summaries |
| Cantera multi-thread safety; whether PyPI wheels allow linking a custom C++ solver | Undocumented in sources found; assume processes and conda-forge/source builds |
| EvoX's move to a PyTorch backend; nanobind overhead claims; Numba 0.68; MkDocs/Zensical status | Secondary or prior knowledge |
| 110× vectorization speedup | Measured locally on an illustrative transform, not on the repo's exact code |

## Conclusion

The strongest version of SLUD is a claim about *representation*, not about distributions. It says that folding the sign bit into the exponent gene gives population-based optimizers a smaller, order-preserving search space for coefficients that are mixed-sign and span many decades, a regime that virtual chemistry creates and that standard kinetics workflows never faced. That claim is falsifiable and has not been made in the BBO literature found here. It also exposes SLUD's real risks: large-magnitude sign changes take several iterations, the near-zero band is unreachable, and the dimension advantage confounds budget comparisons. Each risk becomes an ablation rather than a reviewer's surprise. The current results cannot yet carry the claim, because the analytic problems have all-positive optima and the baselines double the dimension.

The engineering lesson is that simplicity and speed point the same way. Removing per-generation Ray, vectorizing once per population and parallelizing over independent runs makes the cheap tier both faster and easier for a combustion team to read. The same run-level, process-per-worker structure, with loud invariant checks and counted solver failures, carries over unchanged to Cantera flames. Built this way, the package and the journal paper reinforce each other: the paper supplies the evidence, the package makes it reproducible, and six months of visible development later the package qualifies for JOSS.
