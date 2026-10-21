# Python benchmarking frameworks and test suites for black-box / derivative-free / evolutionary optimization (state as of Oct 2026), and which fit a SLUD parametrization study

Method note: package versions and release dates were pulled live from the PyPI JSON API on 2026-10-21 (https://pypi.org/pypi/<name>/json). COCO facts were verified against the C/Python source in `numbbo/coco-experiment` (raw.githubusercontent.com). The `*.github.io` documentation sites (numbbo.github.io, coco-platform.org, iohprofiler.github.io, facebookresearch.github.io) and PMC could not be reached from this sandbox (DNS/proxy 403), so I read the repo READMEs and doc sources on GitHub instead. Items marked **[prior knowledge, not re-fetched]** are well-known bibliographic facts from my training data. They come with canonical DOI or repo URLs, but I did not re-verify them in this session.

### Version and maintenance snapshot (PyPI, fetched 2026-10-21)

| Package (import) | Latest | Released | Status note |
|---|---|---|---|
| coco-experiment (`cocoex`) | 2.8.3 | 2026-10-21 | very active; PyPI name differs from import name — [PyPI](https://pypi.org/project/coco-experiment/), [repo](https://github.com/numbbo/coco-experiment) |
| cocopp | 2.9.1 | 2026-10-15 | active — [PyPI](https://pypi.org/project/cocopp/) |
| ioh (IOHexperimenter) | 0.3.22 | 2025-09-26 | maintained — [PyPI](https://pypi.org/project/ioh/) |
| iohinspector | 0.0.8 | 2026-08-04 | new Python-side analysis package from IOHprofiler, early (0.0.x) — [repo](https://github.com/IOHprofiler/iohinspector) |
| nevergrad | 1.0.12 | 2025-04-23 | no release for ~18 months — [PyPI](https://pypi.org/project/nevergrad/) |
| pymoo | 0.6.2 | 2026-06-28 | active — [PyPI](https://pypi.org/project/pymoo/) |
| cma (pycma) | 4.5.0 | 2026-09-13 | active — [repo](https://github.com/CMA-ES/pycma) |
| modcma | 1.2.0 | 2025-12-03 | modular CMA-ES (IOHprofiler group) — [PyPI](https://pypi.org/project/modcma/) |
| opfunu | 1.0.4 | 2024-06-04 | stale since mid-2024 — [repo](https://github.com/thieu1995/opfunu) |
| mealpy | 3.0.3 | 2025-08-16 | metaheuristic zoo (same author as opfunu) — [repo](https://github.com/thieu1995/mealpy) |
| pycutest | 1.8.2 | 2026-04-28 | active — [repo](https://github.com/jfowkes/pycutest/) |
| scipy | 1.18.1 | 2026-08-21 | — [PyPI](https://pypi.org/project/scipy/) |
| evosax | 0.3.2 | 2026-10-11 | active (JAX) — [repo](https://github.com/RobertTLange/evosax) |
| evox | 1.4.0 | 2026-09-09 | active — [repo](https://github.com/EMI-Group/evox) |
| evotorch | 0.6.1 | 2025-05-14 | slower — [site](https://evotorch.ai) |
| pygmo | 2.20.0 | 2026-10-06 | active — [PyPI](https://pypi.org/project/pygmo/) |
| pyswarms | 1.3.0 | 2021-01-03 | unmaintained — [PyPI](https://pypi.org/project/pyswarms/) |
| optuna | 5.0.0 | 2026-09-07 | active — [PyPI](https://pypi.org/project/optuna/) |
| botorch | 0.18.1 | 2026-06-08 | active — [site](https://botorch.org) |
| smac | 2.4.1 | 2026-09-10 | active — [PyPI](https://pypi.org/project/smac/) |
| yahpo-gym | 1.0.2 | 2024-08-18 | quiet — [repo](https://github.com/slds-lmu/yahpo_gym) |
| hpobench | — | not on PyPI | install from GitHub only — [PyPI query returned 404](https://pypi.org/pypi/hpobench/json) |
| scikit-optimize | 0.10.2 | 2024-06-04 | quiet — [PyPI](https://pypi.org/project/scikit-optimize/) |
| cantera | 3.2.0 | 2025-11-18 | — [site](https://cantera.org) |
| petab / pypesto | 0.9.0 / 0.7.0 | 2026-09 | active (systems-biology parameter estimation) — [petab PyPI](https://pypi.org/project/petab/), [pypesto PyPI](https://pypi.org/project/pypesto/) |

---

## Q1. COCO / BBOB: API, wrapping an optimizer, post-processing, use in papers, suitability for a transform-around-optimizer study, interaction with [-5,5]^n, and ill-conditioned functions

### Takeaway
COCO (`cocoex` + `cocopp`) is the de facto reference that GECCO/TELO/EC reviewers expect for continuous black-box benchmarking. Because a COCO problem is a plain Python callable, a SLUD/LUD/LIN transform can be wrapped around any optimizer as `problem(T(u))`. BBOB optima, however, sit in [-4,4]^n inside a [-5,5]^n domain, and all variables share a single O(1) scale. BBOB therefore cannot exercise SLUD's main claim (a signed optimum whose magnitude spans many decades). Use it as a "no-harm"/control testbed, especially the 10^6-conditioned f2/f10/f11/f12 (ill-conditioning in the Hessian, which is different from bad *variable scaling*). Pair it with a scaled or badly-scaled suite.

### Cited Findings
- **Packaging:** the experimentation module is published on PyPI as `coco-experiment` (2.8.3, 2026-10-21), but the import name is `cocoex`. The README says the "module name (`coco-experiment`) is not the import name", and recommends `pip install -U coco-experiment` — [coco-experiment README](https://github.com/numbbo/coco-experiment/tree/main/build/python), [PyPI](https://pypi.org/project/coco-experiment/).
- **Canonical experiment loop** (official `example_experiment_simple.py`, author N. Hansen): `suite = cocoex.Suite("bbob", "", "")`; `observer = cocoex.Observer("bbob", "result_folder: ...")`; `repeater = cocoex.ExperimentRepeater(budget_multiplier)`. Then `for problem in suite: problem.observe_with(observer); problem(problem.dimension*[0]); xopt = fmin(problem, repeater.initial_solution_proposal(problem)); problem(xopt); repeater.track(problem)`. Post-processing is `cocopp.main(observer.result_folder + ' bfgs!')`, which compares against the archived BFGS data. Swapping in CMA-ES is shown as `cma.fmin2(fun, x0, 2, ...)` — [example_experiment_simple.py](https://raw.githubusercontent.com/numbbo/coco-experiment/main/build/python/example/example_experiment_simple.py).
- **What wrapping means in practice:** the problem object is called directly by the solver (`fmin(problem, x0)`). Any callable composition such as `lambda u: problem(T(u))` therefore gets logged by the observer in the *original* x-space, which is exactly what a parametrization study needs — [example_experiment_simple.py](https://raw.githubusercontent.com/numbbo/coco-experiment/main/build/python/example/example_experiment_simple.py) (the composition itself is my inference from the callable API).
- **Suites in the current C core:** `bbob` (24 functions), `bbob-boxed` (alias SBOX-COST), `bbob-biobj` (55), `bbob-biobj-ext` (92), `bbob-largescale` (24), `bbob-constrained` (54 linearly constrained), `bbob-mixint` (24 functions, mixed-integer, dims 2,3,5,10,20,40), `bbob-biobj-mixint` (92), `bbob-noisy`, and `toy` — [coco_suite.c](https://raw.githubusercontent.com/numbbo/coco-experiment/main/src/coco_suite.c).
- **`bbob` dimensions** are {2, 3, 5, 10, 20, 40} — [suite_bbob.c](https://raw.githubusercontent.com/numbbo/coco-experiment/main/src/suite_bbob.c).
- **Location of the optimum:** `bbob2009_compute_xopt` sets `xopt[i] = 8*floor(1e4*u)/1e4 - 4`, so each optimum coordinate is uniform in [-4,4], on a 1e-4 grid, with 0 replaced by -1e-5. A second routine places the optimum "in the larger space of [-5, 5]^DIM" — [suite_bbob_legacy_code.c](https://raw.githubusercontent.com/numbbo/coco-experiment/main/src/suite_bbob_legacy_code.c).
- **`bbob-boxed` / SBOX-COST:** the same 24 BBOB functions with a strict box (it includes `transform_vars_enforce_box.c`). A FIXME in the source notes that f9, f19, f20 and f24 "have optima that are always close to the origin" — [suite_sbox_cost.c](https://raw.githubusercontent.com/numbbo/coco-experiment/main/src/suite_sbox_cost.c). The suite came out of a GECCO 2023 workshop on strict box-constraint optimization — [SBOX-COST CFP](https://listserv.zib.de/pipermail/opt-net/2023-March/002181.html).
- **Ill-conditioned functions:** the ellipsoid uses `pow(conditioning, i/(n-1)) * x_i^2` with conditioning `1.0e6` for the bbob variants (f2 separable and f10 rotated) and is typed "3-ill-conditioned" — [f_ellipsoid.c](https://raw.githubusercontent.com/numbbo/coco-experiment/main/src/f_ellipsoid.c). The rotated Rosenbrock (f9), ellipsoid (f10), discus (f11), bent cigar (f12), sharp ridge (f13) and different powers (f14) are all allocated in `suite_bbob.c` — [suite_bbob.c](https://raw.githubusercontent.com/numbbo/coco-experiment/main/src/suite_bbob.c). Secondary sources group f10–f14 as "unimodal with high conditioning". For f11, a single direction is "1000 times more sensitive" (i.e. a 10^6 conditioning of the quadratic) — [MetaBox COCO-BBOB doc](https://metaboxdoc.readthedocs.io/en/stable/guide/DS_BL/COCO-BBOB.html), [Revisiting BO in light of COCO, arXiv 2103.16649](https://arxiv.org/pdf/2103.16649).
- **Five BBOB function groups:** separable; low/moderate conditioning; high conditioning and unimodal; multimodal with adequate global structure; multimodal with weak global structure. The BBOB workshop has run at GECCO most years since 2009, and the COCO data archive holds results from almost all workshop algorithms — [OPTION ontology paper, arXiv 2211.11332](https://arxiv.org/pdf/2211.11332), [BBOB 2017 CFP](https://dis.ijs.si/gecco-workshop-on-real-parameter-black-box-optimization-benchmarking-bbob-2017/).
- **Reference papers [prior knowledge, not re-fetched]:** Hansen, Auger, Ros, Mersmann, Tušar, Brockhoff, "COCO: a platform for comparing continuous optimizers in a black-box setting", *Optimization Methods & Software* 36(1):114–144, 2021 — [doi:10.1080/10556788.2020.1808977](https://doi.org/10.1080/10556788.2020.1808977), [arXiv:1603.08785](https://arxiv.org/abs/1603.08785). The function definitions are in Hansen, Finck, Ros, Auger, INRIA RR-6829 (2009) — [hal inria-00362633](https://hal.inria.fr/inria-00362633).
- **cocopp output [prior knowledge, not re-fetched]:** empirical cumulative distribution functions (ECDFs) of runtime to reach a set of targets (bootstrapped, aggregated over functions and dimensions); average runtime (aRT) tables with significance tests against the best 2009 algorithm or a chosen reference; runtime-scaling plots versus dimension; and pairwise scatter plots. Any archived algorithm can be compared by name (as in `'bfgs!'` above) — [cocopp PyPI](https://pypi.org/project/cocopp/).

### Inferences
- **SLUD on BBOB:** SLUD maps u ∈ [-1,1] (or u plus a sign) to ±lb·(ub/lb)^|u|. On BBOB the natural setup is T_SLUD with lb ≈ 1e-k and ub = 5, compared against LIN on [-5,5]. Reaching an optimum at, for example, x* = 2.37 then needs fine resolution in u near the top decade, while coordinates near 0 (|x| < lb) are unreachable unless lb is small. This measures the *cost* of SLUD when scale is already known — a valid and expected control experiment.
- **Badly scaled variants:** to make BBOB test SLUD's actual premise, define y_i = x_i · 10^{k_i} with k_i spread over, say, [-6, 6] and random signs, and evaluate f_bbob(x(y)). Each coordinate's optimum magnitude then spans decades. Because COCO targets are defined on f−f_opt, this per-coordinate rescaling is invisible to the logger: cocopp ECDFs and aRT remain directly comparable to archived BBOB data, while the optimizer sees the badly scaled y-space through SLUD/LUD/LIN. This is a cleanly reviewable "scaled-BBOB" design, but it is my construction and not an official suite.
- **Conditioning versus scaling:** f2 (separable ellipsoid, 10^6) is the BBOB function closest to "badly scaled variables", because its ill-conditioning is axis-aligned and so equivalent to a diagonal variable rescaling. f10 is the rotated counterpart. This makes f2 versus f10 a natural pair: a per-coordinate transform like SLUD might help on f2-like structure but cannot undo rotation.
- **bbob-boxed:** this suite is relevant because SLUD/LUD/LIN all impose hard boxes, so boundary handling differs. Optima near the origin (f9, f19, f20, f24) are pathological for a log-magnitude map with lb > 0.
- **Invariance:** COCO's methodology emphasizes invariance, and CMA-ES is invariant to order-preserving f-transformations and to rotations, while PSO is not. Reviewers will expect at least CMA-ES on BBOB alongside PSO.

### Gaps
- The coco-platform.org / numbbo.github.io pages (getting-started, bbob function definitions, BBOB-2026 workshop CFP) were unreachable here (proxy 403). I could not confirm whether a BBOB workshop ran at GECCO 2026; one web search found no 2026 edition page.
- I did not verify the exact conditioning constants of f11/f12/f13/f14 in source (only f2/f10 = 1e6 was checked in code). The definitions document is the authority.

---

## Q2. IOHprofiler (IOHexperimenter `ioh`, IOHanalyzer, IOHdata): API, loggers, COCO compatibility, ECDF/ERT

### Takeaway
IOHexperimenter (`ioh` 0.3.22) re-hosts the 24 BBOB functions, SBOX-COST and CEC2013/CEC2022 in a C++ core with a Python API and flexible loggers. Its output feeds IOHanalyzer, an R/Shiny web tool that also reads COCO data, for fixed-target (ERT/ECDF) and fixed-budget analyses. The main advantage over COCO for a SLUD study is that arbitrary user functions can be registered and logged (`wrap_problem`), so the MGH badly scaled problems and a NASA-Cp fit can be analyzed with the same ECDF/ERT machinery.

### Cited Findings
- **Problem sets:** PBO (25 pseudo-Boolean), BBOB (24 continuous, "integration of the well-known BBOB problem set"), SBOX-COST (COCO), StarDiscrepancy, W-model, submodular graph problems, CEC 2013 niching and CEC 2022 bound-constrained; a "flexible interface for adding new suites and problems"; an "advanced logging module"; "data format is compatible with IOHanalyzer" — [IOHexperimenter README](https://github.com/IOHprofiler/IOHexperimenter).
- **Citation:** de Nobel, Ye, Vermetten, Wang, Doerr, Bäck, "IOHexperimenter: Benchmarking Platform for Iterative Optimization Heuristics", *Evolutionary Computation* 32(3):205–210, 2024, doi:10.1162/evco_a_00342. The original IOHprofiler paper is arXiv:1810.05281 — [IOHexperimenter README](https://github.com/IOHprofiler/IOHexperimenter).
- **Releases:** `ioh` 0.3.22 (2025-09-26); `iohinspector` 0.0.8 (2026-08-04) is a newer Python analysis package from the same organisation — [PyPI ioh](https://pypi.org/project/ioh/), [iohinspector repo](https://github.com/IOHprofiler/iohinspector).
- **API outline [prior knowledge, not re-fetched; docs site unreachable]:** `ioh.get_problem(fid, instance, dimension, problem_class=ioh.ProblemClass.BBOB)`; `problem.bounds.lb/ub`, `problem.state.evaluations`, `problem.state.current_best`, `problem.optimum`; `logger = ioh.logger.Analyzer(root=..., folder_name=..., algorithm_name=...)`; `problem.attach_logger(logger)`; `problem.reset()` between runs; `ioh.wrap_problem(f, name, problem_class=ioh.ProblemClass.REAL, dimension, lb, ub)` for custom functions; triggers such as `ioh.logger.trigger.ON_IMPROVEMENT`. The docs are at [iohprofiler.github.io/IOHexperimenter/python](https://iohprofiler.github.io/IOHexperimenter/python.html).
- **IOHanalyzer paper [prior knowledge, not re-fetched]:** Wang, Vermetten, Ye, Doerr, Bäck, "IOHanalyzer: Detailed Performance Analyses for Iterative Optimization Heuristics", *ACM TELO* 2(1), 2022 — [doi:10.1145/3510426](https://doi.org/10.1145/3510426), [arXiv:2007.03953](https://arxiv.org/abs/2007.03953). It provides ERT, ECDF (fixed-target), fixed-budget distributions, parameter tracking, and Glicko-2/Deep-Statistical-Comparison rankings, and accepts COCO-format data.

### Inferences
- **IOH versus COCO for SLUD:** IOH is the better choice when the suite must include *non-BBOB* problems (MGH badly scaled, NASA-Cp). Logging stays in x-space and the analysis matches COCO's style. COCO's archive comparison is stronger for BBOB-only claims. A pragmatic setup is COCO (or ioh BBOB) for the control experiments plus IOH `wrap_problem` for the custom problems, with all data analyzed in IOHanalyzer.
- **Tracking:** IOH loggers can record extra tracked attributes (e.g., the SLUD-space position or the sign-flip count) per evaluation, which helps a mechanism analysis. This rests on my knowledge of `logger.watch`; it was not re-verified.

### Gaps
- The Python API docs were unreachable, so method names above are from prior knowledge. Verify them against `ioh` 0.3.22 before use.
- I did not confirm IOHdata's current status or hosting.

---

## Q3. Nevergrad: benchmarking, parametrization, optimizers, NGOpt

### Takeaway
Nevergrad has the richest *parametrization* layer (`ng.p.Log`, `Scalar`, `Array` with bounds and mutation scales), a very large optimizer registry including the NGOpt algorithm-selection "wizard", and a CLI benchmark runner with hundreds of named experiment plans, including `illcondi` and `far_optimum_es`. However, its last release was April 2025, and its built-in plots (regret curves and "fight" win-rate matrices) are less standard in journals than COCO/IOH ECDFs. It is best used as a *source of baselines* (NGOpt, CMA, DE, PSO, (1+1)) rather than as the primary benchmark harness.

### Cited Findings
- **Installation and API:** "`nevergrad` is a Python 3.8+ library", installed with `pip install nevergrad`. A minimal run is `ng.optimizers.NGOpt(parametrization=2, budget=100).minimize(f)`. Mixed spaces use `ng.p.Instrumentation(learning_rate=ng.p.Log(lower=0.001, upper=1.0), batch_size=ng.p.Scalar(lower=1, upper=12).set_integer_casting(), architecture=ng.p.Choice([...]))` — [nevergrad README](https://github.com/facebookresearch/nevergrad).
- **`Log(lower, upper)`:** "describes log distributed data between two bounds. Under the hood this uses a `Scalar` with appropriate specifications for bounds and mutations." `Array` supports `set_bounds`, `set_mutation` and `set_integer_casting` — [parametrization.rst](https://github.com/facebookresearch/nevergrad/blob/main/docs/parametrization.rst). Nevergrad has no signed-log parameter; `Log` requires positive bounds.
- **Benchmark runner:** run `python -m nevergrad.benchmark <experiment_name>` (with `--num_workers`, `--seed`, `--plot`), which writes a CSV with one row per experiment. `python -m nevergrad.benchmark.plotting <csv>` makes "experiment plots" (regret vs budget) and "fight plots" (win-rate matrix). Custom functions derive from `nevergrad.functions.ExperimentFunction`, and extra experiments load via `--imports` (not on Windows) — [benchmarking.rst](https://github.com/facebookresearch/nevergrad/blob/main/docs/benchmarking.rst).
- **Named experiment plans** include `yabbob`, `yahdbbob`, `yaboundedbbob`, `yaboxbbob`, `illcondi`, `illcondipara`, `multimodal`, `deceptive`, `lowbudget`, `parallel`, `noisy`, `far_optimum_es`, `realworld`, `photonics`, `rocket`, `powersystems`, `lsgo`, `pbo_suite` and dozens more — [experiments.py](https://github.com/facebookresearch/nevergrad/blob/main/nevergrad/benchmark/experiments.py).
- **`illcondi`:** "Testing optimizers on ill cond problems. Cigar, Ellipsoid. Both rotated and unrotated. Budget 100, 1000, 10000. Dimension 50." **`far_optimum_es`** runs ES/basic optimizers on `FarOptimumFunction.itercases()` with budgets of 100 to 10000 — [experiments.py](https://github.com/facebookresearch/nevergrad/blob/main/nevergrad/benchmark/experiments.py).
- **Latest release:** nevergrad 1.0.12 (2025-04-23) — [PyPI](https://pypi.org/project/nevergrad/).
- **NGOpt reference [prior knowledge, not re-fetched]:** Meunier, Rakotoarison, Wong, Rozière, Rapin, Teytaud, Moreau, Doerr, "Black-Box Optimization Revisited: Improving Algorithm Selection Wizards through Massive Benchmarking", *IEEE TEVC* 26(3), 2022 — [arXiv:2010.04542](https://arxiv.org/abs/2010.04542). The platform citation is Rapin & Teytaud, "Nevergrad – A gradient-free optimization platform", 2018 — [GitHub](https://github.com/facebookresearch/nevergrad).

### Inferences
- **Most relevant plans:** `far_optimum_es` (optimum far from the initial guess / at unusual scale) and `illcondi` are the Nevergrad experiment plans closest in spirit to SLUD. Citing them shows that scale or offset robustness is a recognized benchmark dimension.
- **LUD baseline:** `ng.p.Log` is effectively the community's standard "LUD" parametrization (positive only). A SLUD paper can position itself as the signed generalization, and Nevergrad's `Log` (plus a separate sign `Choice`) could serve as an off-the-shelf LUD baseline.
- **Maintenance:** with no release for ~18 months, pin the version and state it.

### Gaps
- I did not read the `FarOptimumFunction` source to report its exact offsets and scales.
- I could not confirm how many optimizers are in the registry as of 1.0.12 (I recall several hundred, but have no source).

---

## Q4. pymoo: test problems, benchmark tools, use in papers, version status

### Takeaway
pymoo (0.6.2, June 2026) is primarily a *multi-objective* algorithm library. Its single-objective test problems and benchmarking/analysis tools are thin compared with COCO/IOH, so reviewers will see pymoo as an *algorithm provider* (PSO, DE, GA, CMA-ES wrapper, Nelder–Mead, pattern search), not a benchmark platform. The current pymoo PSO + Ray code can stay as the algorithm layer, but performance data should be logged and analyzed via COCO or IOH.

### Cited Findings
- **Release:** pymoo 0.6.2, released 2026-06-28; the PyPI summary is "Multi-Objective Optimization in Python" — [PyPI](https://pypi.org/project/pymoo/).
- **Reference and contents [prior knowledge, not re-fetched]:** Blank & Deb, "pymoo: Multi-Objective Optimization in Python", *IEEE Access* 8:89497–89509, 2020 — [doi:10.1109/ACCESS.2020.2990567](https://doi.org/10.1109/ACCESS.2020.2990567). Problems are available via `pymoo.problems.get_problem` (single-objective: ackley, rastrigin, rosenbrock, griewank, zakharov, sphere, g1–g24 constrained; multi-objective: ZDT, DTLZ, WFG, MW, CTP, DASCMOP). Single-objective algorithms include GA, DE, BRKGA, Nelder-Mead, PatternSearch, CMA-ES (wrapping pycma), ES, SRES, ISRES, G3PCX and PSO — [pymoo repo](https://github.com/anyoptimization/pymoo).

### Inferences
- **Status:** pymoo is maintained, but its test problem set does not include the MGH badly scaled problems or a BBOB port with instances. Using pymoo-only problems in a journal benchmark would likely draw a "why not BBOB/COCO" comment.
- **Interop:** a pymoo `Problem` can wrap a cocoex/ioh problem (call it in `_evaluate`), so the existing PSO + Ray pipeline can feed COCO/IOH loggers. Ray parallelism must then handle per-process observers (one problem instance per worker), which is my inference.

### Gaps
- I did not fetch the pymoo docs to re-verify the exact problem and algorithm lists in 0.6.2.

---

## Q5. Other suites and libraries: CEC/opfunu, SciPy benchmarks, Moré–Garbow–Hillstrom, CUTEst/PyCUTEst, pycma, GPU ES libraries, HPO benchmarks, BoTorch

### Takeaway
The Moré–Garbow–Hillstrom (MGH) problems are directly available through CUTEst/PyCUTEst: Powell badly scaled is `POWELLBS` and Brown badly scaled is `BROWNBS`, both citing MGH 1981 as their source. This gives the SLUD study an authoritative, citable source for its two badly scaled problems. pycma is the expected CMA-ES baseline. CEC suites via opfunu are popular but less respected by BBOB-community reviewers, and opfunu is stale. GPU ES libraries (evosax, EvoX, EvoTorch) and HPO benchmarks (HPOBench, YAHPO, BoTorch functions) are peripheral to a scaling or parametrization study.

### Cited Findings
- **CUTEst sources:** the `POWELLBS.SIF` header reads "Powell badly scaled problem … Source: Problem 3 in J.J. Moré, B.S. Garbow and K.E. Hillstrom, 'Testing Unconstrained Optimization Software', ACM TOMS 7(1), pp. 17-41, 1981", noting "sum of n-1 sets of 2 groups" (so it is n-dimensional in CUTEst) — [POWELLBS.SIF](https://bitbucket.org/optrove/sif/raw/master/POWELLBS.SIF). `BROWNBS.SIF` reads "Brown badly scaled problem in 2 variables … Source: Problem 4 in J.J. Moré, B.S. Garbow and K.E. Hillstrom … 1981" — [BROWNBS.SIF](https://bitbucket.org/optrove/sif/raw/master/BROWNBS.SIF).
- **PyCUTEst:** version 1.8.2 (2026-04-28) — [repo](https://github.com/jfowkes/pycutest/). Reference [prior knowledge, not re-fetched]: Fowkes, Roberts, Bűrmen, "PyCUTEst: an open source Python package of optimization test problems", *JOSS* 7(78):4377, 2022 — [doi:10.21105/joss.04377](https://doi.org/10.21105/joss.04377). Note that it requires a CUTEst/SIFDecode Fortran build and works on Linux/macOS.
- **Python MGH implementations:** `pyopus.problems.mgh` provides the More-Garbow-Hillstrom functions with first derivatives, translated from MATLAB, and can be used standalone. PyOPUS's last PyPI release is 0.9 (2018-09-17). A Rust crate `mgh` and a Scilab `uncprb` toolbox (35 problems) also exist — [pyopus.problems.mgh](https://fides.fe.uni-lj.si/pyopus/doc/problems.mgh.html), [PyPI pyopus](https://pypi.org/project/pyopus/), [Rust mgh](https://docs.rs/crate/mgh/latest), [Scilab uncprb](https://atoms.scilab.org/toolboxes/uncprb/0.2). I found no maintained PyPI package dedicated to MGH.
- **MGH optima [prior knowledge, from MGH 1981, not re-fetched]:** Powell badly scaled has f* = 0 at x ≈ (1.098e-5, 9.106). Brown badly scaled has f* = 0 at (1e6, 2e-6). Both reward correct magnitude across many decades, which is exactly the SLUD use case — [MGH 1981, doi:10.1145/355934.355936](https://doi.org/10.1145/355934.355936).
- **opfunu:** "An Open-source Python Library for Optimization Benchmark Functions"; latest release 1.0.4 (2024-06-04). It implements CEC 2005–2022 suites plus classical named functions — [PyPI](https://pypi.org/project/opfunu/), [repo](https://github.com/thieu1995/opfunu). IOHexperimenter natively includes CEC 2013 (niching) and CEC 2022 (bound-constrained) — [IOHexperimenter README](https://github.com/IOHprofiler/IOHexperimenter).
- **pycma:** `cma` 4.5.0 (2026-09-13) — [repo](https://github.com/CMA-ES/pycma). COCO's own example shows `cma.fmin2` as the drop-in solver — [example_experiment_simple.py](https://raw.githubusercontent.com/numbbo/coco-experiment/main/build/python/example/example_experiment_simple.py). `modcma` 1.2.0 (2025-12) offers modular CMA-ES variants — [PyPI](https://pypi.org/project/modcma/).
- **GPU/JAX/Torch ES libraries:** evosax 0.3.2 ("Evolution Strategies in JAX", Oct 2026), EvoX 1.4.0 (Sep 2026), EvoTorch 0.6.1 ("built directly on top of PyTorch, created at NNAISENSE", May 2025) — [evosax](https://github.com/RobertTLange/evosax), [evox](https://github.com/EMI-Group/evox), [evotorch](https://evotorch.ai).
- **HPO benchmarks:** YAHPO Gym 1.0.2 (Aug 2024); HPOBench is not on PyPI; bayesmark's last release was 0.0.8 (2020); Optuna 5.0.0 (Sep 2026); SMAC 2.4.1 (Sep 2026); BoTorch 0.18.1 (Jun 2026) — [YAHPO](https://github.com/slds-lmu/yahpo_gym), [bayesmark PyPI](https://pypi.org/project/bayesmark/), [optuna PyPI](https://pypi.org/project/optuna/), [smac PyPI](https://pypi.org/project/smac/), [botorch](https://botorch.org).
- **SciPy:** SciPy maintains a large set of global-optimization benchmark functions (`benchmarks/benchmarks/go_benchmark_functions`, based on Jamil & Yang 2013 and Gavana's collection) used for asv benchmarks of `differential_evolution`, `shgo`, `dual_annealing` and others [prior knowledge, not re-fetched; GitHub API access to scipy was refused] — [scipy repo](https://github.com/scipy/scipy/tree/main/benchmarks/benchmarks).

### Inferences
- **Citing the badly scaled problems:** take Powell and Brown badly scaled from MGH 1981, and optionally validate the implementations against PyCUTEst `POWELLBS`/`BROWNBS`. Mention that CUTEst's POWELLBS is the n-dimensional chained generalization; if you use n > 2 you need to say which version.
- **opfunu:** CEC suites (via opfunu or IOH) are common in "metaheuristic" venues, but BBOB-community reviewers often criticise CEC-only evaluation. Optional at most.
- **GPU/HPO libraries:** GPU ES libraries matter only if the 200 runs × problems × methods budget becomes the bottleneck (the Cp fit is cheap). HPO benchmarks (YAHPO/HPOBench) are relevant only for an "HPO log-scale hyperparameter" side story. BoTorch synthetic functions are standard O(1)-scaled test functions with no added value here.

### Gaps
- I did not verify which SciPy benchmark functions exist in 1.18 (GitHub API access was refused).
- I did not verify whether CUTEst `BROWNBS` has a variable-n version (the header says "in 2 variables" but also "sum of n-1 sets").

---

## Q6. Baseline optimizers reviewers would expect beyond PSO

### Takeaway
A journal-grade parametrization study should show that the SLUD effect holds across *optimizer families*, not just PSO. The expected minimum set is CMA-ES (pycma, with IPOP/BIPOP restarts), Differential Evolution (scipy or L-SHADE-type), Nelder–Mead, a model-based local DFO method (COBYQA/BOBYQA/Powell), uniform random search (in each parametrization), and ideally NGOpt as a "wizard" baseline. All are run under each of SLUD/LUD/LIN.

### Cited Findings
- **CMA-ES as default:** COCO's reference example uses `scipy.optimize.fmin` (Nelder–Mead) as the default solver, suggests `cma.fmin2` as the replacement, and post-processes against BFGS from the archive ('bfgs!') — [example_experiment_simple.py](https://raw.githubusercontent.com/numbbo/coco-experiment/main/build/python/example/example_experiment_simple.py).
- **Nevergrad's ill-conditioning plans:** `illcondi` uses the `"basics"` optimizer group, and the photonics/ceviche plans list DiagonalCMA, PSO, DE, CMA, OnePlusOne and MetaModel variants as standard comparators — [experiments.py](https://github.com/facebookresearch/nevergrad/blob/main/nevergrad/benchmark/experiments.py).
- **Kinetic-model precedent:** benchmarking of kinetic-model calibration compared local and global strategies (enhanced scatter search versus multistart local) in both linear and logarithmic parameter space — [Villaverde et al. 2019, Bioinformatics, via Freiburg benchmarking wiki](https://benchmarking.uni-freiburg.de/index.php/Benchmarking_optimization_methods_for_parameter_estimation_in_large_kinetic_models).
- **Combustion precedent:** a combustion-kinetics study compared 10 stochastic optimizers (MC, LHS, MLE, MCMC, SCE-UA, SA, ROPE, ABC, FSCABC, DDS) — [Ashraf & Pfaendtner, Ind. Eng. Chem. Res. 59:19212–19225, 2020 (NSF PAR)](https://par.nsf.gov/servlets/purl/10276863).

### Inferences
- **Invariance argument:** CMA-ES adapts a full covariance and is rotation-invariant, so it can partially "learn" a bad scaling that PSO/DE cannot. If SLUD helps PSO but not CMA-ES, reviewers will want to see that contrast explicitly. Separable-ish problems (MGH 2D, f2) versus rotated ones (f10) separate the per-coordinate transform effect from correlation learning.
- **Random-search control:** random search under each parametrization is essential. It isolates the *prior/measure* effect of SLUD (the probability mass it puts per decade) from the optimizer's dynamics.
- **Local DFO baseline:** at least one gradient-free local method is needed (SciPy `Powell`, `Nelder-Mead`, and `COBYQA`; I recall COBYQA was added in SciPy 1.14, but this was not re-verified), with multistart. MGH problems are classical local-solver test problems, and reviewers from numerical optimization will ask about them.
- **Restarts and budgets:** use budgets proportional to dimension (COCO convention: budget multiplier × n), report anytime performance (ECDF over targets), and run independent restarts. The current "200 seeded runs" maps naturally onto IOH/COCO "runs per instance". For BBOB, instances (rotations/shifts) are preferred over repeated seeds on one instance.

### Gaps
- I did not find a published, community-agreed "required baseline list". The above is synthesized from platform defaults and domain papers.

---

## Q7. Real-world-like benchmarks for chemical-kinetics / combustion parameter fitting

### Takeaway
I found no standardized, community-maintained combustion-mechanism optimization benchmark suite in Python comparable to BBOB. The closest benchmark-grade analog is from systems biology: the Villaverde et al. 2019 kinetic-model benchmarks, which explicitly test log versus linear parameter space, and the PEtab ecosystem (petab 0.9.0, pyPESTO 0.7.0), whose problem spec declares a per-parameter `parameterScale` (lin/log/log10). Combustion-side precedents (Ashraf & Pfaendtner 2020 with the `kinexns` code, GA-based mechanism optimization) are individual studies, not suites. A Cantera-based fitting problem would have to be self-built.

### Cited Findings
- **Villaverde et al. 2019 (Bioinformatics):** six benchmark kinetic models with 36–383 parameters and 105–7567 data points (metabolic and signalling). Every method was tested "in both linear and logarithmic parameter space". Enhanced scatter search outperformed multistart local optimization by about 2× in computation time. Code (AMICI/MEIGO, MATLAB) is on Zenodo, DOI 10.5281/zenodo.1160342 — [Freiburg benchmarking wiki](https://benchmarking.uni-freiburg.de/index.php/Benchmarking_optimization_methods_for_parameter_estimation_in_large_kinetic_models), [bioRxiv 10.1101/295006](https://www.biorxiv.org/content/10.1101/295006v1.full.pdf+html), [PMC6394396](https://pmc.ncbi.nlm.nih.gov/articles/PMC6394396).
- **PEtab/pyPESTO:** petab 0.9.0 ("Parameter estimation tabular data") and pypesto 0.7.0 ("python-based Parameter EStimation TOolbox") were both released in Sep 2026 — [petab PyPI](https://pypi.org/project/petab/), [pypesto PyPI](https://pypi.org/project/pypesto/). The PEtab `parameterScale` column (lin/log/log10) and the Benchmark-Models-PEtab collection (Hass et al. 2019, *Bioinformatics*) are [prior knowledge, not re-fetched] — [Benchmark-Models-PEtab repo](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab).
- **Ashraf & Pfaendtner 2020:** "Assessing the Performance of Various Stochastic Optimization Methods on Chemical Kinetic Modeling of Combustion", *Ind. Eng. Chem. Res.* 59:19212–19225. It compares 10 algorithms on rate parameters of the San Diego propane mechanism and releases the Python package `kinexns` (CVode solve, sensitivity analysis, parameter optimization), which is not on PyPI — [NSF PAR](https://par.nsf.gov/servlets/purl/10276863).
- **Other combustion work:** GA optimization of a propane mechanism at engine-relevant conditions (ORNL) — [ORNL record](https://impact.ornl.gov/en/publications/genetic-algorithm-optimization-of-a-chemical-kinetic-mechanism-fo/). A comparison of GA, Grey Wolf, GA–PSO hybrid and Whale optimization for Arrhenius coefficients in methane mechanism reduction — [Amirkabir J. Mech. Eng.](https://mej.aut.ac.ir/article_6154_en.html). DeePMO is a deep-learning-guided kinetic-parameter optimizer — [DeePMO pdf](https://ins.sjtu.edu.cn/people/xuzhiqin/pub/deepmo.pdf). Cantera 3.2.0 was released 2025-11-18 — [cantera.org](https://cantera.org).
- **Mechanism optimization frameworks [prior knowledge, not re-fetched]:** Optima/Optima++ (Turányi group, ELTE) optimize Arrhenius parameters in transformed (ln A, n, E/R) space. OptiSMOKE++ (Politecnico di Milano) couples OpenSMOKE++ with DAKOTA. Neither is a Python benchmark suite.

### Inferences
- **Analog for SLUD:** the systems-biology kinetic benchmarks are the strongest "real-world" analog for SLUD. Their parameters (rate constants, scalings) span many decades, and the community has already shown that log-space fitting matters (Villaverde 2019). A SLUD paper could add one or two PEtab benchmark models (via pyPESTO objective functions) as a real-world tier. Note that these parameters are positive, so SLUD's sign-spanning feature is not exercised; it then degenerates to LUD.
- **Where the sign matters:** the NASA 7-coefficient Cp polynomial fit in the current code has signed coefficients spanning about 1e-15 to 1e1. This is a genuinely sign-spanning, multi-decade problem and arguably the most distinctive real-world case. Making it a documented, reproducible benchmark (fixed species set from a Cantera/GRI-Mech YAML, fixed temperature grid, published objective) would itself be a contribution, given that no such suite exists.
- **Combustion rate parameters:** Arrhenius pre-exponential A spans decades (positive), the temperature exponent n is signed O(1), and activation energies can be negative for some reactions. A small Cantera ignition-delay fitting problem (e.g., H2/O2 with about 5 to 10 perturbed A-factors) would be a credible "realistic" tier, but it is expensive to run 200 times.

### Gaps
- I found no reliable source for a public, maintained combustion-mechanism optimization benchmark suite in Python, and no Cantera-based benchmark collection.
- I could not read Villaverde 2019's quantitative log-versus-linear results (PMC unreachable). Check them before citing specific numbers.
- I did not find a study benchmarking CMA-ES against GA/PSO specifically on combustion mechanisms.

---

## Q8. Which framework best fits a "search-space transformation / parametrization" study?

### Takeaway
No single framework covers the SLUD claim. The best-fitting, reviewer-proof design is a three-tier design:
- **Tier 1:** COCO `bbob` (and optionally `bbob-boxed`) through `cocoex`, with the transform wrapped around each optimizer, as a control. Report ECDFs and compare against archived data.
- **Tier 2:** a badly-scaled tier: MGH Powell/Brown badly scaled (cited to MGH 1981 / CUTEst POWELLBS/BROWNBS), plus a rescaled/sign-randomized BBOB variant, logged with IOHexperimenter `wrap_problem` and analyzed in IOHanalyzer.
- **Tier 3:** a real-world tier: the NASA-Cp fit, optionally plus a PEtab kinetic model.

Optimizers come from pycma, SciPy, pymoo (PSO/DE) and Nevergrad (NGOpt, `ng.p.Log` as the LUD reference).

### Cited Findings
- **COCO:** logs any callable and compares against archived algorithms via `cocopp.main(... 'bfgs!')` — [example_experiment_simple.py](https://raw.githubusercontent.com/numbbo/coco-experiment/main/build/python/example/example_experiment_simple.py).
- **IOH:** has a "flexible interface for adding new suites and problems", and its data is IOHanalyzer-compatible — [IOHexperimenter README](https://github.com/IOHprofiler/IOHexperimenter).
- **Nevergrad:** `Log` is the standard log-uniform parameter, with bounds and mutation handled internally — [parametrization.rst](https://github.com/facebookresearch/nevergrad/blob/main/docs/parametrization.rst). Its `illcondi`/`far_optimum_es` plans show that scale robustness is a recognized benchmark axis — [experiments.py](https://github.com/facebookresearch/nevergrad/blob/main/nevergrad/benchmark/experiments.py).
- **BBOB optima** are confined to [-4,4] — [suite_bbob_legacy_code.c](https://raw.githubusercontent.com/numbbo/coco-experiment/main/src/suite_bbob_legacy_code.c). **MGH badly scaled problems** have optima spanning about 11 decades (Brown: 1e6 and 2e-6) — [MGH 1981](https://doi.org/10.1145/355934.355936).
- **Log versus linear** parameter space is an established experimental factor in kinetic-model benchmarking — [Villaverde et al. 2019](https://benchmarking.uni-freiburg.de/index.php/Benchmarking_optimization_methods_for_parameter_estimation_in_large_kinetic_models).

### Inferences
- **Concrete design suggestions** (synthesized; not from a single source):
  1. Treat the parametrization (SLUD/LUD/LIN) as the experimental factor and the optimizer as a blocking factor. Report per-optimizer ECDFs (fixed-target) plus fixed-budget final-error distributions, with nonparametric tests (Mann–Whitney/Wilcoxon with Holm correction, or IOHanalyzer's DSC).
  2. Log in the *original* x-space so all parametrizations are compared on identical f-targets.
  3. Count every evaluation, including LUD's separate sign variable, which changes dimension (n versus 2n). The extra dimension is a confound that reviewers will raise, so report the dimension and treat the sign variable explicitly (e.g., continuous-relaxed versus categorical).
  4. Vary lb/ub (decades spanned) as a sensitivity analysis, because SLUD's performance depends on how tight the a-priori range is.
  5. Include a "known-scale" control (BBOB in LIN on [-5,5]) where SLUD is expected not to help.
  6. Pin and report versions (table above). Use cocoex ≥ 2.8 (the PyPI name is `coco-experiment`), ioh 0.3.22, cma 4.5.0, pymoo 0.6.2 and nevergrad 1.0.12.
- **Precedent:** the closest published precedents for "parametrization as the studied factor" are in systems-biology parameter estimation (log versus linear), not in the BBOB literature. A SLUD paper bridging the two (BBOB methodology plus kinetic-fitting motivation) would be well-positioned for venues such as ACM TELO, Evolutionary Computation, or GECCO (BBOB workshop or Theory/ENUM tracks).

### Gaps
- I did not find any prior BBOB/COCO paper that studies a *signed* log-uniform variable transform. Absence of evidence is not proof, and a dedicated literature search (Google Scholar: "log transformation" + "black-box optimization" + "BBOB") is advisable.
- Whether IOHanalyzer can display two datasets with different "algorithm" labels for the same optimizer under different parametrizations is assumed (label each as `PSO-SLUD`, `PSO-LIN`, and so on), not verified.
