# Fast Python compute for a population-based optimization benchmark (cheap analytic functions vs expensive Cantera evaluations), as of Oct 2026

Method note: during this session WebFetch could not resolve most doc domains (DNS failure) and direct curl was blocked by the proxy. Findings below come from web-search result extracts of the cited pages (mostly primary docs, release notes, and papers), plus one local micro-benchmark that I ran myself (marked **MEASURED (local)**). Statements marked "claim" are vendor or author claims I could not check independently.

## Q1. NumPy vectorization of transform and objectives (pymoo Problem vs ElementwiseProblem); parallelizing across runs instead of within a generation; Ray overhead for millisecond tasks

### Takeaway
For the cheap regime, the main problem is Python-level per-individual work, not a shortage of cores. Vectorizing the SLUD transform and the objectives over the whole (100 x n) population matrix gave about 110x per generation in a local test. After that change, each generation costs tens of microseconds, so any per-generation Ray or IPC dispatch is pure overhead. The right parallel unit is the independent run (seed x parametrization x function), giving 3,600 coarse tasks.

### Cited Findings
- pymoo has two problem styles. In a vectorized `Problem`, `_evaluate` receives the whole population as a matrix (one row per solution) and computes all objective values at once. In an `ElementwiseProblem`, `_evaluate` is called once per solution. The pymoo paper lists vectorized evaluation as the first parallelization option, notes that PyTorch tensors could enable GPU evaluation, and recommends Dask for distributing evaluations across machines — [pymoo paper, arXiv 2002.04504](https://arxiv.org/abs/2002.04504). A third-party summary says the parallelization interfaces are Starmap (threads or processes), Dask and Joblib (`JoblibParallelization`, optional joblib dependency); I could not open the official page — [pymoo skill wiki (secondary)](https://moltchat-agent-commons.onrender.com/wiki/pymoo_skill_(K-Dense_scientific-agent-skills)). Official page to check: https://pymoo.org/problems/parallelization.html
- The latest pymoo release I could find is 0.6.1.6 (25 Nov 2025). I did not find a 2026 release — [piwheels pymoo](https://www.piwheels.org/project/pymoo); [Tessl registry](https://tessl.io/registry/skills/github/K-Dense-AI/scientific-agent-skills/pymoo).
- **MEASURED (local)** on Python 3.13.16, NumPy 2.5.3, a 4-vCPU Linux container, population 100, n=5:
  - Per-individual Python loops (a scalar signed-log-uniform transform with a per-variable loop, plus a 5-coefficient polynomial residual over 100 temperatures, plus Rosenbrock) took **~7.0 ms per generation**.
  - The NumPy-vectorized equivalent (`np.copysign`/`10**` transform, `C @ vander(T).T`, row-wise Rosenbrock) took **~64 µs per generation**, about **110x faster**. Results matched to rtol 1e-9.
  - The transform is an illustrative SLUD-like map, not the repository's exact code. Script: /tmp/claude-0/-home-user-Signed-Log-Uniform-Distribution-SLUD/0f423eee-96ef-56c3-b23e-877b66e98afc/scratchpad/bench/b.py
- Ray's own docs classify too-fine-grained tasks as an anti-pattern: a remote call costs much more than a normal function call. In the docs' example of 10,000 tiny tasks:
  - one task per item took about 1.6 s
  - a plain serial loop took about 0.17 s
  - batches of 1,000 took about 0.03 s

  The recommended fix is to batch work so each task does substantial computation — [Ray anti-pattern: too fine-grained tasks](https://docs.ray.io/en/latest/ray-core/patterns/too-fine-grained-tasks.html).
- Ray's install docs (latest is around Ray 2.59) still mark Windows support as **Beta**. They say multi-node clusters on Windows are untested and that performance on Windows is slower because opening files is slower. They also say Windows has no copy-on-write fork, so new processes can need more memory — [Ray installation docs](https://docs.ray.io/en/latest/ray-overview/installation.html). Windows support moved from alpha (0.8.6) to beta in Ray 1.10 — [Anyscale Ray 1.10 blog](https://www.anyscale.com/blog/ray-version-1-10-has-been-released).

### Inferences
- **Derived estimate:** with the current design, each Ray task (a batch of 10 individuals) holds about 0.7 ms of Python work. That is the same order as Ray's per-task overhead: the docs' example implies roughly 0.1–0.2 ms per task for trivial tasks, plus scheduling and serialization. Within-generation Ray parallelism therefore likely yields little or no speedup, and could be a net slowdown, especially on Windows.
- **Expected total cost (derived from the local measurement, not end-to-end):**
  - Element-wise evaluation: about 3,600 runs x 500 generations x ~7 ms ≈ 3.5 h of single-core evaluation time.
  - Vectorized evaluation: about 3,600 x 500 x ~64 µs ≈ 2 min.
  - The remaining cost would then be pymoo's own per-generation overhead (survival, PSO velocity update, object creation), which I did not measure. It is likely of the same order or larger than the vectorized evaluation.
- **Recommended structure for the cheap regime:**
  1. Use a vectorized `pymoo.core.problem.Problem` with a NumPy SLUD transform.
  2. Run each (function, parametrization, seed) combination as one serial task.
  3. Spread these tasks over cores with `concurrent.futures.ProcessPoolExecutor` or `joblib.Parallel(n_jobs=-1)`. Ray is optional and adds little.
  4. Expect near-linear scaling in core count on top of the ~100x vectorization gain.
  5. Set `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` in workers so BLAS threads do not oversubscribe the cores (standard practice, not sourced here).
- An embarrassingly parallel run-level loop is also simpler for a team than Ray remote batches, and it works identically on Windows.

### Gaps
- I could not open pymoo's official parallelization page to confirm the current class names and signatures (`StarmapParallelization`, `DaskParallelization`, `JoblibParallelization`, Ray).
- I have no measured number for pymoo PSO's internal per-generation overhead.

## Q2. Numba, JAX (+ evosax / EvoX / EvoJAX / TensorNeat) and EvoTorch: status, "thousands of seeds in one vmap", float64 caveats, use in published benchmarking

### Takeaway
Numba is mature and the lowest-friction next step after NumPy. It supports Python 3.14 since 0.63 (Dec 2025), and the current releases are 0.67/0.68 in 2026. JAX-based EC libraries can vmap many seeds in one compiled program, and EvoX reports ≥10x GPU speedups at large scale. However, JAX defaults to float32, and FP64 on consumer GPUs is 1/32–1/64 of FP32 throughput. For the 1e-14 coefficients, use JAX on CPU with x64 or a data-centre GPU. Rewriting a pymoo PSO in JAX is mostly worthwhile when you need tens of thousands of runs, not 3,600.

### Cited Findings
- **Numba versions:**
  - 0.63.0 (8 Dec 2025) added Python 3.14 support and **experimental** initial support for free-threaded CPython builds. The 0.63.0b1 beta came out one day before the Python 3.14 release — [Numba 0.63.0 notes](https://numba.readthedocs.io/en/stable/release/0.63.0-notes.html); [Numba 0.63.0b1 announcement](https://numba.discourse.group/t/ann-numba-0-63-0b1-llvmlite-0-46-0b1-python-3-14-support/3068).
  - 0.65.0 was released 31 Mar 2026 — [Numba 0.65.0 notes](https://numba.readthedocs.io/en/latest/release/0.65.0-notes.html).
  - 0.67.0 was released 11 Aug 2026 — [Numba 0.67.0 notes](https://numba.readthedocs.io/en/stable/release/0.67.0-notes.html).
  - A secondary source (Wikipedia, as summarized by search) gives 0.68.0 as current. An xarray issue reports no cp315t (free-threaded 3.15) Numba wheels on PyPI yet — [xarray issue #11710](https://github.com/pydata/xarray/issues/11710).
  - The deprecated `numba.pycc` cannot build free-threaded AOT modules — [Numba 0.63.0 notes](https://numba.readthedocs.io/en/stable/release/0.63.0-notes.html).
  - Intel SVML support was lost starting with 0.62.0 when llvmlite moved to LLVM 20 — [Numba 0.63.0 notes](https://numba.readthedocs.io/en/0.63.0/release/0.63.0-notes.html).
- **evosax** (Lange, GECCO'23 companion):
  - The paper shows `jax.vmap(run_es_loop, in_axes=(0,None,None))(seed_rng, num_gens)`, which runs many independent ES runs by batching over PRNG keys, and `jax.pmap` to spread the seeds across devices — [evosax paper arXiv 2212.04180](https://arxiv.org/abs/2212.04180); [ACM DL](https://dl.acm.org/doi/10.1145/3583133.3590733).
  - The current README advertises an ask/eval/tell API with full jit/vmap/`lax.scan` support — [evosax GitHub](https://github.com/RobertTLange/evosax).
- **EvoX** (Huang et al., IEEE TEVC):
  - Scalability tests with PSO, DE and CMA-ES on Sphere found that the CPU can win at small dimension or population, but that the GPU "rapidly overtakes as the scale increases, frequently achieving a tenfold or greater speedup". The authors say the benefit is algorithm-dependent. The multi-node tests used Xeon Gold 6240 nodes with 4x RTX 2080 Ti — [EvoX arXiv 2301.12457](https://arxiv.org/pdf/2301.12457).
  - The README's "over 100x speedups" is a claim with unspecified conditions — [EvoX GitHub](https://github.com/EMI-Group/evox).
  - EvoX 1.0 (2025) moved its backend to PyTorch — [EvoX 1.0 release note](https://evox.group/index.php?m=home&c=View&a=index&aid=145) (title only seen; backend detail is from my background knowledge, so verify).
- **EvoJAX** (Google) is a JAX neuroevolution toolkit — [google/evojax](https://github.com/google/evojax).
- **EvoTorch** (NNAISENSE): the technical report describes a PyTorch EC library for high-dimensional problems with GPU support and Ray-based parallelization — [EvoTorch arXiv 2302.12600](https://arxiv.org/pdf/2302.12600v1). The latest PyPI release I found is 0.6.1 (14 May 2025) — [pepy evotorch](https://pepy.tech/project/evotorch).
- **JAX precision:** JAX runs in float32 by default, and `jax.config.update("jax_enable_x64", True)` is a process-global switch — [magpylib-jax precision doc (secondary)](https://magpylib-jax.readthedocs.io/en/latest/precision.html).
- **Consumer-GPU FP64 throughput:** one project doc puts it at about 1/32 of FP32, versus about 1/2 on A100/H100. Another cites RTX 4090 at 1:64 (≈82 TFLOPS FP32 vs ≈1.3 TFLOPS FP64), so the sources conflict on the exact ratio. One project concluded that float64 physics on consumer GPUs nets only 1.3–2.6x before transfer overhead, and chose CPU-only JAX — [heterodyne ADR-006](https://heterodyne.readthedocs.io/en/latest/developer/adrs/adr_006_no_consumer_gpu.html); [heterodyne ADR-001](https://heterodyne.readthedocs.io/en/latest/developer/adrs/adr_001_jax_cpu_only.html).

### Inferences
- **Numba:** `@njit` on the SLUD transform and objectives, or `@vectorize`/`@guvectorize` for the transform, would typically match or exceed the NumPy-vectorized speed for these small kernels. It also avoids temporaries and lets you keep the readable scalar loop that the team already wrote. `prange` only helps for large arrays, so with population 100 and n≤5, process-level parallelism over runs is better.
- **Numba versus plain NumPy:**
  - Numba adds a compile step (seconds) per process.
  - It is useful where vectorization is awkward, such as branching in SLUD near zero and per-element loops.
  - Its main remaining risk is lag behind new CPython and NumPy releases. Historically it has shipped support around each CPython release.
- **JAX / vmap over seeds:** you could run all 200 seeds x 3 parametrizations of one function as a single jit-compiled `lax.scan` over 500 generations. This would likely reach millions of evaluations per second on CPU.
- **JAX costs:**
  - The PSO, and any pymoo-specific operators, must be rewritten in JAX (evosax and EvoX include PSO-like or ES algorithms but are not pymoo).
  - float64 must be enabled globally.
  - On a consumer GPU, FP64 erodes most of the advantage.
  - The JAX optimizers would not be the same code that later drives Cantera, because Cantera is not JAX-traceable.
  - For a team that wants simple code shared across both regimes, this is a poor fit unless the cheap benchmark grows to around 10^5 runs.
- **Use in published benchmarking:** evosax, EvoX and EvoJAX papers appear in GECCO and TEVC and in neuroevolution/RL work. Classical numerical benchmarking (COCO/BBOB, IOHprofiler) mostly uses CPU implementations such as pycma, nevergrad and pymoo. I did not find a source quantifying adoption.

### Gaps
- I could not open the JAX docs to quote the official float64 guidance (the "Default dtypes and the X64 flag" section in jax.readthedocs.io "Sharp bits").
- I did not find a published head-to-head of pymoo vs evosax/EvoX on small-population (100) low-dimensional problems. EvoX's own results suggest the CPU wins at small scale.
- TensorNeat (an EMI-Group JAX NEAT library) was not researched, as it is irrelevant to continuous parameter fitting.

## Q3. Compiled kernels: Cython, pybind11, nanobind, Fortran via f2py (meson)

### Takeaway
For the cheap analytic functions, compiled extensions buy little beyond NumPy or Numba and add build complexity, especially on Windows. They are worth it only when a kernel is already written in C++ or Fortran, or for the Cantera side, where the user already maintains C++/Cython bindings. For new Fortran wrapping, f2py must now use the meson backend on Python ≥3.12.

### Cited Findings
- NumPy 1.26.0 replaced its distutils build with Meson because Python 3.12 removed distutils — [NumPy 1.26.0 release notes](https://numpy.org/devdocs/release/1.26.0-notes.html).
- On Python ≥3.12, f2py defaults to the meson backend, and users are "strongly recommended to switch to meson". Choose the backend with `--backend`, pass dependencies with `--dep`, and select the Fortran compiler via the `FC` environment variable instead of `--fcompiler` — [NumPy f2py: Migrating to meson](https://numpy.org/doc/2.4/f2py/buildtools/distutils-to-meson.html).
- On the free-threaded build, the stable ABI (abi3) is not supported, so compiled extensions need per-version cp314t wheels — [psutil blog on free-threaded wheels](https://psutil.readthedocs.io/latest/blog/2025/wheels-for-free-threaded-python.html).

### Inferences
- **Rough ordering for this workload (background knowledge, not measured here):**
  1. NumPy vectorization: about 100x, measured above.
  2. Numba or Cython typed memoryview loops: usually 1–5x more over NumPy for tiny arrays, because they avoid temporaries and NumPy call overhead.
  3. pybind11 or nanobind C++: a similar kernel speed to Cython, plus about 1 µs of per-call overhead. nanobind's documentation claims lower call overhead and smaller binaries than pybind11, but I did not verify those claims in this session.
- At this scale, the binding choice matters less than calling once per population rather than once per individual.
- **Cantera:** the Cantera Python module is itself Cython over C++. A customized flame solver is best exposed through the same mechanism (Cython `.pyx`, or C++ plus a thin Cython wrapper) rather than introducing pybind11 in parallel.

### Gaps
- I did not retrieve nanobind or pybind11 overhead benchmarks. Check https://nanobind.readthedocs.io/en/latest/benchmark.html
- I found no Cython 3.1/3.2 free-threading status source.

## Q4. Python 3.13/3.14: free-threading (PEP 703/779) and the experimental JIT (PEP 744); library support

### Takeaway
In Python 3.14 (Oct 2025), free-threading is officially supported but optional (PEP 779 "phase II"). It carries a single-thread slowdown of about 5–10% (secondary source). NumPy has free-threaded wheels, Numba support is experimental, and I found no confirmation of Ray or Cantera support for production use. The 3.14 JIT gives roughly 0% gain; measurable JIT gains (≈4–12%) only appear in 3.15. Neither feature changes the advice: vectorize, then use process-level parallelism.

### Cited Findings
- **PEP 779:** accepted 16 Jun 2025 for 3.14. It removes the "experimental" tag from the free-threaded build.
  - SC performance target: stay within 10% degradation. Changes up to 15% must be discussed with the SC.
  - SC memory target: under 15% increase, with a hard limit of 20% — [PEP 779](https://peps.python.org/pep-0779/); [discuss.python.org PEP 779 thread](https://discuss.python.org/t/pep-779-criteria-for-supported-status-for-free-threaded-python/84319).
  - A secondary explainer says the gap was 5–10% by 3.14 — [pydevtools: what is PEP 779](https://pydevtools.com/handbook/explanation/what-is-pep-779/).
  - The Register reported "cautious" free-threaded support at the 3.14 release (8 Oct 2025) — [The Register](https://theregister.com/software/2025/10/08/python-314-released-with-cautious-free-threaded-support/349336).
- **NumPy:**
  - 2.3.0 (Jun 2025), 2.4.0 (Dec 2025) and 2.5.0 (Jun 2026) each improved free-threading support.
  - NumPy warns that supporting free-threaded Python "does not mean that NumPy is thread safe" — [NumPy news](https://numpy.org/news/).
  - cp315t wheels exist for numpy, pandas, scipy and pyarrow — [xarray issue #11710](https://github.com/pydata/xarray/issues/11710).
- **Numba:** experimental free-threading support since 0.63 — [Numba 0.63.0 notes](https://numba.readthedocs.io/en/stable/release/0.63.0-notes.html).
- **Cantera:** PyPI lists a free-threaded CPython 3.14 wheel for **Cantera 3.2.0**, classified "Free Threading :: 1 - Unstable" (from the search summary of the PyPI listing). Cantera 3.1.0 wheels cover Python 3.8–3.13 only — [Cantera 3.1 pip install docs](https://cantera.org/3.1/install/pip.html); [cantera on PyPI](https://pypi.org/project/cantera/).
- **JIT:**
  - Ken Jin (core developer) said the 3.14 JIT ranges from slower than the interpreter to roughly equal — [DevClass, Jul 2025](https://devclass.com/2025/07/09/despite-30-months-work-core-developer-says-pythons-jit-compiler-is-often-slower-than-the-interpreter/).
  - Miguel Grinberg measured 3.14 with the JIT at 1.00x 3.14 without it on Fibonacci. His roughly 27% gain of 3.14 over 3.13 came from non-JIT changes — [Grinberg blog](https://blog.miguelgrinberg.com/post/python-3-14-is-here-how-fast-is-it).
  - Brandt Bucher: 3.14 is about 8% faster than 3.13 overall. The tail-call interpreter adds 3–5% on pyperformance — [discuss.python.org: faster CPython, JIT and 3.14](https://discuss.python.org/t/news-faster-cpython-jit-and-3-14/85326).
  - For 3.15, the JIT reports about 4–12% geometric-mean gains on Tier-1 platforms — [PEP 836](https://peps.python.org/pep-0836/); [LWN](https://lwn.net/Articles/1090825/).
  - PEP 744 itself is Informational (design of the 3.13 JIT) — [PEP 744 source](https://github.com/python/peps/blob/main/peps/pep-0744.rst).

### Inferences
- For the cheap benchmark, simply upgrading 3.12→3.14 may give 10–30% on pure-Python loops (claims and benchmark-specific numbers). That is negligible next to the ~100x from vectorization.
- Free-threading could replace process pools with thread pools for the run-level parallelism (lower memory, no pickling). However, pymoo, Ray and a customized Cantera build have no confirmed free-threading support, and Numba support is experimental. For a team codebase in 2026, processes remain the safe default.

### Gaps
- I could not access the py-free-threading compatibility tracker (https://py-free-threading.github.io/tracking/) for per-package status of Ray, pymoo, scipy, Cython, pybind11 and nanobind.
- No source on Ray's cp314t wheels was found.

## Q5. Parallelism for expensive evaluations (Cantera, seconds to minutes per eval): Ray vs multiprocessing/concurrent.futures vs joblib/loky vs Dask vs mpi4py; asynchronous/steady-state evaluation; pymoo built-ins

### Takeaway
When each evaluation takes seconds to minutes, dispatch overhead (Ray's ~0.1–1 ms, process-pool pickling) becomes irrelevant. The choice then hinges on portability, simplicity, and handling heterogeneous runtimes:
- **Workstation (including Windows):** `concurrent.futures.ProcessPoolExecutor` or joblib/loky, with a per-worker Cantera initializer, is the simplest robust option.
- **Multi-node HPC:** Dask (pymoo has a hook for it) or mpi4py (`mpi4py.futures`) fit SLURM better than Ray. Ray multi-node is untested on Windows.
- **Wildly varying flame solve times:** synchronous generational evaluation idles workers until the slowest evaluation finishes. Asynchronous steady-state schemes recover that time, at some cost to algorithm fidelity.

### Cited Findings
- Ray is Beta on Windows: multi-node untested, slower file I/O, no fork — [Ray installation docs](https://docs.ray.io/en/latest/ray-overview/installation.html).
- **Synchronous waste:** heterogeneous evaluation times degrade synchronous parallel EAs because "parallel processing units will idle until the longest evaluation has completed" — [arXiv 2308.04102 (async EAs for long, variable simulations)](https://arxiv.org/html/2308.04102v3).
- **Asynchronous steady-state trade-offs:** such schemes address heterogeneous evaluation costs, but linear speedup no longer holds, and convergence can be hindered because the algorithm differs from the sequential one. Cost-aware selection is proposed as a remedy, in a diesel-engine MOEA study — [Asynchronous Master/Slave MOEAs and Heterogeneous Evaluation Costs (HAL)](https://hal.archives-ouvertes.fr/hal-00689965).
- **Bias:** asynchronous EAs are biased toward individuals that evaluate quickly — [The Impact of Asynchrony on Parallel Model-Based EAs, arXiv 2303.15543](https://arxiv.org/pdf/2303.15543).
- **Measured speedups:** asynchronous evolution reported over 2x on one task and 14x on a task with highly variable evaluation times (neural architecture search context) — [Cognizant: Asynchronous Evolution of DNN architectures](https://www.cognizant.com/us/en/ai-lab/publications/asynchronous-evolution-neural-architecture-search).
- **pymoo:** evaluation is serial by default, with options for threads, Dask (recommended for clusters), and vectorized matrices — [pymoo paper](https://arxiv.org/abs/2002.04504). pymoo's PSO and GA are generational (synchronous batch). I found no built-in asynchronous mode in pymoo (absence not confirmed from official docs).
- **scipy `differential_evolution`:**
  - `workers` accepts an int or any map-like callable (e.g. `multiprocessing.Pool.map`, called as `workers(func, iterable)`).
  - It forces `updating='deferred'`, requires a picklable `func`, and overrides `vectorized` when `workers != 1` — [SciPy differential_evolution docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html).
- **pagmo2/pygmo:**
  - Islands in an archipelago evolve asynchronously in parallel — [pygmo docs](https://pygmo.readthedocs.io/).
  - Since 2.13, a batch fitness evaluation (bfe) framework with Python evaluators based on multiprocessing (`mp_bfe`) and ipyparallel (`ipyparallel_bfe`) lets selected algorithms evaluate many decision vectors in parallel — [pagmo JOSS paper](https://joss.theoj.org/papers/10.21105/joss.02338.pdf); [pagmo2 issue #135](https://github.com/esa/pagmo2/issues/135).

### Inferences
- **Concrete recommendation for the Cantera regime:**
  1. Keep pymoo, and use an `ElementwiseProblem` that calls the flame solver.
  2. Attach an elementwise runner built on a `ProcessPoolExecutor(max_workers=n_physical_cores)` with `initializer=` that builds the `ct.Solution` and flame object once per worker (see Q6).
  3. Set the population size to a multiple of the core count to minimize idle tail time.
  4. Also cache by parameter vector, and warm-start flames from a stored solution, to cut runtime variance.
- If runtimes vary more than about 2–3x, consider an asynchronous ask/tell loop built from `concurrent.futures.as_completed` around an ask/tell optimizer such as pycma or nevergrad (both expose ask/tell). This reportedly yields 2–14x less wall-clock idle depending on runtime variance (task-specific numbers), at the cost of algorithmic deviation and bias toward fast evaluations.
- **Speedup expectation:**
  - Near-linear in physical cores on a workstation (e.g. 8–16x on a 16-core PC), since dispatch overhead is under 0.1% of a seconds-long evaluation.
  - On HPC, Dask-jobqueue or `mpi4py.futures` across nodes scale to hundreds of workers.
  - Ray also works on Linux clusters but adds a dependency and daemon for no performance benefit in this regime.
- Island models (pygmo archipelago) are a natural fit for HPC: each island is a process, migration is rare, and runtimes are averaged out. However, pygmo is a different optimizer stack from pymoo.

### Gaps
- No primary Ray number for absolute per-task overhead was found beyond the docs' example.
- I could not confirm the current pymoo class names (`StarmapParallelization`, `DaskParallelization`) from the official docs this session.
- I found no published benchmark of Ray vs loky vs Dask vs MPI for seconds-long tasks. I believe none of them matters at that granularity.
- I did not research Dask-jobqueue or mpi4py version specifics.

## Q6. Cantera specifics: thread safety, multiprocessing best practice, Cantera 3.x Python API

### Takeaway
Cantera objects cannot be pickled or shared across processes. The official 3.1/3.2 pattern is a multiprocessing `Pool` whose initializer creates one Cantera object per worker, stored in a module-level global. Thread safety of concurrent `ThermoPhase`/flame use is not documented in anything I found, so use processes, not threads. Cantera 3.1 supports Python 3.8–3.13. Cantera 3.2 adds Python 3.14 and an "unstable" free-threaded wheel.

### Cited Findings
- **Official example "Parallelizing transport property calculations" (3.1 and 3.2):**
  - Cantera objects cannot be passed between Python processes, so each process needs its own copy.
  - Each worker's objects are created once in a pool initializer and kept in a module-level dict.
  - Requires cantera ≥ 2.5.0. In 3.x the transport model is named `'multicomponent'` (2.x used `'Multi'`) — [Cantera 3.2 example](https://cantera.org/3.2/examples/python/transport/multiprocessing_viscosity.html); [Cantera 3.1 example](https://cantera.org/3.1/examples/python/transport/multiprocessing_viscosity.html).
- **Cantera 3.1.0 release notes:**
  - Locks that prevent adding species to `ThermoPhase` objects that are in use were moved from Python to C++. This concerns species addition, not general thread safety — [Cantera 3.1 release notes](https://cantera.org/dev/reference/releasenotes/v3.1.html).
  - The dev docs are labelled "Cantera 4.0.0a1", so a 4.0 release is in development.
- On Windows, Cantera PyPI wheels use a **single-threaded LAPACK** — [Cantera 3.1 pip docs](https://cantera.org/3.1/install/pip.html).
- Cantera 3.2.0 has a cp314t wheel marked "Free Threading :: 1 - Unstable" — [cantera on PyPI](https://pypi.org/project/cantera/).

### Inferences
- **Best practice for the customized 1D flame:**
  1. Build `gas = ct.Solution(mech)` and `flame = ct.FreeFlame(gas, width=...)` once per worker in the initializer.
  2. In each task, set the parameters (e.g. modified rate coefficients via `gas.set_multiplier` or modifying reactions), restore a reference initial solution, solve, and return scalars.
  3. Return only numbers (flame speed, residual, success flag), never Cantera objects.
- Use one process per physical core. Cantera's 1D solver is essentially single-threaded, and Windows wheels use single-threaded LAPACK, so per-process threading will not oversubscribe.
- Handle solver failures inside the worker with try/except, returning a penalty value. A crashed worker kills a `ProcessPoolExecutor` (`BrokenProcessPool`), and joblib/loky recovers more gracefully.
- Set a timeout per evaluation, or use `concurrent.futures` with `wait(timeout)`, because stiff cases can run for minutes.
- The user's customized C++ solver build must match the Cantera version and Python ABI. Pin Cantera 3.1 (Python ≤3.13) or 3.2 (adds 3.14) and avoid the free-threaded build.

### Gaps
- I found no authoritative statement on whether separate `Solution` objects can safely be used concurrently from multiple threads in one process. The Cantera C++ docs and GitHub issues would need checking.
- Cantera 3.2 release date and notes were not retrieved.

## Q7. Alternative optimizer libraries with fast cores (pycma, nevergrad, scipy DE, PyGMO/pagmo2, EvoX GPU)

### Takeaway
For a benchmark whose subject is a parametrization (SLUD) rather than the optimizer, it is better to keep pymoo PSO and vectorize than to switch libraries. Library choice mostly matters for the expensive regime:
- **Ask/tell libraries** (pycma, nevergrad) enable asynchronous evaluation.
- **pagmo2** offers C++ algorithms plus islands with `mp_bfe`.
- **scipy DE** offers trivial `workers=` parallelism.
- **EvoX/evosax** pay off only at large scale on GPU.

### Cited Findings
- **scipy DE:** `workers=-1` or a map-like callable (forces deferred updating), or `vectorized=True` for population-at-once calls — [SciPy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html).
- **pygmo:** asynchronous islands, plus `mp_bfe`/`ipyparallel_bfe` batch fitness evaluators for Python-defined problems (since 2.13) — [pagmo JOSS paper](https://joss.theoj.org/papers/10.21105/joss.02338.pdf); [pygmo docs](https://pygmo.readthedocs.io/).
- **EvoX:** CPU is competitive at small scale, and the GPU gives ≥10x at large scale (paper). The README's "100x" is a claim — [EvoX arXiv 2301.12457](https://arxiv.org/pdf/2301.12457); [EvoX GitHub](https://github.com/EMI-Group/evox).
- **evosax:** vmap or pmap over seeds — [evosax arXiv 2212.04180](https://arxiv.org/abs/2212.04180).
- **EvoTorch:** PyTorch with GPU and Ray parallelism, v0.6.1 (May 2025) — [EvoTorch arXiv](https://arxiv.org/pdf/2302.12600v1); [pepy](https://pepy.tech/project/evotorch).

### Inferences
- pycma (`cma.CMAEvolutionStrategy.ask()/tell()`) and nevergrad (`optimizer.ask()/tell()`, with `num_workers` and executor support) are the standard choices when evaluations are expensive and an asynchronous loop is wanted. This is from background knowledge; I did not retrieve or verify their docs this session.
- The pure-Python overhead of pycma and nevergrad per iteration is in the range of milliseconds, which is irrelevant for Cantera but noticeable for cheap functions (not measured).
- Switching to pagmo2's C++ PSO would remove pymoo's per-generation Python overhead for the cheap benchmark. However, it changes the algorithm implementation being benchmarked, and pygmo installs most easily via conda-forge (background knowledge).

### Gaps
- I did not retrieve current version numbers for pycma, nevergrad, pygmo, evosax or EvoX.
- I found no published per-generation overhead comparison of pymoo vs pycma vs nevergrad vs pagmo on small populations.
