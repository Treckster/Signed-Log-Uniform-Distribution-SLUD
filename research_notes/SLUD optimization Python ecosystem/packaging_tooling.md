# Modern (2026) Python packaging, environment and project tooling for SLUD

Research method note: research date 2026-10-21. In this session, WebFetch could not resolve several documentation hosts (docs.astral.sh, cantera.org, learn.scientific-python.org, joss.readthedocs.io): DNS failure. So wherever possible, primary sources were read from the projects' own GitHub repositories (raw.githubusercontent.com), which hold the source of the same docs. Those are cited by their GitHub URL, and the matching public docs URL is given where it is known. Other items come from search-result summaries and are marked as such. Anything taken from background knowledge that could not be verified this session is listed under Gaps, not under Cited Findings.

Repo context observed (`git ls-files`): loose scripts at the root (`SignedUniLog.py`, `SignedUniLogMANYRUNS.py`, `UniLog(DEPRECATED).py`), a committed `__pycache__/*.pyc`, `.vscode/` and `*.code-workspace` files, a `LATEX/` folder that includes build artifacts (`*.aux`, `*.bbl`, `*.log`, `*.fdb_latexmk`, `*.fls`, `*.synctex.gz`, a vim `.swp` file, several PDFs, a .docx), and benchmark CSVs under `Stats/<problem>/{LIN,LUD,SLUD}.csv`, which are duplicated under `bkup/`.

## 1. Environment / dependency managers (uv, pixi, conda/mamba, Poetry, Hatch, PDM) and Cantera availability

### Takeaway
Use **uv** as the default project and environment manager. SLUD's own runtime dependencies (numpy, pymoo, optionally Ray) are all on PyPI, and Cantera now ships PyPI wheels for Windows, macOS and Linux (3.2.0, Python 3.10–3.14). Add **pixi** only for the team's environment that builds the *customized Cantera C++ solver*. That build needs conda-forge compilers, BLAS/LAPACK, Sundials and similar libraries, and on Windows the conda build of Cantera has multithreaded LAPACK while the PyPI wheel does not.

### Cited Findings
- uv release cadence in 2026: 0.10.0 (early Feb 2026, stabilized preview features such as Python upgrade commands); 0.11.x from March; 0.12.0 on 2026-07-28 (breaking-change release: "projects created with `uv init` now declare a build system and are packaged by default"); 0.12.1 on 2026-07-31. Taken from a search-result summary of third-party release mirrors — [newreleases.io uv 0.11.14](https://newreleases.io/project/pypi/uv/release/0.11.14), [GitLab MR bumping uv_build 0.11.31](https://gitlab.com/sylva-projects/sylva-elements/ci-tooling/sylva-test-framework/-/merge_requests/64). The authoritative source is https://github.com/astral-sh/uv/blob/main/CHANGELOG.md.
- uv's current docs on `main` show `requires = ["uv_build>=0.13.5,<0.14"]`, which implies uv 0.13.x is current as of Oct 2026 — [uv docs: build backend (GitHub source)](https://raw.githubusercontent.com/astral-sh/uv/main/docs/concepts/build-backend.md) (published at https://docs.astral.sh/uv/concepts/build-backend/). This does not conflict with the 0.12 dates above; it is just a later release.
- uv supports PEP 723 inline script metadata:
  - The block is TOML inside comments, from `# /// script` to `# ///`, with `requires-python` and `dependencies` (required even if empty).
  - `uv add --script example.py 'requests<3' 'rich'` writes the block for you.
  - `[tool.uv] exclude-newer = "<RFC 3339 date>"` in the block limits resolution to distributions released before that date, for reproducibility.
  - `uv lock --script` writes `example.py.lock`.
  - A shebang `#!/usr/bin/env -S uv run --script` makes the script directly executable.
  — [uv docs: scripts guide (GitHub source)](https://raw.githubusercontent.com/astral-sh/uv/main/docs/guides/scripts.md) (published at https://docs.astral.sh/uv/guides/scripts/)
- Cantera 3.2.0 wheels are on PyPI for Python 3.10–3.14, with install steps for Linux, macOS and Windows — [Cantera docs: Installing with pip (GitHub source)](https://raw.githubusercontent.com/Cantera/cantera/main/doc/sphinx/install/pip.md) (published at https://cantera.org/3.2/install/pip.html)
- On Windows, "the PyPI packages are compiled using a single-threaded implementation of the LAPACK library", so reactor-network and flame simulations cannot use multiple cores. The docs point Windows users who want optimized LAPACK to the Conda package — [Cantera pip install docs](https://raw.githubusercontent.com/Cantera/cantera/main/doc/sphinx/install/pip.md); same text in the [3.1 docs](https://cantera.org/3.1/install/pip.html)
- The Cantera 3.1 install docs "highly recommend that all new users install the Python interface using Conda". Conda packages exist for Windows, macOS and Linux. Taken from a search-result summary — [Cantera 3.1 install index](https://cantera.org/3.1/install/index.html), [PyPI: Cantera](https://pypi.org/project/Cantera). Note: the 3.2 pip page that was read does not repeat a general "use conda" recommendation; it only gives the Windows LAPACK caveat. Check https://cantera.org/stable/install/ for the current wording.
- Building Cantera wheels from source (3.1) needs SCons and `build`; the wheel is built with `python -m build --wheel`, and Linux also needs OpenBLAS — [Cantera 3.1: Distributing sdist and wheel packages via PyPI](https://cantera.org/3.1/develop/distribution-packages/pypi-sdist-wheel.html)
- pixi stores its configuration either in `pixi.toml` or under `[tool.pixi.*]` in `pyproject.toml` (`pixi init --format pyproject`). Conda packages go in `[tool.pixi.dependencies]`, PyPI packages in `[tool.pixi.pypi-dependencies]` (and `[project].dependencies` are treated as PyPI deps). The local package can be installed editable with `{ path = ".", editable = true }`. If a package is listed in both tables, the conda version wins. Conda dependencies are solved first, then the remaining PyPI dependencies. Taken from search summaries of third-party mirrors — [pydevtools handbook: pixi](https://pydevtools.com/handbook/reference/pixi/), [pixi pyproject reference mirror](https://www.mintlify.com/prefix-dev/pixi/reference/pyproject-toml), [pixi conda/pypi concepts mirror](https://www.mintlify.com/prefix-dev/pixi/concepts/conda-pypi)
- pixi is still pre-1.0. The latest releases found are 0.72.0–0.72.2 (July 2026; 0.72.0 added inline package manifests for building source packages), and 0.69.0 (2026-05-20) added `pixi auth login` for prefix.dev. No 1.0 release was found — [newreleases: pixi v0.72.2](https://newreleases.io/project/github/prefix-dev/pixi/release/v0.72.2), [pixi CHANGELOG v0.69.0](https://pixi.prefix.dev/v0.69.0/CHANGELOG/)
- `pixi.lock` records exact versions and checksums for all platforms. Scientific groups such as the UW Scientific Software Engineering Center have adopted pixi for reproducible research software — [prefix.dev blog: pixi for scientists](https://prefix.dev/blog/pixi_for_scientists), [prefix.dev blog: UW SSEC adopted pixi](https://prefix.dev/blog/seeking-reproducible-research-software-how-the-uw-scientific-software-engineering-center-adopted-pixi). pixi was also presented at SciPy 2025 — [SciPy 2025 talk](https://cfp.scipy.org/scipy2025/talk/GDN8PN/)
- pixi's own Python build backend (`pixi-build-python`) is still a preview feature that must be enabled in the workspace config; it uses uv for installation by default — [pixi-build-python docs](https://pixi.prefix.dev/dev/build/backends/pixi-build-python)
- The Scientific Python `cookie` template (companion to the Scientific Python Development Guide) uses uv for fast CI and task running and includes a `dev` dependency group for `uv run` — [scientific-python/cookie README](https://raw.githubusercontent.com/scientific-python/cookie/main/README.md)
- A pyOpenSci guide pull request (#591, merge status unconfirmed) names Hatch as the recommended all-in-one tool for package developers, presents Hatch together with uv as an installer option, and suggests venv for the simplest pure-Python cases — [pyOpenSci python-package-guide PR #591](https://github.com/pyOpenSci/python-package-guide/pull/591.diff)

### Inferences
- **Recommended split for this team:**
  1. **SLUD package (pure Python core):** use uv. Commit `pyproject.toml` and `uv.lock`. Team workflow: `uv sync`, `uv run pytest`, `uv run python benchmarks/run.py`. uv also installs Python itself (`uv python install 3.12`), which removes the "which Python?" problem on Windows workstations.
  2. **Combustion/Cantera work that needs the *customized* Cantera C++ build:** use a pixi environment (or a conda-forge `environment.yml` for people who already know conda). Get the compilers, sundials, eigen, fmt, yaml-cpp, boost and BLAS from conda-forge. Ordinary users who only need stock Cantera can stay on uv/pip with the PyPI wheel, except on Windows when multicore LAPACK matters.
  3. **One-off experiment scripts:** use PEP 723 inline metadata with `uv run script.py` and `exclude-newer`. This gives reproducible single-file runs without touching the package.
- Poetry and PDM add no value here over uv. They have smaller mindshare in the 2026 scientific Python community, and Poetry historically lagged on PEP 621 support. Hatch is a valid alternative (pyOpenSci leans that way), but uv is simpler to explain: one tool for Python, venv, lock, run and build. Conda/mamba remains fine, but pixi gives conda-forge packages *with a lockfile and tasks*, which plain conda lacks.
- The PEP 621 `[project]` table keeps the package portable: whether a teammate uses uv, pixi, pip or Hatch, the same `pyproject.toml` works.

Minimal uv-centric `pyproject.toml` fragment (illustrative; versions follow the cited docs):
```toml
[project]
name = "slud"
dynamic = ["version"]            # or version = "0.1.0"
description = "Signed Log-Uniform Distribution transforms for black-box optimizers"
readme = "README.md"
license = "MIT"                  # PEP 639 SPDX string; check backend support
requires-python = ">=3.11"
dependencies = ["numpy>=1.26"]

[project.optional-dependencies]
pymoo = ["pymoo>=0.6.1"]
ray = ["ray[default]"]
cantera = ["cantera>=3.1"]

[dependency-groups]              # PEP 735; used by `uv sync`/`uv run`
dev = ["pytest>=8", "hypothesis", "ruff", "pre-commit"]
docs = ["mkdocs-material", "mkdocstrings[python]"]

[build-system]
requires = ["hatchling", "hatch-vcs"]   # or uv_build>=0.13.5,<0.14 if a static version is fine
build-backend = "hatchling.build"
```

Optional pixi overlay (same file) for the Cantera-from-conda-forge environment (illustrative syntax; verify against https://pixi.sh):
```toml
[tool.pixi.workspace]
channels = ["conda-forge"]
platforms = ["linux-64", "win-64", "osx-arm64"]

[tool.pixi.dependencies]
python = "3.12.*"
cantera = ">=3.1"
cxx-compiler = "*"
cmake = "*"
ninja = "*"

[tool.pixi.pypi-dependencies]
slud = { path = ".", editable = true }

[tool.pixi.tasks]
test = "pytest"
```

### Gaps
- Could not open docs.astral.sh or pixi.sh directly (DNS failure in the fetch tool), so the exact feature list and stability status of uv workspaces and of the current pixi pyproject keys (`[tool.pixi.workspace]` vs the older `[tool.pixi.project]`) were not verified first-hand.
- Whether the Cantera PyPI wheel ships C++ headers and a library you can link a customized solver against was not confirmed; the pip page does not say. Assume it does not, and that the C++ development route is conda-forge (`libcantera-devel`) or a source build. Confirm at https://cantera.org/stable/install/.
- No usage survey on "which env manager scientific projects use in 2026" was found. The claims about uv/pixi adoption rest on template defaults (scientific-python/cookie) and blog case studies, not on statistics.
- Ray on Windows: Ray's Windows support has historically been labelled beta. This was not checked this session and is relevant if teammates run SLUD+Ray on Windows.

## 2. Build backends (hatchling, uv_build, scikit-build-core, meson-python, maturin, setuptools+Cython; nanobind/pybind11; f2py; cibuildwheel)

### Takeaway
Start **pure Python** with **hatchling** (or `uv_build`). The SLUD transform is a vectorised NumPy map, and a compiled extension is unlikely to be needed. If a compiled kernel is ever justified, use **scikit-build-core + nanobind** (CMake, which C++ engineers already know), or **meson-python** if Fortran/f2py is involved. Build wheels with **cibuildwheel**. Avoid new setuptools/Cython `setup.py` builds, although pymoo and pycma still use setuptools.

### Cited Findings
- The Scientific Python cookie template offers ten backends: hatchling, uv_build, flit, pdm, poetry, setuptools, pybind11 (setuptools), scikit-build-core (CMake + pybind11), meson-python (Meson + pybind11, no VCS versioning) and maturin (Rust, no VCS versioning). It marks Hatch as "Recommended for pure Python projects", scikit-build as "Recommended for C++ projects" and maturin as "Recommended for Rust projects", and says "Currently, the best choice is probably hatch for pure Python projects." For the C++ backends it uses cibuildwheel to "build wheels for all platforms" — [scientific-python/cookie README](https://raw.githubusercontent.com/scientific-python/cookie/main/README.md)
- `uv_build` "currently only supports pure Python code"; extension modules need another backend, and hatchling is suggested for build scripts or more flexible layouts. It expects `src/<package_name>/__init__.py` by default (the name is normalised, e.g. `Foo-Bar` → `foo_bar`), which you can override with `[tool.uv.build-backend] module-name/module-root`. It follows uv's versioning, so upper-bound it: `requires = ["uv_build>=0.13.5,<0.14"]` — [uv docs: build backend](https://raw.githubusercontent.com/astral-sh/uv/main/docs/concepts/build-backend.md)
- Since July 2025, `uv init` defaults to `uv_build` (it previously used hatchling) — [pydevtools: why uv used hatch as a backend](https://pydevtools.com/handbook/explanation/why-does-uv-use-hatch-as-a-backend/)
- scikit-build-core is a CMake-driven PEP 517 backend that works with pybind11, nanobind, Cython, SWIG and Fortran (via f2py-cmake) — [scikit-build-core: Getting started](https://scikit-build-core.readthedocs.io/en/stable/guide/getting_started.html), [SciPy 2024 proceedings paper on scikit-build-core](https://proceedings.scipy.org/articles/FMKR8387) (search-result summary)
- The canonical nanobind + scikit-build-core example (`wjakob/nanobind_example`) contains:
  - `[build-system] requires = ["scikit-build-core >=0.10", "nanobind >=1.3.2"]`, `build-backend = "scikit_build_core.build"`
  - `[tool.scikit-build] minimum-version = "build-system.requires"`, `build-dir = "build/{wheel_tag}"`, `wheel.py-api = "cp312"` (stable-ABI wheels for CPython ≥3.12)
  - `[tool.cibuildwheel]` that runs `pytest {project}/tests` on each built wheel, and sets `MACOSX_DEPLOYMENT_TARGET = "10.14"` for C++17
  — [nanobind_example pyproject.toml](https://raw.githubusercontent.com/wjakob/nanobind_example/master/pyproject.toml)
- NumPy's docs present meson-python as the way to distribute F2PY-wrapped Fortran. The `pyproject.toml` lists meson-python and NumPy (≥2.0, required by `dependency('numpy')` in `meson.build`) as build requirements with `build-backend = "mesonpy"`. The build needs C and Fortran compilers (gfortran/ifx) and Python ≥3.10. `meson.build` runs f2py through a `custom_target`, and `f2py --build-dir` can generate a skeleton Meson project — [NumPy: Distributing F2PY extensions with meson-python](https://numpy.org/devdocs/f2py/buildtools/meson-python.html), [NumPy 2.4: F2PY using via meson](https://numpy.org/doc/2.4/f2py/buildtools/meson.html) (search-result summary)
- pymoo (an optimizer library SLUD already depends on) still builds with `setuptools.build_meta`, with `requires = ["setuptools>=77", "numpy>=2.0.0", "Cython>=0.29"]`. It builds against NumPy 2.x because the 2.x C-ABI is backward compatible: one wheel loads on NumPy 1.19+ and 2.x, and an unpinned build risks falling back to slow pure-Python code. It also declares `requires-python = ">= 3.10"` (classifiers 3.10–3.14), and extras `parallelization = [joblib, dask[distributed], ray[default]]`, `others`, `visualization`, `full` — [pymoo pyproject.toml](https://raw.githubusercontent.com/anyoptimization/pymoo/main/pyproject.toml)
- pycma (the `cma` package) is pure Python on `setuptools>=70.1` with a dynamic version from `cma.__version__`, depends only on `numpy`, has optional extras (`plotting`, `constrained-solution-tracking`), and lints with ruff — [pycma pyproject.toml](https://raw.githubusercontent.com/CMA-ES/pycma/master/pyproject.toml)

### Inferences
- For SLUD, a compiled extension is almost certainly unnecessary. The transform (sign × log-uniform mapping) is an elementwise NumPy expression that already runs at C speed. The dominant cost is objective evaluations (Cantera simulations), not the transform. Keep the package pure Python with hatchling, so there are no wheels per platform, no cibuildwheel and no compilers for users.
- If the team later wants to expose its customized Cantera C++ code as a Python module, that should be a **separate package** (its own repo or a workspace member) with scikit-build-core + nanobind (or pybind11). CMake matches what C++ engineers know, and nanobind gives smaller, faster bindings with stable-ABI wheels (`wheel.py-api = "cp312"`), which cuts the wheel matrix. Use meson-python only if Fortran or f2py is central. Cython remains fine for small numeric kernels and builds under both scikit-build-core and meson-python, so there is no need for `setup.py`.
- Choosing hatchling over uv_build: hatchling with `hatch-vcs` gives versions from git tags, which suits a Zenodo-per-release workflow. uv_build needs a static version string. Both are acceptable; hatchling is the more conservative and widely documented choice.

Minimal compiled layout (only if ever needed):
```toml
[build-system]
requires = ["scikit-build-core>=0.10", "nanobind>=2"]
build-backend = "scikit_build_core.build"

[tool.scikit-build]
minimum-version = "build-system.requires"
wheel.py-api = "cp312"

[tool.cibuildwheel]
test-requires = ["pytest"]
test-command = "pytest {project}/tests"
```
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(_core STABLE_ABI src/core.cpp)
install(TARGETS _core LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
```
(The CMake snippet follows the nanobind_example pattern from memory. Verify it against https://github.com/wjakob/nanobind_example/blob/master/CMakeLists.txt.)

### Gaps
- The Scientific Python Development Guide's "compiled packaging" page could not be fetched (DNS failure on learn.scientific-python.org; the guessed GitHub source path returned 404), so its exact current wording is not quoted.
- Current exact versions of scikit-build-core, meson-python, nanobind, cibuildwheel, hatchling and ruff as of Oct 2026 were not verified. Only the minimum pins in the cited examples are known.
- Nevergrad's and IOH's build configurations were not checked. Background knowledge: nevergrad is pure Python; IOHexperimenter (`ioh`) is a C++ core with pybind11 bindings and ships wheels. Unverified this session.

## 3. Project layout and dev tooling (src layout, PEP 621, typing, ruff, pytest + hypothesis, pre-commit, CI, docs, versioning)

### Takeaway
Use the layout and tooling the **scientific-python/cookie** template generates: src layout, PEP 621 `pyproject.toml`, ruff (lint and format), pytest, pre-commit, GitHub Actions with trusted PyPI publishing, and a docs folder. Running `uvx sp-repo-review` (repo-review) then audits the repo against the Scientific Python guide automatically.

### Cited Findings
- The cookie template includes:
  - ruff for formatting, linting and autofixes ("Replaces Flake8, isort, pyupgrade, yesqa, pycln, and dozens of plugins")
  - pre-commit (or `prek`), MyPy through pre-commit, and a spell checker
  - a nox file with common targets
  - pytest with a `tests/` folder, plus Codecov upload
  - GitHub Actions with PyPI trusted publishing and Dependabot for actions
  - a ReadTheDocs-ready `docs/` folder with a `docs` dependency group
  - a `dev` dependency group for `uv run`
  - repo-review (`sp-repo-review`), whose checks link back to the guide and which also has a WebAssembly version
  — [scientific-python/cookie README](https://raw.githubusercontent.com/scientific-python/cookie/main/README.md)
- uv_build's default discovery is `src/<package_name>/__init__.py`, i.e. src layout is the default convention — [uv docs: build backend](https://raw.githubusercontent.com/astral-sh/uv/main/docs/concepts/build-backend.md)
- pycma configures ruff in `pyproject.toml` (`[tool.ruff.lint]` and per-file ignores), which shows that even long-standing optimization packages have moved to ruff — [pycma pyproject.toml](https://raw.githubusercontent.com/CMA-ES/pycma/master/pyproject.toml)
- SPEC 0 (Scientific Python Ecosystem Coordination): drop support for a Python version 3 years after its initial release, and for core packages (NumPy, SciPy, etc.) 2 years after release. This applies to feature releases, and the drop happens in the next release in the given quarter — [SPEC 0 source](https://raw.githubusercontent.com/scientific-python/specs/main/spec-0000/index.md) (published at https://scientific-python.org/specs/spec-0000/)

### Inferences
Recommended repo layout for SLUD:
```
slud/
├─ pyproject.toml          # PEP 621 metadata, deps, ruff/pytest config
├─ uv.lock                 # committed
├─ README.md               # what/why/install/quickstart/citation
├─ LICENSE                 # OSI license (MIT/BSD-3/Apache-2.0)
├─ CITATION.cff
├─ CONTRIBUTING.md         # JOSS needs community guidelines
├─ CHANGELOG.md
├─ .gitignore  .pre-commit-config.yaml
├─ .github/workflows/{ci.yml,release.yml}
├─ src/slud/
│   ├─ __init__.py         # public API: SLUD transform, inverse, sampler
│   ├─ transform.py        # pure functions, NumPy-vectorised, type-hinted
│   ├─ pymoo.py            # optional adapter (import pymoo lazily)
│   └─ py.typed
├─ tests/                  # pytest + hypothesis
├─ benchmarks/             # scripts/configs that produced paper results (Ray lives here, not in core)
├─ examples/ or notebooks/ # marimo .py notebooks
├─ docs/                   # MkDocs or Sphinx
└─ paper/                  # paper.md + paper.bib for JOSS (NOT the COBEM LaTeX)
```
- **Typing:** type-hint the public API with `numpy.typing.NDArray[np.float64]` and ship `py.typed`. Run mypy (or pyright) from pre-commit as the cookie template does. Keep it non-strict at first so non-specialists are not blocked.
- **ruff:** a single `[tool.ruff.lint] select = ["E","F","I","B","UP","NPY","PT"]` and `ruff format`. That one tool replaces black, isort and flake8.
- **pytest + hypothesis:** the transform is a natural fit for property-based tests:
  - round-trip `inverse(forward(x)) ≈ x`
  - monotonicity
  - sign preservation
  - output stays inside bounds
  - the distribution is symmetric under sign flip
  - edge cases at 0, at ±bounds, and at extreme magnitudes such as 1e-300/1e300
  Use `hypothesis.extra.numpy.arrays` with `floats(allow_nan=False)` strategies. Add a statistical test (a KS test of sampled log|x| against uniform) marked `@pytest.mark.slow`.
- **CI matrix:** ubuntu/windows/macos × the oldest and newest supported Python (e.g. 3.11 and 3.14, per SPEC 0), with `astral-sh/setup-uv` and `uv run pytest`. Add a separate release workflow that runs `uv build` and publishes with PyPI trusted publishing on tag.
- **Docs:** for a small package, a README plus one docs site is enough. Use MkDocs-Material + mkdocstrings for simplicity, or Sphinx + MyST-NB / pydata-sphinx-theme if the team wants the numpy/scipy look. Note the MkDocs-ecosystem caveat under Gaps. Host on Read the Docs or GitHub Pages.
- **Versioning:** take versions from git tags (hatch-vcs / setuptools-scm) and use SemVer-ish `0.x` until the API stabilises. Each GitHub release then gets a Zenodo DOI automatically (see section 4).

Minimal `.pre-commit-config.yaml` (the revs are placeholders; pin them with `pre-commit autoupdate`):
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: vX.Y.Z
    hooks: [{id: ruff-check, args: [--fix]}, {id: ruff-format}]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: vX.Y.Z
    hooks: [{id: check-added-large-files}, {id: end-of-file-fixer}, {id: trailing-whitespace}]
```

Minimal CI (`.github/workflows/ci.yml`):
```yaml
on: [push, pull_request]
jobs:
  test:
    strategy:
      matrix: {os: [ubuntu-latest, windows-latest, macos-latest], python: ["3.11", "3.14"]}
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with: {python-version: "${{ matrix.python }}"}
      - run: uv run --group dev pytest
```

### Gaps
- MkDocs ecosystem status in 2026 was not verified this session. From background knowledge, unverified: in late 2025 the Material for MkDocs maintainers announced a successor project (Zensical) and a move of Material for MkDocs toward maintenance mode, and MkDocs 1.x core maintenance has been a concern. Check squidfunk/mkdocs-material announcements before committing to it. Sphinx + MyST + pydata-sphinx-theme is the conservative choice used by numpy, scipy and the Scientific Python projects.
- Exact current versions of the `astral-sh/setup-uv` and `actions/checkout` actions were not checked. The tags above are illustrative.
- Jupyter Book 2 (MyST-MD based) status was not researched.

## 4. Research-software publication (JOSS, CITATION.cff, Zenodo, Software Heritage, pyOpenSci, Scientific Python guide/repo-review, SPECs)

### Takeaway
SLUD can become citable **immediately**: add `CITATION.cff`, connect the GitHub repo to Zenodo, and cut a release to get a DOI. Software Heritage gives a permanent SWHID as well. **JOSS is not available yet**: it now requires the repository to have been **public for more than 6 months with sustained development** before submission. Also, a transform whose novelty is mainly the method (already published at COBEM 2025) has to show "substantial scholarly effort" in the *software* itself. pyOpenSci review is an optional route that fast-tracks JOSS acceptance.

### Cited Findings
- JOSS requirements (current `submitting.md` on JOSS's GitHub):
  - an OSI-approved license, with the source browsable and issues openable without approval or payment
  - a short `paper.md` with title, summary, authors, affiliations and key references
  - the implementation must reflect "a substantial scientific effort", and the paper must not focus on new research results produced with the software
  - well-documented software that reviewers can test locally
  - "procedures (such as automated tests) for checking correctness"
  - community guidelines (a CONTRIBUTING file and stated support/governance expectations)
  — [JOSS submitting.md (GitHub)](https://raw.githubusercontent.com/openjournals/joss/main/docs/submitting.md) (published at https://joss.readthedocs.io/en/latest/submitting.html)
- JOSS AI policy: generative AI use is allowed but must be disclosed in an "AI usage disclosure" statement. The statement covers tools and versions, where AI was used, the nature of the assistance, and confirmation that humans reviewed the outputs and made the core design decisions. AI may not be used in conversations with editors or reviewers, except for translation — [JOSS submitting.md](https://raw.githubusercontent.com/openjournals/joss/main/docs/submitting.md)
- JOSS development-history rule: the repo must have been public for more than six months before submission, with active development across that period. A repository developed in a short burst, or dumped publicly just before submission, will not be accepted — [JOSS submitting.md](https://raw.githubusercontent.com/openjournals/joss/main/docs/submitting.md). The review checklist likewise asks for evidence of "sustained development over time … rather than rapid, recent code generation", including releases, public issues/PRs and ideally external engagement — [JOSS review checklist](https://joss.readthedocs.io/en/latest/review_checklist.html) (search-result summary; JOSS also hosts a [pre-2026-submission](https://joss.readthedocs.io/en/pre-2026-submission/review_checklist.html) version of the docs, so the criteria changed in 2026)
- More of the JOSS reviewer checklist: a plain-text OSI license file; the submitting author made major contributions as shown by commit history; installation works as documented; functional and performance claims are confirmed; a statement of need naming the target audience; clear dependency lists. Outcomes are Accept / Minor / Major revisions, with no outright rejection for major revisions — [JOSS review checklist](https://joss.readthedocs.io/en/latest/review_checklist.html) (search-result summary)
- pyOpenSci: packages accepted through pyOpenSci peer review that are in JOSS scope are fast-tracked through JOSS. JOSS accepts the pyOpenSci review and reviews only the paper. Reviews run as public GitHub issues in `pyOpenSci/software-submission` — [pyOpenSci software-submission](https://github.com/pyOpenSci/software-submission); [pyOpenSci 2018 JOSS collaboration notes](https://pyopensci-governance.readthedocs.io/en/latest/reference/meeting-notes/2018/2018-10-09-notes.html) (search-result summaries; the current fast-track terms were not confirmed)
- Zenodo–GitHub integration: each GitHub release is archived with its own version DOI, plus a "concept DOI" that always resolves to the latest version. Zenodo reads `CITATION.cff` (or `.zenodo.json`) for metadata and falls back to repository metadata without it — [CASRAI: DataCite, GitHub, Zenodo software-citation stack](https://www.casrai.org/news/datacite-github-zenodo-software-citation-stack) (secondary/opinion source), example of CITATION.cff + .zenodo.json in practice: [github/spec-kit PR #2291](https://github.com/github/spec-kit/pull/2291)
- Software Heritage archives repositories and issues SWHIDs, which are content-based identifiers not tied to any platform — [ZBW Open Economics: making open code citable](https://openeconomics.zbw.eu/en/knowledgebase/making-open-code-citable)
- The Scientific Python guide has a companion template (`scientific-python/cookie`) and an automated checker (`sp-repo-review`) — [cookie README](https://raw.githubusercontent.com/scientific-python/cookie/main/README.md). SPEC 0 sets dependency/Python support windows — [SPEC 0](https://raw.githubusercontent.com/scientific-python/specs/main/spec-0000/index.md)

### Inferences
- **Timeline implication:** make the repo public (or keep it public) and develop it visibly, with tagged releases, issues and PRs, from now on. The earliest realistic JOSS submission is about 6 months after the package-ization work becomes public. The JOSS paper must frame SLUD as reusable *software* for optimizer users: pymoo/cma/scipy adapters, docs, tests. It must not present COBEM benchmark results as new findings.
- **Do today:** add `CITATION.cff` with a `preferred-citation` pointing to the COBEM 2025 paper (COB-2025-2080) once it is published, plus the software entry itself. Enable the Zenodo GitHub app and release `v0.1.0` to get a DOI badge for the README. Optionally trigger a Software Heritage "Save code now".
- An optional pyOpenSci submission later gives a thorough packaging review from domain-neutral reviewers, plus a JOSS fast track. That is valuable for a team of non-software-specialists.

Minimal `CITATION.cff`:
```yaml
cff-version: 1.2.0
message: "If you use SLUD, please cite it as below."
title: "SLUD: Signed Log-Uniform Distribution transforms for optimizers"
type: software
authors:
  - family-names: "Rezende"
    given-names: "Estevan"
    # orcid: "https://orcid.org/..."
license: MIT
repository-code: "https://github.com/<org>/slud"
version: 0.1.0
date-released: 2026-11-01
# doi: 10.5281/zenodo.XXXXXXX   # concept DOI after first Zenodo release
preferred-citation:
  type: conference-paper
  title: "<COBEM 2025 paper title>"
  authors: [{family-names: "...", given-names: "..."}]
  year: 2025
  conference: {name: "COBEM 2025"}
```
(Author details are placeholders inferred from the repo and user context. Fill them in from the actual paper.)

### Gaps
- The full current (2026) JOSS review checklist and the exact "substantial scholarly effort" thresholds (historically ~3 months of work, plus code-size heuristics) could not be read directly. JOSS's docs site was unreachable and only search summaries were available. The 2026 doc revision suggests the criteria were reworked.
- The current pyOpenSci fast-track terms and scope (e.g. whether optimization/numerical-methods packages are in scope) were not verified from the current pyOpenSci guidebook.
- Official Zenodo and Software Heritage documentation pages were not fetched. The Zenodo flow is well established, but the citations here are secondary.

## 5. Data/results storage, separating the paper from code, LaTeX .gitignore, Git LFS vs Zenodo

### Takeaway
Keep raw benchmark outputs **out of the package repo's history** (or keep only small, final CSV summaries). Archive full benchmark datasets on **Zenodo** as a versioned dataset with its own DOI, not Git LFS. Move the COBEM LaTeX into its own repo or Overleaf project and never commit LaTeX build artifacts. Use Parquet for large or many-run result tables and CSV for small human-readable summaries.

### Cited Findings
- Zenodo issues per-version and concept DOIs per archive and reads CITATION metadata — [CASRAI software-citation stack](https://www.casrai.org/news/datacite-github-zenodo-software-citation-stack) (secondary source). This applies equally to dataset uploads.
- JOSS papers must not focus on new research results obtained with the software — [JOSS submitting.md](https://raw.githubusercontent.com/openjournals/joss/main/docs/submitting.md). This supports keeping paper and results artifacts separate from the software repo.
- The cookie template's pre-commit setup is a natural home for hooks such as `check-added-large-files` — [scientific-python/cookie README](https://raw.githubusercontent.com/scientific-python/cookie/main/README.md) (that README describes pre-commit in general; the specific hook is a standard pre-commit-hooks hook, not verified this session)

### Inferences
- **The current repo has concrete problems:**
  - committed LaTeX build products (`*.aux *.bbl *.blg *.log *.fls *.fdb_latexmk *.synctex.gz`, `.swp`, several PDFs) in `LATEX/`
  - committed `__pycache__/`
  - `bkup/` duplicating `Stats/`
  - editor files (`.vscode/`, `*.code-workspace`)

  Recommended actions:
  1. Move `LATEX/` to a separate `slud-cobem2025-paper` repo or Overleaf (Overleaf can sync to GitHub). Keep only `paper/paper.md` and `paper.bib` for JOSS in the code repo.
  2. Delete `bkup/` and `__pycache__/` from the index.
  3. Keep `Stats/` only if it is small; otherwise publish it to Zenodo and download it on demand in benchmark scripts (e.g. with `pooch`).
  4. Optionally rewrite history with `git filter-repo` before the public 0.1 release to purge large PDFs. Weigh this against JOSS's "visible development history" requirement: rewriting is fine, but history must remain.
- **Git LFS vs Zenodo:** GitHub LFS has storage and bandwidth quotas and makes clones and CI more complex, and it gives no DOI. Zenodo gives a DOI, versioning and long-term preservation, which suits "data behind a paper". Recommend Zenodo.
- **CSV vs Parquet:** for per-run optimizer traces (many runs × generations × metrics), Parquet (via pandas/pyarrow or polars) is columnar, typed, compressed, and keeps dtypes such as float64 exactly. Keep CSV for the small aggregated tables that appear in the paper, because they are readable in any tool. Store run metadata (seed, algorithm, SLUD parameters, package versions from `importlib.metadata`) alongside the data, for example as a sidecar JSON or Parquet key-value metadata.

`.gitignore` additions (the LaTeX entries follow GitHub's TeX.gitignore conventions):
```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
build/
dist/
*.egg-info/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.hypothesis/
# LaTeX build artifacts
*.aux
*.bbl
*.blg
*.log
*.fls
*.fdb_latexmk
*.synctex.gz
*.out
*.toc
*.bcf
*.run.xml
*-eps-converted-to.pdf
# Editors
.vscode/
*.code-workspace
*.swp
# Results (publish to Zenodo instead)
results/raw/
```

### Gaps
- GitHub's TeX.gitignore and Git LFS quota pages were not fetched this session. The LaTeX patterns and the LFS-quota remark come from background knowledge.
- No source was fetched comparing CSV and Parquet specifically for optimization benchmarks. The recommendation is reasoned, not cited.

## 6. Notebook vs script workflows; marimo

### Takeaway
Keep reusable logic in `src/slud`, and drive experiments from **scripts** (PEP 723 or `benchmarks/` modules run with `uv run`). Use **marimo** instead of Jupyter for exploratory or demo notebooks, because marimo notebooks are plain `.py` files: git-diffable, runnable as scripts, testable with pytest, and free of hidden state.

### Cited Findings
- marimo is an open-source reactive Python notebook:
  - notebooks are stored as pure Python, can run as scripts and can be deployed as apps
  - the reactive runtime re-runs or marks stale the dependent cells, which removes hidden state
  - it was developed with computational scientists at Stanford's SLAC
  — [marimo GitHub](https://github.com/marimo-team/marimo), [marimo blog: SLAC](https://marimo.io/blog/slac-marimo)
- marimo notebooks are testable with pytest, reusable as modules, and can be shared as interactive web apps or run in the browser via WASM — [marimo for researchers](https://marimo.io/for-researchers)
- A SLAC-authored article cited by marimo notes that fewer than 4% of notebooks on GitHub were reproducible — [marimo blog: SLAC](https://marimo.io/blog/slac-marimo) (secondary citation of an external study)
- marimo was presented at SciPy 2025 — [SciPy 2025 talk](https://cfp.scipy.org/scipy2025/talk/CG8P37/)
- uv's PEP 723 support (`uv run script.py`, `uv lock --script`, `exclude-newer`) makes standalone experiment scripts reproducible — [uv scripts guide](https://raw.githubusercontent.com/astral-sh/uv/main/docs/guides/scripts.md)

### Inferences
- Structure for SLUD:
  - `benchmarks/run_suite.py` is the CLI that runs pymoo + Ray experiments and writes Parquet. It replaces `SignedUniLogMANYRUNS.py`.
  - `benchmarks/make_figures.py` regenerates the paper figures from stored results.
  - `examples/slud_demo.py` is a marimo notebook showing the distributions (the `dists_example.png` figures). It can be embedded in the docs.
- marimo notebooks can carry PEP 723 headers (marimo supports `--sandbox` with uv). This was not verified this session; see Gaps. With it, a teammate runs `uvx marimo edit --sandbox examples/slud_demo.py` without any setup.

### Gaps
- marimo's current version and its `--sandbox`/uv integration details were not verified in this session.

## 7. Well-regarded small optimization/scientific packages to emulate

### Takeaway
**pycma** is the closest analogue: a small, NumPy-only optimizer package with a dynamic version, optional extras and ruff. It still uses setuptools, though; SLUD should copy its *dependency minimalism*, not its backend. **pymoo** shows the right pattern for heavy optional dependencies (Ray and dask as extras). For the overall template, the **scientific-python/cookie** template is the best model, and nanobind_example covers compiled code if it is ever needed.

### Cited Findings
- pycma: `dependencies = ["numpy"]`, extras `plotting = ["matplotlib"]`, `constrained-solution-tracking = ["moarchiving"]`, version from `cma.__version__`, setuptools ≥70.1 backend, ruff lint config. No `requires-python` is declared and the classifiers still list Python 2.7 — [pycma pyproject.toml](https://raw.githubusercontent.com/CMA-ES/pycma/master/pyproject.toml)
- pymoo:
  - Python ≥3.10
  - core deps: numpy, scipy, moocore, autograd, cma, matplotlib, alive_progress, Deprecated
  - Ray is only an optional extra (`parallelization = ["joblib", "dask[distributed]", "ray[default]"]`)
  - Cython build requirement with a NumPy ≥2.0 build pin for ABI compatibility
  — [pymoo pyproject.toml](https://raw.githubusercontent.com/anyoptimization/pymoo/main/pyproject.toml)
- nanobind_example: the canonical minimal scikit-build-core + nanobind + cibuildwheel setup — [nanobind_example pyproject.toml](https://raw.githubusercontent.com/wjakob/nanobind_example/master/pyproject.toml)
- Cantera itself builds with SCons and ships both PyPI wheels and conda-forge packages — [Cantera 3.1 PyPI distribution docs](https://cantera.org/3.1/develop/distribution-packages/pypi-sdist-wheel.html), [Cantera pip docs](https://raw.githubusercontent.com/Cantera/cantera/main/doc/sphinx/install/pip.md)
- scientific-python/cookie: a cookiecutter/copier template implementing all Scientific Python Development Guide recommendations — [cookie README](https://raw.githubusercontent.com/scientific-python/cookie/main/README.md); referenced from the [scikit-build-core docs](https://scikit-build-core.readthedocs.io/en/stable/guide/getting_started.html)

### Inferences
- Mirror pymoo's approach: SLUD core depends only on numpy (and maybe scipy). pymoo, Ray, cma and Cantera are **optional extras** (`pip install slud[pymoo,ray]`), and Ray is imported only inside `benchmarks/` or an optional parallel helper. This keeps installation trivial on Windows workstations.
- Avoid pycma's legacy traits: no `requires-python` and Python 2.7 classifiers. Declare `requires-python = ">=3.11"`, or follow SPEC 0.
- Bootstrap quickly with `uvx copier copy gh:scientific-python/cookie slud --trust` and choose the hatchling backend. Then delete whatever the team does not want (e.g. nox) so it stays simple.

### Gaps
- The packaging and CI of nevergrad, IOHexperimenter (`ioh`) and cocopp were not examined this session (tool-call budget), so no cited claims are made about them.
- No source was found ranking which optimization packages are "well-regarded" for packaging quality; the selection above is judgment.
