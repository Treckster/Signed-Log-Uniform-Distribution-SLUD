# Prior art on log-scale, signed-log and sign-magnitude parametrizations of optimization search spaces, and where SLUD stands

Scope note: written 2026-10-21. The library facts below come from reading source code. I downloaded the current PyPI wheels with `pip download --no-deps`: optuna 5.0.0, nevergrad 1.0.12, hyperopt 0.3.0, scikit-optimize 0.10.2, ConfigSpace 1.2.2, ax-platform 1.3.1, botorch 0.18.1, ray 2.59.0, cma 4.5.0 and scipy 1.18.1. The line numbers refer to those versions. The GitHub links point to each project's repository, and the file path is given in the text. In this session, direct fetches of arxiv.org, hal.science and github.io were blocked (DNS failure or proxy 403). Some literature facts therefore rest on search-engine summaries of the primary source, and those are flagged. A few DOIs come from memory and are marked "(DOI not verified in-session)".

Reference definition of SLUD, from the project code and paper draft `COB-2025-2080.pdf`:
z ∈ [2·log10 lb, 2·log10 ub], m = log10(lb·ub), h = log10(ub/lb), u = |z−m|/h, s = sign(z−m), x = s·lb·(ub/lb)^u.
With t = (z−m)/h ∈ [−1, 1], this is x = sign(t)·lb·(ub/lb)^{|t|}. The map is monotone increasing in z: from −ub, to −lb as z→m⁻, a jump to +lb, then up to +ub. A uniform z gives a 50/50 mixture of a log-uniform on [lb, ub] and its mirror image on [−ub, −lb].

## Q1. Log-uniform handling in mainstream optimization and HPO tools: does any support a signed log-uniform natively?

### Takeaway
None of the ten libraries inspected (Optuna, Nevergrad, Ax, BoTorch, Hyperopt, Ray Tune, scikit-optimize, ConfigSpace/SMAC, scipy.stats, pycma) has a native input or search-space parameter that is log-uniform over both signs. Each one rejects non-positive bounds when log scale is requested, or can only produce positive values. The only signed-log transforms in these packages (Ax `BilogY`, BoTorch `Bilog`, Ax `PowerTransformY` using Yeo–Johnson) act on outputs (objective or constraint values), not on inputs.

### Cited Findings
- **Optuna 5.0.0.** `FloatDistribution.__init__` raises ``ValueError("`low > 0` must hold for `log=True`...")`` when `log and low <= 0.0` (`optuna/distributions.py` l.148–149). `IntDistribution` requires `low >= 1` for `log=True` (l.351). The docs say the same: "If log is True, low must be larger than 0". `step` cannot be combined with `log=True`. — [Optuna distributions source](https://optuna.readthedocs.io/en/stable/_modules/optuna/distributions.html); [FloatDistribution docs](https://optuna.readthedocs.io/en/v4.5.0/reference/generated/optuna.distributions.FloatDistribution.html)
  - Why negative bounds are forbidden: log=True means sampling uniformly in log(x), and log is undefined for x ≤ 0. Optuna offers no sign-aware variant. A user who needs signs must add a separate categorical or float sign parameter, which is the LUD pattern. — same source
- **Nevergrad 1.0.12.**
  - `ng.p.Log` documents `lower: minimum value if any (> 0)`. It is a wrapper over `Scalar` with an `Exponent` data layer (`nevergrad/parametrization/data.py` l.487–545). When bounded, it defaults to `init = sqrt(lower*upper)` (the geometric midpoint, the same as SLUD's m) and `exponent = exp((ln upper − ln lower)/6)`, which puts "99.7% of values within the bounds".
  - The `Exponent` layer raises "Exponent must be strictly positive" for base ≤ 0 (`_datalayers.py` l.181–187). `Scalar.set_mutation(exponent=...)` likewise allows only exponents > 0 (`data.py` l.262–266).
  - The Exponent layer computes x = base^y, so x is always positive. There is no signed-log parameter, and a grep of the package for asinh/symlog/sign-log found nothing.
  - Source: [nevergrad repo, parametrization/data.py and _datalayers.py](https://github.com/facebookresearch/nevergrad). Docs: [parametrization reference](https://facebookresearch.github.io/nevergrad/parametrization_ref.html), which could not be fetched in-session; findings are from source.
- **Ax 1.3.1.** `RangeParameter` raises `UserInputError("... cannot take log when min <= 0.")` when `log_scale and lower <= 0`. `logit_scale` requires `0 < lower` and `upper < 1` (`ax/core/parameter.py` l.441–444). For ChoiceParameter with log scale: "log_scale requires all values to be positive" (l.827). The input transforms in `ax/adapter/transforms/` include `log.py`, `logit.py`, `unit_x.py` and others, but no signed-log input transform. — [Ax repo](https://github.com/facebook/Ax)
- **Ax output-side signed transforms.**
  - `BilogY` applies f(y) = bound + sign(y−bound)·log(|y−bound|+1) to outcome-constraint metrics, citing Eriksson & Poloczek (AISTATS 2021, SCBO) (`ax/adapter/transforms/bilog_y.py` l.38, 136–138).
  - `PowerTransformY` uses sklearn's Yeo–Johnson on outcomes (`power_transform_y.py` l.42, 320).
  - Sources: [Ax repo](https://github.com/facebook/Ax); [Eriksson & Poloczek 2021, PMLR v130](https://proceedings.mlr.press/v130/eriksson21a/eriksson21a.pdf)
- **BoTorch 0.18.1.**
  - The input transforms are `Normalize`, `InputStandardize`, `Round`, `Log10`, `Warp`, `AppendFeatures`, `InputPerturbation`, `NumericToCategoricalEncoding`, `OneHotToNumeric`, `LearnedFeatureImputation` and others (`botorch/models/transforms/input.py`, class list). None of them is a signed-log input transform.
  - `Warp` applies a Kumaraswamy CDF and assumes inputs lie in the unit cube, clamped to [eps, 1−eps].
  - A signed-log appears only in the outcome transform `Bilog`: `Y_tf = Y.sign() * (Y.abs() + 1.0).log()` (`outcome.py` l.1075–1090).
  - Sources: [BoTorch input transforms source](https://botorch.readthedocs.io/en/stable/_modules/botorch/models/transforms/input.html); [BoTorch repo](https://github.com/pytorch/botorch)
- **Hyperopt 0.3.0.** `loguniform(low, high)` draws `rng.uniform(low, high)` and returns `np.exp(draw)`, with bounds given in log space (`hyperopt/pyll/stochastic.py` l.43–45). Output is therefore always positive, and `qloguniform` and `lognormal` follow the same pattern. No signed variant exists. — [hyperopt repo](https://github.com/hyperopt/hyperopt); usage example `hp.loguniform('learning_rate', np.log(0.01), np.log(0.5))` in [araffin blog](https://araffin.github.io/post/optuna/)
- **Ray Tune 2.59.0.** `LogUniform` raises "LogUniform requires a lower bound greater than 0" (`ray/tune/search/sample.py` l.294–297, l.403–406). — [Ray repo](https://github.com/ray-project/ray)
- **scikit-optimize 0.10.2.** `Real(..., prior='log-uniform')` raises when `low * high <= 0`, so the interval may not straddle or touch zero (`skopt/space/space.py` l.369). I did not test whether an all-negative interval then works through `np.log`. Note that scikit-optimize is effectively unmaintained. — [scikit-optimize repo](https://github.com/scikit-optimize/scikit-optimize)
- **ConfigSpace 1.2.2, used by SMAC3.** `if self.log and self.lower_value <= 0: raise ValueError("Negative lower bound ... for log-scale is not possible.")` (`ConfigSpace/hyperparameters/hp_components.py` l.287–291). — [ConfigSpace repo](https://github.com/automl/ConfigSpace)
- **scipy 1.18.1.** `loguniform` and `reciprocal` are the same `reciprocal_gen`, with support a ≤ x ≤ b and the requirement b > a > 0. `_argcheck` returns `(a > 0) & (b > a)` (`scipy/stats/_continuous_distns.py` l.9225–9334). — [scipy repo](https://github.com/scipy/scipy)
- **pycma 4.5.0.**
  - `scaling_of_variables` is marked "deprecated, rather use fitness_transformations.ScaleCoordinates instead (or CMA_stds)". `typical_x` is "deprecated, use `cma.fitness_transformations.Shifted`" (`cma/options_parameters.py` l.90, 119).
  - A user-supplied genotype→phenotype map `opts['transformation'] = [transform, inverse]` exists, but pycma warns that such transformations "may not be compatible with more recently introduced code features ... and are deprecated" (`evolution_strategy.py` l.719, 1137–1140).
  - In pycma, `symlog` appears only in plotting/logging code (`logger.py` l.2075, 2727).
  - Source: [pycma repo](https://github.com/CMA-ES/pycma)

### Inferences
- "No mainstream HPO or BBO library offers a native signed log-uniform search-space primitive" is a defensible and verifiable claim as of these versions. It should be phrased as "we found no …" and cite the specific validation lines above.
- Nevergrad's `ng.p.Log` default `init = sqrt(lower·upper)` uses the same geometric midpoint m as SLUD. It is worth noting this as a design precedent: the log-space centre of [lb, ub].
- Ax and BoTorch already ship bilog/signed-log transforms for outputs. A reviewer could reasonably ask why the same idea had not been applied to inputs. The answer is that inputs in HPO are rarely mixed-sign and multi-decade, whereas JANAF/NASA coefficients are. That answer is an argument, not a cited fact.

### Gaps
- I did not inspect SMAC3 itself, Optuna's TPE/CMA-ES samplers' internal handling of `log=True` beyond the distribution check, Syne Tune, HEBO, DEAP, pymoo or PlatEMO. A quick check of pymoo and DEAP would strengthen the "no native support" claim for the EC side, which matters most here because SLUD is evaluated with population-based methods.
- I did not verify scikit-optimize's behaviour with an all-negative log-uniform interval.

## Q2. Signed-log transforms elsewhere (symlog, bi-symmetric log, arcsinh, log-modulus, Yeo–Johnson) and their use in optimization input warping

### Takeaway
Signed-log transforms are long established in statistics, data visualization and scientific inference: John & Draper 1980, Burbidge et al. 1988, Yeo & Johnson 2000, matplotlib symlog, and Webber 2013. The closest prior art to SLUD as a sampling prior is GAMBIT/ScannerBit's `double_log_flat_join` prior from high-energy-physics global fits, which joins log-uniform on negative values, flat across zero and log-uniform on positive values. SLUD is essentially that construction (or a matplotlib symlog) with the linear core collapsed to zero width and the interval (−lb, lb) removed. In BO input warping, the standard methods (Snoek et al. 2014 Beta/Kumaraswamy CDF warping) act on [0,1]-normalised inputs and are monotone smooth bijections. They do not address sign, and I found no paper that uses asinh or symlog as a BO input warping for mixed-sign multi-decade parameters.

### Cited Findings
- **Log-modulus transformation.** John & Draper (1980), "An alternative family of transformations", *Applied Statistics / JRSS-C* 29(2):190–197 (JSTOR 2986305), define the modulus transform sign(y)·((|y|+1)^λ − 1)/λ, which for λ = 0 is sign(y)·log(|y|+1). It is a Box–Cox generalisation for data with both signs. — [R scales `transform_boxcox` docs](https://rdrr.io/github/r-lib/scales/man/transform_boxcox.html); [trafo::modulus](https://search.r-project.org/CRAN/refmans/trafo/html/modulus.html); [freerangestats blog](https://freerangestats.info/blog/2015/09/05/creating-a-scale-transformation). Primary source is paywalled: https://doi.org/10.2307/2986305
- **Inverse hyperbolic sine.** Burbidge, Magee & Robb (1988), "Alternative transformations to handle extreme values of the dependent variable", *JASA* 83(401):123–127, propose asinh, ln(x+√(1+x²)), as a log substitute for zero or negative values, with a scale parameter estimated by concentrated likelihood. — [Statalist summary](https://www.stata.com/statalist/archive/2008-04/msg00516.html); [feature-engine ArcSinhTransformer](https://feature-engine.trainindata.com/en/latest/api_doc/transformation/ArcSinhTransformer.html). Primary source: https://doi.org/10.1080/01621459.1988.10478575 (DOI not verified in-session)
- **Bi-symmetric log.** Webber, "A bi-symmetric log transformation for wide-range data", *Meas. Sci. Technol.* 24(2):027001. Repositories list it as 2013, with online publication in late 2012. It is a log-style transform that "can transform negative data to scaled negative data", with a single constant controlling a finite gradient near zero. — [Kent repository](https://kar.kent.ac.uk/32810); [STFC ePubs](https://epubs.stfc.ac.uk/work/10744277); DOI https://doi.org/10.1088/0957-0233/24/2/027001
- **matplotlib symlog / SymLogNorm.** It is "logarithmic in both the positive and negative directions from the origin". Because values near zero tend toward infinity in log space, a linear range (−linthresh, linthresh) is required, and `linscale` sets its visual width relative to a decade. — [matplotlib SymLogNorm docs](https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.colors.SymLogNorm.html); [proplot SymmetricalLogScale](https://proplot.readthedocs.io/en/v0.9.0/api/proplot.scale.SymmetricalLogScale.html)
- **Yeo–Johnson** (Biometrika 2000, 87(4):954–959) is a power-transform family defined on all reals. In the optimization tools, Ax uses it only on outputs (see Q1). — [Ax power_transform_y source](https://github.com/facebook/Ax). Primary source: https://doi.org/10.1093/biomet/87.4.954 (DOI not verified in-session)
- **Input warping for BO.** Snoek, Swersky, Zemel & Adams (ICML 2014) note that practitioners "often first transform the input space using a monotonic function such as the natural logarithm" but that this "is generally a crude approximation to the ideal (unknown) transformation". They learn bijective Beta-CDF warpings of the unit hypercube. — [arXiv:1402.0929](https://arxiv.org/pdf/1402.0929); [Princeton LIPS](https://lips.cs.princeton.edu/bibliography/snoek2014warping/)
- **BoTorch `Warp`** implements the Kumaraswamy-CDF variant. It assumes inputs in the unit cube and is typically chained after `Normalize`. — [BoTorch input transforms source](https://botorch.readthedocs.io/en/stable/_modules/botorch/models/transforms/input.html); [PyePAL usage](https://pypal.readthedocs.io/en/latest/_modules/pyepal/models/botorch_gp.html)
- **JCMwave optimizer** offers a `sinh` warping, but of function values, not inputs. — [JCMwave BO docs](https://optimizer.jcmwave.com/documentation/python/3.0.9/_sources/drivers/BayesianOptimization.rst.txt)
- **Closest prior art: GAMBIT ScannerBit `double_log_flat_join` prior.**
  - Described in "Comparison of statistical sampling methods with ScannerBit, the GAMBIT scanning module" (GAMBIT Scanner Workgroup), Eur. Phys. J. C 77 (2017) 761, arXiv:1705.07959.
  - It "patches together sections uniform in log(−x), uniform in x, and uniform in log x", for parameters "whose desired prior density is positive at zero but logarithmic at large absolute values".
  - Its option `ranges = [lower, flat_start, flat_end, upper]` sets the negative-log, flat and positive-log segments, and the same paper notes `range` is not available for this prior.
  - It was used in GAMBIT's GUT-scale SUSY fits for the trilinear coupling A₀, which can be either sign, as "a symmetric logarithmic prior at large |A_0| truncated to a flat prior at |A_0| < 100 GeV".
  - Sources: [arXiv:1705.07959](https://arxiv.org/abs/1705.07959); [arXiv:1705.07935](https://arxiv.org/pdf/1705.07935); [DOAJ record EPJC Nov 2017](https://doaj.org/article/fb3e1437b676417a813493f58e05acd7). Caveat: the description was obtained via a search-engine summary of the arXiv full text; arXiv itself was not directly fetchable in-session. The exact unit-hypercube mapping and the default mass split across segments still need to be checked in §3.1 of the paper.
- **A related astro sampler utility.** The `elisa` X-ray spectral package has a `bslogu` (bi-symmetric log-uniform) utility module. — [elisa.util.bslogu docs](https://astro-elisa.readthedocs.io/en/v0.2.3/apidoc/elisa.util.bslogu.html). I only saw the module title in the search hit and did not inspect its content.

### Inferences
- **SLUD is a symlog with linscale → 0.** SLUD is the inverse of y = sign(x)·log10(|x|/lb) restricted to |x| ≥ lb. This is a symlog/bi-symmetric-log transform with linear threshold lb and a zero-width linear core, and it is a ScannerBit `double_log_flat_join` with flat_start = −lb and flat_end = +lb whose flat segment receives zero mass.
  - The mathematical object (a "two-sided" or "symmetric" log-uniform) is therefore not new and should not be claimed as such.
  - Defensible novelty: (i) using this single-variable two-sided log parametrization as the decision-variable encoding (genotype→phenotype map) for population-based optimizers, rather than as a Bayesian prior or a plotting scale; (ii) the empirical comparison against the two-variable sign-plus-magnitude encodings LUD and LIN on a combustion kinetics/thermo fitting problem; (iii) the observation that it halves the dimensionality relative to sign-plus-log-magnitude encodings.
- A natural reviewer suggestion is the "SLUD-with-linear-core" generalization, i.e. symlog/ScannerBit style. It would let the optimizer reach exactly zero and would remove the jump. Discussing it as future work, or as an ablation, would pre-empt the "why exclude zero?" criticism.

### Gaps
- I could not open Webber 2013, John & Draper 1980, Burbidge et al. 1988 or ScannerBit §3.1 directly in this session. The facts above come from secondary summaries.
- I found no paper using asinh, symlog or log-modulus as an input warping in Bayesian optimization or evolutionary computation for mixed-sign parameters. A search phrased that way returned only Snoek et al. and JCMwave output warping. This is absence of evidence from a limited search, not proof of absence.
- An exact-phrase search for "signed log-uniform" returned no defining source. All hits were ordinary log-uniform HPO material, e.g. [Bischl et al. HPO review arXiv:2410.22854 hit list](https://arxiv.org/pdf/2410.22854). Searches for "symmetric log-uniform" or "two-sided log-uniform" were not run and are suggested follow-ups.

## Q3. Variable scaling and encodings in evolutionary computation: CMA-ES scaling, log-normal mutation, exponent/mantissa and sign-bit genes. Does anyone use a single variable spanning both signs on a log scale?

### Takeaway
EC has three relevant traditions:
- per-coordinate scaling (CMA-ES `scaling_of_variables`/`CMA_stds`, now deprecated in favour of fitness transformations);
- multiplicative or log-normal step handling (Nevergrad's `Log`/`exponent` mutation, which mutates in log space);
- floating-point-style encodings with separate mantissa and exponent genes, and sometimes a sign bit (IEEE-754 chromosomes, Deb-style M/E genes).

All of these either assume a positive variable or carry the sign as a separate gene or bit, which is structurally LUD. I found no EC paper that uses one continuous gene mapped monotonically onto [−ub, −lb] ∪ [lb, ub] log-uniformly.

### Cited Findings
- **pycma.** Per-variable scale is set by `CMA_stds` or `fitness_transformations.ScaleCoordinates`. The older `scaling_of_variables`/`typical_x` are deprecated. Arbitrary genotype–phenotype `transformation` callables are allowed but also deprecated and flagged as possibly incompatible with newer features. — [pycma repo, `cma/options_parameters.py`, `cma/evolution_strategy.py`](https://github.com/CMA-ES/pycma)
- **Nevergrad `Log`/exponent mutation.** "an exponent of 2.0 will lead to mutations by factors between around 0.5 and 2.0", i.e. multiplicative or log-normal mutation in the internal standardized space. It applies only to positive values. — [nevergrad repo `parametrization/data.py`](https://github.com/facebookresearch/nevergrad)
- **Floating-point encodings.**
  - A floating-point GA paper describes a chromosome laid out as the IEEE-754 64-bit word: 1 sign bit, 11 exponent bits and 52 mantissa bits. It reports faster execution than binary-coded GAs, and notes "some problems have arisen in the implementation of the traditional mutation of exponents". — [Traditional Techniques of GAs Applied to Floating-Point (CiteSeerX)](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1052.2503&rep=rep1&type=pdf)
  - A separate source describes Deb's floating-point encoding, in which "separate genes designated by M and E ... represent both mantissa and exponent of a floating-point parameter". — [IJEAT "Novel Encoding Scheme in GAs"](https://www.ijeat.org/wp-content/uploads/papers/v1i6/F0670081612.pdf). Caveat: these are search-snippet-level findings from lower-tier venues, and the primary Deb reference was not identified.
- **Combustion GA work.** The GA work on reaction-rate parameters by Elliott, Ingham, Tomlin and colleagues (GECCO 2004, LNCS 3103, pp. 932–944 and 945–956) used the modified Arrhenius form A·T^β·exp(−E/RT) with physically bounded ranges. A companion journal paper is "incorporation of physical bounds on rate parameters", Combust. Sci. Technol. 175(4):619–648 (2003). — [GECCO 2004 bib (York)](https://cs.york.ac.uk/rts/docs/GECCO_2004/Conference%20proceedings/bibs/3103/31030945.htm); [Springer LNCS](https://link.springer.com/doi/10.1007/978-3-540-24855-2_106); [GECCO 1999 Harris/Elliott et al.](https://gpbib.pmacs.upenn.edu/gecco1999/RW-759_2.pdf). Whether they encode log A or a signed quantity could not be confirmed from the snippets.

### Inferences
- The LUD baseline (a log-magnitude gene plus a sign gene in [−1, 1]) is the continuous analogue of the IEEE-754 sign-bit-plus-exponent chromosome. SLUD can be framed as folding the sign bit into the exponent gene. Using the floating-point and sign-bit encoding literature as the "sign-magnitude" family that SLUD improves on gives a clean positioning.
- **Topological difference worth stating explicitly.** In SLUD, +ub and −ub are maximally far apart in genotype space: they sit at opposite ends of the z interval, and the path between them crosses every decade down to lb. In LUD a sign flip at any magnitude is a one-coordinate move. SLUD therefore assumes that sign changes are "cheap" only at small magnitudes, which is physically sensible when ±lb are negligible.

### Gaps
- I did not find, or check, a systematic EC survey of encodings for parameters spanning orders of magnitude, e.g. Bäck's or Eiben & Smith's textbooks on representation. Nor did I find any GECCO, ECJ or TEVC paper on "signed logarithmic encoding". Absence is from limited search (about four queries), so a targeted Scholar search with terms such as "logarithmic encoding" "negative" "evolution strategy" is recommended before submission.
- I did not check whether a "logarithmic number system" (a sign bit plus a fixed-point log magnitude, from computer arithmetic) has been used as a GA representation. It is conceptually the LUD encoding.

## Q4. Combustion and chemical-kinetics optimization practice: how A, n, Ea and NASA/JANAF coefficients are parametrized, and whether signed many-decade variables are needed

### Takeaway
Mainstream mechanism optimization parametrizes rate parameters in log or "transformed" coordinates around a nominal value: ln A, n, E/R (Turányi/Optima++), or a normalized factorial variable x = ln(k/k₀)/ln f bounded by the uncertainty factor f (Sheen & Wang MUM-PCE, used in FFCM). These are bounded perturbations of a known prior mechanism. They never need a single variable spanning both signs over many decades: n and E/R are optimized linearly, and A is positive. Virtual chemistry (Cailler et al. 2020) is the exception in kind. Rate and thermodynamic parameters of fictitious species are optimized from scratch by a GA, with no nominal value to perturb, which is the regime where SLUD's need arises. I could not access the paper's parametrization details.

### Cited Findings
- **Turányi group (Nagy et al. 2015, Combust. Flame 162(5):2059–2076; Varga et al. 2015 H₂ mechanism).**
  - Parameters are optimized as "transformed Arrhenius parameters (ln A, n, E/R)", whose uncertainty domain "is convex and centrally symmetric around the mean parameter set".
  - Uncertainty is represented through the covariance of the transformed parameters, and a uniform distribution in the transformed parameters is used when little information is available.
  - Optima++ is the C++ code for this, distributed via ReSpecTh.
  - Sources: [BME seminar abstract](https://old.bme.hu/node/5314); [Galway record](https://research.universityofgalway.ie/en/publications/determination-of-rate-parameters-based-on-both-direct-and-indirec/); [Turányi 2018 slides](https://math.bme.hu/sites/math.bme.hu/files/AlkAnalSzemi/PapersPresentations/Turanyi_BME_8March2018.pdf)
- **MUM-PCE / FFCM (Sheen & Wang; H. Wang lab, Stanford).** Each rate parameter is normalized as x_k = ln(A_k/A_{k,0})/ln f_k, with f_k the uncertainty factor. The rate-coefficient uncertainty is lognormal with median equal to the nominal value. — [FFCM-1 optimization methodology](https://web.stanford.edu/group/haiwanglab/FFCM1/pages/method.html); [FFCM-2 docs](https://web.stanford.edu/group/haiwanglab/FFCM2/docs/Optimization); [mumpce_py, NIST J. Res. 122:39 (2017)](https://www.nist.gov/node/1276741). Sheen & Wang, Combust. Flame 158 (2011) 2358–2374, https://doi.org/10.1016/j.combustflame.2011.05.010 (DOI not verified in-session)
- **Recent EA-based Arrhenius optimization.** Murakami, Bertolino & Parente (Fuel, 2026) report that correlations among Arrhenius parameters make it hard to sample valid sets. They use ICA inside an evolutionary optimizer, giving about 80% more valid selections on a 72-parameter NH₃ mechanism. — [ULB record](https://difusion.ulb.ac.be/vufind/Record/ULB-DIPOT:oai:dipot.ulb.ac.be:2013/403735/TOC)
- **ANSYS Chemkin global mechanism optimization** tunes Arrhenius (and low/high-pressure-limit) parameters, but not PLOG or CHEB formulations. — [Ansys help](https://ansyshelp.ansys.com/public/views/secured/corp/v251/en/reaction_util/rwb_sec_globmechopt.html)
- **Virtual chemistry.**
  - Cailler, Darabiha, Veynante & Fiorina, "Development of a virtual optimized chemistry method. Application to hydrocarbon/air combustion", Combust. Flame 211 (2020) 281–302: "kinetic rate parameters of the virtual reactions and virtual species thermodynamic properties are optimized through a genetic algorithm" to capture flame temperature and CO.
  - Extensions: NO by Maio et al. (CTM 2020), soot by Maldonado Colmán et al. (C&F 2022), and a 2026 generalisation with standardised thermodynamics (Préteseille et al., CTM).
  - Sources: [HAL hal-02494438](https://hal-centralesupelec.archives-ouvertes.fr/hal-02494438); [virtual-chemistry publications](https://virtual-chemistry.em2c.centralesupelec.fr/publications/). DOI 10.1016/j.combustflame.2019.09.016 (not verified in-session)
- **The project's own draft (COB-2025-2080).** It states that the 16-variable virtual-chemistry problem includes JANAF heat-capacity polynomial coefficients for a virtual species and Arrhenius parameters. It says the JANAF coefficients "have to span many orders of magnitude" and either sign. It also says plain log-uniform "would require to double every variable that can also have negative values". — local file `/home/user/Signed-Log-Uniform-Distribution-SLUD/COB-2025-2080.pdf` (abstract and introduction)

### Inferences
- **Mainstream versus virtual chemistry.**
  - In mainstream kinetics UQ and optimization, the log is applied to A (always positive) and to k/k₀ ratios, which are also positive. n and E are perturbed linearly within narrow bounds. So the field's standard tools never face the "mixed sign × many decades" problem.
  - Virtual chemistry and other from-scratch thermo-polynomial fitting do face it. NASA-7 a₁…a₅ coefficients alternate in sign and range from about 1e0 down to about 1e-15 (T⁴ term). They are not perturbations of a nominal value. This contrast is a good motivation paragraph.
  - Caveat: the coefficient-magnitude statement is general knowledge of NASA-7 polynomials, not a cited finding in this note.
- A reviewer from combustion may ask why not optimize physically meaningful thermo quantities (Cp(T) at anchor temperatures, ΔHf, S°) and refit the coefficients. That would make every variable single-signed, or nearly so, and well scaled. The Préteseille et al. (2026) "standardised thermodynamics" direction may do exactly this. The paper should acknowledge this alternative and argue SLUD's generality, or its compatibility with legacy coefficient-level codes.

### Gaps
- I could not access the Cailler 2020 full text (HAL and DOI fetches failed). I therefore cannot state how Cailler et al. bounded or scaled A, β, Ea and the thermo coefficients, or whether they used log scaling. This should be checked in the paper before claiming that "the original method used linear bounds".
- I did not find DeePMO or other ML-based mechanism optimizers' parametrization documentation in this session.
- I could not confirm whether the Elliott/Ingham GA encoded log A.
- I found no combustion paper that handles signed many-decade variables with a single log-type variable (absence from limited search).

## Q5. Likely reviewer criticisms of SLUD and relevant literature

### Takeaway
The main attack surfaces are:
1. exclusion of (−lb, lb), so zero is unreachable;
2. the jump at the midpoint;
3. wasted sampling when |x| ≈ lb is physically negligible;
4. wasted half-space for variables whose sign is actually known;
5. symmetric magnitude bounds for both signs;
6. genotype topology: large +x and large −x are far apart;
7. interaction with PSO velocity and clamping, LHS stratification, and model-based or gradient-based optimizers.

Most of these have direct analogues in the symlog, bi-symmetric-log and ScannerBit literature, which addressed (1) and (2) with a linear or flat core around zero.

### Cited Findings
- **Zero and the near-zero region.** Symlog exists because log scales cannot represent zero: "values close to zero tend toward infinity", so a linear region (−linthresh, linthresh) is introduced. — [matplotlib SymLogNorm](https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.colors.SymLogNorm.html). Webber's bi-symmetric log likewise keeps the gradient finite near zero with one tuning constant. — [Kent repository](https://kar.kent.ac.uk/32810). ScannerBit's `double_log_flat_join` inserts a flat segment so the prior density is positive at zero. — [arXiv:1705.07959](https://arxiv.org/abs/1705.07959). The project draft itself acknowledges as drawbacks "the inability to represent zero values" and the "discontinuity between the lower and upper bounds". — local `COB-2025-2080.pdf`, §2
- **Log transforms as crude approximations.** Snoek et al. argue that a fixed log transform is "generally a crude approximation to the ideal (unknown) transformation" and that learned warpings help BO. A reviewer could ask whether SLUD should be compared against, or combined with, learned warping. — [arXiv:1402.0929](https://arxiv.org/pdf/1402.0929)
- **Unit-cube assumptions.** BoTorch's `Warp` assumes inputs normalised to the unit cube and clamps them to [eps, 1−eps]. Applying SLUD inside a BO pipeline means the GP sees z (or t), and any jump in f at t = 0 is a non-stationarity the GP must model. — [BoTorch input transforms source](https://botorch.readthedocs.io/en/stable/_modules/botorch/models/transforms/input.html)
- **Bounds and sign selection are a known pain point in EA kinetics.** Physical bounds on rate parameters were the subject of a dedicated paper (Combust. Sci. Technol. 175(4), 2003), and Arrhenius-parameter correlations hamper valid sampling (Murakami et al. 2026). — [York GECCO bib](https://cs.york.ac.uk/rts/docs/GECCO_2004/Conference%20proceedings/bibs/3103/31030945.htm); [ULB record](https://difusion.ulb.ac.be/vufind/Record/ULB-DIPOT:oai:dipot.ulb.ac.be:2013/403735/TOC)

### Inferences
These are analysis points to include or pre-empt; they are reasoning, not cited facts.
1. **The jump is small in physical space.** The discontinuity in x is only 2·lb. If lb is chosen below the physically relevant scale, f∘x is nearly continuous at t = 0, and the "discontinuity" objection becomes a parameter-choice argument. Conversely, the real cost is that a fixed fraction of the z-range maps to physically negligible magnitudes, giving a near-flat plateau in the fitness landscape: log10(ub/lb) decades per side are spread uniformly, including irrelevant tiny decades. The same issue exists in LUD, so it is not a differential weakness, but lb should be justified.
2. **Monotone, order-preserving map.** x(z) is monotone increasing, so SLUD preserves ordering. Box constraints in z map exactly onto [−ub, −lb] ∪ [lb, ub], and standard clamp or reflect boundary handling stays valid. Worth stating, because it distinguishes SLUD from non-monotone encodings (e.g. the |z|-based "fold").
3. **Topology and PSO.** Particles near +ub must travel the whole z-range to reach −ub. PSO velocity clamping, typically a fraction of the range, makes such sign changes multi-iteration events, while LUD allows instant sign flips. This may help (stability) or hurt (an optimum of the opposite sign at large magnitude). An ablation with an optimum at −ub/2 versus +ub/2 would answer it.
4. **LHS.** With SLUD, an n-point LHS in z stratifies decades within each sign and allocates exactly half the strata to each sign, guaranteeing balanced sign coverage per variable. With LUD, LHS stratifies the sign variable and the log-magnitude separately, so per-variable marginals are equally balanced but in twice the dimensions. The joint space-filling quality in the doubled space is worse for the same sample size. This is a concrete, testable advantage claim.
5. **Single-sign variables.** For variables known to be positive (e.g. A), SLUD wastes half the range unless the code falls back to plain LUD. The paper should state that SLUD is applied only to mixed-sign variables.
6. **Asymmetric ranges.** SLUD assumes the same [lb, ub] magnitude for both signs. A generalisation with separate (lb⁻, ub⁻, lb⁺, ub⁺) and a sign-split point p is straightforward and is the ScannerBit-style generalisation.
7. **Gradient-based and model-based optimizers.** Near t = 0, dx/dt = x·ln(ub/lb) is finite, but x jumps at t = 0. Gradient methods may stall at the jump, and GP surrogates with stationary kernels will see a kink or step. A symlog-style linear core would fix both.
8. **Dimensionality claim.** SLUD halves the number of decision variables for mixed-sign parameters compared with LUD and LIN. This is objectively true and is the core practical selling point. For CMA-ES, whose sample-size and learning-rate defaults scale with n, and for PSO, halving n changes default population settings. Comparisons should therefore control population size and evaluation budget.

### Gaps
- I found no EC or BO literature specifically analysing optimizer behaviour on sign-plus-magnitude versus single signed-log encodings, i.e. no direct empirical precedent for or against SLUD.
- I found no literature on LHS quality in mixed sign/magnitude encodings. The LHS argument above is analytical, not cited.

### Summary of defensible novelty claims (synthesised from all sections above)
- **Defensible.**
  - None of the major Python HPO/BBO libraries provides signed log-uniform inputs; each rejects non-positive log bounds (code-level evidence in Q1).
  - SLUD is a single-variable, monotone, two-sided log-uniform genotype→phenotype map used as an EA/PSO decision-variable encoding that halves dimensionality versus sign-plus-magnitude encodings.
  - The paper gives an empirical comparison (SLUD versus LUD versus LIN) on virtual-chemistry JANAF/Arrhenius fitting, where mixed-sign, many-decade variables arise because there is no nominal mechanism to perturb.
- **Not defensible without qualification.**
  - That a signed/symmetric logarithmic transform or distribution is new. Prior art includes John & Draper 1980; Burbidge et al. 1988; Webber 2012/13; matplotlib symlog; and especially GAMBIT/ScannerBit `double_log_flat_join` (EPJC 2017), a sampling prior with log-uniform on both signs.
  - That log-scaling of kinetic parameters is new. Prior art includes Turányi's ln A, n, E/R and MUM-PCE's ln(k/k₀)/ln f.
- **Recommended related-work framing.** Present SLUD as "the symlog / double-log prior idea, with the linear core collapsed, transplanted from statistics/visualisation/Bayesian scanning into the decision-variable encoding of metaheuristics", and position LUD and LIN as instances of the sign-bit/sign-gene encoding tradition.
