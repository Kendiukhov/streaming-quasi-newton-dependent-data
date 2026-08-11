# Streaming inversion-free quasi-Newton estimation and inference on dependent data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21891312.svg)](https://doi.org/10.5281/zenodo.21891312)

Archived release: [10.5281/zenodo.21891313](https://doi.org/10.5281/zenodo.21891313) (`release1`).
The concept DOI [10.5281/zenodo.21891312](https://doi.org/10.5281/zenodo.21891312) always resolves
to the most recent release. Cite the version DOI when the point is reproducing the results in the
paper, the concept DOI when the point is the software itself.

Reference implementation and full experimental pipeline for the paper

> **Inference, Not Just Optimisation: Streaming Inversion-Free Quasi-Newton Estimation on
> Dependent Data Streams** (`paper/main.pdf`)

---

## The problem in one paragraph

Streaming quasi-Newton methods fit a model in a single pass, maintaining an inverse
curvature matrix by rank-one Sherman–Morrison updates so that the *total* cost is `O(dN)` —
the same order as SGD — while keeping the asymptotic efficiency of Newton's method
([Godichon-Baggioni & Werge, JMLR 26(59), 2025](https://www.jmlr.org/papers/v26/23-1565.html)).
That theory assumes independent observations. On a serially dependent stream the estimator's
limiting covariance is the **long-run** sandwich

```
sqrt(N) (theta_hat - theta*)  ->  N(0,  H^-1 S H^-1),      S = sum_k Cov(score_0, score_k)
```

not the instantaneous sandwich `H^-1 Gamma_0 H^-1` that the independent-data theory
certifies. The two differ by a factor that does **not** shrink with `N`, so nominal 95%
intervals converge to a coverage below 95% and stay there. On the hourly air-quality stream
studied here that limit is about 34%.

## What this repository provides

**BGSN** (blocked–gapped streaming Newton): the same `O(dN)`-time, `O(d²)`-memory estimator,
with two structural changes that make its *uncertainty* correct too.

* **Blocked gradient steps.** One preconditioned step per contiguous block of `b`
  observations. The block gradients the algorithm already forms are, up to a factor `b`, the
  batch means of the score process — so they *are* an estimator of `S`. The statistically
  correct object and the computational shortcut are the same operation.
* **Gapped curvature updates.** Rank-one curvature updates only every `m`-th observation.
  At a matched number of updates this decorrelates the curvature terms exponentially in `m`,
  whereas the Bernoulli thinning used as the cost knob in the independent-data method only
  does so linearly.

The two changes do different jobs, and we are explicit about which one earns the coverage.
**Blocking is what fixes the inference**: it is what produces a consistent estimate of `S`,
and removing it (using the i.i.d. plug-in covariance instead) is what makes coverage collapse.
**Gapping is a curvature-side cost knob**: it buys a better curvature estimate per rank-one
update, and in our experiments it does *not* move coverage measurably — which is exactly what
the theory predicts, because the preconditioner enters the limit distribution only through a
rate. We report it because it strictly dominates the Bernoulli thinning the independent-data
method uses for the same purpose, not because the intervals need it.

**A stability condition that blocking creates.** Blocking the gradient makes the step's
contraction condition non-trivial. The linearised error recursion has spectral radius below one
only while `gamma_t * lambda_max(Hbar^-1 Hhat_t) < 2`, where `Hhat_t` is the *block* Hessian. On
independent data a block of `b >= d` observations gives a full-rank `Hhat_t` and the condition
holds at `gamma_1 = 1`; on a dependent stream the same block can span far fewer than `b` distinct
covariate directions, `Hhat_t` is near-singular, and a full-length first step overshoots by orders
of magnitude. We measured `||H^-1 Hhat_t||` at `b=20`: median 20 on the AR design and 33 on the
regime-switching design, so the condition is violated on *both* — the AR designs are lucky, not
safe. Without a safeguard, individual runs on the regime-switching design diverge outright; the
paper reports the relative variance with and without, from the ablation table.

The fix is an `O(d)` step-length safeguard: each step is capped at `c_D (1 + ||theta||)`, which
binds 2–7 times per run and provably changes nothing once it stops binding. We also implemented
the step-size shift `gamma_t = 1/(t + t0)` that the condition suggests more directly, measured it
against the safeguard, and rejected it — it slows all `10^4` steps to correct a handful, and on a
short stream `t0` can exceed the number of steps and freeze the estimator. Both are in the code
(`step_cap`, `t0_mult`); the comparison is reported. This is a cost of *our* construction, not of
the method we build on.

**The correct interval is also the cheaper one.** The i.i.d. plug-in score covariance costs
`O(d²)` per *observation*; the blocked long-run covariance costs `O(d²)` per *block*.

## Install

```bash
python3 -m pip install numpy scipy pandas matplotlib numba joblib
```

Python ≥ 3.9. Numba is used for the streaming kernels; everything else is NumPy/SciPy.

## Quick start

```python
import numpy as np
from bgsn import estimators as E

# X: (N, d) covariates in time order, y: (N,) responses.  d = 20 here.
fit = E.streaming_newton(X, y, model="linear", b=20)     # one pass, O(dN) time
lo, hi = fit.ci()                                        # valid under temporal dependence
print(fit.theta, fit.se)
```

`model` is `"linear"` or `"logistic"`. `b` is the block length: it must exceed the
dependence horizon of the score process. The fit reports a free diagnostic for that choice —

```python
u = fit.extras["W"] * fit.extras["Ainv"]                 # the maintained Hbar^{-1}
num = np.abs(np.diag(u @ (fit.extras["S_ft"] - fit.extras["S_bm"]) @ u))
den = np.diag(u @ fit.extras["S_ft"] @ u)
print("block adequacy r_j:", num / den)                  # small (< 0.05) means b is enough
```

— which estimates the relative block-length bias of plain batch means. If it is large,
increase `b`.

To see the failure mode the paper is about, ask for the interval the independent-data theory
prescribes and compare:

```python
bad = E.streaming_newton(X, y, "linear", b=20, variance="plugin")   # i.i.d. plug-in
print(fit.se / bad.se)     # ~ sqrt(kappa): how much too narrow the i.i.d. interval is
```

## Reproducing the paper

```bash
make all           # data + validation tests + all experiments + figures + tables + PDF
make test          # just the validation of every closed-form population quantity
make all NJOBS=8   # more parallelism
make all R=100     # smaller Monte Carlo (fast smoke run)
```

Total compute is a few CPU-hours on a laptop. Every experiment writes a JSON file to
`results/`, every figure a PDF to `figures/`, and `experiments/make_tables.py` emits both the
LaTeX tables and `paper/numbers.tex` — a file of macros holding **every number quoted in the
prose**, so the text cannot drift from the results.

### Note on BLAS threads

`experiments/_env.py` pins BLAS to one thread before NumPy is imported and parallelism is
over Monte Carlo replicates instead. This is not cosmetic: on the development machine an
unpinned tall-skinny `(d, N) @ (N, d)` product with `N ≈ 4·10⁴`, `d ≈ 12` took **18 s**
against **5 ms** pinned — a thread-oversubscription pathology that would otherwise dominate
every timing measurement. Because the machine was shared, the wall-clock numbers in the
paper are corroborated by exact, machine-independent operation counts computed from the
algorithm itself (`experiments/exp7_cost.py`).

## Layout

```
src/bgsn/
  _core.py        numba streaming kernels: the Riccati curvature recursion, the blocked
                  gradient step, the long-run covariance accumulators, ASGD and AdaGrad
  estimators.py   user-facing estimators, sandwich standard errors, offline HAC reference,
                  random-scaling critical values
  models.py       loss / score / rank-one curvature form for least squares and logistic
  streams.py      four dependent DGPs with closed-form or simulated population targets
  dgp.py          DGP objects: a FIXED population plus a sampler (see the warning below)
  realdata.py     the two real hourly streams and the exact conditional oracle used for
                  the semi-synthetic real-data protocol
experiments/
  _env.py         BLAS thread pinning (import before numpy)
  common.py       method registry, Monte Carlo driver, tuning, summaries
  exp0..exp10     one file per experiment; each writes results/expN_*.json
  make_figures.py, make_tables.py
tests/
  test_oracles.py every closed-form S, Gamma_0 and H checked against a long simulation;
                  the two-scale algebraic identity; the streaming dyadic accumulators
                  checked against an explicit batch-means computation
  test_paper_sources.py  no shared fragment truncated; every numeric macro generated
paper/            LaTeX sources for the preprint.  Section bodies, figure captions, tables,
                  the abstract and numbers.tex are shared with the journal submission, so the
                  two versions cannot drift apart.  Every refs.bib entry was verified against
                  a live publisher / arXiv / dataset page; the evidence URLs and the list of
                  corrections are in notes/refs_raw.json
submission/       the Information Sciences (Elsevier) submission package: manuscript in
                  elsarticle format, supplementary material, highlights, cover letter and the
                  required declarations.  `make submission` builds it; submission/README.md
                  lists what the journal requires and what the author must still complete
notes/            provenance: design notes written before the code, the binding list of
                  what we may and may not claim relative to prior art, the adversarial
                  review and what each finding changed, and the bibliography verification
                  records (see notes/README.md)
results/          one JSON per experiment (the evidence behind every number) plus logs
figures/          generated PDFs
```

### Two bugs worth repeating

Both were found by an adversarial review pass (`notes/review_response.md`) and both had been
biasing results in the paper's favour, which is why they are called out here rather than buried.

1. **The i.i.d. plug-in variance was accumulated over the curvature warm-up**, where `theta` is
   still `theta_0` and the scores are large, while the blocked estimators correctly excluded
   those blocks. The dependence-blind competitor therefore looked worse than it is — mildly in
   simulation, badly on a short real segment. All three variance estimators now see exactly the
   same observations.
2. **Four of ten traffic segments had an exactly rank-deficient design**, because `rain_1h` is
   identically zero over stretches of thousands of consecutive hours, so the standardised
   `log1p(rain)` column is constant on some segments. Rainfall was dropped from that feature
   set, and `exp6_real.py` now asserts full rank per segment.

### A third warning worth repeating

`dgp.py` exists because of a bug we made and fixed: if the *design* (`Sigma_x`, `theta*`) is
redrawn per Monte Carlo replicate, the empirical covariance of the estimator mixes sampling
with population variability and cannot be compared to any single oracle sandwich. Every
experiment goes through `DGP.sample(N, seed)`, which varies the realised path only.

## The paper

Two versions are built from one set of sources, so they cannot disagree:

* `paper/main.pdf` — the preprint, single column.
* `submission/manuscript.pdf` — the Information Sciences submission: `elsarticle`, line-numbered,
  a 200-word abstract, ten figures + tables (the journal's guideline for a theoretical article)
  with four further figures in `submission/supplementary.pdf`, and the declarations the journal
  requires.

Both have a **17-page main narrative** (the numbered Sections 1–8) and seven lettered appendices.
Table D.4 is a `longtable` spanning three pages; every other table is a float whose width is capped
by a `\resizebox` guard. `make layout` checks both PDFs for over-wide lines, floats too tall for a
page, and ink outside the text block on any side of any page.
Each long section is split into a core fragment, which the body includes, and a `_detail`
fragment, which the appendix includes — so the derivation of the contraction condition, the full
assumption statements, the secondary experiments, the complete literature survey and the long-form
discussion of every limitation are all still there, one level down. No paragraph appears in both
halves.

`make paper` builds the first, `make submission` the second. The shared fragments are
`paper/sec_*.tex`, `paper/fig_*.tex`, `paper/tab_*.tex`, `paper/abstract.tex` and
`paper/numbers.tex`; the two main files own only their own preamble, title page and float
placement. `tests/test_paper_sources.py` checks that no shared fragment has been truncated and
that every numeric macro used in the prose is actually generated from `results/`.

## Data

Downloaded by `experiments/fetch_data.py` from the UCI Machine Learning Repository:

* **Metro Interstate Traffic Volume** (Hogue) — hourly westbound volume on I-94 between
  Minneapolis and St Paul, 2012-10 to 2018-09, with weather covariates.
* **Beijing Multi-Site Air-Quality Data** (Zhang et al.) — hourly pollutant and weather
  readings at 12 monitoring sites, 2013-03 to 2017-02.

Timestamps are sorted and de-duplicated; missing hours are simply absent, so "lag k" means
"k observations back". Feature construction is fixed in advance and never tuned on the
response.

## What we claim, and what we do not

We claim: convergence, an almost sure rate and a CLT with the long-run sandwich for the
*inversion-free rank-one curvature recursion with gapped updates* under geometric mixing; a
long-run covariance estimator built from block gradients whose block length need not grow
with `N` (hence `Õ(N^{-1/2})` rather than the `N^{-1/8}` of iterate-based batch means under
dependence); an `O(rho^b)` block-length bias for the two-scale correction at block-gradient
resolution; the exact coverage functional for dependence-blind intervals; and that a
deterministic gap beats Bernoulli thinning at matched cost.

We do **not** claim: the `O(dN)` budget or curvature thinning (Godichon-Baggioni & Werge
2025; Godichon-Baggioni, Portier & Sallé 2026); blocked mini-batches as a dependence-breaking
device (Godichon-Baggioni, Werge & Wintenberger); the long-run sandwich limit or the
semiparametric efficiency bound under dependence for first-order methods (Liu, Chen & Shang;
Li, Liang & Zhang; Samsonov et al.); the existence of online long-run covariance estimators
under dependence for first-order methods (Roy & Balasubramanian; Samsonov et al.); online
covariance estimation for streaming second-order methods under independent data
(Kuang, Anitescu & Na; Wang, Du & Na); or the two-scale / flat-top / lugsail correction
itself (Politis & Romano; Politis; Vats & Flegal; Singh, Shukla & Vats).
Section 6 of the paper states each boundary precisely.

## Limitations

Memory is `O(d²)`, not `O(d)`. Geometric mixing is a real restriction and the block length
must respect it — on the air-quality stream, where the residual lag-one autocorrelation is
0.92, the block length needed is an order of magnitude larger than for traffic. The two-scale
covariance is not guaranteed positive semidefinite (a documented fallback is used; it never
triggered). The curvature form `grad^2 F = E[alpha * Phi Phi']` is required. The curvature
warm-up is a tuning constant the asymptotic theory is silent about.

## License

Code: MIT (`LICENSE`). The datasets remain under their original UCI terms and are not
redistributed here.
