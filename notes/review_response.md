# Adversarial review and what it changed

Before finalising, the manuscript and code were put through a five-lens adversarial review
(theory correctness, statistical/experimental design, positioning versus prior art, clarity,
code correctness), with every blocking/major finding independently verified against the files
before being accepted. Forty candidate findings; twenty-eight survived verification. The raw
records are in `review_raw.json` and `review_findings.txt`.

Two of the findings changed *numbers*, not just wording, and both had been biasing results in
the paper's favour. They are listed first, because those are the ones that matter.

## Findings that changed reported numbers

**1. The i.i.d. plug-in variance was accumulated over the curvature warm-up.**
`_core.py` accumulated `Gamma0_hat` from the first observation, including the warm-up blocks
where `theta` is still `theta_0 = 0` and the scores are large, while the blocked estimators
`S_bm`/`S_ft` correctly excluded them. The dependence-blind competitor's variance was
therefore inflated for a purely implementational reason — mildly in simulation (~2%), badly on
real data, where the warm-up can be 15% of a short segment. Fixed by guarding the accumulator
with the warm-up flag so all three estimators see the same observations; every affected
experiment was re-run.

**2. Four of the ten traffic segments had an exactly rank-deficient design.**
`rain_1h` in the Metro series is identically zero over stretches of several thousand
consecutive hours, so the standardised `log1p(rain)` column is constant on some contiguous
segments and collinear with the intercept. The "exact conditional oracle" was then inverting a
singular matrix. Fixed by dropping rainfall from the traffic feature set (temperature and cloud
cover carry the weather signal), and `exp6_real.py` now *asserts* that every evaluation segment
has full rank and reports the worst per-segment condition number, so this cannot recur silently.

## Theory

**3. The weighted-sum lemma was applied with random weights it did not cover.** The original
lemma assumed *deterministic* weights and centred stationary summands, but in all three places
it was used — consistency, and remainder terms (A) and (C) — the weights contain
`Hbar^{-1}` and are random and correlated with the summand. The verification agent produced a
numerical counterexample showing the claimed bound can fail by a factor `T^{1/2}` for bounded
*adapted* weights. Replaced by Gordin's martingale–coboundary decomposition plus a lemma for
adapted weights with a localised deterministic envelope, stated with an explicit rate (three
terms, since the applications use `gamma_t = 1`, for which the increment sum diverges
logarithmically and a summability hypothesis would be too strong). Terms (A) and (C) rewritten
against it; the appendix now also states what the crude Davydov bound gives and why it is not
enough.

**4. The cited Bernstein inequality does not hold under our assumptions.** Merlevède–Peligrad–Rio
Theorem 1 requires uniformly bounded summands; we only assume `2+nu` moments. After the
coboundary decomposition the summands form a martingale difference sequence, so a martingale
law of the iterated logarithm suffices and the inequality is no longer cited for that purpose.

**5. The theorem dependency graph was circular as written.** The curvature theorem
hypothesises a rate for the iterates; the convergence proof uses the curvature theorem; the
rate proof uses both. Split into an *unconditional* lemma (the ridge bound, from the ridge
alone) and a *conditional* rate theorem, with the four-rung bootstrap ladder stated explicitly
as a remark and followed in that order in the appendix.

**6. The Bernoulli-thinning variance ignored `E[Y] = H != 0`.** With a fixed normaliser the
omitted term is `Theta(1)` and swamps the `O(1/m)` effect. The algorithm uses the *realised*
normaliser, so the estimator is a ratio and the mean-induced term enters only at second order;
the proposition and proof now say so and use the delta method.

**7. Missing assumptions and mis-stated regimes.** Added an identifiability assumption (needed
for global convergence and used implicitly before), a fourth-moment condition (needed for the
covariance estimator's variance), and an `L^2` rate theorem (the drift analysis needs a moment
bound, which an almost sure rate with a random constant does not supply). Restated the
covariance theorem uniformly over `b <= C log N`, corrected the operator-norm conversion to
carry its factor `d`, symmetrised `Delta = sum_k k(Gamma_k + Gamma_k')`, and replaced
"algebraically identical to `2 S(2b) - S(b)`" by the correct "equal in expectation, and
identical if disjoint pairs are used".

## Positioning

**8. A false absolute claim about prior art.** "None of it estimates an asymptotic variance"
is wrong: Godichon-Baggioni (2019, JSPI 203:1–19) gives a recursive online estimator of the
asymptotic variance for averaged first-order algorithms under i.i.d. data — by the author of
the base method. Now cited in both the contributions and related work, with the sentence
corrected.

**9. The headline rate compared different objects.** `Otilde(N^{-1/2})` was our rate for `S`,
while `N^{-1/8}` is a rate for the whole sandwich `Sigma = H^{-1} S H^{-1}`. Two changes: the
composite sandwich rate is now stated (limited by the curvature rate, which the ridge exponent
can push arbitrarily close to `N^{-1/2}`), and the rate figure now contains a genuine
head-to-head against an iterate-path overlapping-batch-means estimator of `Sigma` at the block
length its analysis requires — replacing what had been our own estimator run at a bad block
length, which is a straw man.

**10. Two over-claims.** "Caps the rate at `N^{-1/8}`" became "the best published rates are
`N^{-1/8}`, and the minimax ceiling for the Hessian-free trajectory class is `N^{-1/4}`". The
exclusivity claim "available only to an algorithm that forms block gradients and keeps a
curvature estimate" is false — we run the same construction on averaged SGD as a baseline — and
became "available to any method that sees the individual gradients and can supply a curvature
estimate; what the quasi-Newton recursion contributes is that the curvature estimate is already
there". The reason the Ni–Huo minimax bound does not bind is now stated purely in information
terms, with the note that their own optimal estimator does form a Hessian.

**11. Contribution 3 was demoted.** The `O(rho^b)` bias order of a flat-top window under
geometric autocovariance decay is the classical tail-sum property, not a new mechanism. The
contribution is now stated as: it is available at *block-gradient* resolution on a dependent
stream, which is what makes `b ~ log N` admissible, plus the streaming (dyadic) implementation.

## Experimental design and code

**12. Real-data tuning leaked.** One grid search on the traffic stream's first 8000 rows —
which were evaluation segments — supplied the step constants for all twelve air-quality streams
too. Now tuned *per stream* on a leading 15% hold-out that is excluded from every evaluation
segment, with the grid and whether the optimum is interior recorded.

**13. Real-data standard errors ignored clustering.** Protocol A replicates that share a
covariate segment differ only in the error path. Monte Carlo standard errors are now
cluster-robust at the segment level, and the text says explicitly that Protocol A coverage is
conditional on the realised covariate segments.

**14. Protocol B's correction was undocumented.** `R` is now defined, the one-line derivation
and its three assumptions are in the appendix, and both the raw and corrected nominal levels
are reported.

**15. The critical-value script was off by one quantile level.** It reported the 0.975 quantile
of an already-absolute-valued statistic as "the two-sided 95% value" and compared against the
published table without the signed/absolute conversion, so it disagreed at every row even
though the sampler is correct. Fixed; the published table's quantiles are now labelled as
belonging to the *signed* statistic and compared at the matching level.

**16. Two experiments were silently degraded.** The `b ~ N^{3/4}` arm of the rate experiment
was skipped at every `N` by a guard, and `exp8_lrv_rate.json` was not a prerequisite of the
Makefile's `exp` target, so `make all` never produced that figure. Both fixed.

**17. The batch-means theory curve used its own asymptotic expansion.** The comparison is now
against the exact Bartlett-weighted sum, which is closed form for this design, so the residual
at `b = 5` is visibly the `O(rho^b)` term rather than an artefact of the `1/b` expansion, and
the "tracks over the whole range" wording is restricted accordingly.

**18. The promised positive-definiteness fallback was not implemented.** The paper stated a
fallback from the two-scale to the batch-means quadratic form when the former is not positive,
and stated that it never triggered; the code only clipped at zero. The fallback is now
implemented and its rate is counted and reported.

**19. A test was passing for the wrong reason and one was failing.** The test validating the
streaming dyadic accumulators used the wrong block offset under warm-up. Fixed, and extended to
check the two-scale accumulator as well; it now agrees with an explicit computation to `2e-15`.

## Also fixed

Notation collisions (the ridge exponent versus the two mixing decay rates; the curvature vector
versus the normal distribution function; the orthogonal matrix versus the autoregressive
coefficient matrix; the arbitrary small constant versus the block score); a terminology
paragraph defining score, `M`-estimator, sandwich, batch means and the mixing coefficient; the
`b` regime inconsistency between a figure caption and the theorem; figure captions that named
no axes; the abstract's hardcoded numbers, now macro-injected like the rest; and the honest
statement of the warm-up's one-off `O(c_w d^3)` cost, which is why the measured exponent in `d`
exceeds one at the `N` used in the cost table.

## Findings we considered and did not act on

* Re-deriving the covariance estimator's variance under an explicit sub-exponential condition
  to remove the factor `d` in the operator-norm rate. We state the factor instead and confine
  every rate claim to fixed `d`.
* A Berry–Esseen bound, and a rate for the estimator of `S` that is optimal within our own
  information set. Both are listed in the limitations as things we do not prove.
