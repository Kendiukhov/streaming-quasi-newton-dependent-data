# Adversarial review and what it changed

Before finalising, the manuscript and code were put through a five-lens adversarial review
(theory correctness, statistical/experimental design, positioning versus prior art, clarity,
code correctness), with every blocking/major finding independently verified against the files
before being accepted. Forty candidate findings; twenty-eight survived verification. Two further findings came
out of the final re-run rather than the review itself, and are recorded here too because they
changed the algorithm. The raw
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

**3. Our own method diverged on the regime-switching design, and the cause was not the one we
first assumed.** The first diagnosis was an under-long curvature warm-up, and raising
`warm_mult` from 50 to 200 reduced but did not remove the failure: at `warm_mult=200` the
relative error of the point estimate was still `>1` on 7 of 10 replicates, with coverage 0.27.
Tracing the iterates showed the step magnitudes *growing* over the first ten blocks
(6.3 -> 63 -> 234 -> 1963 -> 1.3e4 -> 8.0e4), i.e. an explosive linear recursion, not a slowly
decaying bias. Linearising the update identified the cause exactly: the error map is
`I - gamma_t Hbar^{-1} Hhat_t` with `Hhat_t` the *block* Hessian, and because
`Hbar^{-1} Hhat_t` is a product of PSD matrices this contracts only while
`gamma_t ||Hbar^{-1} Hhat_t|| < 2`. Measuring `||H^{-1} Hhat_t||` directly at `b=20` gave a
median of 20 on the AR design and 33 on the regime-switching design (block maxima 55 and 89),
so `gamma_1 = 1` violates the condition on *both* designs — the AR designs were simply lucky.
Neither gapping (`m=1` fails too) nor a larger `b` (which reduces the amplification but cuts the
number of steps, making the point estimate worse: rmse/efficient 1.82 at `b=20` rising
monotonically to 3.24 at `b=800`) is the fix. The fix is a step-size shift
`gamma_t = c/(t+t0)^a` with `t0` set from the contraction condition on the warm-up window,
eq. (t0) in the paper. This is now a stated contribution with a proposition and proof
(`prop:shift`, `app:shift`), an ablation panel, a unit test that checks the power-iteration
norm against `numpy.linalg.norm(...,2)` to 2e-16 and that the shift restores contraction, and a
limitations paragraph. Effect: on the regime-switching design, 0/10 divergences instead of
7/10, coverage 0.74 instead of 0.27; on the AR designs, coverage 0.948 instead of 0.956 and
relative variance 1.03 instead of 1.00 — a small, honestly reported cost.

**4. The regime-switching design's residual undercoverage was then traced to the central limit
theorem, not to the variance estimator, and this is now shown rather than asserted.** At
`N=2e5` that design's point estimate is still 1.9x the efficient standard error, so *any*
interval built from the correct asymptotic variance must undercover. A new experiment
(`exp10_hard_design.py`) sweeps `N` and reports our coverage beside the coverage of an
infeasible interval using the exact `H^{-1} S H^{-1}` at the same iterate. The two agree to
within 0.004 at every `N` and both rise to nominal (0.914 vs 0.916 at `N=2e6`) while the
dependence-blind interval stays flat. The paper reports every design at the same `N` in Table 1
and resolves this column in its own table and figure.

**5. A second five-lens adversarial review, run after the algorithm changes above, produced
46 candidate findings of which 26 survived independent verification.** The two blocking ones were
both real and both are fixed:

*The ridge lemma was circular.* The vanishing ridge was scaled by the *running* harmonic mean of
the curvature eigenvalues, `varsigma_n = d/tr(Hbar_n^{-1})`. Since that quantity is bounded above
by `d * lambda_min(Hbar_n)`, assuming it bounded below (as the proof did, in a parenthetical) is
equivalent to assuming the lemma's conclusion; and the only unconditional bound available,
`varsigma_k >= lambda_0/W_k = Theta(1/k)`, gives `sum iota_k = O(1)` instead of the required
`Omega(n^{1-q_r})`, with a Gronwall argument showing that is tight. The bound does hold once
`Hbar_t -> H > 0`, but that is three rungs higher in the paper's own ladder. Fixed by *freezing*
the ridge scale on the first curvature-bearing block (`_core.py`): it is then a positive finite
constant measurable with respect to that block, the sum bound holds unconditionally, and the
lemma sits at the bottom of the ladder as advertised. A remark now explains why freezing is
necessary rather than cosmetic.

*The contraction condition was stated as an equivalence with the wrong norm.* "Product of two PSD
matrices, so eigenvalues real and non-negative, so the map is a contraction exactly when
`gamma ||Hbar^{-1} Hhat|| < 2`" is a non-sequitur: real non-negative eigenvalues give a statement
about the *spectral radius*, not about the operator norm of a single step, and `lambda_max` is not
`||.||`. Restated: the spectral radius is below one exactly when
`gamma lambda_max(Hbar^{-1} Hhat) < 2`, for which the operator-norm version is sufficient, and the
text now says why we use the norm (computable, conservative) and what the condition does and does
not rule out.

Four further theory findings were also real: rung (iii) of the ladder (the preliminary almost sure
rate) was asserted but never proved -- now proved as `lem:prelim`, with a bootstrap, and it is the
only place `q_r < 1/4` is needed, so the default exponent was changed from 0.4 to 0.2 to sit
inside the range the theory covers; `lem:incr` was stated under `thm:curv`'s assumptions but used
at a lower rung -- now split into part (a) from the ridge alone and part (b) sharper under
`thm:curv`; two citations of `thm:curv` were really citations of `lem:ridge`; and one almost sure
bound was cited where an L2 bound was needed. The variance part of `thm:lrv` needed `4 + nu`
moments rather than exactly four, because Davydov applied to the quadratic block products needs a
margin -- the assumption now says so.

Three empirical claims were overstated or wrong and are corrected: the PSD fallback was claimed
never to trigger, but it was never recorded on the real streams and does trigger there (now
recorded and reported); the block-adequacy diagnostic divided by the two-scale quadratic form,
which can be negative, and produced numbers of order 1e30 on the traffic stream (now divided by
the batch-means form, which is PSD by construction); and the cost table's caption said "median of
7 runs" when the code reports the minimum over interleaved repetitions, and declared ten columns
while emitting eleven.

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
