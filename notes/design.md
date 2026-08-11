# Design notes: Streaming inversion-free quasi-Newton for dependent data

## Base method (Godichon-Baggioni & Werge, JMLR 26(59), 2025; arXiv:2311.17753)
Update (streaming stochastic Newton, weighted Hessian, "SNA"):
  theta_{t+1} = theta_t - (1/(t+1)) Hbar_{t,w'}^{-1} gbar(theta_t; xi_{t+1})
  H_{t,w'} = H_0 + sum_{i<=t} log(i+1)^{w'} sum_{j<=n} Z_{ij} ( iota_{ij} e e^T + alpha_{ij} Phi_ij Phi_ij^T )
  Hbar = N_{t,Z}^{-1} H_{t,w'};  inverse maintained by Riccati/Sherman-Morrison (rank-1).
Cost: p d^2 N (curvature) + d N (gradient) + d^2 N / n (matvec). p = 1/d, n = d  =>  O(dN).
Result: sqrt(N)(theta - theta*) -> N(0, H^{-1} Sigma H^{-1}), Sigma = E[grad f grad f^T] at theta*.
Assumes i.i.d. blocks; noise is a martingale difference sequence (MDS) w.r.t. F_t.

## What breaks under temporal dependence
1. **Inference.** xi_{t+1} is not independent of F_t. The score process is serially
   correlated, so the correct limit is the *long-run* sandwich
      sqrt(N)(theta_N - theta*) -> N(0, H^{-1} S H^{-1}),  S = sum_{k in Z} Gamma_k,
      Gamma_k = Cov(grad f(theta*; xi_0), grad f(theta*; xi_k)).
   The base method's plug-in Sigma = Gamma_0 is *inconsistent* for S. CIs miscalibrated
   by the factor kappa = (e_j^T H^{-1} S H^{-1} e_j) / (e_j^T H^{-1} Gamma_0 H^{-1} e_j).
   Exact asymptotic coverage of a nominal 1-alpha dependence-blind CI:
      2*Phi(z_{alpha/2}/sqrt(kappa_j)) - 1.
2. **Optimization/consistency.** Robbins-Monro martingale arguments break: the noise
   is no longer MDS. Need blocking / m-dependent approximation / mixingale bounds.
3. **Curvature.** The rank-1 Riccati recursion aggregates *dependent* terms
   alpha_i Phi_i Phi_i^T. Consistency survives (ergodic thm) but the a.s. rate nu in
   ||Hbar_t^{-1} - H^{-1}|| = O(t^{-nu}) needs re-derivation; effective sample size
   shrinks by the integrated autocorrelation time.

## KEY CONCEPTUAL POINT (must be stated clearly; adversarial reviewers will ask)
Serial dependence in the **covariates alone** does NOT create a sandwich mismatch when
the model's conditional mean/score is correctly specified with i.i.d. innovations: then
the score is an MDS and S = Gamma_0. A genuine long-run-variance problem requires the
**score** to be serially correlated (serially correlated errors, latent dynamic factor,
omitted dynamics / misspecification). Dependence in covariates still matters for the
*curvature* estimator's effective sample size and finite-sample behaviour.
=> Our designs deliberately include both channels and we report both.

## Proposed method: BOSQN (Blocked/Gapped Streaming Quasi-Newton)
Two structural knobs, doing two different jobs:
* **Blocking** (block length b_t): the gradient step uses a *contiguous* block average.
  b Var(gbar_block) -> S. Blocking is what makes the long-run variance estimable,
  via non-overlapping batch means (BM) on the block gradients -- at zero extra cost
  and with no extra bandwidth parameter.
* **Gapping** (gap m): curvature rank-1 updates use only every m-th observation
  (equivalently: one designated slot per block). Decorrelates curvature terms,
  restores near-i.i.d. Riccati analysis, and cuts cost.
Neither alone suffices: gapping does not fix inference; blocking alone leaves the
curvature recursion with dependent increments. Theorem states the admissible triple.

**Long-run covariance estimator (streaming, O(d^2) per block, O(d^2 N / b) total):**
   Shat_T = (b/T) sum_{t<=T} (gbar_t(theta_t)) (gbar_t(theta_t))^T          (uncentred BM)
Consistency: drift H(theta_t - theta*) contributes O(log T / T) to Shat -> negligible.
Bias O(1/b) + Var O(1/T) => need b -> infinity and T -> infinity: b ~ N^beta, beta in (0,1).
If b >= d then LRV estimation is free inside the O(dN) budget.

CI: half-width z * sqrt( u_j^T Shat u_j / N ), u_j = Hbar^{-1} e_j (already maintained).
One-time O(d^3) for the full sandwich, or O(d^2) per requested coordinate.

## Honest accounting
* TIME is O(dN) amortized; MEMORY is O(d^2) (same as base method: stores Hbar^{-1}).
  Must state this plainly -- do not claim O(d) memory.

## Baselines
1. SNA-iid: base method + plug-in (dependence-blind) CI. [shows the failure]
2. SNA-iid + our BM variance. [ablation: is the blocking of the *gradient* needed?]
3. ASGD (Polyak-Ruppert) + plug-in.
4. ASGD + batch-means (Chen et al. 2020 / Zhu-Chen-Wu 2023 style).
5. ASGD + random scaling (Lee, Liao, Seo, Shin) -- valid under dependence, no S estimate.
6. AdaGrad (streaming variant from base paper) -- diagonal only.
7. Offline reference: full-sample M-estimator + Newey-West HAC. [gold standard, O(Nd^2)]

## Experiments
E1. Linear regression, AR(1) covariates + AR(1) errors. Sweep phi. Coverage, width, MSE, time.
E2. Logistic regression with latent AR(1) factor (score serially correlated).
E3. Markov-chain regime-switching covariates (2-state HMM) -- non-Gaussian, non-linear dep.
E4. Ill-conditioned design (the base paper's selling point) x dependence. Show Newton >> ASGD.
E5. Theory check: predicted coverage 2*Phi(z/sqrt(kappa))-1 vs measured, 3-digit match.
E6. Real stream: Metro Interstate Traffic Volume (UCI) and/or Beijing PM2.5.
    (a) real covariates + synthetic response with known theta*: exact coverage.
    (b) fully real: disjoint-replicate protocol, target = full-series M-estimator.
E7. Ablations: block length b, gap m, Bernoulli p; robustness to misspecified b.
