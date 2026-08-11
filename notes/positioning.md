# Positioning after the competitor scan (this is binding — do not overclaim)

## MUST NOT CLAIM
1. NOT the two-scale / flat-top / lugsail LRV correction itself. Prior art:
   Politis & Romano (1995); Politis (2011); Liu & Flegal (2018); Vats & Flegal (2022,
   Biometrika, "lugsail"); and applied to SGD covariance in **Singh, Shukla & Vats,
   JMLR 26(258), 2025, Eqs. (15) and (17)** — algebraically identical to ours
   (2 S(2b) - S(b) = S_bm + b(Lam1 + Lam1')). Their Corollary 1 shows the *rate* is
   unchanged and they call an exact bias expression an open problem.
   -> We may claim only: the O(rho^b) bias order at *block-gradient* resolution under
      geometric mixing of the data, and the streaming (dyadic) implementation.
2. NOT the O(dN) cost result / thinned curvature updates. Base paper's own
   p = 1/d, n = d construction; also Godichon-Baggioni, Portier & Salle (2026) mSNA;
   Chen, Lai, Li & Zhang thin Hessian entries with p = O(1/d^2).
3. NOT "first CLT with a long-run sandwich under dependence". Prior: Liu, Chen & Shang
   (phi-mixing, Thm 2); Li, Liang & Zhang (Markov, nonlinear SA, Thm 1 FCLT);
   Samsonov et al. (linear SA, + Berry-Esseen); Li, Liang, Chen & Zhang (controlled
   Markov). All FIRST-ORDER.
4. NOT "we establish asymptotic efficiency under dependence" — Li, Liang & Zhang Thm 3
   owns the semiparametric bound G^{-1} S G^{-T}. We ATTAIN a known bound.
5. NOT "first online long-run covariance estimator under dependent streaming data" —
   Roy & Balasubramanian (Thms 2.1/2.2) and Samsonov et al. (Cor 2) do it for
   first-order SA, both at n^{-1/8}.
6. NOT blocked gradients as a dependence-breaking device — Godichon-Baggioni, Werge &
   Wintenberger (TMLR) own it (n_t = C t^rho, and mu_nu > 0 needs fixed C large enough).
7. NOT "first to notice undercoverage of dependence-blind intervals" — Liu, Chen & Shang
   Proposition 1 (their simulated coverage "hovers around 85%"; our formula gives 83.4%
   for their kappa = 2 — use as external validation).
8. NOT "first online covariance estimator for a streaming second-order method" —
   Kuang, Anitescu & Na (Thm 4.4/Cor 4.5); Wang, Du & Na (Thm 4.6). Both i.i.d.

## WHAT WE DO CLAIM (scoped)
* **T1.** Convergence, a.s. rate, and a CLT with the long-run sandwich for the
  **inversion-free rank-one (Riccati/Sherman-Morrison) curvature recursion with gapped
  updates**, under geometric mixing. Existing dependent-data theory is first-order;
  existing second-order streaming inference (Kuang et al.; Wang et al.) assumes
  **conditionally unbiased gradients** (their Assumption 3.2) — precisely what mixing
  destroys. Ours is the second-order, inverse-maintaining, dependent case.
* **T2 (the sharpest wedge).** Our S-estimator is built from **block gradients**, whose
  serial dependence is inherited from the *data* process, not from the *iterate* path.
  Iterate-based batch means must let the block length grow with n so the iterates mix
  within a block (Samsonov et al. Prop. 2 needs b_n >~ n^gamma), which caps the rate at
  n^{-1/8}. Block gradients need only b >~ log N under geometric mixing, giving
  Otilde(N^{-1/2}). This does not contradict the Ni & Huo minimax rate
  Theta(n^{-(1-alpha)/2}) because that lower bound is for **Hessian-free** estimators
  that observe only the iterate path; a quasi-Newton method already maintains curvature
  and forms block gradients, so it is outside that class. STATE THIS EXPLICITLY.
* **T3.** O(rho^b) bias of the two-scale correction at block-gradient resolution under
  geometric mixing (Singh et al. leave the bias order open; their Cor 1 = rate parity).
  Plus the streaming/dyadic implementation, answering their Remark 4 ("an online
  implementation strategy for EBS estimators remains an open problem").
* **T4.** Gapping vs Bernoulli thinning at matched cost: excess curvature variance
  O(rho^m) vs O(1/m). Modest but exact, and it *replaces* the base method's cost knob
  with a strictly better one.
* **T5.** Exact coverage functional 2 Phi(z/sqrt(kappa)) - 1 for dependence-blind
  intervals, verified to three digits; kappa = (1+phi psi)/(1-phi psi) presented as an
  illustrative corollary (it is textbook), not as a result.
* **T6.** Blocking is *free* for this algorithm's point estimate: with gamma_t = 1/t the
  Newton recursion's leading term is -(1/N) H^{-1} sum_{i<=N} g_i **for every b**, so the
  asymptotic variance does not depend on b. This answers Li, Liang, Chen & Zhang's
  anti-batching argument (Thm 2) head-on. VERIFY NUMERICALLY (efficiency vs b flat).

## OBJECTIONS TO PRE-EMPT IN THE TEXT
* Samsonov Prop. 2 block-growth obstruction -> answered by T2 + the numerical
  decomposition showing the theta_t-drift contributes ~0.001 of S_hat (measured).
* Kuang et al. Sec 4.3 ("directly applicable to conditioned SGD") -> their Assumption 3.2
  requires E[gbar_t | F_{t-1}] = grad F, which fails under mixing.
* Anti-batching (single-sample stream SGD is more sample-efficient) -> T6.
* PSD of S_ft: lugsail estimators need not be PSD (Singh et al. Remark 5). Report the
  empirical fallback rate and document the fallback rule.
* Random-scaling critical values: the correct TWO-SIDED 95% value is 6.811
  (Kiefer-Vogelsang-Bunzel 2000 Table I, 97.5% quantile), reported in Lee-Liao-Seo-Shin
  Table 2 as 3.890 / 5.374 / 6.811 / 8.544 for 90/95/97.5/99%. Several papers use 5.32
  labelled "95%", which is the ONE-SIDED 95% value. Simulate ours and say so.
