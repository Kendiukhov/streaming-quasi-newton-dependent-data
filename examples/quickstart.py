"""Minimal end-to-end example: fit a model on a dependent stream and get valid intervals.

    python examples/quickstart.py

Prints, for a simulated stream whose population quantities are known exactly:
  * the coverage a dependence-blind interval would achieve (and the value predicted by the
    coverage functional 2*Phi(z/sqrt(kappa)) - 1),
  * the coverage our interval achieves,
  * the free block-adequacy diagnostic that tells you whether the block length is big enough.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from bgsn import dgp as DG, estimators as E   # noqa: E402

N, D, B, R = 200_000, 20, 20, 200
PHI = PSI = 0.7

g = DG.make_lin_hom(d=D, phi=PHI, psi=PSI)     # AR(1) covariates AND AR(1) errors
kappa = float(g.oracle.meta["kappa_scalar"])   # (1 + phi psi) / (1 - phi psi)
print(f"stream: N={N:,} d={D}, AR({PHI}) covariates with AR({PSI}) errors")
print(f"variance inflation caused by the dependence: kappa = {kappa:.3f}")
print(f"predicted coverage of a dependence-blind 95% interval: "
      f"{2 * norm.cdf(1.959963985 / np.sqrt(kappa)) - 1:.3f}\n")

cov_ours = cov_blind = 0
adequacy = []
for s in range(R):
    X, y = g.sample(N, 1000 + s)
    fit = E.streaming_newton(X, y, "linear", b=B, seed=s, variance="both")

    Hinv = fit.extras["W"] * fit.extras["Ainv"]          # the maintained curvature inverse
    q = lambda S: np.maximum(np.diag(Hinv @ S @ Hinv), 0.0)   # noqa: E731
    se_ours = np.sqrt(q(fit.extras["S_ft"]) / fit.N)     # long-run  (valid)
    se_blind = np.sqrt(q(fit.extras["Gamma0_hat"]) / fit.N)   # i.i.d.  (invalid)

    err = fit.theta - g.theta_star
    cov_ours += np.mean(np.abs(err) <= 1.959963985 * se_ours)
    cov_blind += np.mean(np.abs(err) <= 1.959963985 * se_blind)
    adequacy.append(np.mean(np.abs(np.diag(Hinv @ (fit.extras["S_ft"]
                                                  - fit.extras["S_bm"]) @ Hinv))
                            / q(fit.extras["S_ft"])))

print(f"over {R} replications, nominal 95%:")
print(f"  dependence-blind interval (i.i.d. plug-in variance): {cov_blind / R:.3f}")
print(f"  BGSN interval (long-run variance)                  : {cov_ours / R:.3f}")
print(f"  block-adequacy diagnostic r_j (small => b is enough): {np.mean(adequacy):.3f}")
