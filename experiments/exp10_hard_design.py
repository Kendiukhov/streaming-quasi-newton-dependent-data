"""On the designs where the point estimate has not converged, whose fault is the undercoverage?

Table 1 reports every design at the same N = 200,000 so that the columns are comparable.  On two
of the four designs that N is not enough for the streaming *point* estimate to reach its
asymptotic regime, for two different reasons:

* Markov: the covariate process is a persistent regime chain with an integrated autocorrelation
  time of about 66 observations, so the effective sample size is a small fraction of N.
* Logistic: the curvature weight alpha = sigma'(x'theta) is largest at theta_0 = 0, where the
  warm-up evaluates it, so the preconditioner built there overestimates the curvature by a
  factor of up to four in some directions and the early steps are correspondingly too short.

An interval built from the *correct* asymptotic variance must undercover in that situation,
whatever estimator supplied the variance, because sqrt(N)(theta_T - theta*) is not yet close to
its limit.  This experiment separates the two explanations.  For a sweep in N we report, side by
side, the coverage of our interval and the coverage of the infeasible interval that uses the
exact H^{-1} S H^{-1} at the same iterate.  If our variance estimator were at fault the two
would diverge; if the point estimate simply has not converged they will agree at every N and
both will rise to nominal together, while the dependence-blind interval will not.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401
import numpy as np
from joblib import Parallel, delayed
import common
from bgsn import dgp as DG, estimators as E

D = 20
# (N, R) pairs.  R shrinks with N: the quantity being measured is the *gap* between our
# coverage and the oracle's, which is 0.02-0.08 on these designs, so a Monte Carlo standard
# error around 0.01 is ample, and the memory footprint at the largest N is what limits us.
SWEEP = [(100_000, 200), (200_000, 200), (500_000, 150)]
R = int(os.environ.get("R", 0)) or None
DESIGNS = [
    ("Markov", lambda: DG.make_markov_cov(d=D, stay=0.97, psi=0.8, noise_scale=0.05),
     (1 + 0.97) / (1 - 0.97)),
    ("Logistic", lambda: common.logit_dgp(d=D, phi=0.7, psi=0.8), None),
]


def run_design(g, tau):
    print(f"\n== {g.name}: kappa={g.oracle.kappa.min():.2f} "
          f"cond(H)={np.linalg.cond(g.oracle.H):.0f}"
          + (f"  integrated autocorrelation time={tau:.0f}" if tau else ""), flush=True)
    ts, H, S = g.theta_star, g.oracle.H, g.oracle.S
    V = np.diag(g.oracle.sandwich_true)

    def one(N, s):
        X, y = g.sample(N, 700_000 + s)
        f = E.streaming_newton(X, y, g.model, b=20, seed=s, variance="both")
        o = E.oracle_fit(f.theta, f.N, H, S)
        u = f.extras["W"] * f.extras["Ainv"]
        se_pl = np.sqrt(np.maximum(np.diag(u @ f.extras["Gamma0_hat"] @ u), 0.0) / f.N)
        return (f.theta - ts, f.covers(ts), o.covers(ts),
                np.abs(f.theta - ts) <= common.Z95 * se_pl, f.extras["n_clip"])

    rows = []
    for N, Rdef in SWEEP:
        Rn = R or Rdef
        t0 = time.time()
        # Each worker holds an N x d design plus the response, so peak memory grows with
        # n_jobs * N.  Cap the product: on a shared machine, thrashing costs far more than
        # the lost parallelism.  ~4e6 observations in flight is comfortable at d = 20.
        nj = max(1, min(common.NJOBS, int(1_000_000 // N)))
        out = Parallel(n_jobs=nj, batch_size=1)(delayed(one)(N, s) for s in range(Rn))
        Em = np.array([o[0] for o in out])
        rmse_rel = np.sqrt((Em ** 2).mean(0)) / np.sqrt(V / N)
        r = dict(N=N, R=Rn,
                 cov_bgsn=float(np.mean([o[1] for o in out])),
                 cov_oracle=float(np.mean([o[2] for o in out])),
                 cov_plugin=float(np.mean([o[3] for o in out])),
                 cov_bgsn_se=float(np.mean([o[1] for o in out], axis=1).std(ddof=1)
                                   / np.sqrt(Rn)),
                 rmse_rel_median=float(np.median(rmse_rel)),
                 rmse_rel_max=float(rmse_rel.max()),
                 # the step-length safeguard should bind a handful of times and NOT more often
                 # as N grows: that is the empirical content of "eventually inactive"
                 n_clip=float(np.median([o[4] for o in out])),
                 secs=time.time() - t0)
        rows.append(r)
        print("  N=%9d  ours=%.3f+-%.3f  oracle-variance=%.3f  dependence-blind=%.3f  "
              "rmse/efficient=%.2f  clips=%.0f  (%.0fs)"
              % (N, r["cov_bgsn"], r["cov_bgsn_se"], r["cov_oracle"], r["cov_plugin"],
                 r["rmse_rel_median"], r["n_clip"], r["secs"]), flush=True)
        del out
    gap = max(abs(r["cov_bgsn"] - r["cov_oracle"]) for r in rows)
    print(f"  largest |ours - oracle-variance| over the sweep: {gap:.4f}", flush=True)
    return dict(dgp=g.name, d=D, b=20, tau=tau,
                kappa=float(g.oracle.kappa.min()),
                cond_H=float(np.linalg.cond(g.oracle.H)),
                max_gap_to_oracle=gap, rows=rows)


if __name__ == "__main__":
    res = {}
    for label, mk, tau in DESIGNS:
        res[label] = run_design(mk(), tau)
    res["max_gap_to_oracle"] = max(v["max_gap_to_oracle"] for v in res.values()
                                   if isinstance(v, dict))
    print(f"\nlargest gap to the oracle interval over both designs and all N: "
          f"{res['max_gap_to_oracle']:.4f}", flush=True)
    common.save("exp10_hard_design.json", res)
