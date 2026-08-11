"""Theory check: the exact asymptotic coverage of dependence-blind intervals.

Proposition (coverage): a nominal 1-alpha dependence-blind interval has asymptotic coverage
2 Phi(z_{alpha/2} / sqrt(kappa)) - 1, where kappa is the variance inflation.  For the
homogeneous AR design kappa = (1 + phi psi)/(1 - phi psi) for every coordinate, so the
prediction is a single parameter-free curve.

All four intervals (two-scale, batch means, i.i.d. plug-in, oracle) are derived from ONE fit
per replicate: they share the point estimate and differ only in which covariance is put in
the sandwich.  That is both four times cheaper and a cleaner comparison, since nothing but
the interval changes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
import common
from bgsn import dgp as DG, estimators as E

D = 20
N = 200_000
B = 20
R = int(os.environ.get("R", 300))
PHIS = [0.0, 0.25, 0.45, 0.6, 0.7, 0.8, 0.85]


def one(seed, g, N, b):
    X, y = g.sample(N, 900_000 + seed)
    f = E.streaming_newton(X, y, g.model, b=b, seed=seed, variance="both")
    U = f.extras["W"] * f.extras["Ainv"]
    Nu = f.N
    Hi = np.linalg.inv(g.oracle.H)

    def se(S, M=U):
        return np.sqrt(np.maximum(np.diag(M @ S @ M), 0.0) / Nu)
    out = dict(err=f.theta - g.theta_star,
               ft=se(f.extras["S_ft"]), bm=se(f.extras["S_bm"]),
               plugin=se(f.extras["Gamma0_hat"]),
               oracle=se(g.oracle.S, Hi),
               fallback=f.extras["n_psd_fallback"])
    return out


if __name__ == "__main__":
    rows = []
    for phi in PHIS:
        g = DG.make_lin_hom(d=D, phi=phi, psi=phi)
        kap = float(g.oracle.meta["kappa_scalar"])
        pred = 2 * norm.cdf(common.Z95 / np.sqrt(kap)) - 1
        res = Parallel(n_jobs=common.NJOBS, batch_size=3)(
            delayed(one)(s, g, N, B) for s in range(R))
        err = np.array([r["err"] for r in res])
        V = np.diag(g.oracle.sandwich_true)
        row = dict(phi=phi, psi=phi, kappa=kap, predicted_plugin_coverage=pred,
                   N=(N // B) * B, R=R,
                   psd_fallback_rate=float(np.mean([r["fallback"] for r in res]) / D),
                   eff=float(np.mean(np.diag(err.T @ err / R * (N // B) * B) / V)))
        for k, lab in (("ft", "BGSN"), ("bm", "BGSN-BM"), ("plugin", "BGSN-plugin"),
                       ("oracle", "Oracle-var")):
            se = np.array([r[k] for r in res])
            cm = np.abs(err) <= common.Z95 * se
            row[f"cov_{lab}"] = float(cm.mean())
            row[f"covse_{lab}"] = float(cm.mean(axis=1).std(ddof=1) / np.sqrt(R))
            row[f"width_{lab}"] = float(np.mean(se * np.sqrt(row["N"]) / np.sqrt(V)))
        rows.append(row)
        print(f"phi=psi={phi:.2f} kappa={kap:5.2f}  predicted={pred:.4f}  "
              f"measured(plugin)={row['cov_BGSN-plugin']:.4f}"
              f"+-{row['covse_BGSN-plugin']:.4f}  BGSN={row['cov_BGSN']:.4f}  "
              f"BM={row['cov_BGSN-BM']:.4f}  oracle={row['cov_Oracle-var']:.4f}",
              flush=True)
    err = [abs(r["cov_BGSN-plugin"] - r["predicted_plugin_coverage"]) for r in rows]
    print(f"\nmax |measured - predicted| over the sweep: {max(err):.4f}")
    common.save("exp2_coverage_law.json",
                dict(d=D, N=N, b=B, R=R, rows=rows, max_abs_error=float(max(err))))
