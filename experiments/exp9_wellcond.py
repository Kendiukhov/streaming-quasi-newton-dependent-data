"""A well-conditioned dependent design, to separate two effects that the main table mixes.

On the ill-conditioned designs of Table 1 the first-order baselines' *point* estimates have
not converged (their RMSE is several times the efficient value), so their intervals are
uninformative for the question this paper is about.  That is the base method's own motivation
-- first-order methods crawl when the curvature is ill-conditioned -- but it means Table 1
cannot answer "does the long-run variance estimator help a first-order method too?".

Here the covariate covariance is the identity, so averaged SGD is efficient and the only thing
that can go wrong is the variance estimate.  Everything else (dependence, block length,
sample size) is unchanged.  If our construction is really about inference rather than about
conditioning, then on this design every method's point estimate should be efficient and only
the choice of covariance should decide coverage.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401
import numpy as np
import common
from bgsn import dgp as DG

D = 20
N = 200_000
B = 20
R = int(os.environ.get("R", 300))
METHODS = ["BGSN", "BGSN-BM", "BGSN-plugin", "SN-iid", "ASGD-2scale", "ASGD-plugin",
           "ASGD-RS", "AdaGrad-2scale", "Oracle-var"]

if __name__ == "__main__":
    # cond_exponent = 0 makes Sigma_x = I; the dependence (phi = psi = 0.7) is unchanged
    g = DG.make_lin_hom(d=D, phi=0.7, psi=0.7, cond_exponent=0.0)
    print(f"well-conditioned design: cond(H)={np.linalg.cond(g.oracle.H):.2f}  "
          f"kappa in [{g.oracle.kappa.min():.2f}, {g.oracle.kappa.max():.2f}]", flush=True)
    t0 = time.time()
    tuned = common.tune_first_order(g, N, B)
    print(f"   tuned: c_asgd={tuned['c_asgd']:.4g} c_adagrad={tuned['c_adagrad']:.4g}"
          f"  ({time.time()-t0:.0f}s)", flush=True)
    recs = common.run_mc(g, N, B, METHODS, R, tuned)
    rows = common.summarise(recs, g.oracle)
    for r in sorted(rows, key=lambda r: -r["coverage"]):
        print(f"   {r['method']:16s} cov={r['coverage']:.3f}+-{r['coverage_se']:.3f} "
              f"width={r['width_rel']:.2f} rmse={r['rmse_rel']:.2f}", flush=True)
    common.save("exp9_wellcond.json",
                dict(dgp=g.name, d=D, N=N, b=B, R=R,
                     cond_H=float(np.linalg.cond(g.oracle.H)),
                     kappa=g.oracle.kappa.tolist(),
                     tuned={k: v for k, v in tuned.items() if not k.endswith("_grid")},
                     tuning_grids={k: v for k, v in tuned.items() if k.endswith("_grid")},
                     rows=rows))
