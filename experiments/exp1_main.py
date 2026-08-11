"""Main comparison: coverage, interval width, efficiency and time on four dependent DGPs.

Every method sees the same stream in the same blocks.  First-order baselines are given the
TRUE Hessian for their sandwich (an advantage no practitioner has) and their step-size
constants are tuned on pilot seeds disjoint from the evaluation seeds.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401
import numpy as np
import common
from bgsn import dgp as DG, streams

D = 20
N = 200_000
B = 20
R = int(os.environ.get("R", 300))
R_HAC = int(os.environ.get("R_HAC", 60))

FAST = [m for m in common.METHODS if m != "Offline-HAC"]
ORACLE_CACHE = os.path.join(common.RESULTS, "logit_oracle_d%d.npz" % D)


def logit_dgp():
    """Build the logistic DGP, caching its simulated long-run covariance oracle."""
    th = np.linspace(-1.0, 1.0, D)
    Sig = streams.illconditioned_cov(D, np.random.default_rng(30_000), 1.0)
    if os.path.exists(ORACLE_CACHE):
        z = np.load(ORACLE_CACHE)
        orc = streams.Oracle(theta_star=th, H=z["H"], Gamma0=z["G0"], S=z["S"],
                             exact=False, oracle_error=float(z["err"]),
                             meta=dict(dgp="logit_copula", cached=True))
    else:
        print("  [building logistic oracle by long simulation ...]")
        t0 = time.time()
        orc = streams.logit_copula_oracle(D, 0.7, 0.8, 1.0, th, Sig,
                                          n_sim=1_000_000, n_rep=8, lag=100)
        np.savez(ORACLE_CACHE, H=orc.H, G0=orc.Gamma0, S=orc.S,
                 err=orc.oracle_error)
        print(f"  [done in {time.time()-t0:.0f}s, relative MC error of the sandwich "
              f"diagonal = {orc.oracle_error:.2e}]")
    return DG.make_logit_copula(d=D, phi=0.7, psi=0.8, cond_exponent=1.0, oracle=orc)


def build():
    return {
        "AR-hom":   DG.make_lin_hom(d=D, phi=0.7, psi=0.7),
        "AR-het":   DG.make_lin_het(d=D, phi_lo=0.0, phi_hi=0.9, psi=0.7),
        "Markov":   DG.make_markov_cov(d=D, stay=0.97, psi=0.8, noise_scale=0.05),
        "Logistic": logit_dgp(),
    }


if __name__ == "__main__":
    out = {}
    for label, g in build().items():
        print(f"== {label}: {g.name}  kappa in [{g.oracle.kappa.min():.2f}, "
              f"{g.oracle.kappa.max():.2f}]  cond(H)={np.linalg.cond(g.oracle.H):.0f}")
        t0 = time.time()
        tuned = common.tune_first_order(g, N, B)
        print(f"   tuned: c_asgd={tuned['c_asgd']:.4g} c_adagrad={tuned['c_adagrad']:.4g}"
              f"  ({time.time()-t0:.0f}s)")
        recs = common.run_mc(g, N, B, FAST, R, tuned)
        recs += common.run_mc(g, N, B, ["Offline-HAC"], R_HAC, tuned)
        rows = common.summarise(recs, g.oracle)
        out[label] = dict(
            dgp=g.name, model=g.model, d=D, N=N, b=B, R=R, R_hac=R_HAC,
            tuned={k: v for k, v in tuned.items() if not k.endswith("_grid")},
            tuning_grids={k: v for k, v in tuned.items() if k.endswith("_grid")},
            kappa=g.oracle.kappa.tolist(),
            cond_H=float(np.linalg.cond(g.oracle.H)),
            oracle_exact=bool(g.oracle.exact),
            oracle_error=g.oracle.oracle_error,
            rows=rows)
        for r in sorted(rows, key=lambda r: -r["coverage"]):
            print(f"   {r['method']:16s} cov={r['coverage']:.3f}±{r['coverage_se']:.3f} "
                  f"[{r['coverage_min']:.2f},{r['coverage_max']:.2f}] "
                  f"width={r['width_rel']:.2f} rmse={r['rmse_rel']:.2f} "
                  f"t={r['secs_median']*1e3:.0f}ms")
        print(f"   ({time.time()-t0:.0f}s total)")
    common.save("exp1_main.json", out)
