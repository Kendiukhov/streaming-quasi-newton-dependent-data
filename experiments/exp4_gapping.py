"""Curvature schedules at matched cost: deterministic gapping vs Bernoulli thinning.

Proposition (gapping): from a stream of length N, taking n rank-one curvature updates at a
deterministic gap m = N/n gives  n Var[Hhat] = Gamma_0^H + O(rho^m), whereas Bernoulli(p =
n/N) thinning of every index gives  n Var[Hhat] = Gamma_0^H + (S^H - Gamma_0^H)/m.  The
excess over the independent-data benchmark is therefore exponentially small in m for
gapping and of order (kappa_H - 1)/m for Bernoulli thinning.

We freeze theta at theta* so that the optimisation dynamics cannot confound the comparison,
run both schedules on the same streams at a matched *expected* number of updates, and
compare the Monte Carlo variance of Hbar with the two predictions.  For AR(1) Gaussian
covariates, Cov(x_i x_i', x_{i+k} x_{i+k}') decays like phi^{2k}, so
kappa_H = (1 + phi^2)/(1 - phi^2) in closed form.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401
import numpy as np
from joblib import Parallel, delayed
import common
from bgsn import dgp as DG, _core

D = 20
N = 200_000
R = int(os.environ.get("R", 200))
MS = [1, 2, 5, 10, 20, 40, 80]
PHI = 0.7


def one(seed, g, m, mode):
    X, y = g.sample(N, 900_000 + seed)
    X = np.ascontiguousarray(X)
    if mode == "gap":
        mm, p = m, 1.0
    else:                      # Bernoulli thinning at the same expected update count
        mm, p = 1, 1.0 / m
    Ainv, W, nc = _core.curvature_only(X, g.theta_star, mm, p, 0.0, 0.0, 0.4, 1e-6,
                                       0 if g.model == "linear" else 1, seed, 20)
    return np.linalg.inv(W * Ainv), nc


if __name__ == "__main__":
    g = DG.make_lin_hom(d=D, phi=PHI, psi=0.7)
    H = g.oracle.H
    kappa_H = (1 + PHI ** 2) / (1 - PHI ** 2)
    rows = []
    for m in MS:
        row = dict(m=m, kappa_H=kappa_H)
        for mode in ("gap", "bern"):
            out = Parallel(n_jobs=common.NJOBS, batch_size=4)(delayed(one)(s, g, m, mode)
                                                   for s in range(R))
            Hs = np.array([o[0] for o in out])
            nc = float(np.mean([o[1] for o in out]))
            # scale-free summary: n * Var of the curvature estimate in the trace norm,
            # normalised by its independent-data benchmark at m = 1
            dev = Hs - H
            v = float(np.mean(np.sum(dev ** 2, axis=(1, 2))) * nc)
            row[f"nvar_{mode}"] = v
            row[f"nupd_{mode}"] = nc
        # predictions, normalised so that both curves start from the same benchmark
        row["pred_excess_bern"] = (kappa_H - 1) / m
        row["pred_excess_gap"] = (PHI ** (2 * m)) * 2 / (1 - PHI ** (2 * m)) if m else np.inf
        rows.append(row)
        print(f"m={m:3d} (n={row['nupd_gap']:.0f}/{row['nupd_bern']:.0f})  "
              f"n*Var gap={row['nvar_gap']:.4g}  bern={row['nvar_bern']:.4g}  "
              f"ratio={row['nvar_bern']/row['nvar_gap']:.3f}  "
              f"predicted excess: gap={row['pred_excess_gap']:.2e} "
              f"bern={row['pred_excess_bern']:.3f}")
    common.save("exp4_gapping.json",
                dict(d=D, N=N, R=R, phi=PHI, kappa_H=kappa_H, rows=rows))
