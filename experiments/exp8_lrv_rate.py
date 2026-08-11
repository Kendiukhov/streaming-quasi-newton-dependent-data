"""Does the long-run covariance estimator really converge at the claimed rate?

Theorem (long-run covariance) says
    ||S_ft - S|| = O_p( sqrt(b/N) + rho^b + (b+d) log^2 N / N ),
so with b ~ c log N the rate is Otilde(N^{-1/2}), whereas plain batch means are stuck at
    ||S_bm - S|| = O_p( sqrt(b/N) + 1/b ),
optimised at b ~ N^{1/3} for O_p(N^{-1/3}).

This experiment measures both directly over four decades of N under three block-length
schedules and fits the empirical exponent.  It is the one claim in the paper that a referee
is most likely to want checked numerically rather than taken on trust, because the headline
comparison with the N^{-1/8} rate of iterate-based batch means rests on it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401
import numpy as np
from joblib import Parallel, delayed
import common
from bgsn import dgp as DG, estimators as E

D = 20
R = int(os.environ.get("R", 30))
NS = [25_000, 50_000, 100_000, 200_000, 400_000]
SCHEDULES = {
    "log":   lambda N: max(int(np.ceil(3.0 * np.log(N))), 8),   # b ~ c log N  (our theory)
    "cbrt":  lambda N: max(int(np.ceil(N ** (1.0 / 3.0))), 8),  # b ~ N^{1/3}  (BM-optimal)
    "n34":   lambda N: max(int(np.ceil(N ** 0.75)), 8),         # b ~ N^{3/4}  (iterate BM)
}


def one(seed, g, N, b):
    """Errors of three estimators of the SAME object, plus the two S-estimators.

    The comparison that matters is of the whole sandwich Sigma = H^{-1} S H^{-1}, because
    that is what an interval uses and what the iterate-path literature estimates.  We
    therefore report (i) our composite Hbar^{-1} S_ft Hbar^{-1}, (ii) the iterate-path
    overlapping-batch-means estimator of Sigma at its required block length, and, for the
    bias/variance discussion, (iii)-(iv) the errors of S_ft and S_bm themselves.
    """
    X, y = g.sample(N, 900_000 + seed)
    f = E.streaming_newton(X, y, g.model, b=b, seed=seed, variance="twoscale",
                           traj_every=1)
    S = g.oracle.S
    Sig = g.oracle.sandwich_true
    U = f.extras["W"] * f.extras["Ainv"]
    nS, nSig = np.linalg.norm(S, 2), np.linalg.norm(Sig, 2)
    # iterate-path OBM on our own trajectory, at the block length that literature requires
    Th = f.extras["traj_theta"]
    bn = int(np.ceil(len(Th) ** 0.75))
    Sig_it = E.iterate_obm_sandwich(Th, (N // b) * b, bn)
    return (np.linalg.norm(f.extras["S_ft"] - S, 2) / nS,
            np.linalg.norm(f.extras["S_bm"] - S, 2) / nS,
            np.linalg.norm(U @ f.extras["S_ft"] @ U - Sig, 2) / nSig,
            np.linalg.norm(Sig_it - Sig, 2) / nSig)


if __name__ == "__main__":
    g = DG.make_lin_hom(d=D, phi=0.7, psi=0.7)
    out = {}
    for name, sched in SCHEDULES.items():
        rows = []
        for N in NS:
            b = sched(N)
            capped = False
            if b > N // 8:      # the estimator needs a handful of blocks to exist at all
                b, capped = N // 8, True
                print(f"  [cap ] schedule={name} N={N}: b capped at N/8 = {b}", flush=True)
            res = Parallel(n_jobs=common.NJOBS, batch_size=2)(
                delayed(one)(s, g, N, b) for s in range(R))
            A = np.array(res)
            row = dict(N=N, b=b, R=R, b_capped=capped)
            for j, k in enumerate(["ft", "bm", "sandwich", "iterate_obm"]):
                row[f"err_{k}"] = float(A[:, j].mean())
                row[f"err_{k}_se"] = float(A[:, j].std(ddof=1) / np.sqrt(R))
            rows.append(row)
            print(f"  schedule={name:5s} N={N:8d} b={b:6d}  "
                  f"S_ft {row['err_ft']:.4f}  S_bm {row['err_bm']:.4f}  "
                  f"sandwich(ours) {row['err_sandwich']:.4f}  "
                  f"sandwich(iterate OBM) {row['err_iterate_obm']:.4f}", flush=True)
        if len(rows) >= 3:
            ln = np.log([r["N"] for r in rows])
            ex = {k: float(np.polyfit(ln, np.log([r[f"err_{k}"] for r in rows]), 1)[0])
                  for k in ["ft", "bm", "sandwich", "iterate_obm"]}
            print(f"  -> schedule={name}: exponents  S_ft {ex['ft']:.3f}  "
                  f"S_bm {ex['bm']:.3f}  sandwich(ours) {ex['sandwich']:.3f}  "
                  f"sandwich(iterate OBM) {ex['iterate_obm']:.3f}")
            out[name] = dict(rows=rows, exponent_ft=ex["ft"], exponent_bm=ex["bm"],
                             exponent_sandwich=ex["sandwich"],
                             exponent_iterate_obm=ex["iterate_obm"])
        else:
            out[name] = dict(rows=rows)
    common.save("exp8_lrv_rate.json",
                dict(d=D, R=R, Ns=NS, schedules=list(SCHEDULES), results=out))
