"""Long-run covariance estimation: bias in b, the two-scale correction, and the drift.

Three claims are tested.
(a) E[S_bm] = S - Delta/b + O(rho^b) with Delta = sum_{k>=1} k (Gamma_k + Gamma_k'): we compare the
    measured ratio with the closed-form prediction 1 - Delta/(bS), which for the
    homogeneous AR design is a scalar.
(b) E[S_ft] = S + O(rho^b): the two-scale estimator should be unbiased at every b.
(c) the theta_t-drift contributes O(b/N) and is negligible: we decompose the algorithm's
    block gradients into the score part at theta* and the drift part Hhat_t(theta_{t-1}-theta*)
    and evaluate the long-run covariance of each separately.  This is the direct answer to
    the objection that iterate blocks must grow for the block average to mix.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401
import numpy as np
from joblib import Parallel, delayed
import common
from bgsn import dgp as DG, estimators as E

D = 20
N = 200_000
R = int(os.environ.get("R", 80))
BS = [5, 10, 20, 40, 80, 160, 320]
PHI = PSI = 0.7


def block_lrv(Gb, b):
    T = len(Gb)
    S = b * (Gb[:, None, :] * Gb[:, :, None]).mean(axis=0)
    C = np.einsum('ti,tj->ij', Gb[:-1], Gb[1:]) / (T - 1)
    return S, S + b * (C + C.T)


def one(seed, g, b):
    X, y = g.sample(N, 900_000 + seed)
    f = E.streaming_newton(X, y, g.model, b=b, seed=seed, traj_every=1)
    d = g.d
    th = f.extras["traj_theta"]           # post-step iterate for each stepping block
    nw = f.extras["n_warm"]               # leading blocks used for curvature only
    T = len(th)
    # block j of the recorded trajectory is stream block nw + j; the *pre-step* parameter
    # for that block is th[j-1] (and theta_0 = 0 for j = 0)
    Xb = X[nw * b:(nw + T) * b].reshape(T, b, d)
    yb = y[nw * b:(nw + T) * b].reshape(T, b)
    thpre = np.vstack([np.zeros((1, d)), th[:-1]])
    # algorithm's block gradients, and the exact decomposition for a linear model:
    #   gbar_t(theta_{t-1}) = Hhat_t (theta_{t-1}-theta*) - (1/b) sum x_i u_i
    r = np.einsum('tbd,td->tb', Xb, thpre) - yb
    Gb_alg = np.einsum('tbd,tb->td', Xb, r) / b
    u = yb - np.einsum('tbd,d->tb', Xb, g.theta_star)
    Gb_score = -np.einsum('tbd,tb->td', Xb, u) / b
    Hb = np.einsum('tbi,tbj->tij', Xb, Xb) / b
    Gb_drift = np.einsum('tij,tj->ti', Hb, thpre - g.theta_star)
    k = slice(T - f.extras["n_blocks_kept"], T)
    dS = np.diag(g.oracle.S)
    res = {}
    for nm, G in (("alg", Gb_alg), ("score", Gb_score), ("drift", Gb_drift)):
        sb, sf = block_lrv(G[k], b)
        res[nm] = (np.diag(sb) / dS, np.diag(sf) / dS)
    res["internal"] = (np.diag(f.extras["S_bm"]) / dS, np.diag(f.extras["S_ft"]) / dS)
    res["nonpsd"] = float(np.mean(np.diag(f.extras["S_ft"]) <= 0))
    return res


if __name__ == "__main__":
    g = DG.make_lin_hom(d=D, phi=PHI, psi=PSI)
    r = PHI * PSI
    kappa = (1 + r) / (1 - r)
    delta_over_S = 2 * (r / (1 - r) ** 2) / kappa      # Delta / S, a scalar here

    def exact_bm_ratio(b):
        """E[S_bm]/S exactly: sum_{|k|<b}(1-|k|/b) r^{|k|} divided by (1+r)/(1-r).

        This is the exact Bartlett-weighted sum, not its 1/b expansion, so the comparison
        with the measurement has no approximation of its own at small b.
        """
        k = np.arange(1, b)
        return float((1.0 + 2.0 * np.sum((1 - k / b) * r ** k)) / kappa)

    rows = []
    for b in BS:
        out = Parallel(n_jobs=common.NJOBS, batch_size=2)(delayed(one)(s, g, b) for s in range(R))
        row = dict(b=b, N=N, R=R,
                   theory_bm=exact_bm_ratio(b),
                   theory_bm_asymptotic=1.0 - delta_over_S / b,
                   nonpsd_rate=float(np.mean([o["nonpsd"] for o in out])))
        for nm in ("alg", "score", "drift", "internal"):
            row[f"bm_{nm}"] = float(np.mean([o[nm][0] for o in out]))
            row[f"ft_{nm}"] = float(np.mean([o[nm][1] for o in out]))
            row[f"bm_{nm}_se"] = float(np.std([np.mean(o[nm][0]) for o in out],
                                              ddof=1) / np.sqrt(R))
            row[f"ft_{nm}_se"] = float(np.std([np.mean(o[nm][1]) for o in out],
                                              ddof=1) / np.sqrt(R))
        rows.append(row)
        print(f"b={b:4d} BM {row['bm_alg']:.4f} (theory {row['theory_bm']:.4f})  "
              f"2scale {row['ft_alg']:.4f}±{row['ft_alg_se']:.4f}  "
              f"drift-only contribution to 2scale {row['ft_drift']:.5f}  "
              f"non-PSD diag {row['nonpsd_rate']:.3f}")
    common.save("exp3_lrv.json",
                dict(d=D, N=N, R=R, phi=PHI, psi=PSI, kappa=kappa,
                     delta_over_S=delta_over_S, rows=rows))
