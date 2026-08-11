"""Ablations and sensitivity: N, block length b, warm-up, ridge, and the p knob.

Each panel isolates one design choice.  All use the homogeneous AR design (d = 20,
kappa = 2.92) unless stated, so the numbers are directly comparable with Table 1.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401
import numpy as np
from joblib import Parallel, delayed
import common
from bgsn import dgp as DG, estimators as E

D = 20
R = int(os.environ.get("R", 250))


def mc(g, N, b, R=R, **kw):
    def one(s):
        X, y = g.sample(N, 900_000 + s)
        f = E.streaming_newton(X, y, g.model, b=b, seed=s, variance="both", **kw)
        u = f.extras["W"] * f.extras["Ainv"]
        q = lambda S: np.maximum(np.diag(u @ S @ u), 0.0)   # noqa: E731
        Nu = (N // b) * b
        return (f.theta - g.theta_star,
                np.sqrt(q(f.extras["S_ft"]) / Nu),
                np.sqrt(q(f.extras["S_bm"]) / Nu),
                np.sqrt(q(f.extras["Gamma0_hat"]) / Nu),
                f.extras["n_psd_fallback"] / max(len(f.theta), 1),
                common.adequacy(f),
                f.extras["t0"], f.extras["n_clip"], f.extras["n_warm"])
    out = Parallel(n_jobs=common.NJOBS, batch_size=3)(delayed(one)(s) for s in range(R))
    Em = np.array([o[0] for o in out])
    Nu = (N // b) * b
    V = np.diag(g.oracle.sandwich_true)
    res = dict(N=Nu, b=b, R=R,
               t0=float(np.median([o[6] for o in out])),
               n_clip=float(np.median([o[7] for o in out])),
               n_warm=float(np.median([o[8] for o in out])),
               eff=float(np.mean(np.diag(Em.T @ Em / R * Nu) / V)),
               nonpsd=float(np.mean([o[4] for o in out])),
               block_adequacy=float(np.nanmean([o[5] for o in out])))
    for j, lab in ((1, "ft"), (2, "bm"), (3, "plugin")):
        Se = np.array([o[j] for o in out])
        res[f"cov_{lab}"] = float(np.mean(np.abs(Em) <= common.Z95 * Se))
        res[f"cov_{lab}_se"] = float(np.mean(np.abs(Em) <= common.Z95 * Se, axis=1)
                                     .std(ddof=1) / np.sqrt(R))
        res[f"width_{lab}"] = float(np.mean(Se * np.sqrt(Nu) / np.sqrt(V)))
    return res


if __name__ == "__main__":
    out = {}
    g = DG.make_lin_hom(d=D, phi=0.7, psi=0.7)

    print("== (a) sample size ==")
    out["N"] = []
    for N in [25_000, 50_000, 100_000, 200_000, 400_000]:
        r = mc(g, N, 20); out["N"].append(r)
        print(f"  N={N:7d} eff={r['eff']:.3f} cov(2scale)={r['cov_ft']:.3f} "
              f"cov(BM)={r['cov_bm']:.3f} cov(plugin)={r['cov_plugin']:.3f}")

    print("== (b) block length b (N = 200k) ==")
    out["b"] = []
    for b in [5, 10, 20, 40, 80, 160, 320, 640]:
        r = mc(g, 200_000, b); out["b"].append(r)
        print(f"  b={b:4d} eff={r['eff']:.3f} cov(2scale)={r['cov_ft']:.3f} "
              f"cov(BM)={r['cov_bm']:.3f} width(2scale)={r['width_ft']:.3f} "
              f"adequacy={r['block_adequacy']:.3f} nonPSD={r['nonpsd']:.4f}")

    print("== (c) curvature warm-up c_w (N = 100k) ==")
    out["warm"] = []
    for wm in [0.5, 5, 25, 50, 100, 200, 500]:
        r = mc(g, 100_000, 20, warm_mult=wm)
        r["warm_mult"] = wm; out["warm"].append(r)
        print(f"  c_w={wm:6.1f} eff={r['eff']:.3f} cov(2scale)={r['cov_ft']:.3f}")

    print("== (c') warm-up on the regime-switching design: why c_w = 200, not 50 ==")
    out["warm_markov"] = []
    gm = DG.make_markov_cov(d=D, stay=0.97, psi=0.8, noise_scale=0.05)
    for wm in [50, 100, 200, 500]:
        r = mc(gm, 200_000, 20, R=120, warm_mult=wm)
        r["warm_mult"] = wm; out["warm_markov"].append(r)
        print(f"  markov c_w={wm:5.0f} eff={r['eff']:.4g} cov(2scale)={r['cov_ft']:.3f}")

    print("== (d) ridge constant and exponent (N = 200k) ==")
    out["ridge"] = []
    for cr, cp in [(0.0, 0.2), (0.001, 0.2), (0.01, 0.2), (0.1, 0.2), (1.0, 0.2),
                   (0.01, 0.05), (0.01, 0.1), (0.01, 0.24), (0.01, 0.4), (0.01, 0.49)]:
        r = mc(g, 200_000, 20, ci_reg=cr, ci_pow=cp)
        r["ci_reg"], r["ci_pow"] = cr, cp; out["ridge"].append(r)
        print(f"  c_iota={cr:<6g} varrho={cp:<5g} eff={r['eff']:.3f} "
              f"cov={r['cov_ft']:.3f}")

    print("== (e) initial curvature scale lam0 (N = 200k) ==")
    out["lam0"] = []
    for l0 in [1e-8, 1e-6, 1e-4, 1e-2, 1.0]:
        r = mc(g, 200_000, 20, lam0=l0)
        r["lam0"] = l0; out["lam0"].append(r)
        print(f"  lam0={l0:<8g} eff={r['eff']:.3f} cov={r['cov_ft']:.3f}")

    print("== (f) Bernoulli thinning p at matched cost, gap m = 1 (N = 200k) ==")
    out["p"] = []
    for p in [1.0, 0.5, 1.0 / 5, 1.0 / D]:
        r = mc(g, 200_000, 20, m=1, p=p)
        r["p"], r["m"] = p, 1; out["p"].append(r)
        print(f"  m=1 p={p:.4f} eff={r['eff']:.3f} cov={r['cov_ft']:.3f}")
    for m in [1, 5, 20]:
        r = mc(g, 200_000, 20, m=m, p=1.0)
        r["p"], r["m"] = 1.0, m; out["p"].append(r)
        print(f"  m={m} p=1      eff={r['eff']:.3f} cov={r['cov_ft']:.3f}")

    print("== (g') strong dependence: block length must respect the horizon ==")
    out["b_strong"] = []
    g2 = DG.make_lin_hom(d=D, phi=0.85, psi=0.85)
    for b in [20, 40, 80, 160, 320]:
        r = mc(g2, 200_000, b)
        r["phi"] = 0.85
        r["kappa"] = float(g2.oracle.meta["kappa_scalar"])
        out["b_strong"].append(r)
        print(f"  phi=psi=0.85 (kappa={r['kappa']:.2f}) b={b:4d} cov(2scale)={r['cov_ft']:.3f}"
              f" cov(BM)={r['cov_bm']:.3f} adequacy={r['block_adequacy']:.3f}"
              f" eff={r['eff']:.2f}")

    print("== (i) safeguarding the blocked step: which device, and is it needed? ==")
    out["shift"] = []
    gm2 = DG.make_markov_cov(d=D, stay=0.97, psi=0.8, noise_scale=0.05)
    VARIANTS = [("none", 0.0, 0.0), ("shift", 0.5, 0.0), ("cap", 0.0, 1.0),
                ("cap 0.25", 0.0, 0.25), ("cap 4", 0.0, 4.0), ("both", 0.5, 1.0)]
    for lab, gg, R_ in (("AR-hom", g, R), ("AR-strong", g2, R), ("Markov", gm2, 120)):
        for vtag, t0m, cap in VARIANTS:
            r = mc(gg, 200_000, 20, R=R_, t0_mult=t0m, step_cap=cap)
            r["t0_mult"], r["step_cap"] = t0m, cap
            r["variant"], r["design"] = vtag, lab
            out["shift"].append(r)
            print(f"  {lab:9s} {vtag:9s} t0={r['t0']:7.1f} clip={r['n_clip']:5.1f} "
                  f"eff={r['eff']:.4g} cov(2scale)={r['cov_ft']:.3f}", flush=True)

    print("== (j) warm-up as a fraction of a SHORT stream (N = 4000, d = 11) ==")
    out["warm_frac"] = []
    g11 = DG.make_lin_hom(d=11, phi=0.75, psi=0.75)
    for wf in [0.02, 0.05, 0.10, 0.25, 1.0]:
        r = mc(g11, 4_000, 40, R=R, warm_frac=wf)
        r["warm_frac"] = wf; out["warm_frac"].append(r)
        print(f"  warm_frac={wf:.2f} warm_blocks={r['n_warm']:4.0f} eff={r['eff']:.4g} "
              f"cov(2scale)={r['cov_ft']:.3f}", flush=True)

    print("== (g) averaged variant (N sweep) ==")
    out["averaged"] = []
    for N in [25_000, 100_000, 400_000]:
        r = mc(g, N, 20, a_step=0.67, w_avg=2.0, averaged=True)
        r["variant"] = "averaged"; out["averaged"].append(r)
        print(f"  N={N:7d} eff={r['eff']:.3f} cov(2scale)={r['cov_ft']:.3f}")

    common.save("exp5_ablations.json", out)
