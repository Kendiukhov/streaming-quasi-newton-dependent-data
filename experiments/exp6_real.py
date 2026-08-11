"""Real hourly streams: road traffic (I-94) and urban air quality (Beijing PM2.5).

Two protocols, because neither alone is conclusive.

PROTOCOL A -- real covariates, synthetic response, EXACT ground truth.
  The covariate stream is real (and strongly dependent); the response is
  y_t = x_t' theta* + u_t with u an AR(1) whose coefficient is set to the AR(1)
  coefficient of the *real* residuals, and theta* / sigma_u are the full-stream
  least-squares fit and residual scale.  Conditionally on the covariate path the
  population targets H, Gamma_0 and S are available in closed form
  (realdata.conditional_oracle), so coverage is measured against an exact truth with
  genuinely real covariate dependence.  Replicates vary the error path within each of
  several disjoint covariate segments.

PROTOCOL B -- fully real response.
  The target is the full-stream M-estimator theta_hat_full; each method is run on disjoint
  contiguous segments and we record coverage of theta_hat_full.  Because the target is
  itself estimated from a superset of each segment,
  Var(theta_seg - theta_full) = Var(theta_seg)(1 - 1/R) and the *correct* nominal level for
  this comparison is 2 Phi(z / sqrt(1 - 1/R)) - 1, which we report alongside 0.95.

Both protocols also report the free block-adequacy diagnostic
  r_j = |u_j'(S_ft - S_bm)u_j| / (u_j' S_ft u_j),
an estimate of the relative block-length bias of plain batch means: small r_j certifies
that b exceeds the dependence horizon.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: F401
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
import common
from bgsn import realdata as RD, estimators as E

R_NOISE_METRO = int(os.environ.get("RN_METRO", 40))
R_NOISE_AIRQ = int(os.environ.get("RN_AIRQ", 6))
METHODS_A = ["BGSN", "BGSN-BM", "BGSN-plugin", "SN-iid", "ASGD-plugin", "ASGD-2scale",
             "ASGD-RS", "Oracle-var"]
METHODS_B = ["BGSN", "BGSN-BM", "BGSN-plugin", "SN-iid", "ASGD-plugin", "ASGD-2scale",
             "ASGD-RS"]


def ar1_coef(r):
    return float(np.dot(r[:-1], r[1:]) / np.dot(r[:-1], r[:-1]))


HOLDOUT_FRAC = 0.15          # leading fraction of each stream reserved for tuning only


def holdout_end(n_total, n_seg):
    """Index at which the evaluation segments begin.

    The first HOLDOUT_FRAC of each stream is used *only* to choose the first-order
    baselines' step-size constants, and is excluded from every evaluation segment.  Tuning
    on data that is later evaluated on would flatter those baselines.
    """
    return int(np.ceil(HOLDOUT_FRAC * n_total / n_seg)) * n_seg


def segments(n_total, n_seg):
    start = holdout_end(n_total, n_seg)
    R = (n_total - start) // n_seg
    return [(start + k * n_seg, start + (k + 1) * n_seg) for k in range(R)]


# --------------------------------------------------------------------------------------
def protocol_a(streams, n_seg, b, psi_map, theta_map, sigma_map, n_noise, tuned):
    """`tuned` maps stream name -> {c_asgd, c_adagrad}."""
    recs, adeq = [], []
    jobs = []
    for name, X in streams:
        for (i0, i1) in segments(len(X), n_seg):
            for rep in range(n_noise):
                jobs.append((name, i0, i1, rep))

    def one(job):
        name, i0, i1, rep = job
        X = dict(streams)[name][i0:i1]
        th, psi, sg = theta_map[name], psi_map[name], sigma_map[name]
        u = RD.ar1_errors(len(X), psi, sg, seed=abs(hash((name, i0, rep))) % (2 ** 31))
        y = X @ th + u
        orc = RD.conditional_oracle(X, psi, sg, th)
        aux = dict(seed=rep, H_true=orc.H, S_true=orc.S, **tuned[name])
        out = []
        for mth in METHODS_A:
            f = common.METHODS[mth](np.ascontiguousarray(X), y, "linear", b, aux)
            out.append((name, mth, (f.theta - th), f.se, f.crit,
                        np.diag(orc.sandwich_true), orc.kappa, (name, i0)))
        f = common.METHODS["BGSN"](np.ascontiguousarray(X), y, "linear", b, aux)
        u_ = f.extras["W"] * f.extras["Ainv"]
        num = np.abs(np.diag(u_ @ (f.extras["S_ft"] - f.extras["S_bm"]) @ u_))
        den = np.maximum(np.diag(u_ @ f.extras["S_ft"] @ u_), 1e-30)
        return out, float(np.mean(num / den))

    res = Parallel(n_jobs=common.NJOBS, batch_size=2)(delayed(one)(j) for j in jobs)
    for out, a in res:
        recs += out
        adeq.append(a)
    return recs, float(np.mean(adeq))


def cluster_se(values, clusters):
    """Cluster-robust Monte Carlo standard error of a mean.

    Protocol A replicates that share a covariate segment are not independent: they differ
    only in the error path.  Treating them as independent would understate the Monte Carlo
    error, so we average within segment first and take the standard error across segments.
    """
    import collections
    g = collections.defaultdict(list)
    for v, c in zip(values, clusters):
        g[c].append(v)
    m = np.array([np.mean(v) for v in g.values()])
    if len(m) < 2:
        return float("nan")
    return float(m.std(ddof=1) / np.sqrt(len(m)))


def summarise_a(recs, n_seg):
    import collections
    grp = collections.defaultdict(list)
    for name, mth, err, se, crit, V, kap, clu in recs:
        grp[(name, mth)].append((err, se, crit, V, kap, clu))
    rows = []
    for (name, mth), rs in grp.items():
        err = np.array([r[0] for r in rs]); se = np.array([r[1] for r in rs])
        crit = rs[0][2]; V = np.array([r[3] for r in rs]); kap = np.array([r[4] for r in rs])
        clu = [r[5] for r in rs]
        covm = np.abs(err) <= crit * se
        wo = 2 * common.Z95 * np.sqrt(V / n_seg)
        # pooled prediction for the dependence-blind interval on this stream
        pred = float(np.mean(2 * norm.cdf(common.Z95 / np.sqrt(kap)) - 1))
        rows.append(dict(stream=name, method=mth, R=len(rs), n=n_seg,
                         n_segments=len(set(clu)),
                         coverage=float(covm.mean()),
                         coverage_se=float(cluster_se(covm.mean(axis=1), clu)),
                         coverage_min=float(covm.mean(axis=0).min()),
                         coverage_max=float(covm.mean(axis=0).max()),
                         width_rel=float(np.mean(2 * crit * se / wo)),
                         rmse_rel=float(np.sqrt(np.mean(err ** 2 * n_seg / V))),
                         kappa_median=float(np.median(kap)),
                         predicted_plugin_coverage=pred))
    return rows


# --------------------------------------------------------------------------------------
def protocol_b(streams, ys, n_seg, b, tuned):
    rows, adeq = [], []
    for name, X in streams:
        y = ys[name]
        th_full = E.offline_mestimator(X, y, "linear")
        segs = segments(len(X), n_seg)
        Rn = len(segs)
        fit_full = E.offline_hac(X, y, "linear", lag=min(500, len(X) // 20))
        recs = []
        for (i0, i1) in segs:
            Xs = np.ascontiguousarray(X[i0:i1]); ysg = np.ascontiguousarray(y[i0:i1])
            aux = dict(seed=i0 % 1000, **tuned[name])
            aux["H_true"] = E.empirical_hessian(Xs, ysg, "linear", th_full)
            for mth in METHODS_B:
                f = common.METHODS[mth](Xs, ysg, "linear", b, aux)
                recs.append((mth, f.theta - th_full, f.se, f.crit))
            f = common.METHODS["BGSN"](Xs, ysg, "linear", b, aux)
            u_ = f.extras["W"] * f.extras["Ainv"]
            num = np.abs(np.diag(u_ @ (f.extras["S_ft"] - f.extras["S_bm"]) @ u_))
            den = np.maximum(np.diag(u_ @ f.extras["S_ft"] @ u_), 1e-30)
            adeq.append(float(np.mean(num / den)))
        import collections
        grp = collections.defaultdict(list)
        for mth, e, se, c in recs:
            grp[mth].append((e, se, c))
        for mth, rs in grp.items():
            err = np.array([r[0] for r in rs]); se = np.array([r[1] for r in rs])
            crit = rs[0][2]
            covm = np.abs(err) <= crit * se
            rows.append(dict(stream=name, method=mth, R=Rn, n=n_seg,
                             coverage=float(covm.mean()),
                             coverage_se=float(covm.mean(axis=1).std(ddof=1)
                                               / np.sqrt(len(rs))),
                             mean_width=float(np.mean(2 * crit * se)),
                             rmse=float(np.sqrt(np.mean(err ** 2)))))
        rows.append(dict(stream=name, method="__target__", R=Rn, n=n_seg,
                         nominal_corrected=float(2 * norm.cdf(
                             common.Z95 / np.sqrt(1 - 1.0 / Rn)) - 1),
                         hac_lag_full=int(fit_full.extras["lag"]),
                         se_full=fit_full.se.tolist(),
                         theta_full=th_full.tolist()))
    return rows, float(np.mean(adeq))


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    metro = RD.load_metro()
    sites = RD.AIRQ_SITES
    airq = [RD.load_airq(s) for s in sites]

    out = dict(datasets=[])
    prep = {}
    for st in [metro] + airq:
        th = E.offline_mestimator(st.X, st.y, "linear")
        r = st.y - st.X @ th
        prep[st.name] = dict(theta=th, sigma=float(r.std()), psi=ar1_coef(r),
                             n=len(st.X), d=st.X.shape[1])
        n_seg_i = 4000 if st.name.startswith("metro") else 5000
        conds, ranks = [], []
        for (i0, i1) in segments(len(st.X), n_seg_i):
            Hs = st.X[i0:i1].T @ st.X[i0:i1] / (i1 - i0)
            conds.append(float(np.linalg.cond(Hs)))
            ranks.append(int(np.linalg.matrix_rank(Hs)))
        prep[st.name]["seg_cond_max"] = max(conds) if conds else float("nan")
        prep[st.name]["seg_rank_min"] = min(ranks) if ranks else -1
        assert prep[st.name]["seg_rank_min"] == st.X.shape[1], (
            f"{st.name}: a segment design is rank deficient "
            f"({prep[st.name]['seg_rank_min']}/{st.X.shape[1]}); the per-segment population "
            f"targets would be undefined")
        out["datasets"].append(dict(name=st.name, n=len(st.X), d=st.X.shape[1],
                                    features=st.feature_names, note=st.note,
                                    resid_ar1=prep[st.name]["psi"],
                                    resid_sd=prep[st.name]["sigma"],
                                    R2=float(1 - r.var() / st.y.var()),
                                    cond_H=float(np.linalg.cond(st.X.T @ st.X / len(st.X))),
                                    seg_cond_max=prep[st.name]["seg_cond_max"],
                                    seg_rank_min=prep[st.name]["seg_rank_min"]))
        print(f"{st.name}: n={len(st.X)} d={st.X.shape[1]} resid AR(1)={prep[st.name]['psi']:.3f} "
              f"R2={1-r.var()/st.y.var():.3f} cond(H)={np.linalg.cond(st.X.T@st.X/len(st.X)):.1f}")

    # tune the first-order step constants PER STREAM on that stream's held-out prefix,
    # which is excluded from every evaluation segment (see `segments`)
    from bgsn import estimators as EE
    grid = np.logspace(-3, 1.5, 19)

    def tune(st, n_seg, b):
        end = holdout_end(len(st.X), n_seg)
        Xp = np.ascontiguousarray(st.X[:end]); yp = np.ascontiguousarray(st.y[:end])
        th = prep[st.name]["theta"]
        chosen, grids = {}, {}
        for mth, key in (("asgd", "c_asgd"), ("adagrad", "c_adagrad")):
            vals = []
            for c in grid:
                f = EE.first_order(Xp, yp, "linear", b=b, c_step=c,
                                   a_step=0.75 if mth == "asgd" else 0.0, method=mth,
                                   variance="randomscaling", seed=0)
                v = float(np.sum((f.theta - th) ** 2))
                vals.append(v if np.isfinite(v) else np.inf)
            j = int(np.argmin(vals))
            chosen[key] = float(grid[j])
            grids[key + "_grid"] = [[float(c), float(v)] for c, v in zip(grid, vals)]
            grids[key + "_interior"] = bool(0 < j < len(grid) - 1)
        return chosen, grids

    tuned_per_stream, tuning_grids = {}, {}
    for st, n_seg, b in [(metro, 4000, 40)] + [(a, 5000, 50) for a in airq]:
        ch, gr = tune(st, n_seg, b)
        tuned_per_stream[st.name] = ch
        tuning_grids[st.name] = gr
        print(f"  tuned on {st.name} hold-out: {ch} "
              f"(interior: {gr['c_asgd_interior']}, {gr['c_adagrad_interior']})", flush=True)
    out["tuned_per_stream"] = tuned_per_stream
    out["tuning_grids"] = tuning_grids
    tuned = tuned_per_stream[metro.name]

    # ---- Protocol A ------------------------------------------------------------------
    print("\n== Protocol A: real covariates, synthetic AR errors, exact oracle ==")
    for label, sts, n_seg, b, n_noise in [
            ("metro", [metro], 4000, 40, R_NOISE_METRO),
            ("airq", airq, 5000, 50, R_NOISE_AIRQ)]:
        streams = [(s.name, np.ascontiguousarray(s.X)) for s in sts]
        recs, adeq = protocol_a(
            streams, n_seg, b,
            {s.name: prep[s.name]["psi"] for s in sts},
            {s.name: prep[s.name]["theta"] for s in sts},
            {s.name: prep[s.name]["sigma"] for s in sts},
            n_noise, tuned_per_stream)
        rows = summarise_a(recs, n_seg)
        out[f"protoA_{label}"] = dict(rows=rows, n=n_seg, b=b, n_noise=n_noise,
                                      block_adequacy=adeq)
        agg = {}
        for r in rows:
            agg.setdefault(r["method"], []).append(r)
        print(f" -- {label} (n={n_seg}, b={b}, block-adequacy={adeq:.3f})")
        for mth, rs in sorted(agg.items(), key=lambda kv: -np.mean([r['coverage'] for r in kv[1]])):
            c = np.mean([r["coverage"] for r in rs]); w = np.mean([r["width_rel"] for r in rs])
            e = np.mean([r["rmse_rel"] for r in rs]); k = np.median([r["kappa_median"] for r in rs])
            print(f"    {mth:14s} cov={c:.3f} width={w:.2f} rmse={e:.2f} (kappa~{k:.2f})")

    # ---- Protocol B ------------------------------------------------------------------
    print("\n== Protocol B: fully real response, target = full-stream M-estimator ==")
    for label, sts, n_seg, b in [("metro", [metro], 4000, 40), ("airq", airq, 5000, 50)]:
        streams = [(s.name, np.ascontiguousarray(s.X)) for s in sts]
        ys = {s.name: np.ascontiguousarray(s.y) for s in sts}
        rows, adeq = protocol_b(streams, ys, n_seg, b, tuned_per_stream)
        out[f"protoB_{label}"] = dict(rows=rows, n=n_seg, b=b, block_adequacy=adeq)
        agg = {}
        for r in rows:
            if r["method"] == "__target__":
                nominal = r["nominal_corrected"]
                continue
            agg.setdefault(r["method"], []).append(r)
        print(f" -- {label} (n={n_seg}, b={b}, adequacy={adeq:.3f}, "
              f"target-corrected nominal={nominal:.4f})")
        for mth, rs in sorted(agg.items(), key=lambda kv: -np.mean([r['coverage'] for r in kv[1]])):
            c = np.mean([r["coverage"] for r in rs])
            print(f"    {mth:14s} cov={c:.3f}  (R={rs[0]['R']} segments x {len(rs)} streams)")

    # ---- block length sweep on real data ---------------------------------------------
    print("\n== Protocol A, block length sweep (metro) ==")
    out["protoA_metro_bsweep"] = []
    for b in [10, 20, 40, 80, 160, 400]:
        recs, adeq = protocol_a([(metro.name, np.ascontiguousarray(metro.X))], 4000, b,
                                {metro.name: prep[metro.name]["psi"]},
                                {metro.name: prep[metro.name]["theta"]},
                                {metro.name: prep[metro.name]["sigma"]},
                                max(R_NOISE_METRO // 2, 10), tuned_per_stream)
        rows = summarise_a(recs, 4000)
        rec = {r["method"]: r["coverage"] for r in rows}
        w = {r["method"]: r["width_rel"] for r in rows}
        out["protoA_metro_bsweep"].append(dict(b=b, adequacy=adeq, coverage=rec, width=w))
        print(f"  b={b:4d} adequacy={adeq:.3f} BGSN={rec['BGSN']:.3f} "
              f"BM={rec['BGSN-BM']:.3f} plugin={rec['BGSN-plugin']:.3f} "
              f"oracle={rec['Oracle-var']:.3f} width(BGSN)={w['BGSN']:.2f}")

    common.save("exp6_real.json", out)
