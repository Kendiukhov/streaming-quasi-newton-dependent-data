"""Shared harness: method registry, Monte Carlo driver, tuning, and result I/O.

Design rules that make the comparison fair (and that we state in the paper):

* All methods consume the *same* stream in the *same* contiguous blocks of length ``b``,
  so every method pays O(d) per observation for the gradient and sees identical data.
* First-order baselines are handed the **true** Hessian for their sandwich, an advantage
  no practitioner has.  This isolates the question the paper is about -- which long-run
  covariance is being estimated -- and removes any suspicion that a baseline was
  crippled.
* Step-size constants for the first-order baselines are tuned by grid search on *pilot*
  seeds that are disjoint from the evaluation seeds, separately for every DGP and N, and
  the selected values are reported.
* The proposed method has no learning rate to tune: the step is 1/t and the scale comes
  from the maintained curvature estimate.
"""
from __future__ import annotations

import _env  # noqa: F401  (must precede numpy; see _env.py)

import json
import os
import sys
import time
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from bgsn import estimators as E  # noqa: E402
from bgsn import dgp as DGPS      # noqa: E402

Z95 = 1.959963984540054
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# Two-sided 95% critical value of the random-scaling pivot.  Simulated in
# exp0_critical_values.py; matches Kiefer-Vogelsang-Bunzel (2000) Table I / Table 2 of
# Lee, Liao, Seo & Shin (2022).  NB: several papers quote 5.32 as "95%"; that is the
# ONE-sided 95% quantile and produces badly under-covering two-sided intervals.
RS_CRIT95 = 6.811


# --------------------------------------------------------------------------------------
# method registry
# --------------------------------------------------------------------------------------
def m_bgsn(X, y, model, b, aux, variance="twoscale", **kw):
    return E.streaming_newton(X, y, model, b=b, m=b, p=1.0, variance=variance,
                              seed=aux["seed"], **kw)


def m_sn_iid(X, y, model, b, aux, variance="plugin", **kw):
    """Base method's curvature schedule: every observation a candidate, Bernoulli(1/d)."""
    d = X.shape[1]
    return E.streaming_newton(X, y, model, b=b, m=1, p=1.0 / d, variance=variance,
                              seed=aux["seed"], **kw)


def m_asgd(X, y, model, b, aux, variance="twoscale"):
    return E.first_order(X, y, model, b=b, c_step=aux["c_asgd"], a_step=0.75,
                         method="asgd", variance=variance,
                         H_for_sandwich=aux.get("H_true"), rs_crit=RS_CRIT95,
                         seed=aux["seed"])


def m_adagrad(X, y, model, b, aux, variance="twoscale"):
    return E.first_order(X, y, model, b=b, c_step=aux["c_adagrad"], a_step=0.0,
                         method="adagrad", variance=variance,
                         H_for_sandwich=aux.get("H_true"), rs_crit=RS_CRIT95,
                         seed=aux["seed"])


def m_offline_hac(X, y, model, b, aux):
    return E.offline_hac(X, y, model, lag=aux.get("hac_lag", 100))


def m_oracle(X, y, model, b, aux):
    f = m_bgsn(X, y, model, b, aux)
    return E.oracle_fit(f.theta, f.N, aux["H_true"], aux["S_true"])


METHODS = {
    # ---- proposed -------------------------------------------------------------------
    "BGSN":            lambda *a: m_bgsn(*a, variance="twoscale"),
    "BGSN-BM":         lambda *a: m_bgsn(*a, variance="batchmeans"),
    "BGSN-plugin":     lambda *a: m_bgsn(*a, variance="plugin"),
    # ---- base method (i.i.d. curvature schedule and i.i.d. variance) ----------------
    "SN-iid":          lambda *a: m_sn_iid(*a, variance="plugin"),
    "SN-iid+2scale":   lambda *a: m_sn_iid(*a, variance="twoscale"),
    # ---- first-order ----------------------------------------------------------------
    "ASGD-plugin":     lambda *a: m_asgd(*a, variance="plugin"),
    "ASGD-2scale":     lambda *a: m_asgd(*a, variance="twoscale"),
    "ASGD-RS":         lambda *a: m_asgd(*a, variance="randomscaling"),
    "AdaGrad-2scale":  lambda *a: m_adagrad(*a, variance="twoscale"),
    # ---- references -----------------------------------------------------------------
    "Offline-HAC":     m_offline_hac,
    "Oracle-var":      m_oracle,
}

# methods that are *not* streaming, flagged in every table
NOT_STREAMING = {"Offline-HAC", "Oracle-var"}


# --------------------------------------------------------------------------------------
# Monte Carlo driver
# --------------------------------------------------------------------------------------
@dataclass
class Rec:
    dgp: str
    method: str
    N: int
    b: int
    d: int
    seed: int
    err: list
    se: list
    crit: float
    secs: float
    psd_fallback: int = 0


def _aux(g, seed, tuned, extra=None):
    a = dict(seed=seed, H_true=g.oracle.H, S_true=g.oracle.S)
    a.update(tuned)
    if extra:
        a.update(extra)
    return a


def run_one_seed(g, N, b, methods, seed, tuned, extra=None):
    X, y = g.sample(N, 900_000 + seed)
    aux = _aux(g, seed, tuned, extra)
    out = []
    for name in methods:
        t0 = time.perf_counter()
        f = METHODS[name](X, y, g.model, b, aux)
        dt = time.perf_counter() - t0
        out.append(Rec(g.name, name, (N // b) * b, b, g.d, seed,
                       (f.theta - g.theta_star).tolist(), f.se.tolist(), f.crit, dt,
                       int(f.extras.get("n_psd_fallback", 0))))
    return out


NJOBS = int(os.environ.get("NJOBS", 5))



def logit_dgp(d: int = 20, phi: float = 0.7, psi: float = 0.8, cond_exponent: float = 1.0):
    """The dependent-logistic DGP, with its simulated long-run covariance oracle cached.

    The oracle for this design has no closed form and costs a long simulation, so it is cached
    on disk and shared by every experiment that uses the design.  Sharing matters for more than
    speed: two experiments that built the oracle independently would be comparing against
    slightly different targets.
    """
    from bgsn import streams as _st, dgp as _dg
    th = np.linspace(-1.0, 1.0, d)
    Sig = _st.illconditioned_cov(d, np.random.default_rng(30_000), cond_exponent)
    cache = os.path.join(RESULTS, "logit_oracle_d%d.npz" % d)
    if os.path.exists(cache):
        z = np.load(cache)
        orc = _st.Oracle(theta_star=th, H=z["H"], Gamma0=z["G0"], S=z["S"],
                         exact=False, oracle_error=float(z["err"]),
                         meta=dict(dgp="logit_copula", cached=True))
    else:
        import time as _t
        print("  [building the logistic oracle by long simulation ...]", flush=True)
        t0 = _t.time()
        orc = _st.logit_copula_oracle(d, phi, psi, cond_exponent, th, Sig,
                                      n_sim=1_000_000, n_rep=8, lag=100)
        np.savez(cache, H=orc.H, G0=orc.Gamma0, S=orc.S, err=orc.oracle_error)
        print(f"  [done in {_t.time()-t0:.0f}s, relative Monte Carlo error of the sandwich "
              f"diagonal = {orc.oracle_error:.2e}]", flush=True)
    return _dg.make_logit_copula(d=d, phi=phi, psi=psi, cond_exponent=cond_exponent,
                                 oracle=orc)


def adequacy(fit) -> float:
    """The free block-adequacy diagnostic, averaged over coordinates.

    ``r_j = |u_j'(S_ft - S_bm)u_j| / (u_j'S_bm u_j)`` with ``u_j = Hbar^{-1}e_j``.  The
    denominator is the plain batch-means quadratic form, NOT the two-scale one: ``S_bm`` is a
    sum of outer products and so is positive semidefinite, whereas ``S_ft`` is a difference of
    two such matrices and its quadratic forms can be negative (Remark 12).  Dividing by the
    two-scale form and flooring it at a tiny positive number, as an earlier version did, turns
    those coordinates into astronomically large numbers and makes the average meaningless on
    real streams.  With the batch-means denominator the diagnostic keeps its interpretation:
    an estimate of the relative block-length bias of plain batch means.
    """
    u = fit.extras["W"] * fit.extras["Ainv"]
    num = np.abs(np.diag(u @ (fit.extras["S_ft"] - fit.extras["S_bm"]) @ u))
    den = np.diag(u @ fit.extras["S_bm"] @ u)
    ok = den > 0
    if not ok.any():
        return float("nan")
    return float(np.mean(num[ok] / den[ok]))


def run_mc(g, N, b, methods, R, tuned, n_jobs=None, extra=None, batch_size=2):
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=n_jobs or NJOBS, batch_size=batch_size)(
        delayed(run_one_seed)(g, N, b, methods, s, tuned, extra) for s in range(R))
    return [asdict(r) for chunk in res for r in chunk]


# --------------------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------------------
def summarise(records, oracle: "DGPS.Oracle", by=("method",)):
    """Coverage, width and efficiency summaries with Monte Carlo standard errors."""
    import collections
    V = np.diag(oracle.sandwich_true)
    groups = collections.defaultdict(list)
    for r in records:
        groups[tuple(r[k] for k in by)].append(r)
    rows = []
    for key, rs in groups.items():
        err = np.array([r["err"] for r in rs])
        se = np.array([r["se"] for r in rs])
        crit = rs[0]["crit"]
        N = rs[0]["N"]
        R = len(rs)
        covmat = np.abs(err) <= crit * se
        cov = covmat.mean()
        # MC se of the pooled coverage, treating replicates as independent (coordinates
        # within a replicate are dependent, so we take the se across replicates)
        cov_se = covmat.mean(axis=1).std(ddof=1) / np.sqrt(R)
        w = 2 * crit * se
        w_oracle = 2 * Z95 * np.sqrt(V / N)
        eff = np.diag(err.T @ err / R * N) / V
        # Coverage checks a single quantile; the CLT claims the whole studentised
        # distribution.  We therefore also report the Kolmogorov-Smirnov distance between
        # err_j / se_j and the standard normal, per coordinate, across replicates.  This is
        # meaningless for the random-scaling method, whose pivot is deliberately
        # non-normal, so it is reported as NaN there.
        if crit == Z95:
            from scipy.stats import kstest
            ks = [kstest((err[:, j] / se[:, j]), "norm").statistic
                  for j in range(err.shape[1])]
            ks_mean, ks_max = float(np.mean(ks)), float(np.max(ks))
            ks_crit = 1.36 / np.sqrt(R)          # 95% KS critical value, one coordinate
        else:
            ks_mean = ks_max = ks_crit = float("nan")
        rows.append(dict(zip(by, key)) | dict(
            R=R, N=N,
            coverage=float(cov), coverage_se=float(cov_se),
            coverage_min=float(covmat.mean(axis=0).min()),
            coverage_max=float(covmat.mean(axis=0).max()),
            width_rel=float(np.mean(w / w_oracle)),
            se_ratio=float(np.mean(se * np.sqrt(N) / np.sqrt(V))),
            rmse_rel=float(np.sqrt(np.mean(eff))),
            ks_mean=ks_mean, ks_max=ks_max, ks_crit95=ks_crit,
            psd_fallback_rate=float(np.mean([r["psd_fallback"] for r in rs])
                                    / err.shape[1]),
            secs_median=float(np.median([r["secs"] for r in rs])),
        ))
    return rows


def save(name, obj):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    print(f"[saved] {path}")
    return path


def load(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------------------
# tuning of first-order step-size constants (pilot seeds, disjoint from evaluation)
# --------------------------------------------------------------------------------------
def tune_first_order(g, N, b, grid=None, R=8, pilot_offset=7_000_000, n_jobs=None):
    from joblib import Parallel, delayed
    if grid is None:
        grid = list(np.logspace(-3, 1.5, 19))

    def loss(c, method):
        errs = []
        for s in range(R):
            X, y = g.sample(N, pilot_offset + s)
            f = E.first_order(X, y, g.model, b=b, c_step=c,
                              a_step=0.75 if method == "asgd" else 0.0,
                              method=method, variance="randomscaling", seed=s)
            errs.append(np.sum((f.theta - g.theta_star) ** 2))
        v = float(np.mean(errs))
        return v if np.isfinite(v) else np.inf

    out = {}
    for method, key in (("asgd", "c_asgd"), ("adagrad", "c_adagrad")):
        vals = Parallel(n_jobs=n_jobs or NJOBS)(delayed(loss)(c, method) for c in grid)
        out[key] = float(grid[int(np.argmin(vals))])
        out[key + "_grid"] = [[float(c), float(v)] for c, v in zip(grid, vals)]
    return out
