"""User-facing estimators and confidence-interval constructions.

Every method returns a :class:`Fit` carrying a point estimate, per-coordinate standard
errors and the critical value to use, so that coverage/width comparisons are
apples-to-apples across very different inference devices (normal sandwich intervals vs
the non-standard random-scaling pivot).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import _core
from .models import get_model

MODEL_CODE = {"linear": 0, "logistic": 1}


# --------------------------------------------------------------------------------------
@dataclass
class Fit:
    name: str
    theta: np.ndarray
    N: int
    se: np.ndarray
    crit: float
    extras: dict = field(default_factory=dict)

    def ci(self):
        h = self.crit * self.se
        return self.theta - h, self.theta + h

    def covers(self, theta_star: np.ndarray) -> np.ndarray:
        lo, hi = self.ci()
        return (theta_star >= lo) & (theta_star <= hi)

    def width(self) -> np.ndarray:
        return 2.0 * self.crit * self.se


def _sandwich_diag(Ainv: np.ndarray, W: float, S: np.ndarray) -> np.ndarray:
    """Diagonal of ``Hbar^{-1} S Hbar^{-1}`` with ``Hbar^{-1} = W Ainv``.

    Only the diagonal is needed, and it is obtained with two O(d^2) products; the full
    d x d sandwich is never required for marginal intervals.
    """
    U = W * Ainv                       # Hbar^{-1} (symmetric)
    return np.diag(U @ S @ U)


def _sandwich_se(Ainv: np.ndarray, W: float, S: np.ndarray, N: int,
                 S_fallback: Optional[np.ndarray] = None):
    """Standard errors, with the documented fallback for a non-PSD two-scale estimate.

    The two-scale (flat-top / lugsail) long-run covariance is a difference of positive
    semidefinite matrices and need not itself be positive semidefinite.  Marginal intervals
    only need the scalars ``u_j' S u_j`` to be positive; where one is not, we fall back to
    the plain batch-means value, which is a positive semidefinite quadratic form by
    construction.  The number of coordinates that needed the fallback is returned so that
    it can be reported rather than hidden.
    """
    v = _sandwich_diag(Ainv, W, S)
    n_fb = 0
    if S_fallback is not None:
        bad = v <= 0.0
        n_fb = int(bad.sum())
        if n_fb:
            v = v.copy()
            v[bad] = _sandwich_diag(Ainv, W, S_fallback)[bad]
    return np.sqrt(np.maximum(v, 0.0) / N), n_fb


def _sandwich_from_H(H: np.ndarray, S: np.ndarray, N: int,
                     S_fallback: Optional[np.ndarray] = None):
    """As :func:`_sandwich_se` but with a supplied Hessian (used by the baselines)."""
    Hi = np.linalg.inv(H)
    v = np.diag(Hi @ S @ Hi)
    n_fb = 0
    if S_fallback is not None:
        bad = v <= 0.0
        n_fb = int(bad.sum())
        if n_fb:
            v = v.copy()
            v[bad] = np.diag(Hi @ S_fallback @ Hi)[bad]
    return np.sqrt(np.maximum(v, 0.0) / N), n_fb


# --------------------------------------------------------------------------------------
# The proposed method and the i.i.d. base method (same kernel, different knobs)
# --------------------------------------------------------------------------------------
def streaming_newton(X, y, model: str, b: int, m: int = None, p: float = 1.0,
                     theta0: Optional[np.ndarray] = None, w_hess: float = 0.0,
                     ci_reg: float = 0.01, ci_pow: float = 0.2, lam0: float = 1e-6,
                     warm_mult: float = 200.0, n_warm: Optional[int] = None,
                     warm_mult_max: float = 1000.0, warm_tol: float = 0.0,
                     auto_ridge: bool = True,
                     a_step: float = 1.0, c_step: float = 1.0, w_avg: float = 0.0,
                     t0_mult: float = 0.0, step_cap: float = 1.0,
                     warm_frac: float = 0.1,
                     averaged: bool = False,
                     seed: int = 0, variance: str = "twoscale",
                     z: float = 1.959963984540054, traj_every: int = 0,
                     name: Optional[str] = None):
    """Streaming inversion-free quasi-Newton estimator.

    ``variance`` selects the long-run covariance estimator:
    ``'twoscale'`` (default) the two-scale / flat-top blocked estimator,
    ``'batchmeans'`` plain non-overlapping batch means,
    ``'plugin'`` the i.i.d. score covariance ``Gamma0hat`` prescribed by the base
    method's asymptotic theory (inconsistent for the long-run covariance).

    ``m`` defaults to ``b`` (exactly one curvature slot per block: maximal gap at the
    prescribed cost).

    **Curvature warm-up.**  The efficient ``gamma_t = 1/t`` schedule must not start while
    the curvature estimate is still uninformative: a 1/t step taken with a bad
    preconditioner leaves a bias that decays only like ``t^{-c}`` with ``c < 1``, and at
    realistic N this dominates the sampling error and destroys interval coverage.  We
    therefore spend at least the first ``warm_mult * d`` observations on curvature updates
    alone (no parameter step), i.e. ``n_warm = ceil(warm_mult * d / b)`` blocks, and continue
    past that until the harmonic mean of the curvature eigenvalues has settled to within
    ``warm_tol`` (capped at ``warm_mult_max * d`` observations).  The self-terminating part
    matters on streams where consecutive observations are near-duplicates -- a persistent
    regime-switching design, say -- because there a fixed number of observations can span very
    few effectively distinct design points.  The whole warm-up consumes O(d) observations and
    ``O(warm_mult d^3)`` flops, O(1) in N, and leaves every asymptotic statement unchanged.

    **Stability shift.**  A blocked preconditioned step is not automatically safe to take at
    full length.  Writing the linearised error recursion,
    ``theta_t - theta* = (I - gamma_t Hbar^{-1} Hhat_t)(theta_{t-1} - theta*) + noise``, it
    contracts only while ``gamma_t ||Hbar^{-1} Hhat_t||_2 < 2``.  On an *independent* stream a
    block of ``b >= d`` observations gives a full-rank ``Hhat_t`` and this holds at
    ``gamma_1 = 1``; on a *dependent* stream the same block can span far fewer than ``b``
    distinct covariate directions, so ``Hhat_t`` is near-singular and the amplification is
    large.  We therefore use ``gamma_t = c_step (t + t0)^{-a_step}`` with
    ``t0 = max(0, t0_mult * c_step * max_t ||Hbar^{-1} Hhat_t||_2 - 1)``, the maximum taken over
    the warm-up blocks only.  ``t0_mult = 0.5`` is the value the contraction condition
    dictates.  ``t0`` is a finite constant fixed before the first parameter step, so it changes
    no asymptotic statement; set ``t0_mult = 0`` to recover the unshifted schedule.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    N, d = X.shape
    if m is None:
        m = b
    if theta0 is None:
        theta0 = np.zeros(d)
    if n_warm is None:
        # A warm-up of c_w d observations is O(1) in N, but on a short stream it can be a
        # large fraction of it, leaving too few blocked steps for the 1/t schedule to do
        # anything.  Cap it at warm_frac of the stream: this binds only when
        # c_w d > warm_frac N, i.e. exactly in the regime where the asymptotic argument for
        # ignoring the warm-up does not apply.
        n_warm = min(int(np.ceil(warm_mult * d / b)),
                     max(1, int(np.floor(warm_frac * N / b))))
    n_warm_max = max(n_warm, min(int(np.ceil(warm_mult_max * d / b)),
                                 max(1, int(np.floor(warm_frac * N / b)))))
    want_plugin = 1 if variance in ("plugin", "both") else 0
    (th, tbar, Ainv, W, S_bm, S_ft, G0, n_curv, nkeep, tn, tt,
     n_warm_used, t0_used, n_clip) = _core.bgsn_run(
        X, y, np.ascontiguousarray(theta0, dtype=np.float64), b, m, p, w_hess,
        ci_reg, ci_pow, lam0, MODEL_CODE[model], seed, want_plugin, traj_every,
        n_warm, 1 if auto_ridge else 0, a_step, c_step, w_avg,
        n_warm_max, warm_tol, t0_mult, step_cap)
    Nused = (N // b) * b
    point = tbar if averaged else th
    Suse = {"plugin": G0, "batchmeans": S_bm, "twoscale": S_ft}.get(variance, S_ft)
    se, n_fb = _sandwich_se(Ainv, W, Suse, Nused,
                            S_fallback=S_bm if Suse is S_ft else None)
    return Fit(name=name or f"SN(b={b},m={m},p={p},{variance})", theta=point, N=Nused,
               se=se, crit=z,
               extras=dict(Ainv=Ainv, W=W, S_bm=S_bm, S_ft=S_ft, Gamma0_hat=G0,
                           theta_last=th, theta_bar=tbar, n_warm=n_warm_used,
                           n_psd_fallback=n_fb, n_curv=n_curv, n_blocks_kept=nkeep,
                           t0=t0_used, n_clip=n_clip, traj_n=tn, traj_theta=tt))


def first_order(X, y, model: str, b: int, c_step: float, a_step: float = 0.75,
                method: str = "asgd", theta0: Optional[np.ndarray] = None,
                adagrad_eps: float = 1e-8, seed: int = 0,
                variance: str = "batchmeans", H_for_sandwich: Optional[np.ndarray] = None,
                z: float = 1.959963984540054, rs_crit: float = 6.747,
                traj_every: int = 0, name: Optional[str] = None):
    """Averaged SGD or streaming AdaGrad, with three inference options.

    ``variance='batchmeans'`` / ``'plugin'`` build the usual sandwich and therefore need
    a Hessian estimate; we supply the *offline* empirical Hessian at the final iterate
    (``H_for_sandwich``) so that the first-order baselines are not penalised for lacking
    curvature information -- this is a deliberately generous choice.
    ``variance='randomscaling'`` uses the studentisation of Lee, Liao, Seo and Shin,
    which needs no long-run covariance estimate at all.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    N, d = X.shape
    if theta0 is None:
        theta0 = np.zeros(d)
    want_plugin = 1 if variance in ("plugin", "both") else 0
    mcode = 0 if method == "asgd" else 1
    tbar, S_bm, S_ft, G0, V_rs, tn, tt = _core.first_order_run(
        X, y, np.ascontiguousarray(theta0, dtype=np.float64), b, c_step, a_step,
        mcode, adagrad_eps, MODEL_CODE[model], want_plugin, traj_every)
    Nused = (N // b) * b
    T = N // b
    n_fb = 0
    if variance == "randomscaling":
        se = np.sqrt(np.maximum(np.diag(V_rs), 0.0) / T)
        crit = rs_crit
    else:
        if H_for_sandwich is None:
            raise ValueError("sandwich inference for first-order methods needs a Hessian")
        Suse = {"plugin": G0, "batchmeans": S_bm, "twoscale": S_ft}.get(variance, S_ft)
        se, n_fb = _sandwich_from_H(H_for_sandwich, Suse, Nused,
                                    S_fallback=S_bm if Suse is S_ft else None)
        crit = z
    return Fit(name=name or f"{method}({variance})", theta=tbar, N=Nused, se=se,
               crit=crit,
               extras=dict(S_bm=S_bm, S_ft=S_ft, Gamma0_hat=G0, V_rs=V_rs,
                           n_psd_fallback=n_fb, traj_n=tn, traj_theta=tt))


def empirical_hessian(X, y, model: str, theta: np.ndarray) -> np.ndarray:
    """``N^{-1} sum_i alpha_i x_i x_i'`` at a given theta (O(Nd^2), offline)."""
    mdl = get_model(model)
    a = mdl.alpha(theta, X, y)
    return (X * a[:, None]).T @ X / X.shape[0]


# --------------------------------------------------------------------------------------
# Offline reference: full-sample M-estimator + HAC covariance
# --------------------------------------------------------------------------------------
def offline_mestimator(X, y, model: str, n_newton: int = 25, tol: float = 1e-12):
    """Full-sample M-estimator (closed form for least squares, Newton for logistic)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if model == "linear":
        return np.linalg.solve(X.T @ X, X.T @ y)
    mdl = get_model(model)
    th = np.zeros(X.shape[1])
    for _ in range(n_newton):
        g = mdl.mean_grad(th, X, y)
        H = empirical_hessian(X, y, model, th)
        stp = np.linalg.solve(H + 1e-12 * np.eye(X.shape[1]), g)
        th = th - stp
        if np.max(np.abs(stp)) < tol:
            break
    return th


def andrews_bandwidth(G: np.ndarray, rho_cap: float = 0.9,
                      abs_cap: int = 1000) -> int:
    """Andrews (1991) automatic bandwidth for the Bartlett kernel via AR(1) plug-ins.

    The AR(1) plug-in contains ``(1 - rho)^{-8}``, which explodes for near-unit-root
    scores and can return a bandwidth of order n.  We cap the fitted ``rho`` at
    ``rho_cap`` and the bandwidth at ``abs_cap``, and always report the value used; the
    real-data section shows that conclusions do not move over a wide bandwidth range.
    """
    n, d = G.shape
    num = 0.0
    den = 0.0
    for j in range(d):
        g = G[:, j]
        v0 = float(g[:-1] @ g[:-1])
        if v0 <= 0:
            continue
        rho = float(g[:-1] @ g[1:]) / v0
        rho = min(max(rho, -rho_cap), rho_cap)
        resid = g[1:] - rho * g[:-1]
        s2 = float(resid @ resid) / (n - 1)
        num += 4.0 * rho ** 2 * s2 ** 2 / (1 - rho) ** 8
        den += s2 ** 2 / (1 - rho) ** 4
    if den <= 0:
        return 1
    a1 = num / den
    L = 1.1447 * (a1 * n) ** (1.0 / 3.0)
    return int(max(1, min(np.floor(L), n // 5, abs_cap)))


def offline_hac(X, y, model: str, lag: Optional[int] = None,
                z: float = 1.959963984540054, name: str = "offline-HAC"):
    """Offline M-estimator with a Newey--West sandwich: the gold-standard reference.

    Costs O(N d^2 + L d^2) time and O(N d) memory, so it is *not* a streaming method;
    we include it to show how close the streaming intervals get to the batch ideal.
    """
    from .streams import bartlett_lrv
    mdl = get_model(model)
    th = offline_mestimator(X, y, model)
    G = mdl.score(th, X, y)
    G = G - G.mean(axis=0, keepdims=True)
    L = andrews_bandwidth(G) if lag is None else lag
    S = bartlett_lrv(G, L)
    H = empirical_hessian(X, y, model, th)
    se, _ = _sandwich_from_H(H, S, X.shape[0])
    return Fit(name=name, theta=th, N=X.shape[0], se=se, crit=z,
               extras=dict(S_hat=S, H=H, lag=L))


def oracle_fit(theta: np.ndarray, N: int, H: np.ndarray, S: np.ndarray,
               z: float = 1.959963984540054, name: str = "oracle-variance") -> Fit:
    """Intervals built from the *true* H and S: a validity check on the CLT itself."""
    return Fit(name=name, theta=theta, N=N, se=_sandwich_from_H(H, S, N)[0], crit=z)


def iterate_obm_sandwich(theta_path: np.ndarray, N: int, b_n: int) -> np.ndarray:
    r"""Overlapping-batch-means covariance built from the ITERATE path.

    This is the comparator used by the dependent-data literature: it estimates the whole
    sandwich ``Sigma = H^{-1} S H^{-1}`` directly from the trajectory, with no gradients and
    no curvature estimate.  In the form of Samsonov et al. (their Eq. 14, matrix version),

        Sigmahat = b_n / (T - b_n + 1) * sum_t (thetabar_{b_n,t} - thetabar)(...)^T,

    where ``thetabar_{b_n,t}`` averages the iterates in the window ``[t, t+b_n)`` and
    ``thetabar`` averages all of them.  The scaling is such that it estimates the covariance
    of ``sqrt(N)(thetabar - theta*)``.

    Its block length must grow with the sample size, because a window has to span enough
    parameter movement for its average to behave like an independent draw -- which is exactly
    the constraint that block *gradients* escape.  Computed here with cumulative sums, so
    O(T d) time; it is included as the honest head-to-head for the rate comparison, not as a
    streaming method.
    """
    Th = np.asarray(theta_path, float)
    T = Th.shape[0]
    b_n = int(max(2, min(b_n, T // 2)))
    C = np.vstack([np.zeros((1, Th.shape[1])), np.cumsum(Th, axis=0)])
    win = (C[b_n:] - C[:-b_n]) / b_n                    # (T-b_n+1, d) window averages
    dev = win - Th.mean(axis=0, keepdims=True)
    Sig = (b_n / dev.shape[0]) * (dev.T @ dev)
    # the iterates are indexed by block, so convert the per-block scaling to per-observation
    return Sig * (N / T)


# --------------------------------------------------------------------------------------
# critical values for the random-scaling pivot
# --------------------------------------------------------------------------------------
def random_scaling_critical_values(levels=(0.90, 0.95, 0.975, 0.99),
                                   n_paths: int = 4_000_000, n_terms: int = 400,
                                   seed: int = 12345, chunk: int = 200_000):
    r"""Quantiles of the random-scaling pivot ``|W(1)| / sqrt(int_0^1 B(r)^2 dr)``.

    ``B(r) = W(r) - r W(1)`` is a Brownian bridge, which is *independent* of ``W(1)`` and
    has the Karhunen--Loeve expansion ``B(r) = sum_k Z_k sqrt(2) sin(k pi r)/(k pi)`` with
    i.i.d. standard normal ``Z_k``.  Since ``sqrt(2) sin(k pi r)`` is orthonormal on
    ``[0, 1]``,

        int_0^1 B(r)^2 dr = sum_{k >= 1} Z_k^2 / (k pi)^2 ,

    so the pivot can be sampled exactly with no time discretisation.  We keep ``n_terms``
    terms and replace the remainder by its mean ``sum_{k > n_terms} (k pi)^{-2}``, whose
    standard deviation is ``O(n_terms^{-3/2})`` and therefore negligible.  This makes the
    quantiles cheap enough to compute at four million draws.
    """
    rng = np.random.default_rng(seed)
    k = np.arange(1, n_terms + 1)
    w = 1.0 / (k * np.pi) ** 2
    tail = float(np.pi ** -2 * (np.pi ** 2 / 6.0 - np.sum(1.0 / k ** 2)))
    out = []
    done = 0
    while done < n_paths:
        c = min(chunk, n_paths - done)
        Z = rng.standard_normal((c, n_terms))
        denom = np.sqrt(Z ** 2 @ w + tail)
        out.append(np.abs(rng.standard_normal(c)) / denom)
        done += c
    stat = np.concatenate(out)
    return {lv: float(np.quantile(stat, lv)) for lv in levels}
