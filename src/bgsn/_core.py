"""Numba-compiled streaming kernels.

All estimators consume the stream in *contiguous blocks* of length ``b`` and never
revisit an observation.  Per-block cost is O(bd) for the gradient plus O(d^2) for the
curvature/covariance algebra, i.e. O(d) amortised work per observation whenever
``b >= d``.  Memory is O(d^2) (the running inverse-curvature matrix and the long-run
covariance accumulator) -- the same order as the base i.i.d. method.

Model codes: 0 = least squares, 1 = logistic regression (labels in {0, 1}).

Naming follows the paper: ``A`` is the *unnormalised* curvature accumulator, ``W`` the
total weight, so the curvature estimate is ``Hbar = A / W`` and the preconditioner is
``Hbar^{-1} = W * Ainv``.  Maintaining ``Ainv`` (rather than ``Hbar^{-1}``) is what
keeps every update an exact rank-one Sherman--Morrison step.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


@njit(cache=True, inline="always")
def _sigmoid(z):
    if z >= 0.0:
        return 1.0 / (1.0 + np.exp(-z))
    e = np.exp(z)
    return e / (1.0 + e)


@njit(cache=True, inline="always")
def _score_scale(model_code, xdot, yi):
    """c such that the score of observation i equals c * x_i."""
    if model_code == 0:
        return xdot - yi
    return _sigmoid(xdot) - yi


@njit(cache=True, inline="always")
def _alpha(model_code, xdot):
    """Curvature scalar: grad^2 F = E[alpha(x'theta) x x']."""
    if model_code == 0:
        return 1.0
    p = _sigmoid(xdot)
    return p * (1.0 - p)


@njit(cache=True)
def _riccati_vec(Ainv, u, c, work):
    """In-place ``A <- A + c u u'`` on the inverse: Ainv <- Ainv - c v v' / (1 + c u'v)."""
    d = Ainv.shape[0]
    for i in range(d):
        s = 0.0
        for j in range(d):
            s += Ainv[i, j] * u[j]
        work[i] = s
    uv = 0.0
    for i in range(d):
        uv += u[i] * work[i]
    fac = c / (1.0 + c * uv)
    for i in range(d):
        wi = fac * work[i]
        for j in range(d):
            Ainv[i, j] -= wi * work[j]


@njit(cache=True)
def _riccati_basis(Ainv, k, c):
    """``A <- A + c e_k e_k'``: the needed vector is simply column k of Ainv."""
    d = Ainv.shape[0]
    fac = c / (1.0 + c * Ainv[k, k])
    col = Ainv[:, k].copy()
    for i in range(d):
        wi = fac * col[i]
        for j in range(d):
            Ainv[i, j] -= wi * col[j]


@njit(cache=True, inline="always")
def _roll_dyadic(acc_prev, acc_cur, d):
    """prev <- cur, cur <- 0 (the 'keep the most recent dyadic window' rule)."""
    for i in range(d):
        for j in range(d):
            acc_prev[i, j] = acc_cur[i, j]
            acc_cur[i, j] = 0.0


@njit(cache=True)
def _finish_lrv(acc_prev, acc_cur, nkeep, crs_prev, crs_cur, npair, b, d):
    """Assemble the two long-run covariance estimators from the block accumulators.

    ``S_bm``  : non-overlapping batch means,  E[S_bm] = S - Delta / b + O(rho^b) with
                Delta = sum_{k>=1} k (Gamma_k + Gamma_k') -- an O(1/b) bias that shrinks only
                linearly in the block length.
    ``S_ft``  : the two-scale ("flat-top") correction ``S_bm + b (Lam1 + Lam1')`` with
                ``Lam1`` the lag-one *block* cross-moment.  Algebraically this is the
                Richardson extrapolation ``2 S_bm(2b) - S_bm(b)``, and it applies the
                kernel weights ``1`` for ``|k| <= b`` and ``(2b-|k|)/b`` for
                ``b < |k| < 2b``, so under geometric mixing its bias is O(rho^b).
    Both cost O(d^2) per block and O(d^2) memory.
    """
    S_bm = np.zeros((d, d))
    S_ft = np.zeros((d, d))
    for a in range(d):
        for bb in range(d):
            S_bm[a, bb] = b * (acc_prev[a, bb] + acc_cur[a, bb]) / nkeep
    if npair > 0:
        # S_ft = S_bm + b (Lam1 + Lam1'); the raw accumulator holds Lam1 alone, which is
        # not symmetric, so both orientations must be added (not averaged).
        for a in range(d):
            for bb in range(d):
                c = (crs_prev[a, bb] + crs_cur[a, bb]
                     + crs_prev[bb, a] + crs_cur[bb, a]) / npair
                S_ft[a, bb] = S_bm[a, bb] + b * c
    else:
        for a in range(d):
            for bb in range(d):
                S_ft[a, bb] = S_bm[a, bb]
    return S_bm, S_ft


@njit(cache=True)
def _block_amplification(X, y, theta, Ainv, W, b, nblk, model_code, n_iter, v0):
    """``max_t`` of ``||Hbar^{-1} Hhat_t||_2`` over the first ``nblk`` blocks.

    ``Hhat_t`` is the block Hessian ``(1/b) sum_{i in block} alpha_i psi_i psi_i^T`` and
    ``Hbar^{-1} = W * Ainv``.  This is the quantity that decides whether a *blocked*
    preconditioned step can be taken at full length: the error recursion contracts only
    while ``gamma_t ||Hbar^{-1} Hhat_t||_2 < 2``, and on a dependent stream a block of
    ``b`` observations can span far fewer than ``b`` distinct covariate directions, so
    ``Hhat_t`` is close to singular and the amplification is large (Section 4.4).

    The spectral norm is obtained by power iteration on ``M^T M`` with ``M`` applied
    factorised, so no ``d x d`` product is ever formed: each iteration costs
    ``O(d^2 + db)``.  ``v0`` is reused across blocks as a warm start, which is why four
    iterations suffice.  Total cost ``O(n_iter * nblk * (d^2 + db))``, one-off and O(1)
    in ``N``.
    """
    d = X.shape[1]
    u = np.zeros(d)
    w = np.zeros(d)
    best = 0.0
    for t in range(nblk):
        i0 = t * b
        sig = 0.0
        for _ in range(n_iter):
            # u = Hhat v
            for a in range(d):
                u[a] = 0.0
            for r in range(b):
                i = i0 + r
                xdot = 0.0
                xv = 0.0
                for j in range(d):
                    xdot += X[i, j] * theta[j]
                    xv += X[i, j] * v0[j]
                c = _alpha(model_code, xdot) * xv / b
                for j in range(d):
                    u[j] += c * X[i, j]
            # w = W * Ainv u   (=  M v)
            for a in range(d):
                acc = 0.0
                for bb in range(d):
                    acc += Ainv[a, bb] * u[bb]
                w[a] = W * acc
            # u = W * Ainv w  (Ainv is symmetric, so this is the first half of M^T w)
            for a in range(d):
                acc = 0.0
                for bb in range(d):
                    acc += Ainv[a, bb] * w[bb]
                u[a] = W * acc
            # v = Hhat u  (= M^T M v)
            for a in range(d):
                v0[a] = 0.0
            for r in range(b):
                i = i0 + r
                xdot = 0.0
                xu = 0.0
                for j in range(d):
                    xdot += X[i, j] * theta[j]
                    xu += X[i, j] * u[j]
                c = _alpha(model_code, xdot) * xu / b
                for j in range(d):
                    v0[j] += c * X[i, j]
            nrm = 0.0
            for a in range(d):
                nrm += v0[a] * v0[a]
            nrm = np.sqrt(nrm)
            if nrm <= 0.0:
                sig = 0.0
                break
            for a in range(d):
                v0[a] /= nrm
            sig = np.sqrt(nrm)
        if sig > best:
            best = sig
    return best


# --------------------------------------------------------------------------------------
# Blocked-Gapped Streaming Newton  (this paper) and the i.i.d. base method
# --------------------------------------------------------------------------------------
@njit(cache=True)
def bgsn_run(X, y, theta0, b, m, p, w_hess, ci_reg, ci_pow, lam0, model_code,
             seed, want_plugin, traj_every, n_warm, auto_ridge,
             a_step, c_step, w_avg, n_warm_max, warm_tol, t0_mult, step_cap):
    """Streaming quasi-Newton with blocked gradients and gapped curvature updates.

    Parameters
    ----------
    b : block length (contiguous in time).
    m : curvature gap.  A rank-one curvature update is attempted only at global
        indices ``i`` with ``i % m == 0``; ``m = 1`` recovers the base method's
        "every observation is a candidate" rule.
    p : Bernoulli retention probability for candidate curvature updates
        (the base method's cost knob).  ``p = 1`` makes the schedule deterministic.
    w_hess : exponent w' of the log-weights ``log(t+1)^{w'}`` on curvature terms.
    ci_reg, ci_pow : constants of the vanishing ridge ``iota_n = ci_reg * n^{-ci_pow}``
        applied along a cycling basis direction; this keeps ``lambda_min(Hbar)`` bounded
        away from zero without ever forming or inverting a matrix.
    lam0 : ``H_0 = lam0 * scale * I`` where ``scale`` is the mean curvature magnitude
        of the first block, making ``lam0`` dimensionless.  Any value well below
        ``warm_updates * lambda_min(H) / scale`` behaves identically (see ablations).
    n_warm : minimum number of leading blocks used for curvature only (no parameter step,
        and every observation contributing a rank-one update).  This costs
        ``O(n_warm * b * d^2)`` flops -- O(1) in N -- and exists purely to stop the
        1/t schedule from starting while the curvature estimate is still rank-deficient.
        The step index restarts at 1 after the warm-up, so the asymptotics are unchanged.
    n_warm_max, warm_tol : the warm-up is *self-terminating*.  Counting observations is not
        enough: on a regime-switching stream, consecutive observations are near-duplicates,
        so a fixed number of them can span very few effectively distinct design points and
        leave the curvature estimate badly conditioned -- which makes the 1/t schedule
        unstable.  We therefore continue past ``n_warm`` blocks until the harmonic mean of
        the current curvature eigenvalues, ``varsigma = d / tr(Hbar^{-1})`` (available for
        free from the running inverse), has changed by less than ``warm_tol`` in relative
        terms over the last block, stopping in any case at ``n_warm_max`` blocks.  Set
        ``warm_tol <= 0`` to disable and use exactly ``n_warm`` blocks.
    auto_ridge : if 1, scale the vanishing ridge by ``varsigma_star``, the mean curvature
        magnitude of the FIRST block carrying curvature slots, frozen thereafter.  This makes
        ``ci_reg`` dimensionless, i.e. the algorithm invariant to rescaling of the loss.  The
        scale must be frozen rather than tracked: a scale proportional to the *running*
        curvature (an earlier version used ``d / tr(Hbar^{-1})``) shrinks exactly when the
        curvature estimate degenerates, and then the ridge lower bound of Lemma 2 presupposes
        its own conclusion.  If 0, the ridge is unscaled and ``ci_reg`` carries units.
    want_plugin : also accumulate the i.i.d. plug-in score covariance
        ``Gamma0hat = N^{-1} sum_i s_i s_i'`` (costs O(d^2) *per observation*).
    step_cap : if > 0, bound each parameter step by ``step_cap * (1 + ||theta_{t-1}||)``.
        Blocking makes this necessary: the error recursion has spectral radius below one only
        while ``gamma_t lambda_max(Hbar^{-1} Hhat_t) < 2``, and on a dependent stream a block
        can span far fewer than ``b`` distinct covariate directions, so ``Hhat_t`` is nearly
        singular and a full first step overshoots by orders of magnitude.  The bound is
        scale-free in theta, costs O(d), binds a handful of times per run, and is inactive
        once the steps are small (Proposition 1).  ``step_cap = 0`` disables it.
    t0_mult : if > 0, use the REJECTED alternative safeguard instead: the step-size shift
        ``gamma_t = c_step (t + t0)^{-a_step}`` with
        ``t0 = t0_mult * c_step * max_t ||Hbar^{-1} Hhat_t|| - 1`` measured over the warm-up
        blocks.  Kept because the paper reports the comparison; it is worse, because it slows
        every step to correct a few, and on a short stream ``t0`` can exceed the number of
        steps.  Set ``t0_mult = 0`` (the default) to use ``step_cap`` alone.
    a_step, c_step : step sequence ``gamma_t = c_step * t^{-a_step}``.  ``a_step = 1``
        with ``c_step = 1`` is the asymptotically efficient no-averaging Newton schedule;
        ``a_step in (1/2, 1)`` with weighted Polyak--Ruppert averaging is the robust
        variant.  Because the step is preconditioned by ``Hbar^{-1}``, ``c_step = 1`` is
        scale free -- no learning rate has to be tuned.
    w_avg : exponent of the log-weights in the averaged iterate
        ``thetabar_t = sum_i log(i+1)^{w_avg} theta_i / sum_i log(i+1)^{w_avg}``.
    traj_every : record the iterate every this many blocks (0 = off).

    Returns
    -------
    theta, theta_bar, Ainv, W, S_bm, S_ft, Gamma0_hat, n_curv, n_blocks_kept,
    traj_n, traj_theta
    """
    N, d = X.shape
    T = N // b
    theta = theta0.copy()
    # H_0 = lam0 * scale * I with scale = mean curvature magnitude of the first block.
    # Multiplying the loss by c multiplies scale by c, so the algorithm is exactly
    # equivariant under rescaling of the objective and lam0 is a dimensionless constant.
    scale = 0.0
    for r in range(b):
        xdot = 0.0
        nrm = 0.0
        for j in range(d):
            xdot += X[r, j] * theta[j]
            nrm += X[r, j] * X[r, j]
        scale += _alpha(model_code, xdot) * nrm
    scale = scale / (b * d)
    if scale <= 0.0:
        scale = 1.0
    Ainv = np.zeros((d, d))
    for i in range(d):
        Ainv[i, i] = 1.0 / (lam0 * scale)
    W = 1.0

    g = np.zeros(d)
    g_prev = np.zeros(d)
    have_prev = False
    work = np.zeros(d)
    step = np.zeros(d)
    tbar = theta0.copy()
    wsum = 0.0
    acc_prev = np.zeros((d, d))
    acc_cur = np.zeros((d, d))
    crs_prev = np.zeros((d, d))
    crs_cur = np.zeros((d, d))
    npair_prev = 0
    npair_cur = 0
    cnt_prev = 0
    cnt_cur = 0
    next_pow = 1
    G0 = np.zeros((d, d))
    n_plugin = 0
    n_curv = 0

    # only post-warm-up blocks take a parameter step, so only those are recorded;
    # sizing this as T would leave n_warm trailing rows of zeros in the trajectory
    n_traj = 0 if traj_every == 0 else max(T - n_warm, 0) // traj_every
    traj_n = np.zeros(max(n_traj, 1))
    traj_theta = np.zeros((max(n_traj, 1), d))
    itraj = 0

    np.random.seed(seed)
    # curvature slots inside a block: at most b of them during warm-up, else b/m
    max_slots = b + 1
    alph = np.zeros(max_slots)
    slot = np.zeros(max_slots, dtype=np.int64)

    warm_done = -1           # index of the first block that takes a parameter step
    warm_stop = -1           # block at which the self-terminating criterion fired
    t0 = 0.0                 # step-size shift, set once when the warm-up ends
    n_clip = 0               # blocks on which the step-length safeguard bound
    varsig_fix = 0.0         # frozen ridge scale, set on the first curvature-bearing block
    v_pow = np.ones(d) / np.sqrt(float(d))   # power-iteration warm start
    varsig_prev = 0.0
    for t in range(T):
        i0 = t * b
        warm = (t < n_warm) or (warm_tol > 0.0 and warm_stop < 0 and t < n_warm_max)
        if warm_done < 0 and not warm:
            warm_done = t
            # Stability shift.  The blocked error recursion contracts only while
            # gamma_t * ||Hbar^{-1} Hhat_t|| < 2, so with gamma_t = c/(t + t0) it is
            # enough to take t0 >= (c/2) * max_t ||Hbar^{-1} Hhat_t||.  We measure that
            # maximum on the warm-up window, which carries no parameter step, so t0 is a
            # finite constant fixed before the first step (Proposition 6).
            if t0_mult > 0.0 and t > 0:
                amp = _block_amplification(X, y, theta, Ainv, W, b, t, model_code,
                                           4, v_pow)
                t0 = t0_mult * c_step * amp - 1.0
                if t0 < 0.0:
                    t0 = 0.0
        # ---- pass 1 over the block at the *pre-step* parameter -------------------
        for j in range(d):
            g[j] = 0.0
        ns = 0
        for r in range(b):
            i = i0 + r
            xdot = 0.0
            for j in range(d):
                xdot += X[i, j] * theta[j]
            c = _score_scale(model_code, xdot, y[i])
            for j in range(d):
                g[j] += c * X[i, j]
            # The i.i.d. plug-in score covariance must be accumulated on the SAME
            # observations as the blocked estimators, i.e. excluding the warm-up blocks,
            # where theta is still theta_0 and the scores are large.  Including them
            # inflates the dependence-blind competitor's variance for a purely
            # implementational reason -- badly so on real data, where the warm-up can be a
            # sizeable fraction of a short segment.
            if want_plugin == 1 and not warm:
                cc = c * c
                for a in range(d):
                    xa = cc * X[i, a]
                    for bb in range(d):
                        G0[a, bb] += xa * X[i, bb]
                n_plugin += 1
            if warm or i % m == 0:
                if warm or p >= 1.0 or np.random.random() < p:
                    alph[ns] = _alpha(model_code, xdot)
                    slot[ns] = i
                    ns += 1
        for j in range(d):
            g[j] /= b

        if varsig_fix <= 0.0 and ns > 0:
            # Scale of the ridge, FROZEN on the first block that carries curvature slots.
            # It is the mean curvature magnitude d^{-1} b^{-1} sum alpha_i ||Psi_i||^2 of that
            # block: positive and finite almost surely under Assumption 3, and measurable with
            # respect to the first block alone.  Freezing it is what makes the ridge lower
            # bound of Lemma 2 unconditional: with a scale that tracked the running curvature
            # the ridge would shrink whenever the curvature estimate did, and the lemma would
            # presuppose its own conclusion.  auto_ridge = 0 uses an unscaled ridge instead,
            # which is dimensionful; the ablations show the two agree.
            acc = 0.0
            for q in range(ns):
                nq = 0.0
                for j in range(d):
                    nq += X[slot[q], j] * X[slot[q], j]
                acc += alph[q] * nq
            varsig_fix = acc / (ns * d)
            if varsig_fix <= 0.0 or auto_ridge != 1:
                varsig_fix = 1.0

        if warm:
            # curvature-only warm-up block: no parameter step, no covariance accrual
            wt = 1.0
            for q in range(ns):
                n_curv += 1
                iota = ci_reg * varsig_fix * n_curv ** (-ci_pow)
                _riccati_basis(Ainv, (n_curv - 1) % d, wt * iota)
                _riccati_vec(Ainv, X[slot[q]], wt * alph[q], work)
                W += wt
            # self-terminating criterion: has the harmonic-mean curvature settled?
            if warm_tol > 0.0 and t + 1 >= n_warm:
                tr = 0.0
                for a in range(d):
                    tr += Ainv[a, a]
                varsig = d / (W * tr)
                if varsig_prev > 0.0:
                    rel = abs(varsig - varsig_prev) / varsig
                    if rel < warm_tol:
                        warm_stop = t         # stop warming from the next block on
                varsig_prev = varsig
            continue

        # ---- Newton step:  theta <- theta - gamma_t * (Hbar^{-1}) g --------------
        if warm_done < 0:
            warm_done = t
        tp = t - warm_done + 1
        gam = c_step * (tp + t0) ** (-a_step)
        for a in range(d):
            s = 0.0
            for bb in range(d):
                s += Ainv[a, bb] * g[bb]
            step[a] = gam * W * s
        # Step-length safeguard.  The blocked error recursion contracts only while
        # gamma_t ||Hbar^{-1} Hhat_t|| < 2, and on a dependent stream a block can span far
        # fewer than b distinct covariate directions, so that norm can be large and the step
        # can overshoot by orders of magnitude (Section 2.1).  Rather than slow the whole
        # schedule -- which would be fatal on a short stream, where the number of steps is
        # not much larger than the norm -- we bound each step relative to the current
        # iterate: ||s_t|| <= c_D (1 + ||theta_{t-1}||).  This is scale-free in theta, costs
        # O(d), and is inactive for all large t because ||s_t|| = O_p(gamma_t) -> 0 while
        # theta_t -> theta*, so it changes no asymptotic statement (Proposition 6).
        if step_cap > 0.0:
            sn = 0.0
            tn = 0.0
            for a in range(d):
                sn += step[a] * step[a]
                tn += theta[a] * theta[a]
            sn = np.sqrt(sn)
            lim = step_cap * (1.0 + np.sqrt(tn))
            if sn > lim:
                sc = lim / sn
                for a in range(d):
                    step[a] *= sc
                n_clip += 1
        for a in range(d):
            theta[a] -= step[a]

        # ---- weighted Polyak--Ruppert average of the post-step iterates ----------
        wt_a = 1.0
        if w_avg != 0.0:
            wt_a = np.log(tp + 1.0) ** w_avg
        wsum += wt_a
        for a in range(d):
            tbar[a] += (wt_a / wsum) * (theta[a] - tbar[a])

        # ---- long-run covariance accumulators -----------------------------------
        if tp == next_pow:
            _roll_dyadic(acc_prev, acc_cur, d)
            _roll_dyadic(crs_prev, crs_cur, d)
            cnt_prev = cnt_cur
            cnt_cur = 0
            npair_prev = npair_cur
            npair_cur = 0
            next_pow *= 2
        for a in range(d):
            ga = g[a]
            for bb in range(d):
                acc_cur[a, bb] += ga * g[bb]
        cnt_cur += 1
        if have_prev:
            for a in range(d):
                ga = g_prev[a]
                for bb in range(d):
                    crs_cur[a, bb] += ga * g[bb]
            npair_cur += 1
        for j in range(d):
            g_prev[j] = g[j]
        have_prev = True

        # ---- curvature updates (gapped, optionally Bernoulli-thinned) ------------
        if ns > 0:
            wt = 1.0
            if w_hess != 0.0:
                wt = np.log(tp + 1.0) ** w_hess
            for q in range(ns):
                n_curv += 1
                iota = ci_reg * varsig_fix * n_curv ** (-ci_pow)
                _riccati_basis(Ainv, (n_curv - 1) % d, wt * iota)
                _riccati_vec(Ainv, X[slot[q]], wt * alph[q], work)
                W += wt

        if traj_every != 0 and tp % traj_every == 0 and itraj < n_traj:
            traj_n[itraj] = (t + 1) * b
            for j in range(d):
                traj_theta[itraj, j] = theta[j]
            itraj += 1

    nkeep = cnt_prev + cnt_cur
    S_bm, S_ft = _finish_lrv(acc_prev, acc_cur, nkeep, crs_prev, crs_cur,
                             npair_prev + npair_cur, b, d)
    if n_plugin > 0:
        for a in range(d):
            for bb in range(d):
                G0[a, bb] /= n_plugin
    return (theta, tbar, Ainv, W, S_bm, S_ft, G0, n_curv, nkeep, traj_n[:itraj],
            traj_theta[:itraj], max(warm_done, 0), t0, n_clip)


# --------------------------------------------------------------------------------------
# First-order baselines: averaged SGD and streaming AdaGrad
# --------------------------------------------------------------------------------------
@njit(cache=True)
def first_order_run(X, y, theta0, b, c_step, a_step, method_code, adagrad_eps,
                    model_code, want_plugin, traj_every):
    """Averaged SGD (``method_code = 0``) or streaming AdaGrad (``method_code = 1``).

    Both are run on the same contiguous blocks as the quasi-Newton method, so all
    estimators see identical data and pay O(d) per observation for the gradient.
    In addition to the Polyak--Ruppert average this routine streams

      * the batch-means and two-scale long-run covariances ``S_bm``, ``S_ft``,
      * the i.i.d. plug-in ``Gamma0_hat`` (optional, O(d^2) per observation),
      * the random-scaling matrix ``V_rs`` of Lee-Liao-Seo-Shin / Kiefer-Vogelsang type,
        accumulated in the streaming form
        ``V = T^{-2}[ sum_s P_s P_s' - T^{-1} sum_s s (P_s P_T' + P_T P_s')
                      + T^{-2} (sum_s s^2) P_T P_T' ]``, ``P_s = sum_{k<=s} theta_k``.

    Returns theta_bar, S_bm, S_ft, Gamma0_hat, V_rs, traj_n, traj_theta_bar
    """
    N, d = X.shape
    T = N // b
    theta = theta0.copy()
    tbar = np.zeros(d)
    G0 = np.zeros((d, d))
    n_plugin = 0
    g = np.zeros(d)
    g_prev = np.zeros(d)
    have_prev = False
    acc_grad2 = np.zeros(d)

    acc_prev = np.zeros((d, d))
    acc_cur = np.zeros((d, d))
    crs_prev = np.zeros((d, d))
    crs_cur = np.zeros((d, d))
    npair_prev = 0
    npair_cur = 0
    cnt_prev = 0
    cnt_cur = 0
    next_pow = 1

    P = np.zeros(d)             # running sum of iterates  P_s
    SPP = np.zeros((d, d))      # sum_s P_s P_s'
    SsP = np.zeros(d)           # sum_s s P_s
    Ss2 = 0.0                   # sum_s s^2

    n_traj = 0 if traj_every == 0 else T // traj_every
    traj_n = np.zeros(max(n_traj, 1))
    traj_theta = np.zeros((max(n_traj, 1), d))
    itraj = 0

    for t in range(T):
        i0 = t * b
        for j in range(d):
            g[j] = 0.0
        for r in range(b):
            i = i0 + r
            xdot = 0.0
            for j in range(d):
                xdot += X[i, j] * theta[j]
            c = _score_scale(model_code, xdot, y[i])
            for j in range(d):
                g[j] += c * X[i, j]
            if want_plugin == 1:
                cc = c * c
                for a in range(d):
                    xa = cc * X[i, a]
                    for bb in range(d):
                        G0[a, bb] += xa * X[i, bb]
                n_plugin += 1
        for j in range(d):
            g[j] /= b

        gam = c_step * (t + 1.0) ** (-a_step)
        if method_code == 0:
            for j in range(d):
                theta[j] -= gam * g[j]
        else:
            for j in range(d):
                acc_grad2[j] += g[j] * g[j]
                theta[j] -= gam * g[j] / (np.sqrt(acc_grad2[j]) + adagrad_eps)

        # Polyak--Ruppert average of the post-update iterates
        for j in range(d):
            tbar[j] += (theta[j] - tbar[j]) / (t + 1.0)

        if t + 1 == next_pow:
            _roll_dyadic(acc_prev, acc_cur, d)
            _roll_dyadic(crs_prev, crs_cur, d)
            cnt_prev = cnt_cur
            cnt_cur = 0
            npair_prev = npair_cur
            npair_cur = 0
            next_pow *= 2
        for a in range(d):
            ga = g[a]
            for bb in range(d):
                acc_cur[a, bb] += ga * g[bb]
        cnt_cur += 1
        if have_prev:
            for a in range(d):
                ga = g_prev[a]
                for bb in range(d):
                    crs_cur[a, bb] += ga * g[bb]
            npair_cur += 1
        for j in range(d):
            g_prev[j] = g[j]
        have_prev = True

        s1 = t + 1.0
        for j in range(d):
            P[j] += theta[j]
        for a in range(d):
            Pa = P[a]
            for bb in range(d):
                SPP[a, bb] += Pa * P[bb]
            SsP[a] += s1 * Pa
        Ss2 += s1 * s1

        if traj_every != 0 and (t + 1) % traj_every == 0 and itraj < n_traj:
            traj_n[itraj] = (t + 1) * b
            for j in range(d):
                traj_theta[itraj, j] = tbar[j]
            itraj += 1

    Tf = float(T)
    V = np.zeros((d, d))
    for a in range(d):
        for bb in range(d):
            V[a, bb] = (SPP[a, bb]
                        - (SsP[a] * P[bb] + P[a] * SsP[bb]) / Tf
                        + Ss2 * P[a] * P[bb] / (Tf * Tf)) / (Tf * Tf)

    nkeep = cnt_prev + cnt_cur
    S_bm, S_ft = _finish_lrv(acc_prev, acc_cur, nkeep, crs_prev, crs_cur,
                             npair_prev + npair_cur, b, d)
    if n_plugin > 0:
        for a in range(d):
            for bb in range(d):
                G0[a, bb] /= n_plugin
    return tbar, S_bm, S_ft, G0, V, traj_n, traj_theta


# --------------------------------------------------------------------------------------
# curvature-only driver: isolates the effect of gapping on Hessian estimation
# --------------------------------------------------------------------------------------
@njit(cache=True)
def curvature_only(X, theta, m, p, w_hess, ci_reg, ci_pow, lam0, model_code, seed, b):
    """Run *only* the curvature recursion at a frozen theta and return ``Hbar``.

    Used by the ablation that compares deterministic gapping against Bernoulli
    thinning at a matched number of rank-one updates, with the optimisation dynamics
    switched off so that the two schedules are compared on estimation quality alone.
    """
    N, d = X.shape
    T = N // b
    Ainv = np.zeros((d, d))
    for i in range(d):
        Ainv[i, i] = 1.0 / lam0
    W = 1.0
    work = np.zeros(d)
    n_curv = 0
    np.random.seed(seed)
    for t in range(T):
        wt = 1.0
        if w_hess != 0.0:
            wt = np.log(t + 2.0) ** w_hess
        for r in range(b):
            i = t * b + r
            if i % m == 0:
                if p >= 1.0 or np.random.random() < p:
                    xdot = 0.0
                    for j in range(d):
                        xdot += X[i, j] * theta[j]
                    n_curv += 1
                    iota = ci_reg * n_curv ** (-ci_pow)
                    _riccati_basis(Ainv, (n_curv - 1) % d, wt * iota)
                    _riccati_vec(Ainv, X[i], wt * _alpha(model_code, xdot), work)
                    W += wt
    return Ainv, W, n_curv
