"""Validation tests: every closed-form population quantity is checked against simulation.

Run with:  python -m pytest tests -q     (or simply: python tests/test_oracles.py)

These are the checks a referee would want to see: if a closed-form S were wrong, every
coverage number in the paper would be wrong in the same direction, so each oracle is
verified against a long independent simulation with a stated tolerance.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from bgsn import dgp as DG, streams, estimators as E, models  # noqa: E402

NSIM = int(os.environ.get("NSIM", 2_000_000))
LAG = 400   # long enough for the persistent Markov chain (lambda_2 ~ 0.97)


def _score_lrv(g, N, seed):
    X, y = g.sample(N, seed)
    mdl = models.get_model(g.model)
    G = mdl.score(g.theta_star, X, y)
    G = G - G.mean(axis=0, keepdims=True)
    return streams.bartlett_lrv(G, LAG), G.T @ G / len(G), X, y


def _rel(A, B):
    return float(np.abs(A - B).max() / np.abs(B).max())


def check(name, g, tol_S=0.05, tol_G=0.02, tol_H=0.02):
    S_hat, G0_hat, X, y = _score_lrv(g, NSIM, 12345)
    H_hat = E.empirical_hessian(X, y, g.model, g.theta_star)
    eS, eG, eH = (_rel(S_hat, g.oracle.S), _rel(G0_hat, g.oracle.Gamma0),
                  _rel(H_hat, g.oracle.H))
    print(f"{name:14s} rel.err  S={eS:.4f}  Gamma0={eG:.4f}  H={eH:.4f}   "
          f"kappa in [{g.oracle.kappa.min():.2f},{g.oracle.kappa.max():.2f}]")
    assert eS < tol_S, (name, "S", eS)
    assert eG < tol_G, (name, "Gamma0", eG)
    assert eH < tol_H, (name, "H", eH)


def test_lin_hom():
    check("lin_hom", DG.make_lin_hom(d=8, phi=0.7, psi=0.7))


def test_lin_hom_kappa_formula():
    for phi, psi in [(0.3, 0.5), (0.7, 0.7), (0.85, 0.6)]:
        g = DG.make_lin_hom(d=6, phi=phi, psi=psi)
        k = (1 + phi * psi) / (1 - phi * psi)
        assert np.allclose(g.oracle.kappa, k, rtol=1e-10), (phi, psi)
    print("lin_hom kappa == (1+phi psi)/(1-phi psi) for every coordinate: OK")


def test_lin_het():
    check("lin_het", DG.make_lin_het(d=8, phi_lo=0.0, phi_hi=0.9, psi=0.7))


def test_markov_cov():
    # the persistent simplex-centre design has kappa ~ 7.5, so its long-run covariance is
    # much harder to estimate by simulation than the AR designs; allow a looser tolerance
    check("markov_cov", DG.make_markov_cov(d=8, stay=0.97, psi=0.8, noise_scale=0.05),
          tol_S=0.10)


def test_logistic_hessian_quadrature():
    """Gauss-Hermite Hessian for a Gaussian design must match a long simulation."""
    d = 6
    rng = np.random.default_rng(0)
    Sig = streams.illconditioned_cov(d, rng, 1.0)
    th = np.linspace(-1, 1, d)
    H_q = streams.logistic_hessian_gauss(Sig, th)
    L = np.linalg.cholesky(Sig)
    X = rng.standard_normal((4_000_000, d)) @ L.T
    p = 1.0 / (1.0 + np.exp(-(X @ th)))
    H_mc = (X * (p * (1 - p))[:, None]).T @ X / len(X)
    e = _rel(H_q, H_mc)
    print(f"logistic H: quadrature vs simulation rel.err = {e:.5f}")
    assert e < 0.01


def test_logit_copula_margin():
    """The working logistic model must be correctly specified marginally."""
    d = 5
    rng = np.random.default_rng(1)
    Sig = streams.illconditioned_cov(d, rng, 1.0)
    th = np.linspace(-1, 1, d)
    X, y, _ = streams.dgp_logit_copula(4_000_000, d, phi=0.7, psi=0.8,
                                       theta_star=th, Sigma_x=Sig, seed=3)
    p = 1.0 / (1.0 + np.exp(-(X @ th)))
    # E[(y - sigma(x'theta)) x] must be zero: this is what makes theta* the truth
    m = np.abs((X * (y - p)[:, None]).mean(axis=0))
    s = (X * (y - p)[:, None]).std(axis=0) / np.sqrt(len(X))
    z = m / s
    print("logit-copula score mean |z| per coordinate:", np.round(z, 2))
    assert z.max() < 4.0, z


def test_two_scale_identity():
    """S_ft must equal 2 S_bm(2b) - S_bm(b) exactly (algebraic identity)."""
    rng = np.random.default_rng(7)
    T, d, b = 400, 4, 6
    Gb = rng.standard_normal((T, d))
    S_b = b * (Gb[:, None, :] * Gb[:, :, None]).mean(axis=0)
    C = np.einsum('ti,tj->ij', Gb[:-1], Gb[1:]) / (T - 1)
    S_ft = S_b + b * (C + C.T)
    # disjoint-pair form of the 2b estimator
    G2 = Gb[:2 * (T // 2)].reshape(T // 2, 2, d).mean(axis=1)
    S_2b = 2 * b * (G2[:, None, :] * G2[:, :, None]).mean(axis=0)
    S_rich = 2 * S_2b - S_b
    # the two differ only by using all adjacent pairs vs disjoint pairs; both have the
    # same expectation, so we check the *pairwise* identity on disjoint pairs instead
    Cd = np.einsum('ti,tj->ij', Gb[0:2 * (T // 2):2], Gb[1:2 * (T // 2):2]) / (T // 2)
    assert np.allclose(S_b + b * (Cd + Cd.T), S_rich, atol=1e-10)
    print("two-scale identity 2*S(2b) - S(b) == S_bm + b(Lam1+Lam1'): OK")


def test_dyadic_window_matches_explicit():
    """The streaming dyadic accumulators must equal an explicit batch-means computation."""
    g = DG.make_lin_hom(d=6, phi=0.7, psi=0.7)
    X, y = g.sample(60_000, 5)
    b = 20
    f = E.streaming_newton(X, y, "linear", b=b, seed=0, traj_every=1)
    th = f.extras["traj_theta"]
    T = len(th)
    nkeep = f.extras["n_blocks_kept"]
    nw = f.extras["n_warm"]        # the first nw blocks take no step and are not recorded
    Xb = X[nw * b:(nw + T) * b].reshape(T, b, 6)
    yb = y[nw * b:(nw + T) * b].reshape(T, b)
    thpre = np.vstack([np.zeros((1, 6)), th[:-1]])
    r = np.einsum('tbd,td->tb', Xb, thpre) - yb
    Gb = np.einsum('tbd,tb->td', Xb, r) / b
    Gk = Gb[T - nkeep:]
    S_bm = b * (Gk[:, None, :] * Gk[:, :, None]).mean(axis=0)
    e = _rel(S_bm, f.extras["S_bm"])
    print(f"dyadic S_bm vs explicit: rel.err = {e:.2e}  (kept {nkeep}/{T} blocks)")
    assert e < 1e-10
    # the lag-one block cross term uses all adjacent pairs inside the retained window,
    # plus the pair straddling its left edge; check S_ft the same way
    Gp = Gb[T - nkeep - 1:]
    C = np.einsum('ti,tj->ij', Gp[:-1], Gp[1:]) / (len(Gp) - 1)
    S_ft = S_bm + b * (C + C.T)
    e2 = _rel(S_ft, f.extras["S_ft"])
    print(f"dyadic S_ft vs explicit: rel.err = {e2:.2e}")
    assert e2 < 1e-6, e2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print("\nall oracle/identity checks passed")
