"""Dependent data-generating processes (DGPs) with *exact* population targets.

Each DGP returns a stream (X, y) of length N together with an ``Oracle`` holding

    H      = grad^2 F(theta*)                      (population Hessian)
    Gamma0 = E[ s(theta*) s(theta*)^T ]            (instantaneous score covariance)
    S      = sum_{k in Z} Cov(s_0(theta*), s_k(theta*))   (long-run score covariance)

``S`` is the object that governs the asymptotic distribution of any streaming
M-estimator on a dependent stream, and ``Gamma0`` is what an i.i.d.-assuming method
implicitly (and wrongly) uses in its place.

Three of the four DGPs have closed-form ``S``; DGP ``logit_copula`` has no tractable
closed form and its oracle is obtained by a long independent simulation whose Monte
Carlo standard error we report (see ``oracle_error``).

Design note (this matters and is easy to get wrong)
---------------------------------------------------
Serial dependence in the *covariates alone* does not make ``S`` differ from
``Gamma0``.  If the conditional mean / conditional law of y given x is correctly
specified with innovations that are independent over time, the score is a martingale
difference sequence and ``S == Gamma0`` exactly, no matter how persistent x is.  A
genuine long-run-variance problem needs the *score* to be serially correlated.  All
DGPs below therefore combine dependent covariates with a serially dependent error /
latent process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import lfilter


# --------------------------------------------------------------------------------------
# Oracle container
# --------------------------------------------------------------------------------------
@dataclass
class Oracle:
    theta_star: np.ndarray
    H: np.ndarray
    Gamma0: np.ndarray
    S: np.ndarray
    exact: bool = True
    oracle_error: Optional[float] = None  # relative MC error of S, if not exact
    meta: dict = field(default_factory=dict)

    @property
    def sandwich_true(self) -> np.ndarray:
        """H^{-1} S H^{-1}: the correct asymptotic covariance of sqrt(N)(theta-theta*)."""
        Hi = np.linalg.inv(self.H)
        return Hi @ self.S @ Hi

    @property
    def sandwich_naive(self) -> np.ndarray:
        """H^{-1} Gamma0 H^{-1}: what an i.i.d.-assuming plug-in converges to."""
        Hi = np.linalg.inv(self.H)
        return Hi @ self.Gamma0 @ Hi

    @property
    def kappa(self) -> np.ndarray:
        """Per-coordinate variance-inflation factor kappa_j (true / naive)."""
        return np.diag(self.sandwich_true) / np.diag(self.sandwich_naive)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def illconditioned_cov(d: int, rng: np.random.Generator, exponent: float = 2.0) -> np.ndarray:
    """Covariance ``M diag((i/d)^exponent) M^T`` with M a random orthogonal matrix.

    This is the ill-conditioned design of Boyer & Godichon-Baggioni used by the base
    paper: eigenvalues span d^exponent orders, and M makes the coordinates strongly
    correlated so that diagonal preconditioners (AdaGrad) cannot exploit the structure.
    """
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    lam = ((np.arange(1, d + 1) / d) ** exponent)
    return (Q * lam) @ Q.T


def _ar1_vector(N: int, d: int, phi: np.ndarray, Sig_innov_chol: np.ndarray,
                x0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate x_t = diag(phi) x_{t-1} + L eta_t, eta_t ~ N(0, I), given x_0.

    Uses ``scipy.signal.lfilter`` per coordinate: O(Nd) with no Python-level time loop.
    """
    eta = rng.standard_normal((N, d))
    W = eta @ Sig_innov_chol.T                      # innovations, N(0, Sig_innov)
    X = np.empty((N, d))
    for j in range(d):
        # y[n] = phi_j y[n-1] + W[n, j], initial state carries phi_j * x0_j
        X[:, j] = lfilter([1.0], [1.0, -phi[j]], W[:, j], zi=[phi[j] * x0[j]])[0]
    return X


def _ar1_scalar(N: int, psi: float, sd_innov: float, u0: float,
                rng: np.random.Generator) -> np.ndarray:
    e = rng.standard_normal(N) * sd_innov
    return lfilter([1.0], [1.0, -psi], e, zi=[psi * u0])[0]


def markov_chain(N: int, P: np.ndarray, pi: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """Sample a stationary finite-state Markov chain of length N by run lengths.

    Holding time in state s is Geometric(1 - P[s, s]); the destination is drawn from
    the normalised off-diagonal row.  The Python loop runs once per *run*, i.e.
    O(N (1 - min_s P[s,s])) times, which is cheap for persistent chains.
    """
    K = P.shape[0]
    off = P.copy()
    np.fill_diagonal(off, 0.0)
    stay = np.diag(P).copy()
    dest_p = off / off.sum(axis=1, keepdims=True)
    dest_cdf = np.cumsum(dest_p, axis=1)

    out = np.empty(N, dtype=np.int64)
    s = int(rng.choice(K, p=pi))
    pos = 0
    while pos < N:
        q = 1.0 - stay[s]
        hold = 1 if q >= 1.0 else int(rng.geometric(q))
        end = min(N, pos + hold)
        out[pos:end] = s
        pos = end
        if pos < N:
            s = int(np.searchsorted(dest_cdf[s], rng.random()))
    return out


def two_state_chain(p01: float, p10: float):
    P = np.array([[1 - p01, p01], [p10, 1 - p10]])
    pi = np.array([p10, p01]) / (p01 + p10)
    return P, pi


# --------------------------------------------------------------------------------------
# DGP 1: linear model, homogeneous AR(1) covariates + AR(1) errors  (closed-form S)
# --------------------------------------------------------------------------------------
def dgp_lin_hom(N: int, d: int, phi: float, psi: float, sigma_u: float = 1.0,
                cond_exponent: float = 2.0, seed: int = 0,
                theta_star: Optional[np.ndarray] = None,
                Sigma_x: Optional[np.ndarray] = None):
    r"""y_t = x_t' theta* + u_t, with

        x_t = phi x_{t-1} + sqrt(1-phi^2) Sigma_x^{1/2} eta_t,   eta_t ~ N(0, I)
        u_t = psi u_{t-1} + sqrt(1-psi^2) sigma_u eps_t,          eps_t ~ N(0, 1)

    x and u independent.  Then the stationary marginals are x ~ N(0, Sigma_x),
    Var(u) = sigma_u^2, and, with score s_t = -(y_t - x_t'theta*) x_t = -u_t x_t,

        H      = Sigma_x
        Gamma0 = sigma_u^2 Sigma_x
        Gamma_k = sigma_u^2 (phi psi)^{|k|} Sigma_x
        S      = sigma_u^2 Sigma_x (1 + phi psi) / (1 - phi psi).

    Hence kappa_j = (1 + phi psi) / (1 - phi psi) for *every* coordinate j -- an exact
    scalar prediction we use to validate the theory.
    """
    rng = np.random.default_rng(seed)
    if Sigma_x is None:
        Sigma_x = illconditioned_cov(d, np.random.default_rng(10_000 + seed), cond_exponent)
    if theta_star is None:
        theta_star = np.linspace(-1.0, 1.0, d)

    L = np.linalg.cholesky(Sigma_x)
    x0 = L @ rng.standard_normal(d)                              # exact stationary start
    X = _ar1_vector(N, d, np.full(d, phi), np.sqrt(1 - phi ** 2) * L, x0, rng)
    u0 = sigma_u * rng.standard_normal()
    u = _ar1_scalar(N, psi, sigma_u * np.sqrt(1 - psi ** 2), u0, rng)
    y = X @ theta_star + u

    r = phi * psi
    orc = Oracle(theta_star=theta_star, H=Sigma_x.copy(),
                 Gamma0=sigma_u ** 2 * Sigma_x,
                 S=sigma_u ** 2 * Sigma_x * (1 + r) / (1 - r),
                 exact=True,
                 meta=dict(dgp="lin_hom", phi=phi, psi=psi, kappa_scalar=(1 + r) / (1 - r)))
    return X, y, orc


# --------------------------------------------------------------------------------------
# DGP 2: linear model, heterogeneous persistence across coordinates (closed-form S)
# --------------------------------------------------------------------------------------
def dgp_lin_het(N: int, d: int, phi_lo: float = 0.0, phi_hi: float = 0.9,
                psi: float = 0.7, sigma_u: float = 1.0, cond_exponent: float = 2.0,
                seed: int = 0, theta_star: Optional[np.ndarray] = None,
                Sigma_eps: Optional[np.ndarray] = None):
    r"""x_t = Phi x_{t-1} + Sigma_eps^{1/2} eta_t with Phi = diag(phi_1..phi_d).

    Stationary covariance: Sigma_x = Sigma_eps ./ (1 - phi_i phi_j) (Hadamard), which
    is positive definite because [1/(1-phi_i phi_j)] = sum_{k>=0} (phi phi')^{o k} is a
    sum of Hadamard powers of a rank-one PSD matrix.  With AR(1) errors,

        E[x_t x_{t+k}'] = Sigma_x Phi^k  (k >= 0),   Gamma_k = sigma_u^2 psi^k Sigma_x Phi^k
        S = sigma_u^2 [ Sigma_x + Sigma_x M + M Sigma_x ],  M = psi Phi (I - psi Phi)^{-1}.

    Because the phi_j differ, kappa_j varies across coordinates: a *matrix*-level test
    that no scalar correction can pass.
    """
    rng = np.random.default_rng(seed)
    if Sigma_eps is None:
        Sigma_eps = illconditioned_cov(d, np.random.default_rng(20_000 + seed), cond_exponent)
    if theta_star is None:
        theta_star = np.linspace(-1.0, 1.0, d)

    phi = np.linspace(phi_lo, phi_hi, d)
    C = 1.0 / (1.0 - np.outer(phi, phi))
    Sigma_x = Sigma_eps * C
    Lx = np.linalg.cholesky(Sigma_x)
    Le = np.linalg.cholesky(Sigma_eps)

    x0 = Lx @ rng.standard_normal(d)
    X = _ar1_vector(N, d, phi, Le, x0, rng)
    u0 = sigma_u * rng.standard_normal()
    u = _ar1_scalar(N, psi, sigma_u * np.sqrt(1 - psi ** 2), u0, rng)
    y = X @ theta_star + u

    M = np.diag(psi * phi / (1.0 - psi * phi))
    S = sigma_u ** 2 * (Sigma_x + Sigma_x @ M + M @ Sigma_x)
    orc = Oracle(theta_star=theta_star, H=Sigma_x.copy(), Gamma0=sigma_u ** 2 * Sigma_x,
                 S=S, exact=True,
                 meta=dict(dgp="lin_het", phi=phi.copy(), psi=psi))
    return X, y, orc


# --------------------------------------------------------------------------------------
# DGP 3: logistic regression, correctly specified margin, serially dependent latent error
# --------------------------------------------------------------------------------------
def _logit_copula_labels(S_lin: np.ndarray, psi: float, rng: np.random.Generator):
    """Labels y_t = 1{ nu_t > -Phi^{-1}(sigma(s_t)) } with nu_t a unit-variance AR(1).

    Equivalent to y_t = 1{s_t + e_t > 0} where e_t = logit(Phi(nu_t)) has an exact
    standard logistic margin.  Hence P(y_t = 1 | x_t) = sigma(x_t'theta*) *exactly*:
    the working logistic likelihood is correctly specified marginally and theta* is the
    true parameter, yet the score is serially correlated through nu.
    """
    from scipy.special import expit, ndtri
    N = S_lin.shape[0]
    nu0 = rng.standard_normal()
    nu = _ar1_scalar(N, psi, np.sqrt(1 - psi ** 2), nu0, rng)
    thr = -ndtri(np.clip(expit(S_lin), 1e-15, 1 - 1e-15))
    return (nu > thr).astype(np.float64)


def dgp_logit_copula(N: int, d: int, phi: float = 0.7, psi: float = 0.8,
                     cond_exponent: float = 1.0, seed: int = 0,
                     theta_star: Optional[np.ndarray] = None,
                     Sigma_x: Optional[np.ndarray] = None,
                     oracle: Optional[Oracle] = None):
    """Logistic regression with AR(1) covariates and Gaussian-copula AR(1) latent error.

    ``H`` and ``Gamma0`` are exact (they coincide by the information equality, which
    holds because the marginal conditional law is correctly specified, and both are
    computed by Gauss--Hermite quadrature).  ``S`` has no closed form and must be
    supplied via ``oracle`` (see :func:`logit_copula_oracle`).
    """
    rng = np.random.default_rng(seed)
    if Sigma_x is None:
        Sigma_x = illconditioned_cov(d, np.random.default_rng(30_000 + seed), cond_exponent)
    if theta_star is None:
        theta_star = np.linspace(-1.0, 1.0, d)

    L = np.linalg.cholesky(Sigma_x)
    x0 = L @ rng.standard_normal(d)
    X = _ar1_vector(N, d, np.full(d, phi), np.sqrt(1 - phi ** 2) * L, x0, rng)
    y = _logit_copula_labels(X @ theta_star, psi, rng)

    if oracle is None:
        H = logistic_hessian_gauss(Sigma_x, theta_star)
        oracle = Oracle(theta_star=theta_star, H=H, Gamma0=H.copy(), S=H.copy(),
                        exact=False, meta=dict(dgp="logit_copula", note="S not filled"))
    return X, y, oracle


def logistic_hessian_gauss(Sigma_x: np.ndarray, theta: np.ndarray,
                           n_nodes: int = 200) -> np.ndarray:
    r"""Exact ``E[sigma(s)(1-sigma(s)) x x^T]`` for ``x ~ N(0, Sigma_x)``, ``s = x'theta``.

    Decompose x = (Sigma_x theta / sigma_s^2) s + x_perp with x_perp independent of s
    (jointly Gaussian).  Then

        E[w(s) x x'] = E[w(s) s^2] (Sigma_x theta theta' Sigma_x)/sigma_s^4
                     + E[w(s)]     (Sigma_x - Sigma_x theta theta' Sigma_x/sigma_s^2),

    and the two scalar Gaussian integrals are evaluated by Gauss--Hermite quadrature.
    """
    from numpy.polynomial.hermite_e import hermegauss
    a = Sigma_x @ theta
    s2 = float(theta @ a)
    if s2 <= 0:
        raise ValueError("theta' Sigma_x theta must be positive")
    nodes, w = hermegauss(n_nodes)              # weights for exp(-x^2/2), sum = sqrt(2 pi)
    w = w / np.sqrt(2 * np.pi)
    s = nodes * np.sqrt(s2)
    p = 1.0 / (1.0 + np.exp(-s))
    wt = p * (1 - p)
    E_w = float(np.sum(w * wt))
    E_ws2 = float(np.sum(w * wt * s ** 2))
    P = np.outer(a, a)
    return E_ws2 * P / s2 ** 2 + E_w * (Sigma_x - P / s2)


def logit_copula_oracle(d: int, phi: float, psi: float, cond_exponent: float,
                        theta_star: np.ndarray, Sigma_x: np.ndarray,
                        n_sim: int = 4_000_000, n_rep: int = 8, lag: int = 200,
                        seed: int = 987) -> Oracle:
    """Long-simulation oracle for ``S`` in :func:`dgp_logit_copula`.

    Because ``theta*`` is known exactly we can evaluate the score at the truth and
    estimate ``S`` by a Bartlett-kernel long-run covariance with a generous bandwidth.
    We repeat over ``n_rep`` independent streams and report the Monte Carlo relative
    standard error of the diagonal of ``H^{-1} S H^{-1}`` so that every downstream
    claim carries its numerical uncertainty.
    """
    H = logistic_hessian_gauss(Sigma_x, theta_star)
    Hi = np.linalg.inv(H)
    Ss, diags = [], []
    for r in range(n_rep):
        X, y, _ = dgp_logit_copula(n_sim, d, phi=phi, psi=psi, seed=seed + r,
                                   theta_star=theta_star, Sigma_x=Sigma_x)
        p = 1.0 / (1.0 + np.exp(-(X @ theta_star)))
        G = (p - y)[:, None] * X                      # (n_sim, d) scores at the truth
        G -= G.mean(axis=0, keepdims=True)
        Sh = bartlett_lrv(G, lag)
        Ss.append(Sh)
        diags.append(np.diag(Hi @ Sh @ Hi))
    S = np.mean(Ss, axis=0)
    D = np.array(diags)
    rel_se = float(np.max(D.std(axis=0, ddof=1) / np.sqrt(n_rep) / np.abs(D.mean(axis=0))))
    return Oracle(theta_star=theta_star, H=H, Gamma0=H.copy(), S=S, exact=False,
                  oracle_error=rel_se,
                  meta=dict(dgp="logit_copula", phi=phi, psi=psi, n_sim=n_sim,
                            n_rep=n_rep, lag=lag))


def bartlett_lrv(G: np.ndarray, lag: int) -> np.ndarray:
    """Newey--West (Bartlett-kernel) long-run covariance of the rows of ``G``."""
    n = G.shape[0]
    S = G.T @ G / n
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        C = G[k:].T @ G[:-k] / n
        S += w * (C + C.T)
    return S


# --------------------------------------------------------------------------------------
# DGP 4: Markov-modulated (non-Gaussian) covariates + AR(1) errors  (closed-form S)
# --------------------------------------------------------------------------------------
def dgp_markov_cov(N: int, d: int, K: Optional[int] = None, stay: float = 0.97,
                   psi: float = 0.8, sigma_u: float = 1.0, noise_scale: float = 0.05,
                   cond_exponent: float = 2.0, seed: int = 0,
                   theta_star: Optional[np.ndarray] = None,
                   path_seed: Optional[int] = None, centres: str = "simplex"):
    r"""Markov-modulated covariates: x_t = V[:, s_t] + Sigma_nu^{1/2} nu_t.

    ``s_t`` is a K-state Markov chain with stationary law pi and the centres are
    centred so that sum_s pi_s V[:, s] = 0.  Then

        Sigma_x = V diag(pi) V' + Sigma_nu
        E[x_t x_{t+k}'] = V diag(pi) P^k V' =: R_k   (k >= 1)
        Gamma_0 = sigma_u^2 Sigma_x,   Gamma_k = sigma_u^2 psi^k R_k
        S = sigma_u^2 [ Sigma_x + V diag(pi) (psi P (I - psi P)^{-1}) V'
                                + ( ... )' ]

    Covariates are a Markov-modulated Gaussian mixture: non-Gaussian marginals and
    non-linear (regime) dependence, unlike the AR designs.

    **Why the centres are a simplex.**  With randomly drawn centres and ``K << d`` the
    regime component ``V diag(pi) V'`` is low rank and lies in high-curvature directions,
    so the sandwich ``H^{-1} Gamma_k H^{-1}`` is negligible against
    ``H^{-1} Gamma_0 H^{-1}`` and the variance inflation ``kappa`` is ~1: a design with
    strong-looking regime dependence but no inferential consequence.  Setting ``K = d+1``
    and taking the centres to be a regular simplex scaled by a target covariance --
    ``V = sqrt(K) * chol(Sigma) * U`` with ``U`` an orthonormal basis of ``1^perp`` in
    ``R^K``, so that ``V pi = 0`` and ``V diag(pi) V' = Sigma`` exactly -- makes the regime
    component full rank and proportional to the target covariance, so
    ``kappa -> (1 + psi lambda_2)/(1 - psi lambda_2)`` as the idiosyncratic noise shrinks
    (``lambda_2`` is the chain's second eigenvalue).  This is the design we use.

    ``seed`` fixes the *population* (the centres V and the noise covariance);
    ``path_seed``, when given, varies only the realised path, so that Monte Carlo
    replicates share one population.
    """
    rng = np.random.default_rng(seed if path_seed is None else path_seed)
    if theta_star is None:
        theta_star = np.linspace(-1.0, 1.0, d)

    if K is None:
        K = d + 1 if centres == "simplex" else 3
    # persistent K-state chain: stay with prob `stay`, otherwise move uniformly
    P = np.full((K, K), (1.0 - stay) / (K - 1))
    np.fill_diagonal(P, stay)
    pi = np.full(K, 1.0 / K)

    Sig_t = illconditioned_cov(d, np.random.default_rng(40_000 + seed), cond_exponent)
    Sigma_nu = noise_scale * Sig_t
    if centres == "simplex":
        from scipy.linalg import null_space
        U = null_space(np.ones((1, K))).T          # (K-1, K), rows orthonormal, U 1 = 0
        if U.shape[0] < d:
            raise ValueError("simplex centres need K >= d + 1")
        V = np.sqrt(K) * np.linalg.cholesky(Sig_t) @ U[:d]
    else:
        V = np.random.default_rng(41_000 + seed).standard_normal((d, K))
    V -= (V * pi).sum(axis=1, keepdims=True) / pi.sum()      # enforce V pi = 0

    s = markov_chain(N, P, pi, rng)
    Lnu = np.linalg.cholesky(Sigma_nu)
    X = V[:, s].T + rng.standard_normal((N, d)) @ Lnu.T
    u0 = sigma_u * rng.standard_normal()
    u = _ar1_scalar(N, psi, sigma_u * np.sqrt(1 - psi ** 2), u0, rng)
    y = X @ theta_star + u

    Vp = V * pi                                             # V diag(pi)
    Sigma_x = Vp @ V.T + Sigma_nu
    Mk = psi * P @ np.linalg.inv(np.eye(K) - psi * P)
    R = Vp @ Mk @ V.T
    S = sigma_u ** 2 * (Sigma_x + R + R.T)
    orc = Oracle(theta_star=theta_star, H=Sigma_x.copy(), Gamma0=sigma_u ** 2 * Sigma_x,
                 S=S, exact=True,
                 meta=dict(dgp="markov_cov", K=K, stay=stay, psi=psi, centres=centres,
                           lambda2=float(stay - (1 - stay) / (K - 1))))
    return X, y, orc


DGPS = {
    "lin_hom": dgp_lin_hom,
    "lin_het": dgp_lin_het,
    "logit_copula": dgp_logit_copula,
    "markov_cov": dgp_markov_cov,
}

DGP_MODEL = {
    "lin_hom": "linear",
    "lin_het": "linear",
    "logit_copula": "logistic",
    "markov_cov": "linear",
}
