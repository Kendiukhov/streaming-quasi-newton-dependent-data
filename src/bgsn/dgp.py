"""DGP objects: a *fixed* population (theta*, Sigma_x, oracle) plus a sampler.

Keeping the population fixed while only the data realisation varies is essential for
Monte Carlo studies of coverage and efficiency: if the design were redrawn per
replicate, the empirical covariance of the estimator would mix sampling variability
with population variability and could not be compared to a single oracle sandwich.
Every experiment in this repository therefore goes through this API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import streams
from .streams import Oracle


@dataclass
class DGP:
    name: str
    model: str          # "linear" | "logistic"
    d: int
    oracle: Oracle
    params: dict

    def sample(self, N: int, seed: int):
        raise NotImplementedError

    # convenience -----------------------------------------------------------------
    @property
    def theta_star(self):
        return self.oracle.theta_star


@dataclass
class _ARLike(DGP):
    def sample(self, N: int, seed: int):
        fn = self.params["fn"]
        kw = dict(self.params["kw"])
        X, y, _ = fn(N=N, seed=seed, **kw)
        return X, y


def _theta_star(d: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, d)


def make_lin_hom(d: int = 20, phi: float = 0.7, psi: float = 0.7, sigma_u: float = 1.0,
                 cond_exponent: float = 2.0, design_seed: int = 0) -> DGP:
    Sigma_x = streams.illconditioned_cov(d, np.random.default_rng(10_000 + design_seed),
                                         cond_exponent)
    th = _theta_star(d)
    _, _, orc = streams.dgp_lin_hom(N=2, d=d, phi=phi, psi=psi, sigma_u=sigma_u,
                                    seed=0, theta_star=th, Sigma_x=Sigma_x)
    return _ARLike(name=f"lin_hom(phi={phi},psi={psi})", model="linear", d=d, oracle=orc,
                   params=dict(fn=streams.dgp_lin_hom,
                               kw=dict(d=d, phi=phi, psi=psi, sigma_u=sigma_u,
                                       theta_star=th, Sigma_x=Sigma_x)))


def make_lin_het(d: int = 20, phi_lo: float = 0.0, phi_hi: float = 0.9, psi: float = 0.7,
                 sigma_u: float = 1.0, cond_exponent: float = 2.0,
                 design_seed: int = 0) -> DGP:
    Sigma_eps = streams.illconditioned_cov(d, np.random.default_rng(20_000 + design_seed),
                                           cond_exponent)
    th = _theta_star(d)
    _, _, orc = streams.dgp_lin_het(N=2, d=d, phi_lo=phi_lo, phi_hi=phi_hi, psi=psi,
                                    sigma_u=sigma_u, seed=0, theta_star=th,
                                    Sigma_eps=Sigma_eps)
    return _ARLike(name=f"lin_het(psi={psi})", model="linear", d=d, oracle=orc,
                   params=dict(fn=streams.dgp_lin_het,
                               kw=dict(d=d, phi_lo=phi_lo, phi_hi=phi_hi, psi=psi,
                                       sigma_u=sigma_u, theta_star=th,
                                       Sigma_eps=Sigma_eps)))


def make_markov_cov(d: int = 20, K=None, stay: float = 0.97, psi: float = 0.8,
                    sigma_u: float = 1.0, noise_scale: float = 0.05,
                    cond_exponent: float = 2.0, design_seed: int = 0,
                    centres: str = "simplex") -> DGP:
    th = _theta_star(d)
    # the Markov DGP builds V and Sigma_nu from `seed`; freeze that by fixing seed=design
    _, _, orc = streams.dgp_markov_cov(N=2, d=d, K=K, stay=stay, psi=psi,
                                       sigma_u=sigma_u, noise_scale=noise_scale,
                                       cond_exponent=cond_exponent, seed=design_seed,
                                       theta_star=th, centres=centres)
    K = orc.meta["K"]

    class _MC(_ARLike):
        def sample(self, N, seed):
            X, y, _ = streams.dgp_markov_cov(
                N=N, d=d, K=K, stay=stay, psi=psi, sigma_u=sigma_u,
                noise_scale=noise_scale, cond_exponent=cond_exponent,
                seed=design_seed, theta_star=th, path_seed=seed, centres=centres)
            return X, y

    return _MC(name=f"markov_cov(K={K},stay={stay},psi={psi})", model="linear", d=d,
               oracle=orc, params=dict(fn=None, kw={}))


def make_logit_copula(d: int = 20, phi: float = 0.7, psi: float = 0.8,
                      cond_exponent: float = 1.0, design_seed: int = 0,
                      oracle: Optional[Oracle] = None, n_sim: int = 2_000_000,
                      n_rep: int = 6, lag: int = 150) -> DGP:
    Sigma_x = streams.illconditioned_cov(d, np.random.default_rng(30_000 + design_seed),
                                         cond_exponent)
    th = _theta_star(d)
    if oracle is None:
        oracle = streams.logit_copula_oracle(d, phi, psi, cond_exponent, th, Sigma_x,
                                             n_sim=n_sim, n_rep=n_rep, lag=lag)
    return _ARLike(name=f"logit_copula(phi={phi},psi={psi})", model="logistic", d=d,
                   oracle=oracle,
                   params=dict(fn=streams.dgp_logit_copula,
                               kw=dict(d=d, phi=phi, psi=psi, theta_star=th,
                                       Sigma_x=Sigma_x, oracle=oracle)))


BUILDERS = {
    "lin_hom": make_lin_hom,
    "lin_het": make_lin_het,
    "markov_cov": make_markov_cov,
    "logit_copula": make_logit_copula,
}
