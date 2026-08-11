"""Statistical models: loss, score (gradient), and the rank-one curvature form.

Every model exposes the *curvature form*

    grad^2 F(theta) = E[ alpha(theta; x, y) * Phi(theta; x, y) Phi(theta; x, y)^T ],

which is what makes an inversion-free (Sherman--Morrison / Riccati) Hessian-inverse
recursion possible.  For generalised linear models Phi = x and alpha is a scalar
depending on the linear predictor, so a curvature update costs O(d^2) rather than
O(d^3).

Conventions
-----------
* ``score(theta, X, y)`` returns the *per-observation* scores as an (n, d) array,
  i.e. row i is grad_theta f(theta; x_i, y_i).  The empirical risk gradient is the
  row mean.
* ``alpha(theta, X, y)`` returns the (n,) vector of curvature scalars.
* ``Phi`` is always ``X`` for the GLMs used here; we keep the method for clarity
  and to make the interface explicit.
"""

from __future__ import annotations

import numpy as np


class Model:
    """Abstract base class for an M-estimation problem."""

    name = "abstract"

    def score(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score_scale(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return c such that score_i = c_i * x_i (available for all GLMs here)."""
        raise NotImplementedError

    def alpha(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def Phi(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return X

    def mean_grad(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Mean score over the rows of (X, y).  O(nd) work, no (n, d) allocation."""
        return X.T @ self.score_scale(theta, X, y) / X.shape[0]

    def loss(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError


class LinearRegression(Model):
    r"""Least squares: :math:`F(\theta) = \tfrac12 E[(y - x^\top\theta)^2]`.

    score_i = -(y_i - x_i^T theta) x_i,  alpha_i = 1,  Phi_i = x_i,
    so grad^2 F = E[x x^T].
    """

    name = "linear"

    def score_scale(self, theta, X, y):
        return X @ theta - y

    def score(self, theta, X, y):
        return self.score_scale(theta, X, y)[:, None] * X

    def alpha(self, theta, X, y):
        return np.ones(X.shape[0])

    def loss(self, theta, X, y):
        r = y - X @ theta
        return 0.5 * float(np.mean(r * r))


class LogisticRegression(Model):
    r"""Logistic regression with labels in :math:`\{0, 1\}`.

    :math:`F(\theta) = E[\log(1 + e^{x^\top\theta}) - y\, x^\top\theta]`,
    score_i = (sigma(x_i^T theta) - y_i) x_i, alpha_i = sigma (1 - sigma),
    so grad^2 F = E[sigma(1-sigma) x x^T].
    """

    name = "logistic"

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable logistic function.
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def score_scale(self, theta, X, y):
        return self._sigmoid(X @ theta) - y

    def score(self, theta, X, y):
        return self.score_scale(theta, X, y)[:, None] * X

    def alpha(self, theta, X, y):
        p = self._sigmoid(X @ theta)
        return p * (1.0 - p)

    def loss(self, theta, X, y):
        z = X @ theta
        # log(1 + e^z) computed stably
        ll = np.logaddexp(0.0, z) - y * z
        return float(np.mean(ll))


MODELS = {"linear": LinearRegression, "logistic": LogisticRegression}


def get_model(name: str) -> Model:
    return MODELS[name]()
