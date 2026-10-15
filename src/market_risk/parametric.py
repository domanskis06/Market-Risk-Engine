"""Gaussian parametric VaR and ES from portfolio mean and volatility."""

import numpy as np
from scipy.stats import norm

from market_risk.portfolio import validate_weights


def estimate_mean_cov(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """returns (T, N) → (mu (N,), cov (N, N)).
    Use np.mean(..., axis=0) and np.cov(..., rowvar=False).
    """
    mu = np.mean(returns, axis = 0)
    cov = np.cov(returns, rowvar = False)
    return mu, cov


def portfolio_mean_std(
    mu: np.ndarray, cov: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """Return (mu_p, sigma_p). Reuse validate_weights from portfolio."""
    validate_weights(weights)
    mu_p = np.dot(weights, mu)
    sigma_p = np.sqrt(weights @ cov @ weights)
    return mu_p, sigma_p


def parametric_var_gaussian(mu_p: float, sigma_p: float, alpha: float = 0.95) -> float:
    """VaR_alpha = -mu_p + sigma_p * norm.ppf(alpha). Validate alpha, sigma_p >= 0."""
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be in (0, 1)")
    if sigma_p < 0:
        raise ValueError("Sigma_p must be non-negative")
    return -mu_p + sigma_p * norm.ppf(alpha)

def parametric_es_gaussian(mu_p: float, sigma_p: float, alpha: float = 0.95) -> float:
    """ES_alpha = -mu_p + sigma_p * norm.pdf(norm.ppf(alpha))/(1-alpha). Validate alpha, sigma_p >= 0."""
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be in (0, 1)")
    if sigma_p < 0:
        raise ValueError("Sigma_p must be non-negative")
    return -mu_p + sigma_p * norm.pdf(norm.ppf(alpha))/(1-alpha)