"""Monte Carlo VaR and ES under a multivariate normal return model."""

import numpy as np

from market_risk.historical import historical_es, historical_var
from market_risk.portfolio import portfolio_losses, validate_weights


def simulate_multivariate_normal(
    mu: np.ndarray,
    cov: np.ndarray,
    n_sims: int,
    seed: int | None = None,
) -> np.ndarray:
    """Return array shape (n_sims, N). Use np.random.default_rng(seed).multivariate_normal."""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mu, cov, size=n_sims)


def monte_carlo_var_es(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.95,
    n_sims: int = 100_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Simulate → portfolio losses → (historical_var, historical_es) on those losses."""
    validate_weights(weights)
    losses = portfolio_losses(simulate_multivariate_normal(mu, cov, n_sims, seed), weights)
    var_mc = historical_var(losses, alpha)
    es_mc = historical_es(losses, alpha)
    return var_mc, es_mc