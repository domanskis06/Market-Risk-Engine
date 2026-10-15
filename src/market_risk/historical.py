"""Empirical (historical) VaR and Expected Shortfall."""

import numpy as np


def historical_var(losses: np.ndarray, alpha: float = 0.95) -> float:
    """Empirical VaR: alpha-quantile of losses. losses 1-D, alpha in (0, 1)."""
    if losses.ndim != 1:
        raise ValueError("Losses must be a 1-D array")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be in (0, 1)")
    return np.quantile(losses, alpha)


def historical_es(losses: np.ndarray, alpha: float = 0.95) -> float:
    """ES = mean of losses that are >= historical_var(losses, alpha)."""
    if losses.ndim != 1:
        raise ValueError("Losses must be a 1-D array")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be in (0, 1)")
    var = historical_var(losses, alpha)
    return np.mean(losses[losses >= var])