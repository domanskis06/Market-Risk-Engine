"""Portfolio returns and losses from asset returns and weights."""

import numpy as np


def validate_weights(weights: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Ensure 1-D weights, non-negative, sum to 1. Return float array.
    Raise ValueError with a clear message otherwise.
    """
    if weights.ndim != 1:
        raise ValueError("Weights must be a 1-D array")
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")
    if np.abs(np.sum(weights) - 1) > tol:
        raise ValueError("Weights must sum to 1")
    return weights.astype(float)

def portfolio_returns(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """returns: (T, N), weights: (N,). Return R_p shape (T,)."""
    validate_weights(weights)
    return np.dot(returns, weights)

def portfolio_losses(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """L = -R_p."""
    validate_weights(weights)
    return -portfolio_returns(returns, weights)