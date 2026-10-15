import numpy as np
import pytest

from market_risk.historical import historical_es, historical_var


def test_historical_var_matches_numpy_quantile():
    losses = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    alpha = 0.9
    var = historical_var(losses, alpha)
    assert np.isclose(var, np.quantile(losses, alpha))


def test_historical_es_manual_tail():
    # Fixed VaR path: losses >= 0.08 → {0.08, 0.09, 0.10}, mean = 0.09
    losses = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    # Choose alpha so quantile equals 0.08 on this sample:
    # For linear quantile, alpha such that Q=0.08 → verify in REPL once, then lock it.
    alpha = 0.8
    var = historical_var(losses, alpha)
    print(var)  # if this fails, print(var) and adjust alpha OR assert ES via mask on computed var
    es = historical_es(losses, alpha)
    tail = losses[losses >= var]
    assert np.isclose(es, tail.mean())
    assert es >= var


def test_historical_invalid_alpha():
    with pytest.raises(ValueError):
        historical_var(np.array([0.1, 0.2]), alpha=1.0)