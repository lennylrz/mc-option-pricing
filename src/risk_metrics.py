"""
risk_metrics.py — Value-at-Risk and Expected Shortfall
======================================================

Implements parametric and historical simulation approaches for
portfolio risk measurement.

Definitions
-----------
Value-at-Risk (VaR) at confidence level α:
    VaR_α = −Q_α(R)

where Q_α is the α-quantile of the portfolio return distribution R.

Expected Shortfall (Conditional VaR) at confidence level α:
    ES_α = −E[R | R ≤ Q_α(R)]

ES is a *coherent* risk measure (sub-additive), unlike VaR.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from numpy.typing import NDArray


# ---------- Historical / Monte Carlo VaR ----------

def historical_var(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
) -> float:
    """
    Compute historical Value-at-Risk.

    Parameters
    ----------
    returns : ndarray
        Array of portfolio returns (can be log or simple returns).
    confidence : float
        Confidence level, e.g. 0.95 for 95 % VaR.

    Returns
    -------
    float
        VaR expressed as a positive loss amount.
    """
    alpha = 1 - confidence
    return float(-np.percentile(returns, 100 * alpha))


def historical_expected_shortfall(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
) -> float:
    """
    Compute historical Expected Shortfall (CVaR).

    Parameters
    ----------
    returns : ndarray
        Portfolio returns.
    confidence : float
        Confidence level.

    Returns
    -------
    float
        Expected Shortfall as a positive loss amount.
    """
    alpha = 1 - confidence
    cutoff = np.percentile(returns, 100 * alpha)
    tail = returns[returns <= cutoff]
    if len(tail) == 0:
        return float(-cutoff)
    return float(-np.mean(tail))


# ---------- Parametric (Gaussian) VaR ----------

def parametric_var(
    mu: float,
    sigma: float,
    confidence: float = 0.95,
    portfolio_value: float = 1.0,
) -> float:
    """
    Gaussian parametric VaR.

    Parameters
    ----------
    mu : float
        Expected return of the portfolio.
    sigma : float
        Standard deviation of portfolio returns.
    confidence : float
        Confidence level.
    portfolio_value : float
        Notional portfolio value (scales the result).

    Returns
    -------
    float
        VaR as a positive loss amount.
    """
    z = norm.ppf(1 - confidence)  # negative quantile
    return float(-portfolio_value * (mu + z * sigma))


def parametric_expected_shortfall(
    mu: float,
    sigma: float,
    confidence: float = 0.95,
    portfolio_value: float = 1.0,
) -> float:
    """Gaussian parametric Expected Shortfall."""
    alpha = 1 - confidence
    z_alpha = norm.ppf(alpha)
    es_return = mu - sigma * norm.pdf(z_alpha) / alpha
    return float(-portfolio_value * es_return)


# ---------- Portfolio simulation ----------

def simulate_portfolio_returns(
    terminal_prices: NDArray[np.float64],
    S0: float,
) -> NDArray[np.float64]:
    """
    Compute simple returns from simulated terminal prices.

    Parameters
    ----------
    terminal_prices : ndarray
        Simulated stock prices at horizon.
    S0 : float
        Initial stock price.

    Returns
    -------
    ndarray
        Simple returns: (S_T - S_0) / S_0
    """
    return (terminal_prices - S0) / S0


def compute_risk_summary(
    returns: NDArray[np.float64],
    portfolio_value: float = 1_000_000.0,
    confidence_levels: list[float] | None = None,
) -> dict:
    """
    Compute a full risk summary table for a set of simulated returns.

    Parameters
    ----------
    returns : ndarray
        Simulated portfolio returns.
    portfolio_value : float
        Notional value of the portfolio.
    confidence_levels : list of float
        Confidence levels to evaluate.

    Returns
    -------
    dict
        Nested dict keyed by confidence level with VaR/ES in both
        percentage and dollar terms.
    """
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.99]

    summary: dict = {}
    for cl in confidence_levels:
        var_pct = historical_var(returns, cl)
        es_pct = historical_expected_shortfall(returns, cl)
        summary[cl] = {
            "VaR_pct": var_pct,
            "ES_pct": es_pct,
            "VaR_dollar": var_pct * portfolio_value,
            "ES_dollar": es_pct * portfolio_value,
        }
    return summary
