"""
pricing.py — Option Pricing via Monte Carlo and Black–Scholes
=============================================================

Implements Monte Carlo estimators for European call and put options,
together with the closed-form Black–Scholes solution for benchmarking
and convergence analysis.

Key formulas
------------
Black–Scholes call price:
    C = S₀·N(d₁) − K·e^{−rT}·N(d₂)

where
    d₁ = [ln(S₀/K) + (r + σ²/2)·T] / (σ√T)
    d₂ = d₁ − σ√T

Monte Carlo estimator:
    Ĉ = e^{−rT} · (1/N) Σ max(S_T − K, 0)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from numpy.typing import NDArray


# ---------- Black–Scholes closed-form ----------

def black_scholes_call(
    S0: float, K: float, r: float, sigma: float, T: float
) -> float:
    """Compute Black–Scholes European call price."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def black_scholes_put(
    S0: float, K: float, r: float, sigma: float, T: float
) -> float:
    """Compute Black–Scholes European put price via put–call parity."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))


# ---------- Monte Carlo estimators ----------

def monte_carlo_european_call(
    terminal_prices: NDArray[np.float64],
    K: float,
    r: float,
    T: float,
) -> dict:
    """
    Price a European call option using Monte Carlo terminal prices.

    Parameters
    ----------
    terminal_prices : ndarray, shape (n_simulations,)
        Simulated stock prices at maturity.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    T : float
        Time to maturity.

    Returns
    -------
    dict with keys:
        price       — discounted Monte Carlo price estimate
        std_error   — standard error of the estimator
        payoffs     — array of discounted individual payoffs
    """
    payoffs = np.maximum(terminal_prices - K, 0.0)
    discount = np.exp(-r * T)
    discounted_payoffs = discount * payoffs
    price = float(np.mean(discounted_payoffs))
    std_error = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(len(payoffs)))
    return {"price": price, "std_error": std_error, "payoffs": discounted_payoffs}


def monte_carlo_european_put(
    terminal_prices: NDArray[np.float64],
    K: float,
    r: float,
    T: float,
) -> dict:
    """Price a European put option using Monte Carlo terminal prices."""
    payoffs = np.maximum(K - terminal_prices, 0.0)
    discount = np.exp(-r * T)
    discounted_payoffs = discount * payoffs
    price = float(np.mean(discounted_payoffs))
    std_error = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(len(payoffs)))
    return {"price": price, "std_error": std_error, "payoffs": discounted_payoffs}


# ---------- Convergence analysis ----------

def convergence_analysis(
    terminal_prices: NDArray[np.float64],
    K: float,
    r: float,
    T: float,
    option_type: str = "call",
    n_points: int = 200,
) -> dict:
    """
    Compute the running Monte Carlo estimate as N increases.

    Useful for visualising estimator convergence.

    Parameters
    ----------
    terminal_prices : ndarray
        Simulated terminal prices.
    K, r, T : float
        Option parameters.
    option_type : {"call", "put"}
        Type of European option.
    n_points : int
        Number of evaluation points along the simulation count axis.

    Returns
    -------
    dict with keys:
        n_sims        — array of simulation counts evaluated
        running_price — running mean of discounted payoffs
        upper_ci      — 95 % confidence interval upper bound
        lower_ci      — 95 % confidence interval lower bound
    """
    if option_type == "call":
        payoffs = np.maximum(terminal_prices - K, 0.0)
    else:
        payoffs = np.maximum(K - terminal_prices, 0.0)

    discount = np.exp(-r * T)
    discounted = discount * payoffs

    total = len(discounted)
    indices = np.unique(
        np.linspace(100, total, n_points, dtype=int)
    )

    running_price = np.array([np.mean(discounted[:n]) for n in indices])
    running_se = np.array(
        [np.std(discounted[:n], ddof=1) / np.sqrt(n) for n in indices]
    )

    return {
        "n_sims": indices,
        "running_price": running_price,
        "upper_ci": running_price + 1.96 * running_se,
        "lower_ci": running_price - 1.96 * running_se,
    }
