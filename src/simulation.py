"""
simulation.py — Monte Carlo Path Generation Engine
====================================================

Generates simulated asset price paths under Geometric Brownian Motion (GBM)
using vectorised NumPy operations. Supports both single-asset and
multi-asset (correlated) simulations.

Mathematical foundation
-----------------------
Under the risk-neutral measure, a stock price S(t) evolves as:

    dS = r·S·dt + σ·S·dW

where r is the risk-free rate, σ is volatility, and W is a standard
Brownian motion. The exact solution is:

    S(t+dt) = S(t) · exp[(r - 0.5·σ²)·dt + σ·√dt·Z]

with Z ~ N(0,1).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_simulations: int,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """
    Simulate asset price paths under Geometric Brownian Motion.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    r : float
        Annualised risk-free interest rate (continuous compounding).
    sigma : float
        Annualised volatility.
    T : float
        Time horizon in years.
    n_steps : int
        Number of discrete time steps per path.
    n_simulations : int
        Number of independent Monte Carlo paths.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    paths : ndarray, shape (n_simulations, n_steps + 1)
        Simulated price paths. Column 0 is S0 for every path.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # Pre-compute drift and diffusion per step
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Draw all random increments at once: shape (n_simulations, n_steps)
    Z = rng.standard_normal((n_simulations, n_steps))

    # Log-returns for each step
    log_returns = drift + diffusion * Z

    # Cumulative sum of log-returns, then exponentiate
    log_paths = np.cumsum(log_returns, axis=1)
    paths = np.column_stack([np.zeros(n_simulations), log_paths])
    paths = S0 * np.exp(paths)

    return paths


def simulate_correlated_gbm(
    S0: NDArray[np.float64],
    r: float,
    sigmas: NDArray[np.float64],
    corr_matrix: NDArray[np.float64],
    T: float,
    n_steps: int,
    n_simulations: int,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """
    Simulate correlated multi-asset GBM paths via Cholesky decomposition.

    Parameters
    ----------
    S0 : array-like, shape (n_assets,)
        Initial prices for each asset.
    r : float
        Risk-free rate.
    sigmas : array-like, shape (n_assets,)
        Per-asset annualised volatilities.
    corr_matrix : ndarray, shape (n_assets, n_assets)
        Correlation matrix (must be positive semi-definite).
    T : float
        Time horizon in years.
    n_steps : int
        Number of time steps.
    n_simulations : int
        Number of Monte Carlo paths.
    seed : int or None
        Random seed.

    Returns
    -------
    paths : ndarray, shape (n_assets, n_simulations, n_steps + 1)
        Simulated price paths for each asset.
    """
    S0 = np.asarray(S0, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    corr_matrix = np.asarray(corr_matrix, dtype=np.float64)
    n_assets = len(S0)
    dt = T / n_steps

    rng = np.random.default_rng(seed)

    # Cholesky factorisation of the correlation matrix
    L = np.linalg.cholesky(corr_matrix)

    # Draw independent normals: (n_assets, n_simulations, n_steps)
    Z_independent = rng.standard_normal((n_assets, n_simulations, n_steps))

    # Correlate the draws: apply L across the asset dimension
    # Reshape for matmul: (n_simulations * n_steps, n_assets) @ L^T
    Z_flat = Z_independent.reshape(n_assets, -1).T  # (N*T, n_assets)
    Z_corr_flat = (L @ Z_flat.T).T  # (N*T, n_assets)
    Z_corr = Z_corr_flat.T.reshape(n_assets, n_simulations, n_steps)

    # Build paths per asset
    paths = np.zeros((n_assets, n_simulations, n_steps + 1))
    for i in range(n_assets):
        drift = (r - 0.5 * sigmas[i] ** 2) * dt
        diffusion = sigmas[i] * np.sqrt(dt)
        log_returns = drift + diffusion * Z_corr[i]
        log_paths = np.cumsum(log_returns, axis=1)
        log_paths = np.column_stack(
            [np.zeros(n_simulations), log_paths]
        )
        paths[i] = S0[i] * np.exp(log_paths)

    return paths


def generate_time_grid(T: float, n_steps: int) -> NDArray[np.float64]:
    """Return an evenly spaced time grid [0, dt, 2·dt, ..., T]."""
    return np.linspace(0.0, T, n_steps + 1)
