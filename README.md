# Monte Carlo Simulation for Option Pricing & Portfolio Risk

A quantitative finance project implementing Monte Carlo methods for European option pricing under Geometric Brownian Motion (GBM) and portfolio risk measurement via Value-at-Risk (VaR) and Expected Shortfall (ES). Includes an interactive Streamlit dashboard, modular Python code, and a research-style Jupyter notebook.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Financial Theory](#financial-theory)
3. [Methodology](#methodology)
4. [Repository Structure](#repository-structure)
5. [Results & Interpretation](#results--interpretation)
6. [Visualisations](#visualisations)
7. [Getting Started](#getting-started)
8. [Possible Extensions](#possible-extensions)
9. [References](#references)

---

## Project Overview

Monte Carlo simulation is a cornerstone technique in computational finance. It is particularly valuable when closed-form solutions are unavailable — for instance, when pricing path-dependent or multi-asset derivatives, or when modelling portfolio-level tail risk.

This project demonstrates:

- **Option pricing** — pricing European calls and puts by simulating risk-neutral asset paths, then benchmarking against the Black–Scholes closed-form solution.
- **Convergence analysis** — tracking how the Monte Carlo estimator converges to the analytical price as the number of simulations increases, including 95 % confidence bands.
- **Portfolio risk measurement** — computing Value-at-Risk and Expected Shortfall from simulated portfolio return distributions at multiple confidence levels.
- **Interactive exploration** — a Streamlit dashboard allowing users to adjust model parameters in real time and observe their impact on prices and risk metrics.

---

## Financial Theory

### Geometric Brownian Motion

Under the risk-neutral pricing framework, the dynamics of a non-dividend-paying stock price $S(t)$ are governed by:

$$dS = r \, S \, dt + \sigma \, S \, dW_t$$

where $r$ is the continuously compounded risk-free rate, $\sigma$ is the annualised volatility, and $W_t$ is a standard Brownian motion.

Applying Itô's lemma to $\ln S$ yields the exact solution:

$$S(t + \Delta t) = S(t) \cdot \exp\!\Big[\Big(r - \tfrac{1}{2}\sigma^2\Big)\Delta t + \sigma\sqrt{\Delta t}\;Z\Big], \quad Z \sim \mathcal{N}(0,1)$$

This log-normal structure guarantees $S > 0$ and permits efficient vectorised simulation.

### Monte Carlo Option Pricing

The risk-neutral pricing formula for a European call with strike $K$ and maturity $T$ is:

$$C = e^{-rT}\;\mathbb{E}^{\mathbb{Q}}\big[\max(S_T - K,\; 0)\big]$$

The Monte Carlo estimator replaces the expectation with a sample mean over $N$ simulated terminal prices:

$$\hat{C} = e^{-rT}\;\frac{1}{N}\sum_{i=1}^{N}\max\!\big(S_T^{(i)} - K,\; 0\big)$$

By the Central Limit Theorem, $\hat{C}$ converges to $C$ at rate $\mathcal{O}(1/\sqrt{N})$, and the standard error of the estimate is:

$$\text{SE} = \frac{\hat{\sigma}_{\text{payoff}}}{\sqrt{N}}$$

### Black–Scholes Benchmark

The closed-form Black–Scholes call price is:

$$C = S_0 \, \Phi(d_1) - K \, e^{-rT} \, \Phi(d_2)$$

$$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)\,T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

This serves as a ground-truth reference for validating the Monte Carlo engine.

### Value-at-Risk and Expected Shortfall

**Value-at-Risk** at confidence level $\alpha$ is defined as:

$$\text{VaR}_\alpha = -Q_\alpha(R)$$

where $Q_\alpha$ is the $\alpha$-quantile of the portfolio return distribution.

**Expected Shortfall** (also called Conditional VaR) captures the average loss in the tail beyond VaR:

$$\text{ES}_\alpha = -\mathbb{E}[R \mid R \leq Q_\alpha(R)]$$

ES is a *coherent* risk measure — in particular it is sub-additive, making it suitable for portfolio aggregation. Both metrics are computed here via full Monte Carlo simulation of the portfolio return distribution.

---

## Methodology

1. **Path generation** — Simulate $N$ independent GBM paths over $T$ years with $M$ time steps using the exact log-normal discretisation.
2. **Option pricing** — Extract terminal prices $S_T^{(i)}$, compute discounted payoffs, and average to obtain $\hat{C}$. Compare with Black–Scholes.
3. **Convergence** — Track the running mean estimate and 95 % confidence interval as $N$ grows from 100 to the full simulation count.
4. **Risk metrics** — Convert terminal prices to portfolio returns, then compute historical VaR and ES at 90 %, 95 %, and 99 % confidence.
5. **Correlated assets** — Extend to multi-asset simulation via Cholesky decomposition of the correlation matrix.

---

## Repository Structure

```
mc-option-pricing/
│
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
├── streamlit_app.py                   ← Interactive dashboard
│
├── notebooks/
│   └── monte_carlo_simulation.ipynb   ← Full research notebook
│
├── src/
│   ├── __init__.py
│   ├── simulation.py                  ← GBM path generation (single & multi-asset)
│   ├── pricing.py                     ← MC + Black–Scholes option pricing
│   └── risk_metrics.py                ← VaR, Expected Shortfall, risk summaries
│
├── plots/                             ← Generated visualisations
│   ├── price_paths.png
│   ├── return_distribution.png
│   ├── convergence.png
│   └── var_visualisation.png
│
├── data/                              ← (placeholder for market data)
└── results/                           ← (placeholder for exported tables)
```

---

## Results & Interpretation

With default parameters ($S_0 = 100$, $K = 105$, $r = 5\%$, $\sigma = 20\%$, $T = 1$ year, $N = 100{,}000$ paths):

| Metric | Value |
|---|---|
| Black–Scholes call price | ≈ $8.02 |
| Monte Carlo call price | ≈ $8.02 ± $0.05 |
| 95 % VaR (1-year) | ≈ 26–28 % |
| 95 % ES (1-year) | ≈ 32–35 % |

The Monte Carlo estimate converges within ±$0.10 of the analytical price after roughly 10,000 simulations, confirming the $\mathcal{O}(1/\sqrt{N})$ convergence rate. Increasing volatility widens both the option price and the portfolio risk tail, consistent with financial intuition.

---

## Visualisations

The notebook and Streamlit app generate four primary plots:

1. **Simulated price paths** — a sample of GBM trajectories with the mean path overlaid, illustrating the dispersion driven by $\sigma$.
2. **Return distribution** — histogram of simulated portfolio returns with VaR and ES thresholds marked.
3. **Convergence plot** — running Monte Carlo estimate versus the Black–Scholes benchmark, with 95 % confidence bands shrinking as $N$ grows.
4. **VaR confidence level curve** — VaR as a function of the confidence level from 80 % to 99.5 %, showing how tail risk escalates at extreme quantiles.

---

## Getting Started

### Prerequisites

Python 3.10 or later.

### Installation

```bash
git clone https://github.com/<your-username>/mc-option-pricing.git
cd mc-option-pricing
pip install -r requirements.txt
```

### Run the Jupyter Notebook

```bash
jupyter notebook notebooks/monte_carlo_simulation.ipynb
```

### Launch the Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

The app opens in your browser and lets you adjust volatility, interest rate, time horizon, and simulation count in real time.

---

## Possible Extensions

- **Variance reduction** — antithetic variates, control variates, importance sampling.
- **Exotic options** — Asian, barrier, and lookback options using the full simulated paths.
- **Stochastic volatility** — Heston model with mean-reverting variance.
- **Jump-diffusion** — Merton model adding Poisson-distributed jumps.
- **Greeks computation** — finite-difference and pathwise sensitivity estimates.
- **Multi-asset portfolio** — correlated GBM with portfolio-level VaR aggregation.
- **Real market data** — calibrate $\sigma$ and $r$ from historical prices and the yield curve.

---

## References

- Hull, J. C. (2022). *Options, Futures, and Other Derivatives*. 11th ed. Pearson.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
- Black, F. & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637–654.
- Artzner, P. et al. (1999). "Coherent Measures of Risk." *Mathematical Finance*, 9(3), 203–228.

---

*Built as a quantitative finance portfolio project. Feedback and contributions welcome.*
