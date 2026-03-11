"""
streamlit_app.py — Interactive Monte Carlo Option Pricing Dashboard
===================================================================

Launch with:  streamlit run streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.simulation import simulate_gbm_paths, generate_time_grid
from src.pricing import (
    black_scholes_call,
    black_scholes_put,
    monte_carlo_european_call,
    monte_carlo_european_put,
    convergence_analysis,
)
from src.risk_metrics import (
    simulate_portfolio_returns,
    historical_var,
    historical_expected_shortfall,
    compute_risk_summary,
)

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="MC Option Pricing & Risk",
    page_icon="📈",
    layout="wide",
)

st.title("Monte Carlo Option Pricing & Portfolio Risk")
st.markdown("*By Lenny Lorenz - L'Économique*")
st.markdown(
    "Adjust the parameters in the sidebar, then click **Run Simulation** "
    "to price European options and compute portfolio risk metrics."
)

# ── Sidebar inputs ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Parameters")
    S0 = st.number_input("Initial stock price (S₀)", value=100.0, min_value=1.0, step=5.0)
    K = st.number_input("Strike price (K)", value=105.0, min_value=1.0, step=5.0)
    r = st.slider("Risk-free rate (r)", 0.00, 0.15, 0.05, 0.005, format="%.3f")
    sigma = st.slider("Volatility (σ)", 0.05, 1.00, 0.20, 0.01, format="%.2f")
    T = st.slider("Time horizon (years)", 0.1, 5.0, 1.0, 0.1)

    st.header("Simulation Settings")
    n_simulations = st.select_slider(
        "Number of simulations",
        options=[1_000, 5_000, 10_000, 50_000, 100_000, 500_000],
        value=50_000,
    )
    n_steps = st.select_slider(
        "Time steps per path",
        options=[50, 100, 252, 504],
        value=252,
    )
    seed = st.number_input("Random seed", value=42, step=1)

    st.header("Portfolio")
    portfolio_value = st.number_input(
        "Portfolio notional ($)", value=1_000_000, step=100_000
    )

    run = st.button("🚀 Run Simulation", width="stretch")

# ── Main panel ───────────────────────────────────────────────────────
if run:
    with st.spinner("Simulating paths …"):
        paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_simulations, seed=int(seed))
        t_grid = generate_time_grid(T, n_steps)
        terminal = paths[:, -1]

    # ── Option pricing ───────────────────────────────────────────────
    bs_call = black_scholes_call(S0, K, r, sigma, T)
    bs_put = black_scholes_put(S0, K, r, sigma, T)
    mc_call = monte_carlo_european_call(terminal, K, r, T)
    mc_put = monte_carlo_european_put(terminal, K, r, T)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BS Call", f"${bs_call:.4f}")
    col2.metric("MC Call", f"${mc_call['price']:.4f}", delta=f"SE {mc_call['std_error']:.4f}")
    col3.metric("BS Put", f"${bs_put:.4f}")
    col4.metric("MC Put", f"${mc_put['price']:.4f}", delta=f"SE {mc_put['std_error']:.4f}")

    # ── Tab layout ───────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Price Paths", "Convergence", "Return Distribution", "VaR Curve"]
    )

    # ── 1. Price paths ───────────────────────────────────────────────
    with tab1:
        n_show = min(200, n_simulations)
        fig1 = go.Figure()
        for i in range(n_show):
            fig1.add_trace(go.Scatter(
                x=t_grid, y=paths[i], mode="lines",
                line=dict(width=0.4, color="rgba(70,130,180,0.15)"),
                showlegend=False,
            ))
        mean_path = paths.mean(axis=0)
        fig1.add_trace(go.Scatter(
            x=t_grid, y=mean_path, mode="lines",
            line=dict(width=2.5, color="#E74C3C"),
            name="Mean path",
        ))
        fig1.update_layout(
            title="Simulated GBM Price Paths",
            xaxis_title="Time (years)", yaxis_title="Stock Price ($)",
            template="plotly_white", height=480,
        )
        st.plotly_chart(fig1, width="stretch")

    # ── 2. Convergence ───────────────────────────────────────────────
    with tab2:
        conv = convergence_analysis(terminal, K, r, T, "call", n_points=300)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=conv["n_sims"], y=conv["running_price"],
            mode="lines", name="MC estimate",
            line=dict(color="#2980B9", width=2),
        ))
        fig2.add_trace(go.Scatter(
            x=conv["n_sims"], y=conv["upper_ci"],
            mode="lines", name="95 % CI",
            line=dict(color="#AED6F1", width=1, dash="dash"),
        ))
        fig2.add_trace(go.Scatter(
            x=conv["n_sims"], y=conv["lower_ci"],
            mode="lines", showlegend=False,
            line=dict(color="#AED6F1", width=1, dash="dash"),
            fill="tonexty", fillcolor="rgba(174,214,241,0.2)",
        ))
        fig2.add_hline(y=bs_call, line_dash="dot", line_color="#E74C3C",
                       annotation_text=f"BS = ${bs_call:.4f}")
        fig2.update_layout(
            title="Monte Carlo Convergence (European Call)",
            xaxis_title="Number of simulations",
            yaxis_title="Option price ($)",
            template="plotly_white", height=480,
        )
        st.plotly_chart(fig2, width="stretch")

    # ── 3. Return distribution ───────────────────────────────────────
    with tab3:
        returns = simulate_portfolio_returns(terminal, S0)
        var95 = historical_var(returns, 0.95)
        es95 = historical_expected_shortfall(returns, 0.95)

        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=returns, nbinsx=200, name="Returns",
            marker_color="rgba(41,128,185,0.6)",
        ))
        fig3.add_vline(x=-var95, line_color="#E74C3C", line_width=2,
                       annotation_text=f"VaR 95 % = {var95:.2%}")
        fig3.add_vline(x=-es95, line_color="#8E44AD", line_width=2, line_dash="dash",
                       annotation_text=f"ES 95 % = {es95:.2%}")
        fig3.update_layout(
            title="Distribution of Simulated Portfolio Returns",
            xaxis_title="Return", yaxis_title="Frequency",
            template="plotly_white", height=480,
        )
        st.plotly_chart(fig3, width="stretch")

        # Risk summary table
        summary = compute_risk_summary(returns, portfolio_value, [0.90, 0.95, 0.99])
        rows = []
        for cl, v in summary.items():
            rows.append({
                "Confidence": f"{cl:.0%}",
                "VaR (%)": f"{v['VaR_pct']:.2%}",
                f"VaR ($)": f"${v['VaR_dollar']:,.0f}",
                "ES (%)": f"{v['ES_pct']:.2%}",
                f"ES ($)": f"${v['ES_dollar']:,.0f}",
            })
        import pandas as pd
        st.subheader("Risk Summary")
        st.table(pd.DataFrame(rows))

    # ── 4. VaR confidence curve ──────────────────────────────────────
    with tab4:
        cls = np.linspace(0.80, 0.995, 100)
        vars_curve = [historical_var(returns, c) for c in cls]
        es_curve = [historical_expected_shortfall(returns, c) for c in cls]

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=cls, y=vars_curve, mode="lines", name="VaR",
            line=dict(color="#E74C3C", width=2),
        ))
        fig4.add_trace(go.Scatter(
            x=cls, y=es_curve, mode="lines", name="Expected Shortfall",
            line=dict(color="#8E44AD", width=2, dash="dash"),
        ))
        fig4.update_layout(
            title="VaR & ES as a Function of Confidence Level",
            xaxis_title="Confidence level",
            yaxis_title="Loss (as fraction of portfolio)",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".1%",
            template="plotly_white", height=480,
        )
        st.plotly_chart(fig4, width="stretch")

else:
    st.info("👈 Set your parameters in the sidebar and click **Run Simulation**.")
