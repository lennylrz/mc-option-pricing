"""
streamlit_app.py — Interactive Monte Carlo Option Pricing Dashboard
===================================================================

Launch with:  streamlit run streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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

# ── Brand colours ────────────────────────────────────────────────
PRIMARY_GREEN  = "#163306"
DARK_FOREST    = "#1F4A0A"
MUTED_GREEN    = "#2D6A1A"
SOFT_SAGE      = "#A8C5A0"
PRESTIGE_GOLD  = "#B8962E"
ACADEMIC_BLACK = "#1A1A1A"
PAPER_WHITE    = "#F7F5F0"
LIGHT_GREY     = "#E8E5DF"

# ── Plotly template matching the brand ───────────────────────────
BRAND_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=PAPER_WHITE,
        plot_bgcolor=PAPER_WHITE,
        font=dict(family="Georgia, Times New Roman, serif", color=ACADEMIC_BLACK, size=13),
        title=dict(font=dict(family="Georgia, Times New Roman, serif", size=20, color=PRIMARY_GREEN)),
        xaxis=dict(
            gridcolor=LIGHT_GREY, linecolor=LIGHT_GREY,
            title_font=dict(color=ACADEMIC_BLACK),
        ),
        yaxis=dict(
            gridcolor=LIGHT_GREY, linecolor=LIGHT_GREY,
            title_font=dict(color=ACADEMIC_BLACK),
        ),
        colorway=[MUTED_GREEN, PRESTIGE_GOLD, DARK_FOREST, SOFT_SAGE, PRIMARY_GREEN],
    )
)

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="MC Option Pricing — L'Économique",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS for brand styling ─────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,700;1,400&display=swap');

    /* Global background */
    .stApp {{
        background-color: {PAPER_WHITE};
    }}

    /* Main text */
    .stApp, .stMarkdown, p, li, span {{
        font-family: 'EB Garamond', Georgia, 'Times New Roman', serif !important;
        color: {ACADEMIC_BLACK};
    }}

    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'EB Garamond', Georgia, serif !important;
        color: {PRIMARY_GREEN} !important;
        font-weight: 700 !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {PRIMARY_GREEN} !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {PAPER_WHITE} !important;
        font-family: 'EB Garamond', Georgia, serif !important;
    }}
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stSelectSlider label {{
        color: {SOFT_SAGE} !important;
    }}

    /* Metrics */
    [data-testid="stMetric"] {{
        background-color: {LIGHT_GREY};
        border-left: 4px solid {PRESTIGE_GOLD};
        padding: 12px 16px;
        border-radius: 2px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {DARK_FOREST} !important;
        font-family: 'EB Garamond', Georgia, serif !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {PRIMARY_GREEN} !important;
        font-family: 'EB Garamond', Georgia, serif !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab"] {{
        font-family: 'EB Garamond', Georgia, serif !important;
        color: {ACADEMIC_BLACK} !important;
        font-size: 15px;
    }}
    .stTabs [aria-selected="true"] {{
        border-bottom-color: {PRESTIGE_GOLD} !important;
        color: {PRIMARY_GREEN} !important;
        font-weight: 700 !important;
    }}

    /* Table */
    .stTable, table {{
        font-family: 'EB Garamond', Georgia, serif !important;
    }}
    thead tr th {{
        background-color: {PRIMARY_GREEN} !important;
        color: {PAPER_WHITE} !important;
    }}

    /* Button */
    .stButton > button {{
        background-color: {PRESTIGE_GOLD} !important;
        color: {PAPER_WHITE} !important;
        font-family: 'EB Garamond', Georgia, serif !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 2px !important;
        letter-spacing: 0.5px;
    }}
    .stButton > button:hover {{
        background-color: {DARK_FOREST} !important;
    }}

    /* Divider line */
    .gold-divider {{
        border: none;
        border-top: 2px solid {PRESTIGE_GOLD};
        margin: 0.5rem 0 1.5rem 0;
    }}

    /* Subtitle styling */
    .subtitle {{
        font-family: 'EB Garamond', Georgia, serif;
        font-style: italic;
        color: {MUTED_GREEN};
        font-size: 1.1rem;
        margin-top: -0.5rem;
    }}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────
st.title("Monte Carlo Option Pricing & Portfolio Risk")
st.markdown('<p class="subtitle">By Lenny Lorenz, President of L\'Économique</p>', unsafe_allow_html=True)
st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
st.markdown(
    "Adjust the parameters in the sidebar, then click **Run Simulation** "
    "to price European options and compute portfolio risk metrics."
)

# ── Sidebar inputs ───────────────────────────────────────────────
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

    run = st.button("Run Simulation")

# ── Main panel ───────────────────────────────────────────────────
if run:
    with st.spinner("Simulating paths …"):
        paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_simulations, seed=int(seed))
        t_grid = generate_time_grid(T, n_steps)
        terminal = paths[:, -1]

    # ── Option pricing ───────────────────────────────────────────
    bs_call = black_scholes_call(S0, K, r, sigma, T)
    bs_put = black_scholes_put(S0, K, r, sigma, T)
    mc_call = monte_carlo_european_call(terminal, K, r, T)
    mc_put = monte_carlo_european_put(terminal, K, r, T)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BS Call", f"${bs_call:.4f}")
    col2.metric("MC Call", f"${mc_call['price']:.4f}", delta=f"SE {mc_call['std_error']:.4f}")
    col3.metric("BS Put", f"${bs_put:.4f}")
    col4.metric("MC Put", f"${mc_put['price']:.4f}", delta=f"SE {mc_put['std_error']:.4f}")

    # ── Tab layout ───────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Price Paths", "Convergence", "Return Distribution", "VaR Curve"]
    )

    # ── 1. Price paths ───────────────────────────────────────────
    with tab1:
        n_show = min(200, n_simulations)
        fig1 = go.Figure()
        for i in range(n_show):
            fig1.add_trace(go.Scatter(
                x=t_grid, y=paths[i], mode="lines",
                line=dict(width=0.4, color="rgba(45,106,26,0.12)"),
                showlegend=False,
            ))
        mean_path = paths.mean(axis=0)
        fig1.add_trace(go.Scatter(
            x=t_grid, y=mean_path, mode="lines",
            line=dict(width=2.5, color=PRESTIGE_GOLD),
            name="Mean path",
        ))
        fig1.update_layout(
            title="Simulated GBM Price Paths",
            xaxis_title="Time (years)", yaxis_title="Stock Price ($)",
            template=BRAND_TEMPLATE, height=500,
            legend=dict(font=dict(size=13)),
        )
        st.plotly_chart(fig1, width="stretch")

    # ── 2. Convergence ───────────────────────────────────────────
    with tab2:
        conv = convergence_analysis(terminal, K, r, T, "call", n_points=300)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=conv["n_sims"], y=conv["running_price"],
            mode="lines", name="MC estimate",
            line=dict(color=DARK_FOREST, width=2),
        ))
        fig2.add_trace(go.Scatter(
            x=conv["n_sims"], y=conv["upper_ci"],
            mode="lines", name="95 % CI",
            line=dict(color=SOFT_SAGE, width=1, dash="dash"),
        ))
        fig2.add_trace(go.Scatter(
            x=conv["n_sims"], y=conv["lower_ci"],
            mode="lines", showlegend=False,
            line=dict(color=SOFT_SAGE, width=1, dash="dash"),
            fill="tonexty", fillcolor="rgba(168,197,160,0.2)",
        ))
        fig2.add_hline(y=bs_call, line_dash="dot", line_color=PRESTIGE_GOLD,
                       annotation_text=f"BS = ${bs_call:.4f}",
                       annotation_font=dict(color=PRESTIGE_GOLD, size=13))
        fig2.update_layout(
            title="Monte Carlo Convergence (European Call)",
            xaxis_title="Number of simulations",
            yaxis_title="Option price ($)",
            template=BRAND_TEMPLATE, height=500,
        )
        st.plotly_chart(fig2, width="stretch")

    # ── 3. Return distribution ───────────────────────────────────
    with tab3:
        returns = simulate_portfolio_returns(terminal, S0)
        var95 = historical_var(returns, 0.95)
        es95 = historical_expected_shortfall(returns, 0.95)

        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=returns, nbinsx=200, name="Returns",
            marker_color=MUTED_GREEN, opacity=0.7,
        ))
        fig3.add_vline(x=-var95, line_color=PRESTIGE_GOLD, line_width=2.5)
        fig3.add_vline(x=-es95, line_color=PRIMARY_GREEN, line_width=2, line_dash="dash")
        fig3.add_annotation(
            x=-var95, y=1.06, yref="paper", showarrow=False,
            text=f"<b>VaR 95% = {var95:.2%}</b>",
            font=dict(color=PRESTIGE_GOLD, size=14, family="Georgia, serif"),
            xanchor="left", xshift=8,
        )
        fig3.add_annotation(
            x=-es95, y=0.94, yref="paper", showarrow=False,
            text=f"<b>ES 95% = {es95:.2%}</b>",
            font=dict(color=PRIMARY_GREEN, size=14, family="Georgia, serif"),
            xanchor="right", xshift=-8,
        )
        fig3.update_layout(
            title="Distribution of Simulated Portfolio Returns",
            xaxis_title="Return", yaxis_title="Frequency",
            template=BRAND_TEMPLATE, height=500,
        )
        st.plotly_chart(fig3, width="stretch")

        # Risk summary table
        summary = compute_risk_summary(returns, portfolio_value, [0.90, 0.95, 0.99])
        rows = []
        for cl, v in summary.items():
            rows.append({
                "Confidence": f"{cl:.0%}",
                "VaR (%)": f"{v['VaR_pct']:.2%}",
                "VaR ($)": f"${v['VaR_dollar']:,.0f}",
                "ES (%)": f"{v['ES_pct']:.2%}",
                "ES ($)": f"${v['ES_dollar']:,.0f}",
            })
        st.subheader("Risk Summary")
        st.table(pd.DataFrame(rows))

    # ── 4. VaR confidence curve ──────────────────────────────────
    with tab4:
        cls = np.linspace(0.80, 0.995, 100)
        vars_curve = [historical_var(returns, c) for c in cls]
        es_curve = [historical_expected_shortfall(returns, c) for c in cls]

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=cls, y=vars_curve, mode="lines", name="VaR",
            line=dict(color=PRESTIGE_GOLD, width=2.5),
        ))
        fig4.add_trace(go.Scatter(
            x=cls, y=es_curve, mode="lines", name="Expected Shortfall",
            line=dict(color=DARK_FOREST, width=2.5, dash="dash"),
        ))
        fig4.update_layout(
            title="VaR & ES as a Function of Confidence Level",
            xaxis_title="Confidence level",
            yaxis_title="Loss (as fraction of portfolio)",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".1%",
            template=BRAND_TEMPLATE, height=500,
        )
        st.plotly_chart(fig4, width="stretch")

else:
    st.info("Set your parameters in the sidebar and click **Run Simulation**.")
