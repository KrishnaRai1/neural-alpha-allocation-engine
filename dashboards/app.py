import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Neural Alpha Allocation Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"]  {
    background-color: #050816;
    color: white;
    font-family: 'Inter', sans-serif;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1020 0%, #111827 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}

.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
}

.dashboard-title {
    font-size: 48px;
    font-weight: 700;
    color: white;
}

.subtitle {
    color: #9CA3AF;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.image(
    "https://images.unsplash.com/photo-1642052502503-95f4d8f5f8f5",
    use_container_width=True
)

st.sidebar.markdown("## Neural Allocation Terminal")

selected_asset = st.sidebar.selectbox(
    "Asset Universe",
    [
        "AAPL",
        "MSFT",
        "NVDA",
        "TSLA",
        "META",
        "GOOGL",
        "AMZN"
    ]
)

selected_model = st.sidebar.selectbox(
    "Forecasting Architecture",
    [
        "LSTM",
        "Transformer",
        "Ensemble AI"
    ]
)

forecast_days = st.sidebar.slider(
    "Forecast Horizon",
    7,
    180,
    30
)

risk_profile = st.sidebar.radio(
    "Risk Profile",
    [
        "Conservative",
        "Balanced",
        "Aggressive"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("""
Deep Learning + Portfolio Optimization + Financial NLP
""")

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="dashboard-title">
Neural Alpha Allocation Engine
</div>

<div class="subtitle">
Institutional Quantitative Intelligence Platform
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# MARKET OVERVIEW
# =========================================================

ticker1, ticker2, ticker3, ticker4 = st.columns(4)

ticker1.metric("NASDAQ", "18,420", "+1.8%")
ticker2.metric("S&P 500", "5,421", "+0.9%")
ticker3.metric("BTC", "$104,220", "+3.1%")
ticker4.metric("VIX", "13.8", "-4.2%")

st.markdown("---")

# =========================================================
# KPI METRICS
# =========================================================

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Portfolio Return",
    "18.42%",
    "+2.14%"
)

metric2.metric(
    "Sharpe Ratio",
    "1.87",
    "+0.18"
)

metric3.metric(
    "Annual Volatility",
    "11.26%",
    "-1.02%"
)

metric4.metric(
    "Maximum Drawdown",
    "-6.18%",
    "-0.42%"
)

st.markdown("---")

# =========================================================
# MAIN TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Portfolio Analytics",
    "🧠 AI Forecasting",
    "📰 Sentiment Intelligence",
    "⚠️ Risk Engine",
    "🚀 Allocation Engine"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.subheader("Portfolio Performance")

    returns = np.random.randn(300).cumsum()

    performance_df = pd.DataFrame({
        "Portfolio Returns": returns
    })

    fig = px.line(
        performance_df,
        y="Portfolio Returns",
        template="plotly_dark"
    )

    fig.update_layout(
        paper_bgcolor="#050816",
        plot_bgcolor="#050816"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Asset Allocation")

        pie_fig = go.Figure(
            data=[
                go.Pie(
                    labels=[
                        "AAPL",
                        "MSFT",
                        "NVDA",
                        "TSLA",
                        "META"
                    ],
                    values=[
                        22,
                        18,
                        24,
                        16,
                        20
                    ],
                    hole=0.4
                )
            ]
        )

        pie_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#050816"
        )

        st.plotly_chart(
            pie_fig,
            use_container_width=True
        )

    with col2:

        st.markdown("### Portfolio Statistics")

        stats_df = pd.DataFrame({
            "Metric": [
                "Alpha",
                "Beta",
                "Sortino Ratio",
                "Information Ratio",
                "Tracking Error"
            ],
            "Value": [
                0.18,
                1.04,
                1.62,
                0.92,
                2.14
            ]
        })

        st.dataframe(
            stats_df,
            use_container_width=True
        )

# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.subheader("AI Forecasting Models")

    forecast_df = pd.DataFrame({
        "LSTM": np.random.randn(150).cumsum(),
        "Transformer": np.random.randn(150).cumsum(),
        "Ensemble": np.random.randn(150).cumsum()
    })

    forecast_fig = px.line(
        forecast_df,
        template="plotly_dark"
    )

    forecast_fig.update_layout(
        paper_bgcolor="#050816",
        plot_bgcolor="#050816"
    )

    st.plotly_chart(
        forecast_fig,
        use_container_width=True
    )

    st.progress(91)

    st.caption("AI Forecast Confidence Score")

    model_df = pd.DataFrame({
        "Model": [
            "LSTM",
            "Transformer",
            "Ensemble AI"
        ],
        "RMSE": [
            0.018,
            0.015,
            0.011
        ],
        "Accuracy": [
            "82%",
            "87%",
            "92%"
        ]
    })

    st.dataframe(
        model_df,
        use_container_width=True
    )

# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.subheader("Financial Sentiment Intelligence")

    sentiment_df = pd.DataFrame({
        "Sentiment": [
            "Positive",
            "Neutral",
            "Negative"
        ],
        "Score": [
            64,
            22,
            14
        ]
    })

    sentiment_fig = px.bar(
        sentiment_df,
        x="Sentiment",
        y="Score",
        template="plotly_dark"
    )

    sentiment_fig.update_layout(
        paper_bgcolor="#050816",
        plot_bgcolor="#050816"
    )

    st.plotly_chart(
        sentiment_fig,
        use_container_width=True
    )

    news_df = pd.DataFrame({
        "Headline": [
            "AI demand accelerates semiconductor rally",
            "Technology sector volatility stabilizes",
            "Institutional capital rotates into growth assets"
        ],
        "Sentiment": [
            "Positive",
            "Neutral",
            "Positive"
        ]
    })

    st.dataframe(
        news_df,
        use_container_width=True
    )

# =========================================================
# TAB 4
# =========================================================

with tab4:

    st.subheader("Quantitative Risk Monitoring")

    risk_df = pd.DataFrame(
        np.random.randn(10, 5),
        columns=[
            "AAPL",
            "MSFT",
            "NVDA",
            "TSLA",
            "META"
        ]
    )

    st.dataframe(
        risk_df.style.background_gradient(cmap="viridis")
    )

    monte_carlo = np.cumsum(np.random.randn(1000))

    mc_df = pd.DataFrame({
        "Monte Carlo Simulation": monte_carlo
    })

    mc_fig = px.line(
        mc_df,
        template="plotly_dark"
    )

    mc_fig.update_layout(
        paper_bgcolor="#050816",
        plot_bgcolor="#050816"
    )

    st.plotly_chart(
        mc_fig,
        use_container_width=True
    )

# =========================================================
# TAB 5
# =========================================================

with tab5:

    st.subheader("Dynamic Allocation Engine")

    allocation_df = pd.DataFrame({
        "Asset": [
            "AAPL",
            "MSFT",
            "NVDA",
            "TSLA",
            "META",
            "AMZN"
        ],
        "Expected Return": [
            14.2,
            12.4,
            18.1,
            10.6,
            13.2,
            11.9
        ],
        "Optimized Weight": [
            0.22,
            0.18,
            0.24,
            0.14,
            0.12,
            0.10
        ]
    })

    alloc_fig = px.bar(
        allocation_df,
        x="Asset",
        y="Optimized Weight",
        color="Expected Return",
        template="plotly_dark"
    )

    alloc_fig.update_layout(
        paper_bgcolor="#050816",
        plot_bgcolor="#050816"
    )

    st.plotly_chart(
        alloc_fig,
        use_container_width=True
    )

    with st.expander("View Allocation Methodology"):

        st.write("""
        The optimization framework integrates:

        - Sharpe Ratio Optimization
        - Deep Learning Forecasting
        - Transformer Temporal Attention
        - Volatility Targeting
        - FinBERT Sentiment Signals
        - Dynamic Portfolio Rebalancing
        """)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("""
Neural Alpha Allocation Engine © 2026

Institutional Quantitative Research Platform
""")
