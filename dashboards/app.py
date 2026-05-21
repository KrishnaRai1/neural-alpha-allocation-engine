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

html, body, [class*="css"] {
    background-color: #050816;
    color: white;
    font-family: 'Inter', sans-serif;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0B1020 0%,
        #111827 100%
    );

    border-right: 1px solid rgba(255,255,255,0.05);
}

.block-container {
    padding-top: 2rem;
}

.hero-container {
    background:
        linear-gradient(
            135deg,
            rgba(37,99,235,0.18),
            rgba(168,85,247,0.18)
        );

    padding: 45px;

    border-radius: 30px;

    border: 1px solid rgba(255,255,255,0.08);

    margin-bottom: 30px;

    box-shadow:
        0 0 40px rgba(0,0,0,0.35);
}

.hero-title {
    font-size: 60px;
    font-weight: 700;
    color: white;
    margin-bottom: 12px;
}

.hero-subtitle {
    font-size: 22px;
    color: #D1D5DB;
}

.metric-container {
    display: flex;
    gap: 20px;
    margin-top: 10px;
    margin-bottom: 35px;
}

.metric-card {
    flex: 1;

    background:
        linear-gradient(
            145deg,
            rgba(17,24,39,0.92),
            rgba(31,41,55,0.88)
        );

    border: 1px solid rgba(255,255,255,0.08);

    padding: 30px;

    border-radius: 24px;

    backdrop-filter: blur(14px);

    box-shadow:
        0 0 35px rgba(0,0,0,0.35);

    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow:
        0 0 40px rgba(59,130,246,0.28);
}

.metric-title {
    color: #9CA3AF;
    font-size: 16px;
    margin-bottom: 10px;
}

.metric-value {
    color: white;
    font-size: 42px;
    font-weight: 700;
}

.metric-positive {
    color: #22C55E;
    font-size: 16px;
    margin-top: 10px;
}

.metric-negative {
    color: #EF4444;
    font-size: 16px;
    margin-top: 10px;
}

.dashboard-card {
    background:
        linear-gradient(
            145deg,
            rgba(17,24,39,0.92),
            rgba(31,41,55,0.88)
        );

    border-radius: 24px;

    padding: 25px;

    border: 1px solid rgba(255,255,255,0.06);

    margin-bottom: 25px;
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

st.sidebar.markdown("# Neural Allocation Terminal")

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
    "Forecasting Model",
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

rebalance_frequency = st.sidebar.selectbox(
    "Rebalancing Frequency",
    [
        "Daily",
        "Weekly",
        "Monthly"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("""
Deep Learning + Portfolio Optimization + Financial NLP
""")

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero-container">

<div class="hero-title">
Neural Alpha Allocation Engine
</div>

<div class="hero-subtitle">
Institutional Deep Learning Portfolio Intelligence Platform
</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# MARKET OVERVIEW
# =========================================================

market1, market2, market3, market4 = st.columns(4)

market1.metric("NASDAQ", "18,420", "+1.8%")
market2.metric("S&P 500", "5,421", "+0.9%")
market3.metric("BTC", "$104,220", "+3.1%")
market4.metric("VIX", "13.8", "-4.2%")

st.markdown("---")

# =========================================================
# PREMIUM KPI CARDS
# =========================================================

st.markdown("""
<div class="metric-container">

<div class="metric-card">
    <div class="metric-title">
        Portfolio Return
    </div>

    <div class="metric-value">
        18.42%
    </div>

    <div class="metric-positive">
        ▲ +2.14%
    </div>
</div>

<div class="metric-card">
    <div class="metric-title">
        Sharpe Ratio
    </div>

    <div class="metric-value">
        1.87
    </div>

    <div class="metric-positive">
        ▲ +0.18
    </div>
</div>

<div class="metric-card">
    <div class="metric-title">
        Annual Volatility
    </div>

    <div class="metric-value">
        11.26%
    </div>

    <div class="metric-negative">
        ▼ -1.02%
    </div>
</div>

<div class="metric-card">
    <div class="metric-title">
        Max Drawdown
    </div>

    <div class="metric-value">
        -6.18%
    </div>

    <div class="metric-positive">
        ▲ Recovery Improving
    </div>
</div>

</div>
""", unsafe_allow_html=True)

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

    st.markdown("## Portfolio Performance")

    returns = np.random.randn(250).cumsum()

    performance_df = pd.DataFrame({
        "Portfolio Returns": returns
    })

    fig = px.line(
        performance_df,
        y="Portfolio Returns",
        template="plotly_dark"
    )

    fig.update_traces(
        line=dict(
            width=4,
            color="#60A5FA"
        )
    )

    fig.update_layout(
        paper_bgcolor="#050816",
        plot_bgcolor="#050816",

        font=dict(color="white"),

        xaxis=dict(
            showgrid=False
        ),

        yaxis=dict(
            gridcolor="rgba(255,255,255,0.08)"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

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

                    hole=0.5
                )
            ]
        )

        pie_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#050816",
            plot_bgcolor="#050816"
        )

        st.plotly_chart(
            pie_fig,
            use_container_width=True
        )

    with col2:

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

    st.markdown("## AI Forecasting Models")

    forecast_df = pd.DataFrame({
        "LSTM": np.random.randn(120).cumsum(),
        "Transformer": np.random.randn(120).cumsum(),
        "Ensemble": np.random.randn(120).cumsum()
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

    st.markdown("## Financial Sentiment Intelligence")

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

    st.markdown("## Quantitative Risk Monitoring")

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

    monte_carlo = np.cumsum(
        np.random.randn(1000)
    )

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

    st.markdown("## Dynamic Allocation Engine")

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
