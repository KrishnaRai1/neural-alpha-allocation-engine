import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Neural Alpha Allocation Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #050816;
    color: white;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 25%),
        radial-gradient(circle at top right, rgba(168,85,247,0.15), transparent 25%),
        linear-gradient(180deg, #050816 0%, #0B1120 100%);
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0B1020 0%,
            #111827 100%
        );

    border-right: 1px solid rgba(255,255,255,0.06);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background:
        linear-gradient(
            135deg,
            rgba(37,99,235,0.22),
            rgba(168,85,247,0.18)
        );

    border: 1px solid rgba(255,255,255,0.08);

    padding: 45px;

    border-radius: 30px;

    margin-bottom: 30px;

    box-shadow:
        0 0 50px rgba(0,0,0,0.35);
}

.hero-title {
    font-size: 64px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

.hero-subtitle {
    color: #CBD5E1;
    font-size: 22px;
}

.metric-card {
    background:
        linear-gradient(
            145deg,
            rgba(17,24,39,0.95),
            rgba(31,41,55,0.88)
        );

    border: 1px solid rgba(255,255,255,0.06);

    border-radius: 24px;

    padding: 28px;

    box-shadow:
        0 0 25px rgba(0,0,0,0.35);

    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-6px);

    box-shadow:
        0 0 35px rgba(59,130,246,0.22);
}

.metric-title {
    color: #94A3B8;
    font-size: 15px;
    margin-bottom: 10px;
}

.metric-value {
    color: white;
    font-size: 42px;
    font-weight: 700;
}

.metric-positive {
    color: #22C55E;
    margin-top: 10px;
}

.metric-negative {
    color: #EF4444;
    margin-top: 10px;
}

.panel {
    background:
        linear-gradient(
            145deg,
            rgba(17,24,39,0.92),
            rgba(31,41,55,0.88)
        );

    border-radius: 26px;

    padding: 22px;

    border: 1px solid rgba(255,255,255,0.05);

    margin-bottom: 25px;

    box-shadow:
        0 0 22px rgba(0,0,0,0.28);
}

.sidebar-title {
    font-size: 28px;
    font-weight: 700;
    color: white;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("""
<div class="sidebar-title">
Neural AI Terminal
</div>
""", unsafe_allow_html=True)

selected_asset = st.sidebar.selectbox(
    "Asset Universe",
    [
        "NVDA",
        "AAPL",
        "MSFT",
        "TSLA",
        "META",
        "AMZN",
        "GOOGL"
    ]
)

selected_model = st.sidebar.selectbox(
    "Forecast Model",
    [
        "LSTM",
        "Transformer",
        "Ensemble AI"
    ]
)

forecast_horizon = st.sidebar.slider(
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

st.sidebar.info("""
Deep Learning + Portfolio Optimization + Financial NLP + Institutional Quant Research
""")

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">

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

m1, m2, m3, m4 = st.columns(4)

m1.metric("NASDAQ", "18,420", "+1.8%")
m2.metric("S&P 500", "5,421", "+0.9%")
m3.metric("BTC", "$104,200", "+3.6%")
m4.metric("VIX", "13.2", "-4.1%")

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# PREMIUM KPI CARDS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Portfolio Return</div>
        <div class="metric-value">18.42%</div>
        <div class="metric-positive">▲ +2.14%</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Sharpe Ratio</div>
        <div class="metric-value">1.87</div>
        <div class="metric-positive">▲ +0.18</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Annual Volatility</div>
        <div class="metric-value">11.26%</div>
        <div class="metric-negative">▼ -1.02%</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Max Drawdown</div>
        <div class="metric-value">-6.18%</div>
        <div class="metric-positive">▲ Recovery Improving</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# MAIN DASHBOARD GRID
# =========================================================

left_col, right_col = st.columns([2.2,1])

# =========================================================
# LEFT SIDE
# =========================================================

with left_col:

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
        height=520,

        paper_bgcolor="#111827",
        plot_bgcolor="#111827",

        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10
        ),

        font=dict(color="white"),

        xaxis=dict(
            showgrid=False
        ),

        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:

        st.markdown("### Portfolio Allocation")

        pie_fig = go.Figure(
            data=[
                go.Pie(
                    labels=[
                        "NVDA",
                        "AAPL",
                        "MSFT",
                        "META",
                        "AMZN"
                    ],

                    values=[
                        24,
                        22,
                        18,
                        16,
                        20
                    ],

                    hole=0.55
                )
            ]
        )

        pie_fig.update_layout(
            height=350,
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            template="plotly_dark"
        )

        st.plotly_chart(
            pie_fig,
            use_container_width=True
        )

    with bottom_right:

        st.markdown("### Risk Heatmap")

        heatmap_data = np.random.randn(8,5)

        heatmap_df = pd.DataFrame(
            heatmap_data,

            columns=[
                "AAPL",
                "MSFT",
                "NVDA",
                "TSLA",
                "META"
            ]
        )

        st.dataframe(
            heatmap_df.style.background_gradient(
                cmap="viridis"
            ),

            use_container_width=True
        )

# =========================================================
# RIGHT SIDE
# =========================================================

with right_col:

    st.markdown("### AI Signal Strength")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",

            value=91,

            title={
                'text': "Forecast Confidence"
            },

            gauge={
                'axis': {
                    'range': [0,100]
                },

                'bar': {
                    'color': "#3B82F6"
                },

                'bgcolor': "#111827",

                'steps': [
                    {
                        'range': [0,50],
                        'color': "#1F2937"
                    },

                    {
                        'range': [50,80],
                        'color': "#374151"
                    },

                    {
                        'range': [80,100],
                        'color': "#2563EB"
                    }
                ]
            }
        )
    )

    gauge.update_layout(
        height=320,
        paper_bgcolor="#111827",
        font=dict(color="white")
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    st.markdown("### Live AI Signals")

    signal_df = pd.DataFrame({
        "Signal": [
            "Momentum",
            "Volatility",
            "Sentiment",
            "Liquidity",
            "Allocation"
        ],

        "Status": [
            "Bullish",
            "Stable",
            "Positive",
            "Strong",
            "Optimized"
        ]
    })

    st.dataframe(
        signal_df,
        use_container_width=True
    )

    st.markdown("### Portfolio Diagnostics")

    st.metric(
        "Expected Return",
        "16.8%",
        "+1.2%"
    )

    st.metric(
        "Sortino Ratio",
        "2.14",
        "+0.21"
    )

    st.metric(
        "Tracking Error",
        "1.84%",
        "-0.3%"
    )

# =========================================================
# ADVANCED ANALYTICS
# =========================================================

st.markdown("---")

a1, a2 = st.columns(2)

with a1:

    st.markdown("## Monte Carlo Simulation")

    monte = np.cumsum(
        np.random.randn(300)
    )

    monte_df = pd.DataFrame({
        "Simulation": monte
    })

    monte_fig = px.line(
        monte_df,
        y="Simulation",
        template="plotly_dark"
    )

    monte_fig.update_layout(
        height=380,

        paper_bgcolor="#111827",
        plot_bgcolor="#111827",

        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        )
    )

    st.plotly_chart(
        monte_fig,
        use_container_width=True
    )

with a2:

    st.markdown("## Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": [
            "LSTM",
            "Transformer",
            "Ensemble AI"
        ],

        "Accuracy": [
            82,
            88,
            93
        ]
    })

    bar_fig = px.bar(
        comparison_df,
        x="Model",
        y="Accuracy",
        color="Accuracy",
        template="plotly_dark"
    )

    bar_fig.update_layout(
        height=380,

        paper_bgcolor="#111827",
        plot_bgcolor="#111827"
    )

    st.plotly_chart(
        bar_fig,
        use_container_width=True
    )

# =========================================================
# SENTIMENT INTELLIGENCE
# =========================================================

st.markdown("---")

st.markdown("## Financial Sentiment Intelligence")

news1, news2, news3 = st.columns(3)

with news1:
    st.success("""
    AI chip demand accelerates semiconductor momentum.
    """)

with news2:
    st.info("""
    Institutional capital rotating into growth equities.
    """)

with news3:
    st.warning("""
    Volatility expectations remain elevated near earnings.
    """)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("""
Neural Alpha Allocation Engine © 2026

Institutional Quantitative Research & Deep Learning Infrastructure
""")
```
