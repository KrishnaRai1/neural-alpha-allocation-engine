import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    .main {
        background-color: #0B0F19;
        color: white;
    }

    .stMetric {
        background-color: #111827;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #1F2937;
    }

    .css-1d391kg {
        background-color: #111827;
    }

    h1, h2, h3, h4 {
        color: #F9FAFB;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    .dashboard-card {
        background-color: #111827;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #1F2937;
        margin-bottom: 20px;
    }

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Neural Allocation Controls")

selected_asset = st.sidebar.selectbox(
    "Select Asset",
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
        "Ensemble"
    ]
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Days)",
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
    "Rebalance Frequency",
    [
        "Daily",
        "Weekly",
        "Monthly"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("""
Institutional Quantitative Research Environment

Deep Learning + Financial NLP + Portfolio Optimization
""")

# =========================================================
# HEADER
# =========================================================

st.title("Neural Alpha Allocation Engine")

st.markdown("""
### Institutional Deep Learning Portfolio Intelligence Platform

An advanced quantitative finance framework integrating:
- LSTM Forecasting Models
- Transformer-Based Asset Allocation
- FinBERT Financial Sentiment Analysis
- Portfolio Optimization Pipelines
- Quantitative Risk Monitoring
""")

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
    "Portfolio Analytics",
    "Forecasting Models",
    "Sentiment Intelligence",
    "Risk Monitoring",
    "Allocation Engine"
])

# =========================================================
# TAB 1 - PORTFOLIO ANALYTICS
# =========================================================

with tab1:

    st.subheader("Portfolio Performance Analytics")

    portfolio_returns = np.random.randn(300).cumsum()

    performance_df = pd.DataFrame({
        "Portfolio Returns": portfolio_returns
    })

    st.line_chart(performance_df)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Asset Allocation")

        allocation_df = pd.DataFrame({
            "Asset": [
                "AAPL",
                "MSFT",
                "NVDA",
                "TSLA",
                "META"
            ],
            "Allocation": [
                22,
                18,
                24,
                16,
                20
            ]
        })

        st.bar_chart(
            allocation_df.set_index("Asset")
        )

    with col2:

        st.markdown("### Portfolio Statistics")

        stats_df = pd.DataFrame({
            "Metric": [
                "Alpha",
                "Beta",
                "Sortino Ratio",
                "Tracking Error",
                "Information Ratio"
            ],
            "Value": [
                0.18,
                1.04,
                1.62,
                2.14,
                0.92
            ]
        })

        st.dataframe(
            stats_df,
            use_container_width=True
        )

# =========================================================
# TAB 2 - FORECASTING MODELS
# =========================================================

with tab2:

    st.subheader("Deep Learning Forecasting Models")

    forecast_df = pd.DataFrame({
        "LSTM Forecast": np.random.randn(120).cumsum(),
        "Transformer Forecast": np.random.randn(120).cumsum(),
        "Ensemble Signal": np.random.randn(120).cumsum()
    })

    st.line_chart(forecast_df)

    st.markdown("""
    ### Forecasting Pipeline

    The neural forecasting engine integrates:

    - Sequential LSTM architectures
    - Transformer-based temporal learning
    - Ensemble prediction systems
    - Rolling portfolio optimization
    - Dynamic asset allocation strategies
    """)

    model_metrics = pd.DataFrame({
        "Model": [
            "LSTM",
            "Transformer",
            "Ensemble"
        ],
        "RMSE": [
            0.018,
            0.015,
            0.012
        ],
        "Accuracy": [
            "82%",
            "86%",
            "91%"
        ]
    })

    st.dataframe(
        model_metrics,
        use_container_width=True
    )

# =========================================================
# TAB 3 - SENTIMENT INTELLIGENCE
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

    st.bar_chart(
        sentiment_df.set_index("Sentiment")
    )

    st.markdown("""
    ### FinBERT NLP Pipeline

    The sentiment engine processes:
    - Financial news streams
    - Earnings reports
    - Market commentary
    - Institutional sentiment indicators
    - Multi-asset financial narratives

    Sentiment signals are integrated directly into the portfolio allocation engine.
    """)

    news_table = pd.DataFrame({
        "Headline": [
            "AI demand accelerates semiconductor rally",
            "Technology sector volatility stabilizes",
            "Institutional allocation shifts toward growth"
        ],
        "Sentiment": [
            "Positive",
            "Neutral",
            "Positive"
        ]
    })

    st.dataframe(
        news_table,
        use_container_width=True
    )

# =========================================================
# TAB 4 - RISK MONITORING
# =========================================================

with tab4:

    st.subheader("Quantitative Risk Monitoring")

    risk_metrics = pd.DataFrame({
        "Risk Metric": [
            "Portfolio Beta",
            "Value at Risk",
            "Expected Shortfall",
            "Volatility",
            "Tail Risk"
        ],
        "Current Value": [
            1.06,
            -4.21,
            -6.14,
            11.26,
            2.81
        ]
    })

    st.dataframe(
        risk_metrics,
        use_container_width=True
    )

    st.markdown("### Portfolio Risk Distribution")

    risk_distribution = np.random.normal(
        0,
        1,
        1000
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(
        risk_distribution,
        bins=40
    )

    st.pyplot(fig)

# =========================================================
# TAB 5 - ALLOCATION ENGINE
# =========================================================

with tab5:

    st.subheader("Dynamic Allocation Engine")

    allocation_engine_df = pd.DataFrame({
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

    st.dataframe(
        allocation_engine_df,
        use_container_width=True
    )

    st.markdown("""
    ### Optimization Framework

    Allocation optimization integrates:
    - Sharpe ratio maximization
    - Volatility minimization
    - Dynamic risk budgeting
    - Neural forecasting signals
    - Sentiment-enhanced allocation adjustments
    """)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
### Neural Alpha Allocation Engine

Built for:
- Quantitative Finance Research
- Deep Learning Portfolio Optimization
- Financial NLP Workflows
- Institutional Asset Allocation Research
- Systematic Trading Analytics
""")
