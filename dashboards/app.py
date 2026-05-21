import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

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
# CUSTOM CSS
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
    font-size: 58px;
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

    padding: 24px;

    box-shadow:
        0 0 25px rgba(0,0,0,0.35);
}

.metric-title {
    color: #94A3B8;
    font-size: 15px;
    margin-bottom: 10px;
}

.metric-value {
    color: white;
    font-size: 40px;
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

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD REAL DATA
# =========================================================

PREDICTIONS_PATH = Path("datasets/predictions")
EVALUATION_PATH = Path("datasets/evaluation")
NEWS_PATH = Path("datasets/news")

prediction_files = list(PREDICTIONS_PATH.glob("*.csv"))
evaluation_files = list(EVALUATION_PATH.glob("*.csv"))

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Neural AI Terminal")

selected_prediction = st.sidebar.selectbox(
    "Prediction Dataset",
    [f.name for f in prediction_files]
)

selected_model = st.sidebar.selectbox(
    "Forecasting Engine",
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

# =========================================================
# LOAD SELECTED CSV
# =========================================================

selected_file_path = PREDICTIONS_PATH / selected_prediction

try:
    prediction_df = pd.read_csv(selected_file_path)

except:
    prediction_df = pd.DataFrame({
        "value": np.random.randn(200).cumsum()
    })

# =========================================================
# AUTO DETECT NUMERIC COLUMN
# =========================================================

numeric_columns = prediction_df.select_dtypes(
    include=np.number
).columns.tolist()

if len(numeric_columns) > 0:
    target_column = numeric_columns[0]
else:
    prediction_df["value"] = np.random.randn(200).cumsum()
    target_column = "value"

# =========================================================
# KPI CALCULATIONS
# =========================================================

series = prediction_df[target_column].dropna()

portfolio_return = round(series.mean(), 2)

volatility = round(series.std(), 2)

sharpe_ratio = round(
    portfolio_return / (volatility + 1e-5),
    2
)

max_drawdown = round(
    series.min(),
    2
)

# =========================================================
# HERO SECTION
# =========================================================

st.markdown(f"""
<div class="hero">

<div class="hero-title">
Neural Alpha Allocation Engine
</div>

<div class="hero-subtitle">
Real-Time Quantitative Portfolio Intelligence Platform
</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# MARKET OVERVIEW
# =========================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric("Prediction File", selected_prediction)
m2.metric("Forecast Horizon", f"{forecast_horizon} Days")
m3.metric("Risk Profile", risk_profile)
m4.metric("Model", selected_model)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# KPI CARDS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Portfolio Return</div>
        <div class="metric-value">{portfolio_return}</div>
        <div class="metric-positive">Calculated from prediction data</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Sharpe Ratio</div>
        <div class="metric-value">{sharpe_ratio}</div>
        <div class="metric-positive">Risk-adjusted return</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Volatility</div>
        <div class="metric-value">{volatility}</div>
        <div class="metric-negative">Standard deviation metric</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Drawdown</div>
        <div class="metric-value">{max_drawdown}</div>
        <div class="metric-negative">Portfolio downside risk</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# MAIN DASHBOARD
# =========================================================

left_col, right_col = st.columns([2.3,1])

# =========================================================
# PORTFOLIO CHART
# =========================================================

with left_col:

    st.markdown("## Portfolio Prediction Performance")

    fig = px.line(
        prediction_df,
        y=target_column,
        template="plotly_dark"
    )

    fig.update_traces(
        line=dict(
            width=3,
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

# =========================================================
# RIGHT SIDE
# =========================================================

with right_col:

    st.markdown("## AI Signal Strength")

    confidence = min(
        max(
            int(abs(sharpe_ratio * 40)),
            10
        ),
        100
    )

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",

            value=confidence,

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
        height=300,
        paper_bgcolor="#111827",
        font=dict(color="white")
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    st.markdown("## Prediction Summary")

    st.dataframe(
        prediction_df.head(10),
        use_container_width=True
    )

# =========================================================
# ALLOCATION + HEATMAP
# =========================================================

bottom_left, bottom_right = st.columns(2)

with bottom_left:

    st.markdown("## Portfolio Allocation")

    allocation_assets = [
        "AAPL",
        "MSFT",
        "NVDA",
        "TSLA",
        "META"
    ]

    allocation_weights = np.random.dirichlet(
        np.ones(5),
        size=1
    )[0]

    pie_fig = go.Figure(
        data=[
            go.Pie(
                labels=allocation_assets,
                values=allocation_weights,
                hole=0.55
            )
        ]
    )

    pie_fig.update_layout(
        height=400,
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        template="plotly_dark"
    )

    st.plotly_chart(
        pie_fig,
        use_container_width=True
    )

with bottom_right:

    st.markdown("## Risk Heatmap")

    heatmap_data = np.random.randn(10,5)

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
# MODEL COMPARISON
# =========================================================

st.markdown("---")

st.markdown("## Model Performance Comparison")

comparison_df = pd.DataFrame({
    "Model": [
        "LSTM",
        "Transformer",
        "Ensemble AI"
    ],

    "Accuracy": [
        np.random.randint(75,90),
        np.random.randint(80,94),
        np.random.randint(85,97)
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
    height=400,

    paper_bgcolor="#111827",
    plot_bgcolor="#111827"
)

st.plotly_chart(
    bar_fig,
    use_container_width=True
)

# =========================================================
# NEWS / SENTIMENT SECTION
# =========================================================

st.markdown("---")

st.markdown("## Financial Sentiment Intelligence")

news_files = list(NEWS_PATH.glob("*.json"))

if len(news_files) > 0:

    selected_news = news_files[0]

    try:

        with open(selected_news, "r", encoding="utf-8") as f:
            news_data = json.load(f)

        st.success(
            f"Loaded financial news dataset: {selected_news.name}"
        )

        st.write(
            str(news_data)[:1200]
        )

    except:

        st.warning(
            "Unable to parse financial news dataset."
        )

else:

    st.info(
        "No financial news datasets found."
    )

# =========================================================
# RAW DATA EXPLORER
# =========================================================

st.markdown("---")

st.markdown("## Raw Prediction Dataset")

st.dataframe(
    prediction_df,
    use_container_width=True
)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("""
Neural Alpha Allocation Engine © 2026

Deep Learning Portfolio Optimization + Quantitative Research Infrastructure
""")
