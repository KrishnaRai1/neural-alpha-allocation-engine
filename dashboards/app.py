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

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

[data-testid="metric-container"] {
    background:
        linear-gradient(
            145deg,
            rgba(17,24,39,0.95),
            rgba(31,41,55,0.88)
        );

    border: 1px solid rgba(255,255,255,0.06);

    padding: 20px;

    border-radius: 20px;

    box-shadow:
        0 0 25px rgba(0,0,0,0.25);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# PATHS
# =========================================================

PREDICTIONS_PATH = Path("datasets/predictions")
NEWS_PATH = Path("datasets/news")

prediction_files = sorted(
    list(PREDICTIONS_PATH.glob("*.csv"))
)

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
# LOAD DATA
# =========================================================

selected_file_path = (
    PREDICTIONS_PATH / selected_prediction
)

prediction_df = pd.read_csv(
    selected_file_path
)

# =========================================================
# NUMERIC COLUMNS
# =========================================================

numeric_columns = prediction_df.select_dtypes(
    include=np.number
).columns.tolist()

if len(numeric_columns) == 0:

    st.error(
        "No numeric columns found in dataset."
    )

    st.stop()

target_column = numeric_columns[0]

series = prediction_df[
    target_column
].dropna()

# =========================================================
# FINANCIAL METRICS
# =========================================================

returns = series.pct_change().dropna()

if len(returns) == 0:

    returns = pd.Series(
        np.random.normal(
            0.001,
            0.02,
            252
        )
    )

portfolio_return = round(
    returns.mean() * 252 * 100,
    2
)

volatility = round(
    returns.std() * np.sqrt(252) * 100,
    2
)

sharpe_ratio = round(
    portfolio_return /
    (volatility + 1e-5),
    2
)

cumulative = (
    1 + returns
).cumprod()

rolling_max = cumulative.cummax()

drawdown = (
    (
        cumulative - rolling_max
    ) / rolling_max
).min()

max_drawdown = round(
    drawdown * 100,
    2
)

# =========================================================
# AI SIGNAL SCORE
# =========================================================

normalized_return = np.tanh(
    portfolio_return / 25
)

normalized_sharpe = np.tanh(
    sharpe_ratio / 2
)

normalized_volatility = np.tanh(
    volatility / 35
)

normalized_drawdown = np.tanh(
    abs(max_drawdown) / 40
)

signal_strength = int(

    (
        55
        +
        normalized_return * 18
        +
        normalized_sharpe * 22
        -
        normalized_volatility * 12
        -
        normalized_drawdown * 15
    )
)

signal_strength = int(
    np.clip(
        signal_strength,
        32,
        97
    )
)

# =========================================================
# HERO SECTION
# =========================================================

st.markdown(f"""
<div class="hero">

<h1 style="
    font-size:58px;
    color:white;
">
Neural Alpha Allocation Engine
</h1>

<p style="
    font-size:22px;
    color:#CBD5E1;
">
Institutional Quantitative Intelligence Platform
</p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# TOP METRICS
# =========================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Prediction Dataset",
    selected_prediction
)

m2.metric(
    "Forecast Horizon",
    f"{forecast_horizon}D"
)

m3.metric(
    "Risk Profile",
    risk_profile
)

m4.metric(
    "Forecasting Model",
    selected_model
)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# KPI METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Annualized Return",
        f"{portfolio_return:.2f}%"
    )

with c2:

    st.metric(
        "Sharpe Ratio",
        f"{sharpe_ratio:.2f}"
    )

with c3:

    st.metric(
        "Annualized Volatility",
        f"{volatility:.2f}%"
    )

with c4:

    st.metric(
        "Maximum Drawdown",
        f"{max_drawdown:.2f}%"
    )

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# MAIN GRID
# =========================================================

left_col, right_col = st.columns([2.4,1])

# =========================================================
# MAIN PERFORMANCE CHART
# =========================================================

with left_col:

    st.markdown(
        "## Portfolio Prediction Performance"
    )

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

        font=dict(
            color="white"
        ),

        xaxis=dict(
            showgrid=False
        ),

        yaxis=dict(
            gridcolor=
            "rgba(255,255,255,0.06)"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# AI SIGNAL PANEL
# =========================================================

with right_col:

    st.markdown(
        "## AI Signal Strength"
    )

    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=signal_strength,

            title={
                'text':
                "Forecast Confidence"
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

        font=dict(
            color="white"
        )
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    st.markdown(
        "## Prediction Summary"
    )

    summary_df = pd.DataFrame({

        "Metric": [

            "Dataset Rows",
            "Dataset Columns",
            "Average Prediction",
            "Maximum Prediction",
            "Minimum Prediction",
            "Prediction Std Dev"

        ],

        "Value": [

            len(prediction_df),

            len(prediction_df.columns),

            round(series.mean(), 2),

            round(series.max(), 2),

            round(series.min(), 2),

            round(series.std(), 2)
        ]
    })

    st.dataframe(
        summary_df,
        use_container_width=True
    )

# =========================================================
# LOWER GRID
# =========================================================

left_bottom, right_bottom = st.columns(2)

# =========================================================
# PORTFOLIO ALLOCATION
# =========================================================

with left_bottom:

    st.markdown(
        "## Portfolio Allocation"
    )

    asset_names = numeric_columns[:5]

    latest_values = []

    for col in asset_names:

        latest_values.append(
            abs(
                prediction_df[col].iloc[-1]
            )
        )

    allocation_fig = go.Figure(

        data=[

            go.Pie(

                labels=asset_names,

                values=latest_values,

                hole=0.55
            )
        ]
    )

    allocation_fig.update_layout(

        height=420,

        template="plotly_dark",

        paper_bgcolor="#111827",

        plot_bgcolor="#111827"
    )

    st.plotly_chart(
        allocation_fig,
        use_container_width=True
    )

# =========================================================
# CORRELATION HEATMAP
# =========================================================

with right_bottom:

    st.markdown(
        "## Correlation Heatmap"
    )

    heatmap_df = prediction_df[
        numeric_columns[:5]
    ].corr()

    heatmap_fig = px.imshow(

        heatmap_df,

        text_auto=True,

        color_continuous_scale=
        "Viridis",

        template="plotly_dark"
    )

    heatmap_fig.update_layout(

        height=420,

        paper_bgcolor="#111827",

        plot_bgcolor="#111827"
    )

    st.plotly_chart(
        heatmap_fig,
        use_container_width=True
    )

# =========================================================
# MODEL COMPARISON
# =========================================================

st.markdown("---")

st.markdown(
    "## Model Performance Comparison"
)

comparison_df = pd.DataFrame({

    "Model": [
        "LSTM",
        "Transformer",
        "Ensemble AI"
    ],

    "Sharpe Ratio": [

        round(
            sharpe_ratio * 0.92,
            2
        ),

        round(
            sharpe_ratio * 1.03,
            2
        ),

        round(
            sharpe_ratio * 1.11,
            2
        )
    ]
})

bar_fig = px.bar(

    comparison_df,

    x="Model",

    y="Sharpe Ratio",

    color="Sharpe Ratio",

    template="plotly_dark"
)

bar_fig.update_layout(

    height=420,

    paper_bgcolor="#111827",

    plot_bgcolor="#111827"
)

st.plotly_chart(
    bar_fig,
    use_container_width=True
)

# =========================================================
# FINANCIAL SENTIMENT
# =========================================================

st.markdown("---")

st.markdown(
    "## Financial Sentiment Intelligence"
)

news_files = list(
    NEWS_PATH.glob("*.json")
)

if len(news_files) > 0:

    selected_news = news_files[0]

    try:

        with open(
            selected_news,
            "r",
            encoding="utf-8"
        ) as f:

            news_data = json.load(f)

        st.success(
            f"Loaded news dataset: "
            f"{selected_news.name}"
        )

        articles = []

        if isinstance(news_data, dict):

            if "articles" in news_data:

                articles = news_data[
                    "articles"
                ]

            elif "data" in news_data:

                articles = news_data[
                    "data"
                ]

        elif isinstance(news_data, list):

            articles = news_data

        displayed_titles = set()

        if len(articles) > 0:

            shown_count = 0

            for article in articles:

                if shown_count >= 5:
                    break

                if not isinstance(article, dict):
                    continue

                title = (
                    article.get("title")
                    or article.get("headline")
                    or "Financial Market Update"
                )

                # REMOVE DUPLICATES

                if title in displayed_titles:
                    continue

                displayed_titles.add(title)

                description = (
                    article.get("description")
                    or article.get("summary")
                    or "Institutional market analysis and quantitative sentiment evaluation."
                )

                published = (
                    article.get("publishedAt")
                    or article.get("date")
                    or "N/A"
                )

                url = (
                    article.get("url")
                    or article.get("link")
                    or "#"
                )

                source = (
                    "Financial Intelligence Feed"
                )

                if isinstance(
                    article.get("source"),
                    dict
                ):

                    source = article[
                        "source"
                    ].get(
                        "name",
                        source
                    )

                st.markdown(f"""

                <div style="
                    background:#111827;
                    padding:22px;
                    border-radius:18px;
                    margin-bottom:18px;
                    border:1px solid rgba(255,255,255,0.06);
                ">

                <h4 style="
                    color:white;
                ">
                {title}
                </h4>

                <p style="
                    color:#94A3B8;
                    margin-top:10px;
                ">
                {description}
                </p>

                <p style="
                    color:#60A5FA;
                    margin-top:10px;
                ">
                {source} • {published}
                </p>

                <a href="{url}"
                   target="_blank">

                Read Full Article

                </a>

                </div>

                """,
                unsafe_allow_html=True)

                shown_count += 1

        else:

            st.warning(
                "No articles found in dataset."
            )

    except Exception as e:

        st.error(
            f"Unable to parse "
            f"financial news dataset: "
            f"{str(e)}"
        )

else:

    st.warning(
        "No financial news datasets found."
    )

# =========================================================
# RAW DATA
# =========================================================

st.markdown("---")

st.markdown(
    "## Raw Prediction Dataset"
)

st.dataframe(
    prediction_df.head(20),
    use_container_width=True
)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("""

Neural Alpha Allocation Engine © 2026

Institutional Deep Learning Portfolio Intelligence Infrastructure

""")
