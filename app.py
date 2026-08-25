"""
Streamlit Web Dashboard — Empirical Financial Sentiment & Market Correlation Pipeline

Provides an interactive user interface to:
1. Select stock tickers & lookback windows
2. Execute data collection, VADER sentiment scoring, and financial analytics
3. View interactive Plotly dual-axis price/sentiment charts
4. Inspect statistical correlations, scatter trendlines, and raw headline scores

Run locally via:
    streamlit run app.py
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.data_fetcher import fetch_market_prices, fetch_news_headlines
from src.text_processor import process_news_dataframe
from src.sentiment_engine import SentimentEngine
from src.financial_analytics import create_master_dataset
from src.statistical_analysis import run_full_statistical_analysis

# Page Config
st.set_page_config(
    page_title="Financial Sentiment & Market Correlation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme aesthetic & KPI styling
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 18px 22px;
        border: 1px solid #2e3545;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00e676;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #a0aab8;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


def load_existing_data():
    master_path = "data/processed/master_sentiment_prices.csv"
    raw_news_path = "data/raw/raw_news.csv"
    report_path = "outputs/reports/summary_report.json"

    master_df = pd.read_csv(master_path) if os.path.exists(master_path) else pd.DataFrame()
    raw_news_df = pd.read_csv(raw_news_path) if os.path.exists(raw_news_path) else pd.DataFrame()
    
    stats_summary = {}
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            stats_summary = json.load(f)

    return master_df, raw_news_df, stats_summary


def run_pipeline_flow(tickers, days_back):
    with st.spinner("Fetching market prices & news headlines..."):
        prices_df = fetch_market_prices(tickers=tickers, days_back=days_back)
        news_df = fetch_news_headlines(tickers=tickers, days_back=days_back)

    if prices_df.empty:
        st.error("Market price collection returned an empty dataset. Check internet connection or tickers.")
        return pd.DataFrame(), pd.DataFrame(), {}

    with st.spinner("Cleaning headlines & performing market time alignment..."):
        processed_news = process_news_dataframe(news_df)

    with st.spinner("Scoring sentiment via NLTK VADER..."):
        engine = SentimentEngine()
        scored_news = engine.score_news_dataframe(processed_news)
        daily_sentiment = engine.aggregate_daily_sentiment(scored_news)

    with st.spinner("Computing financial metrics & merging master table..."):
        master_df = create_master_dataset(
            prices_df=prices_df,
            daily_sentiment_df=daily_sentiment,
            output_path="data/processed/master_sentiment_prices.csv"
        )

    with st.spinner("Executing empirical statistical analysis..."):
        stats_summary = run_full_statistical_analysis(master_df)
        os.makedirs("outputs/reports", exist_ok=True)
        with open("outputs/reports/summary_report.json", "w") as f:
            json.dump(stats_summary, f, indent=2)

    return master_df, news_df, stats_summary


def main():
    st.title("📈 Empirical Market Sentiment & Price Correlation Pipeline")
    st.caption("Quantitative analysis of financial news headline sentiment (NLTK VADER) vs daily stock returns.")

    # Sidebar Controls
    st.sidebar.header("🕹️ Pipeline Controls")
    
    default_tickers = ["AAPL", "MSFT", "NVDA", "RELIANCE.NS", "TCS.NS"]
    selected_tickers = st.sidebar.multiselect(
        "Select Basket Tickers",
        options=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "RELIANCE.NS", "TCS.NS", "INFY.NS"],
        default=default_tickers
    )

    custom_ticker = st.sidebar.text_input("Add Custom Ticker (e.g. TSLA, INFY.NS)", value="")
    if custom_ticker and custom_ticker not in selected_tickers:
        selected_tickers.append(custom_ticker.strip().upper())

    days_back = st.sidebar.slider("Lookback Window (Days)", min_value=7, max_value=90, value=30, step=1)
    
    run_btn = st.sidebar.button("🚀 Run Analysis Pipeline", type="primary", use_container_width=True)

    # Initial data load
    if run_btn:
        master_df, news_df, stats_summary = run_pipeline_flow(selected_tickers, days_back)
    else:
        master_df, news_df, stats_summary = load_existing_data()

    if master_df.empty:
        st.info("No master dataset found. Select parameters in the sidebar and click **Run Analysis Pipeline**.")
        return

    # Top KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    acc = stats_summary.get("directional_accuracy", {}).get("overall_directional_acc_pct", 0)
    same_corr = stats_summary.get("correlations", {}).get("pooled_same_day", {})
    pearson_r = same_corr.get("pearson_r", 0)
    pearson_p = same_corr.get("pearson_p", 1.0)
    sample_size = same_corr.get("sample_size", len(master_df))
    total_headlines = master_df["Headline_Count"].sum()

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{acc:.1f}%</div>
            <div class="metric-label">Directional Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        sig_str = " (Sig)" if pearson_p < 0.05 else " (Not Sig)"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{pearson_r:+.3f}</div>
            <div class="metric-label">Pearson r {sig_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_headlines}</div>
            <div class="metric-label">Headlines Analyzed</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{sample_size}</div>
            <div class="metric-label">Trading Days Analyzed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Price & Sentiment Timeline", "📊 Statistical Analytics", "📰 News Headline Inspector"])

    # TAB 1: DUAL AXIS TIME SERIES
    with tab1:
        st.subheader("Dual-Axis Time-Series: Stock Price vs Compound Sentiment")
        
        tickers = master_df["Ticker"].unique()
        selected_stock = st.selectbox("Select Stock Ticker to Inspect", options=tickers)

        sub_df = master_df[master_df["Ticker"] == selected_stock].sort_values("Date")

        if not sub_df.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Stock price line
            fig.add_trace(
                go.Scatter(
                    x=sub_df["Date"],
                    y=sub_df["Close_Price"],
                    name="Close Price ($)",
                    line=dict(color="#00e676", width=2.5),
                    mode="lines+markers"
                ),
                secondary_y=False
            )

            # Sentiment bar colors (Green = Pos, Red = Neg, Grey = Neu)
            colors = ["#00e676" if s >= 0.05 else "#ff1744" if s <= -0.05 else "#90a4ae" for s in sub_df["Mean_Sentiment"]]

            fig.add_trace(
                go.Bar(
                    x=sub_df["Date"],
                    y=sub_df["Mean_Sentiment"],
                    name="Mean Sentiment",
                    marker_color=colors,
                    opacity=0.6
                ),
                secondary_y=True
            )

            fig.update_layout(
                template="plotly_dark",
                height=500,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Trading Date"),
                hovermode="x unified"
            )

            fig.update_yaxes(title_text="Stock Close Price ($)", secondary_y=False, showgrid=True, gridcolor="#2e3545")
            fig.update_yaxes(title_text="Compound Sentiment Score", secondary_y=True, range=[-1.0, 1.0], showgrid=False)

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for the selected stock ticker.")

    # TAB 2: STATISTICAL ANALYTICS
    with tab2:
        st.subheader("Scatter Plot & Regression Trendline")

        # Scatter plot sentiment vs returns
        scatter_fig = px.scatter(
            master_df,
            x="Mean_Sentiment",
            y="Daily_Return_Pct",
            color="Ticker",
            hover_data=["Date", "Headline_Count", "Close_Price"],
            trendline="ols",
            title="Daily Compound Sentiment vs Stock Return (%)",
            template="plotly_dark",
            height=450
        )
        scatter_fig.update_layout(
            xaxis=dict(title="Mean Sentiment Score (S_bar)", range=[-1.05, 1.05]),
            yaxis=dict(title="Daily Return (%)")
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Statistical Correlation Breakdown")
        
        per_ticker = stats_summary.get("correlations", {}).get("per_ticker", {})
        if per_ticker:
            per_ticker_df = pd.DataFrame(per_ticker).T
            per_ticker_df.index.name = "Ticker"
            st.dataframe(per_ticker_df.style.highlight_max(axis=0, color="#1b5e20"), use_container_width=True)

    # TAB 3: NEWS HEADLINE INSPECTOR
    with tab3:
        st.subheader("Scraped News Headlines & VADER Sentiment Breakdown")
        if not news_df.empty:
            st.dataframe(news_df, use_container_width=True, height=400)
        else:
            st.info("Raw news data table not currently cached. Run the pipeline to populate.")


if __name__ == "__main__":
    main()
