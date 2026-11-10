"""
Streamlit Web Dashboard — Empirical Financial Sentiment & Market Correlation Pipeline

Redesigned with custom dark glassmorphism styling, refined typography,
glowing KPI cards, and custom Plotly visual themes.
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
    page_title="AlphaSentiment | Market Correlation Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — Glassmorphism, Google Fonts, glowing cards & modern typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 15% 15%, #111827 0%, #030712 100%);
        color: #f3f4f6;
    }
    
    /* Header Gradient */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-bottom: 25px;
        font-weight: 400;
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 14px 40px -10px rgba(56, 189, 248, 0.15);
    }
    
    .kpi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #9CA3AF;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .kpi-value {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 8px 0;
        letter-spacing: -0.03em;
    }
    
    .kpi-badge {
        display: inline-block;
        font-size: 0.75rem;
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
    
    .badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-blue {
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    .badge-purple {
        background: rgba(168, 85, 247, 0.15);
        color: #C084FC;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    /* Streamlit Tab Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(17, 24, 39, 0.5);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre;
        border-radius: 8px;
        color: #9CA3AF;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1F2937 !important;
        color: #38BDF8 !important;
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
    with st.spinner("⚡ Connecting to market feeds & fetching news..."):
        prices_df = fetch_market_prices(tickers=tickers, days_back=days_back)
        news_df = fetch_news_headlines(tickers=tickers, days_back=days_back)

    if prices_df.empty:
        st.error("Market price collection returned an empty dataset. Please check tickers.")
        return pd.DataFrame(), pd.DataFrame(), {}

    with st.spinner("🧠 Processing headlines & executing market timestamp shift rules..."):
        processed_news = process_news_dataframe(news_df)

    with st.spinner("🔮 Computing VADER sentiment polarity index..."):
        engine = SentimentEngine()
        scored_news = engine.score_news_dataframe(processed_news)
        daily_sentiment = engine.aggregate_daily_sentiment(scored_news)

    with st.spinner("📊 Building master financial feature dataset..."):
        master_df = create_master_dataset(
            prices_df=prices_df,
            daily_sentiment_df=daily_sentiment,
            output_path="data/processed/master_sentiment_prices.csv"
        )

    with st.spinner("📐 Calculating statistical correlations & regressions..."):
        stats_summary = run_full_statistical_analysis(master_df)
        os.makedirs("outputs/reports", exist_ok=True)
        with open("outputs/reports/summary_report.json", "w") as f:
            json.dump(stats_summary, f, indent=2)

    return master_df, news_df, stats_summary


def main():
    # Hero Header
    st.markdown('<h1 class="hero-title">AlphaSentiment Lab</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Empirical market sentiment analytics & daily equity price return correlation pipeline.</p>', unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.markdown("### 🎛️ Parameters")
    
    default_tickers = ["AAPL", "MSFT", "NVDA", "RELIANCE.NS", "TCS.NS"]
    selected_tickers = st.sidebar.multiselect(
        "Equity Basket Tickers",
        options=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "RELIANCE.NS", "TCS.NS", "INFY.NS"],
        default=default_tickers
    )

    custom_ticker = st.sidebar.text_input("Add Ticker (e.g. AMD, TSLA)", value="")
    if custom_ticker and custom_ticker not in selected_tickers:
        selected_tickers.append(custom_ticker.strip().upper())

    days_back = st.sidebar.slider("Lookback Window (Days)", min_value=7, max_value=90, value=30, step=1)
    
    run_btn = st.sidebar.button("🚀 Re-Run Empirical Pipeline", type="primary", use_container_width=True)

    # Data Load
    if run_btn:
        master_df, news_df, stats_summary = run_pipeline_flow(selected_tickers, days_back)
    else:
        master_df, news_df, stats_summary = load_existing_data()

    if master_df.empty:
        st.info("No active dataset. Adjust parameters in sidebar and click **Re-Run Empirical Pipeline**.")
        return

    # Extract Stats Metrics
    acc = stats_summary.get("directional_accuracy", {}).get("overall_directional_acc_pct", 0)
    same_corr = stats_summary.get("correlations", {}).get("pooled_same_day", {})
    pearson_r = same_corr.get("pearson_r", 0)
    pearson_p = same_corr.get("pearson_p", 1.0)
    spearman_r = same_corr.get("spearman_r", 0)
    sample_size = same_corr.get("sample_size", len(master_df))
    total_headlines = master_df["Headline_Count"].sum()

    # Glass KPI Cards Row
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-header">
                <span>Directional Accuracy</span>
                <span>🎯</span>
            </div>
            <div class="kpi-value" style="color: #34D399;">{acc:.1f}%</div>
            <span class="kpi-badge badge-green">Up/Down Match Rate</span>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        sig_text = "p < 0.05 (Sig)" if pearson_p < 0.05 else f"p={pearson_p:.2f} (Not Sig)"
        color_hex = "#38BDF8" if pearson_r >= 0 else "#F43F5E"
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-header">
                <span>Pearson Correlation</span>
                <span>🔗</span>
            </div>
            <div class="kpi-value" style="color: {color_hex};">{pearson_r:+.3f}</div>
            <span class="kpi-badge badge-blue">{sig_text}</span>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-header">
                <span>News Volume Scraped</span>
                <span>📰</span>
            </div>
            <div class="kpi-value" style="color: #C084FC;">{total_headlines}</div>
            <span class="kpi-badge badge-purple">VADER Scored Headlines</span>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="glass-card">
            <div class="kpi-header">
                <span>Sample Trading Days</span>
                <span>📅</span>
            </div>
            <div class="kpi-value" style="color: #FBBF24;">{sample_size}</div>
            <span class="kpi-badge badge-blue">Observations</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Price & Sentiment Story", "📊 Quantitative Correlations", "📰 Headline Feed Inspector"])

    # TAB 1: DUAL AXIS TIME SERIES
    with tab1:
        col_select, col_info = st.columns([1, 2])
        with col_select:
            selected_stock = st.selectbox("Select Equity Ticker to Inspect", options=master_df["Ticker"].unique())

        sub_df = master_df[master_df["Ticker"] == selected_stock].sort_values("Date")

        if not sub_df.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            close_col = "Close_Price" if "Close_Price" in sub_df.columns else "Close"
            # Closing price smooth spline
            fig.add_trace(
                go.Scatter(
                    x=sub_df["Date"],
                    y=sub_df[close_col],
                    name="Close Price ($)",
                    line=dict(color="#38BDF8", width=3, shape="spline"),
                    mode="lines+markers",
                    marker=dict(size=6, color="#38BDF8", symbol="circle")
                ),
                secondary_y=False
            )

            # Sentiment bars (Emerald Pos, Rose Neg, Slate Neu)
            colors = [
                "#10B981" if s >= 0.05 else "#F43F5E" if s <= -0.05 else "#64748B"
                for s in sub_df["Mean_Sentiment"]
            ]

            fig.add_trace(
                go.Bar(
                    x=sub_df["Date"],
                    y=sub_df["Mean_Sentiment"],
                    name="Mean Sentiment (S_bar)",
                    marker_color=colors,
                    opacity=0.65,
                    marker=dict(line=dict(width=0))
                ),
                secondary_y=True
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(11, 15, 25, 0.7)",
                height=520,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
                xaxis=dict(
                    title="Trading Date",
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.05)",
                    zerolinecolor="rgba(255, 255, 255, 0.1)"
                ),
                hovermode="x unified"
            )

            fig.update_yaxes(
                title_text="Closing Price ($)",
                secondary_y=False,
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.05)"
            )
            fig.update_yaxes(
                title_text="Compound Sentiment Score",
                secondary_y=True,
                range=[-1.05, 1.05],
                showgrid=False
            )

            st.plotly_chart(fig, use_container_width=True)

    # TAB 2: QUANTITATIVE CORRELATIONS
    with tab2:
        st.markdown("### Scatter Regression Analysis")
        
        return_col = "Daily_Return_Pct" if "Daily_Return_Pct" in master_df.columns else "Return_Pct"
        close_col = "Close_Price" if "Close_Price" in master_df.columns else "Close"
        hover_items = ["Date", "Headline_Count"]
        if close_col in master_df.columns:
            hover_items.append(close_col)

        scatter_fig = px.scatter(
            master_df,
            x="Mean_Sentiment",
            y=return_col,
            color="Ticker",
            hover_data=hover_items,
            trendline="ols",
            color_discrete_sequence=["#38BDF8", "#34D399", "#C084FC", "#FBBF24", "#F43F5E"],
            template="plotly_dark",
            height=460
        )
        scatter_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11, 15, 25, 0.7)",
            xaxis=dict(title="Daily Mean Sentiment Score (S_bar)", range=[-1.05, 1.05], gridcolor="rgba(255, 255, 255, 0.05)"),
            yaxis=dict(title="Daily Price Return (%)", gridcolor="rgba(255, 255, 255, 0.05)")
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Per-Ticker Correlation Matrix")
        
        per_ticker = stats_summary.get("correlations", {}).get("per_ticker", {})
        if per_ticker:
            per_ticker_df = pd.DataFrame(per_ticker).T
            per_ticker_df.index.name = "Ticker Symbol"
            st.dataframe(
                per_ticker_df.style.format("{:.4f}").highlight_max(axis=0, color="rgba(16, 185, 129, 0.3)"),
                use_container_width=True
            )

    # TAB 3: NEWS FEED INSPECTOR
    with tab3:
        st.markdown("### Scraped News Articles & VADER Polarities")
        if not news_df.empty:
            st.dataframe(
                news_df,
                use_container_width=True,
                height=420
            )
        else:
            st.info("Raw headline feed table not cached. Click **Re-Run Empirical Pipeline** to populate.")


if __name__ == "__main__":
    main()
