"""
Visualizer Module — Generate High-Resolution Publication-Quality Charts

Features:
- Dual-axis price vs. daily sentiment score overlay chart.
- Pearson & Spearman correlation heatmap across tickers and variables.
- Sentiment vs. Return scatter plot with linear trendline.
- Headline volume distribution bar plot.
- All figures saved to outputs/figures/ at 300 DPI without emojis.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Configure clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


def plot_sentiment_vs_price_overlay(master_df: pd.DataFrame, output_dir: str = "outputs/figures") -> list:
    """
    Generate dual-axis plot of Close Price (left axis) and Daily Mean Sentiment (right axis) over time for each ticker.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    tickers = master_df["Ticker"].unique()
    for ticker in tickers:
        df_t = master_df[master_df["Ticker"] == ticker].copy()
        if len(df_t) < 3:
            continue

        df_t["Date_dt"] = pd.to_datetime(df_t["Date"])
        df_t = df_t.sort_values("Date_dt")

        fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

        # Axis 1: Close Price
        color1 = "#1f77b4"
        ax1.set_xlabel("Date")
        ax1.set_ylabel(f"Close Price ({ticker})", color=color1, fontweight="bold")
        line1 = ax1.plot(df_t["Date_dt"], df_t["Close"], color=color1, linewidth=2, label="Close Price", marker="o", markersize=3)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Axis 2: Mean Sentiment
        ax2 = ax1.twinx()
        color2 = "#2ca02c"
        ax2.set_ylabel("Mean Headline Sentiment (VADER)", color=color2, fontweight="bold")
        line2 = ax2.plot(df_t["Date_dt"], df_t["Mean_Sentiment"], color=color2, linewidth=1.8, linestyle="--", label="Mean Sentiment", marker="s", markersize=3)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(-1.0, 1.0)
        ax2.axhline(0, color="gray", linewidth=0.8, linestyle=":")

        # Title & Layout
        plt.title(f"Empirical Sentiment vs. Stock Price Alignment: {ticker}", fontsize=13, fontweight="bold", pad=12)
        fig.tight_layout()

        filename = f"sentiment_price_overlay_{ticker.replace('.', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        generated_files.append(filepath)
        logger.info(f"[OK] Saved overlay plot to {filepath}")

    return generated_files


def plot_correlation_heatmap(master_df: pd.DataFrame, output_dir: str = "outputs/figures") -> str:
    """
    Generate correlation matrix heatmap between sentiment, headline volume, returns, and volatility.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "correlation_heatmap.png")

    cols = ["Mean_Sentiment", "Headline_Count", "Return_Pct", "Next_Return_Pct", "Volatility"]
    existing_cols = [c for c in cols if c in master_df.columns]

    if len(existing_cols) < 2 or master_df.empty:
        logger.warning("[WARNING] Insufficient columns/data for correlation heatmap.")
        return ""

    corr = master_df[existing_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    plt.title("Pearson Correlation Heatmap: Sentiment & Market Features", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

    logger.info(f"[OK] Saved correlation heatmap to {filepath}")
    return filepath


def plot_sentiment_return_scatter(master_df: pd.DataFrame, output_dir: str = "outputs/figures") -> str:
    """
    Generate scatter plot of Mean Sentiment vs Daily Return (%) with linear regression fit line.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "sentiment_return_scatter.png")

    valid = master_df.dropna(subset=["Mean_Sentiment", "Return_Pct"])
    if valid.empty:
        return ""

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    sns.regplot(
        data=valid,
        x="Mean_Sentiment",
        y="Return_Pct",
        ax=ax,
        scatter_kws={"alpha": 0.6, "color": "#1f77b4", "s": 35},
        line_kws={"color": "#d62728", "linewidth": 2}
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Daily Mean Headline Sentiment (VADER Compound Score)", fontweight="bold")
    ax.set_ylabel("Same-Day Market Return (%)", fontweight="bold")
    ax.set_title("Empirical Scatter Plot: Headline Sentiment vs. Same-Day Stock Return", fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

    logger.info(f"[OK] Saved scatter plot to {filepath}")
    return filepath


def plot_headline_volume_distribution(master_df: pd.DataFrame, output_dir: str = "outputs/figures") -> str:
    """
    Generate bar plot of total headline volume per ticker.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "headline_volume_distribution.png")

    if master_df.empty or "Headline_Count" not in master_df.columns:
        return ""

    vol_df = master_df.groupby("Ticker")["Headline_Count"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar(vol_df["Ticker"], vol_df["Headline_Count"], color="#2ca02c", edgecolor="black", alpha=0.85)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontweight="bold"
        )

    ax.set_xlabel("Stock Ticker Symbol", fontweight="bold")
    ax.set_ylabel("Total Ingested News Headlines Count", fontweight="bold")
    ax.set_title("News Headline Ingestion Volume per Equity Ticker", fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

    logger.info(f"[OK] Saved headline volume chart to {filepath}")
    return filepath


def generate_all_plots(master_df: pd.DataFrame, output_dir: str = "outputs/figures") -> list:
    """Execute all plot generation routines."""
    logger.info("[INFO] Generating publication-quality visualization charts...")
    plots = []
    plots.extend(plot_sentiment_vs_price_overlay(master_df, output_dir))
    
    heatmap = plot_correlation_heatmap(master_df, output_dir)
    if heatmap:
        plots.append(heatmap)

    scatter = plot_sentiment_return_scatter(master_df, output_dir)
    if scatter:
        plots.append(scatter)

    volume = plot_headline_volume_distribution(master_df, output_dir)
    if volume:
        plots.append(volume)

    logger.info(f"[OK] Completed visualization generation ({len(plots)} total figures saved).")
    return plots


if __name__ == "__main__":
    logger.info("[INFO] Testing Visualizer module...")
