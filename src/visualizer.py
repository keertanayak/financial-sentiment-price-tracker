"""
Visualizer Module — Generate High-Resolution Publication-Quality Charts

Features:
- Dual-axis price vs. daily sentiment score overlay chart (Close line + Green/Red sentiment bars).
- Pearson & Spearman correlation heatmap across tickers and variables.
- Sentiment vs. Return scatter plot with linear trendline.
- Headline volume distribution bar plot.
- Confusion matrix heatmap for baseline directional logistic regression (Blueprint 8.3).
- Sentiment category donut chart showing Pos/Neg/Neu breakdown (Blueprint 8.4).
- All figures saved to outputs/figures/ at 300 DPI without emojis.
"""

import os
import json
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
    Generate dual-axis plot of Close Price (left axis) and Daily Mean Sentiment (right axis) over time.
    Uses Green bars for Positive sentiment, Red bars for Negative sentiment per Blueprint 8.1.
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

        close_col = "Close_Price" if "Close_Price" in df_t.columns else "Close"

        fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

        # Axis 1: Close Price
        color1 = "#1f77b4"
        ax1.set_xlabel("Date")
        ax1.set_ylabel(f"Close Price ({ticker})", color=color1, fontweight="bold")
        ax1.plot(df_t["Date_dt"], df_t[close_col], color=color1, linewidth=2, label="Close Price", marker="o", markersize=3)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Axis 2: Mean Sentiment as bar stem plot (Green = Pos, Red = Neg, Gray = Neu)
        ax2 = ax1.twinx()
        bar_colors = [
            "#2ca02c" if s >= 0.05 else "#d62728" if s <= -0.05 else "#9e9e9e"
            for s in df_t["Mean_Sentiment"]
        ]
        ax2.set_ylabel("Mean Headline Sentiment (VADER)", color="#2ca02c", fontweight="bold")
        ax2.bar(df_t["Date_dt"], df_t["Mean_Sentiment"], color=bar_colors, alpha=0.6, width=0.8, label="Mean Sentiment")
        ax2.tick_params(axis="y", labelcolor="#2ca02c")
        ax2.set_ylim(-1.05, 1.05)
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

    ret_col = "Daily_Return_Pct" if "Daily_Return_Pct" in master_df.columns else "Return_Pct"
    cols = ["Mean_Sentiment", "Headline_Count", ret_col, "Next_Return_Pct", "Volatility"]
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

    ret_col = "Daily_Return_Pct" if "Daily_Return_Pct" in master_df.columns else "Return_Pct"
    valid = master_df.dropna(subset=["Mean_Sentiment", ret_col])
    if valid.empty:
        return ""

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    sns.regplot(
        data=valid,
        x="Mean_Sentiment",
        y=ret_col,
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


def plot_confusion_matrix_heatmap(stats_summary: dict = None, output_dir: str = "outputs/figures") -> str:
    """
    Generate Confusion Matrix Heatmap for Baseline Directional Logistic Regression (Blueprint Section 8.3).
    Displays True Negatives, False Positives, False Negatives, True Positives.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "confusion_matrix_heatmap.png")

    cm = None
    if stats_summary and isinstance(stats_summary, dict):
        cm = stats_summary.get("logistic_regression", {}).get("confusion_matrix")
    
    if cm is None:
        report_path = "outputs/reports/summary_report.json"
        if os.path.exists(report_path):
            try:
                with open(report_path, "r") as f:
                    rep = json.load(f)
                    cm = rep.get("logistic_regression", {}).get("confusion_matrix")
            except Exception:
                pass

    if cm is None or len(cm) != 2:
        logger.warning("[WARNING] No valid 2x2 confusion matrix found to plot.")
        return ""

    cm_arr = np.array(cm)
    total = cm_arr.sum()
    labels = np.array([
        [f"TN\n{cm_arr[0,0]}\n({cm_arr[0,0]/total*100:.1f}%)", f"FP\n{cm_arr[0,1]}\n({cm_arr[0,1]/total*100:.1f}%)"],
        [f"FN\n{cm_arr[1,0]}\n({cm_arr[1,0]/total*100:.1f}%)", f"TP\n{cm_arr[1,1]}\n({cm_arr[1,1]/total*100:.1f}%)"]
    ])

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm_arr,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Pred Down/Flat (0)", "Pred Up (1)"],
        yticklabels=["Actual Down/Flat (0)", "Actual Up (1)"],
        linewidths=1.5,
        ax=ax
    )

    plt.title("Baseline Logistic Regression: Confusion Matrix Heatmap", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Market Direction", fontweight="bold")
    plt.ylabel("Actual Market Direction", fontweight="bold")
    fig.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

    logger.info(f"[OK] Saved confusion matrix heatmap to {filepath}")
    return filepath


def plot_sentiment_donut_chart(master_df: pd.DataFrame, output_dir: str = "outputs/figures") -> str:
    """
    Generate Sentiment Category Donut Chart showing proportion of Positive vs Negative vs Neutral headlines (Blueprint 8.4).
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "sentiment_category_donut.png")

    pos_total = int(master_df["Pos_Count"].sum()) if "Pos_Count" in master_df.columns else 0
    neg_total = int(master_df["Neg_Count"].sum()) if "Neg_Count" in master_df.columns else 0
    neu_total = int(master_df["Neu_Count"].sum()) if "Neu_Count" in master_df.columns else 0

    # If count is 0 in master, attempt reading from raw_news or sentiment classes
    if (pos_total + neg_total + neu_total) == 0:
        counts = master_df["Dominant_Class"].value_counts()
        pos_total = counts.get("Positive", 0)
        neg_total = counts.get("Negative", 0)
        neu_total = counts.get("Neutral", 0)

    total = pos_total + neg_total + neu_total
    if total == 0:
        logger.warning("[WARNING] No headline sentiments to plot in donut chart.")
        return ""

    sizes = [pos_total, neg_total, neu_total]
    labels = [f"Positive ({pos_total})", f"Negative ({neg_total})", f"Neutral ({neu_total})"]
    colors = ["#2ca02c", "#d62728", "#7f7f7f"]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )

    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    for t in texts:
        t.set_fontweight("bold")

    ax.text(0, 0, f"Total\n{total}", ha="center", va="center", fontsize=13, fontweight="bold", color="#333333")
    plt.title("Headline Sentiment Polarity Distribution (Basket Total)", fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

    logger.info(f"[OK] Saved sentiment donut chart to {filepath}")
    return filepath


def generate_all_plots(master_df: pd.DataFrame, output_dir: str = "outputs/figures", stats_summary: dict = None) -> list:
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

    cm_plot = plot_confusion_matrix_heatmap(stats_summary, output_dir)
    if cm_plot:
        plots.append(cm_plot)

    donut_plot = plot_sentiment_donut_chart(master_df, output_dir)
    if donut_plot:
        plots.append(donut_plot)

    logger.info(f"[OK] Completed visualization generation ({len(plots)} total figures saved).")
    return plots


if __name__ == "__main__":
    logger.info("[INFO] Testing Visualizer module...")
