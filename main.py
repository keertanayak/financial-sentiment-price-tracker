"""
Main Pipeline Orchestrator — End-to-End Sentiment & Market Price Correlation Pipeline

Executes all phases:
1. Ingestion (prices & news) via src.data_fetcher
2. Text Cleaning & Alignment via src.text_processor
3. Sentiment Scoring via src.sentiment_engine
4. Financial Analytics & Merging via src.financial_analytics
5. Empirical Statistical Analysis via src.statistical_analysis
6. Publication Chart Generation via src.visualizer

Usage:
    python main.py --tickers AAPL MSFT NVDA --days-back 30
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

from src.data_fetcher import fetch_market_prices, fetch_news_headlines
from src.text_processor import process_news_dataframe
from src.sentiment_engine import SentimentEngine
from src.financial_analytics import create_master_dataset
from src.statistical_analysis import run_full_statistical_analysis
from src.visualizer import generate_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Financial Sentiment & Market Correlation Computational Pipeline")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "NVDA", "RELIANCE.NS", "TCS.NS"],
        help="List of stock ticker symbols to analyze"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=30,
        help="Number of days to look back for market prices and headlines"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date YYYY-MM-DD"
    )
    return parser.parse_args()


def run_pipeline():
    args = parse_args()

    logger.info("STARTING FINANCIAL SENTIMENT & MARKET CORRELATION PIPELINE")
    logger.info(f"Target Tickers: {args.tickers}")
    logger.info(f"Lookback Window: {args.days_back} days")

    # Phase 1: Data Ingestion
    logger.info("\n[PHASE 1] Fetching raw market prices & news headlines...")
    raw_prices_df = fetch_market_prices(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        days_back=args.days_back
    )
    raw_news_df = fetch_news_headlines(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        days_back=args.days_back
    )

    if raw_prices_df.empty:
        logger.error("[FAILED] Market price collection failed or returned empty dataframe.")
        sys.exit(1)

    # Phase 2: Text Cleaning & Time Alignment
    logger.info("\n[PHASE 2] Cleaning headlines & performing market time alignment...")
    processed_news_df = process_news_dataframe(raw_news_df)

    # Phase 3: Sentiment Scoring
    logger.info("\n[PHASE 3] Scoring headline sentiment via NLTK VADER...")
    engine = SentimentEngine()
    scored_news_df = engine.score_news_dataframe(processed_news_df)
    daily_sentiment_df = engine.aggregate_daily_sentiment(scored_news_df)

    # Phase 4: Financial Feature Engineering & Master Table Construction
    logger.info("\n[PHASE 4] Computing financial metrics & creating master dataset...")
    master_df = create_master_dataset(
        prices_df=raw_prices_df,
        daily_sentiment_df=daily_sentiment_df,
        output_path="data/processed/master_sentiment_prices.csv"
    )

    if master_df.empty:
        logger.error("[FAILED] Master dataset creation failed.")
        sys.exit(1)

    # Phase 5: Empirical Statistical Analysis
    logger.info("\n[PHASE 5] Executing empirical statistical analysis...")
    stats_summary = run_full_statistical_analysis(master_df)

    # Save summary report JSON
    os.makedirs("outputs/reports", exist_ok=True)
    report_path = "outputs/reports/summary_report.json"
    with open(report_path, "w") as f:
        json.dump(stats_summary, f, indent=2)
    logger.info(f"[OK] Saved statistical analysis summary to {report_path}")

    # Phase 6: Visualization Chart Generation
    logger.info("\n[PHASE 6] Generating publication figures...")
    plots = generate_all_plots(master_df, output_dir="outputs/figures")

    # Final Pipeline Summary
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"Master Dataset Saved : data/processed/master_sentiment_prices.csv ({len(master_df)} rows)")
    logger.info(f"Summary Report Saved : {report_path}")
    logger.info(f"Figures Generated    : {len(plots)} charts in outputs/figures/")

    pooled_same = stats_summary.get("correlations", {}).get("pooled_same_day", {})
    logger.info("\nKey Statistical Finding (Same-Day Sentiment vs Stock Return):")
    logger.info(f"  Pearson Correlation r  : {pooled_same.get('pearson_r')} (p-val: {pooled_same.get('pearson_p')})")
    logger.info(f"  Spearman Correlation r : {pooled_same.get('spearman_r')} (p-val: {pooled_same.get('spearman_p')})")
    logger.info(f"  Directional Accuracy   : {stats_summary.get('directional_accuracy', {}).get('overall_directional_acc_pct')}%")


if __name__ == "__main__":
    run_pipeline()
