"""
Financial Analytics Module — Return Calculation & Master Dataset Alignment

Features:
- Compute Daily Log Return R_t = ln(P_t / P_{t-1}).
- Compute Daily Percentage Return Delta P_t = (P_t - P_{t-1}) / P_{t-1} * 100.
- Compute Binary Directional Target Y_t = 1 if Delta P_t > 0 else 0.
- Compute Intraday Volatility sigma_t = (High - Low) / Low.
- Merge aggregated daily sentiment scores with price features into data/processed/master_sentiment_prices.csv.
- Zero-News-Day imputation for dates with market prices but no news headlines.
"""

import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def compute_financial_metrics(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily financial metrics per ticker:
    - Log Return (Return_Log)
    - Pct Return (Return_Pct)
    - Next Day Log Return (Next_Return_Log) — shift -1 for predictive correlation
    - Price Direction (Price_Direction: 1 for up, 0 for down/flat)
    - Intraday Volatility (Volatility)
    """
    if prices_df.empty:
        logger.warning("[WARNING] Price dataframe is empty.")
        return prices_df

    df = prices_df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)

    calculated_dfs = []
    for ticker, group in df.groupby("Ticker"):
        g = group.copy()

        # Prices
        close = g["Close"].astype(float)
        high = g["High"].astype(float)
        low = g["Low"].astype(float)

        # Log return R_t
        g["Return_Log"] = np.log(close / close.shift(1))

        # Percentage return Delta P_t
        g["Return_Pct"] = ((close - close.shift(1)) / close.shift(1)) * 100.0

        # Next-day return R_{t+1} for forward-looking correlation
        g["Next_Return_Log"] = g["Return_Log"].shift(-1)
        g["Next_Return_Pct"] = g["Return_Pct"].shift(-1)

        # Binary Direction Y_t: 1 if positive return, 0 otherwise
        g["Price_Direction"] = (g["Return_Pct"] > 0).astype(int)

        # Intraday Volatility sigma_t = (High - Low) / Low
        g["Volatility"] = np.where(low > 0, (high - low) / low, 0.0)

        # Drop first row per ticker (since t-1 return cannot be computed)
        g_clean = g.dropna(subset=["Return_Log"]).copy()
        calculated_dfs.append(g_clean)

    if not calculated_dfs:
        return pd.DataFrame()

    result_df = pd.concat(calculated_dfs, ignore_index=True)
    logger.info(f"[OK] Calculated financial metrics for {len(result_df)} trading rows.")
    return result_df


def create_master_dataset(
    prices_df: pd.DataFrame,
    daily_sentiment_df: pd.DataFrame,
    output_path: str = "data/processed/master_sentiment_prices.csv"
) -> pd.DataFrame:
    """
    Merge daily price metrics with daily aggregated sentiment scores.
    Imputes zero-news days with Mean_Sentiment = 0.0 and No_News = True.
    """
    fin_df = compute_financial_metrics(prices_df)

    if fin_df.empty:
        logger.error("[ERROR] Financial metrics calculation returned empty dataset.")
        return pd.DataFrame()

    # Align column names for merging
    if "Trading_Day" in daily_sentiment_df.columns:
        sent_df = daily_sentiment_df.rename(columns={"Trading_Day": "Date"}).copy()
    else:
        sent_df = daily_sentiment_df.copy()

    if not sent_df.empty and "Date" in sent_df.columns:
        sent_df["Date"] = pd.to_datetime(sent_df["Date"]).dt.strftime("%Y-%m-%d")

    # Left join prices with sentiment so every trading day is preserved
    master = pd.merge(
        fin_df,
        sent_df,
        on=["Ticker", "Date"],
        how="left"
    )

    # Impute Zero-News Days
    master["Headline_Count"] = master["Headline_Count"].fillna(0).astype(int)
    master["Mean_Sentiment"] = master["Mean_Sentiment"].fillna(0.0)
    master["Pos_Count"] = master["Pos_Count"].fillna(0).astype(int)
    master["Neg_Count"] = master["Neg_Count"].fillna(0).astype(int)
    master["Neu_Count"] = master["Neu_Count"].fillna(0).astype(int)
    master["Dominant_Class"] = master["Dominant_Class"].fillna("Neutral")
    master["No_News"] = master["No_News"].fillna(True).astype(bool)

    # Standardized alias columns for downstream compatibility (spec vs convention)
    master["Close_Price"] = master["Close"]
    master["Daily_Return_Pct"] = master["Return_Pct"]

    # Reorder columns standardly
    cols = [
        "Ticker", "Date", "Close", "Close_Price", "Return_Pct", "Daily_Return_Pct", "Return_Log",
        "Next_Return_Pct", "Next_Return_Log", "Price_Direction", "Volatility",
        "Headline_Count", "Mean_Sentiment", "Pos_Count", "Neg_Count", "Neu_Count",
        "Dominant_Class", "No_News"
    ]
    cols = [c for c in cols if c in master.columns]
    master = master[cols].sort_values(by=["Ticker", "Date"]).reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        master.to_csv(output_path, index=False)
        logger.info(f"[OK] Saved master sentiment-price dataset to {output_path} ({len(master)} rows)")

    return master


if __name__ == "__main__":
    logger.info("[INFO] Testing Financial Analytics module...")
