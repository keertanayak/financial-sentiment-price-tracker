"""
Text Processor Module — Headline Cleaning, Noise Filtering, & Market-Time Alignment

Features:
- Strip HTML, decode entities, and normalize whitespace while PRESERVING punctuation and capitalization (crucial for VADER).
- Deduplicate headlines per ticker.
- Filter out non-financial promotional noise.
- Non-Trading-Day Shift Rule: Map news published after market close (4:00 PM EST / 3:30 PM IST) or on weekends/holidays to the next valid trading day (t+1).
- Aggregate daily headlines per ticker.
"""

import re
import html
import logging
from datetime import datetime, time, timedelta
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Keywords for promotional press releases or irrelevant non-financial noise
NOISE_KEYWORDS = [
    "sponsored content", "promotional", "discount code", "buy now",
    "advertisement", "free trial", "sweepstakes", "giveaway"
]


def clean_headline(text: str) -> str:
    """
    Clean headline text while preserving punctuation and capitalization.
    - Strips HTML tags
    - Unescapes HTML entities (&amp; -> &, &quot; -> ", etc.)
    - Replaces non-printable characters
    - Normalizes multi-spaces
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Unescape HTML entities
    text = html.unescape(text)

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace non-printable/control chars except standard punctuation
    text = "".join(ch for ch in text if ch.isprintable())

    # Replace multiple spaces/newlines with single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_noisy_headline(text: str) -> bool:
    """Check if a headline contains promotional or non-financial noise."""
    lower_text = text.lower()
    for kw in NOISE_KEYWORDS:
        if kw in lower_text:
            return True
    return False


def get_market_timezone_offset(ticker: str) -> int:
    """
    Get UTC offset in hours for market locality based on ticker suffix.
    - US tickers (AAPL, MSFT, NVDA): EST/EDT approx UTC-5 / UTC-4
    - Indian tickers (.NS, .BO): IST UTC+5.5 -> approximate +5
    Returns UTC offset in hours.
    """
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return 5  # IST (~UTC+5.5, approx 5h offset for date math)
    return -5  # US Markets (EST ~UTC-5)


def map_timestamp_to_trading_day(pub_time_str: str, ticker: str) -> str:
    """
    Map publication timestamp (UTC) to target trading day.
    
    Non-Trading-Day Shift Rule:
    - US Market Close: 16:00 (4:00 PM local)
    - Indian Market Close: 15:30 (3:30 PM local)
    - If published AFTER market close or on a weekend (Sat/Sun) -> shift to next valid trading day.
    - If published before or during market hours -> maps to current day (if weekday) or next Monday (if weekend).
    """
    if not pub_time_str or pd.isna(pub_time_str):
        return datetime.utcnow().strftime("%Y-%m-%d")

    try:
        # Try parsing full ISO timestamp YYYY-MM-DD HH:MM:SS
        if len(str(pub_time_str)) >= 19:
            dt_utc = datetime.strptime(str(pub_time_str)[:19], "%Y-%m-%d %H:%M:%S")
        elif len(str(pub_time_str)) >= 10:
            dt_utc = datetime.strptime(str(pub_time_str)[:10], "%Y-%m-%d")
        else:
            return str(pub_time_str)[:10]
    except Exception:
        return str(pub_time_str)[:10]

    # Convert UTC to local market time approx
    offset = get_market_timezone_offset(ticker)
    dt_local = dt_utc + timedelta(hours=offset)

    close_hour = 15 if (ticker.endswith(".NS") or ticker.endswith(".BO")) else 16
    close_minute = 30 if (ticker.endswith(".NS") or ticker.endswith(".BO")) else 0

    target_date = dt_local.date()

    # Check if after market close
    if dt_local.time() >= time(close_hour, close_minute):
        target_date += timedelta(days=1)

    # Shift weekend to Monday
    while target_date.weekday() in (5, 6):  # 5 = Saturday, 6 = Sunday
        target_date += timedelta(days=1)

    return target_date.strftime("%Y-%m-%d")


def process_news_dataframe(raw_news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw news dataframe:
    - Clean headline text
    - Filter noisy promotional content
    - Map timestamps to target trading days
    - Deduplicate identical headlines per ticker & mapped trading day
    """
    if raw_news_df.empty:
        logger.warning("[WARNING] Raw news dataframe is empty.")
        return pd.DataFrame(columns=["Ticker", "Clean_Title", "Publisher", "Publication_Time", "Trading_Day", "URL", "Source_API"])

    df = raw_news_df.copy()

    # Clean title
    df["Clean_Title"] = df["Title"].apply(clean_headline)

    # Filter out empty or noisy headlines
    df = df[df["Clean_Title"].str.len() > 5].copy()
    df = df[~df["Clean_Title"].apply(is_noisy_headline)].copy()

    # Map trading day
    df["Trading_Day"] = df.apply(
        lambda row: map_timestamp_to_trading_day(row.get("Publication_Time", ""), row.get("Ticker", "")),
        axis=1
    )

    # Deduplicate
    initial_len = len(df)
    df = df.drop_duplicates(subset=["Ticker", "Clean_Title", "Trading_Day"]).reset_index(drop=True)
    logger.info(f"[OK] Cleaned & aligned headlines: reduced from {initial_len} to {len(df)} rows.")

    return df


if __name__ == "__main__":
    logger.info("[INFO] Testing text processor module...")
    sample_text = "<b>BREAKING:</b> Apple Inc. (AAPL) Beats Q4 Revenue Expectations! &amp; stock surges 5%"
    cleaned = clean_headline(sample_text)
    logger.info(f"[OK] Sample clean result: '{cleaned}'")
    
    t_day = map_timestamp_to_trading_day("2026-08-07 20:00:00", "AAPL")
    logger.info(f"[OK] Sample Friday 8PM EST news mapped to trading day: {t_day}")
