"""
Data Fetcher Module — Ingest Market Prices & Financial News Headlines

Features:
- Ingest OHLCV + Adjusted Close via yfinance for configurable tickers.
- Ingest news headlines per ticker from NewsAPI / GNews (if API key set) with yfinance fallback.
- Exponential backoff retry logic for external API resilience.
- Save raw data to data/raw/raw_prices.csv and data/raw/raw_news.csv.
"""

import os
import sys
import time
import logging
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime, timedelta
import pandas as pd
import requests
import yfinance as yf

# Try loading .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Setup logging without emojis
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "RELIANCE.NS", "TCS.NS"]
DEFAULT_DAYS_BACK = 30


def retry_with_backoff(func: Callable, max_retries: int = 3, backoff_factor: float = 1.5, *args, **kwargs) -> Any:
    """
    Execute a function with exponential backoff retry.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"[ERROR] Function {func.__name__} failed after {max_retries} attempts: {e}")
                raise e
            sleep_time = backoff_factor ** attempt
            logger.warning(f"[WARNING] Attempt {attempt} failed for {func.__name__}: {e}. Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)


def fetch_market_prices(
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days_back: int = DEFAULT_DAYS_BACK,
    output_path: Optional[str] = "data/raw/raw_prices.csv"
) -> pd.DataFrame:
    """
    Fetch daily stock price history (OHLCV + Adj Close) for given tickers via yfinance.
    
    Returns DataFrame with columns:
    [Date, Ticker, Open, High, Low, Close, Adj_Close, Volume]
    """
    tickers = tickers or DEFAULT_TICKERS
    if not end_date:
        end_dt = datetime.utcnow()
        end_date = end_dt.strftime("%Y-%m-%d")
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if not start_date:
        start_dt = end_dt - timedelta(days=days_back)
        start_date = start_dt.strftime("%Y-%m-%d")

    logger.info(f"[INFO] Fetching market prices for {tickers} from {start_date} to {end_date}...")

    all_records = []
    for ticker in tickers:
        def _download():
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval="1d", auto_adjust=False)
            return df

        try:
            df = retry_with_backoff(_download, max_retries=3)
            if df.empty:
                logger.warning(f"[WARNING] No price data returned for {ticker}")
                continue

            df = df.reset_index()
            # Normalize Date column
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            
            df["Ticker"] = ticker
            
            # Map column names standardly
            rename_map = {
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Adj Close": "Adj_Close",
                "Volume": "Volume"
            }
            for col in rename_map:
                if col not in df.columns and col.lower() in df.columns:
                    rename_map[col.lower()] = rename_map[col]

            if "Adj Close" not in df.columns and "Close" in df.columns:
                df["Adj_Close"] = df["Close"]

            df = df.rename(columns={"Adj Close": "Adj_Close"})
            
            keep_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj_Close", "Volume"]
            existing_cols = [c for c in keep_cols if c in df.columns]
            df = df[existing_cols]

            all_records.append(df)
            logger.info(f"[OK] Fetched {len(df)} price rows for {ticker}")

        except Exception as e:
            logger.error(f"[ERROR] Failed to fetch prices for {ticker}: {e}")

    if not all_records:
        logger.warning("[WARNING] No price data collected across all tickers!")
        master_df = pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Adj_Close", "Volume"])
    else:
        master_df = pd.concat(all_records, ignore_index=True)
        master_df = master_df.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        master_df.to_csv(output_path, index=False)
        logger.info(f"[OK] Saved market prices to {output_path} ({len(master_df)} total records)")

    return master_df


def fetch_news_newsapi(tickers: list, start_date: str, end_date: str, api_key: str) -> list:
    """Fetch news articles using NewsAPI.org."""
    articles_list = []
    url = "https://newsapi.org/v2/everything"

    for ticker in tickers:
        logger.info(f"[INFO] NewsAPI: Searching headlines for {ticker}...")
        params = {
            "q": ticker,
            "from": start_date,
            "to": end_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
            "apiKey": api_key,
        }

        def _request():
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()

        try:
            data = retry_with_backoff(_request, max_retries=3)
            if data.get("status") == "ok":
                items = data.get("articles", [])
                logger.info(f"[OK] NewsAPI returned {len(items)} articles for {ticker}")
                for item in items:
                    pub_raw = item.get("publishedAt", "")
                    pub_ts = pub_raw.replace("Z", "").replace("T", " ")[:19] if pub_raw else ""
                    articles_list.append({
                        "Ticker": ticker,
                        "Title": item.get("title", ""),
                        "Publisher": item.get("source", {}).get("name", "Unknown"),
                        "Publication_Time": pub_ts,
                        "URL": item.get("url", ""),
                        "Source_API": "NewsAPI"
                    })
            else:
                logger.warning(f"[WARNING] NewsAPI error response for {ticker}: {data.get('message')}")
        except Exception as e:
            logger.error(f"[ERROR] NewsAPI request failed for {ticker}: {e}")

    return articles_list


def fetch_news_gnews(tickers: list, start_date: str, end_date: str, api_key: str) -> list:
    """Fetch news articles using GNews API."""
    articles_list = []
    url = "https://gnews.io/api/v4/search"

    from_iso = f"{start_date}T00:00:00Z"
    to_iso = f"{end_date}T23:59:59Z"

    for ticker in tickers:
        logger.info(f"[INFO] GNews: Searching headlines for {ticker}...")
        params = {
            "q": ticker,
            "lang": "en",
            "from": from_iso,
            "to": to_iso,
            "max": 100,
            "token": api_key,
        }

        def _request():
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()

        try:
            data = retry_with_backoff(_request, max_retries=3)
            items = data.get("articles", [])
            logger.info(f"[OK] GNews returned {len(items)} articles for {ticker}")
            for item in items:
                pub_raw = item.get("publishedAt", "")
                pub_ts = pub_raw.replace("Z", "").replace("T", " ")[:19] if pub_raw else ""
                articles_list.append({
                    "Ticker": ticker,
                    "Title": item.get("title", ""),
                    "Publisher": item.get("source", {}).get("name", "Unknown"),
                    "Publication_Time": pub_ts,
                    "URL": item.get("url", ""),
                    "Source_API": "GNews"
                })
        except Exception as e:
            logger.error(f"[ERROR] GNews request failed for {ticker}: {e}")

    return articles_list


def fetch_news_yfinance(tickers: list) -> list:
    """Fallback: Fetch news feed using yfinance."""
    articles_list = []
    for ticker in tickers:
        logger.info(f"[INFO] yfinance fallback: Fetching news for {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            news = stock.news or []
            logger.info(f"[OK] yfinance returned {len(news)} articles for {ticker}")

            for item in news:
                if not isinstance(item, dict):
                    continue

                title = ""
                publisher = "Yahoo Finance"
                pub_ts = ""
                url_val = ""

                content = item.get("content", item)
                if isinstance(content, dict):
                    title = content.get("title", "")
                    url_val = content.get("canonicalUrl", {}).get("url", "") if isinstance(content.get("canonicalUrl"), dict) else content.get("url", "")
                    pub_date = content.get("pubDate", "")
                    if pub_date:
                        pub_ts = pub_date.replace("Z", "").replace("T", " ")[:19]
                    provider = content.get("provider", {})
                    if isinstance(provider, dict):
                        publisher = provider.get("displayName", publisher)

                if not title:
                    title = item.get("title", "")
                if not url_val:
                    url_val = item.get("link", "")
                if not pub_ts:
                    epoch = item.get("providerPublishTime", None)
                    if epoch:
                        pub_ts = datetime.utcfromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")

                if not pub_ts:
                    pub_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

                if title:
                    articles_list.append({
                        "Ticker": ticker,
                        "Title": title,
                        "Publisher": publisher,
                        "Publication_Time": pub_ts,
                        "URL": url_val,
                        "Source_API": "yfinance"
                    })
        except Exception as e:
            logger.error(f"[ERROR] yfinance news fetch failed for {ticker}: {e}")

    return articles_list


def fetch_news_headlines(
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days_back: int = DEFAULT_DAYS_BACK,
    output_path: Optional[str] = "data/raw/raw_news.csv"
) -> pd.DataFrame:
    """
    Fetch news headlines per ticker using available primary API (NewsAPI/GNews) with yfinance fallback.
    
    Returns DataFrame with columns:
    [Ticker, Title, Publisher, Publication_Time, URL, Source_API]
    """
    tickers = tickers or DEFAULT_TICKERS
    if not end_date:
        end_dt = datetime.utcnow()
        end_date = end_dt.strftime("%Y-%m-%d")
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if not start_date:
        start_dt = end_dt - timedelta(days=days_back)
        start_date = start_dt.strftime("%Y-%m-%d")

    news_api_key = os.environ.get("NEWS_API_KEY", "")
    gnews_api_key = os.environ.get("GNEWS_API_KEY", "")

    articles = []
    if news_api_key:
        logger.info("[INFO] Using NewsAPI.org as primary news source...")
        articles = fetch_news_newsapi(tickers, start_date, end_date, news_api_key)
    elif gnews_api_key:
        logger.info("[INFO] Using GNews API as primary news source...")
        articles = fetch_news_gnews(tickers, start_date, end_date, gnews_api_key)
    
    # If primary API yielded few or no articles, supplement/fallback with yfinance
    if len(articles) < len(tickers) * 3:
        logger.info("[INFO] Primary news API key not set or returned low volume; invoking yfinance fallback...")
        yf_articles = fetch_news_yfinance(tickers)
        articles.extend(yf_articles)

    df = pd.DataFrame(articles, columns=["Ticker", "Title", "Publisher", "Publication_Time", "URL", "Source_API"])
    
    # Drop exact duplicate headlines for same ticker
    if not df.empty:
        df = df.drop_duplicates(subset=["Ticker", "Title"]).reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"[OK] Saved news headlines to {output_path} ({len(df)} unique headlines)")

    return df


if __name__ == "__main__":
    logger.info("[INFO] Starting Phase 1 Data Ingestion Spike...")
    prices_df = fetch_market_prices()
    news_df = fetch_news_headlines()
    logger.info("[OK] Phase 1 Data Ingestion completed successfully.")
