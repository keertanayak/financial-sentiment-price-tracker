"""
Unit Tests for Sentiment Analysis & Text Processing Modules

Executes pytest unit test suite for:
- Text cleaning and HTML stripping
- Market-time alignment shift logic
- NLTK VADER sentiment scoring accuracy
- Financial return calculation
"""

import pytest
import pandas as pd
from src.text_processor import clean_headline, map_timestamp_to_trading_day, is_noisy_headline
from src.sentiment_engine import SentimentEngine
from src.financial_analytics import compute_financial_metrics


def test_clean_headline_html_stripping():
    raw = "<h1>Apple Inc.</h1> (AAPL) reports <b>record</b> Q4 profits &amp; revenue growth!"
    cleaned = clean_headline(raw)
    assert "<h1>" not in cleaned
    assert "<b>" not in cleaned
    assert "&amp;" not in cleaned
    assert "Apple Inc. (AAPL) reports record Q4 profits & revenue growth!" == cleaned


def test_clean_headline_preserves_capitalization_and_punctuation():
    raw = "SURGE! Nvidia stock explodes +15% after HUGE earnings beat!!!"
    cleaned = clean_headline(raw)
    assert "SURGE!" in cleaned
    assert "HUGE" in cleaned
    assert "!!!" in cleaned


def test_is_noisy_headline():
    noisy = "Get a 50% discount code with sponsored content buy now!"
    legit = "Microsoft announces strategic partnership with OpenAI"
    assert is_noisy_headline(noisy) is True
    assert is_noisy_headline(legit) is False


def test_map_timestamp_to_trading_day_weekend_shift():
    # Saturday news -> map to Monday
    sat_ts = "2026-08-08 12:00:00"  # Saturday
    mapped_day = map_timestamp_to_trading_day(sat_ts, "AAPL")
    # Saturday 8th -> Sunday 9th -> Monday 10th
    assert mapped_day == "2026-08-10"


def test_map_timestamp_to_trading_day_after_hours_shift():
    # Friday 10:00 PM UTC (22:00 UTC = 5:00 PM EST) -> after market close 4PM EST -> shift to Monday
    fri_evening = "2026-08-07 22:00:00"
    mapped_day = map_timestamp_to_trading_day(fri_evening, "AAPL")
    assert mapped_day == "2026-08-10"


def test_vader_scoring_positive_negative_neutral():
    engine = SentimentEngine()

    pos_res = engine.score_headline("Company reports record profits and soaring revenue beat")
    assert pos_res["sentiment_class"] == "Positive"
    assert pos_res["compound"] >= 0.05

    neg_res = engine.score_headline("Company faces massive fraud lawsuit and bankruptcy warning")
    assert neg_res["sentiment_class"] == "Negative"
    assert neg_res["compound"] <= -0.05

    neu_res = engine.score_headline("Company schedules annual shareholder meeting for Tuesday")
    assert neu_res["sentiment_class"] == "Neutral"


def test_financial_metrics_computation():
    sample_data = {
        "Date": ["2026-08-03", "2026-08-04", "2026-08-05"],
        "Ticker": ["AAPL", "AAPL", "AAPL"],
        "Open": [100.0, 102.0, 105.0],
        "High": [103.0, 106.0, 108.0],
        "Low": [99.0, 101.0, 104.0],
        "Close": [102.0, 105.0, 103.0],
        "Adj_Close": [102.0, 105.0, 103.0],
        "Volume": [1000, 1200, 1100]
    }
    df = pd.DataFrame(sample_data)
    res = compute_financial_metrics(df)

    assert len(res) == 2  # First row dropped due to return calculation
    assert "Return_Log" in res.columns
    assert "Return_Pct" in res.columns
    assert "Price_Direction" in res.columns

    # Day 2 Close 105 vs Day 1 Close 102 -> positive return
    assert res.iloc[0]["Price_Direction"] == 1
    # Day 3 Close 103 vs Day 2 Close 105 -> negative return
    assert res.iloc[1]["Price_Direction"] == 0
