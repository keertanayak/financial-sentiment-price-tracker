"""
Sentiment Engine Module — NLTK VADER Scoring & Daily Sentiment Index Aggregation

Features:
- Score cleaned financial headlines using NLTK VADER SentimentIntensityAnalyzer.
- Classify individual headlines as Positive (>= 0.05), Negative (<= -0.05), or Neutral.
- Compute daily aggregated sentiment index (Mean Compound Score S_bar_t, Headline Volume N_t) per ticker per trading day.
- Impute Zero-News Days with S_bar_t = 0.0 and no_news flag.
"""

import logging
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is downloaded silently
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class SentimentEngine:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def score_headline(self, text: str) -> dict:
        """
        Calculate VADER scores for a single headline.
        Returns dict with keys: pos, neg, neu, compound, sentiment_class
        """
        if not text or not isinstance(text, str):
            return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0, "sentiment_class": "Neutral"}

        scores = self.sia.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            sentiment_class = "Positive"
        elif compound <= -0.05:
            sentiment_class = "Negative"
        else:
            sentiment_class = "Neutral"

        return {
            "pos": scores["pos"],
            "neg": scores["neg"],
            "neu": scores["neu"],
            "compound": compound,
            "sentiment_class": sentiment_class,
        }

    def score_news_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a dataframe of cleaned news articles.
        Expects column 'Clean_Title'. Adds VADER score columns.
        """
        if df.empty or "Clean_Title" not in df.columns:
            logger.warning("[WARNING] Dataframe empty or missing 'Clean_Title' column.")
            return df

        scored_df = df.copy()
        scores_list = [self.score_headline(title) for title in scored_df["Clean_Title"]]

        scored_df["VADER_Pos"] = [s["pos"] for s in scores_list]
        scored_df["VADER_Neg"] = [s["neg"] for s in scores_list]
        scored_df["VADER_Neu"] = [s["neu"] for s in scores_list]
        scored_df["VADER_Compound"] = [s["compound"] for s in scores_list]
        scored_df["Sentiment_Class"] = [s["sentiment_class"] for s in scores_list]

        logger.info(f"[OK] Scored {len(scored_df)} headlines with VADER.")
        return scored_df

    def aggregate_daily_sentiment(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate headline scores by Ticker and Trading_Day.
        
        Returns DataFrame with columns:
        [Ticker, Trading_Day, Headline_Count, Mean_Sentiment, Pos_Count, Neg_Count, Neu_Count, Dominant_Class, No_News]
        """
        if scored_df.empty:
            logger.warning("[WARNING] Cannot aggregate empty scored dataframe.")
            return pd.DataFrame(columns=[
                "Ticker", "Trading_Day", "Headline_Count", "Mean_Sentiment",
                "Pos_Count", "Neg_Count", "Neu_Count", "Dominant_Class", "No_News"
            ])

        grouped = scored_df.groupby(["Ticker", "Trading_Day"])

        records = []
        for (ticker, t_day), group in grouped:
            count = len(group)
            mean_compound = group["VADER_Compound"].mean()
            pos_cnt = (group["Sentiment_Class"] == "Positive").sum()
            neg_cnt = (group["Sentiment_Class"] == "Negative").sum()
            neu_cnt = (group["Sentiment_Class"] == "Neutral").sum()

            if mean_compound >= 0.05:
                dom_class = "Positive"
            elif mean_compound <= -0.05:
                dom_class = "Negative"
            else:
                dom_class = "Neutral"

            records.append({
                "Ticker": ticker,
                "Trading_Day": t_day,
                "Headline_Count": count,
                "Mean_Sentiment": round(mean_compound, 4),
                "Pos_Count": pos_cnt,
                "Neg_Count": neg_cnt,
                "Neu_Count": neu_cnt,
                "Dominant_Class": dom_class,
                "No_News": False
            })

        agg_df = pd.DataFrame(records)
        agg_df = agg_df.sort_values(by=["Ticker", "Trading_Day"]).reset_index(drop=True)
        logger.info(f"[OK] Aggregated sentiment into {len(agg_df)} ticker-day records.")
        return agg_df


if __name__ == "__main__":
    logger.info("[INFO] Testing SentimentEngine module...")
    engine = SentimentEngine()
    test_score = engine.score_headline("Company reports record quarterly profit and dividend increase")
    logger.info(f"[OK] Test headline score: {test_score}")
