"""
Statistical Analysis Module — Empirical Correlation, Directional Accuracy, & Predictive Evaluation

Features:
- Pearson and Spearman rank correlation between mean headline sentiment S_bar_t and returns (same-day R_t and next-day R_{t+1}).
- Directional accuracy test (% positive sentiment days with positive return).
- Logistic Regression baseline model with Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
- Per-ticker and pooled statistical summaries.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def compute_correlations(df: pd.DataFrame) -> dict:
    """
    Compute Pearson & Spearman correlations between Mean_Sentiment and returns:
    1. Same-Day Return (Return_Pct)
    2. Next-Day Return (Next_Return_Pct)
    
    Calculates pooled statistics and per-ticker breakdowns.
    """
    if df.empty or "Mean_Sentiment" not in df.columns:
        logger.warning("[WARNING] Master dataset empty or missing 'Mean_Sentiment'")
        return {}

    # Filter out zero-news days if required, or analyze all days
    valid_same = df.dropna(subset=["Mean_Sentiment", "Return_Pct"])
    valid_next = df.dropna(subset=["Mean_Sentiment", "Next_Return_Pct"])

    # Pooled Same-Day Correlation
    pearson_same_r, pearson_same_p = stats.pearsonr(valid_same["Mean_Sentiment"], valid_same["Return_Pct"])
    spearman_same_r, spearman_same_p = stats.spearmanr(valid_same["Mean_Sentiment"], valid_same["Return_Pct"])

    # Pooled Next-Day Correlation
    if len(valid_next) > 2:
        pearson_next_r, pearson_next_p = stats.pearsonr(valid_next["Mean_Sentiment"], valid_next["Next_Return_Pct"])
        spearman_next_r, spearman_next_p = stats.spearmanr(valid_next["Mean_Sentiment"], valid_next["Next_Return_Pct"])
    else:
        pearson_next_r, pearson_next_p = 0.0, 1.0
        spearman_next_r, spearman_next_p = 0.0, 1.0

    ticker_breakdowns = {}
    for ticker, group in df.groupby("Ticker"):
        g_same = group.dropna(subset=["Mean_Sentiment", "Return_Pct"])
        if len(g_same) >= 3 and g_same["Mean_Sentiment"].std() > 0:
            pr, pp = stats.pearsonr(g_same["Mean_Sentiment"], g_same["Return_Pct"])
            sr, sp = stats.spearmanr(g_same["Mean_Sentiment"], g_same["Return_Pct"])
        else:
            pr, pp, sr, sp = 0.0, 1.0, 0.0, 1.0

        ticker_breakdowns[ticker] = {
            "pearson_r": round(float(pr), 4),
            "pearson_p": round(float(pp), 4),
            "spearman_r": round(float(sr), 4),
            "spearman_p": round(float(sp), 4),
            "sample_size": len(g_same)
        }

    results = {
        "pooled_same_day": {
            "pearson_r": round(float(pearson_same_r), 4),
            "pearson_p": round(float(pearson_same_p), 4),
            "spearman_r": round(float(spearman_same_r), 4),
            "spearman_p": round(float(spearman_same_p), 4),
            "sample_size": len(valid_same)
        },
        "pooled_next_day": {
            "pearson_r": round(float(pearson_next_r), 4),
            "pearson_p": round(float(pearson_next_p), 4),
            "spearman_r": round(float(spearman_next_r), 4),
            "spearman_p": round(float(spearman_next_p), 4),
            "sample_size": len(valid_next)
        },
        "per_ticker": ticker_breakdowns
    }

    logger.info(f"[OK] Calculated correlations (Same-Day Pearson r={results['pooled_same_day']['pearson_r']}, p={results['pooled_same_day']['pearson_p']})")
    return results


def compute_directional_accuracy(df: pd.DataFrame) -> dict:
    """
    Compute directional accuracy:
    - What % of positive sentiment days (Mean_Sentiment >= 0.05) had positive returns (Return_Pct > 0)?
    - What % of negative sentiment days (Mean_Sentiment <= -0.05) had negative returns (Return_Pct < 0)?
    - Overall directional agreement rate.
    """
    if df.empty:
        return {}

    pos_days = df[df["Mean_Sentiment"] >= 0.05]
    neg_days = df[df["Mean_Sentiment"] <= -0.05]
    non_zero = df[df["Mean_Sentiment"].abs() >= 0.05]

    pos_correct = (pos_days["Return_Pct"] > 0).sum() if len(pos_days) > 0 else 0
    pos_acc = (pos_correct / len(pos_days) * 100.0) if len(pos_days) > 0 else 0.0

    neg_correct = (neg_days["Return_Pct"] < 0).sum() if len(neg_days) > 0 else 0
    neg_acc = (neg_correct / len(neg_days) * 100.0) if len(neg_days) > 0 else 0.0

    total_correct = pos_correct + neg_correct
    total_acc = (total_correct / len(non_zero) * 100.0) if len(non_zero) > 0 else 0.0

    results = {
        "positive_sentiment_days": len(pos_days),
        "positive_directional_acc_pct": round(pos_acc, 2),
        "negative_sentiment_days": len(neg_days),
        "negative_directional_acc_pct": round(neg_acc, 2),
        "total_active_sentiment_days": len(non_zero),
        "overall_directional_acc_pct": round(total_acc, 2)
    }

    logger.info(f"[OK] Directional Accuracy: Overall={results['overall_directional_acc_pct']}% (N={results['total_active_sentiment_days']})")
    return results


def run_baseline_logistic_regression(df: pd.DataFrame) -> dict:
    """
    Fit baseline Logistic Regression: Mean_Sentiment + Headline_Count -> Price_Direction
    Calculates Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.
    """
    valid = df.dropna(subset=["Mean_Sentiment", "Headline_Count", "Price_Direction"]).copy()

    if len(valid) < 10 or valid["Price_Direction"].nunique() < 2:
        logger.warning("[WARNING] Insufficient data or single-class target for Logistic Regression.")
        return {"status": "insufficient_data"}

    X = valid[["Mean_Sentiment", "Headline_Count"]]
    y = valid["Price_Direction"]

    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y, y_prob)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y, y_pred).tolist()

    results = {
        "status": "success",
        "sample_size": len(valid),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": cm,
        "coefficients": {
            "intercept": round(float(model.intercept_[0]), 4),
            "mean_sentiment_coef": round(float(model.coef_[0][0]), 4),
            "headline_count_coef": round(float(model.coef_[0][1]), 4)
        }
    }

    logger.info(f"[OK] Logistic Regression Baseline: Accuracy={results['accuracy']}, ROC-AUC={results['roc_auc']}")
    return results


def run_full_statistical_analysis(master_df: pd.DataFrame) -> dict:
    """Run all statistical analysis functions on master sentiment-prices dataframe."""
    logger.info("[INFO] Starting empirical statistical analysis...")
    corr_results = compute_correlations(master_df)
    dir_results = compute_directional_accuracy(master_df)
    lr_results = run_baseline_logistic_regression(master_df)

    return {
        "correlations": corr_results,
        "directional_accuracy": dir_results,
        "logistic_regression": lr_results
    }


if __name__ == "__main__":
    logger.info("[INFO] Testing Statistical Analysis module...")
