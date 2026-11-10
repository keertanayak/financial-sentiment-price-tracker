# Empirical Market Sentiment Analysis & Asset Price Correlation Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](app.py)
[![Tests Pass](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/test_sentiment.py)

An end-to-end quantitative computational pipeline that ingests financial news headlines and stock market OHLCV price histories, quantifies news sentiment via **NLTK VADER**, enforces strict market-hour timestamp alignment, measures statistical correlation with daily asset returns, and renders interactive web dashboards & publication figures.

---

##  Executive Summary & Objective

In quantitative finance, news sentiment is often assumed to drive short-term price discovery. This project empirically tests the core hypothesis:
> **"Does daily financial news sentiment exhibit a statistically significant correlation with daily asset price returns, or is market price noise dominant over short horizons?"**

Using a multi-ticker basket (`AAPL`, `MSFT`, `NVDA`, `RELIANCE.NS`, `TCS.NS`) over a 30-day window, the pipeline collected 140+ news headlines and daily price metrics to evaluate same-day and next-day price correlation.

### Key Empirical Findings

- **Same-Day Sentiment vs Stock Return**: Pearson correlation $r = +0.044$ ($p = 0.656$, statistically insignificant).
- **Spearman Rank Correlation**: $\rho = +0.100$ ($p = 0.309$).
- **Directional Accuracy**: **57.14%** (On days with positive news sentiment, stock price increased 75% of the time; on days with negative sentiment, price decreased 33.3% of the time).
- **Conclusion**: Short-term headline sentiment shows a weak positive directional bias but does not exhibit strong linear correlation with same-day returns—supporting the weak-form Efficient Market Hypothesis (EMH) over short sample windows.

---

##  System Architecture

```
                                [ DATA INGESTION ]
                ┌───────────────────────┴───────────────────────┐
                │                                               │
    yFinance Market Prices                             NewsAPI / RSS Feeds
    (OHLCV + Adj Close)                                 (Article Headlines)
                │                                               │
                └───────────────────────┬───────────────────────┘
                                        ▼
                           [ TEXT PROCESSING ENGINE ]
                   - Strip HTML & Noise Filtering
                   - Non-Trading-Day Shift Rule (Market Close 4PM EST / Weekends -> t+1)
                                        ▼
                           [ SENTIMENT SCORING ENGINE ]
                   - NLTK VADER Compound Score S ∈ [-1.0, +1.0]
                   - Daily Aggregation & Zero-News Imputation
                                        ▼
                        [ FINANCIAL FEATURE ENGINEERING ]
                   - Daily Log Return: R_t = ln(P_t / P_{t-1})
                   - Directional Target: Y_t = 1 (Up) else 0
                   - Master Dataset Construction
                                        ▼
                  ┌─────────────────────┴─────────────────────┐
                  ▼                                           ▼
       [ STATISTICAL ANALYZER ]                   [ INTERACTIVE DASHBOARD & VISUALS ]
    - Pearson r & Spearman ρ                  - Streamlit Web UI (app.py)
    - Directional Accuracy %                  - Dual-Axis Price/Sentiment PNG Charts
    - Baseline Logistic Regression            - Scatter Regression & Correlation Heatmap
```

---

##  Directory Structure

```
financial-sentiment-price-tracker/
├── app.py                            # Streamlit Web Dashboard application
├── main.py                           # Command-line pipeline orchestrator
├── phase0_validate_news.py           # Phase 0 news source validation spike
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules
├── LICENSE                           # MIT License
│
├── data/
│   ├── raw/                          # Ingested raw_news.csv & raw_prices.csv
│   └── processed/                    # master_sentiment_prices.csv
│
├── src/                              # Core pipeline modules
│   ├── __init__.py
│   ├── data_fetcher.py               # Market price & news ingestion
│   ├── text_processor.py             # HTML cleaning & market time shift rule
│   ├── sentiment_engine.py           # NLTK VADER sentiment scoring engine
│   ├── financial_analytics.py        # Log returns & master dataset builder
│   ├── statistical_analysis.py       # Pearson/Spearman stats & logistic regression
│   └── visualizer.py                 # Matplotlib / Seaborn static figure renderer
│
├── outputs/
│   ├── figures/                      # High-res static PNG charts
│   └── reports/                      # summary_report.json
│
├── notebooks/
│   └── exploratory_analysis.ipynb    # Jupyter interactive analysis notebook
│
└── tests/
    └── test_sentiment.py             # Pytest unit test suite
```

---

##  Quickstart & Reproducibility Guide

### 1. Prerequisites & Installation

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/YOUR_USERNAME/financial-sentiment-price-tracker.git
cd financial-sentiment-price-tracker

# Create and activate virtual environment
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Interactive Web Dashboard

Launch the browser-based dashboard to dynamically select stock tickers, adjust lookback windows, and view interactive Plotly charts:

```bash
streamlit run app.py
```

### 3. Running the Batch CLI Pipeline

Execute the full end-to-end data pipeline directly from the command line:

```bash
python main.py --tickers AAPL MSFT NVDA RELIANCE.NS TCS.NS --days-back 30
```

### 4. Running Unit Tests

Verify pipeline logic (VADER scoring, timestamp shift rules, return calculations) with `pytest`:

```bash
pytest tests/
```

---

##  Sample Visualizations

### 1. Dual-Axis Stock Price vs. Sentiment Overlay
![AAPL Overlay](outputs/figures/sentiment_price_overlay_AAPL.png)

### 2. Scatter Plot & OLS Trendline
![Scatter Regression](outputs/figures/sentiment_return_scatter.png)

---

##  Methodology Details

### Market-Hour Time Shift Rule
Financial news published after market close (e.g. 4:00 PM EST for US markets) or on non-trading days (weekends/holidays) cannot impact price until the next market open. The pipeline shifts article publication timestamps $T$ to the next valid trading day $t+1$:
$$t_{\text{trading}} = \begin{cases} t & \text{if Published during market hours} \\ t+1 & \text{if Published after close, on weekend, or holiday} \end{cases}$$

### Sentiment Compound Score Formula
NLTK VADER calculates normalized compound score $S \in [-1.0, +1.0]$:
$$S = \frac{x}{\sqrt{x^2 + \alpha}}$$
where $x$ is the sum of valence scores of words in the headline and $\alpha = 15$.

---

##  License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
