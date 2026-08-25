# Implementation Plan: Empirical Market Sentiment Analysis & Asset Price Correlation Pipeline

Build an automated, offline computational pipeline that ingests financial news headlines and daily stock market data for a configurable basket of equities, computes headline sentiment scores via NLTK VADER, aligns news timestamps to trading days, calculates daily price returns & metrics, measures their statistical correlation, and generates visualization charts & a comprehensive README report.

## User Review Required

> [!IMPORTANT]
> **News API Key Required**:
> Phase 0 & Phase 1 use **NewsAPI.org** or **GNews API** as the primary news source (the phased plan explicitly warns that `yfinance`'s built-in news feed is unreliable — often returning very few headlines with inconsistent coverage). `yfinance` is kept only as a fallback.
> - If you have a NewsAPI key (`NEWS_API_KEY`) or GNews API key (`GNEWS_API_KEY`), please set it in your environment or `.env` file.
> - If no API key is provided, the pipeline will fallback to `yfinance` news / public RSS feeds.

> [!NOTE]
> **Statistical Honesty (per spec)**:
> With 5–10 tickers over 7–30 days (~50–300 data points), correlation and logistic regression are likely to come back **weak or statistically insignificant**. This is expected and will be stated plainly in the README's Limitations section — not hidden or overstated.

> [!NOTE]
> **Scope Trim Under Time Pressure**:
> If the timeline gets tight, **Section 7.3 (logistic regression, ROC-AUC, confusion matrix) is the first thing to cut**. Correlation (7.1) and directional accuracy (7.2) already answer the core research question on their own.

## Open Questions

1. **Stock Basket & Time Window**:
   - Default: `['AAPL', 'MSFT', 'NVDA', 'RELIANCE.NS', 'TCS.NS']` over a 14–30 day window. Does this basket align with your target assets?
2. **Primary News Provider**:
   - Would you like to supply a free `NewsAPI.org` or `GNews` API key, or shall we build `data_fetcher.py` with multi-source fallback (NewsAPI → GNews → yfinance/RSS)?

## Design Decisions & Deviations from Blueprint

> [!NOTE]
> **Data Paths**: Blueprint Section 9 specifies raw CSVs as `data/raw_news.csv` and `data/raw_prices.csv` (flat), but Blueprint Section 11's directory tree shows `data/raw/` and `data/processed/` subdirectories. **We follow the directory tree (Section 11)** and place files under `data/raw/` and `data/processed/` respectively, as this is cleaner project organization.

> [!NOTE]
> **`src/statistical_analysis.py`**: The blueprint's directory tree (Section 11) lists only 5 files under `src/`. We add a 6th file `src/statistical_analysis.py` to separate the statistical/hypothesis-testing logic (Section 7) from the financial feature engineering (`financial_analytics.py`). This is a deliberate deviation for better separation of concerns.

---

## Directory Structure

```
financial-sentiment-price-tracker/
├── data/
│   ├── raw/                          # raw_news.csv, raw_prices.csv
│   └── processed/                    # master_sentiment_prices.csv
├── outputs/
│   ├── figures/                      # High-res PNG charts
│   └── reports/
├── src/
│   ├── data_fetcher.py
│   ├── text_processor.py
│   ├── sentiment_engine.py
│   ├── financial_analytics.py
│   ├── statistical_analysis.py       # Added (not in blueprint tree)
│   └── visualizer.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── tests/
│   └── test_sentiment.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Phased Implementation

Each phase has a clear deliverable and a **"Checkpoint — do not proceed until"** gate.

---

### Phase 0: Environment & Validation Spike (0.5–1 day)

**Goal**: De-risk the single biggest unknown before writing real pipeline code.

**Deliverables**:
- Set up the project folder using the exact repo structure above
- Create a virtual environment; install: `yfinance`, `nltk`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `requests`, `python-dotenv`, `pytest`
- Generate [requirements.txt](file:///c:/Users/Nayaks/OneDrive/Desktop/K/sentiment/requirements.txt)
- Sign up for a free NewsAPI.org or GNews API key (or confirm user-provided key)
- Write a **throwaway test script** (not part of the real pipeline) that fetches headlines for 2–3 tickers over a 7-day window and prints how many headlines come back per ticker per day

> [!CAUTION]
> **Checkpoint — do not proceed until**: You've confirmed you're getting a reasonable number of headlines (**ideally 3+ per ticker per day** on average) for at least one news source. If NewsAPI's free tier is too thin, test GNews before writing any pipeline code around it. **This is the one thing that can quietly break the entire project later.**

---

### Phase 1: Data Ingestion (1.5–2 days)

**Goal**: Build [src/data_fetcher.py](file:///c:/Users/Nayaks/OneDrive/Desktop/K/sentiment/src/data_fetcher.py) — reliable, reusable fetching for both news and prices.

**Deliverables**:

#### [NEW] [data_fetcher.py](file:///c:/Users/Nayaks/OneDrive/Desktop/K/sentiment/src/data_fetcher.py)
- **Market Prices**: Ingest OHLCV + Adjusted Close via `yfinance` (daily `1d` interval) for the configured ticker basket
- **News Ingestion**: Ingest per article: Title (String), Publisher (String), Publication Timestamp (UTC → `YYYY-MM-DD HH:MM:SS`), Article URL (String), Associated Ticker (String) — from primary API (NewsAPI/GNews), with `yfinance` as fallback
- **Resilience**: Implement exponential backoff retry wrapper for all API calls (blueprint Section 10.2)
- **Missing Market Data**: Handle trading holidays and weekend gaps cleanly without creating NaN values in return calculations (blueprint Section 10.3)
- **Outputs**: Save raw data to `data/raw/raw_news.csv` and `data/raw/raw_prices.csv`

> [!WARNING]
> **Checkpoint — do not proceed until**: Open both CSVs and manually sanity-check: Do the dates make sense? Are tickers correctly tagged? Is there at least some news volume for every ticker in your basket? If one ticker has almost no news, consider swapping it out now.

---

### Phase 2: Text Cleaning & Time Alignment (1–1.5 days)

**Goal**: Build [src/text_processor.py](file:///c:/Users/Nayaks/OneDrive/Desktop/K/sentiment/src/text_processor.py).

**Deliverables**:

#### [NEW] [text_processor.py](file:///c:/Users/Nayaks/OneDrive/Desktop/K/sentiment/src/text_processor.py)
- **Text Cleaning** (Section 4.1):
  - Strip HTML tags, escape entities, remove non-printable characters
  - **Preserve punctuation and capitalization** — critical for VADER, which uses ALL-CAPS and exclamation marks for sentiment intensity
  - Deduplicate identical headlines published across multiple syndicates
- **Text Noise Filtering** (Section 10.4):
  - Filter out non-financial headlines or promotional press releases (e.g., publisher-based allowlist, keyword heuristics for promotional content)
- **Timestamp Normalization** (Section 4.2):
  - Convert article publication timestamps to local exchange market time (EST for US markets, IST for NSE/BSE)
- **Non-Trading-Day Shift Rule** (Section 4.2):
  - If news is published after market close (e.g., after 4:00 PM EST) or on a weekend/holiday → map to the next valid trading day ($t+1$)
  - **This is the most methodologically important detail in the whole project** — every downstream statistic depends on it
- **Daily Aggregation**:
  - Group same-day headlines per ticker on trading day $t$

> [!WARNING]
> **Checkpoint — do not proceed until**: Pick one ticker and manually trace 3–4 individual headlines through the shift logic by hand. Confirm a Friday-evening headline correctly lands on Monday, and a mid-day Tuesday headline stays on Tuesday. Catching a bug here prevents silently wrong results downstream.

---

### Phase 3: Sentiment Scoring (1 day)

**Goal**: Build [src/sentiment_engine.py](file:///c:/Users/Nayaks/OneDrive/Desktop/K/sentiment/src/sentiment_engine.py).

**Deliverables**:

#### [NEW] [sentiment_engine.py](file:///c:/Users/Nayaks/OneDrive/Desktop/K/sentiment/src/sentiment_engine.py)
- **VADER Scoring** (Section 5.1–5.2):
  - Run NLTK `SentimentIntensityAnalyzer` on every cleaned headline
  - Extract per headline: Positive Score ($S_{pos} \in [0, 1]$), Negative Score ($S_{neg} \in [0, 1]$), Neutral Score ($S_{neu} \in [0, 1]$), Compound Score ($S_{compound} \in [-1.0, +1.0]$)
- **Sentiment Classification Thresholds** (Section 5.2):
  - $S_{compound} \ge 0.05$ → **Positive**
  - $S_{compound} \le -0.05$ → **Negative**
  - $-0.05 < S_{compound} < 0.05$ → **Neutral**
- **Daily Aggregated Sentiment Index** (Section 5.3):
  - Daily Mean Compound Score: $\bar{S}_t = \frac{1}{N} \sum_{j=1}^{N} S_{compound, j}$
  - Headline Volume: $N_t$ (total count of news events on day $t$)
- **Zero-News-Day Rule** (Section 10.1):
  - Impute $\bar{S}_t = 0.0$ and set flag `no_news = True`

> **Checkpoint**: Spot-check 5–10 headlines against their VADER scores manually. Does an obviously positive headline ("Company X beats earnings expectations") score positive? Does an obviously negative one score negative? This catches silent pipeline bugs (e.g., accidentally scoring the wrong column).

---

### Phase 4: Financial Analytics & Feature Engineering (1 day)

**Goal**: Build [src/financial_analytics.py](file:///c:/Users/Nayaks/OneDrive/Desktop/K/sentiment/src/financial_analytics.py).

**Deliverables**:

#### [NEW] [financial_analytics.py](file:///c:/Users/Nayaks/OneDrive/Desktop/K/sentiment/src/financial_analytics.py)
- **Returns & Metrics** (Section 6):
  1. Daily Log Return: $R_t = \ln\left(\frac{P_{close, t}}{P_{close, t-1}}\right)$
  2. Daily Percentage Return: $\Delta P_t = \frac{P_{close, t} - P_{close, t-1}}{P_{close, t-1}} \times 100$
  3. Directional Target: $Y_t = 1$ if $\Delta P_t > 0$ (Up), else $0$ (Down/Flat)
  4. Intraday Volatility: $\sigma_t = \frac{P_{high, t} - P_{low, t}}{P_{low, t}}$
- **First-Row Handling**: Drop first row per ticker (no $t-1$ exists) — don't let NaN silently propagate
- **Missing Market Data**: Handle trading holidays and weekend gaps cleanly without creating NaN values in return calculations
- **Master Table Merge**: Merge sentiment (Phase 3) with price/return data (Phase 1) into `data/processed/master_sentiment_prices.csv`
  - Schema: `[Ticker, Date, Headline_Count, Mean_Sentiment, Sentiment_Class, Close_Price, Daily_Return_Pct, Price_Direction]`

> **Checkpoint**: Open the master table and confirm row counts make sense (~tickers × trading days), and that there are no unexpected NaNs outside the intentional first-day drop.

---

### Phase 5: Statistical Analysis (1–1.5 days)

**Goal**: The actual empirical test — this is the core "finding" of the project. Build [src/statistical_analysis.py](file:///c:/Users/Nayaks/OneDrive/Desktop/K/sentiment/src/statistical_analysis.py).

**Deliverables**:

#### [NEW] [statistical_analysis.py](file:///c:/Users/Nayaks/OneDrive/Desktop/K/sentiment/src/statistical_analysis.py)
- **Correlation** (Section 7.1):
  - Pearson and Spearman rank correlation between $\bar{S}_t$ and both same-day return ($R_t$) and next-day return ($R_{t+1}$)
- **Directional Accuracy Test** (Section 7.2):
  - What percentage of positive-sentiment days ($\bar{S}_t > 0.05$) saw a positive return ($R_t > 0$)?
- **Baseline Logistic Regression** (Section 7.3 — *optional/stretch, first to cut under time pressure*):
  - Train simple Logistic Regression using $\bar{S}_t$ and $N_t$ to predict $Y_t$
  - Metrics: Precision, Recall, F1-Score, ROC-AUC Score, Confusion Matrix
- Write actual numeric results in a scratch note — needed for README Key Findings

> **Checkpoint**: You should be able to state in one sentence what you found — e.g., "same-day sentiment showed a weak positive Spearman correlation (ρ ≈ 0.14, not statistically significant at n=180) with same-day returns." A vague or unstated result means you're not ready for visualization.

---

### Phase 6: Visualization (1–1.5 days)

**Goal**: Build [src/visualizer.py](file:///c:/Users/Nayaks/OneDrive/Desktop/K/sentiment/src/visualizer.py).

**Deliverables**:

#### [NEW] [visualizer.py](file:///c:/Users/Nayaks/OneDrive/Desktop/K/sentiment/src/visualizer.py)

Generate and export **static high-resolution PNG** charts (Matplotlib / Seaborn) to `outputs/figures/`:

1. **Dual-Axis Time-Series Chart** (Section 8.1):
   - Primary Y-Axis (Left): Daily Closing Price ($P_{close}$)
   - Secondary Y-Axis (Right): Daily Compound Sentiment ($\bar{S}_t$) as bar/stem plot — **Green for Positive, Red for Negative**
2. **Scatter Plot with OLS Trend Line** (Section 8.2):
   - X-Axis: Daily Mean Sentiment ($\bar{S}_t$)
   - Y-Axis: Daily Stock Price Return ($\Delta P_t$)
   - Overlay: OLS Regression line showing overall trend
3. **Confusion Matrix Heatmap** (Section 8.3):
   - TP, FP, TN, FN for the directional prediction model — **only if logistic regression was completed in Phase 5**
4. **Sentiment Category Donut Chart** (Section 8.4):
   - Proportion of Positive vs Negative vs Neutral headlines across the basket

Export format: **High-resolution PNG** (and optionally PDF) to `outputs/figures/`

> **Checkpoint**: Look at each chart as if you were a stranger seeing it for the first time. Does the visual story match the numeric result from Phase 5? If a chart looks like it shows a strong pattern but the correlation was weak, double-check for plotting bugs (e.g., unintentional smoothing or axis scaling that visually exaggerates the relationship).

---

### Phase 7: Testing, README & Packaging (1–1.5 days)

**Goal**: Make the project reproducible and presentable.

**Deliverables**:

#### [NEW] [tests/test_sentiment.py](file:///c:/Users/Nayaks/OneDrive/Desktop/K/sentiment/tests/test_sentiment.py)
- Confirm VADER scoring returns expected sign on known example headlines
- Confirm the shift-rule function handles a weekend date correctly
- Doesn't need to be exhaustive — enough to show testing discipline

#### [NEW] [notebooks/exploratory_analysis.ipynb](file:///c:/Users/Nayaks/OneDrive/Desktop/K/sentiment/notebooks/exploratory_analysis.ipynb)
- Interactive notebook demonstrating data exploration, sample visualizations, and intermediate analysis steps

#### [NEW] [README.md](file:///c:/Users/Nayaks/OneDrive/Desktop/K/sentiment/README.md) (Section 12)
Portfolio-ready documentation containing:
1. **Executive Summary**: Brief overview of the empirical objective
2. **System Architecture Diagram**: ASCII or graphic flowchart illustrating the pipeline
3. **Key Findings Section**: Embedded chart images showing sentiment-return correlations with actual numeric results
4. **Reproducibility Guide**: Exact terminal commands to install dependencies, run the pipeline, and reproduce figures
5. **Limitations & Future Scope**: Explicitly mention small-sample caveat and VADER-vs-FinBERT tradeoff

#### [NEW] [.gitignore](file:///c:/Users/Nayaks/OneDrive/Desktop/K/sentiment/.gitignore), [LICENSE](file:///c:/Users/Nayaks/OneDrive/Desktop/K/sentiment/LICENSE)

> [!IMPORTANT]
> **Final Checkpoint**: Clone your own repo into a fresh folder and follow your own README instructions exactly to confirm it actually reproduces from scratch. Someone with no context should be able to read the README top to bottom and understand the question, method, honest result, and limitations in under 5 minutes.

---

## Phase Summary

| Phase | Focus | Est. Time | Key File(s) |
|-------|-------|-----------|-------------|
| 0 | Environment + news source validation | 0.5–1 day | throwaway test script, `requirements.txt` |
| 1 | Data ingestion | 1.5–2 days | `src/data_fetcher.py` |
| 2 | Text cleaning + time alignment | 1–1.5 days | `src/text_processor.py` |
| 3 | Sentiment scoring | 1 day | `src/sentiment_engine.py` |
| 4 | Financial analytics + feature engineering | 1 day | `src/financial_analytics.py` |
| 5 | Statistical analysis | 1–1.5 days | `src/statistical_analysis.py` |
| 6 | Visualization | 1–1.5 days | `src/visualizer.py` |
| 7 | Testing, README, packaging | 1–1.5 days | `tests/`, `README.md`, `notebooks/` |

**Total estimated time: ~8–11 days at 10–15 hrs/week**

---

## Verification Plan

### Automated Tests
- `pytest tests/test_sentiment.py` — VADER scoring sign tests + shift-rule tests
- End-to-end pipeline run verifying CSV generation and figure output without errors

### Manual Verification
- Inspect `data/raw/raw_news.csv` and `data/raw/raw_prices.csv` for correct date alignment and column schema
- Inspect `data/processed/master_sentiment_prices.csv` for expected row counts and absence of unexpected NaNs
- Inspect `outputs/figures/` for clearly rendered, high-resolution charts
- Final fresh-clone reproducibility test following README instructions
