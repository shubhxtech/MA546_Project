# India NLP Quant Research Terminal
### Team: Dhurandhar | IIT Mandi | MA546 Project | May 2025

---

## Team Members

| Member | Roll No. | Role |
|---|---|---|
| Aditi Gupta | B23XXX | Data & NLP Lead |
| Siddhi Pogakwar | B23XXX | Machine Learning Lead |
| Anamika Godara | B23XXX | Quant Strategy Lead |
| Shubh Sahu | B23358 | Architecture & UI Lead |

---

## What This Project Does

An end-to-end algorithmic trading system that:
1. Reads Indian financial news (Economic Times, 2022–2025)
2. Scores each article using **FinBERT** (sentiment) + **DeBERTa-v3** (relevance filtering)
3. Converts scores to cross-sectional Z-scores (relative strength signals)
4. Predicts 21-day forward returns using Ridge Regression + Random Forest ensemble
5. Detects market regime (Bull / Bear / Sideways) using a Gaussian HMM
6. Optimises portfolio weights using Mean-Variance or Mean-Semivariance (SLSQP)
7. Displays everything in a live React/TypeScript dashboard

---

## Dataset

### 1. News Corpus (Pre-processed — Included in this package)

The NLP scores for 100,000+ Economic Times headlines (2022–2025) are **already pre-computed** and included as:

```
nlp_cache.json   (47 MB)
```

> ✅ **You do NOT need to re-download the raw data or re-run the NLP models.**
> The cache loads automatically on startup and is sufficient for all backtesting.

**Original raw dataset (Kaggle):**
> 📦 [Economic Times Headlines India 2022–2025](https://www.kaggle.com/datasets/abhiaero/economic-times-headlines-india-2022-to-2025)
> Author: abhiaero | Kaggle

The dataset contains four CSV files:
```
economic_times_headlines_2022.csv
economic_times_headlines_2023.csv
economic_times_headlines_2024.csv
economic_times_headlines_2025.csv
```

To download the raw data (optional — cache is already pre-processed):
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (requires Kaggle API key at ~/.kaggle/kaggle.json)
kaggle datasets download -d abhiaero/economic-times-headlines-india-2022-to-2025
unzip economic-times-headlines-india-2022-to-2025.zip -d data/raw_headlines/
```

Or download manually from:
https://www.kaggle.com/datasets/abhiaero/economic-times-headlines-india-2022-to-2025

---

### 2. Market Price Data (Downloaded automatically at runtime)

Stock price data is fetched live from **Yahoo Finance** via `yfinance`. No manual download required.

**Universe:** 50+ NSE-listed equities across 9 sectors (Banking, IT, Energy, Auto, Pharma, FMCG, Industrial, Finance, Consumer)

**Benchmark:** NIFTY 50 Index (`^NSEI`)

To prefetch price data manually (optional, for offline use):
```bash
python -c "import yfinance as yf; yf.download('^NSEI', start='2022-01-01', end='2025-12-31')"
```

---

### 3. Earnings Transcripts (Included)

Quarterly earnings call transcript data for 50+ companies is included in:
```
data/transcripts/   (JSON files per company per quarter)
```
Example: `data/transcripts/RELIANCE_2022-Q1.json`

---

## Quick Start (3 steps)

### Step 1 — Install Python dependencies

```bash
pip install torch transformers scikit-learn scipy numpy pandas hmmlearn shap yfinance flask
```

> **Apple Silicon Mac:** The above command works as-is. PyTorch will use MPS (Metal) automatically.
>
> **Linux with NVIDIA GPU:**
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> pip install transformers scikit-learn scipy numpy pandas hmmlearn shap yfinance flask
> ```

Minimum Python version: **3.10**

---

### Step 2 — Start the backend server

```bash
cd Dhurandhar_B23358/Code
python live_server.py
```

The server starts at **http://localhost:5000**

Because `nlp_cache.json` is pre-included, the system starts instantly without downloading or running any NLP models. You will see:
```
INFO | Server ready on http://localhost:5000
```

---

### Step 3 — Start the frontend dashboard

Open a second terminal:
```bash
cd Dhurandhar_B23358/Code/frontend
npm install
npm run dev
```

Dashboard opens at **http://localhost:5173**

> **Note:** `npm install` only needs to be run once. After that, just `npm run dev`.

---

## Run a Backtest (No UI required)

To run a full walk-forward backtest from the command line:
```bash
python main.py
```

Results are printed to stdout and saved in `outputs/`:
- `outputs/portfolio_returns.csv`
- `outputs/portfolio_weights.csv`
- `outputs/performance_metrics.csv`
- `outputs/daily_scores.csv`

---

## Test the NLP Pipeline Standalone

```bash
python news_trading_pipeline.py
```

Runs 4 sample headlines through the complete 8-stage pipeline and prints sentiment scores. This does **not** require the server to be running.

---

## About the NLP Cache (`nlp_cache.json`)

The cache contains pre-scored sentiment output for all 100,000+ articles:

```json
{
  "headline_md5_id": {
    "headline": "HDFC Bank posts record Q3 profit...",
    "timestamp": "2023-10-15T09:30:00+05:30",
    "ticker": "HDFCBANK",
    "sentiment_score": 72.4,
    "sentiment": "POSITIVE",
    "esg_score": 0.12,
    "sdg_score": 0.08
  },
  ...
}
```

**Why is this included?** Running FinBERT + DeBERTa on 100k articles takes ~4–6 hours on a CPU. The cache means the backtest and live server run in seconds.

**To regenerate the cache** (if you want to re-score with different models):
```bash
# WARNING: Takes 4-6 hours on CPU, ~45 min on GPU
python news_trading_pipeline.py --rebuild-cache
```

---

## Configuration

Key parameters are in `config.py`:

| Parameter | Default | Effect |
|---|---|---|
| Signal half-life | 36 hours | Decay rate of news signal |
| Holding period | 21 days | How long each portfolio is held |
| Max stock weight | 15% | Concentration cap per stock |
| Max sector weight | 35% | Sector diversification cap |
| Min stocks in portfolio | 8 | Minimum holdings |
| Rebalance frequency | Month-end | When portfolio is rebalanced |

Live-editable without restart:
- `factor_config.json` — factor weights (Quality, Momentum, Value, etc.)
- `regime_parameters.json` — per-regime strategy parameters (Bull/Bear/Sideways)

---

## Project Structure

```
Dhurandhar_B23358/
├── Report/
│   └── Dhurandhar_B23358_Project_report.tex   ← full academic report (LaTeX)
├── Slides/
│   └── Dhurandhar_B23358_Project_slides.tex   ← presentation slides (LaTeX)
└── Code/
    ├── README.md                    ← this file
    ├── live_server.py               ← START HERE: main backend + REST API
    ├── main.py                      ← batch backtest CLI
    ├── news_trading_pipeline.py     ← 8-stage NLP pipeline
    ├── ml_portfolio.py              ← ML models + portfolio optimiser
    ├── factor_engine.py             ← 5-factor fundamental scoring
    ├── regime_detector.py           ← HMM market regime detection
    ├── tone_shift_detector.py       ← earnings transcript tone-shift
    ├── earnings_scraper.py          ← transcript ingestion
    ├── insider_scraper.py           ← insider trading data
    ├── insider_signal_engine.py     ← insider signal processing
    ├── config.py                    ← full configuration + stock universe
    ├── factor_config.json           ← factor weights (editable live)
    ├── regime_parameters.json       ← regime strategy params
    ├── nlp_cache.json               ← 47 MB pre-scored NLP output ✅ INCLUDED
    ├── data/
    │   └── transcripts/             ← earnings call JSON files
    ├── outputs/                     ← backtest results (CSV + PNG)
    └── frontend/                    ← React/TypeScript dashboard
        ├── package.json
        └── src/
            ├── App.tsx
            ├── utils.ts
            └── components/
                ├── ResearchTab.tsx
                ├── BacktestTab.tsx
                ├── PortfolioTab.tsx
                ├── AnalyticsTab.tsx
                ├── ScreenerTab.tsx
                ├── SignalMatrixTab.tsx
                ├── JournalTab.tsx
                ├── SettingsTab.tsx
                └── Layout.tsx
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install torch` |
| `ModuleNotFoundError: No module named 'hmmlearn'` | Run `pip install hmmlearn` |
| `Port 5000 already in use` | Run `lsof -ti:5000 | xargs kill` or change port in `live_server.py` |
| Frontend shows blank / cannot connect | Make sure `python live_server.py` is running first |
| `yfinance` download fails | Check internet connection; Yahoo Finance may throttle — retry after 30s |
| HMM error on first backtest | Need at least 100 trading days of price history loaded — use start date ≥ 2022-06-01 |

---

## Dependencies Summary

```
Python >= 3.10
torch >= 2.0
transformers >= 4.38
scikit-learn >= 1.4
scipy >= 1.11
numpy >= 1.24
pandas >= 2.0
hmmlearn >= 0.3
shap >= 0.44
yfinance >= 0.2
flask >= 3.0

Node.js >= 18.x  (for frontend only)
```
