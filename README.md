# India NLP Quant Research Terminal
### A News-Driven Sustainable Portfolio System — IIT Mandi

**Team:** NLPQuant | **Course:** B.Tech Project | **Date:** May 2025

---

## System Overview

A full-stack algorithmic trading system that reads Indian financial news, scores it with transformer NLP models (FinBERT + DeBERTa-v3), and constructs an optimised equity portfolio from the resulting signals. The backend exposes a live REST API consumed by a React/TypeScript dashboard.

---

## Required Libraries / Dependencies

### Python (Backend)
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
flask >= 3.0          # or equivalent HTTP server
```

Install all Python dependencies:
```bash
pip install torch transformers scikit-learn scipy numpy pandas hmmlearn shap yfinance flask
```

> **Note on PyTorch:** If you have an Apple Silicon Mac, install the MPS-enabled build:
> ```bash
> pip install torch torchvision torchaudio
> ```
> On Linux with CUDA:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```

### Node.js (Frontend)
```
Node.js >= 18.x
npm >= 9.x
```

Install frontend dependencies:
```bash
cd frontend
npm install
```

---

## Steps to Run the Code

### Step 1 — Start the Python Backend Server

```bash
cd /path/to/NLPQuant_Project_code
python live_server.py
```

The server will start on **http://localhost:5000** (or the port configured in `live_server.py`).

On first run, the HuggingFace models (`ProsusAI/finbert` and `cross-encoder/nli-deberta-v3-small`) will be downloaded automatically (~500 MB). Subsequent runs use the local cache.

### Step 2 — Start the React Frontend

Open a second terminal:
```bash
cd /path/to/NLPQuant_Project_code/frontend
npm run dev
```

The dashboard will open at **http://localhost:5173**

### Step 3 — Run a Backtest (Optional CLI)

To run a batch backtest from the command line without the UI:
```bash
python main.py
```

Results will be printed to stdout and saved in the `outputs/` directory.

### Step 4 — Test the NLP Pipeline Directly

To test the 8-stage NLP pipeline standalone:
```bash
python news_trading_pipeline.py
```

This runs a built-in test with 4 sample headlines and prints the sentiment scores and target allocations.

---

## Configuration

All key hyperparameters are in `config.py`. The most commonly adjusted parameters:

| Parameter | Location | Default | Effect |
|---|---|---|---|
| Sentiment model | `config.py` → `PipelineConfig` | `ProsusAI/finbert` | NLP model used |
| Signal half-life | `config.py` → `PIPELINE_CONFIG` | 36 hours | Decay rate of news signal |
| Holding period | `config.py` → `PIPELINE_CONFIG` | 21 days | Portfolio holding duration |
| Max stock weight | `config.py` → `PIPELINE_CONFIG` | 15% | Concentration cap |
| Factor weights | `factor_config.json` | Quality 25%, Momentum 20%... | Five-factor composite weights |
| Regime parameters | `regime_parameters.json` | Bull/Bear/Sideways configs | Strategy per market regime |

---

## File Structure

```
NLPQuant_Project_code/
├── live_server.py             # Main backend server + REST API (run this first)
├── main.py                    # Batch backtest CLI entry point
├── news_trading_pipeline.py   # 8-stage NLP pipeline
├── ml_portfolio.py            # ML models + portfolio optimiser
├── factor_engine.py           # 5-factor fundamental scoring
├── regime_detector.py         # HMM market regime detection
├── tone_shift_detector.py     # Earnings transcript tone-shift
├── earnings_scraper.py        # Quarterly transcript ingestion
├── insider_scraper.py         # Insider trading data scraper
├── insider_signal_engine.py   # Insider signal processing
├── config.py                  # Full configuration + stock universe
├── factor_config.json         # Live-editable factor weights
├── regime_parameters.json     # Per-regime strategy configuration
├── nlp_cache.json             # Pre-scored NLP output cache (48.9 MB)
├── README.md                  # This file
└── frontend/                  # React dashboard
    ├── package.json           # Node dependencies
    ├── vite.config.ts
    └── src/
        ├── App.tsx
        ├── main.tsx
        ├── index.css
        ├── utils.ts           # API client
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

## NLP Model Downloads (First Run)

The system downloads two HuggingFace models on first run:
- `ProsusAI/finbert` (~440 MB) — Finance sentiment analysis
- `cross-encoder/nli-deberta-v3-small` (~180 MB) — Zero-shot relevance filtering

These are cached locally at `~/.cache/huggingface/hub/` and do not need to be re-downloaded.

To pre-download (useful for offline environments):
```bash
python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='ProsusAI/finbert'); pipeline('zero-shot-classification', model='cross-encoder/nli-deberta-v3-small')"
```

---

## Troubleshooting

| Error | Solution |
|---|---|
| `ModuleNotFoundError: torch` | Run `pip install torch` |
| `Port 5000 already in use` | Change port in `live_server.py` or kill conflicting process |
| `nlp_cache.json not found` | The file must be present; it contains pre-scored NLP data |
| Frontend shows blank page | Ensure backend is running on port 5000 before opening the UI |
| HuggingFace download fails | Check internet connection; or use a VPN if behind a firewall |

---

## Team

| Member | Roll No. | Role |
|---|---|---|
| Aditi Gupta | B21000 | Data & NLP Lead |
| Siddhi Pogakwar | B21001 | Machine Learning Lead |
| Anamika Godara | B21002 | Quant Strategy Lead |
| Shubh Sahu | B21003 | Architecture & UI Lead |
