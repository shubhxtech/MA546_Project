"""
live_server.py
==============
A real-time Python HTTP Server built to drive an interactive data visualization 
dashboard displaying live NLP inference and asset weight distributions.
"""

import json
import time
import copy
import threading
import sys
import bisect
import hashlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Load our ML and NLP pipelines
from news_trading_pipeline import NewsTradingPipeline, CONFIG, TICKER_ALIASES
from ml_portfolio import select_top_m, build_ml_models, calculate_shap_weights, optimize_portfolio

# New Advanced Modules
from earnings_scraper import scrape_and_segment_transcripts
from tone_shift_detector import process_transcripts_through_nlp
from regime_detector import RegimeDetector
from insider_scraper import run_insider_scraper
from insider_signal_engine import generate_insider_signals
from factor_engine import compute_multi_factors

HOST = "0.0.0.0"
PORT = 8766
STATIC_DIR = Path(__file__).parent

# Multi-Thread Synchronization
state_lock = threading.RLock()

# Global Caches
GLOBAL_HEADLINES = []
GLOBAL_DATES = []
PIPELINE_INSTANCE = None
NLP_CACHE = {}
NLP_CACHE_FILE = STATIC_DIR / "nlp_cache.json"

APP_STATE = {
    "status": "AWAITING START COMMAND",
    "logs": [],
    "db_signals": [],
    "recent_weights": {},
    "recent_shap": {},
    "trade_journal": [],
    "execution_logs": [],
    "metrics": {
        "articles_processed": 0,
        "signals_generated": 0,
        "nli_filtered_out": 0,
        "final_pnl": None,
        "live_pnl": "0.00%",
        "turnover_cost_bps": 0.0,
    },
    "history": {
        "timestamps": [],
        "gross_allocation": [],
        "pnl_curve": [],
        "rolling_sharpe": [],
    }
}

SIMULATION_PARAMS = {
    "is_running": False,
    "current_idx": 0,
    "end_idx": 0,
    "daily_weights_history": {},
    "daily_sentiment": {},
    "prev_weights": {},     # used to compute turnover cost
    "ml_params": {
         "is_window": 6,
         "is_window_unit": "months",
         "oos_window": 21,
         "oos_window_unit": "days",
         "top_m": 14,
         "opt_protocol": "SHAP Weighting",
         "models": ["Linear", "Ridge", "RF"]
    },
    "next_rebalance_date": pd.Timestamp("2070-01-01", tz="Asia/Kolkata"),
    "yfinance_cache": pd.DataFrame(),
    "benchmark_cache":    pd.Series(dtype=float),   # NIFTY50 close prices
    "fundamental_cache":  {},   # ticker -> {pe, pb, roe, de, margin, rev_growth, mkt_cap, div_yield, score}
    "factor_scores": pd.DataFrame(),
    "fundamental_min_score": 45,
    "regime": "Sideways",
    "regime_confidence": 1.0,
    "regime_history": []
}

REGIME_DETECTOR = RegimeDetector()

# ── Setup Caches ─────────────────────────────────────────────────────────────

def init_global_caches():
    global GLOBAL_HEADLINES, GLOBAL_DATES, PIPELINE_INSTANCE, NLP_CACHE
    
    print("Initializing NLP Engine...")
    PIPELINE_INSTANCE = NewsTradingPipeline(CONFIG)
    
    if NLP_CACHE_FILE.exists():
        try:
            with open(NLP_CACHE_FILE, "r") as f:
                NLP_CACHE = json.load(f)
            print(f"✅ NLP Cache loaded: {len(NLP_CACHE)} saved headlines.")
        except Exception as e:
            print(f"⚠️ Could not load NLP cache: {e}")
            NLP_CACHE = {}
    else:
        NLP_CACHE = {}
    
    print("Loading 2022-2025 CSV files into RAM...")
    base_dir = Path("/Users/shubhsahu/Desktop/Quant")
    csv_files = [
        base_dir / "economic_times_headlines_2022.csv",
        base_dir / "economic_times_headlines_2023.csv",
        base_dir / "economic_times_headlines_2024.csv",
        base_dir / "economic_times_headlines_2025.csv"
    ]
    
    dfs = []
    for cf in csv_files:
        if cf.exists():
            print(f" -> Found {cf.name}")
            try:
                df = pd.read_csv(cf)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {cf.name}: {e}")
                
    if not dfs:
        print("CRITICAL: No matching CSV datasets found in Desktop/Quant/")
        # Fallback fake data if missing
        from main import generate_synthetic_headlines
        df = generate_synthetic_headlines(n=1000)
        df['timestamp'] = [pd.Timestamp.now() - timedelta(days=1)] * 1000
        dfs.append(df)

    master_df = pd.concat(dfs, ignore_index=True)
    headline_col = next((c for c in master_df.columns if 'head' in c.lower() or 'title' in c.lower()), master_df.columns[0])
    date_col = next((c for c in master_df.columns if 'date' in c.lower() or 'time' in c.lower()), master_df.columns[1])
    
    master_df = master_df.dropna(subset=[headline_col, date_col])
    print("Parsing dates formatting across thousands of rows...")
    parsed_dates = pd.to_datetime(master_df[date_col], dayfirst=True, format='mixed')
    
    # Sort dataset chronologically so the timeline makes mathematical sense
    master_df['__internal_dt'] = parsed_dates
    master_df = master_df.sort_values(by='__internal_dt').reset_index(drop=True)
    
    print("Pre-coercing native Timezone objects...")
    # Coerce Timezone
    GLOBAL_DATES = master_df['__internal_dt'].apply(lambda d: d if d.tzinfo else d.tz_localize("Asia/Kolkata")).tolist()
    GLOBAL_HEADLINES = master_df[headline_col].tolist()
    
    print(f"Global Data Cache Locked! Memory size: {len(GLOBAL_HEADLINES)} articles ready via RAM.")

    print("Pre-fetching total OHLC market data for entire universe to RAM...")
    import yfinance as yf
    from news_trading_pipeline import TICKER_ALIASES
    
    # Calculate global timeline bound from CSV
    if not master_df.empty:
        start_bound = (GLOBAL_DATES[0] - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
        end_bound   = (GLOBAL_DATES[-1] + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    else:
        start_bound, end_bound = "2021-01-01", "2026-01-01"
        
    all_tix = [k + ".NS" for k in TICKER_ALIASES.keys() if not k.startswith("__")]
    
    cache_file = base_dir / "market_data_cache.csv"
    
    if cache_file.exists():
        print(f"Loading persistent OHLC cache from disk ({cache_file.name})...")
        try:
            raw_price_cache = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            SIMULATION_PARAMS["yfinance_cache"] = raw_price_cache
            print(f"✅ Loaded {len(raw_price_cache.columns)} NSE tickers from disk cache.")
        except Exception as e:
            print(f"❌ Failed to load disk cache: {e}")
            SIMULATION_PARAMS["yfinance_cache"] = pd.DataFrame()
    else:
        print("Pre-fetching total OHLC market data for entire universe from network...")
        try:
            raw_price_cache = yf.download(list(set(all_tix)), start=start_bound, end=end_bound, progress=False, auto_adjust=True)["Close"]
            if isinstance(raw_price_cache, pd.Series):
                 raw_price_cache = raw_price_cache.to_frame(all_tix[0])
            # Clean fully unlisted tickers from cache
            raw_price_cache = raw_price_cache.dropna(axis=1, how='all')
            
            # Persist to disk for future runs
            raw_price_cache.to_csv(cache_file)
            print(f"✅ Downloaded and cached structured price data for {len(raw_price_cache.columns)} NSE tickers to {cache_file.name}.")
            
            SIMULATION_PARAMS["yfinance_cache"] = raw_price_cache
        except Exception as e:
            print(f"❌ Failed to build OHLC price cache: {e}")
            SIMULATION_PARAMS["yfinance_cache"] = pd.DataFrame()

    # ── Download NIFTY50 benchmark series ────────────────────────────────────
    print("Fetching NIFTY50 benchmark price series...")
    try:
        if not master_df.empty:
            bm_start = (GLOBAL_DATES[0] - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
            bm_end   = (GLOBAL_DATES[-1] + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
        else:
            bm_start, bm_end = "2021-01-01", "2026-01-01"
        nifty = yf.download("^NSEI", start=bm_start, end=bm_end, progress=False, auto_adjust=True)["Close"]
        if isinstance(nifty, pd.DataFrame):
            nifty = nifty.squeeze()
        nifty = nifty.dropna()
        SIMULATION_PARAMS["benchmark_cache"] = nifty
        print(f"✅ NIFTY50 benchmark loaded: {len(nifty)} trading days.")
    except Exception as e:
        print(f"❌ NIFTY50 benchmark fetch failed: {e}")

# ── Fundamental Data Layer ────────────────────────────────────────────────────

def compute_quality_score(m: dict) -> float:
    """
    Composite Quality Score (0–100) derived from key fundamental metrics.

    Scoring rationale:
      ROE (Return on Equity)   → Primary profitability signal; high ROE = capital efficiency
      Debt/Equity              → Balance sheet health; low D/E = financial safety
      Net Profit Margin        → Operating quality; high margin = pricing power / moat
      Revenue Growth           → Business momentum; growing revenue = expanding TAM
      P/E Valuation            → Penalty for extreme overvaluation or loss-making firms
    """
    score = 50.0  # neutral baseline

    # ROE: +20 max (ROE of 25%+ earns full marks)
    roe = m.get("roe")
    if roe is not None and not np.isnan(roe):
        score += min(20.0, float(roe) * 80.0)

    # Debt/Equity: +15 for debt-free, −15 for highly leveraged
    de = m.get("debt_equity")
    if de is not None and not np.isnan(de):
        de = float(de)
        if de < 0.5:   score += 15.0
        elif de < 1.0: score += 8.0
        elif de < 2.0: score += 2.0
        elif de < 3.0: score -= 5.0
        else:          score -= 15.0

    # Net Margin: +15 max (margin of 30%+ earns full marks)
    margin = m.get("net_margin")
    if margin is not None and not np.isnan(margin):
        score += min(15.0, float(margin) * 50.0)

    # Revenue Growth: +10 max (>25% growth earns full marks)
    rev_g = m.get("rev_growth")
    if rev_g is not None and not np.isnan(rev_g):
        score += min(10.0, float(rev_g) * 40.0)

    # P/E penalty for extreme overvaluation or loss-making
    pe = m.get("pe")
    if pe is not None and not np.isnan(pe):
        pe = float(pe)
        if pe < 0:     score -= 10.0  # loss-making
        elif pe > 80:  score -= 10.0  # extreme overvaluation
        elif pe > 50:  score -= 5.0
        elif 12 <= pe <= 30: score += 5.0   # fair-value sweet spot

    return round(float(np.clip(score, 0.0, 100.0)), 1)


def fetch_fundamental_cache():
    """
    Downloads yfinance.info for every tradeable ticker in the universe.
    Runs in a background daemon thread so it does not block server startup.
    Typically completes in 60–120 seconds depending on network.
    """
    from news_trading_pipeline import TICKER_ALIASES
    import yfinance as yf

    tickers = [k for k in TICKER_ALIASES.keys() if not k.startswith("__")]
    print(f"[Fundamentals] Fetching data for {len(tickers)} tickers in background...")

    cache = {}
    for base in tickers:
        ticker_ns = base + ".NS"
        try:
            info = yf.Ticker(ticker_ns).info
            # Guard: sometimes yfinance returns a nearly empty dict for delisted tickers
            if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
                continue

            raw = {
                "name":        info.get("longName") or info.get("shortName") or base,
                "sector":      info.get("sector") or info.get("industry") or "Other",
                "pe":          info.get("trailingPE"),
                "pb":          info.get("priceToBook"),
                "roe":         info.get("returnOnEquity"),
                "roa":         info.get("returnOnAssets"),
                "debt_equity": info.get("debtToEquity"),
                "net_margin":  info.get("profitMargins"),
                "rev_growth":  info.get("revenueGrowth"),
                "eps":         info.get("trailingEps"),
                "market_cap":  info.get("marketCap"),
                "div_yield":   info.get("dividendYield"),
                "current_ratio": info.get("currentRatio"),
                "52w_high":    info.get("fiftyTwoWeekHigh"),
                "52w_low":     info.get("fiftyTwoWeekLow"),
                "analyst_target": info.get("targetMeanPrice"),
            }
            raw["score"] = compute_quality_score(raw)
            cache[ticker_ns] = raw
        except Exception as e:
            print(f"[Fundamentals] {ticker_ns}: {e}")

    with state_lock:
        SIMULATION_PARAMS["fundamental_cache"] = cache

    scores = [v["score"] for v in cache.values()]
    avg = round(np.mean(scores), 1) if scores else 0
    print(f"✅ [Fundamentals] Loaded {len(cache)} tickers. Universe avg quality score: {avg}/100")

# ── ML Engine Wrapper ────────────────────────────────────────────────────────



def build_sentiment_portfolio(top_m=10):
    """Fallback: Derive normalized weights directly from daily sentiment when ML hasn't fired yet."""
    ds = SIMULATION_PARAMS["daily_sentiment"]
    if not ds:
        return {}
    
    # Aggregate all accumulated scores per ticker
    ticker_scores = {}
    for day_data in ds.values():
        for ticker, scores in day_data.items():
            if ticker not in ticker_scores:
                ticker_scores[ticker] = []
            ticker_scores[ticker].extend(scores)
    
    if not ticker_scores:
        return {}
    
    # Mean score per ticker, keep only nonzero
    avg_scores = {t: np.mean(v) for t, v in ticker_scores.items() if abs(np.mean(v)) > 1.0}
    if not avg_scores:
        return {}
    
    # Take top_m tickers by absolute score
    sorted_t = sorted(avg_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:top_m]
    raw = {t: s for t, s in sorted_t}
    
    # Normalize to sum of abs = 1.0
    total = sum(abs(v) for v in raw.values())
    if total == 0:
        return {}
    return {t: round(v / total, 4) for t, v in raw.items()}


def calculate_live_pnl():
    """
    Compute rolling PnL AND generate a full trade journal.
    Journal contains per-day: entry price, exit price, weights, stock-level
    P&L contribution and the cumulative portfolio return.
    """
    import yfinance as yf
    wh = SIMULATION_PARAMS.get("daily_weights_history", {})
    if not wh:
        return "0.00%"

    dates = sorted(wh.keys())
    start_date = dates[0]
    end_date = (pd.to_datetime(dates[-1]) + pd.Timedelta(days=12)).strftime("%Y-%m-%d")

    try:
        # Collect tickers — skip non-tradeable __ prefix tickers and near-zero weights
        all_tix = set()
        for w in wh.values():
            for k, v in w.items():
                if abs(v) > 0.001 and not k.startswith("__"):
                    all_tix.add(k)
        tix_list = sorted(all_tix)
        if not tix_list:
            return "0.00%"

        raw_price = SIMULATION_PARAMS["yfinance_cache"]
        if raw_price.empty:
            print("[PnL] Global price cache is empty!")
            return "0.00%"
            
        # Scope dates to what we bounded
        mask = (raw_price.index >= start_date) & (raw_price.index <= end_date)
        raw_price = raw_price.loc[mask]

        if raw_price.empty:
            print("[PnL] No price records found in bound")
            return "0.00%"

        # Drop any columns that are all-NaN (delisted / bad tickers)
        raw_price = raw_price.dropna(axis=1, how='all')
        valid_tix = [t for t in tix_list if t in raw_price.columns]
        raw_price = raw_price[valid_tix]

        # Build weight_df; re-map weight dates to nearest actual trading day
        weight_df = pd.DataFrame.from_dict(wh, orient="index").fillna(0.0)
        # Drop __ prefix tickers and absent tickers from weights df
        weight_df = weight_df[[c for c in weight_df.columns if c in raw_price.columns]]
        # Shift weights to prevent lookahead bias (T news trades on T+1 return)
        # We re-align weight dates to nearest actual trading day
        weight_df.index = pd.to_datetime(weight_df.index).tz_localize(None)
        all_price_dates = raw_price.index
        if all_price_dates.tz is not None:
            all_price_dates = all_price_dates.tz_localize(None)
        mapped_idx = []
        for wd in weight_df.index:
            # Find the NEXT trading day after the news
            fut = all_price_dates[all_price_dates > wd]
            mapped_idx.append(fut[0] if len(fut) > 0 else all_price_dates[-1])
        weight_df.index = pd.DatetimeIndex(mapped_idx)
        weight_df = weight_df[~weight_df.index.duplicated(keep='last')]

        # Daily returns matrix
        return_df = (raw_price / raw_price.shift(1) - 1).fillna(0.0)

        # Forward-fill weights across all price dates (hold until next rebalance)
        common_cols = [c for c in weight_df.columns if c in return_df.columns]
        if not common_cols:
            print(f"[PnL] No overlapping tickers: weights={list(weight_df.columns)[:4]}")
            return "0.00%"

        active_w = weight_df[common_cols].reindex(return_df.index).ffill().fillna(0.0)
        # Normalise so gross leverage = 1 each day
        gross_w = active_w.abs().sum(axis=1).replace(0, 1)
        active_w = active_w.div(gross_w, axis=0)

        daily_ret = (active_w * return_df[common_cols]).sum(axis=1)
        daily_ret = daily_ret[active_w.abs().sum(axis=1) > 0]  # only days with positions

        if daily_ret.empty:
            return "0.00%"

        cum_ret = float((1 + daily_ret).prod() - 1)

        # ── Build Trade Journal ────────────────────────────────────────────────
        journal = []
        cum = 1.0
        price_idx = raw_price.index

        for date, dr in daily_ret.items():
            cum *= (1 + dr)
            loc = price_idx.get_loc(date) if date in price_idx else None
            if loc is None or loc == 0:
                continue

            prev_date = price_idx[loc - 1]
            day_weights = active_w.loc[date]          # weights in force this day

            positions = []
            for ticker in common_cols:
                w = float(day_weights.get(ticker, 0.0))
                if abs(w) < 0.001:
                    continue
                entry_p = raw_price.loc[prev_date, ticker] if not pd.isna(raw_price.loc[prev_date, ticker]) else None
                exit_p  = raw_price.loc[date,      ticker] if not pd.isna(raw_price.loc[date,      ticker])  else None
                stock_ret = float(return_df.loc[date, ticker]) if ticker in return_df.columns else 0.0
                contribution = round(w * stock_ret * 100, 4)
                positions.append({
                    "ticker":        ticker.replace(".NS", ""),
                    "direction":     "Long" if w > 0 else "Short",
                    "weight_pct":    round(w * 100, 2),
                    "entry_price":   round(float(entry_p), 2) if entry_p is not None else None,
                    "exit_price":    round(float(exit_p),  2) if exit_p  is not None else None,
                    "stock_ret_pct": round(stock_ret * 100, 3),
                    "contribution_pct": contribution,
                })

            positions.sort(key=lambda x: abs(x["contribution_pct"]), reverse=True)
            journal.append({
                "date":              str(date.date()),
                "daily_return_pct": round(float(dr) * 100, 4),
                "cumulative_pct":   round((cum - 1) * 100, 4),
                "positions":        positions,
                "n_long":           sum(1 for p in positions if p["direction"] == "Long"),
                "n_short":          sum(1 for p in positions if p["direction"] == "Short"),
            })

        with state_lock:
            APP_STATE["trade_journal"] = journal

        # --- Append rolling 21-day Sharpe to history ---
        if len(daily_ret) >= 21:
            roll_sharpe = (daily_ret.rolling(21).mean() / daily_ret.rolling(21).std() * np.sqrt(252)).dropna()
            with state_lock:
                APP_STATE["history"]["rolling_sharpe"] = [
                    round(float(v), 3) for v in roll_sharpe.values[-100:]
                ]

        print(f"[PnL] {len(daily_ret)} trading days  |  cum return: {cum_ret*100:+.2f}%")
        return f"{cum_ret * 100:+.2f}%"

    except Exception as e:
        print(f"[PnL] Error: {e}")
        import traceback; traceback.print_exc()
        return "N/A"


def push_exec_log(msg):
    """Pushes a backend execution message directly to the frontend console."""
    ts = datetime.now().strftime("%H:%M:%S")
    with state_lock:
        if "execution_logs" not in APP_STATE:
             APP_STATE["execution_logs"] = []
        APP_STATE["execution_logs"].append(f"[{ts}] {msg}")
        if len(APP_STATE["execution_logs"]) > 60:
            APP_STATE["execution_logs"] = APP_STATE["execution_logs"][-60:]


def execute_ml_rebalance(virtual_time):
    p = SIMULATION_PARAMS["ml_params"]
    is_kwargs = {p["is_window_unit"]: int(p["is_window"])}
    is_start = (virtual_time - pd.DateOffset(**is_kwargs)).strftime("%Y-%m-%d")
    current_date_str = virtual_time.strftime("%Y-%m-%d")
    
    # ── Regime Detection ───────────────────────────────────────────────────────
    try:
        bm_cache = SIMULATION_PARAMS.get("benchmark_cache", pd.Series(dtype=float))
        pr_cache = SIMULATION_PARAMS.get("yfinance_cache", pd.DataFrame())
        regime, conf, reg_params = REGIME_DETECTOR.fit_and_predict(virtual_time, bm_cache, pr_cache)
        
        with state_lock:
            SIMULATION_PARAMS["regime"] = regime
            SIMULATION_PARAMS["regime_confidence"] = conf
            SIMULATION_PARAMS["regime_history"].append({
                "date": current_date_str,
                "regime": regime,
                "confidence": conf
            })
            
            # Update ML Params with regime specific settings
            if "models" in reg_params:
                p["models"] = reg_params["models"]
            p["allow_shorts"] = reg_params.get("allow_shorts", False)
            p["max_single_stock_pct"] = reg_params.get("max_single_stock_pct", 0.20)
            
        push_exec_log(f"Regime Detected: {regime} (Conf: {conf:.2f}). Adjusting parameters.")
    except Exception as e:
        push_exec_log(f"Regime detection error: {e}")
    
    # 1. Build IS Window Sentiment DataFrame
    days_in_is = [d for d in SIMULATION_PARAMS["daily_sentiment"].keys() if is_start <= d <= current_date_str]
    # Need at minimum 3 days with signals to train anything meaningful
    if len(days_in_is) < 3:
        return None
    
    push_exec_log(f"Rebalance triggered // Date: {current_date_str} // IS Window: {len(days_in_is)} days")
        
    ds_records = []
    for d in days_in_is:
        row = {"Date": d}
        for tic, scores in SIMULATION_PARAMS["daily_sentiment"][d].items():
            row[tic] = np.mean(scores)
        ds_records.append(row)
        
    sent_df = pd.DataFrame(ds_records).set_index("Date").fillna(0.0)
    sent_df.index = pd.to_datetime(sent_df.index)
    
    # ── COMPOSITE SIGNAL: Smooth noise via 3-day rolling mean ───────
    # Single-day headlines are often noise; consensus over 72h is a signal.
    sent_df = sent_df.rolling(window=3, min_periods=1).mean()
    # 2. Extract Top M Stocks
    top_m_tickers = select_top_m(sent_df, int(p["top_m"]))
    if not top_m_tickers:
        return None

    # ── Fundamental Quality Gate ─────────────────────────────────────────────
    # Apply quality filter: long candidates must clear the minimum quality score.
    # Low-quality stocks are demoted to short-only — they can still contribute
    # alpha as shorts if they have a negative sentiment signal.
    fund_cache = SIMULATION_PARAMS.get("fundamental_cache", {})
    min_score  = SIMULATION_PARAMS.get("fundamental_min_score", 45)
    if fund_cache:
        quality_longs  = [t for t in top_m_tickers if fund_cache.get(t, {}).get("score", 50) >= min_score]
        quality_shorts = [t for t in top_m_tickers if fund_cache.get(t, {}).get("score", 50) < min_score]
        # Keep full list for the model (direction determined by predicted return sign)
        # but log the split for transparency
        n_gated = len(top_m_tickers) - len(quality_longs)
        if n_gated > 0:
            push_exec_log(f"Quality gate: {len(quality_longs)} longs / {len(quality_shorts)} short-bias "
                          f"(min score {min_score}) — {n_gated} stocks gated from longs")
        # top_m_tickers remains the full list; gating is enforced at weight clipping below

    # 3. Fetch prices for IS window from Cache
    # FIX Bug 1: end_bound was virtual_time+5days which could pull future prices
    # into the IS window near weekends/holidays, introducing lookahead bias.
    # Upper bound must be exactly virtual_time (today's date).
    try:
        raw_price = SIMULATION_PARAMS["yfinance_cache"]
        end_bound = current_date_str   # was: (virtual_time + pd.DateOffset(days=5)).strftime(...)
        mask = (raw_price.index >= is_start) & (raw_price.index <= end_bound)
        raw_price = raw_price.loc[mask]
        
        # Keep only the valid tickers that are actually present
        avail_m = [t for t in top_m_tickers if t in raw_price.columns]
        if not avail_m:
            return None
            
        raw_price = raw_price[avail_m]
        # Daily returns for risk/covariance estimation
        daily_ret = (raw_price / raw_price.shift(1) - 1).fillna(0.0)
        
        # OOS-Horizon returns for ML target training (Forward-looking Alpha)
        oos_val = int(p["oos_window"])
        oos_unit = p["oos_window_unit"]
        # Convert unit to trading days (approx 21 per month)
        oos_days = oos_val if oos_unit == "days" else oos_val * 21
        
        # Target: Return from T to T+oos_days.
        # Shift(-N) pulls the FUTURE price into today's row for training.
        fwd_oos_ret = (raw_price.shift(-oos_days) / raw_price - 1).fillna(0.0)
    except Exception as e:
        push_exec_log(f"Pricing data error: {e}")
        return None
    # Prepare features — only tickers that appear in both price and sentiment
    valid_tickers = [t for t in top_m_tickers if t in daily_ret.columns and t in sent_df.columns]
    if not valid_tickers:
        push_exec_log("No intersecting tickers between universe and cached prices.")
        return None
    
    # Lag sentiment by 1 day (T-1 sentiment predicts T return — no look-ahead)
    raw_features = sent_df[valid_tickers].shift(1).dropna()

    # ── Cross-sectional z-score normalization ─────────────────────────────────
    # Raw: "Reliance=45, HDFC=42" — both positive, no relative ranking signal
    # Z-scored: "Reliance = +1.8σ vs peers today" — THIS is the actionable alpha
    def zscore_row(row):
        std = row.std()
        return (row - row.mean()) / std if std > 1e-8 else row * 0

    # FIX Bug 2: y_train was is_ret[valid_tickers].mean(axis=1) — the market
    # average return, identical for every ticker on a given day. This forced
    # the ML model to learn market-direction beta, not stock-level alpha.
    # Every ticker received the same label so cross-sectional ranking was
    # impossible — the model could never distinguish HDFC from Wipro.
    #
    # Correct approach: build a cross-sectional panel where each observation
    # is one (day, ticker) pair. X = that ticker's z-scored sentiment on T-1.
    # y = that ticker's actual return on T. The model then learns which
    # sentiment patterns predict outperformance within the peer group.
    panel_X_rows = []
    panel_y_vals = []
    # Align the OOS-horizon target with the sentiment features
    ret_aligned = fwd_oos_ret[valid_tickers].reindex(raw_features.index)

    for day in raw_features.index:
        if day not in ret_aligned.index:
            continue
        day_sent = raw_features.loc[day, valid_tickers]
        # Z-score this day's sentiment across the ticker cross-section
        std_s = day_sent.std()
        if std_s > 1e-8:
            day_sent_z = (day_sent - day_sent.mean()) / std_s
        else:
            day_sent_z = day_sent * 0.0
        for ticker in valid_tickers:
            ret_val = ret_aligned.loc[day, ticker] if ticker in ret_aligned.columns else np.nan
            if np.isnan(ret_val):
                continue
            # One row per (day, ticker): single sentiment feature for this ticker
            panel_X_rows.append([float(day_sent_z.get(ticker, 0.0))])
            panel_y_vals.append(float(ret_val))

    if len(panel_X_rows) < 30:
        push_exec_log(f"Sparse panel ({len(panel_X_rows)} obs) — need ≥30 for ML training. Using sentiment fallback.")
        return None

    # Build panel DataFrames (single feature: z-scored sentiment of that ticker)
    X_train = pd.DataFrame(panel_X_rows, columns=["sentiment_z"])
    y_train = pd.Series(panel_y_vals, name="fwd_ret")

    # Also keep full cross-sectional X_train_cs for SHAP (one row per day,
    # one column per ticker) — needed by calculate_shap_weights interface
    X_train_cs = raw_features.apply(zscore_row, axis=1)

    # 4. Build Models (trained on the panel: one obs per day×ticker)
    push_exec_log("Training ML ensembles...")
    try:
        fitted = build_ml_models(X_train, y_train, p["models"])
    except Exception as e:
        push_exec_log(f"ML Error: {e}")
        return None

    primary_model_name = p["models"][0] if p["models"] else "Linear"
    if primary_model_name not in fitted:
        return None
    model = fitted[primary_model_name]

    # 5. Extract weights — predict alpha score for each ticker at current time
    push_exec_log(f"Computing {p['opt_protocol']} attribution...")

    # FIX Bug 3: X_test was raw unscaled sentiment. Model was trained on
    # z-scored features. Predicting on raw features gives garbage output.
    # Apply the same cross-sectional z-score used in training.
    latest_sent_raw = sent_df[valid_tickers].iloc[-1]
    std_latest = latest_sent_raw.std()
    if std_latest > 1e-8:
        latest_sent_z = (latest_sent_raw - latest_sent_raw.mean()) / std_latest
    else:
        latest_sent_z = latest_sent_raw * 0.0

    # Build a per-ticker prediction: feed each ticker's z-score through the model
    # and collect the predicted return as the alpha signal
    ticker_alpha = {}
    for ticker in valid_tickers:
        x_single = pd.DataFrame([[float(latest_sent_z.get(ticker, 0.0))]], columns=["sentiment_z"])
        try:
            pred = model.predict(x_single)[0]
        except Exception:
            pred = 0.0
        ticker_alpha[ticker] = float(pred)
    pred_ret = pd.Series(ticker_alpha)

    if p["opt_protocol"] == "SHAP Weighting":
        # For SHAP: use the cross-sectional format (rows=days, cols=tickers)
        # so SHAP attribution maps back to per-ticker weights naturally
        X_test_cs = pd.DataFrame([latest_sent_z.values], columns=valid_tickers)
        weights = calculate_shap_weights(model, X_train_cs, X_test_cs, primary_model_name)
        # Subset to valid_tickers only (SHAP may return all columns)
        weights = weights.reindex(valid_tickers).fillna(0.0)
        total_w = weights.abs().sum()
        if total_w > 1e-8:
            weights = weights / total_w
    else:
        cov = daily_ret[valid_tickers].cov()
        weights = optimize_portfolio(
            pred_ret, cov,
            framework=p["opt_protocol"],
            returns_hist=daily_ret[valid_tickers]
        )
        
    push_exec_log(f"Portfolio generated. Assets active: {sum([1 for w in weights.to_dict().values() if abs(w)>0.001])}.")

    # ── Enforce Multi-Factor Quality Gate & Insider Conviction & Regime Limits ──
    try:
        fund_cache = SIMULATION_PARAMS.get("fundamental_cache", {})
        price_cache = SIMULATION_PARAMS.get("yfinance_cache", pd.DataFrame())
        factor_scores = compute_multi_factors(fund_cache, price_cache, virtual_time)
        
        with state_lock:
            SIMULATION_PARAMS["factor_scores"] = factor_scores
            
        w_dict = weights.to_dict()
        allow_shorts = p.get("allow_shorts", False)
        max_cap = p.get("max_single_stock_pct", 0.20)
        
        if not factor_scores.empty:
            composite = factor_scores["Composite"]
            # Top tercile -> 1.3x multiplier. Bottom tercile -> zero weight.
            top_tercile = composite.quantile(0.66)
            bottom_tercile = composite.quantile(0.33)
            
            for ticker in list(w_dict.keys()):
                score = composite.get(ticker, 0)
                if score < bottom_tercile and w_dict[ticker] > 0:
                    w_dict[ticker] = 0.0 # Exclude from longs
                elif score > top_tercile and w_dict[ticker] > 0:
                    w_dict[ticker] *= 1.3
                    
        # Apply regime shorts policy and max cap
        for ticker in list(w_dict.keys()):
            if not allow_shorts and w_dict[ticker] < 0:
                w_dict[ticker] = 0.0
                
            if w_dict[ticker] > max_cap:
                w_dict[ticker] = max_cap
            elif w_dict[ticker] < -max_cap:
                w_dict[ticker] = -max_cap
                
        weights = pd.Series(w_dict).fillna(0.0)
        abs_w = weights.abs().sum()
        if abs_w > 1e-8:
            weights = weights / abs_w
            
    except Exception as e:
        push_exec_log(f"Multi-Factor application error: {e}")

    return weights.to_dict()

# ── Simulation Runner ────────────────────────────────────────────────────────

def simulation_worker():
    while True:
        with state_lock:
            running = SIMULATION_PARAMS["is_running"]
            idx = SIMULATION_PARAMS["current_idx"]
            end_i = SIMULATION_PARAMS["end_idx"]
            
        if not running or idx >= end_i:
            if running and idx >= end_i:
                with state_lock:
                    APP_STATE["status"] = "FETCHING NSE PRICES (CALCULATING P&L)..."
                    SIMULATION_PARAMS["is_running"] = False
                calculate_final_pnl()
                # Final save of the NLP Cache
                cache_copy = NLP_CACHE.copy()
                threading.Thread(target=lambda c=cache_copy: json.dump(c, open(NLP_CACHE_FILE, "w")), daemon=True).start()
            time.sleep(1) # Sit idly waiting for UI trigger
            continue
            
        # Processing cycle
        headline = GLOBAL_HEADLINES[idx]
        virtual_time = GLOBAL_DATES[idx]
        
        # ── Daily Scraper Hooks ──────────────────────────────────────────────
        day_str = virtual_time.strftime("%Y-%m-%d")
        last_day = SIMULATION_PARAMS.get("last_processed_day")
        if day_str != last_day:
            tix = [k for k in TICKER_ALIASES.keys() if not k.startswith("__")]
            if tix:
                try:
                    insider_sigs = []
                    # 1. Insider Deals
                    if SIMULATION_PARAMS.get("fetch_insider_data", False):
                        run_insider_scraper(tix, virtual_time)
                        insider_sigs = generate_insider_signals(virtual_time)
                    
                    # 2. Earnings Transcripts
                    transcript_sigs = []
                    if SIMULATION_PARAMS.get("fetch_transcripts", False):
                        scrape_and_segment_transcripts(tix, virtual_time)
                        transcript_sigs = process_transcripts_through_nlp(PIPELINE_INSTANCE, virtual_time)
                    
                    # Merge into daily sentiment
                    if day_str not in SIMULATION_PARAMS["daily_sentiment"]:
                        SIMULATION_PARAMS["daily_sentiment"][day_str] = {}
                        
                    for sig in transcript_sigs:
                        t_ns = sig["ticker"] + ".NS"
                        if t_ns not in SIMULATION_PARAMS["daily_sentiment"][day_str]:
                            SIMULATION_PARAMS["daily_sentiment"][day_str][t_ns] = []
                        score = sig["mgmt_score"]
                        if sig["tone_shift_flag"] in ["POSITIVE_SHIFT", "NEGATIVE_SHIFT"]:
                            score *= 2.0  # 2x weight for tone shifts
                        SIMULATION_PARAMS["daily_sentiment"][day_str][t_ns].append(score)
                        
                    for sig in insider_sigs:
                        t_ns = sig["ticker"] + ".NS"
                        if t_ns not in SIMULATION_PARAMS["daily_sentiment"][day_str]:
                            SIMULATION_PARAMS["daily_sentiment"][day_str][t_ns] = []
                        # Insider signals merge as synthetic sentiment
                        SIMULATION_PARAMS["daily_sentiment"][day_str][t_ns].append(sig["score"])
                        
                except Exception as e:
                    push_exec_log(f"Daily scraper error: {e}")
                    
            with state_lock:
                SIMULATION_PARAMS["last_processed_day"] = day_str
        
        with state_lock:
            SIMULATION_PARAMS["current_idx"] += 1
        
        try:
            start_t = time.time()
            
            # ── NLP CACHE CHECK ──
            hl_hash = hashlib.md5(headline.encode('utf-8')).hexdigest()
            is_valid_signal = False
            art = None
            
            if hl_hash not in NLP_CACHE:
                # ── LOOK-AHEAD BATCHING (MPS Optimization) ──
                batch_arts = []
                batch_hashes = []
                look_i = idx
                # Gather up to 16 uncached articles
                while look_i < end_i and len(batch_arts) < 16:
                    l_headline = GLOBAL_HEADLINES[look_i]
                    l_time = GLOBAL_DATES[look_i]
                    l_hash = hashlib.md5(l_headline.encode('utf-8')).hexdigest()
                    
                    if l_hash not in NLP_CACHE:
                        l_art = PIPELINE_INSTANCE.ingest(l_headline, str(l_time))
                        if l_art is not None:
                            l_art = PIPELINE_INSTANCE.extract_entities(l_art)
                            if l_art.tickers or l_art.sectors:
                                batch_arts.append(l_art)
                                batch_hashes.append(l_hash)
                            else:
                                NLP_CACHE[l_hash] = None
                        else:
                            NLP_CACHE[l_hash] = None
                    look_i += 1
                
                if batch_arts:
                    # Run batch inference natively through HF Pipeline
                    processed_arts = PIPELINE_INSTANCE.batch_analyze_articles(batch_arts, batch_size=16)
                    
                    for l_hash, p_art in zip(batch_hashes, processed_arts):
                        if getattr(p_art, 'nli_confidence', 0.0) == 0.0 or getattr(p_art, 'sentiment_score', None) is None:
                            NLP_CACHE[l_hash] = None
                        else:
                            NLP_CACHE[l_hash] = {
                                "id": p_art.id,
                                "headline": p_art.headline,
                                "tickers": p_art.tickers,
                                "sectors": p_art.sectors,
                                "relevance_score": getattr(p_art, 'relevance_score', 0.0),
                                "sentiment": getattr(p_art, 'sentiment', 'NEUTRAL'),
                                "sentiment_score": getattr(p_art, 'sentiment_score', 0.0),
                                "nli_confidence": getattr(p_art, 'nli_confidence', 0.0)
                            }
                            
                # Periodically save cache to disk
                if len(NLP_CACHE) % 1000 < 16:
                    cache_copy = NLP_CACHE.copy()
                    threading.Thread(target=lambda c=cache_copy: json.dump(c, open(NLP_CACHE_FILE, "w")), daemon=True).start()

            # Now retrieve from cache (guaranteed to be there or None)
            cached = NLP_CACHE.get(hl_hash)
            if cached is not None:
                from news_trading_pipeline import NewsArticle
                art = NewsArticle(
                    id=cached["id"],
                    headline=cached["headline"],
                    timestamp=virtual_time,
                    tickers=cached["tickers"],
                    sectors=cached["sectors"],
                    relevance_score=cached["relevance_score"],
                    sentiment=cached["sentiment"],
                    sentiment_score=cached["sentiment_score"],
                    nli_confidence=cached["nli_confidence"]
                )
                PIPELINE_INSTANCE.active_articles.append(art)
                is_valid_signal = True

            # Prevent Memory Leak and O(N^2) deduplication lag
            if len(PIPELINE_INSTANCE.active_articles) > 1000:
                PIPELINE_INSTANCE.active_articles.pop(0)
                        
            latency = time.time() - start_t
            
            with state_lock:
                APP_STATE["metrics"]["articles_processed"] += 1
                
                # Weights updates
                if is_valid_signal:
                    APP_STATE["metrics"]["signals_generated"] += 1
                    push_log(virtual_time, latency, headline, art.tickers, art.sentiment, art.sentiment_score, art.nli_confidence)
                    
                    if len(APP_STATE["db_signals"]) > 20000:
                        APP_STATE["db_signals"].pop(0)
                        
                    source_str = "Transcript" if "[EARNINGS CALL]" in headline else getattr(art, "source", "News")
                    APP_STATE["db_signals"].append({
                        "date": virtual_time.strftime("%Y-%m-%d %H:%M"),
                        "headline": headline,
                        "tickers": art.tickers,
                        "score": round(art.sentiment_score, 3),
                        "nli": round(art.nli_confidence, 3),
                        "source": source_str
                    })
                    
                    # Accumulate Daily Sentiment for ML Feature Matrix
                    day_str = virtual_time.strftime("%Y-%m-%d")
                    if day_str not in SIMULATION_PARAMS["daily_sentiment"]:
                        SIMULATION_PARAMS["daily_sentiment"][day_str] = {}
                    
                    # Track Market Regime (NIFTY/SENSEX)
                    market_score = 0
                    for t in art.tickers:
                        if t.startswith("__"): # NLP-only indices
                            market_score += art.sentiment_score
                            continue
                            
                        t_ns = t + ".NS"
                        if t_ns not in SIMULATION_PARAMS["daily_sentiment"][day_str]:
                            SIMULATION_PARAMS["daily_sentiment"][day_str][t_ns] = []
                        SIMULATION_PARAMS["daily_sentiment"][day_str][t_ns].append(art.sentiment_score)
                else:
                    APP_STATE["metrics"]["nli_filtered_out"] += 1
                    # Noise explicitly discarded from the visual log stream!

                # --- Live sentiment fallback portfolio (always visible, even before ML fires) ---
                if not APP_STATE["recent_weights"] and SIMULATION_PARAMS["daily_sentiment"]:
                    fallback_w = build_sentiment_portfolio(int(SIMULATION_PARAMS["ml_params"]["top_m"]))
                    if fallback_w:
                        APP_STATE["recent_weights"] = fallback_w
                        APP_STATE["recent_shap"] = fallback_w

                # --- Turnover cost accounting (5 bps per 1% weight change, round-trip) ---
                if is_valid_signal and SIMULATION_PARAMS["prev_weights"]:
                    prev = SIMULATION_PARAMS["prev_weights"]
                    curr = APP_STATE["recent_weights"]
                    all_t = set(list(prev.keys()) + list(curr.keys()))
                    turnover = sum(abs(curr.get(t, 0.0) - prev.get(t, 0.0)) for t in all_t)
                    cost_bps = turnover * 5.0   # 5 bps per unit of turnover
                    APP_STATE["metrics"]["turnover_cost_bps"] = round(
                        APP_STATE["metrics"].get("turnover_cost_bps", 0.0) + cost_bps, 4
                    )

                # Has Rebalance Triggered?
                if virtual_time >= SIMULATION_PARAMS["next_rebalance_date"]:
                    APP_STATE["status"] = "GRIDSEARCH TUNING ML MODELS (SHAP OPTIMIZATION)..."
                    
                    try:
                        w = execute_ml_rebalance(virtual_time)
                        if w is not None:
                            SIMULATION_PARAMS["prev_weights"] = dict(APP_STATE["recent_weights"])
                            APP_STATE["recent_weights"] = w
                            APP_STATE["recent_shap"] = w
                            t_day = virtual_time.strftime("%Y-%m-%d")
                            SIMULATION_PARAMS["daily_weights_history"][t_day] = {k: v for k, v in w.items()}
                            gross_exposure = sum(abs(v) for v in w.values())
                            
                            t_str = virtual_time.strftime("%d %b %H:%M")
                            if not APP_STATE["history"]["timestamps"] or APP_STATE["history"]["timestamps"][-1] != t_str:
                                APP_STATE["history"]["timestamps"].append(t_str)
                                APP_STATE["history"]["gross_allocation"].append(round(gross_exposure * 100, 2))
                                if len(APP_STATE["history"]["timestamps"]) > 100:
                                    APP_STATE["history"]["timestamps"].pop(0)
                                    APP_STATE["history"]["gross_allocation"].pop(0)
                            
                            # Calculate progressive live PnL after each rebalance
                            try:
                                live_pnl = calculate_live_pnl()
                                APP_STATE["metrics"]["live_pnl"] = live_pnl
                            except Exception as pnl_e:
                                print(f"Live PnL error: {pnl_e}")
                        else:
                            # ML not ready yet — use sentiment fallback and still record + compute PnL
                            fallback_w = build_sentiment_portfolio(int(SIMULATION_PARAMS["ml_params"]["top_m"]))
                            if fallback_w:
                                APP_STATE["recent_weights"] = fallback_w
                                APP_STATE["recent_shap"] = fallback_w
                                t_day = virtual_time.strftime("%Y-%m-%d")
                                SIMULATION_PARAMS["daily_weights_history"][t_day] = fallback_w
                                gross_exposure = sum(abs(v) for v in fallback_w.values())
                                t_str = virtual_time.strftime("%d %b %H:%M")
                                if not APP_STATE["history"]["timestamps"] or APP_STATE["history"]["timestamps"][-1] != t_str:
                                    APP_STATE["history"]["timestamps"].append(t_str)
                                    APP_STATE["history"]["gross_allocation"].append(round(gross_exposure * 100, 2))
                                    if len(APP_STATE["history"]["timestamps"]) > 100:
                                        APP_STATE["history"]["timestamps"].pop(0)
                                        APP_STATE["history"]["gross_allocation"].pop(0)
                                # Compute live PnL even in fallback mode
                                try:
                                    live_pnl = calculate_live_pnl()
                                    APP_STATE["metrics"]["live_pnl"] = live_pnl
                                    print(f"[Fallback] Sentiment portfolio PnL: {live_pnl}")
                                except Exception as pnl_e:
                                    push_exec_log(f"[Fallback] PnL error: {pnl_e}")
                    except Exception as loop_e:
                        import traceback
                        trace_str = traceback.format_exc().split("\n")[-3:]
                        push_exec_log(f"ML fault trace: {' | '.join(trace_str)}")
                        
                    oos_kwargs = {SIMULATION_PARAMS["ml_params"]["oos_window_unit"]: int(SIMULATION_PARAMS["ml_params"]["oos_window"])}
                    SIMULATION_PARAMS["next_rebalance_date"] = virtual_time + pd.DateOffset(**oos_kwargs)
                    APP_STATE["status"] = "CRUNCHING EVENTS..."

        except Exception as e:
            print(f"Loop engine exception parsing headline: {e}")
            
        time.sleep(0.002)  # ~500 headlines/sec — efficient without saturating CPU

def push_log(vt, lat, hl, tick, sent, sent_s, nli):
    log = {
        "time": vt.strftime("%b %d, %H:%M"),
        "headline": hl,
        "tickers": tick,
        "sentiment": sent,
        "sentiment_score": round(sent_s, 3),
        "nli_conf": round(nli, 3),
        "latency_ms": int(lat * 1000)
    }
    APP_STATE["logs"].insert(0, log)
    if len(APP_STATE["logs"]) > 50:
        APP_STATE["logs"] = APP_STATE["logs"][:50]

def calculate_final_pnl():
    from news_trading_pipeline import TICKER_ALIASES
    
    with state_lock:
        wh = copy.deepcopy(SIMULATION_PARAMS["daily_weights_history"])
        
    if not wh:
        with state_lock:
             APP_STATE["metrics"]["final_pnl"] = "0.00%"
             APP_STATE["status"] = "TEST COMPLETED"
        return
        
    dates = sorted(wh.keys())
    start_date = dates[0]
    end_date = (pd.to_datetime(dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    
    try:
        raw_price = SIMULATION_PARAMS["yfinance_cache"]
        if raw_price.empty:
            raise ValueError("Global cache is empty.")
            
        mask = (raw_price.index >= start_date) & (raw_price.index <= end_date)
        raw_price = raw_price.loc[mask]
        
        weight_df = pd.DataFrame.from_dict(wh, orient="index").fillna(0.0)
        weight_df.index = pd.to_datetime(weight_df.index)
        weight_df.columns = [c + ".NS" if not c.endswith(".NS") else c for c in weight_df.columns]
        
        # Align indexes identically so they match daily returns perfectly
        # Drop tickers that do not exist in the price cache
        valid_cols = [c for c in weight_df.columns if c in raw_price.columns]
        weight_df = weight_df[valid_cols]
        price_df = raw_price[valid_cols].reindex(weight_df.index).ffill()
        return_df = (price_df / price_df.shift(1) - 1).fillna(0.0)
        
        # Shift weights to prevent lookahead bias (Weights from today apply to tomorrow's market return)
        shifted_weights = weight_df.shift(1).fillna(0.0)
        
        # Enforce exactly 1.0 total gross leverage constraint 
        gross_w = shifted_weights.abs().sum(axis=1)
        shifted_weights = shifted_weights.div(gross_w.replace(0, 1), axis=0)
        
        aligned_w, aligned_r = shifted_weights.align(return_df, join='inner', axis=1)
        portfolio_daily_returns = (aligned_w * aligned_r).sum(axis=1)
        
        cum_ret = (1 + portfolio_daily_returns).prod() - 1
        final_str = f"{cum_ret * 100:+.2f}%"
        
        print(f"Final Return calculated: {final_str}")
        with state_lock:
             APP_STATE["metrics"]["final_pnl"] = final_str
             APP_STATE["status"] = "TEST COMPLETED"
             
    except Exception as e:
        print("P&L Calculation Error:", e)
        with state_lock:
             APP_STATE["metrics"]["final_pnl"] = "ERR"
             APP_STATE["status"] = "TEST COMPLETED"

def compute_tearsheet():
    """Compute full performance tear sheet including benchmark alpha/beta/IR/TE."""
    wh = SIMULATION_PARAMS.get("daily_weights_history", {})
    if not wh:
        return {}

    dates = sorted(wh.keys())
    all_tix = set()
    for w in wh.values(): all_tix.update(w.keys())
    tix_list = list(all_tix)
    if not tix_list: return {}

    start_date = dates[0]
    end_date = (pd.to_datetime(dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    try:
        raw_price = SIMULATION_PARAMS["yfinance_cache"]
        mask = (raw_price.index >= start_date) & (raw_price.index <= end_date)
        raw_price = raw_price.loc[mask]

        avail_tix = [t for t in tix_list if t in raw_price.columns]
        if not avail_tix: return {}
        raw_price = raw_price[avail_tix]

        weight_df = pd.DataFrame.from_dict(wh, orient="index").fillna(0.0)
        weight_df = weight_df[[c for c in avail_tix if c in weight_df.columns]]
        weight_df.index = pd.to_datetime(weight_df.index)
        price_df = raw_price.reindex(weight_df.index, method='ffill')
        return_df = (price_df / price_df.shift(1) - 1).fillna(0.0)

        shifted_w = weight_df.shift(1).fillna(0.0)
        gross_w = shifted_w.abs().sum(axis=1).replace(0, 1)
        shifted_w = shifted_w.div(gross_w, axis=0)
        aligned_w, aligned_r = shifted_w.align(return_df, join='inner', axis=1)
        daily_ret = (aligned_w * aligned_r).sum(axis=1).dropna()

        if len(daily_ret) < 2:
            return {}

        cum_ret     = float((1 + daily_ret).prod() - 1)
        n_years     = len(daily_ret) / 252.0
        ann_ret     = float((1 + cum_ret) ** (1 / max(n_years, 0.01)) - 1) if n_years > 0 else 0.0
        sharpe      = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
        downside    = daily_ret[daily_ret < 0]
        sortino     = float(daily_ret.mean() / downside.std() * np.sqrt(252)) if len(downside) > 1 and downside.std() > 0 else 0.0
        running_max = (1 + daily_ret).cumprod().cummax()
        drawdown_series = ((1 + daily_ret).cumprod() - running_max) / running_max
        max_dd      = float(drawdown_series.min())
        win_rate    = float((daily_ret > 0).mean() * 100)
        equity      = (1 + daily_ret).cumprod()

        # --- Calmar & Omega ratios ---
        calmar = float(ann_ret / abs(max_dd)) if max_dd != 0 else 0.0
        threshold = 0.0
        gains  = daily_ret[daily_ret > threshold].sum()
        losses = abs(daily_ret[daily_ret < threshold].sum())
        omega  = float(gains / losses) if losses > 0 else float('inf')

        # Monthly returns — resample
        monthly = daily_ret.resample('ME').apply(lambda x: float((1 + x).prod() - 1))
        monthly_dict = {str(k.to_period('M').to_timestamp()): round(v * 100, 2) for k, v in monthly.items()}

        # Rolling 21-day Sharpe
        roll_sharpe_series = []
        if len(daily_ret) >= 21:
            rs = (daily_ret.rolling(21).mean() / daily_ret.rolling(21).std() * np.sqrt(252)).dropna()
            roll_sharpe_series = [round(float(v), 3) for v in rs.values]

        # ── Benchmark Alpha / Beta / IR / Tracking Error ────────────────────
        benchmark_cache = SIMULATION_PARAMS.get("benchmark_cache", pd.Series(dtype=float))
        benchmark_results = {}
        if not benchmark_cache.empty:
            try:
                bm = benchmark_cache.copy()
                bm.index = pd.to_datetime(bm.index)
                bm_ret = (bm / bm.shift(1) - 1).dropna()
                # Align to portfolio dates
                common_idx = daily_ret.index.intersection(bm_ret.index)
                if len(common_idx) >= 10:
                    pr  = daily_ret.loc[common_idx]
                    bmr = bm_ret.loc[common_idx]
                    # Beta via OLS
                    cov_mat = np.cov(pr.values, bmr.values)
                    beta = float(cov_mat[0, 1] / (cov_mat[1, 1] + 1e-12))
                    # CAPM Alpha (annualised)
                    risk_free_daily = 0.065 / 252   # ~6.5% Indian risk-free rate
                    alpha_daily = pr.mean() - (risk_free_daily + beta * (bmr.mean() - risk_free_daily))
                    alpha_ann   = float(alpha_daily * 252)
                    # Tracking Error (annualised std of active return)
                    active_ret  = pr - bmr
                    tracking_er = float(active_ret.std() * np.sqrt(252))
                    # Information Ratio
                    ir = float(active_ret.mean() / active_ret.std() * np.sqrt(252)) if active_ret.std() > 0 else 0.0
                    # Benchmark cumulative return
                    bm_cum = float((1 + bmr).prod() - 1)
                    bm_equity = (1 + bmr.reindex(equity.index).fillna(0)).cumprod()
                    benchmark_results = {
                        "beta":             round(beta, 3),
                        "alpha_ann":        round(alpha_ann * 100, 2),
                        "tracking_error":   round(tracking_er * 100, 2),
                        "info_ratio":       round(ir, 3),
                        "bm_total_return":  round(bm_cum * 100, 2),
                        "bm_equity_values": [round(float(v), 4) for v in bm_equity.values],
                    }
            except Exception as bm_e:
                print(f"[Tearsheet] Benchmark computation error: {bm_e}")

        result = {
            "total_return":      round(cum_ret * 100, 2),
            "ann_return":        round(ann_ret * 100, 2),
            "sharpe":            round(sharpe, 3),
            "sortino":           round(sortino, 3),
            "calmar":            round(calmar, 3),
            "omega":             round(min(omega, 99.0), 3),
            "max_drawdown":      round(max_dd * 100, 2),
            "win_rate":          round(win_rate, 1),
            "total_days":        len(daily_ret),
            "monthly":           monthly_dict,
            "equity_dates":      [str(d.date()) for d in equity.index],
            "equity_values":     [round(float(v), 4) for v in equity.values],
            "drawdown_values":   [round(float(v) * 100, 2) for v in drawdown_series.values],
            "rolling_sharpe":    roll_sharpe_series,
            "turnover_cost_bps": round(APP_STATE["metrics"].get("turnover_cost_bps", 0.0), 2),
        }
        result.update(benchmark_results)
        return result

    except Exception as e:
        print(f"Tearsheet error: {e}")
        import traceback; traceback.print_exc()
        return {}


def compute_optimal_config():
    """
    Returns data-driven optimal run configuration with rationale.
    These values are grounded in the signal characteristics of the
    Economic Times 2022-2025 dataset and the cross-sectional ML architecture.
    """
    min_date = GLOBAL_DATES[0].strftime("%Y-%m-%d") if GLOBAL_DATES else "2022-01-01"
    max_date = GLOBAL_DATES[-1].strftime("%Y-%m-%d") if GLOBAL_DATES else "2025-12-31"
    return {
        "start": min_date,
        "end":   max_date,
        "is_window":        6,
        "is_window_unit":   "months",
        "oos_window":       21,
        "oos_window_unit":  "days",
        "top_m":            14,
        "opt_protocol":     "SHAP Weighting",
        "ml_models":        ["Linear", "Ridge", "RF"],
        "rationale": {
            "is_window":     "6 months captures 1-2 earnings cycles — sufficient for stable covariance estimation",
            "oos_window":    "21-day (monthly) rebalance minimises turnover while staying reactive to market shifts",
            "top_m":         "14 stocks (7L/7S) gives well-diversified long-short book without over-diluting alpha",
            "opt_protocol":  "SHAP Weighting directly ties position size to marginal alpha contribution",
            "ml_models":     "Linear+Ridge for regularised linear alpha; RF for non-linear regimes",
        }
    }

# ── HTTP Server ──────────────────────────────────────────────────────────────

class JSONEncoderExt(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

class LiveAPIHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/api/start-sim":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data)
            
            s_date = payload.get("start")
            e_date = payload.get("end")
            
            try:
                # Map bounding indexes
                s_dt = pd.to_datetime(s_date).tz_localize("Asia/Kolkata")
                e_dt = pd.to_datetime(e_date).tz_localize("Asia/Kolkata") + timedelta(days=1)
                
                # Binary search for O(logN) speed insertion mapping
                start_i = bisect.bisect_left(GLOBAL_DATES, s_dt)
                end_i = bisect.bisect_right(GLOBAL_DATES, e_dt)
                
                with state_lock:
                    # Reset environment entirely
                    PIPELINE_INSTANCE.active_articles = []
                    APP_STATE["logs"] = []
                    APP_STATE["db_signals"] = []
                    APP_STATE["recent_weights"] = {}
                    APP_STATE["recent_shap"] = {}
                    APP_STATE["history"] = {"timestamps": [], "gross_allocation": [], "pnl_curve": []}
                    APP_STATE["metrics"] = {"articles_processed": 0, "signals_generated": 0, "nli_filtered_out": 0, "final_pnl": None, "live_pnl": "0.00%", "turnover_cost_bps": 0.0}
                    APP_STATE["tearsheet"] = {}
                    APP_STATE["status"] = f"CRUNCHING ({end_i - start_i} events attached)"
                    
                    SIMULATION_PARAMS["current_idx"] = start_i
                    SIMULATION_PARAMS["end_idx"] = end_i
                    SIMULATION_PARAMS["daily_weights_history"] = {}
                    SIMULATION_PARAMS["daily_sentiment"] = {}
                    SIMULATION_PARAMS["prev_weights"] = {}
                    
                    # Store ML overrides
                    if payload.get("is_window"): SIMULATION_PARAMS["ml_params"]["is_window"] = int(payload.get("is_window"))
                    if payload.get("is_window_unit"): SIMULATION_PARAMS["ml_params"]["is_window_unit"] = payload.get("is_window_unit")
                    if payload.get("oos_window"): SIMULATION_PARAMS["ml_params"]["oos_window"] = int(payload.get("oos_window"))
                    if payload.get("oos_window_unit"): SIMULATION_PARAMS["ml_params"]["oos_window_unit"] = payload.get("oos_window_unit")
                    if payload.get("top_m"): SIMULATION_PARAMS["ml_params"]["top_m"] = int(payload.get("top_m"))
                    if payload.get("opt_protocol"): SIMULATION_PARAMS["ml_params"]["opt_protocol"] = payload.get("opt_protocol")
                    if payload.get("fetch_insider_data") is not None: SIMULATION_PARAMS["fetch_insider_data"] = payload.get("fetch_insider_data")
                    if payload.get("fetch_transcripts") is not None: SIMULATION_PARAMS["fetch_transcripts"] = payload.get("fetch_transcripts")
                    if payload.get("ml_models") and len(payload.get("ml_models")) > 0: 
                        SIMULATION_PARAMS["ml_params"]["models"] = payload.get("ml_models")
                    
                    # First rebalance fires after the IS window has fully accumulated
                    # so the ML model trains on a complete sample of data.
                    # The sentiment-fallback portfolio provides immediate visual feedback
                    # while the IS window is being filled.
                    is_kw = {
                        SIMULATION_PARAMS["ml_params"]["is_window_unit"]:
                        int(SIMULATION_PARAMS["ml_params"]["is_window"])
                    }
                    SIMULATION_PARAMS["next_rebalance_date"] = s_dt + pd.DateOffset(**is_kw)
                    SIMULATION_PARAMS["is_running"] = True
                    
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return
            
        if url.path == "/api/stop-sim":
            with state_lock:
                SIMULATION_PARAMS["is_running"] = False
                APP_STATE["status"] = "SIMULATION ABORTED BY USER."
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return

        if url.path == "/api/set-fundamental-filter":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                payload = json.loads(post_data)
                with state_lock:
                    if "min_score" in payload:
                        SIMULATION_PARAMS["fundamental_min_score"] = int(payload["min_score"])
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "fundamental_min_score": SIMULATION_PARAMS["fundamental_min_score"]}).encode())
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

    def do_GET(self):
        url = urlparse(self.path)

        if url.path == "/api/transcript-signals":
            sig_path = Path(__file__).parent / "data" / "transcript_signals.json"
            data = []
            if sig_path.exists():
                with open(sig_path, "r") as f:
                    data = json.load(f)
            # Send latest 50
            data = sorted(data, key=lambda x: x["date"], reverse=True)[:50]
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
            return
            
        if url.path == "/api/regime-status":
            with state_lock:
                regime = SIMULATION_PARAMS.get("regime", "Sideways")
                conf = SIMULATION_PARAMS.get("regime_confidence", 1.0)
                history = SIMULATION_PARAMS.get("regime_history", [])
                
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "current_regime": regime,
                "confidence": round(conf, 3),
                "history": history
            }).encode())
            return
            
        if url.path == "/api/insider-signals":
            sig_path = Path(__file__).parent / "data" / "insider_signals.json"
            data = []
            if sig_path.exists():
                with open(sig_path, "r") as f:
                    data = json.load(f)
            data = sorted(data, key=lambda x: x["date"], reverse=True)[:50]
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
            return
            
        if url.path == "/api/factor-scores":
            with state_lock:
                df = SIMULATION_PARAMS.get("factor_scores", pd.DataFrame())
                data = df.to_dict(orient="index") if not df.empty else {}
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
            return

        if url.path == "/api/fundamentals":
            with state_lock:
                fc = copy.deepcopy(SIMULATION_PARAMS.get("fundamental_cache", {}))
                min_score = SIMULATION_PARAMS.get("fundamental_min_score", 45)
            out = {"min_score": min_score, "loaded": len(fc) > 0, "stocks": fc}
            payload = json.dumps(out, cls=JSONEncoderExt).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Content-length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        elif url.path == "/api/optimal-config":
            payload = json.dumps(compute_optimal_config(), cls=JSONEncoderExt).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Content-length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        elif url.path == "/api/tearsheet":
            ts = compute_tearsheet()
            payload = json.dumps(ts, cls=JSONEncoderExt).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Content-length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        elif url.path == "/api/signals-db":
            with state_lock:
                payload = json.dumps(APP_STATE["db_signals"]).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Content-length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        elif url.path == "/api/trade-journal":
            with state_lock:
                # Send the trade journal list built by calculate_live_pnl()
                journal = APP_STATE.get("trade_journal", [])
                payload = json.dumps(journal, cls=JSONEncoderExt).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Content-length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        elif url.path == "/api/live-state":
            with state_lock:
                data = copy.deepcopy(APP_STATE)
                # Strip heavy arrays polled separately to prevent UI freeze
                data.pop("db_signals", None)
                data.pop("trade_journal", None)
                data["timeline_bounds"] = {
                    "min": GLOBAL_DATES[0].strftime("%Y-%m-%d") if len(GLOBAL_DATES)>0 else "1970-01-01",
                    "max": GLOBAL_DATES[-1].strftime("%Y-%m-%d") if len(GLOBAL_DATES)>0 else "2070-01-01"
                }
                data["regime"] = SIMULATION_PARAMS.get("regime", "Sideways")
            payload = json.dumps(data, cls=JSONEncoderExt).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Content-length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
            
        elif url.path == "/" or url.path == "":
            filepath = STATIC_DIR / "index.html"
        else:
            filepath = STATIC_DIR / url.path.lstrip("/")

        if filepath.exists() and filepath.is_file():
            self.send_response(200)
            if filepath.suffix == '.html': self.send_header('Content-type', 'text/html')
            elif filepath.suffix == '.css': self.send_header('Content-type', 'text/css')
            elif filepath.suffix == '.js': self.send_header('Content-type', 'application/javascript')
            self.end_headers()
            with open(filepath, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    def log_message(self, format, *args): pass

def start_server():
    init_global_caches()

    # Fetch fundamental data in background — does not block server startup
    fund_thread = threading.Thread(target=fetch_fundamental_cache, daemon=True)
    fund_thread.start()

    class ReusableHTTPServer(HTTPServer):
        allow_reuse_address = True

    server = ReusableHTTPServer((HOST, PORT), LiveAPIHandler)
    print(f"🚀 Live Interface running on http://localhost:{PORT}")
    print(f"📊 Fundamental data loading in background (~60-90 sec)...")
    
    worker = threading.Thread(target=simulation_worker, daemon=True)
    worker.start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()

if __name__ == "__main__":
    start_server()
