import pandas as pd
import numpy as np
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "factor_config.json"

def get_factor_weights():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            weights = json.load(f)
            total = sum(weights.values())
            if total > 0:
                return {k: v / total for k, v in weights.items()}
    return {
        "Quality": 0.25,
        "Momentum": 0.20,
        "Low-Vol": 0.20,
        "Profitability": 0.20,
        "Value": 0.15
    }

def zscore_series(s: pd.Series) -> pd.Series:
    """Computes cross-sectional z-score."""
    if s.empty or s.std() == 0:
        return s * 0
    return (s - s.mean()) / s.std()

def compute_multi_factors(fund_cache: dict, price_hist: pd.DataFrame, current_date: pd.Timestamp) -> pd.DataFrame:
    """
    Computes the 5-factor model and composite score for the active universe.
    Returns a DataFrame indexed by ticker with z-scores for each factor + Composite.
    """
    tickers = list(fund_cache.keys())
    if not tickers or price_hist.empty:
        return pd.DataFrame()
        
    weights = get_factor_weights()
    factors = []
    
    # Isolate relevant price history (up to current_date)
    # Get 252 trading days (~1 year) of history for momentum and volatility
    try:
        hist = price_hist.loc[:current_date.strftime('%Y-%m-%d')].tail(252)
        if len(hist) < 21:  # Need at least a month
            return pd.DataFrame()
            
        ret_1m = hist.pct_change(21).iloc[-1]
        ret_12m = hist.pct_change(len(hist)-1).iloc[-1]
        # Momentum: 12m minus 1m (skip recent reversal)
        mom_raw = ret_12m - ret_1m
        
        # Low-Volatility: negative of 252d realized volatility
        daily_ret = hist.pct_change().dropna()
        vol_raw = -daily_ret.std() * np.sqrt(252)
        
    except Exception as e:
        print(f"[Factor Engine] Price history error: {e}")
        mom_raw = pd.Series({t: 0 for t in tickers})
        vol_raw = pd.Series({t: 0 for t in tickers})
        
    for t in tickers:
        f = fund_cache[t]
        
        # 1. Quality Factor
        roe = f.get("roe") or 0.0
        de = f.get("debt_equity") or 1.0
        cr = f.get("current_ratio") or 1.0
        rev_g = f.get("rev_growth") or 0.0
        q_raw = roe + (1/max(de, 0.1)) + cr + rev_g
        
        # 2. Profitability Factor
        gm = f.get("net_margin") or 0.0 # using net margin as proxy for gm/opm if others missing
        roa = f.get("roa") or roe # proxy for ROIC
        prof_raw = gm + roa
        
        # 3. Value Factor
        pe = f.get("pe") or 50.0
        if pe <= 0: pe = 100.0 # penalty
        ey = 1 / pe
        pb = f.get("pb") or 5.0
        bp = 1 / max(pb, 0.1)
        dy = f.get("div_yield") or 0.0
        val_raw = ey + bp + dy
        
        factors.append({
            "ticker": t,
            "Quality_Raw": q_raw,
            "Profitability_Raw": prof_raw,
            "Value_Raw": val_raw,
        })
        
    df = pd.DataFrame(factors).set_index("ticker")
    df["Momentum_Raw"] = mom_raw.reindex(df.index).fillna(0)
    df["Low-Vol_Raw"] = vol_raw.reindex(df.index).fillna(0)
    
    # Z-score all factors
    res = pd.DataFrame(index=df.index)
    res["Quality"] = zscore_series(df["Quality_Raw"])
    res["Momentum"] = zscore_series(df["Momentum_Raw"])
    res["Low-Vol"] = zscore_series(df["Low-Vol_Raw"])
    res["Profitability"] = zscore_series(df["Profitability_Raw"])
    res["Value"] = zscore_series(df["Value_Raw"])
    
    # Calculate Composite
    res["Composite"] = (
        res["Quality"] * weights.get("Quality", 0) +
        res["Momentum"] * weights.get("Momentum", 0) +
        res["Low-Vol"] * weights.get("Low-Vol", 0) +
        res["Profitability"] * weights.get("Profitability", 0) +
        res["Value"] * weights.get("Value", 0)
    )
    
    return res

