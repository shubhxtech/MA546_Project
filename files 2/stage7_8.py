"""
stage7_8.py
===========
Stage 7  — Universe Ranking        (Feng Stage 10: composite score → long universe)
Stage 8  — ML Weight Optimization  (Feng Stage 11: 5-model ensemble)
Stage 9  — Portfolio Construction  (Feng Stage 12: monthly rebalance + evaluation)

Models compared (mirroring Feng et al.):
  1. Ridge Regression       (linear baseline)
  2. LASSO Regression       (feature-selecting linear)
  3. Random Forest          (nonlinear ensemble — Feng's top performer)
  4. Gradient Boosting      (XGBoost-style sequential ensemble)
  5. MLP Neural Network     (shallow feedforward network)

All models use a rolling 12-month training window, retrained monthly.
Portfolio weights are held for ~21 trading days (1 calendar month).

Output: portfolio_returns, performance_metrics, weight_history DataFrames
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SKPipeline

from config import STOCK_UNIVERSE, PIPELINE_CONFIG, SECTORS

warnings.filterwarnings("ignore")
logger = logging.getLogger("Pipeline.Stage7_9")

np.random.seed(PIPELINE_CONFIG["seed"])

# ═════════════════════════════════════════════════════════════════
#  SYNTHETIC PRICE DATA GENERATOR
#  (used when real price data is unavailable; replace with
#   yfinance / NSE data in production)
# ═════════════════════════════════════════════════════════════════

import yfinance as yf

def fetch_online_prices(
    daily_scores: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Fetch real market prices from Yahoo Finance API.
    Maps local tickers strictly to NSE (.NS) domain.
    Outputs computed log returns perfectly aligned with the score dates.
    """
    dates   = sorted(daily_scores["date"].unique())
    tickers = list(STOCK_UNIVERSE.keys())

    if not dates:
        return pd.DataFrame()

    start_date = (dates[0] - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end_date   = (dates[-1] + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    # FIX: Yahoo Finance uses different NSE symbols for some tickers:
    #   BAJAJ-AUTO → BAJAJ_AUTO.NS   (hyphen not valid in Yahoo tickers)
    #   M&M        → M%26M.NS doesn't work — use MM.NS
    YAHOO_OVERRIDES = {
        "BAJAJ-AUTO": "BAJAJ-AUTO.NS",   # actually works with hyphen on Yahoo
        "M&M":        "M%26M.NS",        # URL-encoded ampersand
    }
    ns_tickers  = []
    reverse_map = {}   # yahoo_symbol → local ticker
    for t in tickers:
        yt = YAHOO_OVERRIDES.get(t, f"{t}.NS")
        ns_tickers.append(yt)
        reverse_map[yt] = t

    logger.info(f"Downloading yfinance data for {len(tickers)} Indian tickers from {start_date} to {end_date}...")
    
    try:
        raw_price_data = yf.download(
            ns_tickers, start=start_date, end=end_date,
            progress=False, auto_adjust=True,   # FIX: use adjusted prices for splits
        )["Close"]
    except Exception as e:
        logger.error(f"Failed to fetch from yfinance: {e}")
        return pd.DataFrame()

    # FIX: rename back using reverse_map (handles overridden symbols correctly)
    raw_price_data.rename(columns=reverse_map, inplace=True)

    # FIX: yfinance returns a tz-aware DatetimeIndex (UTC); our dates are tz-naive.
    # Strip tz before reindexing to avoid alignment failure.
    if hasattr(raw_price_data.index, "tz") and raw_price_data.index.tz is not None:
        raw_price_data.index = raw_price_data.index.tz_localize(None)

    # Convert dates to DatetimeIndex for reindex compatibility
    date_index = pd.DatetimeIndex(dates)

    # Reindex to match sentiment dates EXACTLY. 
    # Forward-fill weekends/holidays so dates align properly
    price_df = raw_price_data.reindex(date_index).ffill()

    # FIX: replace zero/NaN prices before log to avoid -inf returns
    price_df = price_df.replace(0, np.nan).ffill().bfill()

    # Compute daily log returns (we need returns, not raw prices)
    return_df = np.log(price_df / price_df.shift(1)).fillna(0.0)
    return_df.index.name = "date"
    
    logger.info(f"Real market data loaded: {len(tickers)} stocks × {len(dates)} days")
    return return_df


# ═════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════

def build_feature_matrix(
    daily_scores: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build monthly feature matrix for ML models.
    Features per stock (as of each month-end):
      - sentiment_score_mean   (trailing 21d)
      - sentiment_score_std
      - esg_score_mean
      - sdg_score_mean
      - composite_score_mean
      - composite_score_momentum (21d vs 63d mean)
      - trailing_return_21d
      - trailing_volatility_21d
      - trailing_return_63d
      - sector_dummies (one-hot)
      - esg_prior (H=1, M=0.5, L=0)
    Target: forward_return_21d (next calendar month)
    """
    # FIX: ensure date column is proper datetime (it may be date objects after
    # forward-fill reset_index in stage 6), otherwise reindex against returns
    # (which has a DatetimeIndex) silently produces all NaNs.
    daily_scores = daily_scores.copy()
    daily_scores["date"] = pd.to_datetime(daily_scores["date"])

    score_pivot = daily_scores.pivot(
        index="date", columns="ticker", values=[
            "composite_score", "sentiment_score", "esg_score", "sdg_score"
        ]
    )
    score_pivot.columns = ["_".join(c) for c in score_pivot.columns]
    score_pivot = score_pivot.reindex(index=returns.index).ffill(limit=5)

    tickers = list(STOCK_UNIVERSE.keys())
    records = []

    month_ends = returns.resample("ME").last().index

    for me in month_ends:
        # Window indices
        window_21 = returns.loc[:me].tail(21)
        window_63 = returns.loc[:me].tail(63)
        score_21  = score_pivot.loc[:me].tail(21)
        score_63  = score_pivot.loc[:me].tail(63)

        if len(window_21) < 10:  # not enough history
            continue

        # Forward return — next ~21 trading days
        future = returns.loc[me:].iloc[1:22]
        if len(future) < 5:
            continue

        for ticker in tickers:
            if ticker not in returns.columns:
                continue

            info = STOCK_UNIVERSE[ticker]

            # Return & vol features
            tr_21  = window_21[ticker].sum()
            tr_63  = window_63[ticker].sum() if len(window_63) >= 10 else np.nan
            vol_21 = window_21[ticker].std() * np.sqrt(252)

            # Score features
            c21 = score_21.get(f"composite_score_{ticker}", pd.Series(dtype=float))
            c63 = score_63.get(f"composite_score_{ticker}", pd.Series(dtype=float))
            s21 = score_21.get(f"sentiment_score_{ticker}", pd.Series(dtype=float))
            e21 = score_21.get(f"esg_score_{ticker}", pd.Series(dtype=float))
            d21 = score_21.get(f"sdg_score_{ticker}", pd.Series(dtype=float))

            comp_mean_21 = c21.mean() if len(c21) > 0 else 0.0
            comp_mean_63 = c63.mean() if len(c63) > 0 else 0.0
            comp_momentum = comp_mean_21 - comp_mean_63

            # Sector one-hot
            sector_feats = {f"sector_{s}": int(info["sector"] == s) for s in SECTORS}

            # ESG prior
            esg_prior = {"H": 1.0, "M": 0.5, "L": 0.0}.get(info["esg_prior"], 0.5)

            # Target: forward 21-day cumulative return
            fwd_ret = future[ticker].sum() if ticker in future.columns else np.nan

            row = {
                "month_end":          me,
                "ticker":             ticker,
                "sector":             info["sector"],
                "sentiment_mean":     s21.mean() if len(s21) > 0 else 0.0,
                "sentiment_std":      s21.std()  if len(s21) > 1 else 0.0,
                "esg_mean":           e21.mean() if len(e21) > 0 else 0.0,
                "sdg_mean":           d21.mean() if len(d21) > 0 else 0.0,
                "composite_mean":     comp_mean_21,
                "composite_momentum": comp_momentum,
                "trailing_ret_21":    tr_21,
                "trailing_vol_21":    vol_21,
                "trailing_ret_63":    tr_63,
                "esg_prior":          esg_prior,
                "forward_ret_21":     fwd_ret,
                **sector_feats,
            }
            records.append(row)

    features = pd.DataFrame(records)
    logger.info(f"Feature matrix: {len(features):,} rows "
                f"({features['month_end'].nunique()} months × {len(tickers)} tickers)")
    return features


# ═════════════════════════════════════════════════════════════════
#  STAGE 7 — UNIVERSE RANKING  (Feng Stage 10)
# ═════════════════════════════════════════════════════════════════

def rank_universe(
    daily_scores: pd.DataFrame,
    month_end: pd.Timestamp,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """
    Rank all tickers by composite score as of month_end.
    Returns a DataFrame with rank and universe_tier columns.
    Tiers: LONG (top 25%), NEUTRAL (mid 50%), EXCLUDE (bottom 25%).
    """
    cutoff_start = month_end - pd.Timedelta(days=lookback_days)
    w_cfg = PIPELINE_CONFIG["composite_weights"]

    recent = daily_scores[
        (daily_scores["date"] >= cutoff_start) &
        (daily_scores["date"] <= month_end)
    ]

    if recent.empty:
        return pd.DataFrame()

    # Average scores over lookback window
    agg = (
        recent.groupby("ticker")[
            ["sentiment_score", "esg_score", "sdg_score", "composite_score"]
        ]
        .mean()
        .reset_index()
    )

    # Recompute composite with Feng weights (in case individual scores changed)
    agg["composite_final"] = (
        w_cfg["sentiment"] * agg["sentiment_score"].clip(-1, 1) +
        w_cfg["esg"]       * agg["esg_score"].clip(-1, 1) +
        w_cfg["sdg"]       * agg["sdg_score"].clip(0, 1)
    )

    # Add ESG prior boost from config
    for _, row in agg.iterrows():
        prior  = STOCK_UNIVERSE.get(row["ticker"], {}).get("esg_prior", "M")
        boost  = {"H": 0.05, "M": 0.0, "L": -0.05}[prior]
        agg.loc[agg["ticker"] == row["ticker"], "composite_final"] += boost

    agg["composite_final"] = agg["composite_final"].clip(-1, 1)
    agg["rank"]            = agg["composite_final"].rank(ascending=False, method="first")
    n                      = len(agg)

    def _tier(rank):
        if rank <= n * 0.25:  return "LONG"
        if rank <= n * 0.75:  return "NEUTRAL"
        return "EXCLUDE"

    agg["universe_tier"] = agg["rank"].apply(_tier)
    agg["month_end"]     = month_end
    return agg.sort_values("composite_final", ascending=False)


# ═════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS  (Feng Stage 11)
# ═════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "sentiment_mean", "sentiment_std", "esg_mean", "sdg_mean",
    "composite_mean", "composite_momentum",
    "trailing_ret_21", "trailing_vol_21", "trailing_ret_63",
    "esg_prior",
] + [f"sector_{s}" for s in SECTORS]


def _make_models() -> dict:
    """Instantiate all 5 model families."""
    return {
        "Ridge": SKPipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0, random_state=PIPELINE_CONFIG["seed"])),
        ]),
        "LASSO": SKPipeline([
            ("scaler", StandardScaler()),
            ("model",  Lasso(alpha=0.01, max_iter=2000,
                             random_state=PIPELINE_CONFIG["seed"])),
        ]),
        "RandomForest": SKPipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=200, max_depth=5, min_samples_leaf=3,
                n_jobs=-1, random_state=PIPELINE_CONFIG["seed"],
            )),
        ]),
        "GradientBoosting": SKPipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=PIPELINE_CONFIG["seed"],
            )),
        ]),
        "MLP": SKPipeline([
            ("scaler", StandardScaler()),
            ("model",  MLPRegressor(
                hidden_layer_sizes=(64, 32), activation="relu",
                max_iter=500, early_stopping=True, validation_fraction=0.15,
                learning_rate_init=0.001,
                random_state=PIPELINE_CONFIG["seed"],
            )),
        ]),
    }


# ═════════════════════════════════════════════════════════════════
#  STAGE 8 — ML WEIGHT OPTIMIZATION + PORTFOLIO CONSTRUCTION
# ═════════════════════════════════════════════════════════════════

def _portfolio_weights_from_forecasts(
    forecasts: pd.Series,
    universe_tier: pd.Series,
    sector: pd.Series,
) -> pd.Series:
    """
    Convert per-ticker return forecasts → portfolio weights.
    Constraints:
      - Long-only
      - Exclude tickers in EXCLUDE tier
      - Max single stock: 10%
      - Max single sector: 30%
      - Weights sum to 1
    Uses quadratic optimisation (min negative expected return subject to constraints).
    Falls back to rank-proportional weights if optimiser fails.
    """
    config     = PIPELINE_CONFIG
    candidates = universe_tier[universe_tier == "LONG"].index
    if len(candidates) < config["min_stocks_in_portfolio"]:
        # Relax to NEUTRAL stocks too
        candidates = universe_tier[universe_tier != "EXCLUDE"].index

    f = forecasts.loc[candidates].fillna(0.0)
    s = sector.loc[candidates]
    n = len(f)

    if n == 0:
        return pd.Series(dtype=float)

    # Simple rank-proportional starting point
    ranks    = f.rank(ascending=True)
    w0       = (ranks / ranks.sum()).values

    def _neg_return(w):
        return -(w * f.values).sum()

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    # Sector constraints
    for sec in s.unique():
        sec_mask = (s == sec).values
        constraints.append({
            "type": "ineq",
            "fun": lambda w, m=sec_mask: config["max_sector_weight"] - (w * m).sum(),
        })

    bounds = [(0.0, config["max_stock_weight"])] * n

    try:
        result = minimize(
            _neg_return, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if result.success:
            w_opt = np.clip(result.x, 0, None)
            w_opt = w_opt / w_opt.sum()
            return pd.Series(w_opt, index=candidates)
    except Exception:
        pass

    # Fallback: rank-proportional weights clipped at constraints
    w_fallback = (f.clip(lower=0) + 1e-6)
    w_fallback = w_fallback / w_fallback.sum()
    w_fallback = w_fallback.clip(upper=config["max_stock_weight"])
    w_fallback = w_fallback / w_fallback.sum()
    return w_fallback


def run_ml_portfolio(
    features: pd.DataFrame,
    returns: pd.DataFrame,
    daily_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Rolling window ML portfolio (Feng Stages 11–12).

    For each month_end:
      1. Rank universe (Stage 7)
      2. Train each model on 12-month rolling window
      3. Predict next-month returns
      4. Ensemble → portfolio weights (Stage 8)
      5. Record portfolio returns over next holding period

    Returns:
      portfolio_returns  : DataFrame of monthly returns per model + ensemble
      weight_history     : DataFrame of monthly weights per ticker
      model_performance  : rolling Spearman IC per model
    """
    train_months = PIPELINE_CONFIG["rolling_train_months"]
    hold_days    = PIPELINE_CONFIG["holding_period_days"]
    config       = PIPELINE_CONFIG

    models       = _make_models()
    model_names  = list(models.keys())

    # Ensemble weights — start equal, updated monthly based on IC
    ens_weights  = {m: 1.0 / len(models) for m in model_names}
    ic_history   = {m: [] for m in model_names}   # Spearman IC trail (3-month)

    month_ends     = sorted(features["month_end"].unique())
    port_records   = []
    weight_records = []

    logger.info(f"Running ML portfolio: {len(month_ends)} months, "
                f"{len(model_names)} models")

    for i, me in enumerate(month_ends):
        # Need at least train_months of prior data
        train_end   = me
        train_start = month_ends[max(0, i - train_months)]

        train_feat = features[
            (features["month_end"] >= train_start) &
            (features["month_end"] <  train_end)
        ].copy()

        test_feat  = features[features["month_end"] == me].copy()

        if len(train_feat) < 50 or len(test_feat) == 0:
            continue

        # Prepare X, y
        X_train = train_feat[FEATURE_COLS].fillna(0.0)
        y_train = train_feat["forward_ret_21"].fillna(0.0)
        X_test  = test_feat[FEATURE_COLS].fillna(0.0)

        # Universe ranking
        ranking = rank_universe(daily_scores, me)
        if ranking.empty:
            continue
        ranking = ranking.set_index("ticker")

        # ── Train & predict each model ────────────────────────────
        predictions = {}
        for name, mdl in models.items():
            try:
                mdl.fit(X_train, y_train)
                preds = mdl.predict(X_test)
                predictions[name] = pd.Series(preds, index=test_feat["ticker"].values)

                # Spearman IC on the latest training month cross-section (same
                # calendar month as test, but still in-sample — used only to
                # blend ensemble weights, not as a published OOS metric).
                if "forward_ret_21" in train_feat.columns:
                    last_train_me = train_feat["month_end"].max()
                    val_slice = train_feat[train_feat["month_end"] == last_train_me]
                    if len(val_slice) >= 5:
                        X_val = val_slice[FEATURE_COLS].fillna(0.0)
                        y_val = val_slice["forward_ret_21"].fillna(0.0)
                        p_val = mdl.predict(X_val)
                        ic, _ = spearmanr(p_val, y_val.values)
                        ic_history[name].append(ic if not np.isnan(ic) else 0.0)
                    else:
                        ic_history[name].append(0.0)

            except Exception as e:
                logger.warning(f"  Model {name} failed at {me}: {e}")
                predictions[name] = pd.Series(
                    0.0, index=test_feat["ticker"].values
                )

        # ── Update ensemble weights from rolling IC ───────────────
        if i >= 3:
            ic_recent = {}
            for name in model_names:
                trail = ic_history[name][-3:]  # last 3 months
                ic_recent[name] = max(np.mean(trail), 1e-4)
            total_ic = sum(ic_recent.values())
            ens_weights = {n: ic_recent[n] / total_ic for n in model_names}

        # ── Ensemble forecast ─────────────────────────────────────
        all_tickers = test_feat["ticker"].values
        ens_pred    = pd.Series(0.0, index=all_tickers)
        for name in model_names:
            ens_pred = ens_pred.add(
                predictions[name].reindex(all_tickers).fillna(0) * ens_weights[name],
                fill_value=0,
            )
        predictions["Ensemble"] = ens_pred

        # ── Portfolio weights for each model ──────────────────────
        # Prepare sector map for candidates
        ticker_sector = pd.Series(
            {t: STOCK_UNIVERSE[t]["sector"] for t in all_tickers
             if t in STOCK_UNIVERSE}
        )
        tier_map = ranking["universe_tier"].reindex(all_tickers).fillna("NEUTRAL")

        model_weights = {}
        for name in list(model_names) + ["Ensemble"]:
            wts = _portfolio_weights_from_forecasts(
                predictions[name],
                tier_map,
                ticker_sector.reindex(all_tickers).fillna("Other"),
            )
            model_weights[name] = wts

        # Equal-weight benchmark (top-quartile universe)
        long_u = tier_map[tier_map == "LONG"].index
        if len(long_u) > 0:
            ew_weights = pd.Series(1.0 / len(long_u), index=long_u)
        else:
            ew_weights = pd.Series(1.0 / len(all_tickers), index=all_tickers)
        model_weights["EqualWeight"] = ew_weights

        # Sentiment-only benchmark (ablation) — same 30d lookback as rank_universe
        sent_cutoff = me - pd.Timedelta(days=30)
        sent_scores = (
            daily_scores[
                (daily_scores["date"] >= sent_cutoff)
                & (daily_scores["date"] <= me)
            ]
            .groupby("ticker")["sentiment_score"]
            .mean()
            .reindex(all_tickers)
            .fillna(0.0)
        )
        sent_tier = sent_scores.rank(ascending=False, pct=True).apply(
            lambda p: "LONG" if p <= 0.25 else ("NEUTRAL" if p <= 0.75 else "EXCLUDE")
        )
        model_weights["SentimentOnly"] = _portfolio_weights_from_forecasts(
            sent_scores, sent_tier, ticker_sector.reindex(all_tickers).fillna("Other")
        )

        # ── Realised returns over holding period ──────────────────
        future_end_i = returns.index.searchsorted(me)
        future_rets  = returns.iloc[future_end_i : future_end_i + hold_days]

        if len(future_rets) == 0:
            continue

        # future_rets are log returns; for fixed weights, daily portfolio
        # simple return is sum_i w_i * (exp(r_i) - 1), not sum_i w_i * r_i.
        arith_day = np.expm1(future_rets)

        for name, wts in model_weights.items():
            wts_aligned = wts.reindex(returns.columns).fillna(0.0)
            if wts_aligned.sum() > 0:
                wts_aligned = wts_aligned / wts_aligned.sum()

            port_ret_series = arith_day.mul(wts_aligned, axis=1).sum(axis=1)
            cum_ret         = (1 + port_ret_series).prod() - 1

            port_records.append({
                "month_end":     me,
                "model":         name,
                "return_21d":    round(cum_ret, 6),
                "return_ann":    round((1 + cum_ret) ** (252 / hold_days) - 1, 6),
                "volatility":    round(port_ret_series.std() * np.sqrt(252), 6),
                "n_stocks":      int((wts > 0.001).sum()) if len(wts) > 0 else 0,
            })

        # ── Record weights ────────────────────────────────────────
        for ticker, wt in model_weights.get("Ensemble", {}).items():
            weight_records.append({
                "month_end": me,
                "ticker":    ticker,
                "weight":    round(wt, 4),
                "tier":      tier_map.get(ticker, "NEUTRAL"),
            })

        recs_me = [r for r in port_records if r["month_end"] == me]
        ens_ret = next((r["return_21d"] for r in recs_me if r["model"] == "Ensemble"), None)
        ew_ret = next((r["return_21d"] for r in recs_me if r["model"] == "EqualWeight"), None)
        if ens_ret is not None and ew_ret is not None:
            logger.info(f"  {me.date()} | Ensemble: {ens_ret:+.3%} | EW: {ew_ret:+.3%}")

    port_df    = pd.DataFrame(port_records)
    weight_df  = pd.DataFrame(weight_records)
    return port_df, weight_df, ic_history


# ═════════════════════════════════════════════════════════════════
#  PERFORMANCE METRICS  (Feng Stage 12 evaluation)
# ═════════════════════════════════════════════════════════════════

def compute_performance(port_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annualised performance metrics per model.
    Metrics: Ann. Return, Ann. Volatility, Sharpe, Max Drawdown,
             Calmar, Hit Rate (vs EqualWeight benchmark), Win %
    """
    records = []
    ew_rets = port_df[port_df["model"] == "EqualWeight"].set_index("month_end")["return_21d"]

    for model, grp in port_df.groupby("model"):
        grp   = grp.sort_values("month_end")
        rets  = grp["return_21d"].values

        ann_ret  = np.mean(rets) * 12
        # FIX: use ddof=1 (sample std) to match pandas/finance convention.
        # numpy default ddof=0 overstates vol with small samples (< 36 months).
        ann_vol  = np.std(rets, ddof=1) * np.sqrt(12)
        # FIX: subtract approximate Indian risk-free rate (6.5% p.a. ≈ 0.54%/month)
        # to get a true Sharpe. Previously was just return/vol (information ratio).
        RISK_FREE_MONTHLY = 0.065 / 12
        sharpe   = (np.mean(rets) - RISK_FREE_MONTHLY) / (np.std(rets, ddof=1) + 1e-10) * np.sqrt(12) if ann_vol > 0 else 0.0

        # Max Drawdown
        cum      = (1 + pd.Series(rets)).cumprod()
        rolling_max = cum.cummax()
        drawdowns   = (cum - rolling_max) / rolling_max
        max_dd      = drawdowns.min()

        calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

        # Hit rate vs equal-weight
        common_dates = grp["month_end"].values
        ew_aligned   = ew_rets.reindex(common_dates).fillna(0.0).values
        hit_rate     = (rets > ew_aligned).mean()

        win_pct      = (rets > 0).mean()

        records.append({
            "Model":         model,
            "Ann. Return":   f"{ann_ret:.2%}",
            "Ann. Vol":      f"{ann_vol:.2%}",
            "Sharpe Ratio":  f"{sharpe:.3f}",
            "Max Drawdown":  f"{max_dd:.2%}",
            "Calmar Ratio":  f"{calmar:.3f}",
            "Hit Rate vs EW":f"{hit_rate:.2%}",
            "Win %":         f"{win_pct:.2%}",
            "N Months":      len(rets),
        })

    perf_df = pd.DataFrame(records).set_index("Model")
    return perf_df
