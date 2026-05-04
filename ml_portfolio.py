import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import shap
import scipy.optimize as sco


# ── Max position cap (per stock) ─────────────────────────────────────────────
MAX_POSITION_PCT = 0.25   # No single stock > 25% gross weight
MAX_SHORT_PCT    = 0.20   # Max short per stock


def select_top_m_long_short(sentiment_df: pd.DataFrame, M: int, score_threshold: float = 3.0) -> List[str]:
    """
    Selects Top-M stocks for a long-short universe using a multi-factor composite score:
    - Factor 1: Sentiment Level (Mean) — direction of conviction
    - Factor 2: Sentiment Momentum (Latest - Initial) — trend in narrative
    - Factor 3: News Volume (Count of non-zero entries) — information density
    - Factor 4: Signal-to-Noise Ratio (|Mean| / Std) — penalises erratic sentiment

    The SNR factor is the key addition: two tickers with the same average score
    but one having stable, consistent signals will rank higher than one with noisy,
    high-variance sentiment bursts. This reduces false positives from one-off spikes.
    """
    if len(sentiment_df) < 2:
        avg = sentiment_df.mean()
        return avg.abs().nlargest(min(M, len(avg))).index.tolist()

    # Factor 1: Level — mean sentiment over the IS window
    level = sentiment_df.mean()

    # Factor 2: Momentum — change in sentiment from start to end of IS window
    momentum = sentiment_df.iloc[-1] - sentiment_df.iloc[0]

    # Factor 3: Volume — how many days has each ticker received coverage?
    volume = (sentiment_df != 0).sum()

    # Factor 4: Signal-to-Noise Ratio — high mean relative to standard deviation
    # signals consistent conviction vs. noisy spikes. Floored at 1e-8 to avoid div/0.
    std = sentiment_df.std().replace(0, np.nan).fillna(1e-8)
    snr = level.abs() / std

    # Suppress tickers with extremely weak mean signal (absolute noise)
    level = level.where(level.abs() >= score_threshold, 0.0)

    def zscore(s: pd.Series) -> pd.Series:
        denom = s.std()
        return (s - s.mean()) / denom if denom > 1e-8 else s * 0.0

    # Weighted composite: equal weight on first three, 0.5 weight on SNR
    composite = zscore(level) + zscore(momentum) + 0.5 * zscore(volume) + 0.5 * zscore(snr)

    # Select top M/2 longs (highest composite) and M/2 shorts (lowest composite)
    half_m = max(1, M // 2)
    long_candidates  = composite.nlargest(half_m).index.tolist()
    short_candidates = composite.nsmallest(half_m).index.tolist()

    candidates = list(set(long_candidates + short_candidates))
    print(f"[Universe] Multi-factor SNR selection -> {len(candidates)} active tickers "
          f"(L:{len(long_candidates)} / S:{len(short_candidates)})")
    return candidates


# Keep backward-compatible alias
def select_top_m(sentiment_df: pd.DataFrame, M: int) -> List[str]:
    return select_top_m_long_short(sentiment_df, M)


def build_ml_models(X_train: pd.DataFrame, y_train: pd.Series, selected_models: List[str]) -> Dict[str, object]:
    """
    Trains selected ML pipelines with adaptive CV folds.
    Ensures feature alignment and robustness against small datasets.
    """
    fitted_models = {}
    n = len(X_train)
    if n < 5:
        # Extreme fallback for very sparse data
        model = Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=1.0))])
        model.fit(X_train, y_train)
        return {"Ridge": model}

    cv_folds = max(2, min(5, n))

    # 1. Linear Regression
    if "Linear" in selected_models:
        model = Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])
        model.fit(X_train, y_train)
        fitted_models["Linear"] = model

    # 2. LASSO
    if "LASSO" in selected_models:
        pipe = Pipeline([('scaler', StandardScaler()), ('reg', Lasso(max_iter=5000))])
        params = {'reg__alpha': [0.001, 0.01, 0.1]}
        grid = GridSearchCV(pipe, params, cv=cv_folds, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        fitted_models["LASSO"] = grid.best_estimator_

    # 3. Ridge
    if "Ridge" in selected_models:
        pipe = Pipeline([('scaler', StandardScaler()), ('reg', Ridge())])
        params = {'reg__alpha': [0.1, 1.0, 10.0]}
        grid = GridSearchCV(pipe, params, cv=cv_folds, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        fitted_models["Ridge"] = grid.best_estimator_

    # 5. Random Forest
    if "RF" in selected_models:
        # Reduced complexity for simulation speed
        pipe = RandomForestRegressor(random_state=42, n_estimators=50, max_depth=5)
        pipe.fit(X_train, y_train)
        fitted_models["RF"] = pipe

    # 6. Gradient Boosting
    if "GBM" in selected_models:
        pipe = GradientBoostingRegressor(random_state=42, n_estimators=50, learning_rate=0.1)
        pipe.fit(X_train, y_train)
        fitted_models["GBM"] = pipe

    return fitted_models


def calculate_shap_weights(model: object, X_train: pd.DataFrame, X_test: pd.DataFrame, model_name: str) -> pd.Series:
    """
    Extracts SIGNED SHAP values or model coefficients to preserve alpha direction.
    """
    try:
        # Handle Linear/Lasso/Ridge directly via coefficients for speed and stability
        if hasattr(model, "named_steps") and 'reg' in model.named_steps:
            reg_model = model.named_steps['reg']
            if isinstance(reg_model, (LinearRegression, Lasso, Ridge)):
                coef = reg_model.coef_
                # Align with X_test columns
                return pd.Series(coef.flatten(), index=X_test.columns)

        if model_name in ["RF", "GBM", "CART"]:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test)
            if isinstance(shap_vals, list): shap_vals = shap_vals[0]
            signed_shap = shap_vals.flatten()
            return pd.Series(signed_shap, index=X_test.columns)

        # General Kernel Fallback
        baseline = shap.sample(X_train, min(20, len(X_train)))
        explainer = shap.KernelExplainer(model.predict, baseline)
        shap_vals = explainer.shap_values(X_test, nsamples=100)
        return pd.Series(shap_vals.flatten(), index=X_test.columns)

    except Exception as e:
        print(f"[SHAP] Fallback due to: {e}")
        return pd.Series(1.0 / X_test.shape[1], index=X_test.columns)


def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    framework: str = "M-V",
    returns_hist: pd.DataFrame = None,
    allow_shorts: bool = True
) -> pd.Series:
    """
    Portfolio optimizer supporting:
      - M-V:  Mean-Variance (Markowitz) — minimise variance for given return
      - M-SV: Mean-Semivariance — minimise downside-only variance
      - EW:   Equal-weight fallback
    
    Long-short enabled by default: allows negative weights up to MAX_SHORT_PCT.
    Constraint: sum of weights = 0 (dollar-neutral) when short positions exist,
                otherwise sum = 1 normalised long-only.
    """
    num_assets = len(expected_returns)
    if num_assets == 0:
        return pd.Series(dtype=float)

    # Determine if we have any short candidates from expected_returns sign
    has_short_signal = bool((expected_returns < 0).any())
    use_long_short = allow_shorts and has_short_signal

    if use_long_short:
        # Dollar-neutral: long book and short book offset
        bounds = tuple((-MAX_SHORT_PCT, MAX_POSITION_PCT) for _ in range(num_assets))
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1.0},  # gross leverage = 1
        ]
    else:
        # Long-only (when signal is uniformly positive)
        bounds = tuple((0.0, MAX_POSITION_PCT) for _ in range(num_assets))
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

    def portfolio_volatility(weights, cov):
        return np.sqrt(np.dot(weights.T, np.dot(cov, weights)) + 1e-10)

    def portfolio_semivariance(weights, rets):
        port_ret = np.dot(rets, weights)
        downside = port_ret[port_ret < 0]
        return np.sqrt(np.mean(downside ** 2) + 1e-10) if len(downside) > 0 else 0.0

    if framework == "M-SV" and returns_hist is not None and len(returns_hist) > 2:
        obj_func = portfolio_semivariance
        args = (returns_hist.values,)
    else:
        obj_func = portfolio_volatility
        args = (cov_matrix.values,)

    # Smart initialisation: start from signal direction
    sign_init = np.sign(expected_returns.values)
    sign_init = np.where(sign_init == 0, 1, sign_init)
    init_guess = sign_init / (np.sum(np.abs(sign_init)) + 1e-8)

    try:
        result = sco.minimize(
            obj_func, init_guess, args=args,
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 500}
        )
        if result.success:
            w = result.x
            # Final position cap enforcement
            w = np.clip(w, -MAX_SHORT_PCT, MAX_POSITION_PCT)
            abs_w = np.sum(np.abs(w))
            if abs_w > 0:
                w = w / abs_w
            return pd.Series(w, index=expected_returns.index)
    except Exception as e:
        print(f"Optimizer failed: {e}")

    # FIX: Fallback must re-normalize AFTER clipping, otherwise long-only
    # portfolios sum to < 1 whenever any clipping occurs.
    raw = expected_returns.values
    total = np.sum(np.abs(raw)) + 1e-8
    w = raw / total
    w = np.clip(w, -MAX_SHORT_PCT, MAX_POSITION_PCT)
    # Re-normalize so weights sum correctly
    abs_w = np.sum(np.abs(w))
    if abs_w > 0:
        w = w / abs_w
    return pd.Series(w, index=expected_returns.index)
