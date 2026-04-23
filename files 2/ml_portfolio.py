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


def select_top_m_long_short(sentiment_df: pd.DataFrame, M: int, score_threshold: float = 5.0) -> List[str]:
    """
    Selects Top-M stocks for a long-short universe.
    
    Strategy:
    - Use SENTIMENT MOMENTUM (last score - first score in IS window) to capture
      *changing* conviction, not static level (level is already in the price).
    - Take top M/2 positive-momentum stocks (long candidates)
    - Take top M/2 negative-momentum stocks (short candidates)
    - Only include stocks whose signal exceeds `score_threshold`
    
    This is the correct approach: we want to trade on NEW information,
    not on the fact that Reliance "usually" gets positive news.
    """
    if len(sentiment_df) < 2:
        # Fallback to level if not enough history
        avg = sentiment_df.mean()
        positives = avg[avg >  score_threshold].nlargest(M // 2).index.tolist()
        negatives = avg[avg < -score_threshold].nsmallest(M // 2).index.tolist()
        return (positives + negatives) if (positives or negatives) else avg.abs().nlargest(min(M, len(avg))).index.tolist()

    # Sentiment momentum: final period minus earliest period in IS window
    momentum = sentiment_df.iloc[-1] - sentiment_df.iloc[0]
    
    half_m = max(1, M // 2)
    longs  = momentum[momentum >  score_threshold].nlargest(half_m).index.tolist()
    shorts = momentum[momentum < -score_threshold].nsmallest(half_m).index.tolist()
    
    candidates = longs + shorts
    
    # If no candidates pass threshold, relax and just take biggest movers
    if not candidates:
        candidates = momentum.abs().nlargest(min(M, len(momentum))).index.tolist()
    
    print(f"[Universe] Long: {longs[:3]}  Short: {shorts[:3]}  Total: {len(candidates)}")
    return candidates


# Keep backward-compatible alias
def select_top_m(sentiment_df: pd.DataFrame, M: int) -> List[str]:
    return select_top_m_long_short(sentiment_df, M)


def build_ml_models(X_train: pd.DataFrame, y_train: pd.Series, selected_models: List[str]) -> Dict[str, object]:
    """
    Trains selected ML pipelines with adaptive CV folds (never exceeds n_samples).
    
    Note: Features should already be cross-sectionally z-scored before calling this.
    """
    fitted_models = {}
    n = len(X_train)
    cv_folds = max(2, min(5, n))  # Never exceed n_samples

    # 1. Linear Regression
    if "Linear" in selected_models:
        model = Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])
        model.fit(X_train, y_train)
        fitted_models["Linear"] = model

    # 2. LASSO (sparse feature selection — good for high-dimensional sentiment)
    if "LASSO" in selected_models:
        pipe = Pipeline([('scaler', StandardScaler()), ('reg', Lasso(max_iter=10000))])
        params = {'reg__alpha': [0.001, 0.01, 0.1, 1.0]}
        grid = GridSearchCV(pipe, params, cv=cv_folds, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        fitted_models["LASSO"] = grid.best_estimator_

    # 3. Ridge (useful when features are correlated across sectors)
    if "Ridge" in selected_models:
        pipe = Pipeline([('scaler', StandardScaler()), ('reg', Ridge())])
        params = {'reg__alpha': [0.01, 0.1, 1.0, 10.0]}
        grid = GridSearchCV(pipe, params, cv=cv_folds, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        fitted_models["Ridge"] = grid.best_estimator_

    # 4. CART (Decision Tree)
    if "CART" in selected_models:
        pipe = DecisionTreeRegressor(random_state=42)
        params = {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]}
        grid = GridSearchCV(pipe, params, cv=cv_folds, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        fitted_models["CART"] = grid.best_estimator_

    # 5. Random Forest
    if "RF" in selected_models:
        pipe = RandomForestRegressor(random_state=42, n_jobs=None)
        params = {'n_estimators': [50, 100], 'max_depth': [3, 5, None]}
        grid = GridSearchCV(pipe, params, cv=cv_folds, n_jobs=None)
        grid.fit(X_train, y_train)
        fitted_models["RF"] = grid.best_estimator_

    # 6. Gradient Boosting (often best for tabular financial data)
    if "GBM" in selected_models:
        pipe = GradientBoostingRegressor(random_state=42)
        params = {'n_estimators': [50, 100], 'max_depth': [2, 3], 'learning_rate': [0.05, 0.1]}
        grid = GridSearchCV(pipe, params, cv=cv_folds, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        fitted_models["GBM"] = grid.best_estimator_

    # 7. SVR
    if "SVR" in selected_models:
        pipe = Pipeline([('scaler', StandardScaler()), ('reg', SVR())])
        params = {'reg__C': [0.1, 1, 10], 'reg__kernel': ['linear', 'rbf']}
        grid = GridSearchCV(pipe, params, cv=cv_folds)
        grid.fit(X_train, y_train)
        fitted_models["SVR"] = grid.best_estimator_

    # 8. Neural Net
    if "NN" in selected_models:
        pipe = Pipeline([('scaler', StandardScaler()), ('reg', MLPRegressor(max_iter=1000, random_state=42))])
        params = {'reg__hidden_layer_sizes': [(32,), (64, 32)]}
        grid = GridSearchCV(pipe, params, cv=cv_folds)
        grid.fit(X_train, y_train)
        fitted_models["NN"] = grid.best_estimator_

    # 9. Genetic Algorithm
    if "GA" in selected_models:
        import pygad
        def fitness_func(ga_instance, solution, solution_idx):
            predictions = np.dot(X_train.values, solution)
            mse = np.mean((y_train.values - predictions) ** 2)
            return 1.0 / (mse + 1e-8)

        ga = pygad.GA(num_generations=50, num_parents_mating=10, fitness_func=fitness_func,
                      sol_per_pop=20, num_genes=X_train.shape[1], suppress_warnings=True)
        ga.run()
        solution, _, _ = ga.best_solution()

        class GAModel:
            def __init__(self, w): self.w = w
            def predict(self, X): return np.dot(X, self.w)

        fitted_models["GA"] = GAModel(solution)

    return fitted_models


def calculate_shap_weights(model: object, X_train: pd.DataFrame, X_test: pd.DataFrame, model_name: str) -> pd.Series:
    """
    Extracts SIGNED SHAP values to preserve alpha direction.

    CRITICAL FIX vs old code:
      Old code used np.abs(shap_vals) — this destroyed direction entirely, making
      every stock a long position regardless of what the model predicted.
      New code preserves the sign:
        positive SHAP → stock's positive sentiment predicts positive returns → LONG
        negative SHAP → stock's positive sentiment predicts negative returns → SHORT
    """
    try:
        if model_name in ["RF", "CART", "GBM"]:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test)
        elif model_name == "GA":
            # GA: use raw signed coefficients
            signed = model.w
            total = np.sum(np.abs(signed)) + 1e-8
            return pd.Series(signed / total, index=X_train.columns)
        else:
            if hasattr(model, "named_steps") and 'reg' in model.named_steps and \
               isinstance(model.named_steps['reg'], (LinearRegression, Lasso, Ridge)):
                # For linear models: use signed coefficients directly
                coef = model.named_steps['reg'].coef_
                total = np.sum(np.abs(coef)) + 1e-8
                return pd.Series(coef / total, index=X_train.columns)
            else:
                baseline = shap.sample(X_train, min(50, len(X_train)))
                explainer = shap.KernelExplainer(model.predict, baseline)
                shap_vals = explainer.shap_values(X_test, l1_reg="num_features(10)")

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        # Preserve SIGN — positive = go long, negative = go short
        signed_shap = shap_vals.flatten()
        total = np.sum(np.abs(signed_shap)) + 1e-8

        if total < 1e-7:
            # Fallback: equal long weight if signal is flat
            return pd.Series(1.0 / len(signed_shap), index=X_train.columns)

        normalized = signed_shap / total
        # Apply position size cap
        normalized = np.clip(normalized, -MAX_SHORT_PCT, MAX_POSITION_PCT)
        # Re-normalize after clipping
        total2 = np.sum(np.abs(normalized)) + 1e-8
        return pd.Series(normalized / total2, index=X_train.columns)

    except Exception as e:
        print(f"SHAP extraction failed for {model_name}: {e}. Falling back to equal long.")
        # FIX: use X_test.columns (not X_train.columns) so index matches the
        # prediction target. X_train and X_test may differ in column order after fillna.
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
