import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import json
from pathlib import Path

REGIME_PARAMS_PATH = Path(__file__).parent / "regime_parameters.json"

class RegimeDetector:
    def __init__(self):
        self.model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
        self.is_fitted = False
        self.state_map = {} # maps hidden state index to regime name
        
    def _extract_features(self, nifty_hist: pd.Series, price_cache: pd.DataFrame) -> pd.DataFrame:
        """
        Features: NIFTY 5d return, 20d volatility, Advance-Decline ratio
        """
        if nifty_hist.empty:
            return pd.DataFrame()
            
        df = pd.DataFrame(index=nifty_hist.index)
        # NIFTY returns
        ret = nifty_hist.pct_change()
        df['ret_5d'] = nifty_hist.pct_change(5)
        df['vol_20d'] = ret.rolling(20).std() * np.sqrt(252)
        
        # Advance-Decline
        if not price_cache.empty:
            univ_ret = price_cache.pct_change()
            adv = (univ_ret > 0).sum(axis=1)
            dec = (univ_ret < 0).sum(axis=1)
            df['ad_ratio'] = adv / (dec + 1e-5)
        else:
            df['ad_ratio'] = 1.0 # fallback
            
        return df.dropna()

    def fit_and_predict(self, current_date: pd.Timestamp, nifty_hist: pd.Series, price_cache: pd.DataFrame):
        # We fit on all data available up to current_date to avoid lookahead
        hist = nifty_hist.loc[:current_date.strftime('%Y-%m-%d')]
        p_cache = price_cache.loc[:current_date.strftime('%Y-%m-%d')] if not price_cache.empty else price_cache
        
        features = self._extract_features(hist, p_cache)
        if len(features) < 100:
            return "Sideways", 1.0, {} # Not enough data
            
        X = features.values
        
        try:
            self.model.fit(X)
            self.is_fitted = True
            
            # Predict hidden states for the whole history
            hidden_states = self.model.predict(X)
            
            # Map states to Bull/Bear/Sideways based on mean return of that state
            state_returns = []
            for i in range(3):
                idx = (hidden_states == i)
                # Compute annualized return for periods in this state
                if sum(idx) > 0:
                    mean_ret = features['ret_5d'][idx].mean() * 50 # roughly annualized 5d return
                else:
                    mean_ret = 0.0
                state_returns.append((i, mean_ret))
                
            state_returns.sort(key=lambda x: x[1]) # sort by return ascending
            
            self.state_map = {
                state_returns[0][0]: "Bear",
                state_returns[1][0]: "Sideways",
                state_returns[2][0]: "Bull"
            }
            
            # Get current state
            curr_state_idx = hidden_states[-1]
            curr_regime = self.state_map[curr_state_idx]
            
            # Get probabilities
            probs = self.model.predict_proba(X[-1].reshape(1, -1))[0]
            confidence = probs[curr_state_idx]
            
            if confidence < 0.55:
                curr_regime = "AMBIGUOUS"
                
            # Load params
            params = {"allow_shorts": False, "max_single_stock_pct": 0.15, "quality_gate_modifier": 0}
            if REGIME_PARAMS_PATH.exists():
                with open(REGIME_PARAMS_PATH, "r") as f:
                    all_params = json.load(f)
                    if curr_regime in all_params:
                        params = all_params[curr_regime]
                    elif curr_regime == "AMBIGUOUS":
                        # Blend Bull and Sideways 50/50
                        p1 = all_params.get("Bull", {})
                        p2 = all_params.get("Sideways", {})
                        params["max_single_stock_pct"] = (p1.get("max_single_stock_pct", 0.15) + p2.get("max_single_stock_pct", 0.15)) / 2
                        params["allow_shorts"] = False
                        
            return curr_regime, float(confidence), params
            
        except Exception as e:
            print(f"[RegimeDetector] Error fitting HMM: {e}")
            return "Sideways", 1.0, {"allow_shorts": False, "max_single_stock_pct": 0.15}
