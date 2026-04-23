"""
News-Driven Trading Signal Pipeline
====================================
Dataset: Economic Times Headlines (India) 2022-2025
Model: DeBERTa-v3 for sentiment + NLI-based relevance filtering

Pipeline Stages:
  1. Data Ingestion & Validation
  2. Preprocessing & Deduplication
  3. Entity Extraction (Stock/Sector Tagging)
  4. Relevance Filtering (NLI + Rule-based)
  5. Sentiment Analysis (DeBERTa fine-tuned on financial text)
  6. Signal Aggregation & Scoring
  7. Risk Gating & Position Sizing
  8. Signal Output / Backtest Interface
"""

import re
import json
import hashlib
import logging
import warnings
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
from collections import defaultdict
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("NewsTrading")


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class PipelineConfig:
    sentiment_model: str = "ProsusAI/finbert"
    deberta_model: str = "cross-encoder/nli-deberta-v3-small"
    use_deberta: bool = True
    nli_entailment_threshold: float = 0.6
    keyword_boost: bool = True
    sentiment_confidence_min: float = 0.60
    mixed_sentiment_threshold: float = 0.15
    dedup_window_hours: int = 6
    similarity_threshold: float = 0.85
    signal_decay_hours: int = 24
    min_articles_for_signal: int = 2
    max_signal_age_hours: int = 48
    market_open: str = "09:15"
    market_close: str = "15:30"
    max_position_size_pct: float = 0.05
    blackout_minutes_before_close: int = 30
    output_path: str = "trading_signals.csv"

CONFIG = PipelineConfig()

# ─────────────────────────────────────────────
#  ENTITY DICTIONARY
# ─────────────────────────────────────────────

TICKER_ALIASES = {
    # Banking
    "HDFCBANK":     ["HDFC Bank", "HDFC", "Housing Development Finance"],
    "ICICIBANK":    ["ICICI Bank", "ICICI"],
    "SBIN":         ["SBI", "State Bank of India", "State Bank"],
    "KOTAKBANK":    ["Kotak Mahindra", "Kotak Bank"],
    "AXISBANK":     ["Axis Bank"],
    "INDUSINDBK":   ["IndusInd Bank", "IndusInd"],
    "BANDHANBNK":   ["Bandhan Bank"],
    "FEDERALBNK":   ["Federal Bank"],
    # IT
    "TCS":          ["Tata Consultancy", "TCS"],
    "INFY":         ["Infosys"],
    "WIPRO":        ["Wipro"],
    "HCLTECH":      ["HCL Technologies", "HCL Tech"],
    "TECHM":        ["Tech Mahindra"],
    "LTIM":         ["LTIMindtree", "Larsen Toubro Infotech", "LTI"],
    "MPHASIS":      ["Mphasis"],
    "PERSISTENT":   ["Persistent Systems"],
    "COFORGE":      ["Coforge", "NIIT Technologies"],
    # Energy & Oil
    "RELIANCE":     ["Reliance Industries", "RIL", "Reliance Jio", "Reliance Retail", "Mukesh Ambani"],
    "ONGC":         ["ONGC", "Oil and Natural Gas"],
    "COALINDIA":    ["Coal India"],
    "NTPC":         ["NTPC"],
    "POWERGRID":    ["Power Grid"],
    "TATAPOWER":    ["Tata Power"],
    "ADANIGREEN":   ["Adani Green", "Adani Energy"],
    "ADANIPORTS":   ["Adani Ports", "Mundra Port"],
    # Auto
    "TATAMOTORS":   ["Tata Motors", "Jaguar Land Rover", "JLR"],
    "MARUTI":       ["Maruti Suzuki", "Maruti"],
    "M&M":          ["Mahindra", "M&M"],
    "BAJAJ-AUTO":   ["Bajaj Auto"],
    "HEROMOTOCO":   ["Hero MotoCorp", "Hero Honda"],
    "EICHERMOT":    ["Eicher Motors", "Royal Enfield"],
    "TVSMOTOR":     ["TVS Motor"],
    # Pharma
    "SUNPHARMA":    ["Sun Pharma", "Sun Pharmaceutical"],
    "DRREDDY":      ["Dr Reddy", "Dr. Reddy's"],
    "CIPLA":        ["Cipla"],
    "DIVISLAB":     ["Divi's Laboratories", "Divis Lab"],
    "BIOCON":       ["Biocon"],
    "LUPIN":        ["Lupin"],
    # FMCG
    "NESTLEIND":    ["Nestle India", "Nestle", "Maggi"],
    "HINDUNILVR":   ["Hindustan Unilever", "HUL"],
    "ITC":          ["ITC", "India Tobacco"],
    "BRITANNIA":    ["Britannia"],
    "DABUR":        ["Dabur"],
    "MARICO":       ["Marico", "Parachute"],
    # Cement / Industrial
    "ULTRACEMCO":   ["UltraTech Cement", "UltraTech"],
    "SHREECEM":     ["Shree Cement"],
    "AMBUJACEM":    ["Ambuja Cement"],
    "LT":           ["L&T Limited", "Larsen Toubro Limited", "Larsen and Toubro", "L&T"],
    "BHEL":         ["BHEL", "Bharat Heavy Electricals"],
    # Finance / Insurance
    "HDFCLIFE":     ["HDFC Life"],
    "SBILIFE":      ["SBI Life"],
    "BAJAJFINSV":   ["Bajaj Finserv"],
    "BAJFINANCE":   ["Bajaj Finance"],
    "CHOLAFIN":     ["Cholamandalam", "Chola Finance"],
    # Consumer / Others
    "TITAN":        ["Titan", "Tanishq"],
    "PIDILITIND":   ["Pidilite", "Fevicol"],
    "ASIANPAINT":   ["Asian Paints"],
    "TATACONSUM":   ["Tata Consumer", "Tata Tea", "Tetley"],
    # === INDICES: used for NLP entity tagging ONLY, not tradeable ===
    # These map to keys starting with '__' so they're filtered before yfinance
    "__NIFTY":      ["Nifty", "NSE", "Nifty 50", "Indian market", "equity market"],
    "__BANKNIFTY":  ["Bank Nifty", "banking sector", "PSU banks"],
    "__SENSEX":     ["Sensex", "BSE", "Bombay Stock Exchange"],
}

SECTOR_MAP = {
    "BANKING":    ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK", "BANDHANBNK", "FEDERALBNK"],
    "IT":         ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "MPHASIS", "PERSISTENT", "COFORGE"],
    "ENERGY":     ["RELIANCE", "ONGC", "COALINDIA", "NTPC", "POWERGRID", "TATAPOWER", "ADANIGREEN", "ADANIPORTS"],
    "AUTO":       ["TATAMOTORS", "MARUTI", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "TVSMOTOR"],
    "PHARMA":     ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "BIOCON", "LUPIN"],
    "FMCG":       ["NESTLEIND", "HINDUNILVR", "ITC", "BRITANNIA", "DABUR", "MARICO", "TATACONSUM"],
    "INDUSTRIAL": ["ULTRACEMCO", "SHREECEM", "AMBUJACEM", "LT", "BHEL"],
    "FINANCE":    ["HDFCLIFE", "SBILIFE", "BAJAJFINSV", "BAJFINANCE", "CHOLAFIN"],
    "CONSUMER":   ["TITAN", "PIDILITIND", "ASIANPAINT"],
}

# Tickers that begin with __ are NLP-only context signals, never traded
NON_TRADEABLE_PREFIX = "__"

ALIAS_TO_TICKER = {}
for ticker, aliases in TICKER_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_TICKER[alias.lower()] = ticker

MARKET_KEYWORDS = [
    "stock", "share", "equity", "nse", "bse", "sensex", "nifty",
    "ipo", "fii", "dii", "earnings", "profit", "revenue", "quarterly",
    "results", "dividend", "merger", "acquisition", "buyback",
    "rbi", "repo rate", "inflation", "gdp", "trade deficit",
    "rupee", "dollar", "crude oil", "interest rate", "bond yield",
    "sebi", "mutual fund", "fpo", "delisting", "rights issue",
    "capex", "ebitda", "margin", "guidance", "outlook", "forecast",
    "rally", "fall", "surge", "crash", "correction", "bull", "bear",
    "investment", "foreign exchange", "forex", "commodity", "gold",
]

NON_MARKET_PATTERNS = [
    r"\bcricket\b", r"\bfootball\b", r"\bbolly\b", r"\bfilm\b",
    r"\bmarriage\b", r"\belection result\b(?!.*market)",
    r"\bcrime\b", r"\bmurder\b", r"\baward\b",
    r"\bweather\b", r"\bcyclone\b(?!.*insurance)",
]

# ─────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class NewsArticle:
    id: str
    headline: str
    timestamp: datetime
    source: str = "Economic Times"
    tickers: List[str] = field(default_factory=list)
    sectors: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    sentiment: str = "NEUTRAL"
    sentiment_score: float = 0.0
    nli_confidence: float = 0.0
    decayed_weight: float = 1.0


# ─────────────────────────────────────────────
#  NLP TRADING PIPELINE
# ─────────────────────────────────────────────

class NewsTradingPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = "mps"  # Will dynamically fallback to cpu below
        self._init_models()
        self.active_articles: List[NewsArticle] = []

    def _init_models(self):
        if not self.config.use_deberta:
            logger.info("DeBERTa models disabled. Operating in simulated testing mode.")
            return

        try:
            import torch
            from transformers import pipeline
            
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                
            logger.info(f"Loading HuggingFace modules on {self.device}...")
            # DeBERTa-v3 NLI zero-shot
            self.nli_classifier = pipeline(
                "zero-shot-classification", 
                model=self.config.deberta_model,
                device=self.device
            )
            # FinBERT-tone Sentiment
            self.sentiment_classifier = pipeline(
                "sentiment-analysis", 
                model=self.config.sentiment_model,
                device=self.device
            )
            logger.info("Transformers successfully loaded into memory.")
        except ImportError:
            logger.error("torch or transformers not found. Cannot load NLP architectures.")
            self.config.use_deberta = False
            self.device = "cpu"

    # STAGES 1 & 2: Ingestion & Deduplication
    def ingest(self, headline: str, timestamp_str: str) -> Optional[NewsArticle]:
        try:
            dt = pd.to_datetime(timestamp_str)
            if dt.tz is None: 
                dt = dt.tz_localize("Asia/Kolkata")
        except Exception:
            return None

        clean_hl = headline.strip()
        if len(clean_hl) < 10: 
            return None

        # Level 1 Hash Deduplication
        hash_id = hashlib.md5(f"{clean_hl.lower()}{dt}".encode()).hexdigest()[:12]
        
        # Check rolling window for near-duplicates (Jaccard omitted for brevity)
        for existing in self.active_articles:
            if existing.id == hash_id:
                return None
                
        article = NewsArticle(id=hash_id, headline=clean_hl, timestamp=dt)
        return article

    # STAGE 3: Entity Extraction
    def extract_entities(self, article: NewsArticle) -> NewsArticle:
        hl_lower = article.headline.lower()
        
        # Tag Tickers
        for alias, ticker in ALIAS_TO_TICKER.items():
            if alias in hl_lower and ticker not in article.tickers:
                article.tickers.append(ticker)
                
        # Tag Sectors implicitly
        for ticker in article.tickers:
            for sector_name, sector_tickers in SECTOR_MAP.items():
                if ticker in sector_tickers and sector_name not in article.sectors:
                    article.sectors.append(sector_name)
                    
        return article

    # STAGE 4: Relevance Filtering (NLI)
    def filter_relevance(self, article: NewsArticle) -> bool:
        hl_lower = article.headline.lower()
        
        # Rule-based drop
        if any(re.search(pat, hl_lower) for pat in NON_MARKET_PATTERNS):
            return False
            
        # DeBERTa NLI Gate
        if self.config.use_deberta:
            try:
                res = self.nli_classifier(
                    article.headline, 
                    candidate_labels=["financial market news", "corporate developments", "unrelated lifestyle news"]
                )
                if res["labels"][0] == "unrelated lifestyle news":
                    return False
                article.nli_confidence = res["scores"][0]
            except Exception as e:
                logger.error(f"NLI failure: {str(e)}")
                
        return True

    # STAGE 5: Sentiment Analysis
    def analyze_sentiment(self, article: NewsArticle) -> NewsArticle:
        if not self.config.use_deberta:
            pos = sum(1 for w in MARKET_KEYWORDS if w in article.headline.lower())
            neg = sum(1 for w in ["crash", "drop", "plunge", "loss", "fail"] if w in article.headline.lower())
            base_score = 1.0 if pos > neg else (-1.0 if neg > pos else 0.0)
            article.sentiment_score = base_score * 100.0
            article.sentiment = "POSITIVE" if article.sentiment_score > 0 else ("NEGATIVE" if article.sentiment_score < 0 else "NEUTRAL")
            return article
            
        try:
            # 1. FinBERT Score
            out_fin = self.sentiment_classifier(article.headline)[0]
            fin_label = out_fin['label'].upper()
            fin_score = out_fin['score']
            fin_numeric = fin_score if fin_label == "POSITIVE" else (-fin_score if fin_label == "NEGATIVE" else 0.0)
            
            # 2. DeBERTa Score (Zero-Shot Dual Layer)
            out_deb = self.nli_classifier(
                article.headline, 
                candidate_labels=["positive market outlook", "negative market outlook", "neutral outlook"]
            )
            # Find probability for positive and negative
            deb_probs = dict(zip(out_deb["labels"], out_deb["scores"]))
            deb_pos = deb_probs.get("positive market outlook", 0.0)
            deb_neg = deb_probs.get("negative market outlook", 0.0)
            deb_numeric = deb_pos - deb_neg  # Ranges roughly -1.0 to 1.0
            
            # 3. Average & Scale to -100 to +100
            combined_score = (fin_numeric + deb_numeric) / 2.0
            final_abs_score = combined_score * 100.0
            
            article.sentiment_score = final_abs_score
            article.sentiment = "POSITIVE" if final_abs_score > 10 else ("NEGATIVE" if final_abs_score < -10 else "NEUTRAL")
            
        except Exception as e:
            logger.error(f"Sentiment failure: {str(e)}")
            
        return article

    # STAGE 6: Signal Aggregation & Scoring
    def compute_decayed_signals(self, current_time: datetime) -> Dict[str, float]:
        signal_book = defaultdict(float)
        
        # Drop stale articles
        cutoff = current_time - timedelta(hours=self.config.max_signal_age_hours)
        self.active_articles = [a for a in self.active_articles if a.timestamp >= cutoff]
        
        for art in self.active_articles:
            age_hours = (current_time - art.timestamp).total_seconds() / 3600
            if age_hours <= 0: age_hours = 0.1
            
            # Exponential decay
            decay_factor = np.exp(-0.693 * (age_hours / self.config.signal_decay_hours))
            art.decayed_weight = float(art.sentiment_score * decay_factor)
            
            if abs(art.decayed_weight) < 0.1:
                continue
                
            for tick in art.tickers:
                signal_book[tick] += art.decayed_weight
                
        return dict(signal_book)

    # STAGE 7: Risk Gating & Position Sizing
    def generate_target_weights(self, signals: Dict[str, float]) -> Dict[str, float]:
        weights = {}
        total_abs_score = sum(abs(v) for v in signals.values())
        if total_abs_score == 0:
            return weights
            
        for ticker, raw_score in signals.items():
            # Cap at max_position_size
            pct = raw_score / (total_abs_score + 1e-6) # Normalize
            target_w = np.clip(pct, -self.config.max_position_size_pct, self.config.max_position_size_pct)
            if abs(target_w) > 0.01:
                weights[ticker] = target_w
                
        return weights

    # STAGE 8: Main Orchestrator / Pipeline Runner
    def process_live_headline(self, headline: str, timestamp: datetime) -> Optional[Dict[str, float]]:
        # Stage 1/2
        art = self.ingest(headline, timestamp)
        if not art: return None
        
        # Stage 3
        art = self.extract_entities(art)
        if not art.tickers and not art.sectors: return None
        
        # Stage 4
        if not self.filter_relevance(art): return None
        
        # Stage 5
        art = self.analyze_sentiment(art)
        
        self.active_articles.append(art)
        
        # Stage 6/7
        signals = self.compute_decayed_signals(timestamp)
        weights = self.generate_target_weights(signals)
        return weights


if __name__ == "__main__":
    import time
    
    # Simple Pipeline Test
    logger.info("Initializing Test NLP Pipeline...")
    pipeline = NewsTradingPipeline(CONFIG)
    
    test_headlines = [
        "HDFC Bank posts record Q3 profit up 25% YoY due to strong retail business",
        "TCS faces massive downgrade after failing to hit revenue guidance",
        "IPL Final scores huge TRP ratings across nation", # Should be ignored
        "Reliance launches $5B clean energy solar farm project",
    ]
    
    dt = pd.Timestamp.now(tz="Asia/Kolkata")
    
    logger.info("-" * 40)
    for i, h in enumerate(test_headlines):
        # We simulate them occurring linearly in the past
        sim_dt = dt - timedelta(hours=(len(test_headlines)-i)*2)
        logger.info(f"Processing: {h}")
        res = pipeline.process_live_headline(h, sim_dt)
        if res:
            logger.info(f"Target Returns/Allocations: {res}")
        else:
            logger.info(f"Filtered out or No Signal Generated.")
        logger.info("-" * 40)
        
    logger.info("Pipeline test execution complete.")