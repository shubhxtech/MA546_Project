"""
stage5_6.py
===========
Stage 5 — Sentiment Scoring   (rule-based lexicon; swap in DeBERTa when available)
Stage 6 — Score Consolidation  (Sentiment + ESG proxy + SDG proxy per ticker per day)
           → Feng et al. Stage 9 equivalent

Output: daily_scores DataFrame indexed by (date, ticker) with columns:
        sentiment_score, esg_score, sdg_score, composite_score,
        article_count, confidence, is_imputed
"""

import logging
import warnings
from itertools import chain

import numpy as np
import pandas as pd

from config import (
    POSITIVE_WORDS, NEGATIVE_WORDS, NEGATION_WORDS,
    ESG_KEYWORDS, SDG_KEYWORDS, STOCK_UNIVERSE,
    PIPELINE_CONFIG,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger("Pipeline.Stage5_6")


# ═════════════════════════════════════════════════════════════════
#  STAGE 5 — SENTIMENT SCORING
# ═════════════════════════════════════════════════════════════════

def _lexicon_sentiment(headline: str) -> dict:
    """
    Rule-based lexicon sentiment scorer (no external model required).
    Returns: {label: str, pos: float, neg: float, neu: float, confidence: float}

    Logic:
      1. Count positive / negative keyword hits
      2. Apply negation window: if a negation word precedes a sentiment word
         within 3 tokens, flip that word's contribution
      3. Normalise to probabilities
      4. Apply neutral band: if |pos-neg| < threshold → NEUTRAL
    """
    hl_lower = headline.lower()
    tokens   = hl_lower.split()
    n        = len(tokens)

    pos_count = 0.0
    neg_count = 0.0

    for i, tok in enumerate(tokens):
        # Check 3-token negation window before current token
        window_start = max(0, i - 3)
        negated = any(tokens[j] in NEGATION_WORDS for j in range(window_start, i))

        if tok in POSITIVE_WORDS:
            if negated:
                neg_count += 0.7  # negated positive → mostly negative
            else:
                pos_count += 1.0
        elif tok in NEGATIVE_WORDS:
            if negated:
                pos_count += 0.5  # negated negative → weakly positive
            else:
                neg_count += 1.0

    total = pos_count + neg_count
    if total == 0:
        return {"label": "NEUTRAL", "pos": 0.0, "neg": 0.0,
                "neu": 1.0, "confidence": 0.5}

    pos_p = pos_count / (total + 1e-9)
    neg_p = neg_count / (total + 1e-9)
    neu_p = max(0.0, 1.0 - pos_p - neg_p)

    # Intensity boost for superlatives / intensifiers in headline
    boost_words = ["record", "massive", "biggest", "historic", "worst",
                   "best", "surge", "crash", "soar", "plunge"]
    has_boost   = any(w in hl_lower for w in boost_words)
    if has_boost:
        pos_p = min(pos_p * 1.15, 1.0)
        neg_p = min(neg_p * 1.15, 1.0)
        # Renormalise
        s = pos_p + neg_p + neu_p
        pos_p, neg_p, neu_p = pos_p/s, neg_p/s, max(0, neu_p/s)

    confidence = max(pos_p, neg_p, neu_p)
    band       = PIPELINE_CONFIG["sentiment_neutral_band"]

    if abs(pos_p - neg_p) < band:
        label = "NEUTRAL"
    elif pos_p > neg_p:
        label = "POSITIVE"
    else:
        label = "NEGATIVE"

    return {"label": label, "pos": pos_p, "neg": neg_p,
            "neu": neu_p, "confidence": confidence}


def score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply sentiment scorer to every headline.
    Adds columns: sentiment_label, sentiment_pos, sentiment_neg,
                  sentiment_neu, sentiment_conf, sentiment_score
    sentiment_score ∈ [-1, 1]:  pos_p - neg_p
    """
    logger.info("Stage 5: Sentiment scoring...")
    conf_min = PIPELINE_CONFIG["sentiment_confidence_min"]

    results = df["headline"].apply(_lexicon_sentiment)
    df["sentiment_label"] = results.apply(lambda x: x["label"])
    df["sentiment_pos"]   = results.apply(lambda x: x["pos"])
    df["sentiment_neg"]   = results.apply(lambda x: x["neg"])
    df["sentiment_neu"]   = results.apply(lambda x: x["neu"])
    df["sentiment_conf"]  = results.apply(lambda x: x["confidence"])
    df["sentiment_score"] = df["sentiment_pos"] - df["sentiment_neg"]

    # Quarantine low-confidence articles (flag but keep for ESG/SDG scoring)
    df["low_confidence"] = df["sentiment_conf"] < conf_min
    quarantine_n = df["low_confidence"].sum()
    logger.info(f"  Quarantined (low confidence): {quarantine_n:,} articles")
    logger.info(f"  Score distribution:\n{df['sentiment_label'].value_counts().to_string()}")
    return df


# ═════════════════════════════════════════════════════════════════
#  ESG & SDG TAGGING  (per article)
# ═════════════════════════════════════════════════════════════════

def _tag_esg(headline: str) -> dict:
    """Return ESG sub-scores: {E: 0/1, S: 0/1, G: 0/1}."""
    hl = headline.lower()
    return {
        "E": int(any(kw in hl for kw in ESG_KEYWORDS["E"])),
        "S": int(any(kw in hl for kw in ESG_KEYWORDS["S"])),
        "G": int(any(kw in hl for kw in ESG_KEYWORDS["G"])),
    }


def _tag_sdg(headline: str) -> dict:
    """Return SDG hit flags: {SDG7: 0/1, ..., SDG17: 0/1}."""
    hl = headline.lower()
    return {
        sdg: int(any(kw in hl for kw in kws))
        for sdg, kws in SDG_KEYWORDS.items()
    }


def tag_esg_sdg(df: pd.DataFrame) -> pd.DataFrame:
    """Add ESG and SDG tag columns to article-level DataFrame."""
    logger.info("Tagging ESG & SDG dimensions...")
    esg_tags = df["headline"].apply(_tag_esg).apply(pd.Series)
    esg_tags.columns = ["esg_E", "esg_S", "esg_G"]

    sdg_tags = df["headline"].apply(_tag_sdg).apply(pd.Series)

    df = pd.concat([df, esg_tags, sdg_tags], axis=1)
    df["is_esg"] = (df[["esg_E", "esg_S", "esg_G"]].sum(axis=1) > 0).astype(int)
    df["is_sdg"] = (df[[c for c in df.columns if c.startswith("SDG")]].sum(axis=1) > 0).astype(int)
    return df


# ═════════════════════════════════════════════════════════════════
#  STAGE 6 — DAILY SCORE AGGREGATION  (Feng Stage 9)
# ═════════════════════════════════════════════════════════════════

def _time_decay_weight(timestamps: pd.Series, halflife_hours: float) -> np.ndarray:
    """
    Exponential time-decay weights relative to the most recent article.
    w_i = exp(-λ * Δt_i)  where λ = ln(2) / halflife
    """
    max_ts = timestamps.max()
    delta_hours = (max_ts - timestamps).dt.total_seconds() / 3600
    lam = np.log(2) / halflife_hours
    return np.exp(-lam * delta_hours.values)


def _expand_to_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand articles tagged to multiple tickers into one row per ticker.
    Sector-only articles are broadcast to ALL tickers in that sector
    with a 0.4 weight multiplier (sector_broadcast flag).
    """
    rows = []
    sector_to_tickers = {}
    for ticker, info in STOCK_UNIVERSE.items():
        sec = info["sector"]
        sector_to_tickers.setdefault(sec, []).append(ticker)

    for _, row in df.iterrows():
        if row["tickers"]:
            for ticker in row["tickers"]:
                r = row.to_dict()
                r["ticker"]           = ticker
                r["sector_broadcast"] = False
                r["broadcast_weight"] = 1.0
                rows.append(r)
        elif row["is_sector_only"]:
            for sector in row["sectors"]:
                for ticker in sector_to_tickers.get(sector, []):
                    r = row.to_dict()
                    r["ticker"]           = ticker
                    r["sector_broadcast"] = True
                    r["broadcast_weight"] = 0.4   # Feng-style partial weight
                    rows.append(r)

    return pd.DataFrame(rows)


def aggregate_daily_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (date, ticker), aggregate article-level scores into:
      sentiment_score ∈ [-1,1]  — time-decay weighted average
      esg_score       ∈ [-1,1]  — coverage-weighted directional ESG
      sdg_score       ∈ [0,1]   — fraction of articles with any SDG tag
      composite_score ∈ [-1,1]  — weighted combination (Feng weights)
      article_count             — number of articles (after quality filter)
      confidence                — average confidence of included articles

    Corner cases handled:
    - Zero articles for a ticker on a day → score = NaN (forward-filled later)
    - All low-confidence articles → use them but flag low reliability
    - Contradictory signals (mixed pos/neg with high counts) → score damped
    """
    logger.info("Stage 6: Daily score aggregation...")

    halflife = PIPELINE_CONFIG["signal_decay_halflife_hours"]
    min_art  = PIPELINE_CONFIG["min_articles_for_conviction"]
    w_cfg    = PIPELINE_CONFIG["composite_weights"]
    sdg_cols = [c for c in df.columns if c.startswith("SDG")]

    df["date"] = df["timestamp"].dt.date
    long_df    = _expand_to_tickers(df)

    records = []

    for (date, ticker), grp in long_df.groupby(["date", "ticker"]):
        # Drop low-confidence rows only if we have enough without them
        high_conf = grp[~grp["low_confidence"]]
        work_grp  = high_conf if len(high_conf) >= min_art else grp

        n = len(work_grp)
        if n == 0:
            continue

        # Time-decay weights × broadcast weights
        decay_w  = _time_decay_weight(work_grp["timestamp"], halflife)
        broad_w  = work_grp["broadcast_weight"].values
        weights  = decay_w * broad_w
        weights  = weights / weights.sum()          # normalise to sum=1

        # ── Sentiment score ───────────────────────────────────────
        sent_raw = (work_grp["sentiment_score"].values * weights).sum()

        # Contradiction damping: if both strong positive AND strong negative
        # articles exist, reduce conviction
        pos_frac = (work_grp["sentiment_score"] > 0.15).mean()
        neg_frac = (work_grp["sentiment_score"] < -0.15).mean()
        contradiction = min(pos_frac, neg_frac)     # high if both directions
        sent_score = sent_raw * (1 - contradiction)

        # ── ESG score ─────────────────────────────────────────────
        esg_coverage = work_grp["is_esg"].mean()    # fraction of ESG articles
        if esg_coverage > 0:
            esg_grp   = work_grp[work_grp["is_esg"] == 1]
            e_w       = weights[work_grp["is_esg"].values == 1]
            e_w       = e_w / e_w.sum() if e_w.sum() > 0 else e_w
            # ESG score = directional sentiment of ESG-tagged articles × coverage
            esg_sents  = esg_grp["sentiment_score"].values
            esg_direct = (esg_sents * e_w).sum()
            esg_score  = esg_direct * esg_coverage
        else:
            esg_score = 0.0

        # ── SDG score ─────────────────────────────────────────────
        sdg_coverage = work_grp["is_sdg"].mean()
        sdg_score    = sdg_coverage   # [0,1], no directionality (SDG is positive by nature)

        # ── Composite score (Feng weights) ────────────────────────
        composite = (
            w_cfg["sentiment"] * sent_score +
            w_cfg["esg"]       * esg_score +
            w_cfg["sdg"]       * sdg_score
        )
        composite = float(np.clip(composite, -1.0, 1.0))

        records.append({
            "date":             date,
            "ticker":           ticker,
            "sentiment_score":  round(sent_score, 4),
            "esg_score":        round(esg_score, 4),
            "sdg_score":        round(sdg_score, 4),
            "composite_score":  round(composite, 4),
            "article_count":    n,
            "confidence":       round(work_grp["sentiment_conf"].mean(), 4),
            "contradiction":    round(contradiction, 4),
            "low_reliability":  (len(high_conf) < min_art),
            "is_imputed":       False,
        })

    daily = pd.DataFrame(records)
    daily["date"] = pd.to_datetime(daily["date"])

    # ── Forward-fill missing days (up to N business days) ────────
    daily = _forward_fill_scores(daily)

    logger.info(f"  Daily score records: {len(daily):,}  "
                f"({daily['ticker'].nunique()} tickers × "
                f"{daily['date'].nunique()} days)")
    return daily


def _forward_fill_scores(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Create a full date-ticker grid and forward-fill missing scores
    up to PIPELINE_CONFIG['score_forward_fill_days'] business days.
    Missing days beyond this are left as NaN.
    """
    max_fill = PIPELINE_CONFIG["score_forward_fill_days"]
    score_cols = ["sentiment_score", "esg_score", "sdg_score", "composite_score"]

    tickers    = daily["ticker"].unique()
    date_range = pd.bdate_range(daily["date"].min(), daily["date"].max())

    full_idx   = pd.MultiIndex.from_product(
        [date_range, tickers], names=["date", "ticker"]
    )
    daily = (
        daily
        .set_index(["date", "ticker"])
        .reindex(full_idx)
    )
    # FIX: is_imputed was bool before reindex; newly inserted rows become NaN.
    # isna() correctly marks new rows as imputed=True, real rows stay False.
    daily["is_imputed"] = daily["is_imputed"].isna()

    # FIX: groupby on a MultiIndex level must use level= parameter, not column name.
    # The old code raised a KeyError on some pandas versions when ticker was in index.
    for col in score_cols:
        daily[col] = (
            daily.groupby(level="ticker")[col]
            .transform(lambda s: s.ffill(limit=max_fill))
        )

    return daily.reset_index()


# ═════════════════════════════════════════════════════════════════
#  CONVENIENCE: Run stages 5–6 in sequence
# ═════════════════════════════════════════════════════════════════

def run_stages_5_to_6(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_articles  — article-level DataFrame with all scores
      daily_scores — daily (date, ticker) aggregated score DataFrame
    """
    df = score_sentiment(df)
    df = tag_esg_sdg(df)
    daily_scores = aggregate_daily_scores(df)
    return df, daily_scores
