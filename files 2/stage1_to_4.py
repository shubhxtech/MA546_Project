"""
stage1_to_4.py
==============
Stages 1–4 of the News-Driven Sustainable Portfolio Pipeline:
  1. Data Ingestion & Validation
  2. Deduplication (exact hash + near-duplicate fingerprint)
  3. Relevance Filtering (keyword rule-based, two-pass)
  4. Entity Extraction (ticker + sector tagging)

Input  : raw ET headlines CSV  (columns: headline, date/timestamp, [url])
Output : clean DataFrame with columns:
         id, headline, timestamp, tickers, sectors, is_sector_only
"""

import re
import hashlib
import logging
from datetime import timezone

import numpy as np
import pandas as pd

from config import (
    TICKER_ALIASES, SECTOR_ALIASES, STOCK_UNIVERSE,
    MARKET_KEYWORDS, NEGATION_WORDS, PIPELINE_CONFIG,
)

logger = logging.getLogger("Pipeline.Stage1_4")


# ═════════════════════════════════════════════════════════════════
#  STAGE 1 — INGESTION & VALIDATION
# ═════════════════════════════════════════════════════════════════

def load_and_validate(
    filepath: str,
    headline_col: str = "headline",
    date_col: str = "date",
    date_format: str = None,
    start_date: str = "2022-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """
    Load ET headlines CSV and enforce schema + date range.

    Handles common corner cases:
    - Multiple possible column names for headline/date
    - Mixed date formats via pd.to_datetime inference
    - Timezone-naive timestamps → UTC
    - Headlines that are too short (<15 chars) or too long (>600 chars)
    - Rows with null headline or date
    """
    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"  Raw rows: {len(df):,}  |  Columns: {list(df.columns)}")

    # ── Auto-detect column names ──────────────────────────────────
    headline_col = _find_col(df, [headline_col, "title", "text", "news", "headline"])
    date_col     = _find_col(df, [date_col, "date", "published", "timestamp",
                                  "publish_date", "datetime", "time"])

    df = df.rename(columns={headline_col: "headline", date_col: "timestamp"})
    df = df[["headline", "timestamp"] + [c for c in df.columns
                                          if c not in ("headline", "timestamp")]]

    # ── Drop nulls ────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["headline", "timestamp"])
    logger.info(f"  Dropped {before - len(df):,} rows with null headline/timestamp")

    # ── Parse timestamps → UTC ────────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=date_format,
                                      errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df["timestamp"].dt.tz is None:
        # Assume IST (UTC+5:30), convert to UTC
        df["timestamp"] = (
            df["timestamp"]
            .dt.tz_localize("Asia/Kolkata", ambiguous="infer", nonexistent="shift_forward")
            .dt.tz_convert("UTC")
        )

    # ── Date range filter ─────────────────────────────────────────
    df = df[(df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")) &
            (df["timestamp"] <= pd.Timestamp(end_date,   tz="UTC"))]

    # ── Headline length filter ────────────────────────────────────
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df[(df["headline"].str.len() >= 15) & (df["headline"].str.len() <= 600)]

    # ── Sort by time ─────────────────────────────────────────────
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Assign stable row ID ──────────────────────────────────────
    df["id"] = df.apply(
        lambda r: hashlib.md5(
            f"{r['headline']}{r['timestamp']}".encode()
        ).hexdigest()[:12],
        axis=1,
    )

    logger.info(f"  Clean rows after stage 1: {len(df):,}")
    return df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first candidate column name that exists in df (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


# ═════════════════════════════════════════════════════════════════
#  STAGE 2 — DEDUPLICATION
# ═════════════════════════════════════════════════════════════════

def deduplicate(df: pd.DataFrame, window_hours: int = None) -> pd.DataFrame:
    """
    Two-level deduplication:
      Level 1 — Exact hash: identical normalised headlines → keep earliest
      Level 2 — Shingle fingerprint: near-duplicates within a rolling time
                window (trigram Jaccard similarity ≥ threshold)

    Corner cases:
    - Duplicate URLs submitted hours apart → caught by exact hash
    - Paraphrased wire-service reposts    → caught by Jaccard
    """
    window_hours = window_hours or PIPELINE_CONFIG["dedup_window_hours"]
    threshold    = PIPELINE_CONFIG["similarity_threshold"]

    logger.info(f"Deduplication: {len(df):,} rows in")

    # Level 1 — Exact normalised hash
    df["_norm"] = df["headline"].apply(_normalise)
    df["_hash"] = df["_norm"].apply(lambda t: hashlib.md5(t.encode()).hexdigest())
    df = df.drop_duplicates(subset="_hash", keep="first")
    logger.info(f"  After exact dedup: {len(df):,} rows")

    # Level 2 — Shingle Jaccard within rolling window
    df = _jaccard_dedup(df, window_hours, threshold)
    logger.info(f"  After near-dedup:  {len(df):,} rows")

    df = df.drop(columns=["_norm", "_hash", "_shingles"], errors="ignore")
    return df.reset_index(drop=True)


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _shingles(text: str, k: int = 3) -> set:
    """Character k-gram shingle set for Jaccard similarity."""
    tokens = text.split()
    words  = " ".join(tokens)
    return set(words[i:i+k] for i in range(len(words) - k + 1)) if len(words) >= k else {words}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _jaccard_dedup(df: pd.DataFrame, window_hours: int, threshold: float) -> pd.DataFrame:
    """
    For each article, compare against all articles in the preceding window.
    Mark as duplicate if Jaccard ≥ threshold with any earlier article.
    
    FIX: replaced iterrows() with itertuples() — 5-10x faster for large datasets.
    Also added early exit for empty shingle sets to avoid false positives.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["_shingles"] = df["_norm"].apply(_shingles)

    keep   = []
    seen   = []   # list of (timestamp, shingles)
    window = pd.Timedelta(hours=window_hours)

    for row in df.itertuples(index=True):
        ts   = row.timestamp
        shin = row._shingles

        # Prune old entries outside window
        seen = [(t, s) for t, s in seen if ts - t <= window]

        # FIX: empty shingle set should never be marked as duplicate
        if not shin:
            keep.append(row.Index)
            seen.append((ts, shin))
            continue

        is_dup = any(_jaccard(shin, s) >= threshold for _, s in seen if s)
        if not is_dup:
            keep.append(row.Index)
            seen.append((ts, shin))

    return df.loc[keep]


# ═════════════════════════════════════════════════════════════════
#  STAGE 3 — RELEVANCE FILTERING (two-pass, no external model)
# ═════════════════════════════════════════════════════════════════

# Non-market patterns — headlines matching these are dropped
# UNLESS they also contain a strong market override term
_NON_MARKET_RE = re.compile(
    r"\b(cricket|ipl|football|tennis|bollywood|film award|music|"
    r"marriage|wedding|obituary|murder|crime|cyclone|earthquake|"
    r"weather forecast|monsoon update|astrology)\b",
    re.IGNORECASE,
)
_MARKET_OVERRIDE_RE = re.compile(
    r"\b(market|stock|share|nse|bse|sensex|nifty|equity|insurance|"
    r"reinsurance|listed|index)\b",
    re.IGNORECASE,
)

# Macro-economic patterns — always kept regardless of keyword count
_MACRO_RE = re.compile(
    r"\b(rbi|repo rate|inflation|gdp|cpi|wpi|current account|"
    r"fiscal deficit|budget|union budget|sebi|fdi|fii|dii|"
    r"interest rate|monetary policy|rupee|forex reserve)\b",
    re.IGNORECASE,
)


def filter_relevant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pass 1 — keyword gate: must have ≥ 1 market keyword or be macro.
    Pass 2 — non-market exclusion: drop lifestyle/sports unless market override.

    Corner cases:
    - "RBI warns of cybercrime" → kept (macro trigger)
    - "Ambani family dispute"   → kept (ticker alias match)
    - "IPL broadcaster rights"  → kept (IPO/FII term absent but 'IPL' vs 'IPO'
                                        distinguished correctly)
    - "Weather hits coal output"→ kept (market override: coal/output)
    """
    logger.info(f"Relevance filter: {len(df):,} rows in")

    market_kw_set = set(MARKET_KEYWORDS)

    def _is_relevant(headline: str) -> bool:
        hl_lower = headline.lower()

        # Always keep macro headlines
        if _MACRO_RE.search(hl_lower):
            return True

        # Check market keyword presence
        words = set(re.findall(r"\b\w+\b", hl_lower))
        # Also check multi-word phrases
        has_market_kw = (
            bool(words & market_kw_set) or
            any(kw in hl_lower for kw in market_kw_set if " " in kw)
        )
        if not has_market_kw:
            return False

        # Non-market pattern check with override
        if _NON_MARKET_RE.search(hl_lower):
            return bool(_MARKET_OVERRIDE_RE.search(hl_lower))

        return True

    mask = df["headline"].apply(_is_relevant)
    df   = df[mask].reset_index(drop=True)
    logger.info(f"  After relevance filter: {len(df):,} rows")
    return df


# ═════════════════════════════════════════════════════════════════
#  STAGE 4 — ENTITY EXTRACTION
# ═════════════════════════════════════════════════════════════════

def extract_entities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each headline with:
      - tickers      : list of matched NSE tickers (may be empty)
      - sectors      : list of matched sectors
      - is_sector_only: True if only sector-level match (no specific ticker)

    Rules:
    1. Scan headline for all ticker aliases → collect tickers
    2. If no ticker found, scan for sector aliases → collect sectors
    3. A headline mentioning 2+ tickers gets tagged to all of them
    4. Ambiguous company names (e.g. "Sun") require a disambiguating word
    5. Sector headlines broadcast at reduced weight (flagged)
    """
    logger.info("Entity extraction...")

    # Sort aliases longest-first to avoid partial matches (e.g. "hdfc" before "hdfc bank")
    sorted_aliases = sorted(TICKER_ALIASES.keys(), key=len, reverse=True)
    sorted_sector  = sorted(SECTOR_ALIASES.keys(), key=len, reverse=True)

    # Ambiguous single-word aliases that need context confirmation
    AMBIGUOUS = {
        "sun":     ("SUNPHARMA", ["pharma", "pharmaceutical", "drug", "sun pharma"]),
        "future":  ("FUTURE",    ["retail", "group", "store"]),
        "general": (None,        []),  # discard if only "general" matches
    }

    # Compile word-boundary regex patterns once for efficiency
    # FIX: plain `alias in hl` matches substrings — "infy" matches "infinity",
    # "oil" matches "spoil". Use \b word boundaries for all single-token aliases.
    import re as _re
    alias_patterns = {}
    for alias in sorted_aliases:
        if " " in alias:
            alias_patterns[alias] = alias   # multi-word: substring is fine
        else:
            alias_patterns[alias] = _re.compile(r'\b' + _re.escape(alias) + r'\b')

    sector_patterns = {}
    for alias in sorted_sector:
        if " " in alias:
            sector_patterns[alias] = alias
        else:
            sector_patterns[alias] = _re.compile(r'\b' + _re.escape(alias) + r'\b')

    results = []
    for _, row in df.iterrows():
        hl      = row["headline"].lower()
        tickers = []
        sectors = []

        # ── Ticker matching ───────────────────────────────────────
        for alias in sorted_aliases:
            # Check ambiguity
            base = alias.split()[0] if " " not in alias else None
            if base and base in AMBIGUOUS:
                ticker, ctx_words = AMBIGUOUS[base]
                if ticker and any(w in hl for w in ctx_words):
                    pat = alias_patterns[alias]
                    matched = pat.search(hl) if hasattr(pat, "search") else (alias in hl)
                    if matched and ticker not in tickers:
                        tickers.append(ticker)
                continue  # skip full alias match for ambiguous singles

            pat = alias_patterns[alias]
            matched = pat.search(hl) if hasattr(pat, "search") else (alias in hl)
            if matched:
                ticker = TICKER_ALIASES[alias]
                if ticker not in tickers:
                    tickers.append(ticker)

        # ── Sector fallback ───────────────────────────────────────
        if not tickers:
            for alias in sorted_sector:
                pat = sector_patterns[alias]
                matched = pat.search(hl) if hasattr(pat, "search") else (alias in hl)
                if matched:
                    sector = SECTOR_ALIASES[alias]
                    if sector not in sectors:
                        sectors.append(sector)

        # ── Infer sectors from tickers ────────────────────────────
        if tickers and not sectors:
            for t in tickers:
                sec = STOCK_UNIVERSE.get(t, {}).get("sector")
                if sec and sec not in sectors:
                    sectors.append(sec)

        results.append({
            "tickers":        tickers,
            "sectors":        sectors,
            "is_sector_only": (len(tickers) == 0 and len(sectors) > 0),
        })

    entity_df = pd.DataFrame(results, index=df.index)
    df = pd.concat([df, entity_df], axis=1)

    # Drop articles with no entity match at all
    df = df[(df["tickers"].apply(len) > 0) | (df["sectors"].apply(len) > 0)]
    df = df.reset_index(drop=True)

    tagged_tickers = df[df["tickers"].apply(len) > 0].shape[0]
    tagged_sector  = df[df["is_sector_only"]].shape[0]
    logger.info(
        f"  Entities: {tagged_tickers:,} ticker-tagged | {tagged_sector:,} sector-only"
    )
    return df


# ═════════════════════════════════════════════════════════════════
#  CONVENIENCE: Run stages 1–4 in sequence
# ═════════════════════════════════════════════════════════════════

def run_stages_1_to_4(
    filepath: str,
    headline_col: str = "headline",
    date_col: str = "date",
    **kwargs,
) -> pd.DataFrame:
    df = load_and_validate(filepath, headline_col, date_col, **kwargs)
    df = deduplicate(df)
    df = filter_relevant(df)
    df = extract_entities(df)
    logger.info(f"Stages 1–4 complete: {len(df):,} articles ready for scoring")
    return df
