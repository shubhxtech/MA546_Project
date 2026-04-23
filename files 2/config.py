"""
config.py
=========
Central configuration for the News-Driven Sustainable Portfolio Pipeline.
Covers 25 large-cap NSE stocks across 6 sectors, chosen for:
  - High ET news coverage frequency
  - Sector diversity for portfolio construction
  - Availability of BRSR/ESG disclosures
"""

# ── Stock Universe ────────────────────────────────────────────────────────────
STOCK_UNIVERSE = {
    # Ticker (NSE) : { display name, sector, ESG tier (H/M/L prior) }
    "HDFCBANK":   {"name": "HDFC Bank",               "sector": "Banking",  "esg_prior": "H"},
    "ICICIBANK":  {"name": "ICICI Bank",               "sector": "Banking",  "esg_prior": "H"},
    "SBIN":       {"name": "State Bank of India",      "sector": "Banking",  "esg_prior": "M"},
    "KOTAKBANK":  {"name": "Kotak Mahindra Bank",      "sector": "Banking",  "esg_prior": "H"},
    "AXISBANK":   {"name": "Axis Bank",                "sector": "Banking",  "esg_prior": "M"},

    "TCS":        {"name": "Tata Consultancy Services","sector": "IT",       "esg_prior": "H"},
    "INFY":       {"name": "Infosys",                  "sector": "IT",       "esg_prior": "H"},
    "WIPRO":      {"name": "Wipro",                    "sector": "IT",       "esg_prior": "H"},
    "HCLTECH":    {"name": "HCL Technologies",         "sector": "IT",       "esg_prior": "M"},
    "TECHM":      {"name": "Tech Mahindra",            "sector": "IT",       "esg_prior": "M"},

    "RELIANCE":   {"name": "Reliance Industries",      "sector": "Energy",   "esg_prior": "M"},
    "ONGC":       {"name": "ONGC",                     "sector": "Energy",   "esg_prior": "L"},
    "NTPC":       {"name": "NTPC Limited",             "sector": "Energy",   "esg_prior": "M"},
    "POWERGRID":  {"name": "Power Grid Corp",          "sector": "Energy",   "esg_prior": "M"},
    "ADANIGREEN": {"name": "Adani Green Energy",       "sector": "Energy",   "esg_prior": "H"},

    "TATAMOTORS": {"name": "Tata Motors",              "sector": "Auto",     "esg_prior": "M"},
    "MARUTI":     {"name": "Maruti Suzuki",            "sector": "Auto",     "esg_prior": "M"},
    "M&M":        {"name": "Mahindra & Mahindra",      "sector": "Auto",     "esg_prior": "M"},
    "BAJAJ-AUTO": {"name": "Bajaj Auto",               "sector": "Auto",     "esg_prior": "M"},

    "SUNPHARMA":  {"name": "Sun Pharmaceutical",       "sector": "Pharma",   "esg_prior": "H"},
    "DRREDDY":    {"name": "Dr Reddy's Labs",          "sector": "Pharma",   "esg_prior": "H"},
    "CIPLA":      {"name": "Cipla",                    "sector": "Pharma",   "esg_prior": "H"},

    "ASIANPAINT": {"name": "Asian Paints",             "sector": "Consumer", "esg_prior": "H"},
    "HINDUNILVR": {"name": "Hindustan Unilever",       "sector": "Consumer", "esg_prior": "H"},
    "ITC":        {"name": "ITC Limited",              "sector": "Consumer", "esg_prior": "M"},
}

SECTORS = list({v["sector"] for v in STOCK_UNIVERSE.values()})

# ── Ticker Alias Dictionary ───────────────────────────────────────────────────
# Maps lowercase text fragments → canonical ticker
TICKER_ALIASES = {
    # Banking
    "hdfc bank": "HDFCBANK", "hdfc": "HDFCBANK", "hdfcbank": "HDFCBANK",
    "housing development finance": "HDFCBANK",
    "icici bank": "ICICIBANK", "icici": "ICICIBANK",
    "state bank of india": "SBIN", "sbi": "SBIN", "state bank": "SBIN",
    "kotak mahindra bank": "KOTAKBANK", "kotak bank": "KOTAKBANK", "kotak": "KOTAKBANK",
    "axis bank": "AXISBANK",

    # IT
    "tata consultancy": "TCS", "tcs": "TCS",
    "infosys": "INFY", "infy": "INFY",
    "wipro": "WIPRO",
    "hcl technologies": "HCLTECH", "hcl tech": "HCLTECH", "hcltech": "HCLTECH",
    "tech mahindra": "TECHM",

    # Energy
    "reliance industries": "RELIANCE", "ril": "RELIANCE", "reliance": "RELIANCE",
    "reliance jio": "RELIANCE", "reliance retail": "RELIANCE",
    "ongc": "ONGC", "oil and natural gas": "ONGC",
    "ntpc": "NTPC",
    "power grid": "POWERGRID", "powergrid": "POWERGRID",
    "adani green": "ADANIGREEN", "adani green energy": "ADANIGREEN",

    # Auto
    "tata motors": "TATAMOTORS", "jaguar land rover": "TATAMOTORS", "jlr": "TATAMOTORS",
    "maruti suzuki": "MARUTI", "maruti": "MARUTI",
    "mahindra": "M&M", "m&m": "M&M",
    "bajaj auto": "BAJAJ-AUTO",

    # Pharma
    "sun pharma": "SUNPHARMA", "sun pharmaceutical": "SUNPHARMA",
    "dr reddy": "DRREDDY", "dr. reddy": "DRREDDY",
    "cipla": "CIPLA",

    # Consumer
    "asian paints": "ASIANPAINT",
    "hindustan unilever": "HINDUNILVR", "hul": "HINDUNILVR",
    "itc": "ITC", "itc limited": "ITC",
}

# Sector-level aliases — matched when no specific ticker found
SECTOR_ALIASES = {
    "banking": "Banking",  "banks": "Banking",  "bank nifty": "Banking",
    "psu bank": "Banking", "npa": "Banking",    "rbi": "Banking",
    "information technology": "IT", "it sector": "IT", "software": "IT",
    "tech sector": "IT",   "nasdaq": "IT",
    "oil": "Energy",       "crude": "Energy",   "petroleum": "Energy",
    "power sector": "Energy", "renewable": "Energy", "solar": "Energy",
    "automobile": "Auto",  "auto sector": "Auto", "ev": "Auto",
    "electric vehicle": "Auto",
    "pharma": "Pharma",    "pharmaceutical": "Pharma", "drug": "Pharma",
    "fmcg": "Consumer",    "consumer": "Consumer", "fmcg sector": "Consumer",
    "paints": "Consumer",
}

# ── ESG / SDG Keyword Taxonomy ────────────────────────────────────────────────
ESG_KEYWORDS = {
    "E": [
        "emission", "carbon", "climate", "pollution", "renewable", "solar", "wind",
        "green energy", "sustainability", "environment", "net zero", "fossil fuel",
        "water conservation", "biodiversity", "deforestation", "waste management",
        "clean energy", "ev", "electric vehicle", "carbon footprint", "esg",
    ],
    "S": [
        "employee", "labour", "labor", "workers", "community", "diversity", "inclusion",
        "safety", "health", "welfare", "social", "csr", "women empowerment",
        "gender equality", "minimum wage", "working conditions", "supply chain",
        "human rights", "charitable", "philanthropy",
    ],
    "G": [
        "board", "governance", "audit", "sebi", "compliance", "transparency",
        "promoter", "shareholding", "dividend", "buyback", "fraud", "corruption",
        "regulatory", "corporate governance", "independent director", "disclosure",
        "annual report", "brsr", "whistleblower",
    ],
}

SDG_KEYWORDS = {
    "SDG7":  ["solar", "wind", "clean energy", "renewable", "ntpc", "adani green",
               "power sector", "electricity access", "energy transition"],
    "SDG8":  ["employment", "jobs", "wage", "labour", "factory", "manufacturing",
               "workforce", "skill development", "gdp growth"],
    "SDG9":  ["infrastructure", "technology", "innovation", "5g", "digitalisation",
               "startup", "r&d", "research", "fdi"],
    "SDG13": ["climate", "net zero", "carbon", "emission", "paris agreement",
               "climate change", "global warming", "flood", "disaster"],
    "SDG17": ["partnership", "fdi", "trade", "export", "global", "investment",
               "bilateral", "economic cooperation"],
}

# ── Sentiment Keywords (rule-based fallback, no model required) ───────────────
POSITIVE_WORDS = [
    "profit", "growth", "surge", "rally", "gain", "rise", "beat", "record",
    "strong", "robust", "upgrade", "outperform", "expand", "acquisition",
    "revenue", "boom", "recovery", "positive", "bullish", "optimistic",
    "breakthrough", "contract", "order", "win", "launch", "increase",
    "dividend", "buyback", "inflow", "approval", "success",
]
NEGATIVE_WORDS = [
    "loss", "fall", "drop", "crash", "decline", "miss", "downgrade", "cut",
    "weak", "risk", "concern", "fraud", "penalty", "fine", "probe", "investigation",
    "slowdown", "layoff", "default", "debt", "bearish", "sell-off", "outflow",
    "ban", "reject", "warning", "slump", "negative", "underperform",
]
NEGATION_WORDS = ["no", "not", "without", "fails", "failed", "unable", "denies", "denied"]

# ── Pipeline Hyperparameters ──────────────────────────────────────────────────
PIPELINE_CONFIG = {
    # Deduplication
    "dedup_window_hours": 6,
    "similarity_threshold": 0.85,

    # Relevance
    "min_market_keywords": 1,

    # Sentiment
    "sentiment_neutral_band": 0.10,   # |pos-neg| < this → NEUTRAL
    "sentiment_confidence_min": 0.55,

    # Score aggregation
    # FIX: 12h half-life was too aggressive — overnight news lost >50% weight
    # before market open. 36h spans a full trading day + overnight properly.
    "signal_decay_halflife_hours": 36,
    # FIX: 5-day forward-fill propagated stale signals an entire week.
    # 3 days gives enough coverage over weekends without over-smoothing.
    "score_forward_fill_days": 3,
    "min_articles_for_conviction": 2,

    # Portfolio / Feng parameters
    # FIX: ESG weight 0.35 was overweighted relative to what keyword-only tagging
    # can reliably detect. Shifted weight to sentiment which has cleaner signal.
    "composite_weights": {           # Stage 10 ranking weights
        "sentiment": 0.50,
        "esg": 0.25,
        "sdg": 0.25,
    },
    "rolling_train_months": 12,      # Stage 11 rolling window
    "holding_period_days": 21,       # ~1 month
    "rebalance_freq": "ME",          # Month-End

    # Position constraints
    # FIX: 10% cap forced near-equal weights in a 10+ stock portfolio, defeating
    # the purpose of ML ranking. Raised to 15% so top-conviction picks matter.
    "max_stock_weight": 0.15,
    # FIX: 30% sector cap was frequently binding in Banking/IT heavy signals.
    # 35% gives slightly more room without losing diversification discipline.
    "max_sector_weight": 0.35,
    "min_stocks_in_portfolio": 8,
    "long_only": True,

    # Benchmark
    "equal_weight_benchmark": True,

    # Random seed
    "seed": 42,
}

# ── Market-relevant keywords (Stage 3 pass-1) ─────────────────────────────────
MARKET_KEYWORDS = [
    "stock", "share", "equity", "nse", "bse", "sensex", "nifty",
    "ipo", "fii", "dii", "earnings", "profit", "revenue", "quarterly",
    "results", "dividend", "merger", "acquisition", "buyback",
    "rbi", "repo rate", "inflation", "gdp", "trade deficit",
    "rupee", "dollar", "crude oil", "interest rate", "bond yield",
    "sebi", "mutual fund", "fpo", "delisting", "rights issue",
    "capex", "ebitda", "margin", "guidance", "outlook", "forecast",
    "rally", "fall", "surge", "crash", "correction", "bull", "bear",
    "investment", "forex", "commodity", "gold", "order", "contract",
] + list(TICKER_ALIASES.keys())
