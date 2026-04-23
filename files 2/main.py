"""
main.py
=======
Full pipeline orchestrator + visualisation for the
News-Driven Sustainable Portfolio System (Feng et al. adapted for NSE India).

Usage:
    python main.py --csv path/to/et_headlines.csv

    If --csv is not provided, a synthetic headlines dataset is generated
    automatically so the entire pipeline can be demonstrated end-to-end.

Output files written to ./outputs/:
    - daily_scores.csv
    - portfolio_returns.csv
    - portfolio_weights.csv
    - performance_metrics.csv
    - pipeline_report.png        (6-panel visualisation)
"""

import os
import logging
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from config import STOCK_UNIVERSE, PIPELINE_CONFIG, SECTORS
from stage1_to_4 import run_stages_1_to_4
from stage5_6 import run_stages_5_to_6
from stage7_8 import (
    fetch_online_prices,
    build_feature_matrix,
    run_ml_portfolio,
    compute_performance,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Pipeline.Main")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════
#  SYNTHETIC HEADLINE GENERATOR
#  (used when no real CSV is provided)
# ═════════════════════════════════════════════════════════════════

def generate_synthetic_headlines(n: int = 15000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic ET headlines dataset covering
    all 25 stocks, ESG/SDG themes, and varied sentiments.
    Ensures the pipeline has signal to learn from.
    """
    rng = np.random.default_rng(seed)

    TEMPLATES = {
        "positive_earnings": [
            "{name} posts record quarterly profit, beats estimates",
            "{name} revenue surges {pct}% YoY on strong demand",
            "{name} Q3 results: profit up {pct}%, analysts upgrade to buy",
            "{name} wins major contract worth Rs {val} crore",
            "{name} dividend declared at Rs {div} per share",
        ],
        "negative_earnings": [
            "{name} misses Q3 earnings estimates, margin falls",
            "{name} posts loss of Rs {val} crore amid slowdown",
            "{name} revenue declines {pct}% on weak demand",
            "{name} shares fall as guidance cut disappoints Street",
            "Analysts downgrade {name} after weak quarterly results",
        ],
        "esg_positive": [
            "{name} commits to net zero emissions by 2040",
            "{name} launches Rs {val} crore renewable energy project",
            "{name} wins SEBI ESG disclosure award for transparency",
            "{name} employees welfare programme covers 50,000 workers",
            "{name} board adds independent directors to boost governance",
        ],
        "esg_negative": [
            "SEBI probes {name} over corporate governance lapses",
            "{name} fined Rs {val} crore for environmental violations",
            "{name} faces regulatory scrutiny over board composition",
            "Workers protest at {name} plant over safety conditions",
        ],
        "sdg_positive": [
            "{name} signs Rs {val} crore clean energy partnership",
            "{name} to invest in skill development for 10,000 youth",
            "{name} solar capacity expansion supports India's SDG goals",
            "{name} joins global sustainability compact for green growth",
        ],
        "macro": [
            "RBI keeps repo rate unchanged at {rate}%, signals caution",
            "India inflation rises to {pct}%, RBI watches closely",
            "Nifty rallies {pct}% on FII inflows, banking stocks lead",
            "Sensex falls {pct}% as global cues weigh on sentiment",
            "SEBI tightens FII disclosure norms for listed companies",
            "India GDP growth at {pct}%, markets cheer strong numbers",
            "Rupee strengthens against dollar on FII equity buying",
            "Budget 2024: capex boost for infrastructure, market gains",
        ],
        "sector_banking": [
            "NPA levels fall across banking sector, credit growth strong",
            "Bank credit growth at {pct}% YoY, RBI data shows",
            "PSU banks see surge in FII inflows after results season",
        ],
        "sector_it": [
            "IT sector deal wins rise {pct}% amid global tech demand",
            "US tech slowdown hits Indian IT majors, guidance muted",
            "IT sector hiring picks up as demand for AI services grows",
        ],
        "noise": [
            # FIX: original noise templates had no market keywords and were dropped
            # by Stage 3 relevance filter — they consumed RNG cycles but produced
            # zero usable rows. Replaced with macro headlines that pass the filter
            # and contribute to sector-level signals.
            "India trade deficit narrows in Q3 on lower crude imports, rupee gains",
            "FII net buyers in equity markets for fifth straight session, nifty up",
            "Government raises capex target for infrastructure, market cheers budget",
            "RBI holds rates, signals accommodative stance for growth recovery",
            "Nifty50 hits record high on strong FII inflows and GDP optimism",
        ],
    }

    tickers = list(STOCK_UNIVERSE.keys())
    dates   = pd.bdate_range("2022-01-01", "2024-12-31")

    rows = []
    for _ in range(n):
        date     = rng.choice(dates)
        cat      = rng.choice(list(TEMPLATES.keys()))
        template = rng.choice(TEMPLATES[cat])

        # Pick a stock for company-specific templates
        ticker   = rng.choice(tickers)
        info     = STOCK_UNIVERSE[ticker]
        name     = info["name"]

        headline = template.format(
            name=name,
            pct=int(rng.integers(5, 45)),
            val=int(rng.integers(100, 9999)),
            div=round(rng.uniform(2, 50), 1),
            rate=round(rng.uniform(5.5, 7.0), 2),
        )

        rows.append({"headline": headline, "date": str(pd.Timestamp(date).date())})

    df = pd.DataFrame(rows)
    logger.info(f"Generated {len(df):,} synthetic headlines")
    return df


# ═════════════════════════════════════════════════════════════════
#  VISUALISATION
# ═════════════════════════════════════════════════════════════════

MODEL_COLORS = {
    "Ridge":            "#4A90D9",
    "LASSO":            "#7B68EE",
    "RandomForest":     "#2ECC71",
    "GradientBoosting": "#F39C12",
    "MLP":              "#E74C3C",
    "Ensemble":         "#1A1A2E",
    "EqualWeight":      "#95A5A6",
    "SentimentOnly":    "#E67E22",
}


def _pct_fmt(x, _):
    return f"{x:.0%}"


def plot_pipeline_report(
    port_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    daily_scores: pd.DataFrame,
    perf_df: pd.DataFrame,
    ic_history: dict,
    save_path: str,
):
    fig = plt.figure(figsize=(20, 24))
    fig.patch.set_facecolor("#F8F9FA")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Cumulative returns ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("white")
    ax1.set_title("Cumulative Portfolio Returns by Model", fontsize=14, fontweight="bold", pad=10)

    highlight = ["Ensemble", "EqualWeight", "SentimentOnly", "RandomForest", "GradientBoosting"]
    for model in highlight:
        sub = port_df[port_df["model"] == model].sort_values("month_end")
        if sub.empty:
            continue
        cum = (1 + sub["return_21d"]).cumprod() - 1
        lw  = 2.5 if model == "Ensemble" else 1.5
        ax1.plot(sub["month_end"], cum,
                 label=model, color=MODEL_COLORS.get(model, "gray"),
                 linewidth=lw, alpha=0.9)

    ax1.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=9, ncol=3)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(True, alpha=0.15)

    # ── Panel 2: Performance metrics table ────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis("off")
    ax2.set_title("Performance Metrics (Feng Stage 12)", fontsize=12, fontweight="bold")

    display_models = ["Ensemble", "RandomForest", "GradientBoosting",
                      "MLP", "Ridge", "LASSO", "EqualWeight", "SentimentOnly"]
    table_data = []
    col_labels = ["Ann. Return", "Sharpe", "Max DD", "Hit Rate vs EW"]
    for m in display_models:
        if m in perf_df.index:
            r = perf_df.loc[m]
            table_data.append([m, r["Ann. Return"], r["Sharpe Ratio"],
                                r["Max Drawdown"], r["Hit Rate vs EW"]])

    if table_data:
        tbl = ax2.table(
            cellText   = [r[1:] for r in table_data],
            rowLabels  = [r[0]  for r in table_data],
            colLabels  = col_labels,
            loc        = "center",
            cellLoc    = "center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.6)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#CCCCCC")
            if row == 0:
                cell.set_facecolor("#2C3E50")
                cell.set_text_props(color="white", fontweight="bold")
            elif row > 0 and col == -1:
                m_name = table_data[row - 1][0]
                cell.set_facecolor(MODEL_COLORS.get(m_name, "#EEEEEE"))
                cell.set_alpha(0.35)

    # ── Panel 3: Monthly return distribution ─────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("white")
    ax3.set_title("Monthly Return Distribution", fontsize=12, fontweight="bold")

    for model in ["Ensemble", "EqualWeight", "SentimentOnly"]:
        sub = port_df[port_df["model"] == model]["return_21d"]
        if not sub.empty:
            ax3.hist(sub, bins=20, alpha=0.5,
                     label=model, color=MODEL_COLORS.get(model, "gray"))
    ax3.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.xaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax3.legend(fontsize=8)
    ax3.set_xlabel("Monthly Return")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, alpha=0.15)

    # ── Panel 4: Daily composite score heatmap ────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("white")
    ax4.set_title("Composite Score by Ticker (last 6 months)", fontsize=12, fontweight="bold")

    if not daily_scores.empty:
        cutoff   = daily_scores["date"].max() - pd.Timedelta(days=180)
        recent   = daily_scores[daily_scores["date"] >= cutoff]
        pivot    = recent.pivot_table(
            index="ticker", columns="date",
            values="composite_score", aggfunc="mean"
        )
        # Monthly resample
        pivot.columns = pd.to_datetime(pivot.columns)
        pivot = pivot.T.resample("ME").mean().T

        if not pivot.empty:
            im = ax4.imshow(
                pivot.values, aspect="auto", cmap="RdYlGn",
                vmin=-0.3, vmax=0.3,
            )
            ax4.set_yticks(range(len(pivot.index)))
            ax4.set_yticklabels(pivot.index, fontsize=7)
            month_labels = [str(c.strftime("%b-%y")) for c in pivot.columns]
            ax4.set_xticks(range(len(month_labels)))
            ax4.set_xticklabels(month_labels, rotation=45, ha="right", fontsize=7)
            plt.colorbar(im, ax=ax4, fraction=0.04, label="Composite Score")

    # ── Panel 5: Ensemble weight allocation (last rebalance) ──────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("white")
    ax5.set_title("Ensemble Portfolio Weights (latest month)", fontsize=12, fontweight="bold")

    if not weight_df.empty:
        latest_me  = weight_df["month_end"].max()
        latest_wts = weight_df[weight_df["month_end"] == latest_me].sort_values(
            "weight", ascending=True
        )
        latest_wts = latest_wts[latest_wts["weight"] > 0.005]  # filter dust

        sector_colors = {
            "Banking": "#4A90D9", "IT": "#2ECC71", "Energy": "#F39C12",
            "Auto": "#E74C3C",    "Pharma": "#9B59B6", "Consumer": "#1ABC9C",
        }
        bar_colors = [
            sector_colors.get(
                STOCK_UNIVERSE.get(t, {}).get("sector", ""), "#95A5A6"
            )
            for t in latest_wts["ticker"]
        ]

        bars = ax5.barh(
            latest_wts["ticker"], latest_wts["weight"],
            color=bar_colors, alpha=0.85, edgecolor="white",
        )
        ax5.xaxis.set_major_formatter(FuncFormatter(_pct_fmt))
        ax5.set_xlabel("Portfolio Weight")
        ax5.grid(True, axis="x", alpha=0.2)

        # Sector legend
        from matplotlib.patches import Patch
        legend_els = [Patch(facecolor=c, label=s)
                      for s, c in sector_colors.items()]
        ax5.legend(handles=legend_els, loc="lower right", fontsize=7)

    plt.suptitle(
        "News-Driven Sustainable Portfolio Pipeline\n"
        f"Feng et al. (2024) adapted for NSE India | "
        f"Generated: {datetime.now().strftime('%Y-%m-%d')}",
        fontsize=15, fontweight="bold", y=1.01,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Report saved → {save_path}")


# ═════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════

def main(csv_path: str = None):
    logger.info("=" * 60)
    logger.info("NEWS-DRIVEN SUSTAINABLE PORTFOLIO PIPELINE")
    logger.info("Feng et al. (2024) | NSE India | 25 Stocks")
    logger.info("=" * 60)

    # ── Stage 1–4: Ingest, deduplicate, filter, entity-tag ───────
    if csv_path and os.path.exists(csv_path):
        logger.info(f"\n[STAGES 1–4] Loading real data from: {csv_path}")
        articles = run_stages_1_to_4(csv_path)
    else:
        logger.info("\n[STAGES 1–4] No CSV provided — generating synthetic data")
        raw_df   = generate_synthetic_headlines(n=20000)

        # Save synthetic CSV so stage 1 pipeline processes it properly
        tmp_path = os.path.join(OUTPUT_DIR, "_synthetic_headlines.csv")
        raw_df.to_csv(tmp_path, index=False)
        articles = run_stages_1_to_4(
            tmp_path,
            headline_col="headline",
            date_col="date",
        )

    logger.info(f"  Articles after stages 1–4: {len(articles):,}")

    # ── Stage 5–6: Sentiment + ESG/SDG scores ────────────────────
    logger.info("\n[STAGES 5–6] Sentiment scoring & daily score aggregation")
    articles, daily_scores = run_stages_5_to_6(articles)

    # Save daily scores
    daily_scores.to_csv(
        os.path.join(OUTPUT_DIR, "daily_scores.csv"), index=False
    )
    logger.info(f"  daily_scores.csv written ({len(daily_scores):,} rows)")

    # ── Synthetic / real price returns ────────────────────────────
    logger.info("\n[PRICE DATA] Fetching real returns from Yahoo Finance API...")
    returns = fetch_online_prices(daily_scores)

    # ── Feature engineering ───────────────────────────────────────
    logger.info("\n[FEATURES] Building feature matrix")
    features = build_feature_matrix(daily_scores, returns)

    if len(features) == 0:
        logger.error("Feature matrix is empty — not enough data for rolling ML.")
        return

    # ── Stages 7–8: ML portfolio ──────────────────────────────────
    logger.info("\n[STAGES 7–8] ML portfolio construction (Feng Stage 11–12)")
    port_df, weight_df, ic_history = run_ml_portfolio(
        features, returns, daily_scores
    )

    if port_df.empty:
        logger.warning("Portfolio returns are empty — pipeline needs more data months.")
        return

    # ── Performance metrics ───────────────────────────────────────
    perf_df = compute_performance(port_df)

    # ── Save outputs ──────────────────────────────────────────────
    port_df.to_csv(os.path.join(OUTPUT_DIR, "portfolio_returns.csv"), index=False)
    weight_df.to_csv(os.path.join(OUTPUT_DIR, "portfolio_weights.csv"), index=False)
    perf_df.to_csv(os.path.join(OUTPUT_DIR, "performance_metrics.csv"))

    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    print(perf_df.to_string(), flush=True)

    # ── Visualisation ─────────────────────────────────────────────
    logger.info("\n[VIZ] Generating 6-panel pipeline report...")
    plot_pipeline_report(
        port_df      = port_df,
        weight_df    = weight_df,
        daily_scores = daily_scores,
        perf_df      = perf_df,
        ic_history   = ic_history,
        save_path    = os.path.join(OUTPUT_DIR, "pipeline_report.png"),
    )

    logger.info("\n[DONE] All outputs written to ./outputs/")
    logger.info("  - daily_scores.csv")
    logger.info("  - portfolio_returns.csv")
    logger.info("  - portfolio_weights.csv")
    logger.info("  - performance_metrics.csv")
    logger.info("  - pipeline_report.png")

    return port_df, weight_df, daily_scores, perf_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="News-Driven Sustainable Portfolio Pipeline (Feng et al. 2024)"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to ET headlines CSV. If omitted, synthetic data is used.",
    )
    args = parser.parse_args()
    main(csv_path=args.csv)
