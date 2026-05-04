import json
from pathlib import Path
from datetime import datetime
import numpy as np

# In production, this uses pdfplumber to parse BSE PDFs from:
# https://www.bseindia.com/corporates/ann.html

def scrape_and_segment_transcripts(universe_tickers: list, current_date: datetime):
    print(f"[Earnings Scraper] Checking for new transcripts on {current_date.strftime('%Y-%m-%d')}...")
    
    # Simulate a transcript drop with 5% probability per day for any given ticker
    # (Earnings season burst)
    new_transcripts = []
    
    for ticker in universe_tickers:
        ticker = ticker.replace(".NS", "")
        if np.random.rand() < 0.05:
            # Generate a synthetic segmented transcript
            year = current_date.year
            quarter = (current_date.month - 1) // 3 + 1
            
            # Simulated segments with embedded sentiments
            is_good_qtr = np.random.rand() > 0.4
            
            if is_good_qtr:
                mgmt = "Ladies and gentlemen, we are pleased to report a strong quarter with record revenues and expanded margins."
                fin = f"Revenue grew by {np.random.randint(15, 30)}% year-over-year. EBITDA margins improved by 200 basis points."
                guidance = "Looking ahead, we expect this momentum to continue. We are raising our full year guidance."
                qa = "Analyst: Great quarter. Are you seeing any demand softness? Mgmt: No, demand remains robust across all verticals."
            else:
                mgmt = "Ladies and gentlemen, it was a challenging quarter impacted by macroeconomic headwinds and supply chain issues."
                fin = f"Revenue declined by {np.random.randint(2, 10)}% year-over-year. Margins contracted due to input cost inflation."
                guidance = "Given the near-term uncertainty, we are cautious and lowering our outlook for the second half."
                qa = "Analyst: When do you see recovery? Mgmt: We expect pressure to continue for at least two more quarters."
            
            new_transcripts.append({
                "ticker": ticker,
                "date": current_date.strftime("%Y-%m-%d"),
                "quarter": f"{year}-Q{quarter}",
                "segments": {
                    "Management Opening Statement": mgmt,
                    "Financial Highlights": fin,
                    "Outlook & Guidance": guidance,
                    "Q&A Session": qa
                }
            })
            
    # Save to disk
    out_dir = Path(__file__).parent / "data" / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for t in new_transcripts:
        path = out_dir / f"{t['ticker']}_{t['quarter']}.json"
        with open(path, "w") as f:
            json.dump(t, f, indent=2)
            
    return new_transcripts
