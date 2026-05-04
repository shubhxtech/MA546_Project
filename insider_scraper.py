import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

# In a true production environment, this would use requests/BeautifulSoup or Selenium
# to scrape https://www.bseindia.com/corporates/Insider_Trading_new.aspx
# For the purpose of this simulation, we generate synthetic insider disclosures 
# for the active universe of tickers.

def run_insider_scraper(universe_tickers: list, current_date: datetime):
    print(f"[Insider Scraper] Fetching insider/bulk deals for {current_date.strftime('%Y-%m-%d')}...")
    
    records = []
    # Generate 1-3 random insider deals per day
    num_deals = np.random.randint(1, 4)
    
    for _ in range(num_deals):
        ticker = np.random.choice(universe_tickers).replace(".NS", "")
        # Weights for relation
        relation = np.random.choice(
            ["Promoter", "Director", "KMP", "Institutional", "FII", "DII"], 
            p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.1]
        )
        direction = np.random.choice(["Buy", "Sell"], p=[0.3, 0.7]) # insiders sell more often than buy
        
        # Value in INR Lakhs (10 Lakhs to 100 Crores)
        value_lakhs = np.random.lognormal(mean=4, sigma=2) * 10 
        
        records.append({
            "ticker": ticker,
            "date": current_date.strftime("%Y-%m-%d"),
            "acquirer": f"Entity_{np.random.randint(100,999)}",
            "relation": relation,
            "direction": direction,
            "value_lakhs": round(value_lakhs, 2)
        })
        
    df = pd.DataFrame(records)
    out_path = Path(__file__).parent / "data" / "insider_raw.csv"
    out_path.parent.mkdir(exist_ok=True)
    
    if out_path.exists():
        df.to_csv(out_path, mode='a', header=False, index=False)
    else:
        df.to_csv(out_path, index=False)
        
    return df
