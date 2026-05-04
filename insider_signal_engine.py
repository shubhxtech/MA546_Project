import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def generate_insider_signals(current_date: datetime):
    raw_path = Path(__file__).parent / "data" / "insider_raw.csv"
    sig_path = Path(__file__).parent / "data" / "insider_signals.json"
    
    if not raw_path.exists():
        return {}
        
    df = pd.read_csv(raw_path)
    if df.empty:
        return {}
        
    # Filter to last 30 days of data for processing
    df['date_obj'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    # Normalize current_date to tz-naive to avoid datetime64 vs Timestamp mismatch
    if hasattr(current_date, 'tzinfo') and current_date.tzinfo is not None:
        cutoff_date = pd.Timestamp(current_date).tz_localize(None) - pd.Timedelta(days=30)
    else:
        cutoff_date = pd.Timestamp(current_date) - pd.Timedelta(days=30)
    df = df[df['date_obj'] >= cutoff_date]
    
    # Ignore transactions below 10 Lakhs
    df = df[df['value_lakhs'] >= 10.0]
    
    signals = []
    
    # Process each ticker
    for ticker, group in df.groupby("ticker"):
        for _, row in group.iterrows():
            base_score = 0.0
            rel = row['relation']
            
            if rel == "Promoter": base_score = 0.8
            elif rel == "Director": base_score = 0.5
            elif rel == "KMP": base_score = 0.3
            elif rel in ["Institutional", "DII"]: base_score = 0.4
            elif rel == "FII": base_score = 0.3
            
            if row['direction'] == "Sell":
                base_score *= -1.0
                
            conviction_level = "NORMAL"
            multiplier = 1.0
            
            # Check for cluster buying/selling in a 5-day window around this date
            row_date = pd.Timestamp(row['date_obj'])
            w_start = row_date - pd.Timedelta(days=5)
            w_end = row_date + pd.Timedelta(days=5)
            cluster_mask = (group['date_obj'] >= w_start) & (group['date_obj'] <= w_end) & (group['direction'] == row['direction'])
            cluster_count = cluster_mask.sum()
            
            if cluster_count == 2: multiplier *= 1.3
            elif cluster_count >= 3: multiplier *= 1.6
            
            # High conviction filter (> 5 Crores = 500 Lakhs)
            if row['value_lakhs'] >= 500:
                conviction_level = "HIGH_CONVICTION"
                multiplier *= 1.5
                
            final_score = np.clip(base_score * multiplier, -1.0, 1.0)
            
            signals.append({
                "ticker": ticker,
                "date": row['date'],
                "insider_type": rel,
                "direction": row['direction'],
                "value_lakhs": row['value_lakhs'],
                "score": round(final_score, 3),
                "conviction_level": conviction_level
            })
            
    # Save to JSON
    with open(sig_path, "w") as f:
        json.dump(signals, f, indent=2)
        
    return signals
