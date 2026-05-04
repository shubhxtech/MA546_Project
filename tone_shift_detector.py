import json
import numpy as np
from pathlib import Path
from datetime import datetime

def compute_tone_shift(ticker: str, current_qtr_score: float, current_date: datetime) -> dict:
    sig_path = Path(__file__).parent / "data" / "transcript_signals.json"
    
    # Load historical scores to compute trailing 4-quarter average
    hist_scores = []
    if sig_path.exists():
        try:
            with open(sig_path, "r") as f:
                all_signals = json.load(f)
                for s in all_signals:
                    if s["ticker"] == ticker:
                        hist_scores.append(s["mgmt_score"])
        except Exception:
            pass
            
    # Need at least 2 quarters of history to establish a baseline properly, 
    # but for simulation we will allow 1.
    if not hist_scores:
        avg_score = current_qtr_score
        std_dev = 0.1 # default
    else:
        # Take last 4
        recent = hist_scores[-4:]
        avg_score = np.mean(recent)
        std_dev = np.std(recent)
        if std_dev == 0: std_dev = 0.1
        
    delta = current_qtr_score - avg_score
    z_score = delta / std_dev
    
    tone_shift_flag = "NEUTRAL"
    if z_score > 0.25: # As requested: delta > 0.25 standard deviations
        tone_shift_flag = "POSITIVE_SHIFT"
    elif z_score < -0.25:
        tone_shift_flag = "NEGATIVE_SHIFT"
        
    return {
        "ticker": ticker,
        "date": current_date.strftime("%Y-%m-%d"),
        "tone_shift_flag": tone_shift_flag,
        "tone_delta": round(float(delta), 3),
        "z_score": round(float(z_score), 3)
    }

def process_transcripts_through_nlp(pipeline, current_date: datetime):
    # This simulates passing the scraped transcripts through the DeBERTa+FinBERT pipeline
    out_dir = Path(__file__).parent / "data" / "transcripts"
    sig_path = Path(__file__).parent / "data" / "transcript_signals.json"
    
    if not out_dir.exists():
        return []
        
    # Read existing signals
    all_signals = []
    if sig_path.exists():
        try:
            with open(sig_path, "r") as f:
                all_signals = json.load(f)
        except Exception:
            pass
            
    new_signals = []
    
    # Check for transcripts dumped today
    for path in out_dir.glob("*.json"):
        with open(path, "r") as f:
            t = json.load(f)
            
        if t["date"] != current_date.strftime("%Y-%m-%d"):
            continue
            
        # Process segments through NLP
        segment_scores = {}
        for seg_name, text in t["segments"].items():
            # This calls the method we will add to news_trading_pipeline.py
            score, conf = pipeline.process_transcript_segment(text, seg_name)
            segment_scores[seg_name] = round(score, 3)
            
        # Tone shift detection on Management Opening
        mgmt_score = segment_scores.get("Management Opening Statement", 0.0)
        shift_data = compute_tone_shift(t["ticker"], mgmt_score, current_date)
        
        sig = {
            "ticker": t["ticker"],
            "date": t["date"],
            "quarter": t["quarter"],
            "segment_scores": segment_scores,
            "mgmt_score": mgmt_score,
            "tone_shift_flag": shift_data["tone_shift_flag"],
            "tone_delta": shift_data["tone_delta"],
            "confidence": 0.85 # Mock confidence
        }
        
        new_signals.append(sig)
        
    if new_signals:
        all_signals.extend(new_signals)
        with open(sig_path, "w") as f:
            json.dump(all_signals, f, indent=2)
            
    return new_signals
