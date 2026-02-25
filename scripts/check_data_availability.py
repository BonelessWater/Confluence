"""
Check data availability for tweet-ticker analysis.

This script checks if required data files exist and provides guidance.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import FEATURES_PARQUET, TWEET_PARQUET, DATA_DIR, TICKERS

def check_data():
    """Check if required data files exist."""
    print("="*80)
    print("DATA AVAILABILITY CHECK")
    print("="*80)
    
    all_good = True
    
    # Check tweet data
    print("\n1. Tweet Data:")
    print(f"   Features file: {FEATURES_PARQUET}")
    if FEATURES_PARQUET.exists():
        print(f"   ✓ Found ({FEATURES_PARQUET.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print(f"   ✗ Not found")
        all_good = False
        
        # Check alternative
        alt_path = TWEET_PARQUET
        print(f"   Checking alternative: {alt_path}")
        if alt_path.exists():
            print(f"   ✓ Alternative found ({alt_path.stat().st_size / 1024 / 1024:.2f} MB)")
            print(f"   Note: You may need to run precompute_features.py to generate features")
        else:
            print(f"   ✗ Alternative also not found")
    
    # Check market data
    print("\n2. Market Data:")
    found_tickers = []
    missing_tickers = []
    
    for ticker in TICKERS:
        price_path = DATA_DIR / f'{ticker}.parquet'
        if price_path.exists():
            size_mb = price_path.stat().st_size / 1024 / 1024
            print(f"   ✓ {ticker}: Found ({size_mb:.2f} MB)")
            found_tickers.append(ticker)
        else:
            print(f"   ✗ {ticker}: Not found")
            missing_tickers.append(ticker)
    
    if missing_tickers:
        all_good = False
        print(f"\n   Missing data for {len(missing_tickers)} tickers: {missing_tickers}")
    
    # Summary
    print("\n" + "="*80)
    if all_good:
        print("✓ All required data files are available!")
        print("  You can run: python scripts/run_all_tweet_ticker_methods.py")
    else:
        print("✗ Some data files are missing.")
        print("\nTo fix:")
        print("1. Ensure tweet data exists:")
        print(f"   - {FEATURES_PARQUET}")
        print(f"   - Or: {TWEET_PARQUET} (then run precompute_features.py)")
        print("\n2. Ensure market data exists:")
        print(f"   - Place parquet files in: {DATA_DIR}/")
        print(f"   - Expected files: {[f'{t}.parquet' for t in TICKERS]}")
        print("\n3. Market data format:")
        print("   - Should have 'ts_event' or 'timestamp' column")
        print("   - Should have 'close' column (and optionally OHLC)")
        print("   - Should be in parquet format")
    
    print("="*80)
    return all_good

if __name__ == '__main__':
    check_data()
