"""
Data Preparation Guide and Helper

This script helps you understand what data is needed and provides options
to either locate existing data or create sample data for testing.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import FEATURES_PARQUET, TWEET_PARQUET, DATA_DIR, TICKERS

def check_existing_data():
    """Check for any existing data files."""
    print("="*80)
    print("SEARCHING FOR EXISTING DATA FILES")
    print("="*80)
    
    found_files = []
    
    # Check for tweet data in various locations
    print("\n1. Searching for tweet data files...")
    search_paths = [
        FEATURES_PARQUET,
        TWEET_PARQUET,
        DATA_DIR / "trump-truth-social-archive" / "data" / "*.parquet",
        Path("tweet_analysis") / "*.txt",
        Path("tweet_analysis") / "*.csv",
        Path("data") / "*.parquet",
        Path("data") / "*.csv",
    ]
    
    for path in search_paths:
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / 1024 / 1024
                print(f"   ✓ Found: {path} ({size_mb:.2f} MB)")
                found_files.append(path)
            elif '*' in str(path):
                # Try glob pattern
                import glob
                matches = glob.glob(str(path))
                for match in matches:
                    match_path = Path(match)
                    if match_path.exists():
                        size_mb = match_path.stat().st_size / 1024 / 1024
                        print(f"   ✓ Found: {match_path} ({size_mb:.2f} MB)")
                        found_files.append(match_path)
    
    # Check for market data
    print("\n2. Searching for market data files...")
    for ticker in TICKERS:
        price_path = DATA_DIR / f'{ticker}.parquet'
        if price_path.exists():
            size_mb = price_path.stat().st_size / 1024 / 1024
            print(f"   ✓ {ticker}: {price_path} ({size_mb:.2f} MB)")
            found_files.append(price_path)
    
    return found_files


def show_data_requirements():
    """Show what data format is required."""
    print("\n" + "="*80)
    print("DATA REQUIREMENTS")
    print("="*80)
    
    print("\n1. Tweet Data File:")
    print(f"   Required: {TWEET_PARQUET}")
    print("   Format: Parquet file")
    print("   Required columns:")
    print("     - id: Unique tweet ID")
    print("     - content: Tweet text content")
    print("     - created_at: Timestamp")
    print("     - embedding: Array/list of embedding values (optional but recommended)")
    print("     - replies_count, reblogs_count, favourites_count: Engagement metrics")
    
    print("\n2. Market Data Files:")
    print(f"   Location: {DATA_DIR}/")
    print("   Format: Parquet files, one per ticker")
    print("   Required files:")
    for ticker in TICKERS:
        print(f"     - {ticker}.parquet")
    print("\n   Required columns:")
    print("     - ts_event or timestamp: Datetime index")
    print("     - close: Closing price (required)")
    print("     - open, high, low: OHLC prices (optional)")
    print("     - volume: Trading volume (optional)")
    
    print("\n3. Output File (will be generated):")
    print(f"   {FEATURES_PARQUET}")
    print("   This file is created by running: python scripts/precompute_features.py")


def show_alternatives():
    """Show alternative approaches if data is missing."""
    print("\n" + "="*80)
    print("ALTERNATIVE APPROACHES")
    print("="*80)
    
    print("\nIf you don't have the required data files, you have these options:")
    
    print("\n1. Use Existing Text File:")
    print("   Found: tweet_analysis/trump_tweets_20251026.txt")
    print("   You could convert this to the required format.")
    print("   However, you'll still need:")
    print("     - Embeddings for each tweet (can generate using OpenAI API or sentence-transformers)")
    print("     - Market data files")
    
    print("\n2. Download Market Data:")
    print("   You can download market data using:")
    print("     - yfinance: pip install yfinance")
    print("     - databento: For high-frequency data")
    print("     - Your broker's API")
    
    print("\n3. Generate Sample Data (for testing only):")
    print("   The precompute_features.py script can generate synthetic market data")
    print("   if you have tweet data but not market data.")
    print("   However, you still need the tweet data file with embeddings.")
    
    print("\n4. Check Other Locations:")
    print("   - Check if data is in a different directory")
    print("   - Check if files have different names")
    print("   - Check if data is in a cloud storage location")


def main():
    """Main function."""
    print("="*80)
    print("DATA PREPARATION GUIDE")
    print("="*80)
    
    found_files = check_existing_data()
    show_data_requirements()
    
    if not found_files:
        print("\n" + "="*80)
        print("WARNING: NO DATA FILES FOUND")
        print("="*80)
        show_alternatives()
    else:
        print("\n" + "="*80)
        print(f"✓ Found {len(found_files)} data file(s)")
        print("="*80)
        print("\nNext steps:")
        if TWEET_PARQUET.exists() or FEATURES_PARQUET.exists():
            print("1. Tweet data found - you can proceed!")
        else:
            print("1. Tweet data still needed - see alternatives above")
        
        market_data_count = sum(1 for t in TICKERS if (DATA_DIR / f'{t}.parquet').exists())
        if market_data_count == len(TICKERS):
            print("2. All market data found - ready to go!")
        else:
            print(f"2. Market data: {market_data_count}/{len(TICKERS)} tickers found")
            print("   Missing:", [t for t in TICKERS if not (DATA_DIR / f'{t}.parquet').exists()])


if __name__ == '__main__':
    main()
