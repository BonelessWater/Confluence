"""
Pre-compute features and add them to the parquet file for faster backtesting.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engineer import FeatureEngineer
from config.settings import TWEET_PARQUET, FEATURES_PARQUET, TICKERS, DATA_DIR
import polars as pl

# Configuration
TWEET_PARQUET_INPUT = TWEET_PARQUET
OUTPUT_PARQUET = FEATURES_PARQUET

def load_price_data_for_ticker(ticker: str, tweets_df: pd.DataFrame):
    """Load real price data from parquet file for a ticker."""
    price_path = DATA_DIR / f'{ticker}.parquet'
    
    if not price_path.exists():
        raise FileNotFoundError(
            f"Price data not found for {ticker} at {price_path}\n"
            f"Please provide market data parquet files in data/ directory."
        )
    
    print(f"Loading {ticker} from {price_path}...")
    price_pl = pl.read_parquet(price_path)
    price_df = price_pl.to_pandas()
    
    # Handle timestamp column
    if 'ts_event' in price_df.columns:
        price_df['timestamp'] = pd.to_datetime(price_df['ts_event'])
    elif 'timestamp' in price_df.columns:
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
    else:
        raise ValueError(f"{ticker} parquet has no ts_event or timestamp column")
    
    # Convert UTC -> US/Eastern (naive) to match tweet created_at
    if price_df['timestamp'].dt.tz is not None:
        price_df['timestamp'] = (
            price_df['timestamp']
            .dt.tz_convert('US/Eastern')
            .dt.tz_localize(None)
        )
    
    price_df = price_df.set_index('timestamp').sort_index()
    
    # Filter to tweet date range (with buffer) to reduce memory
    min_date = tweets_df['created_at'].min() - pd.Timedelta(days=60)
    max_date = tweets_df['created_at'].max() + pd.Timedelta(days=1)
    price_df = price_df[(price_df.index >= min_date) & (price_df.index <= max_date)]
    
    print(f"  Loaded {len(price_df):,} bars ({price_df.index.min()} to {price_df.index.max()})")
    
    return price_df


def align_and_add_price_features(tweets_df: pd.DataFrame, price_df: pd.DataFrame, ticker: str, feature_engineer: FeatureEngineer):
    """Align tweets with prices and add price features using vectorized operations."""
    print(f"\nProcessing {ticker} (using vectorized merge_asof)...")

    # Calculate price features
    price_features = feature_engineer.calculate_price_features(price_df)
    price_features = feature_engineer.fill_missing_values(price_features)

    # Reset index for price features to have 'timestamp' as a column
    price_features = price_features.reset_index()

    # Prepare tweets dataframe
    tweets_sorted = tweets_df[['created_at']].copy()
    tweets_sorted['tweet_idx'] = tweets_sorted.index
    tweets_sorted = tweets_sorted.sort_values('created_at')
    
    # Sort price features
    price_features = price_features.sort_values('timestamp')

    # Merge asof to find closest price bar (first bar >= tweet_time)
    aligned_df = pd.merge_asof(
        tweets_sorted,
        price_features,
        left_on='created_at',
        right_on='timestamp',
        direction='forward'
    )
    
    # Drop rows where timestamp is NaT (tweet is later than available market data)
    aligned_df = aligned_df.dropna(subset=['timestamp'])

    aligned_df['ticker'] = ticker
    aligned_df = aligned_df.rename(columns={'timestamp': 'entry_time'})
    aligned_df = aligned_df.drop(columns=['created_at'])

    print(f"Aligned {len(aligned_df)} tweets with price features for {ticker}")
    return aligned_df


def process_single_ticker(ticker_data):
    """Helper function to process a single ticker in parallel."""
    ticker, tweets_df = ticker_data
    try:
        price_df = load_price_data_for_ticker(ticker, tweets_df)
        feature_engineer = FeatureEngineer()
        ticker_features = align_and_add_price_features(tweets_df, price_df, ticker, feature_engineer)
        return ticker_features
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return pd.DataFrame()


def main():
    """Main function to pre-compute all features."""

    print("="*80)
    print("PRE-COMPUTING FEATURES FOR TWEET DATA")
    print("="*80)

    # Load tweet data
    print("\nLoading tweet data...")
    if not TWEET_PARQUET_INPUT.exists():
        raise FileNotFoundError(
            f"\nTweet data file not found: {TWEET_PARQUET_INPUT}\n"
            "Please ensure the tweet data file exists.\n"
            "Expected location: data/trump-truth-social-archive/data/truth_archive_with_embeddings.parquet"
        )
    
    tweets_df = pd.read_parquet(TWEET_PARQUET_INPUT)
    tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'])
    # Strip timezone if present (convert to naive Eastern)
    if tweets_df['created_at'].dt.tz is not None:
        tweets_df['created_at'] = (
            tweets_df['created_at']
            .dt.tz_convert('US/Eastern')
            .dt.tz_localize(None)
        )

    print(f"Loaded {len(tweets_df)} tweets")
    print(f"Date range: {tweets_df['created_at'].min()} to {tweets_df['created_at'].max()}")

    # Initialize feature engineer
    feature_engineer = FeatureEngineer()

    # Calculate tweet features (independent of ticker)
    print("\nCalculating tweet features...")
    tweet_features = feature_engineer.calculate_tweet_features(tweets_df)

    # Add ticker-specific price features
    # Filter to tickers that have data files
    available_tickers = [t for t in TICKERS if (DATA_DIR / f'{t}.parquet').exists()]
    skipped_tickers = [t for t in TICKERS if t not in available_tickers]
    if skipped_tickers:
        print(f"\nSkipping tickers without data files: {skipped_tickers}")
    print(f"Processing tickers: {available_tickers}")

    # Parallel processing of tickers
    all_ticker_data = []
    
    import concurrent.futures
    import multiprocessing
    max_workers = min(len(available_tickers), multiprocessing.cpu_count() or 4)
    print(f"\nProcessing {len(available_tickers)} tickers in parallel using {max_workers} cores...")

    ticker_args = [(t, tweets_df) for t in available_tickers]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_ticker, args): args[0] for args in ticker_args}
        for future in concurrent.futures.as_completed(futures):
            ticker_features = future.result()
            if not ticker_features.empty:
                all_ticker_data.append(ticker_features)

    # Combine all ticker data
    print("\nCombining all ticker features...")
    if not all_ticker_data:
        raise ValueError("No ticker features could be calculated")
    combined_features = pd.concat(all_ticker_data, ignore_index=True)

    # Merge with original tweet data
    print("Merging with original tweet data...")

    # Prepare tweet features with prefixed names
    tweet_features_prefixed = tweet_features.copy()
    tweet_features_prefixed.columns = [f'tweet_{col}' for col in tweet_features_prefixed.columns]
    
    # Prepare enriched tweets dataframe (original metadata + computed tweet features)
    enriched_tweets = tweets_df[['id', 'created_at', 'content', 'url']].copy()
    enriched_tweets = enriched_tweets.rename(columns={
        'id': 'tweet_id', 
        'created_at': 'tweet_time', 
        'content': 'tweet_content', 
        'url': 'tweet_url'
    })
    enriched_tweets = pd.concat([enriched_tweets, tweet_features_prefixed], axis=1)
    enriched_tweets['tweet_idx'] = enriched_tweets.index

    # Vectorized merge
    final_df = combined_features.merge(enriched_tweets, on='tweet_idx', how='inner')

    print(f"\nFinal dataset shape: {final_df.shape}")
    print(f"Total features: {len(final_df.columns)}")

    # Save to parquet
    print(f"\nSaving to {OUTPUT_PARQUET}...")
    # Ensure directory exists
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(OUTPUT_PARQUET, index=False)

    file_size_mb = os.path.getsize(OUTPUT_PARQUET) / (1024 * 1024)
    print(f"Saved! File size: {file_size_mb:.2f} MB")

    print("\n" + "="*80)
    print("FEATURE PRE-COMPUTATION COMPLETE")
    print("="*80)
    print(f"\nFeatures saved to: {OUTPUT_PARQUET}")
    print(f"Total rows: {len(final_df):,}")
    print(f"Total columns: {len(final_df.columns):,}")
    print("\nColumn categories:")

    # Categorize columns
    price_cols = [c for c in final_df.columns if c.startswith(('ema_', 'vol_', 'rsi_', 'macd_', 'bb_', 'atr_', 'momentum_', 'dist_'))]
    tweet_cols = [c for c in final_df.columns if c.startswith('tweet_')]
    embedding_cols = [c for c in final_df.columns if c.startswith('embedding_')]
    meta_cols = [c for c in final_df.columns if c in ['ticker', 'tweet_id', 'tweet_time', 'entry_time', 'tweet_idx']]

    print(f"  Price features: {len(price_cols)}")
    print(f"  Tweet features: {len(tweet_cols)}")
    print(f"  Embedding features: {len(embedding_cols)}")
    print(f"  Metadata columns: {len(meta_cols)}")

if __name__ == "__main__":
    main()
