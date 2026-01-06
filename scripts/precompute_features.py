"""
Pre-compute features and add them to the parquet file for faster backtesting.
"""

import pandas as pd
import numpy as np
import sys
import os
from feature_engineering import FeatureEngineer

# Configuration
TWEET_PARQUET = r"C:\Users\domdd\Documents\GitHub\confluence\data\trump-truth-social-archive\data\truth_archive_with_embeddings.parquet"
OUTPUT_PARQUET = r"C:\Users\domdd\Documents\GitHub\confluence\data\trump-truth-social-archive\data\truth_archive_with_features.parquet"

def load_price_data_for_ticker(ticker: str, tweets_df: pd.DataFrame):
    """Load and create synthetic 5-minute price data for a ticker."""
    import yfinance as yf

    start_date = tweets_df['created_at'].min()
    end_date = tweets_df['created_at'].max()

    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")

    stock = yf.Ticker(ticker)
    daily_df = stock.history(start=start_date, end=end_date, interval='1d')

    if daily_df.empty:
        print(f"Warning: No data for {ticker}, using recent data")
        daily_df = stock.history(period='2y', interval='1d')

    print(f"Creating synthetic 5-minute bars...")
    intraday_bars = []

    for date, row in daily_df.iterrows():
        market_open = pd.Timestamp(date.date()) + pd.Timedelta(hours=9, minutes=30)
        num_bars = 78

        open_price = row['Open']
        close_price = row['Close']
        high_price = row['High']
        low_price = row['Low']
        volume = row['Volume']

        daily_return = (close_price - open_price) / open_price

        np.random.seed(int(date.timestamp()))
        returns = np.random.normal(daily_return / num_bars, 0.0001, num_bars)
        returns = returns * (daily_return / returns.sum()) if returns.sum() != 0 else returns

        prices = [open_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        prices = np.array(prices)
        prices = np.clip(prices, low_price, high_price)

        for i in range(num_bars):
            timestamp = market_open + pd.Timedelta(minutes=5*i)

            bar = {
                'timestamp': timestamp,
                'open': prices[i],
                'high': max(prices[i], prices[i+1]),
                'low': min(prices[i], prices[i+1]),
                'close': prices[i+1],
                'volume': volume / num_bars
            }
            intraday_bars.append(bar)

    intraday_df = pd.DataFrame(intraday_bars)
    intraday_df.set_index('timestamp', inplace=True)

    return intraday_df


def align_and_add_price_features(tweets_df: pd.DataFrame, price_df: pd.DataFrame, ticker: str, feature_engineer: FeatureEngineer):
    """Align tweets with prices and add price features."""

    print(f"\n{'='*80}")
    print(f"Processing {ticker}")
    print(f"{'='*80}")

    # Calculate price features
    price_features = feature_engineer.calculate_price_features(price_df)

    # Fill missing values
    price_features = feature_engineer.fill_missing_values(price_features)

    aligned_features = []

    for idx, tweet in tweets_df.iterrows():
        tweet_time = tweet['created_at']

        # Find closest price bar
        future_bars = price_df[price_df.index >= tweet_time]

        if len(future_bars) == 0:
            continue

        entry_bar_time = future_bars.index[0]

        # Get price features at this time
        if entry_bar_time in price_features.index:
            features_dict = price_features.loc[entry_bar_time].to_dict()
            features_dict['tweet_idx'] = idx
            features_dict['ticker'] = ticker
            features_dict['entry_time'] = entry_bar_time
            aligned_features.append(features_dict)

    aligned_df = pd.DataFrame(aligned_features)
    print(f"Aligned {len(aligned_df)} tweets with price features")

    return aligned_df


def main():
    """Main function to pre-compute all features."""

    print("="*80)
    print("PRE-COMPUTING FEATURES FOR TWEET DATA")
    print("="*80)

    # Load tweet data
    print("\nLoading tweet data...")
    tweets_df = pd.read_parquet(TWEET_PARQUET)
    tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'])
    tweets_df['created_at'] = tweets_df['created_at'].dt.tz_localize(None)

    print(f"Loaded {len(tweets_df)} tweets")
    print(f"Date range: {tweets_df['created_at'].min()} to {tweets_df['created_at'].max()}")

    # Initialize feature engineer
    feature_engineer = FeatureEngineer()

    # Calculate tweet features (independent of ticker)
    print("\nCalculating tweet features...")
    tweet_features = feature_engineer.calculate_tweet_features(tweets_df)

    # Add ticker-specific price features
    tickers = ['SPY', 'QQQ', 'DIA', 'IWM', 'TLT', 'GLD']

    all_ticker_data = []

    for ticker in tickers:
        # Load price data
        price_df = load_price_data_for_ticker(ticker, tweets_df)

        # Align and calculate features
        ticker_features = align_and_add_price_features(tweets_df, price_df, ticker, feature_engineer)

        all_ticker_data.append(ticker_features)

    # Combine all ticker data
    print("\nCombining all ticker features...")
    combined_features = pd.concat(all_ticker_data, ignore_index=True)

    # Merge with original tweet data
    print("Merging with original tweet data...")

    # For each ticker, merge tweet features
    final_data = []

    for ticker in tickers:
        ticker_data = combined_features[combined_features['ticker'] == ticker].copy()

        # Merge tweet features
        for idx, row in ticker_data.iterrows():
            tweet_idx = int(row['tweet_idx'])

            # Get original tweet data
            original_tweet = tweets_df.iloc[tweet_idx]

            # Combine
            combined_row = row.to_dict()

            # Add tweet-specific features
            tweet_feat_row = tweet_features.iloc[tweet_idx]
            for col in tweet_feat_row.index:
                combined_row[f'tweet_{col}'] = tweet_feat_row[col]

            # Add original tweet metadata (non-redundant)
            combined_row['tweet_id'] = original_tweet['id']
            combined_row['tweet_time'] = original_tweet['created_at']
            combined_row['tweet_content'] = original_tweet['content']
            combined_row['tweet_url'] = original_tweet['url']

            final_data.append(combined_row)

    final_df = pd.DataFrame(final_data)

    print(f"\nFinal dataset shape: {final_df.shape}")
    print(f"Total features: {len(final_df.columns)}")

    # Save to parquet
    print(f"\nSaving to {OUTPUT_PARQUET}...")
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
