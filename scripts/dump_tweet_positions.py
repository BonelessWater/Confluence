"""
Dump tweet and position data for analysis.

Creates two output files:
1. tweet_position_dump.csv - For each tweet, positions and sizes (with dates)
2. top_movers_dump.csv - Same data sorted by top movers (absolute return)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import argparse

from config.settings import FEATURES_PARQUET, DATA_DIR, OUTPUT_DIR


def load_tweets(features_path: Path) -> pd.DataFrame:
    """Load tweet data with entry_time and ticker for matching."""
    if not features_path.exists():
        alt_path = DATA_DIR / "trump-truth-social-archive" / "data" / "truth_archive_with_embeddings.parquet"
        if alt_path.exists():
            features_path = alt_path
        else:
            raise FileNotFoundError(
                f"Tweet data not found at {FEATURES_PARQUET} or {alt_path}\n"
                "Run: python scripts/precompute_features.py"
            )

    df = pd.read_parquet(features_path)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    if df['entry_time'].dt.tz is not None:
        df['entry_time'] = df['entry_time'].dt.tz_convert('US/Eastern').dt.tz_localize(None)

    # Normalize to seconds for matching (trades may have slight precision differences)
    df['entry_time_norm'] = df['entry_time'].dt.floor('s')

    cols = ['tweet_id', 'tweet_content', 'entry_time', 'ticker', 'entry_time_norm']
    available = [c for c in cols if c in df.columns]
    return df[available].drop_duplicates(subset=['entry_time_norm', 'ticker'])


def load_trades(trades_path: Path) -> pd.DataFrame:
    """Load trades from a method's output."""
    df = pd.read_csv(trades_path)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['entry_time_norm'] = df['entry_time'].dt.floor('s')
    return df


def create_dumps(
    method: str = 'sentiment',
    output_dir: Path = None,
    features_path: Path = None
) -> None:
    """
    Create tweet_position_dump.csv and top_movers_dump.csv.

    Args:
        method: Backtest method name (sentiment, correlation, embedding, etc.)
        output_dir: Where to save dumps (default: output/tweet_ticker_methods/{method}/)
        features_path: Path to features parquet (default: from config)
    """
    output_dir = output_dir or OUTPUT_DIR / 'tweet_ticker_methods' / method
    features_path = features_path or FEATURES_PARQUET

    trades_path = output_dir / 'trades.csv'
    if not trades_path.exists():
        raise FileNotFoundError(
            f"Trades file not found: {trades_path}\n"
            f"Run backtest first: python scripts/run_all_tweet_ticker_methods.py --method {method}"
        )

    print("=" * 60)
    print("TWEET & POSITION DUMP")
    print("=" * 60)
    print(f"Method: {method}")
    print(f"Trades: {trades_path}")
    print()

    # Load data
    print("Loading tweets...")
    tweets_df = load_tweets(features_path)
    print(f"  Loaded {len(tweets_df)} tweet-ticker pairs")

    print("Loading trades...")
    trades_df = load_trades(trades_path)
    print(f"  Loaded {len(trades_df)} trades")

    # Merge trades with tweets on entry_time + ticker
    tweet_cols = ['tweet_id', 'tweet_content', 'entry_time_norm', 'ticker']
    merged = pd.merge(
        trades_df,
        tweets_df[tweet_cols],
        on=['entry_time_norm', 'ticker'],
        how='left'
    )

    # Drop merge helper column
    merged = merged.drop(columns=['entry_time_norm'], errors='ignore')

    # Add date columns
    merged['tweet_date'] = pd.to_datetime(merged['entry_time']).dt.date
    merged['position_entry_date'] = pd.to_datetime(merged['entry_time']).dt.date
    merged['position_exit_date'] = pd.to_datetime(merged['exit_time']).dt.date

    # Position size as % of capital
    merged['position_size_pct'] = (merged['weight'] * 100).round(4)

    # Reorder columns for clarity
    col_order = [
        'tweet_id', 'tweet_date', 'tweet_content',
        'entry_time', 'exit_time',
        'position_entry_date', 'position_exit_date',
        'ticker', 'weight', 'position_size_pct',
        'influence_score', 'gross_return', 'net_return',
        'transaction_cost', 'pnl', 'capital'
    ]
    cols = [c for c in col_order if c in merged.columns]
    merged = merged[cols]

    # 1. Tweet-position dump (by tweet / chronological)
    dump1 = merged.sort_values(['entry_time', 'ticker']).reset_index(drop=True)
    dump1_path = output_dir / 'tweet_position_dump.csv'
    dump1.to_csv(dump1_path, index=False)
    print(f"\nSaved: {dump1_path}")
    print(f"  Rows: {len(dump1)}")

    # 2. Top movers dump (by absolute net_return)
    merged['abs_net_return'] = merged['net_return'].abs()
    dump2 = merged.sort_values('abs_net_return', ascending=False).reset_index(drop=True)
    dump2 = dump2.drop(columns=['abs_net_return'])
    dump2_path = output_dir / 'top_movers_dump.csv'
    dump2.to_csv(dump2_path, index=False)
    print(f"\nSaved: {dump2_path}")
    print(f"  Rows: {len(dump2)} (sorted by |net_return|)")

    # Summary
    matched = merged['tweet_id'].notna().sum()
    unmatched = merged['tweet_id'].isna().sum()
    print(f"\nTweet matching: {matched} matched, {unmatched} unmatched")
    if unmatched > 0:
        print("  (Unmatched trades may be from tweets filtered before backtest)")


def main():
    parser = argparse.ArgumentParser(description='Dump tweet and position data')
    parser.add_argument('--method', type=str, default='sentiment',
                        choices=['sentiment', 'correlation', 'embedding', 'bag_of_words',
                                 'ensemble', 'multi_timeframe', 'llm'],
                        help='Backtest method to use')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Override output directory')
    args = parser.parse_args()

    create_dumps(method=args.method, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
