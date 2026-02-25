"""
Main script to run all tweet-ticker relationship discovery methods and backtest them.

This script:
1. Loads tweet and market data
2. Trains all 5 methods (correlation, bag-of-words, embedding, LLM, ensemble)
3. Backtests each method as a trading strategy
4. Generates quantstats equity curves and performance reports
5. Organizes results by method in the file system
"""

import sys
import os
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import polars as pl
import warnings
from typing import Dict, List, Optional
warnings.filterwarnings('ignore')

from config.settings import (
    FEATURES_PARQUET, TICKERS, DATA_DIR, OUTPUT_DIR,
    INITIAL_CAPITAL, HOLDING_PERIOD_MINUTES
)

from src.tweet_ticker_analysis.correlation_discovery import CorrelationDiscovery
from src.tweet_ticker_analysis.bag_of_words_scorer import BagOfWordsScorer
from src.tweet_ticker_analysis.embedding_scorer import EmbeddingScorer
from src.tweet_ticker_analysis.llm_scorer import LLMScorer
from src.tweet_ticker_analysis.ensemble_scorer import EnsembleScorer
from src.tweet_ticker_analysis.backtest_tweet_strategies import TweetStrategyBacktester
from src.tweet_ticker_analysis.tweet_cleaner import TweetCleaner
from src.tweet_ticker_analysis.correlation_validator import CorrelationValidator
from src.tweet_ticker_analysis.sentiment_analyzer import FinancialSentimentAnalyzer, SentimentScorer
from src.tweet_ticker_analysis.market_regime import MarketRegimeDetector, MarketRegime
from src.tweet_ticker_analysis.position_sizing import PositionSizer, AdaptivePositionSizer
from src.tweet_ticker_analysis.multi_timeframe_scorer import MultiTimeframeScorer
from src.tweet_ticker_analysis.cross_asset_analyzer import CrossAssetAnalyzer

# Try to import quantstats
try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False
    print("Warning: quantstats not installed. Install with: pip install quantstats")
    print("  Equity curves will still be generated but quantstats reports will be skipped.")


def load_data():
    """Load tweet and market data."""
    print("="*80)
    print("LOADING DATA")
    print("="*80)

    # Check if features file exists, try alternatives
    features_path = FEATURES_PARQUET
    if not features_path.exists():
        print(f"\nWarning: {features_path} not found")
        # Try alternative path
        alt_path = DATA_DIR / "trump-truth-social-archive" / "data" / "truth_archive_with_embeddings.parquet"
        if alt_path.exists():
            print(f"  Trying alternative: {alt_path}")
            features_path = alt_path
        else:
            raise FileNotFoundError(
                f"\nData file not found: {FEATURES_PARQUET}\n"
                f"Alternative also not found: {alt_path}\n\n"
                "Please ensure one of the following exists:\n"
                f"  1. {FEATURES_PARQUET}\n"
                f"  2. {alt_path}\n\n"
                "You may need to run:\n"
                "  python scripts/precompute_features.py\n"
                "to generate the features file."
            )

    # Load tweets
    print(f"\nLoading tweets from {features_path}...")
    tweets_df = pd.read_parquet(features_path)
    
    # Ensure required columns exist
    required_cols = ['tweet_id', 'tweet_content', 'entry_time', 'ticker']
    missing_cols = [col for col in required_cols if col not in tweets_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure entry_time is timezone-naive (Eastern)
    tweets_df['entry_time'] = pd.to_datetime(tweets_df['entry_time'])
    if tweets_df['entry_time'].dt.tz is not None:
        tweets_df['entry_time'] = (
            tweets_df['entry_time']
            .dt.tz_convert('US/Eastern')
            .dt.tz_localize(None)
        )

    print(f"  Loaded {len(tweets_df)} tweet-ticker pairs")
    print(f"  Unique tweets: {tweets_df['tweet_id'].nunique()}")
    print(f"  Tickers: {tweets_df['ticker'].unique()}")
    
    # Clean tweets (remove HTML tags, etc.)
    print("\n" + "="*80)
    print("CLEANING TWEETS")
    print("="*80)
    cleaner = TweetCleaner()
    tweets_df = cleaner.clean_tweets(tweets_df, text_column='tweet_content')
    
    # Filter to most important tweets
    print("\n" + "="*80)
    print("FILTERING IMPORTANT TWEETS")
    print("="*80)
    tweets_df = cleaner.filter_important_tweets(
        tweets_df, 
        top_percentile=0.4,  # Keep top 40% most important
        min_score=None
    )
    
    print(f"\nFinal dataset: {len(tweets_df)} tweet-ticker pairs")

    # Load market data
    print("\nLoading market data...")
    price_data = {}
    missing_tickers = []
    
    for ticker in TICKERS:
        price_path = DATA_DIR / f'{ticker}.parquet'
        if price_path.exists():
            try:
                price_pl = pl.read_parquet(price_path)
                # Convert to pandas
                price_df = price_pl.to_pandas()
                
                # Ensure timestamp column exists and handle timezone
                if 'ts_event' in price_df.columns:
                    price_df['timestamp'] = pd.to_datetime(price_df['ts_event'])
                    # Convert UTC -> US/Eastern (naive) to match tweet entry_time
                    if price_df['timestamp'].dt.tz is not None:
                        price_df['timestamp'] = (
                            price_df['timestamp']
                            .dt.tz_convert('US/Eastern')
                            .dt.tz_localize(None)
                        )
                    price_df = price_df.set_index('timestamp').sort_index()
                elif 'timestamp' in price_df.columns:
                    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
                    if price_df['timestamp'].dt.tz is not None:
                        price_df['timestamp'] = (
                            price_df['timestamp']
                            .dt.tz_convert('US/Eastern')
                            .dt.tz_localize(None)
                        )
                    price_df = price_df.set_index('timestamp').sort_index()
                elif price_df.index.name == 'timestamp' or isinstance(price_df.index, pd.DatetimeIndex):
                    if price_df.index.tz is not None:
                        price_df.index = price_df.index.tz_convert('US/Eastern').tz_localize(None)
                else:
                    print(f"  Warning: {ticker} price data missing timestamp, skipping")
                    continue

                # Ensure OHLC columns exist
                required_price_cols = ['close']
                if all(col in price_df.columns for col in required_price_cols):
                    price_data[ticker] = price_df
                    print(f"  Loaded {ticker}: {len(price_df):,} bars "
                          f"({price_df.index.min()} to {price_df.index.max()})")
                else:
                    print(f"  Warning: {ticker} missing required price columns")
            except Exception as e:
                print(f"  Error loading {ticker}: {e}")
                missing_tickers.append(ticker)
        else:
            print(f"  {ticker}: File not found")
            missing_tickers.append(ticker)
    
    if len(missing_tickers) > 0:
        print(f"\nWarning: Missing market data for {len(missing_tickers)} tickers: {missing_tickers}")
        print("  The script will continue with available tickers.")
        print("  Ensure market data files exist in: data/{TICKER}.parquet")
    
    if len(price_data) == 0:
        raise FileNotFoundError(
            "\nNo market data found!\n"
            "Please provide market data parquet files in data/ directory.\n"
            f"Expected files: {[f'data/{t}.parquet' for t in TICKERS]}"
        )

    # Filter to only tickers that have price data
    available_tickers = [t for t in TICKERS if t in price_data]
    print(f"\nAvailable tickers with market data: {available_tickers}")
    if len(available_tickers) < len(TICKERS):
        skipped = [t for t in TICKERS if t not in price_data]
        print(f"  Skipping tickers without data: {skipped}")
    
    # Filter tweets to only available tickers
    tweets_df = tweets_df[tweets_df['ticker'].isin(available_tickers)].copy()
    print(f"  Tweets after filtering to available tickers: {len(tweets_df)}")

    # Calculate forward returns
    print("\nCalculating forward returns...")
    returns_df = calculate_forward_returns(tweets_df, price_data)
    
    # Validate correlations
    print("\n" + "="*80)
    print("VALIDATING CORRELATIONS")
    print("="*80)
    validator = CorrelationValidator(min_correlation=0.03, min_p_value=0.1)
    validated = validator.validate_relationships(tweets_df, returns_df, available_tickers, return_horizon='30m')
    
    # Filter to validated tickers (optional - set False to use all available tickers)
    use_validated_only = False
    if use_validated_only:
        tweets_df, returns_df = validator.filter_by_validation(tweets_df, returns_df, available_tickers)
        valid_tickers = validator.get_valid_tickers()
        if len(valid_tickers) > 0:
            print(f"\nUsing only validated tickers: {valid_tickers}")
        else:
            valid_tickers = available_tickers
    else:
        print(f"\nUsing all available tickers: {available_tickers}")
        valid_tickers = available_tickers

    return tweets_df, price_data, returns_df, validator, valid_tickers


def calculate_forward_returns(tweets_df: pd.DataFrame, price_data: Dict) -> pd.DataFrame:
    """Calculate forward returns for all tickers using high-performance vector operations."""
    print(f"  Calculating forward returns for {len(tweets_df)} tweets across {len(price_data)} tickers...")
    
    # Sort tweets by time for merge_asof
    tweets_sorted = tweets_df.sort_values('entry_time').copy()
    all_returns = []

    horizons = [5, 15, 30, 60]

    for ticker in TICKERS:
        if ticker not in price_data:
            continue

        price_df = price_data[ticker]
        if 'close' not in price_df.columns:
            continue

        # Prepare price data for merge_asof (must be sorted by key)
        price_df = price_df.sort_index()
        
        # Filter tweets for this ticker
        ticker_tweets = tweets_sorted[tweets_sorted['ticker'] == ticker].copy()
        if ticker_tweets.empty:
            continue
            
        # Match entry prices
        # merge_asof matches on 'entry_time' to the nearest 'index' in price_df
        temp_df = pd.merge_asof(
            ticker_tweets[['tweet_id', 'entry_time']],
            price_df[['close']],
            left_on='entry_time',
            right_index=True,
            direction='forward'  # Match the same or next available price
        )
        temp_df.rename(columns={'close': 'entry_price'}, inplace=True)

        # Calculate exit prices for each horizon
        for horizon in horizons:
            exit_col = f'exit_price_{horizon}m'
            ticker_tweets[f'exit_time_{horizon}m'] = ticker_tweets['entry_time'] + pd.Timedelta(minutes=horizon)
            
            # Match exit prices
            matched_exit = pd.merge_asof(
                ticker_tweets[[f'exit_time_{horizon}m']],
                price_df[['close']],
                left_on=f'exit_time_{horizon}m',
                right_index=True,
                direction='forward'
            )
            
            temp_df[f'{ticker}_{horizon}m'] = (matched_exit['close'] - temp_df['entry_price']) / temp_df['entry_price']
            # Fill NaNs where price wasn't found (e.g., end of data)
            temp_df[f'{ticker}_{horizon}m'] = temp_df[f'{ticker}_{horizon}m'].fillna(0.0)

        all_returns.append(temp_df.drop(columns=['entry_time', 'entry_price']))

    if not all_returns:
        return pd.DataFrame()

    returns_df = pd.concat(all_returns, ignore_index=True)
    return returns_df


def run_method(method_name: str, scorer, tweets_df: pd.DataFrame, 
               returns_df: pd.DataFrame, price_data: Dict, 
               output_dir: Path, return_horizon: str = '30m',
               valid_tickers: Optional[List[str]] = None):
    """
    Run a single method: train, backtest, and save results.

    Args:
        method_name: Name of the method
        scorer: Scorer instance
        tweets_df: Tweet data
        returns_df: Forward returns
        price_data: Price data dictionary
        output_dir: Output directory
        return_horizon: Return horizon for scoring
    """
    print(f"\n{'='*80}")
    print(f"METHOD: {method_name.upper()}")
    print(f"{'='*80}")

    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    # Train the method
    try:
        if isinstance(scorer, EnsembleScorer):
            scorer.train_all_methods(tweets_df, returns_df, valid_tickers if valid_tickers else TICKERS, return_horizon)
            # Filter to test period (no look-ahead) - use any trained component
            for comp, attr in [(scorer.correlation_scorer, 'use_correlation'),
                               (scorer.bow_scorer, 'use_bow'),
                               (scorer.embedding_scorer, 'use_embedding')]:
                if getattr(scorer, attr, False) and comp is not None and hasattr(comp, 'get_backtest_tweets'):
                    tweets_df = comp.get_backtest_tweets(tweets_df)
                    break
            if len(tweets_df) == 0:
                print("  Warning: No tweets in test period after train split")
                return None
            print(f"  Backtesting ensemble on {len(tweets_df)} test-period tweets")
        elif isinstance(scorer, CorrelationDiscovery):
            scorer.discover_relationships(tweets_df, returns_df, valid_tickers if valid_tickers else TICKERS, return_horizon=return_horizon)
            tweets_df = scorer.get_backtest_tweets(tweets_df)
            if len(tweets_df) == 0:
                print("  Warning: No tweets in test period after train split")
                return None
            print(f"  Backtesting on {len(tweets_df)} test-period tweets")

            # --- Feature Importance ---
            print("\n" + "="*60)
            print("FEATURE IMPORTANCE (correlation method)")
            print("="*60)
            if scorer.correlation_matrix:
                for t in scorer.correlation_matrix:
                    scorer.print_feature_importance(t, top_n=15)
                fi_df = scorer.get_all_feature_importance()
                if not fi_df.empty:
                    fi_path = method_dir / 'feature_importance.csv'
                    fi_df.to_csv(fi_path, index=False)
                    print(f"\n  [OK] Feature importance saved to: {fi_path}")
        elif isinstance(scorer, BagOfWordsScorer):
            scorer.discover_keywords(tweets_df, returns_df, valid_tickers if valid_tickers else TICKERS, return_horizon=return_horizon)
            tweets_df = scorer.get_backtest_tweets(tweets_df)
            if len(tweets_df) == 0:
                print("  Warning: No tweets in test period after train split")
                return None
            print(f"  Backtesting on {len(tweets_df)} test-period tweets")
        elif isinstance(scorer, EmbeddingScorer):
            scorer.train_models(tweets_df, returns_df, valid_tickers if valid_tickers else TICKERS, return_horizon=return_horizon)
            tweets_df = scorer.get_backtest_tweets(tweets_df)
            if len(tweets_df) == 0:
                print("  Warning: No tweets in test period after train split")
                return None
            print(f"  Backtesting on {len(tweets_df)} test-period tweets")
        elif isinstance(scorer, LLMScorer):
            print("  LLM method doesn't require training (zero-shot)")
        elif isinstance(scorer, SentimentScorer):
            print("  Sentiment method doesn't require training (zero-shot)")
        elif isinstance(scorer, MultiTimeframeScorer):
            # Train the base scorer if it needs training
            base_scorer = scorer.base_scorer
            if isinstance(base_scorer, CorrelationDiscovery):
                base_scorer.discover_relationships(tweets_df, returns_df, valid_tickers if valid_tickers else TICKERS, return_horizon=return_horizon)
                tweets_df = base_scorer.get_backtest_tweets(tweets_df)
            elif isinstance(base_scorer, BagOfWordsScorer):
                base_scorer.discover_keywords(tweets_df, returns_df, valid_tickers if valid_tickers else TICKERS, return_horizon=return_horizon)
                tweets_df = base_scorer.get_backtest_tweets(tweets_df)
            elif isinstance(base_scorer, EmbeddingScorer):
                base_scorer.train_models(tweets_df, returns_df, valid_tickers if valid_tickers else TICKERS, return_horizon=return_horizon)
                tweets_df = base_scorer.get_backtest_tweets(tweets_df)
            if len(tweets_df) == 0:
                print("  Warning: No tweets in test period")
                return None
            print("  Multi-timeframe scorer initialized")
        else:
            print(f"  Warning: Unknown scorer type: {type(scorer)}")
    except Exception as e:
        print(f"  Error training {method_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Backtest
    print(f"\nBacktesting {method_name}...")
    backtester = TweetStrategyBacktester(
        initial_capital=INITIAL_CAPITAL,
        holding_period_minutes=HOLDING_PERIOD_MINUTES,
        apply_transaction_costs=True,
        max_trades_per_day=3,  # Limit trades per day for better quality
        use_time_filter=True,  # Filter to 12-14 ET (best performing hours)
        min_hour=12,
        max_hour=14,
        use_regime_detection=True,  # Enable market regime detection
        use_advanced_sizing=True  # Enable Kelly Criterion position sizing
    )

    try:
        # Use adaptive threshold based on method
        # Lower thresholds = more trades; retry with lower threshold if no trades
        if method_name == 'correlation':
            min_threshold = 0.0002  # Slightly higher for correlation
        elif method_name == 'bag_of_words':
            min_threshold = 0.0003  # Lowered - BOW scores are typically small (excess returns)
        elif method_name == 'embedding':
            min_threshold = 0.0005  # Medium threshold for embeddings
        elif method_name == 'ensemble':
            min_threshold = 0.0001  # Very low - ensemble blends methods, scores can be small
        elif method_name == 'sentiment':
            min_threshold = 0.0005  # Medium threshold for sentiment
        elif method_name == 'multi_timeframe':
            min_threshold = 0.0003  # Lower threshold (already filtered by timeframe selection)
        else:
            min_threshold = 0.0
        
        # Also filter by absolute value to catch both positive and negative signals
        min_threshold = abs(min_threshold)
        
        # Check if scorer supports multi-timeframe
        use_multi_timeframe = isinstance(scorer, MultiTimeframeScorer)
        
        # Use validated tickers if provided, otherwise use all
        tickers_to_use = valid_tickers if valid_tickers is not None else TICKERS
        
        # Try backtest; retry with lower threshold if no trades
        for attempt, threshold in enumerate([min_threshold, min_threshold * 0.5, 0.0]):
            effective_threshold = max(threshold, 1e-6)  # Avoid exact zero
            if attempt > 0:
                print(f"  Retrying with lower threshold: {effective_threshold:.6f}")
            trades_df, equity_df, final_capital = backtester.backtest_strategy(
                tweets_df=tweets_df,
                scorer=scorer,  # Use the scorer passed to this function
                tickers=tickers_to_use,  # Use only validated tickers
                price_data=price_data,
                return_horizon=return_horizon,
                min_score_threshold=effective_threshold,  # Filter low-confidence trades
                use_multi_timeframe=use_multi_timeframe,
                returns_df=returns_df  # Pass returns_df for multi-timeframe
            )
            if len(trades_df) > 0:
                if attempt > 0:
                    print(f"  Got {len(trades_df)} trades with lower threshold")
                break
        else:
            print(f"  Warning: No trades generated for {method_name} (tried thresholds down to 0)")
            return None

        # Calculate metrics
        total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
        num_trades = len(trades_df)
        win_rate = (trades_df['net_return'] > 0).mean() if len(trades_df) > 0 else 0.0
        avg_return = trades_df['net_return'].mean() if len(trades_df) > 0 else 0.0

        print(f"\n  Results:")
        print(f"    Total Return: {total_return*100:.2f}%")
        print(f"    Final Capital: ${final_capital:,.2f}")
        print(f"    Number of Trades: {num_trades}")
        print(f"    Win Rate: {win_rate*100:.2f}%")
        print(f"    Avg Return per Trade: {avg_return*100:.4f}%")

        # Save results
        trades_df.to_csv(method_dir / 'trades.csv', index=False)
        equity_df.to_csv(method_dir / 'equity_curve.csv', index=False)

        # Save metrics
        metrics = {
            'method': method_name,
            'total_return': total_return,
            'final_capital': final_capital,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_return': avg_return
        }

        with open(method_dir / 'metrics.txt', 'w') as f:
            f.write(f"{method_name.upper()} RESULTS\n")
            f.write("="*60 + "\n\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    if 'return' in key or 'rate' in key:
                        f.write(f"{key}: {value*100:.2f}%\n")
                    else:
                        f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")

        # Generate quantstats full tearsheet (with SPY benchmark if available)
        if len(equity_df) > 0 and QUANTSTATS_AVAILABLE:
            try:
                print(f"\n  Generating quantstats tearsheet...")
                
                equity_df_copy = equity_df.copy()
                equity_df_copy['time'] = pd.to_datetime(equity_df_copy['time'])
                equity_df_copy = equity_df_copy.set_index('time').sort_index()
                equity_series = equity_df_copy['equity']
                
                # Resample to daily for proper benchmark comparison
                daily_equity = equity_series.resample('D').last().ffill().dropna()
                returns_series = daily_equity.pct_change().dropna()
                
                if len(returns_series) > 2:
                    # Try to add SPY benchmark
                    benchmark_returns = None
                    if price_data and 'SPY' in price_data:
                        spy = price_data['SPY']
                        if 'close' in spy.columns:
                            tmin, tmax = returns_series.index.min(), returns_series.index.max()
                            spy_daily = spy['close'].resample('D').last().dropna()
                            spy_daily = spy_daily[(spy_daily.index >= tmin - pd.Timedelta(days=5)) & 
                                                  (spy_daily.index <= tmax + pd.Timedelta(days=5))]
                            benchmark_returns = spy_daily.pct_change().dropna()
                            common = returns_series.index.intersection(benchmark_returns.index)
                            if len(common) > 5:
                                ret_a = returns_series.reindex(common).ffill().dropna()
                                bench_a = benchmark_returns.reindex(common).ffill().dropna()
                                common_idx = ret_a.index.intersection(bench_a.index)
                                if len(common_idx) > 5:
                                    benchmark_returns = bench_a.loc[common_idx]
                                    returns_series = ret_a.loc[common_idx]
                    
                    # Full tearsheet with benchmark
                    if benchmark_returns is not None and len(benchmark_returns) > 5:
                        qs.reports.html(
                            returns_series,
                            benchmark_returns,
                            output=(method_dir / 'quantstats_report.html').as_posix(),
                            title=f'{method_name.upper()} Strategy vs SPY'
                        )
                    else:
                        qs.reports.html(
                            returns_series,
                            output=(method_dir / 'quantstats_report.html').as_posix(),
                            title=f'{method_name.upper()} Strategy'
                        )
                    
                    qs.plots.returns(returns_series, savefig=(method_dir / 'equity_curve.png').as_posix())
                    qs.plots.monthly_heatmap(returns_series, savefig=(method_dir / 'monthly_heatmap.png').as_posix())
                    
                    print(f"    Quantstats tearsheet saved to {method_dir / 'quantstats_report.html'}")
            except Exception as e:
                print(f"    Warning: Quantstats generation failed: {e}")
                import traceback
                traceback.print_exc()

        # Also create simple matplotlib equity curve
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 6))
            # Use original equity_df which still has 'time' column
            equity_df_plot = equity_df.copy()
            equity_df_plot['time'] = pd.to_datetime(equity_df_plot['time'])
            ax.plot(equity_df_plot['time'], equity_df_plot['equity'], linewidth=2)
            ax.set_title(f'{method_name.upper()} - Equity Curve')
            ax.set_xlabel('Time')
            ax.set_ylabel('Equity ($)')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(method_dir / 'equity_curve_simple.png', dpi=150)
            plt.close()
        except Exception as e:
            print(f"    Warning: Simple equity curve generation failed: {e}")
            import traceback
            traceback.print_exc()

        return {
            'method': method_name,
            'metrics': metrics,
            'trades_df': trades_df,
            'equity_df': equity_df
        }

    except Exception as e:
        print(f"  Error backtesting {method_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run tweet-ticker backtesting methods')
    parser.add_argument('--method', type=str, default=None,
                        choices=['correlation', 'bag_of_words', 'embedding', 'llm',
                                 'ensemble', 'sentiment', 'multi_timeframe'],
                        help='Run only a specific method (default: run all)')
    args = parser.parse_args()
    selected_method = args.method

    print("="*80)
    print("TWEET-TICKER RELATIONSHIP DISCOVERY & BACKTESTING")
    print("="*80)
    if selected_method:
        print(f"  Running method: {selected_method}")
    else:
        print("  Running all methods")

    # Create output directory
    output_dir = OUTPUT_DIR / 'tweet_ticker_methods'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load data (returns valid_tickers based on use_validated_only in load_data)
    tweets_df, price_data, returns_df, validator, valid_tickers = load_data()
    print(f"\n✓ Backtesting with tickers: {valid_tickers}")

    # Merge returns into tweets_df for easier access (if tweet_id exists in returns_df)
    if 'tweet_id' in returns_df.columns and 'tweet_id' in tweets_df.columns:
        tweets_df = pd.merge(tweets_df, returns_df, on='tweet_id', how='left')
    else:
        # If no tweet_id match, try to align by index
        print("  Warning: Could not merge returns by tweet_id, aligning by index")
        if len(returns_df) == len(tweets_df):
            for col in returns_df.columns:
                if col != 'tweet_id':
                    tweets_df[col] = returns_df[col].values

    # Define methods to run
    methods = []

    def should_run(name):
        return selected_method is None or selected_method == name

    # Method 1: Correlation
    if should_run('correlation'):
        methods.append(('correlation', CorrelationDiscovery(
            train_ratio=0.7,           # No look-ahead: train on first 70%, backtest on last 30%
            p_value_max=0.05,            # Only statistically significant features
            min_abs_correlation=0.02,    # Filter noise
            use_price_features=False     # Avoid look-ahead from price alignment
        )))

    # Method 2: Bag-of-Words (refined) - run early for comparison
    if should_run('bag_of_words'):
        methods.append(('bag_of_words', BagOfWordsScorer(
            max_features=2000,
            ngram_range=(1, 3),
            max_keywords_per_ticker=60,   # Focus on strongest signals
            p_value_threshold=0.12,      # Balanced - avoid noisy keywords
            min_occurrences=6,            # Slightly higher for robustness
            train_ratio=0.7                # No look-ahead: train on first 70%
        )))

    # Method 3: Embedding
    if should_run('embedding'):
        methods.append(('embedding', EmbeddingScorer(alpha=1.0, k_neighbors=50, train_ratio=0.7)))

    # Method 4: LLM (optional - only if API key available)
    if should_run('llm'):
        try:
            llm_scorer = LLMScorer(use_llm=True)
            if llm_scorer.use_llm:
                methods.append(('llm', llm_scorer))
            else:
                print("\nSkipping LLM method (no API key available)")
        except:
            print("\nSkipping LLM method")

    # Method 5: Ensemble
    if should_run('ensemble'):
        methods.append(('ensemble', EnsembleScorer(
            use_correlation=True,
            use_bow=True,
            use_embedding=True,
            use_llm=False  # Set to True if LLM is available
        )))

    # Method 6: Sentiment Analysis (FinBERT)
    if should_run('sentiment'):
        try:
            sentiment_analyzer = FinancialSentimentAnalyzer()
            sentiment_scorer = SentimentScorer(sentiment_analyzer)
            methods.append(('sentiment', sentiment_scorer))
            print("\nAdded sentiment analysis method")
        except Exception as e:
            print(f"\nSkipping sentiment method: {e}")

    # Method 7: Multi-Timeframe Analysis
    if should_run('multi_timeframe'):
        try:
            base_correlation = CorrelationDiscovery(
                train_ratio=0.7, p_value_max=0.05, min_abs_correlation=0.02, use_price_features=False
            )
            multi_timeframe_scorer = MultiTimeframeScorer(base_correlation, horizons=[5, 15, 30, 60, 240])
            methods.append(('multi_timeframe', multi_timeframe_scorer))
            print("\nAdded multi-timeframe analysis method")
        except Exception as e:
            print(f"\nSkipping multi-timeframe method: {e}")

    # Run each method
    results = {}
    for method_name, scorer in methods:
        result = run_method(
            method_name=method_name,
            scorer=scorer,
            tweets_df=tweets_df,
            returns_df=returns_df,
            price_data=price_data,
            output_dir=output_dir,
            return_horizon='30m',
            valid_tickers=valid_tickers  # Pass validated tickers
        )
        if result:
            results[method_name] = result
        else:
            # Include in comparison with zeros so user sees all methods were attempted
            results[method_name] = {
                'metrics': {
                    'method': method_name,
                    'total_return': 0.0,
                    'final_capital': INITIAL_CAPITAL,
                    'num_trades': 0,
                    'win_rate': 0.0,
                    'avg_return': 0.0
                },
                'trades_df': pd.DataFrame(),
                'equity_df': pd.DataFrame([{'time': pd.Timestamp.now(), 'equity': INITIAL_CAPITAL}])
            }

    # Generate comparison report
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*80}")

    comparison_data = []
    for method_name, result in results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Method': method_name,
            'Total Return (%)': metrics['total_return'] * 100,
            'Final Capital ($)': metrics['final_capital'],
            'Number of Trades': metrics['num_trades'],
            'Win Rate (%)': metrics['win_rate'] * 100,
            'Avg Return per Trade (%)': metrics['avg_return'] * 100
        })

    if len(comparison_data) == 0:
        print("\nNo methods produced results. Skipping comparison report.")
        return

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Total Return (%)', ascending=False)
    comparison_df.to_csv(output_dir / 'method_comparison.csv', index=False)

    print("\nMethod Comparison:")
    print(comparison_df.to_string(index=False))

    # Generate comparison equity curves plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for method_name, result in results.items():
            metrics = result.get('metrics', {})
            if metrics.get('num_trades', 0) == 0:
                continue  # Skip zero-trade methods in plot
            equity_df = result['equity_df'].copy()
            if len(equity_df) == 0:
                continue
                
            # Ensure time column exists and is datetime
            if 'time' not in equity_df.columns:
                continue
                
            equity_df['time'] = pd.to_datetime(equity_df['time'])
            equity_df = equity_df.set_index('time')
            
            # Normalize to starting capital
            equity_df['equity_normalized'] = (equity_df['equity'] / INITIAL_CAPITAL - 1) * 100
            
            ax.plot(equity_df.index, equity_df['equity_normalized'], 
                   label=method_name, linewidth=2, alpha=0.8)
        
        ax.set_title('Equity Curves Comparison - All Methods', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'all_methods_comparison.png', dpi=150)
        plt.close()
        print(f"\nComparison plot saved to {output_dir / 'all_methods_comparison.png'}")
    except Exception as e:
        print(f"Warning: Comparison plot generation failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*80}")
    print("ALL METHODS COMPLETED")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nSummary:")
    for method_name, result in results.items():
        metrics = result['metrics']
        status = f"{metrics['total_return']*100:.2f}% return, {metrics['num_trades']} trades"
        if metrics['num_trades'] == 0:
            status += " (no trades)"
        print(f"  {method_name}: {status}")


if __name__ == '__main__':
    main()
