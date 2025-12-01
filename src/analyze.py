#!/usr/bin/env python3
"""
Multi-stock L1 consolidation analysis.

Analyzes the predictive power of different L1 (Level 1 market data) consolidation
methods across multiple stocks. Methods include:
- Consensus Mid: Simple midpoint across all exchanges
- Consensus Microprice: Volume-weighted microprice
- EMA Estimate: Exponential Moving Average filter
- Kalman Filter: Kalman filtering for state estimation

Usage:
    python analyze.py --parquet <path> [--output <dir>] [--max-rows <n>]
"""

import argparse
import os
import sys
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from L1_transform.per_exchange_state import MultiExchangeBook
from L1_transform.consensus_mid import compute_consensus_mid
from L1_transform.consensus_microprice import compute_consensus_micro
from L1_transform.ema_estimate import EMAEfficientPrice
from L1_transform.kalman_filter import KalmanEfficientPrice
from L1_transform.metrics import PredictionMetrics, evaluate_method

warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Multi-stock L1 consolidation method analysis'
    )
    parser.add_argument(
        '--parquet', type=str, required=True,
        help='Path to parquet file with market data'
    )
    parser.add_argument(
        '--output', type=str, default='../output',
        help='Output directory for results (default: ../output)'
    )
    parser.add_argument(
        '--max-rows', type=int, default=50000,
        help='Max rows to process per stock (default: 50000)'
    )
    parser.add_argument(
        '--tickers', type=str,
        help='Comma-separated tickers to analyze (default: all)'
    )
    return parser.parse_args()


def extract_stock_data(df_pl, ticker: str) -> pd.DataFrame:
    """Extract data for a specific ticker from polars dataframe."""
    df = df_pl.filter(pl.col('Ticker') == ticker).to_pandas()
    return df


def create_l1_snapshots(df_valid: pd.DataFrame) -> pd.DataFrame:
    """Create L1 snapshots from raw market data."""
    snapshots = []

    def get_l1_snapshot(group):
        """Extract L1 for each exchange in a time window."""
        bids = group[group['EventType'].str.contains('BID')]
        asks = group[group['EventType'].str.contains('ASK')]
        result = {}

        for exchange in group['Exchange'].unique():
            ex_bids = bids[bids['Exchange'] == exchange]
            ex_asks = asks[asks['Exchange'] == exchange]

            if len(ex_bids) > 0 and len(ex_asks) > 0:
                bid_row = ex_bids.iloc[-1]
                ask_row = ex_asks.iloc[-1]

                if bid_row['Price'] > 0 and ask_row['Price'] > 0:
                    result[exchange] = {
                        'bid': bid_row['Price'],
                        'bid_size': bid_row['Quantity'],
                        'ask': ask_row['Price'],
                        'ask_size': ask_row['Quantity'],
                        'timestamp': bid_row['Timestamp_ns']
                    }
        return result

    # Create time windows
    df_valid['window'] = (df_valid['Timestamp_ns'] * 10).astype(int)

    for window_id, group in df_valid.groupby('window'):
        l1_data = get_l1_snapshot(group)
        for exchange, quote in l1_data.items():
            snapshots.append({
                'window': window_id,
                'timestamp': quote['timestamp'],
                'exchange': exchange,
                'bid': quote['bid'],
                'ask': quote['ask'],
                'bid_size': quote['bid_size'],
                'ask_size': quote['ask_size']
            })

    return pd.DataFrame(snapshots) if snapshots else pd.DataFrame()


def generate_predictions(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions using all methods."""
    results = {
        'timestamp': [],
        'actual_next_price': [],
        'consensus_mid_pred': [],
        'consensus_micro_pred': [],
        'ema_estimate_pred': [],
        'kalman_filter_pred': []
    }

    ema = EMAEfficientPrice(alpha=0.15)
    kalman = KalmanEfficientPrice(process_var=0.02, obs_var=0.5)

    windows_list = sorted(snapshot_df['window'].unique())

    for i in range(len(windows_list) - 1):
        window_id = windows_list[i]
        next_window_id = windows_list[i + 1]

        window_data = snapshot_df[snapshot_df['window'] == window_id]
        next_window_data = snapshot_df[snapshot_df['window'] == next_window_id]

        if len(window_data) < 2 or len(next_window_data) < 2:
            continue

        timestamp = window_data['timestamp'].iloc[0]
        next_timestamp = next_window_data['timestamp'].iloc[0]

        # Build current market book
        book = MultiExchangeBook()
        for _, row in window_data.iterrows():
            book.update_quote(row['exchange'], 'BID', row['bid'], row['bid_size'], timestamp)
            book.update_quote(row['exchange'], 'ASK', row['ask'], row['ask_size'], timestamp)

        # Get predictions
        mid = compute_consensus_mid(book, timestamp)
        micro = compute_consensus_micro(book, timestamp)

        if mid is None or micro is None:
            continue

        ema_val = ema.update(mid)
        kalman_val = kalman.update(mid)

        # Build next market book for actual price
        next_book = MultiExchangeBook()
        for _, row in next_window_data.iterrows():
            next_book.update_quote(row['exchange'], 'BID', row['bid'], row['bid_size'], next_timestamp)
            next_book.update_quote(row['exchange'], 'ASK', row['ask'], row['ask_size'], next_timestamp)

        actual_next = compute_consensus_mid(next_book, next_timestamp)

        if actual_next is not None:
            results['timestamp'].append(timestamp)
            results['actual_next_price'].append(actual_next)
            results['consensus_mid_pred'].append(mid)
            results['consensus_micro_pred'].append(micro)
            results['ema_estimate_pred'].append(ema_val)
            results['kalman_filter_pred'].append(kalman_val)

    return pd.DataFrame(results)


def measure_method_times(snapshot_df: pd.DataFrame) -> dict:
    """Measure computation times for each method."""
    book = MultiExchangeBook()
    test_timestamp = snapshot_df['timestamp'].iloc[0]

    for _, row in snapshot_df[snapshot_df['window'] == snapshot_df['window'].iloc[0]].iterrows():
        book.update_quote(row['exchange'], 'BID', row['bid'], row['bid_size'], row['timestamp'])
        book.update_quote(row['exchange'], 'ASK', row['ask'], row['ask_size'], row['timestamp'])

    num_iterations = 500
    computation_times = {}

    # Consensus Mid
    start = time.perf_counter()
    for _ in range(num_iterations):
        compute_consensus_mid(book, test_timestamp)
    computation_times['consensus_mid'] = (time.perf_counter() - start) / num_iterations

    # Consensus Micro
    start = time.perf_counter()
    for _ in range(num_iterations):
        compute_consensus_micro(book, test_timestamp)
    computation_times['consensus_micro'] = (time.perf_counter() - start) / num_iterations

    # EMA
    ema = EMAEfficientPrice(alpha=0.15)
    start = time.perf_counter()
    for _ in range(num_iterations):
        ema.update(test_timestamp)
    computation_times['ema_estimate'] = (time.perf_counter() - start) / num_iterations

    # Kalman
    kalman = KalmanEfficientPrice(process_var=0.02, obs_var=0.5)
    start = time.perf_counter()
    for _ in range(num_iterations):
        kalman.update(test_timestamp)
    computation_times['kalman_filter'] = (time.perf_counter() - start) / num_iterations

    return computation_times


def process_stock(df: pd.DataFrame, ticker: str, max_rows: int = 50000) -> dict:
    """Process a single stock and return metrics."""
    print(f"\nProcessing {ticker}...")

    try:
        # Limit rows if needed
        if len(df) > max_rows:
            df = df.head(max_rows)

        df['Timestamp_ns'] = pd.to_datetime(df['Timestamp']).values.astype(np.int64) / 1e9

        # Filter for valid L1 quotes
        df_valid = df[(df['Price'] > 0) & (df['Quantity'] > 0)].copy()

        if len(df_valid) == 0:
            print(f"  [SKIP] No valid quotes found")
            return None

        # Create L1 snapshots
        snapshot_df = create_l1_snapshots(df_valid)

        if len(snapshot_df) < 10:
            print(f"  [SKIP] Insufficient snapshot data")
            return None

        # Generate predictions
        results_df = generate_predictions(snapshot_df)

        if len(results_df) < 10:
            print(f"  [SKIP] Insufficient predictions generated")
            return None

        # Calculate metrics
        metrics = {}
        actual = results_df['actual_next_price'].values

        for method in ['consensus_mid_pred', 'consensus_micro_pred', 'ema_estimate_pred', 'kalman_filter_pred']:
            method_name = method.replace('_pred', '')
            predicted = results_df[method].values

            # Get accuracy and quality metrics
            acc_metrics = PredictionMetrics.calculate_accuracy_metrics(actual, predicted)
            qual_metrics = PredictionMetrics.calculate_quality_metrics(actual, predicted)

            metrics[method_name] = {
                **acc_metrics,
                **qual_metrics,
                'predictions': len(results_df)
            }

        # Measure computation times
        computation_times = measure_method_times(snapshot_df)
        for method in computation_times:
            metrics[method]['time_us'] = computation_times[method] * 1e6

        print(f"  [OK] Processed {len(results_df)} predictions from {len(snapshot_df)} snapshots")
        return {'ticker': ticker.upper(), 'metrics': metrics, 'results_df': results_df}

    except Exception as e:
        print(f"  [ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main analysis workflow."""
    args = parse_arguments()

    # Determine paths
    base_dir = Path(__file__).parent.parent

    # Handle parquet file path
    parquet_file = Path(args.parquet)
    if not parquet_file.is_absolute():
        # Try relative to base_dir first
        candidate = base_dir / parquet_file
        if candidate.exists():
            parquet_file = candidate.resolve()
        else:
            # If not found and path starts with .., resolve from current dir
            parquet_file = Path(args.parquet).resolve()

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        # Try relative to base_dir first
        candidate = base_dir / output_dir
        output_dir = candidate.resolve()
        os.makedirs(output_dir, exist_ok=True)

    # Verify inputs
    if not parquet_file.exists():
        print(f"ERROR: Parquet file not found: {parquet_file}")
        sys.exit(1)

    print(f"Output directory: {output_dir}")
    print(f"Loading parquet file: {parquet_file.name}")

    # Load and process data
    df_pl = pl.read_parquet(str(parquet_file))
    print(f"Loaded {len(df_pl)} rows")

    # Get tickers
    tickers = df_pl['Ticker'].unique().to_list()
    if args.tickers:
        requested = {t.strip().upper() for t in args.tickers.split(',')}
        tickers = [t for t in tickers if t.upper() in requested]

    print(f"Found {len(tickers)} stock(s): {', '.join(sorted(tickers))}")

    # Process all stocks
    print("\n" + "="*100)
    print("MULTI-STOCK L1 CONSOLIDATION METHOD ANALYSIS")
    print("="*100)

    all_results = {}
    for ticker in sorted(tickers):
        stock_data = extract_stock_data(df_pl, ticker)
        result = process_stock(stock_data, ticker, max_rows=args.max_rows)
        if result:
            all_results[result['ticker']] = result

    if not all_results:
        print("\nNo valid stocks could be processed.")
        sys.exit(1)

    # Generate summary
    print("\n" + "="*100)
    print("SUMMARY REPORT")
    print("="*100)

    summary_data = []
    for ticker, data in all_results.items():
        metrics = data['metrics']
        for method, method_metrics in metrics.items():
            summary_data.append({
                'Stock': ticker,
                'Method': method,
                'MAE': method_metrics['mae'],
                'RMSE': method_metrics['rmse'],
                'MAPE': method_metrics['mape'],
                'R-squared': method_metrics['r_squared'],
                'Correlation': method_metrics['correlation'],
                'Time (us)': method_metrics['time_us'],
                'Predictions': method_metrics['predictions']
            })

    summary_df = pd.DataFrame(summary_data)

    # Debug: Check what's in summary_df
    print(f"\n[DEBUG] Summary contains {len(summary_df)} rows from {summary_df['Stock'].nunique()} unique stocks")
    print(f"[DEBUG] Stocks in summary: {sorted(summary_df['Stock'].unique())[:20]}...")  # Show first 20

    # Print by stock
    for ticker in sorted(all_results.keys()):
        print(f"\n{ticker}:")
        print(f"{'Method':<25} | {'MAE':<12} | {'R-squared':<12} | {'Time (us)':<12}")
        print("-" * 70)

        ticker_data = summary_df[summary_df['Stock'] == ticker].sort_values('MAE')
        for _, row in ticker_data.iterrows():
            print(f"{row['Method']:<25} | {row['MAE']:<12.6f} | {row['R-squared']:<12.6f} | {row['Time (us)']:<12.4f}")

        best = ticker_data.iloc[0]
        print(f"\n  BEST: {best['Method']} (MAE: {best['MAE']:.6f})")

    # Cross-stock comparison
    print("\n" + "="*100)
    print("CROSS-STOCK METHOD PERFORMANCE")
    print("="*100)

    for method in ['consensus_mid', 'consensus_micro', 'ema_estimate', 'kalman_filter']:
        method_data = summary_df[summary_df['Method'] == method]
        print(f"\n{method.upper()}:")
        print(f"  Average MAE: {method_data['MAE'].mean():.6f}")
        print(f"  Average R-squared: {method_data['R-squared'].mean():.6f}")
        print(f"  Average Time: {method_data['Time (us)'].mean():.4f} microseconds")
        print(f"  Best for: {method_data.loc[method_data['MAE'].idxmin(), 'Stock']} (MAE: {method_data['MAE'].min():.6f})")

    # Overall ranking
    print("\n" + "="*100)
    print("OVERALL METHOD RANKING (Across All Stocks)")
    print("="*100)

    method_scores = {}
    for method in ['consensus_mid', 'consensus_micro', 'ema_estimate', 'kalman_filter']:
        method_data = summary_df[summary_df['Method'] == method]
        method_scores[method] = {
            'mae': method_data['MAE'].mean(),
            'r2': method_data['R-squared'].mean(),
            'time_us': method_data['Time (us)'].mean()
        }

    # Normalize and rank
    all_mae = summary_df['MAE'].values
    all_r2 = summary_df['R-squared'].values
    all_time = summary_df['Time (us)'].values

    min_mae, max_mae = all_mae.min(), all_mae.max()
    min_r2, max_r2 = all_r2.min(), all_r2.max()
    min_time, max_time = all_time.min(), all_time.max()

    print(f"\n{'Method':<25} | {'Avg MAE':<12} | {'Avg R2':<12} | {'Avg Time':<12} | {'Score':<10}")
    print("-" * 75)

    scores = {}
    for method in ['consensus_mid', 'consensus_micro', 'ema_estimate', 'kalman_filter']:
        mae_score = 10.0 * (max_mae - method_scores[method]['mae']) / (max_mae - min_mae) if max_mae != min_mae else 5.0
        r2_score = 10.0 * (method_scores[method]['r2'] - min_r2) / (max_r2 - min_r2) if max_r2 != min_r2 else 5.0
        time_score = 10.0 * (max_time - method_scores[method]['time_us']) / (max_time - min_time) if max_time != min_time else 5.0

        overall = (mae_score * 0.50) + (r2_score * 0.35) + (time_score * 0.15)
        scores[method] = overall

        print(f"{method:<25} | {method_scores[method]['mae']:<12.6f} | {method_scores[method]['r2']:<12.6f} | {method_scores[method]['time_us']:<12.4f} | {overall:<10.2f}")

    # Final ranking
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'Rank':<6} | {'Method':<25} | {'Score':<10}")
    print("-" * 45)
    for i, (method, score) in enumerate(ranking, 1):
        print(f"{i:<6} | {method:<25} | {score:<10.2f}")

    # Save results
    summary_csv_path = output_dir / 'l1_multi_stock_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n[OK] Summary saved to {summary_csv_path}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Multi-Stock L1 Method Comparison ({len(all_results)} stocks)', fontsize=16, fontweight='bold')

    # Plot 1: MAE by stock and method
    ax = axes[0, 0]
    pivot_mae = summary_df.pivot(index='Stock', columns='Method', values='MAE')
    pivot_mae.plot(kind='bar', ax=ax)
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Prediction Accuracy (MAE) by Stock')
    ax.legend(title='Method', fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: R-squared by stock and method
    ax = axes[0, 1]
    pivot_r2 = summary_df.pivot(index='Stock', columns='Method', values='R-squared')
    pivot_r2.plot(kind='bar', ax=ax)
    ax.set_ylabel('R-squared')
    ax.set_title('Prediction Quality (R-squared) by Stock')
    ax.legend(title='Method', fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Average metrics
    ax = axes[1, 0]
    methods = list(method_scores.keys())
    mae_vals = [method_scores[m]['mae'] for m in methods]
    r2_vals = [method_scores[m]['r2'] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, mae_vals, width, label='Avg MAE', alpha=0.8)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, r2_vals, width, label='Avg R-squared', alpha=0.8, color='orange')

    ax.set_xlabel('Method')
    ax.set_ylabel('MAE')
    ax2.set_ylabel('R-squared')
    ax.set_title('Average Performance Across All Stocks')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # Plot 4: Computation time
    ax = axes[1, 1]
    time_vals = [method_scores[m]['time_us'] for m in methods]
    bars = ax.bar(methods, time_vals, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Time (microseconds)')
    ax.set_title('Average Computation Time')
    for bar, val in zip(bars, time_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    viz_path = output_dir / 'l1_multi_stock_comparison.png'
    plt.savefig(viz_path, dpi=100, bbox_inches='tight')
    print(f"[OK] Visualization saved to {viz_path}")
    plt.close()

    print("\n[COMPLETE] Multi-stock analysis finished!")
    print(f"Processed {len(all_results)} stock(s)")


if __name__ == '__main__':
    main()
