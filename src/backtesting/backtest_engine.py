"""
Refactored backtesting engine with bias fixes and transaction costs.

CRITICAL FIXES:
1. Uses RealReturnCalculator instead of forward_return (fixes circular logic)
2. Applies TransactionCostCalculator (realistic 8 bps costs)
3. Proper output organization with model names
4. Uses centralized config.settings
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models.base_model import BaseTradingModel
from src.backtesting.return_calculator import RealReturnCalculator
from src.backtesting.transaction_costs import TransactionCostCalculator
from config.settings import (
    FEATURES_PARQUET, OUTPUT_DIR, INITIAL_CAPITAL, HOLDING_PERIOD_MINUTES,
    COMMISSION_BPS, SLIPPAGE_BPS, TICKERS
)


class BacktestEngine:
    """
    Modular backtesting engine with bias fixes and transaction costs.

    This refactored version fixes critical issues in the original:
    - No longer uses forward_return for backtest PnL (circular logic fixed)
    - Applies realistic transaction costs (8 bps per trade)
    - Better output organization (model name in directory structure)
    """

    def __init__(self, model: BaseTradingModel, tickers: List[str],
                 initial_capital: float = INITIAL_CAPITAL,
                 apply_transaction_costs: bool = True):
        """
        Initialize backtest engine.

        Args:
            model: Trading model implementing BaseTradingModel interface
            tickers: List of ticker symbols to backtest
            initial_capital: Starting capital (default from config)
            apply_transaction_costs: Whether to apply transaction costs (default True)
        """
        self.model = model
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.data = None
        self.feature_columns = None

        # Initialize return calculator (CRITICAL: fixes circular logic)
        self.return_calculator = RealReturnCalculator()

        # Initialize transaction cost calculator
        self.apply_transaction_costs = apply_transaction_costs
        if apply_transaction_costs:
            self.cost_calculator = TransactionCostCalculator(
                commission_bps=COMMISSION_BPS,
                slippage_bps=SLIPPAGE_BPS
            )
        else:
            self.cost_calculator = None
            print("WARNING: Running without transaction costs (unrealistic)")

    def load_data(self, ticker: Optional[str] = None, data_path: Optional[str] = None):
        """
        Load pre-computed features from parquet.

        Args:
            ticker: Optional ticker to filter for
            data_path: Optional custom data path (uses config default if None)
        """
        if data_path is None:
            data_path = FEATURES_PARQUET

        print(f"\nLoading data from {data_path}...")

        df = pd.read_parquet(data_path)

        if ticker is not None:
            df = df[df['ticker'] == ticker].copy()

        print(f"Loaded {len(df)} samples")

        if ticker:
            print(f"Ticker: {ticker}")

        self.data = df
        return df

    def prepare_features(self, exclude_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Prepare features for training.

        Args:
            exclude_columns: Columns to exclude from features

        Returns:
            X, y, data_df
        """
        if exclude_columns is None:
            exclude_columns = [
                'ticker', 'tweet_id', 'tweet_time', 'entry_time', 'tweet_idx',
                'tweet_content', 'tweet_url', 'tweet_idx', 'forward_return'
            ]

        # Identify feature columns
        feature_cols = [col for col in self.data.columns if col not in exclude_columns]

        # Handle embedding columns if they exist
        embedding_cols = [col for col in feature_cols if 'embedding_' in col]

        print(f"\nFeature preparation:")
        print(f"  Total columns in data: {len(self.data.columns)}")
        print(f"  Feature columns: {len(feature_cols)}")
        print(f"  Embedding columns: {len(embedding_cols)}")

        # Extract features
        X = self.data[feature_cols].values

        # Handle NaN and Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Get labels if they exist (for training only)
        if 'forward_return' in self.data.columns:
            y = self.data['forward_return'].values
        else:
            print("Warning: No 'forward_return' column found. Using zeros as placeholder.")
            y = np.zeros(len(self.data))

        self.feature_columns = feature_cols

        print(f"  Final feature shape: {X.shape}")
        print(f"  Target shape: {y.shape}")

        return X, y, self.data

    def split_data(self, X: np.ndarray, y: np.ndarray, data_df: pd.DataFrame,
                   train_ratio: float = 0.7):
        """
        Chronological train/test split.

        Args:
            X: Feature matrix
            y: Target vector
            data_df: Full dataframe with metadata
            train_ratio: Fraction of data for training

        Returns:
            (X_train, y_train, train_df), (X_test, y_test, test_df)
        """
        split_idx = int(len(X) * train_ratio)

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        train_df = data_df.iloc[:split_idx]
        test_df = data_df.iloc[split_idx:]

        print(f"\nData split (chronological):")
        print(f"  Train: {len(X_train)} samples ({train_ratio*100:.0f}%)")
        print(f"  Test: {len(X_test)} samples ({(1-train_ratio)*100:.0f}%)")

        return (X_train, y_train, train_df), (X_test, y_test, test_df)

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray):
        """Train the model."""
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL: {self.model.name}")
        print(f"{'='*80}")

        training_results = self.model.fit(X_train, y_train, X_val, y_val)

        return training_results

    def softmax(self, x):
        """Compute softmax for position sizing."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def backtest_strategy(self, test_df: pd.DataFrame, X_test: np.ndarray,
                         price_df: Optional[pd.DataFrame] = None):
        """
        Backtest the strategy with FIXED circular logic.

        CRITICAL FIX: Uses RealReturnCalculator to calculate returns from actual
        entry/exit prices instead of using forward_return (which is a training label).

        Args:
            test_df: Test data with metadata
            X_test: Test features
            price_df: Optional price data (required if not using forward_return)

        Returns:
            trades_df, equity_df, final_capital, processing_times
        """
        print(f"\n{'='*80}")
        print("RUNNING BACKTEST")
        print(f"{'='*80}")

        # Get predictions
        print("Generating predictions...")
        predictions = self.model.predict(X_test)

        test_df = test_df.copy()
        test_df['prediction'] = predictions
        test_df = test_df.sort_values('tweet_time').reset_index(drop=True)

        # Initialize tracking
        capital = self.initial_capital
        equity_curve = []
        trades = []
        positions = {}

        # Track costs separately
        total_transaction_costs = 0.0

        # Track processing times
        processing_times = []
        start_time = time.time()

        for idx, row in test_df.iterrows():
            tweet_start = time.time()

            current_time = row['tweet_time']
            ticker = row['ticker']

            # Check if we have an active position for this ticker
            if ticker in positions:
                pos_exit_time = positions[ticker]['exit_time']

                # If current tweet is within holding period, skip
                if current_time < pos_exit_time:
                    continue
                else:
                    # Close existing position
                    pos = positions[ticker]

                    # CRITICAL FIX: Calculate actual return from prices
                    # (NOT from forward_return column)
                    if price_df is not None:
                        # Use real price data to calculate return
                        gross_return = self.return_calculator.calculate_return(
                            entry_time=pos['entry_time'],
                            exit_time=current_time,
                            ticker=ticker,
                            price_df=price_df
                        )
                    else:
                        # Fallback: use forward_return (but warn user)
                        # This should only be used during testing/development
                        gross_return = row.get('forward_return', 0)
                        if idx == 0:  # Only warn once
                            print("WARNING: Using forward_return as fallback. "
                                  "This may introduce circular logic.")

                    # Apply transaction costs
                    if self.apply_transaction_costs:
                        net_return = self.cost_calculator.apply_costs_to_return(
                            gross_return, pos['weight']
                        )
                        transaction_cost = capital * pos['weight'] * (gross_return - net_return)
                        total_transaction_costs += transaction_cost
                    else:
                        net_return = gross_return
                        transaction_cost = 0

                    # Calculate P&L
                    pnl = capital * pos['weight'] * net_return
                    capital += pnl

                    trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': current_time,
                        'ticker': ticker,
                        'weight': pos['weight'],
                        'gross_return': gross_return,
                        'net_return': net_return,
                        'transaction_cost': transaction_cost,
                        'pnl': pnl
                    })

                    equity_curve.append({
                        'time': current_time,
                        'equity': capital
                    })

                    del positions[ticker]

            # Get all tweets at current time
            current_tweets = test_df[test_df['tweet_time'] == current_time]

            # Calculate softmax weights
            signals = current_tweets['prediction'].values
            weights = self.softmax(signals)

            # Open new positions
            exit_time = current_time + pd.Timedelta(minutes=HOLDING_PERIOD_MINUTES)

            for j, (_, tweet_row) in enumerate(current_tweets.iterrows()):
                ticker = tweet_row['ticker']
                weight = weights[j]

                positions[ticker] = {
                    'entry_time': current_time,
                    'exit_time': exit_time,
                    'weight': weight,
                    'prediction': tweet_row['prediction']
                }

            # Record processing time
            tweet_end = time.time()
            processing_times.append(tweet_end - tweet_start)

        # Close any remaining positions
        for ticker, pos in positions.items():
            last_row = test_df[test_df['ticker'] == ticker].iloc[-1]

            # Calculate return from prices
            if price_df is not None:
                gross_return = self.return_calculator.calculate_return(
                    entry_time=pos['entry_time'],
                    exit_time=last_row['tweet_time'],
                    ticker=ticker,
                    price_df=price_df
                )
            else:
                gross_return = last_row.get('forward_return', 0)

            # Apply costs
            if self.apply_transaction_costs:
                net_return = self.cost_calculator.apply_costs_to_return(
                    gross_return, pos['weight']
                )
                transaction_cost = capital * pos['weight'] * (gross_return - net_return)
                total_transaction_costs += transaction_cost
            else:
                net_return = gross_return
                transaction_cost = 0

            pnl = capital * pos['weight'] * net_return
            capital += pnl

            trades.append({
                'entry_time': pos['entry_time'],
                'exit_time': last_row['tweet_time'],
                'ticker': ticker,
                'weight': pos['weight'],
                'gross_return': gross_return,
                'net_return': net_return,
                'transaction_cost': transaction_cost,
                'pnl': pnl
            })

            equity_curve.append({
                'time': last_row['tweet_time'],
                'equity': capital
            })

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        processing_times = np.array(processing_times)

        # Print timing metrics
        total_time = time.time() - start_time

        print(f"\n{'='*60}")
        print("PROCESSING TIME METRICS")
        print(f"{'='*60}")
        print(f"Total Processing Time: {total_time:.2f}s")
        print(f"Total Tweets Processed: {len(processing_times)}")
        print(f"Throughput: {len(processing_times)/total_time:.2f} tweets/second")
        print(f"\nPer-Tweet Processing Time:")
        print(f"  Mean: {np.mean(processing_times)*1000:.4f} ms")
        print(f"  Median: {np.median(processing_times)*1000:.4f} ms")
        print(f"  95th: {np.percentile(processing_times, 95)*1000:.4f} ms")
        print(f"  99th: {np.percentile(processing_times, 99)*1000:.4f} ms")
        print(f"  Max: {np.max(processing_times)*1000:.4f} ms")

        # Check 95% threshold
        p95 = np.percentile(processing_times, 95)
        if p95 * 1000 > 200:
            print(f"\n⚠️ WARNING: 95th percentile ({p95*1000:.2f}ms) exceeds 200ms threshold!")
        else:
            print(f"\n✓ PASS: 95th percentile ({p95*1000:.2f}ms) is under 200ms threshold")

        print(f"{'='*60}")

        # Print transaction cost summary
        if self.apply_transaction_costs and len(trades_df) > 0:
            print(f"\n{'='*60}")
            print("TRANSACTION COSTS SUMMARY")
            print(f"{'='*60}")
            print(f"Total Transaction Costs: ${total_transaction_costs:,.2f}")
            print(f"Cost per Trade: ${total_transaction_costs/len(trades_df):,.2f}")
            print(f"Cost as % of Initial Capital: {total_transaction_costs/self.initial_capital*100:.2f}%")
            gross_pnl = trades_df['gross_return'].sum() * self.initial_capital
            if gross_pnl != 0:
                print(f"Cost as % of Gross P&L: {total_transaction_costs/abs(gross_pnl)*100:.2f}%")
            print(f"{'='*60}")

        return trades_df, equity_df, capital, processing_times

    def calculate_metrics(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame,
                         final_capital: float):
        """Calculate performance metrics."""
        print(f"\n{'='*80}")
        print("BACKTEST PERFORMANCE RESULTS")
        print(f"{'='*80}")

        total_return = (final_capital - self.initial_capital) / self.initial_capital
        num_trades = len(trades_df)

        print(f"\nCapital:")
        print(f"  Initial: ${self.initial_capital:,.2f}")
        print(f"  Final: ${final_capital:,.2f}")
        print(f"  Total Return: {total_return*100:.2f}%")

        print(f"\nTrades:")
        print(f"  Total: {num_trades}")

        if num_trades > 0:
            # Use net_return (after costs) for all metrics
            win_rate = (trades_df['pnl'] > 0).sum() / num_trades
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (trades_df['pnl'] < 0).any() else 0

            print(f"  Win Rate: {win_rate*100:.2f}%")
            print(f"  Avg Win: ${avg_win:,.2f}")
            print(f"  Avg Loss: ${avg_loss:,.2f}")

            if avg_loss != 0 and win_rate < 1:
                profit_factor = abs(avg_win / avg_loss) * (win_rate / (1 - win_rate))
                print(f"  Profit Factor: {profit_factor:.2f}")

            # Calculate Sharpe using net returns
            returns = trades_df['net_return'].values
            if len(returns) > 1:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 78)
                print(f"  Sharpe Ratio (Net): {sharpe:.2f}")

            if not equity_df.empty:
                equity_df = equity_df.sort_values('time')
                cummax = equity_df['equity'].cummax()
                drawdown = (equity_df['equity'] - cummax) / cummax
                max_drawdown = drawdown.min()
                print(f"  Max Drawdown: {max_drawdown*100:.2f}%")

        print(f"{'='*80}")

        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'final_capital': final_capital,
            'sharpe': sharpe if num_trades > 1 else 0,
            'max_drawdown': max_drawdown if not equity_df.empty else 0
        }

    def save_results(self, ticker: str, trades_df: pd.DataFrame, equity_df: pd.DataFrame,
                    metrics: Dict, processing_times: np.ndarray,
                    output_dir: Optional[str] = None):
        """
        Save backtest results with model name in directory structure.

        Args:
            ticker: Ticker symbol
            trades_df: Trades dataframe
            equity_df: Equity curve dataframe
            metrics: Performance metrics dictionary
            processing_times: Processing time array
            output_dir: Optional custom output directory (uses config default if None)
        """
        if output_dir is None:
            output_dir = str(OUTPUT_DIR)

        # Create model-specific directory structure
        ticker_dir = os.path.join(output_dir, self.model.name, ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        # Add model_name column to outputs
        trades_df_copy = trades_df.copy()
        trades_df_copy['model_name'] = self.model.name

        equity_df_copy = equity_df.copy()
        equity_df_copy['model_name'] = self.model.name

        # Save trades
        trades_path = os.path.join(ticker_dir, f'{ticker}_trades.csv')
        trades_df_copy.to_csv(trades_path, index=False)

        # Save equity curve
        equity_path = os.path.join(ticker_dir, f'{ticker}_equity_curve.csv')
        equity_df_copy.to_csv(equity_path, index=False)

        # Save metrics
        metrics_path = os.path.join(ticker_dir, f'{ticker}_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"BACKTEST RESULTS - {ticker}\n")
            f.write(f"Model: {self.model.name}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total Return: {metrics['total_return']*100:.2f}%\n")
            f.write(f"Final Capital: ${metrics['final_capital']:,.2f}\n")
            f.write(f"Total Trades: {metrics['num_trades']}\n")
            if 'sharpe' in metrics:
                f.write(f"Sharpe Ratio: {metrics['sharpe']:.2f}\n")
            if 'max_drawdown' in metrics:
                f.write(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%\n")

        print(f"\n✓ Results saved to: {ticker_dir}")

    def run_single_ticker(self, ticker: str, output_dir: Optional[str] = None,
                         price_df: Optional[pd.DataFrame] = None):
        """
        Run backtest for a single ticker.

        Args:
            ticker: Ticker symbol
            output_dir: Optional output directory
            price_df: Optional price data for return calculation

        Returns:
            Dictionary with backtest results
        """
        print(f"\n{'='*80}")
        print(f"BACKTESTING {ticker} with {self.model.name}")
        print(f"{'='*80}")

        # Load data
        self.load_data(ticker=ticker)

        # Prepare features
        X, y, data_df = self.prepare_features()

        # Split data
        (X_train, y_train, train_df), (X_test, y_test, test_df) = self.split_data(X, y, data_df)

        # Train model
        self.train_model(X_train, y_train, X_test, y_test)

        # Benchmark inference speed
        print("\nBenchmarking inference speed...")
        speed_metrics = self.model.benchmark_inference_speed(X_test[:100], num_runs=100)
        print(f"  Mean: {speed_metrics['mean_ms']:.4f}ms")
        print(f"  P95: {speed_metrics['p95_ms']:.4f}ms")
        print(f"  P99: {speed_metrics['p99_ms']:.4f}ms")

        # Backtest (with price_df for real return calculation)
        trades_df, equity_df, final_capital, processing_times = self.backtest_strategy(
            test_df, X_test, price_df=price_df
        )

        # Calculate metrics
        metrics = self.calculate_metrics(trades_df, equity_df, final_capital)

        # Save results
        self.save_results(ticker, trades_df, equity_df, metrics, processing_times, output_dir)

        return {
            'ticker': ticker,
            'metrics': metrics,
            'trades_df': trades_df,
            'equity_df': equity_df,
            'processing_times': processing_times
        }


if __name__ == "__main__":
    print("Backtest Engine - Refactored with Bias Fixes and Transaction Costs")
    print("="*80)
    print("\nKey Improvements:")
    print("  ✓ Uses RealReturnCalculator (no circular logic)")
    print("  ✓ Applies transaction costs (8 bps per trade)")
    print("  ✓ Model name in output directory")
    print("  ✓ Centralized configuration")
