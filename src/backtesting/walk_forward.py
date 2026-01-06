"""
Walk-forward validation for trading models.

This implements expanding window validation to test temporal stability
and avoid overfitting to specific time periods.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models.base_model import BaseTradingModel
from src.backtesting.backtest_engine import BacktestEngine
from config.settings import WF_INITIAL_TRAIN_SIZE, WF_TEST_WINDOW_SIZE, WF_STEP_SIZE


class WalkForwardValidator:
    """
    Walk-forward validation with expanding window.

    This approach trains on progressively larger historical windows and tests
    on out-of-sample periods. It's more realistic than simple train/test split
    because it tests performance across multiple time periods.

    Example:
        Window 1: Train [0:1000], Test [1000:1200]
        Window 2: Train [0:1100], Test [1100:1300]
        Window 3: Train [0:1200], Test [1200:1400]
        ...
    """

    def __init__(self, initial_train_size: int = WF_INITIAL_TRAIN_SIZE,
                 test_window_size: int = WF_TEST_WINDOW_SIZE,
                 step_size: int = WF_STEP_SIZE):
        """
        Initialize walk-forward validator.

        Args:
            initial_train_size: Number of samples in initial training set
            test_window_size: Number of samples in each test window
            step_size: Number of samples to move forward each iteration
        """
        self.initial_train_size = initial_train_size
        self.test_window_size = test_window_size
        self.step_size = step_size

        print(f"Walk-Forward Validation Configuration:")
        print(f"  Initial train size: {initial_train_size}")
        print(f"  Test window size: {test_window_size}")
        print(f"  Step size: {step_size}")

    def create_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create expanding window train/test splits.

        Args:
            n_samples: Total number of samples

        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []

        # Start with initial window
        train_end = self.initial_train_size

        while train_end + self.test_window_size <= n_samples:
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(train_end, train_end + self.test_window_size)

            splits.append((train_indices, test_indices))

            # Move forward
            train_end += self.step_size

        print(f"\nCreated {len(splits)} walk-forward splits")
        print(f"Total samples used: {len(splits[-1][1][-1]) if splits else 0} / {n_samples}")

        return splits

    def run_walk_forward(self, model: BaseTradingModel, X: np.ndarray, y: np.ndarray,
                        data_df: pd.DataFrame, ticker: str,
                        backtest_engine: Optional[BacktestEngine] = None,
                        price_df: Optional[pd.DataFrame] = None):
        """
        Run walk-forward validation.

        Args:
            model: Trading model to validate
            X: Feature matrix
            y: Target vector
            data_df: Full dataset with metadata
            ticker: Ticker symbol
            backtest_engine: Optional backtest engine (creates new if None)
            price_df: Optional price data for return calculation

        Returns:
            Dictionary with walk-forward results
        """
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD VALIDATION: {model.name} on {ticker}")
        print(f"{'='*80}")

        # Create backtest engine if not provided
        if backtest_engine is None:
            backtest_engine = BacktestEngine(
                model=model,
                tickers=[ticker],
                apply_transaction_costs=True
            )

        # Create splits
        splits = self.create_splits(len(X))

        results = []

        for i, (train_idx, test_idx) in enumerate(splits):
            print(f"\n{'-'*80}")
            print(f"WALK-FORWARD ITERATION {i+1}/{len(splits)}")
            print(f"{'-'*80}")
            print(f"Train: samples [0:{len(train_idx)}] ({len(train_idx)} samples)")
            print(f"Test: samples [{test_idx[0]}:{test_idx[-1]+1}] ({len(test_idx)} samples)")

            # Split data
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            test_df = data_df.iloc[test_idx]

            # Use a portion of train as validation (last 15%)
            val_split = int(len(X_train) * 0.85)
            X_train_sub, X_val = X_train[:val_split], X_train[val_split:]
            y_train_sub, y_val = y_train[:val_split], y_train[val_split:]

            print(f"  Training on {len(X_train_sub)} samples, validating on {len(X_val)}")

            # Train model
            try:
                model.fit(X_train_sub, y_train_sub, X_val, y_val)

                # Backtest on test set
                trades_df, equity_df, final_capital, _ = backtest_engine.backtest_strategy(
                    test_df, X_test, price_df=price_df
                )

                # Calculate metrics
                metrics = backtest_engine.calculate_metrics(trades_df, equity_df, final_capital)

                # Evaluate model predictions
                eval_metrics = model.evaluate(X_test, y_test)

                results.append({
                    'iteration': i + 1,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'train_start': 0,
                    'train_end': len(train_idx),
                    'test_start': test_idx[0],
                    'test_end': test_idx[-1] + 1,
                    'metrics': metrics,
                    'model_metrics': eval_metrics,
                    'trades_df': trades_df,
                    'equity_df': equity_df
                })

                print(f"\nIteration {i+1} Results:")
                print(f"  Total Return: {metrics['total_return']*100:.2f}%")
                print(f"  Sharpe Ratio: {metrics.get('sharpe', 0):.2f}")
                print(f"  Model Correlation: {eval_metrics['correlation']:.4f}")
                print(f"  Directional Accuracy: {eval_metrics['directional_accuracy']*100:.2f}%")

            except Exception as e:
                print(f"\n⚠️ ERROR in iteration {i+1}: {e}")
                print("Skipping this iteration...")
                continue

        if not results:
            print("\n❌ Walk-forward validation failed - no successful iterations")
            return None

        # Aggregate results
        avg_return = np.mean([r['metrics']['total_return'] for r in results])
        std_return = np.std([r['metrics']['total_return'] for r in results])
        avg_sharpe = np.mean([r['metrics'].get('sharpe', 0) for r in results])
        avg_correlation = np.mean([r['model_metrics']['correlation'] for r in results])
        avg_directional = np.mean([r['model_metrics']['directional_accuracy'] for r in results])

        print(f"\n{'='*80}")
        print("WALK-FORWARD VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"Model: {model.name}")
        print(f"Ticker: {ticker}")
        print(f"Number of Iterations: {len(results)}")
        print(f"\nBacktest Performance:")
        print(f"  Average Return: {avg_return*100:.2f}% ± {std_return*100:.2f}%")
        print(f"  Average Sharpe: {avg_sharpe:.2f}")
        print(f"\nModel Performance:")
        print(f"  Average Correlation: {avg_correlation:.4f}")
        print(f"  Average Directional Accuracy: {avg_directional*100:.2f}%")

        # Check for degradation over time
        returns = [r['metrics']['total_return'] for r in results]
        if len(returns) >= 3:
            early_returns = np.mean(returns[:len(returns)//3])
            late_returns = np.mean(returns[-len(returns)//3:])
            degradation = (early_returns - late_returns) / abs(early_returns) if early_returns != 0 else 0

            print(f"\nTemporal Stability:")
            print(f"  Early period return: {early_returns*100:.2f}%")
            print(f"  Late period return: {late_returns*100:.2f}%")
            print(f"  Degradation: {degradation*100:.2f}%")

            if degradation > 0.3:
                print(f"  ⚠️ WARNING: Significant performance degradation over time!")
            elif degradation > 0.1:
                print(f"  ⚠️ CAUTION: Moderate performance degradation")
            else:
                print(f"  ✓ Good temporal stability")

        print(f"{'='*80}")

        return {
            'iterations': results,
            'avg_return': avg_return,
            'std_return': std_return,
            'avg_sharpe': avg_sharpe,
            'avg_correlation': avg_correlation,
            'avg_directional': avg_directional,
            'num_iterations': len(results)
        }

    def save_walk_forward_results(self, wf_results: Dict, output_path: str):
        """
        Save walk-forward results to file.

        Args:
            wf_results: Walk-forward results dictionary
            output_path: Path to save results
        """
        import json

        # Convert results to JSON-serializable format
        summary = {
            'avg_return': float(wf_results['avg_return']),
            'std_return': float(wf_results['std_return']),
            'avg_sharpe': float(wf_results['avg_sharpe']),
            'avg_correlation': float(wf_results['avg_correlation']),
            'avg_directional': float(wf_results['avg_directional']),
            'num_iterations': wf_results['num_iterations'],
            'iterations': []
        }

        # Add iteration summaries
        for iteration in wf_results['iterations']:
            summary['iterations'].append({
                'iteration': iteration['iteration'],
                'train_size': iteration['train_size'],
                'test_size': iteration['test_size'],
                'total_return': float(iteration['metrics']['total_return']),
                'sharpe': float(iteration['metrics'].get('sharpe', 0)),
                'correlation': float(iteration['model_metrics']['correlation']),
                'directional_accuracy': float(iteration['model_metrics']['directional_accuracy'])
            })

        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Walk-forward results saved to: {output_path}")


def demonstrate_walk_forward():
    """Demonstration of walk-forward validation."""
    print("Walk-Forward Validation Demonstration")
    print("="*80)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 2000
    n_features = 50

    # Create features
    X = np.random.randn(n_samples, n_features)

    # Create target with time-varying relationship
    # Early period: feature 0 is important
    # Late period: feature 1 becomes important
    y = np.zeros(n_samples)
    for i in range(n_samples):
        if i < n_samples // 2:
            y[i] = 0.5 * X[i, 0] + 0.1 * np.random.randn()
        else:
            y[i] = 0.5 * X[i, 1] + 0.1 * np.random.randn()

    # Create data_df
    data_df = pd.DataFrame({
        'ticker': ['TEST'] * n_samples,
        'tweet_time': pd.date_range('2024-01-01', periods=n_samples, freq='5min'),
        'forward_return': y
    })

    # Create walk-forward validator
    validator = WalkForwardValidator(
        initial_train_size=1000,
        test_window_size=200,
        step_size=100
    )

    # Create splits
    splits = validator.create_splits(n_samples)

    print(f"\nExample splits:")
    for i, (train_idx, test_idx) in enumerate(splits[:3]):
        print(f"  Split {i+1}: Train [0:{len(train_idx)}], Test [{test_idx[0]}:{test_idx[-1]+1}]")

    print("\n" + "="*80)
    print("Walk-forward validation demonstration complete!")
    print("\nNOTE: To run full validation, provide a trained model and use run_walk_forward()")


if __name__ == "__main__":
    demonstrate_walk_forward()
