"""
Run backtest and generate equity curve visualizations.

This script:
1. Loads pre-computed features
2. Trains models (Linear, XGBoost, optionally Attention)
3. Runs backtests with realistic transaction costs
4. Generates equity curve plots
5. Saves all results to output directory
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.linear_model import LinearModel
from src.models.xgboost_model import XGBoostModel
from src.backtesting.backtest_engine import BacktestEngine
from config.settings import TICKERS, INITIAL_CAPITAL

def plot_equity_curves(results_dict, ticker, output_dir):
    """
    Plot equity curves for all models on a single chart.

    Args:
        results_dict: Dictionary with model names as keys and equity_df as values
        ticker: Ticker symbol
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(14, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, (model_name, result) in enumerate(results_dict.items()):
        equity_df = result['equity_df']
        if not equity_df.empty:
            equity_df = equity_df.sort_values('time')
            # Convert to datetime if needed
            times = pd.to_datetime(equity_df['time'])
            equity = equity_df['equity'].values

            # Calculate cumulative return
            returns = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

            plt.plot(times, returns, label=model_name, linewidth=2,
                    color=colors[idx % len(colors)])

    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.title(f'Equity Curves - {ticker}\n(With 8 bps Transaction Costs)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, f'{ticker}_equity_curves_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Equity curve plot saved to: {plot_path}")

    plt.show()

def plot_individual_equity_curve(equity_df, model_name, ticker, metrics, output_dir):
    """
    Plot individual equity curve with metrics.

    Args:
        equity_df: Equity curve dataframe
        model_name: Name of the model
        ticker: Ticker symbol
        metrics: Performance metrics dict
        output_dir: Directory to save plot
    """
    if equity_df.empty:
        print(f"⚠ No equity data for {model_name}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Sort by time
    equity_df = equity_df.sort_values('time')
    times = pd.to_datetime(equity_df['time'])
    equity = equity_df['equity'].values

    # Plot 1: Equity curve
    ax1.plot(times, equity, linewidth=2, color='#1f77b4', label='Portfolio Value')
    ax1.axhline(y=INITIAL_CAPITAL, color='black', linestyle='--',
                linewidth=1, alpha=0.5, label='Initial Capital')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.set_title(f'{model_name} - {ticker} Backtest Results\n' +
                  f'Total Return: {metrics["total_return"]*100:.2f}% | ' +
                  f'Sharpe: {metrics.get("sharpe", 0):.2f} | ' +
                  f'Max DD: {metrics.get("max_drawdown", 0)*100:.2f}%',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Drawdown
    cummax = pd.Series(equity).cummax()
    drawdown = (equity - cummax) / cummax * 100
    ax2.fill_between(times, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')

    plt.tight_layout()

    # Save plot
    model_dir = os.path.join(output_dir, model_name, ticker)
    os.makedirs(model_dir, exist_ok=True)
    plot_path = os.path.join(model_dir, f'{ticker}_backtest_visualization.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Individual plot saved to: {plot_path}")

    plt.close()

def main():
    """Run backtests and generate visualizations."""

    print("="*80)
    print("TRADING BACKTEST WITH EQUITY CURVES")
    print("="*80)

    # Configuration
    ticker = 'SPY'  # Start with SPY
    run_attention = False  # Set to True if you want to run Attention model (slower)

    print(f"\nConfiguration:")
    print(f"  Ticker: {ticker}")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"  Transaction Costs: 8 bps per trade")
    print(f"  Models: Linear, XGBoost" + (", Attention" if run_attention else ""))

    # Create models
    models = [
        LinearModel(name="LinearRegression", n_features=50, alpha=1.0),
        XGBoostModel(name="XGBoost", max_depth=5, n_estimators=100, learning_rate=0.1),
    ]

    if run_attention:
        from src.models.attention_model import AttentionModel
        models.append(AttentionModel(name="AttentionModel"))

    # Run backtests
    results_dict = {}

    for model in models:
        print(f"\n{'='*80}")
        print(f"Running backtest for {model.name}")
        print(f"{'='*80}")

        try:
            # Create backtest engine
            engine = BacktestEngine(
                model=model,
                tickers=[ticker],
                apply_transaction_costs=True
            )

            # Run backtest
            result = engine.run_single_ticker(ticker)

            # Store results
            results_dict[model.name] = result

            # Generate individual plot
            plot_individual_equity_curve(
                equity_df=result['equity_df'],
                model_name=model.name,
                ticker=ticker,
                metrics=result['metrics'],
                output_dir='output'
            )

        except Exception as e:
            print(f"⚠ Error running {model.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Generate comparison plot
    if len(results_dict) > 0:
        print(f"\n{'='*80}")
        print("Generating comparison plot")
        print(f"{'='*80}")

        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)

        plot_equity_curves(results_dict, ticker, str(output_dir))

        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        for model_name, result in results_dict.items():
            metrics = result['metrics']
            print(f"\n{model_name}:")
            print(f"  Total Return: {metrics['total_return']*100:>8.2f}%")
            print(f"  Final Capital: ${metrics['final_capital']:>12,.2f}")
            print(f"  Sharpe Ratio: {metrics.get('sharpe', 0):>8.2f}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:>8.2f}%")
            print(f"  Total Trades: {metrics['num_trades']:>8,}")

    print(f"\n{'='*80}")
    print("✓ BACKTEST COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to:")
    print(f"  Individual: output/{{model_name}}/{ticker}/")
    print(f"  Comparison: output/{ticker}_equity_curves_comparison.png")

if __name__ == "__main__":
    main()
