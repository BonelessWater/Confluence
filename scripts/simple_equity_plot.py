"""
Simple script to create equity curve visualization without matplotlib display.
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

from src.models.linear_model import LinearModel
from src.models.xgboost_model import XGBoostModel
from config.settings import INITIAL_CAPITAL

def generate_synthetic_data(n_samples=2000):
    """Generate synthetic data."""
    np.random.seed(42)
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(minutes=30*i) for i in range(n_samples)]

    features = {}
    for i in range(100):
        if i < 20:
            autocorr = np.zeros(n_samples)
            autocorr[0] = np.random.randn()
            for j in range(1, n_samples):
                autocorr[j] = 0.7 * autocorr[j-1] + 0.3 * np.random.randn()
            features[f'feature_{i}'] = autocorr
        else:
            features[f'feature_{i}'] = np.random.randn(n_samples)

    df = pd.DataFrame(features)
    df['ticker'] = 'SPY'
    df['tweet_time'] = timestamps
    df['entry_time'] = timestamps
    df['tweet_id'] = [f'tweet_{i}' for i in range(n_samples)]

    signal = (0.3 * df['feature_0'] + 0.2 * df['feature_1'] - 0.15 * df['feature_2'])
    noise = np.random.randn(n_samples) * 5
    df['forward_return'] = (signal + noise) / 100
    df['forward_return'] = df['forward_return'].clip(-0.05, 0.05)

    return df

def run_simple_backtest(df, model):
    """Run simple backtest."""
    exclude_cols = ['ticker', 'tweet_id', 'tweet_time', 'entry_time', 'forward_return']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].values
    X = np.nan_to_num(X)
    y = df['forward_return'].values

    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    test_df = df.iloc[split_idx:].copy()

    # Train
    model.fit(X_train, y_train, X_test, y_test)

    # Predict
    predictions = model.predict(X_test)

    # Backtest
    capital = INITIAL_CAPITAL
    equity_curve = []

    for i in range(len(test_df)):
        row = test_df.iloc[i]
        weight = 1.0 / (1.0 + np.exp(-predictions[i]))

        gross_return = row['forward_return']
        net_return = gross_return - 0.0008

        pnl = capital * weight * net_return
        capital += pnl

        equity_curve.append({
            'time': row['tweet_time'],
            'equity': capital
        })

    equity_df = pd.DataFrame(equity_curve)
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

    return equity_df, total_return, capital

def main():
    """Run and plot."""
    print("Generating synthetic data and running backtests...")

    df = generate_synthetic_data()

    # Run backtests
    models = [
        ('Linear Regression', LinearModel(name="LinearRegression", n_features=50, alpha=1.0)),
        ('XGBoost', XGBoostModel(name="XGBoost", max_depth=5, n_estimators=100)),
    ]

    results = {}
    for name, model in models:
        print(f"\nRunning {name}...")
        equity_df, total_return, final_capital = run_simple_backtest(df, model)
        results[name] = {
            'equity_df': equity_df,
            'return': total_return,
            'capital': final_capital
        }
        print(f"  Return: {total_return*100:.2f}%")
        print(f"  Final Capital: ${final_capital:,.2f}")

    # Create plot
    print("\nCreating equity curve plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    colors = {'Linear Regression': '#1f77b4', 'XGBoost': '#ff7f0e'}

    for name, result in results.items():
        equity_df = result['equity_df'].sort_values('time')
        times = equity_df['time']
        equity = equity_df['equity'].values
        returns_pct = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        # Plot 1: Returns
        ax1.plot(times, returns_pct, label=name, linewidth=2.5, color=colors[name])

        # Plot 2: Drawdown
        cummax = pd.Series(equity).cummax()
        drawdown = (equity - cummax) / cummax * 100
        ax2.plot(times, drawdown, label=name, linewidth=1.5, color=colors[name])

    # Format
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Backtest Results - SPY\n(With 8 bps Transaction Costs)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.fill_between(equity_df['time'], drawdown, 0, alpha=0.2, color='red')
    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'SPY_equity_curves_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[SUCCESS] Equity curve saved to: {output_file}")

    # Also save results summary
    summary_file = output_dir / 'backtest_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("BACKTEST RESULTS SUMMARY\n")
        f.write("="*60 + "\n\n")
        for name, result in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Total Return: {result['return']*100:.2f}%\n")
            f.write(f"  Final Capital: ${result['capital']:,.2f}\n")
            f.write("\n")
    print(f"[SUCCESS] Summary saved to: {summary_file}")

    print("\nDone!")

if __name__ == "__main__":
    main()
