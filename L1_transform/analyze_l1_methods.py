import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math
import warnings
warnings.filterwarnings('ignore')

# Import the methods from L1_transform
import sys
sys.path.insert(0, r'C:\Users\domdd\Documents\GitHub\confluence')

from L1_transform.per_exchange_state import MultiExchangeBook, ExchangeQuote
from L1_transform.consensus_microprice import compute_consensus_micro
from L1_transform.consensus_mid import compute_consensus_mid
from L1_transform.ema_estimate import EMAEfficientPrice
from L1_transform.kalman_filter import KalmanEfficientPrice

# Load data
print("Loading A.csv...")
df = pd.read_csv(r'C:\Users\domdd\Documents\GitHub\confluence\A.csv', nrows=50000)
print(f"Loaded {len(df)} rows")

# Parse timestamp to numeric
df['Timestamp_ns'] = pd.to_datetime(df['Timestamp']).values.astype(np.int64) / 1e9

# Filter for valid L1 quotes (bid and ask with non-zero prices)
df_valid = df[(df['Price'] > 0) & (df['Quantity'] > 0)].copy()

# Process quotes: for each timestamp, we need the latest bid and ask for each exchange
def get_l1_snapshot(group):
    """Extract L1 (best bid/ask) for each exchange at a timestamp"""
    bids = group[group['EventType'].str.contains('BID')]
    asks = group[group['EventType'].str.contains('ASK')]

    result = {}
    for exchange in group['Exchange'].unique():
        ex_bids = bids[bids['Exchange'] == exchange]
        ex_asks = asks[asks['Exchange'] == exchange]

        if len(ex_bids) > 0 and len(ex_asks) > 0:
            bid_row = ex_bids.iloc[-1]  # Latest bid
            ask_row = ex_asks.iloc[-1]  # Latest ask

            if bid_row['Price'] > 0 and ask_row['Price'] > 0:
                result[exchange] = {
                    'bid': bid_row['Price'],
                    'bid_size': bid_row['Quantity'],
                    'ask': ask_row['Price'],
                    'ask_size': ask_row['Quantity'],
                    'timestamp': bid_row['Timestamp_ns']
                }
    return result

# Group by 100ms windows to get snapshots
df_valid['window'] = (df_valid['Timestamp_ns'] * 10).astype(int)
snapshots = []

print("Creating L1 snapshots...")
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

snapshot_df = pd.DataFrame(snapshots)
print(f"Created {len(snapshot_df)} L1 snapshots across {snapshot_df['window'].nunique()} windows")

# Process windows and apply consolidation methods
results = {
    'timestamp': [],
    'consensus_mid': [],
    'consensus_micro': [],
    'ema_estimate': [],
    'kalman_filter': []
}

ema = EMAEfficientPrice(alpha=0.15)
kalman = KalmanEfficientPrice(process_var=0.02, obs_var=0.5)

print("Applying consolidation methods...")
for window_id in sorted(snapshot_df['window'].unique())[:5000]:  # Limit for performance
    window_data = snapshot_df[snapshot_df['window'] == window_id]

    if len(window_data) < 2:  # Need at least 2 exchanges
        continue

    # Create MultiExchangeBook
    book = MultiExchangeBook()
    timestamp = window_data['timestamp'].iloc[0]

    for _, row in window_data.iterrows():
        book.update_quote(
            row['exchange'],
            'BID',
            row['bid'],
            row['bid_size'],
            timestamp
        )
        book.update_quote(
            row['exchange'],
            'ASK',
            row['ask'],
            row['ask_size'],
            timestamp
        )

    # Apply methods
    mid = compute_consensus_mid(book, timestamp)
    micro = compute_consensus_micro(book, timestamp)

    if mid is not None and micro is not None:
        # For EMA and Kalman, use the midprice as input
        ema_val = ema.update(mid)
        kalman_val = kalman.update(mid)

        results['timestamp'].append(timestamp)
        results['consensus_mid'].append(mid)
        results['consensus_micro'].append(micro)
        results['ema_estimate'].append(ema_val)
        results['kalman_filter'].append(kalman_val)

results_df = pd.DataFrame(results)
print(f"\nGenerated {len(results_df)} consolidated estimates")

# Calculate comparative metrics
print("\n" + "="*80)
print("QUANTITATIVE COMPARISON OF L1 CONSOLIDATION METHODS")
print("="*80)

# 1. Spread analysis
print("\n1. SPREAD ANALYSIS (how tight/volatile the estimates are):")
for method in ['consensus_mid', 'consensus_micro', 'ema_estimate', 'kalman_filter']:
    values = results_df[method].values
    spread = np.ptp(values)  # peak-to-peak range
    std = np.std(values)
    cv = std / np.mean(values) * 100  # coefficient of variation
    print(f"  {method:20} | Range: {spread:10.4f} | Std: {std:10.6f} | CV: {cv:6.2f}%")

# 2. Deviation from consensus
print("\n2. DEVIATION FROM CONSENSUS (agreement between methods):")
consensus_mean = results_df[['consensus_mid', 'consensus_micro', 'ema_estimate', 'kalman_filter']].mean(axis=1)

for method in ['consensus_mid', 'consensus_micro', 'ema_estimate', 'kalman_filter']:
    deviation = np.abs(results_df[method] - consensus_mean)
    print(f"  {method:20} | Mean deviation: {deviation.mean():10.6f} | Max deviation: {deviation.max():10.6f}")

# 3. Price differences
print("\n3. RELATIVE PRICE DIFFERENCES:")
print(f"  Consensus Mid vs Consensus Micro:")
diff = np.abs(results_df['consensus_mid'] - results_df['consensus_micro']) / results_df['consensus_mid'] * 100
print(f"    Mean difference: {diff.mean():.4f}% | Median: {diff.median():.4f}% | Max: {diff.max():.4f}%")

print(f"\n  Consensus Mid vs EMA:")
diff = np.abs(results_df['consensus_mid'] - results_df['ema_estimate']) / results_df['consensus_mid'] * 100
print(f"    Mean difference: {diff.mean():.4f}% | Median: {diff.median():.4f}% | Max: {diff.max():.4f}%")

print(f"\n  Consensus Mid vs Kalman:")
diff = np.abs(results_df['consensus_mid'] - results_df['kalman_filter']) / results_df['consensus_mid'] * 100
print(f"    Mean difference: {diff.mean():.4f}% | Median: {diff.median():.4f}% | Max: {diff.max():.4f}%")

# 4. Smoothness (rate of change)
print("\n4. SMOOTHNESS (lower = less jittery):")
for method in ['consensus_mid', 'consensus_micro', 'ema_estimate', 'kalman_filter']:
    roc = np.abs(np.diff(results_df[method].values))
    print(f"  {method:20} | Mean |dP|: {roc.mean():10.6f} | Std |dP|: {roc.std():10.6f}")

# 5. Autocorrelation (trend-following ability)
print("\n5. AUTOCORRELATION (higher = better trend following):")
for method in ['consensus_mid', 'consensus_micro', 'ema_estimate', 'kalman_filter']:
    values = results_df[method].values
    acf = np.corrcoef(values[:-1], values[1:])[0, 1]
    print(f"  {method:20} | Lag-1 Autocorr: {acf:.6f}")

# Save results
results_df.to_csv(r'C:\Users\domdd\Documents\GitHub\confluence\L1_transform\l1_methods_comparison.csv', index=False)
print("\n[OK] Results saved to l1_methods_comparison.csv")

# Create visualizations
print("\nGenerating visualizations...")

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('L1 Order Book Consolidation Methods Comparison', fontsize=16, fontweight='bold')

# Plot 1: Time series of all methods
ax = axes[0, 0]
ax.plot(range(len(results_df)), results_df['consensus_mid'], label='Consensus Mid', alpha=0.7, linewidth=1)
ax.plot(range(len(results_df)), results_df['consensus_micro'], label='Consensus Micro', alpha=0.7, linewidth=1)
ax.plot(range(len(results_df)), results_df['ema_estimate'], label='EMA', alpha=0.7, linewidth=1)
ax.plot(range(len(results_df)), results_df['kalman_filter'], label='Kalman', alpha=0.7, linewidth=1)
ax.set_xlabel('Observation Index')
ax.set_ylabel('Estimated Price')
ax.set_title('Time Series Comparison (First 500 obs)')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, min(500, len(results_df)))

# Plot 2: Distribution of prices
ax = axes[0, 1]
ax.hist(results_df['consensus_mid'], bins=50, alpha=0.5, label='Consensus Mid')
ax.hist(results_df['consensus_micro'], bins=50, alpha=0.5, label='Consensus Micro')
ax.hist(results_df['ema_estimate'], bins=50, alpha=0.5, label='EMA')
ax.hist(results_df['kalman_filter'], bins=50, alpha=0.5, label='Kalman')
ax.set_xlabel('Estimated Price')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Price Estimates')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Deviation from consensus mean
ax = axes[1, 0]
consensus_mean = results_df[['consensus_mid', 'consensus_micro', 'ema_estimate', 'kalman_filter']].mean(axis=1)
for method in ['consensus_mid', 'consensus_micro', 'ema_estimate', 'kalman_filter']:
    deviation = np.abs(results_df[method] - consensus_mean)
    ax.plot(range(min(1000, len(deviation))), deviation.values[:1000], label=method, alpha=0.7, linewidth=0.8)
ax.set_xlabel('Observation Index')
ax.set_ylabel('Absolute Deviation from Mean')
ax.set_title('Deviation from Consensus (First 1000 obs)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Rate of change (smoothness)
ax = axes[1, 1]
for method in ['consensus_mid', 'consensus_micro', 'ema_estimate', 'kalman_filter']:
    roc = np.abs(np.diff(results_df[method].values))
    ax.plot(range(min(1000, len(roc))), roc[:1000], label=method, alpha=0.7, linewidth=0.8)
ax.set_xlabel('Observation Index')
ax.set_ylabel('|Δ Price|')
ax.set_title('Price Change Magnitude (Smoothness, First 1000 obs)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Box plot comparison
ax = axes[2, 0]
data_for_box = [
    results_df['consensus_mid'].values,
    results_df['consensus_micro'].values,
    results_df['ema_estimate'].values,
    results_df['kalman_filter'].values
]
bp = ax.boxplot(data_for_box, labels=['Consensus Mid', 'Consensus Micro', 'EMA', 'Kalman'])
ax.set_ylabel('Estimated Price')
ax.set_title('Distribution Comparison')
ax.grid(alpha=0.3, axis='y')

# Plot 6: Method differences
ax = axes[2, 1]
diff_mid_micro = np.abs(results_df['consensus_mid'] - results_df['consensus_micro']) / results_df['consensus_mid'] * 100
diff_mid_ema = np.abs(results_df['consensus_mid'] - results_df['ema_estimate']) / results_df['consensus_mid'] * 100
diff_mid_kalman = np.abs(results_df['consensus_mid'] - results_df['kalman_filter']) / results_df['consensus_mid'] * 100

ax.plot(range(min(1000, len(diff_mid_micro))), diff_mid_micro.values[:1000], label='vs Consensus Micro', alpha=0.7, linewidth=1)
ax.plot(range(min(1000, len(diff_mid_ema))), diff_mid_ema.values[:1000], label='vs EMA', alpha=0.7, linewidth=1)
ax.plot(range(min(1000, len(diff_mid_kalman))), diff_mid_kalman.values[:1000], label='vs Kalman', alpha=0.7, linewidth=1)
ax.set_xlabel('Observation Index')
ax.set_ylabel('Relative Difference (%)')
ax.set_title('Consensus Mid vs Other Methods (First 1000 obs)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\domdd\Documents\GitHub\confluence\L1_transform\l1_methods_comparison.png', dpi=100, bbox_inches='tight')
print("[OK] Visualization saved to l1_methods_comparison.png")
plt.show()

# Create scoring summary
print("\n" + "="*80)
print("SUMMARY SCORING")
print("="*80)

methods = ['consensus_mid', 'consensus_micro', 'ema_estimate', 'kalman_filter']
scores = {}

# Metric 1: Smoothness (lower RoC is better) - lower is better
for method in methods:
    roc = np.abs(np.diff(results_df[method].values)).mean()
    scores.setdefault(method, {})['smoothness'] = roc

# Metric 2: Consistency (lower std is better)
for method in methods:
    std = np.std(results_df[method].values)
    scores.setdefault(method, {})['consistency'] = std

# Metric 3: Agreement with other methods (lower deviation)
consensus_mean = results_df[['consensus_mid', 'consensus_micro', 'ema_estimate', 'kalman_filter']].mean(axis=1)
for method in methods:
    deviation = np.abs(results_df[method] - consensus_mean).mean()
    scores.setdefault(method, {})['agreement'] = deviation

# Normalize and rank
print("\nNormalized Scores (0-10, higher is better):")
print(f"{'Method':<20} | {'Smoothness':<12} | {'Consistency':<12} | {'Agreement':<12} | {'Overall':<10}")
print("-" * 70)

normalized_scores = {}
for metric in ['smoothness', 'consistency', 'agreement']:
    values = [scores[m][metric] for m in methods]
    max_val = max(values)
    min_val = min(values)
    if max_val == min_val:
        for m in methods:
            normalized_scores.setdefault(m, {})[metric] = 5.0
    else:
        for m in methods:
            # Invert for smoothness and consistency (lower is better)
            normalized_scores.setdefault(m, {})[metric] = 10.0 * (max_val - scores[m][metric]) / (max_val - min_val)

for method in methods:
    scores_vals = normalized_scores[method]
    overall = (scores_vals['smoothness'] + scores_vals['consistency'] + scores_vals['agreement']) / 3
    print(f"{method:<20} | {scores_vals['smoothness']:<12.2f} | {scores_vals['consistency']:<12.2f} | {scores_vals['agreement']:<12.2f} | {overall:<10.2f}")

# Rank
print("\n" + "="*80)
print("FINAL RANKING (by overall score)")
print("="*80)
rankings = [(method, (normalized_scores[method]['smoothness'] + normalized_scores[method]['consistency'] + normalized_scores[method]['agreement']) / 3) for method in methods]
rankings.sort(key=lambda x: x[1], reverse=True)

for i, (method, score) in enumerate(rankings, 1):
    print(f"{i}. {method:20} - Score: {score:.2f}/10")
